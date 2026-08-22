"""Redis-backed extraction worker.

Consumes extraction jobs, calls Groq for structured parsing,
updates node props, and triggers re-embed if needed.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, cast
from urllib.parse import urlsplit

import redis

from activekg.common.control_plane import (
    ControlPlaneUnauthorized,
    ControlPlaneUnavailable,
    verify_control_plane_authorization,
)
from activekg.embedding.queue import enqueue_embedding_job
from activekg.extraction.client import (
    ExtractionClient,
    ExtractionError,
    assert_extraction_models_configured,
)
from activekg.extraction.prompt import get_extraction_version
from activekg.extraction.queue import (
    EXTRACTION_DLQ_KEY,
    EXTRACTION_QUEUE_KEY,
    clear_extraction_pending,
    move_due_extraction_retries,
    schedule_extraction_retry,
)
from activekg.extraction.schema import ExtractionStatus
from activekg.graph.repository import GraphRepository
from activekg.privacy.models import CandidatePrivacyDecision
from activekg.privacy.repository import CandidatePrivacyRepository, CandidatePrivacyUnavailable

logger = logging.getLogger(__name__)

# Healthcheck port (configurable via env)
HEALTHCHECK_PORT = int(os.getenv("EXTRACTION_HEALTHCHECK_PORT", "8080"))


class WorkerHealthState:
    """Thread-safe in-memory readiness state; request handlers perform no I/O."""

    def __init__(self, poll_interval_seconds: float) -> None:
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._poll_interval = poll_interval_seconds
        self._running = True
        self._loop_status = "starting"
        self._redis_status = "starting"
        self._loop_observed_at: float | None = None
        self._database_status = "starting"
        self._database_observed_at: float | None = None
        self._provider_status = "starting"
        self._provider_failures = 0

    def provider_configured(self) -> None:
        with self._lock:
            self._provider_status = "configured"
            self._provider_failures = 0

    def provider_success(self) -> None:
        with self._lock:
            self._provider_status = "ready"
            self._provider_failures = 0

    def provider_failure(self) -> None:
        with self._lock:
            self._provider_failures += 1
            self._provider_status = "error" if self._provider_failures >= 2 else "degraded"

    def loop_cycle_success(self) -> None:
        with self._lock:
            self._loop_status = "ready"
            self._redis_status = "ready"
            self._loop_observed_at = time.monotonic()

    def loop_error(self) -> None:
        with self._lock:
            self._loop_status = "error"
            self._redis_status = "error"
            self._loop_observed_at = time.monotonic()

    def database_success(self) -> None:
        with self._lock:
            self._database_status = "ready"
            self._database_observed_at = time.monotonic()

    def database_error(self) -> None:
        with self._lock:
            self._database_status = "error"
            self._database_observed_at = time.monotonic()

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._loop_status = "stopped"
        self._stopped.set()

    def wait(self, seconds: float) -> bool:
        """Return true when shutdown interrupts the monitor wait."""

        return self._stopped.wait(seconds)

    def snapshot(self) -> tuple[bool, dict[str, str]]:
        now = time.monotonic()
        with self._lock:
            loop_status = self._loop_status
            redis_status = self._redis_status
            database_status = self._database_status
            provider_status = self._provider_status
            loop_observed_at = self._loop_observed_at
            database_observed_at = self._database_observed_at
            running = self._running

        loop_stale_after = max(10.0, 3.0 * self._poll_interval)
        if not running or loop_observed_at is None or now - loop_observed_at > loop_stale_after:
            loop_status = "stale" if running else "stopped"
            redis_status = "stale" if running else redis_status
        if database_observed_at is None or now - database_observed_at > 90.0:
            database_status = "stale"

        components = {
            "loop": loop_status,
            "redis": redis_status,
            "database": database_status,
            "provider": provider_status,
        }
        ready = (
            loop_status == "ready"
            and redis_status == "ready"
            and database_status == "ready"
            and provider_status in {"configured", "ready", "degraded"}
        )
        return ready, components


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Constant-cost liveness and authenticated in-memory readiness."""

    def _write_json(self, status: int, payload: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        if status == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = urlsplit(self.path).path
        state = cast(WorkerHealthState, self.server.health_state)
        status, payload = worker_health_response(path, self.headers.get("Authorization"), state)
        if payload is None:
            self.send_response(status)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self._write_json(status, payload)

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def worker_health_response(
    path: str,
    authorization: str | None,
    state: WorkerHealthState,
) -> tuple[int, bytes | None]:
    """Return one dependency-free worker health/readiness response contract."""

    if path == "/health":
        return 200, b'{"status":"alive","service":"extraction-worker"}'
    if path != "/readyz":
        return 404, None

    try:
        verify_control_plane_authorization(authorization)
    except ControlPlaneUnavailable:
        return (
            503,
            b'{"detail":{"code":"CONTROL_PLANE_AUTH_UNAVAILABLE",'
            b'"message":"Operational authentication is unavailable."}}',
        )
    except ControlPlaneUnauthorized:
        return (
            401,
            b'{"detail":{"code":"CONTROL_PLANE_AUTH_REQUIRED",'
            b'"message":"Operational authentication is required."}}',
        )

    ready, components = state.snapshot()
    payload = json.dumps(
        {
            "status": "ready" if ready else "not_ready",
            "components": components,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return (200 if ready else 503), payload


def start_healthcheck_server(state: WorkerHealthState) -> HTTPServer:
    """Start healthcheck HTTP server in background thread."""
    server = HTTPServer(("0.0.0.0", HEALTHCHECK_PORT), HealthCheckHandler)
    server.health_state = state  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Healthcheck server started on port {HEALTHCHECK_PORT}")
    return server


class ExtractionWorker:
    """Worker that processes extraction queue jobs."""

    def __init__(
        self,
        redis_client: redis.Redis,
        repo: GraphRepository,
        extraction_client: ExtractionClient,
        health_state: WorkerHealthState,
        privacy_repository: CandidatePrivacyRepository | None = None,
        *,
        poll_interval_seconds: float = 1.0,
        max_attempts: int = 2,  # Only one fallback attempt
        retry_base_seconds: float = 10.0,
        retry_max_seconds: float = 60.0,
    ):
        """Initialize extraction worker.

        Args:
            redis_client: Redis client for queue operations
            repo: Graph repository for DB operations
            extraction_client: Groq extraction client
            poll_interval_seconds: How often to poll queue
            max_attempts: Max extraction attempts (default 2 = primary + fallback)
            retry_base_seconds: Base delay for retries
            retry_max_seconds: Max delay for retries
        """
        self.redis_client = redis_client
        self.repo = repo
        self.extraction_client = extraction_client
        self.health_state = health_state
        self.privacy_repository = privacy_repository
        self.poll_interval = poll_interval_seconds
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.running = True

        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info(
            "Extraction worker initialized",
            extra={
                "poll_interval": poll_interval_seconds,
                "max_attempts": max_attempts,
            },
        )

    def _shutdown_handler(self, signum, frame):
        logger.info("Extraction worker shutting down", extra={"signal": signum})
        self.running = False
        self.health_state.stop()

    def _process_job(self, raw: bytes | str) -> None:
        """Process a single extraction job."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")

        try:
            job = cast(dict[str, Any], json.loads(raw))
        except Exception as e:
            logger.error("Invalid job payload", extra={"error": str(e)})
            return

        node_id = cast(str | None, job.get("node_id"))
        tenant_id = cast(str | None, job.get("tenant_id"))
        attempts = job.get("attempts", 0) + 1
        job["attempts"] = attempts

        if not node_id:
            logger.error("Job missing node_id")
            return

        try:
            decision = self._privacy_decision(node_id)
            if self._privacy_blocks_node(decision, tenant_id):
                self._update_extraction_status(
                    node_id,
                    tenant_id,
                    ExtractionStatus(status="skipped", error="privacy_restricted"),
                    enforce_privacy=False,
                )
                clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return
            # Get node
            node = self.repo.get_node(node_id, tenant_id=tenant_id)
            if not node:
                self._update_extraction_status(
                    node_id,
                    tenant_id,
                    ExtractionStatus(status="failed", error="node_not_found"),
                )
                clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            decision = self._privacy_decision(node_id)
            if self._privacy_blocks_node(decision, tenant_id):
                self._update_extraction_status(
                    node_id,
                    tenant_id,
                    ExtractionStatus(status="skipped", error="privacy_restricted"),
                    enforce_privacy=False,
                )
                clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            self._update_extraction_status(
                node_id, tenant_id, ExtractionStatus(status="processing")
            )

            # Get raw text for extraction (not embedding text with prefix)
            text = self.repo.load_payload_text(node)
            if not text or len(text) < 100:
                self._update_extraction_status(
                    node_id,
                    tenant_id,
                    ExtractionStatus(status="skipped", error="insufficient_text"),
                )
                clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            # Extract
            current_version = get_extraction_version()
            result, model_used = self.extraction_client.extract(text)
            self.health_state.provider_success()

            # Build status
            status = ExtractionStatus(
                status="ready",
                confidence=result.confidence,
                extracted_at=datetime.now(timezone.utc).isoformat(),
                extraction_version=current_version,
                model_used=model_used,
            )

            # Update node props with extracted fields + status
            extracted_props = result.to_props()
            status_props = status.to_props()
            all_props = {**extracted_props, **status_props}

            decision = self._privacy_decision(node_id)
            if self._privacy_blocks_node(decision, tenant_id):
                self._update_extraction_status(
                    node_id,
                    tenant_id,
                    ExtractionStatus(status="skipped", error="privacy_restricted"),
                    enforce_privacy=False,
                )
                clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return
            self._update_node_props(node_id, tenant_id, all_props)

            logger.info(
                "Extraction completed",
                extra={
                    "node_id": node_id,
                    "model": model_used,
                    "confidence": result.confidence,
                    "skills_count": len(result.skills_raw) or len(result.primary_skills),
                    "titles_count": len(result.primary_titles) or len(result.recent_job_titles),
                },
            )

            # Post-extraction: sync platform applicants to global memory
            self._maybe_sync_to_global_memory(node_id, tenant_id, node, result)

            # Trigger re-embed if extraction version differs from embedding version
            node_embed_version = (node.props or {}).get("extraction_version")
            if node_embed_version != current_version:
                self._trigger_reembed(node_id, tenant_id)

            clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)

        except CandidatePrivacyUnavailable:
            try:
                self._update_extraction_status(
                    node_id,
                    tenant_id,
                    ExtractionStatus(status="skipped", error="privacy_restricted"),
                    enforce_privacy=False,
                )
            except Exception:
                # If the authority database is unavailable, do not turn
                # uncertainty into provider retries.  Redis settlement is the
                # privacy-safe operation that remains available.
                pass
            clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)
            return
        except ExtractionError as e:
            self.health_state.provider_failure()
            error_msg = str(e)
            logger.warning(
                "Extraction failed",
                extra={"node_id": node_id, "attempts": attempts, "error": error_msg},
            )

            if attempts >= self.max_attempts:
                # Max attempts reached - mark failed, move to DLQ
                self._update_extraction_status(
                    node_id,
                    tenant_id,
                    ExtractionStatus(status="failed", error=error_msg),
                )
                job["error"] = error_msg
                job["failed_at"] = time.time()
                self.redis_client.lpush(EXTRACTION_DLQ_KEY, json.dumps(job))
                clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            # Schedule retry
            delay = min(
                self.retry_base_seconds * (2 ** max(0, attempts - 1)),
                self.retry_max_seconds,
            )
            self._update_extraction_status(node_id, tenant_id, ExtractionStatus(status="queued"))
            job["error"] = error_msg
            schedule_extraction_retry(self.redis_client, job, delay_seconds=delay)

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Extraction job failed unexpectedly",
                extra={"node_id": node_id, "tenant_id": tenant_id, "error": error_msg},
            )

            if attempts >= self.max_attempts:
                self._update_extraction_status(
                    node_id,
                    tenant_id,
                    ExtractionStatus(status="failed", error=error_msg),
                )
                job["error"] = error_msg
                job["failed_at"] = time.time()
                self.redis_client.lpush(EXTRACTION_DLQ_KEY, json.dumps(job))
                clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            delay = min(
                self.retry_base_seconds * (2 ** max(0, attempts - 1)),
                self.retry_max_seconds,
            )
            self._update_extraction_status(node_id, tenant_id, ExtractionStatus(status="queued"))
            job["error"] = error_msg
            schedule_extraction_retry(self.redis_client, job, delay_seconds=delay)

    def _update_extraction_status(
        self,
        node_id: str,
        tenant_id: str | None,
        status: ExtractionStatus,
        *,
        enforce_privacy: bool = True,
    ) -> None:
        """Update extraction status in node props."""
        self._update_node_props(
            node_id,
            tenant_id,
            status.to_props(),
            enforce_privacy=enforce_privacy,
        )

    def _update_node_props(
        self,
        node_id: str,
        tenant_id: str | None,
        props: dict[str, Any],
        *,
        enforce_privacy: bool = True,
    ) -> None:
        """Merge props into node."""
        with self.repo._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                privacy_sql = (
                    "AND (candidate_privacy_node_decision(id) = 'allow' OR ("
                    "candidate_privacy_node_decision(id) = 'block_global' "
                    "AND tenant_id IS NOT NULL AND tenant_id IS NOT DISTINCT FROM %s))"
                    if enforce_privacy
                    else ""
                )
                params: tuple[Any, ...] = (json.dumps(props), node_id)
                if enforce_privacy:
                    params = (*params, tenant_id)
                cur.execute(
                    f"""
                    UPDATE nodes
                    SET props = COALESCE(props, '{{}}'::jsonb) || %s::jsonb,
                        updated_at = now()
                    WHERE id = %s
                    {privacy_sql}
                    """,
                    params,
                )
                if enforce_privacy and cur.rowcount != 1:
                    raise CandidatePrivacyUnavailable("candidate privacy restriction applies")

    def _maybe_sync_to_global_memory(
        self,
        node_id: str,
        tenant_id: str | None,
        node: Any,
        result: Any,
    ) -> None:
        """After extraction, sync platform applicant nodes to global_candidates."""
        if os.getenv("GLOBAL_MEMORY_ENABLED", "false").lower() != "true":
            return

        metadata = node.metadata or {}
        # platform_applicant = candidate-submitted (public per consent);
        # org_upload = recruiter/bulk import (stays org-private via visibility).
        if metadata.get("provenance_type") not in ("platform_applicant", "org_upload"):
            return

        try:
            decision = self._privacy_decision(node_id)
            if decision is not CandidatePrivacyDecision.ALLOW:
                return
            from activekg.api.global_memory import sync_applicant_to_global_memory

            sync_applicant_to_global_memory(
                node_id=node_id,
                tenant_id=tenant_id,
                node_props=node.props or {},
                extracted_result=result,
                metadata=metadata,
            )
            logger.info(
                "Synced applicant to global memory",
                extra={"node_id": node_id, "tenant_id": tenant_id},
            )
        except CandidatePrivacyUnavailable:
            raise
        except Exception as e:
            if getattr(e, "detail", None) in {
                "candidate_privacy_restricted",
                "candidate_privacy_unavailable",
            }:
                raise CandidatePrivacyUnavailable(
                    "candidate privacy authority is unavailable"
                ) from e
            # Non-blocking: global memory sync failure must not break extraction
            logger.warning(
                "Failed to sync applicant to global memory (non-blocking)",
                extra={"node_id": node_id, "error": str(e)},
            )

    def _trigger_reembed(self, node_id: str, tenant_id: str | None) -> None:
        """Enqueue re-embed job for node."""
        decision = self._privacy_decision(node_id)
        if self._privacy_blocks_node(decision, tenant_id):
            return
        job_id = enqueue_embedding_job(
            self.redis_client,
            node_id,
            tenant_id,
            action="reembed",
            force=True,  # Force re-embed even if pending
        )
        if job_id:
            logger.info(
                "Triggered re-embed after extraction",
                extra={"node_id": node_id, "embed_job_id": job_id},
            )

    def _privacy_decision(self, node_id: str) -> CandidatePrivacyDecision:
        if self.privacy_repository is None:
            raise CandidatePrivacyUnavailable("candidate privacy authority is unavailable")
        return self.privacy_repository.node_decision(node_id)

    @staticmethod
    def _privacy_blocks_node(decision: CandidatePrivacyDecision, tenant_id: str | None) -> bool:
        """Global opt-out blocks public work while preserving owned tenant work."""
        return decision in {
            CandidatePrivacyDecision.BLOCK_ALL,
            CandidatePrivacyDecision.REVIEW,
        } or (decision is CandidatePrivacyDecision.BLOCK_GLOBAL and tenant_id is None)

    def run(self) -> None:
        """Main worker loop."""
        while self.running:
            try:
                # Move due retries to main queue
                move_due_extraction_retries(self.redis_client, limit=200)

                # Block-pop from queue
                item = self.redis_client.brpop(
                    EXTRACTION_QUEUE_KEY, timeout=int(self.poll_interval)
                )
                # A completed retry-move + BRPOP cycle, including an empty pop,
                # proves both the loop and Redis path are responsive.
                self.health_state.loop_cycle_success()
                if not item:
                    continue

                _, payload = item
                self._process_job(payload)

            except Exception as e:
                self.health_state.loop_error()
                logger.error("Worker loop error", extra={"error": str(e)})
                time.sleep(self.poll_interval)


def start_database_monitor(
    state: WorkerHealthState,
    dsn: str,
    *,
    interval_seconds: float = 30.0,
) -> threading.Thread:
    """Monitor DB readiness on a separate bounded connection every 30 seconds."""

    def monitor() -> None:
        import psycopg

        while True:
            try:
                with psycopg.connect(dsn, connect_timeout=2) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SET statement_timeout = 1000")
                        cur.execute("SELECT 1")
                        if cur.fetchone() != (1,):
                            raise RuntimeError("database readiness query failed")
                state.database_success()
            except Exception as exc:
                state.database_error()
                logger.warning(
                    "Extraction worker database readiness failed",
                    extra={"error_type": type(exc).__name__},
                )
            if state.wait(interval_seconds):
                return

    thread = threading.Thread(target=monitor, daemon=True, name="extraction-db-readiness")
    thread.start()
    return thread


def start_extraction_worker() -> None:
    """CLI entrypoint for extraction worker."""
    from activekg.common.metrics import get_redis_client
    from activekg.common.schema_control import SchemaControlError, assert_startup_schema_ready

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        dsn = assert_startup_schema_ready(require_privacy_hmac=False)
    except SchemaControlError as exc:
        logger.error("Schema readiness refused", extra={"error_type": type(exc).__name__})
        sys.exit(1)

    assert_extraction_models_configured()

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("GROQ_API_KEY not set")
        sys.exit(1)

    poll_interval = float(os.getenv("EXTRACTION_WORKER_POLL_INTERVAL", "1.0"))
    health_state = WorkerHealthState(poll_interval)
    health_state.provider_configured()

    # Start healthcheck server for Railway
    start_healthcheck_server(health_state)
    start_database_monitor(health_state, dsn)

    redis_client = get_redis_client()
    repo = GraphRepository(dsn)
    privacy_repository = CandidatePrivacyRepository(dsn)
    extraction_client = ExtractionClient(api_key=groq_key)

    worker = ExtractionWorker(
        redis_client=redis_client,
        repo=repo,
        extraction_client=extraction_client,
        health_state=health_state,
        privacy_repository=privacy_repository,
        poll_interval_seconds=poll_interval,
        max_attempts=int(os.getenv("EXTRACTION_MAX_ATTEMPTS", "2")),
        retry_base_seconds=float(os.getenv("EXTRACTION_RETRY_BASE_SECONDS", "10")),
        retry_max_seconds=float(os.getenv("EXTRACTION_RETRY_MAX_SECONDS", "60")),
    )
    worker.run()


if __name__ == "__main__":
    start_extraction_worker()
