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

import redis

from activekg.embedding.queue import enqueue_embedding_job
from activekg.extraction.client import ExtractionClient, ExtractionError
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

logger = logging.getLogger(__name__)

# Healthcheck port (configurable via env)
HEALTHCHECK_PORT = int(os.getenv("EXTRACTION_HEALTHCHECK_PORT", "8080"))


class HealthCheckHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for healthcheck endpoint."""

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"healthy","service":"extraction-worker"}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def start_healthcheck_server() -> HTTPServer:
    """Start healthcheck HTTP server in background thread."""
    server = HTTPServer(("0.0.0.0", HEALTHCHECK_PORT), HealthCheckHandler)
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

        # Mark as processing
        self._update_extraction_status(node_id, tenant_id, ExtractionStatus(status="processing"))

        try:
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

            self._update_node_props(node_id, tenant_id, all_props)

            logger.info(
                "Extraction completed",
                extra={
                    "node_id": node_id,
                    "model": model_used,
                    "confidence": result.confidence,
                    "skills_count": len(result.primary_skills),
                    "titles_count": len(result.recent_job_titles),
                },
            )

            # Trigger re-embed if extraction version differs from embedding version
            node_embed_version = (node.props or {}).get("extraction_version")
            if node_embed_version != current_version:
                self._trigger_reembed(node_id, tenant_id)

            clear_extraction_pending(self.redis_client, node_id, tenant_id=tenant_id)

        except ExtractionError as e:
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
        self, node_id: str, tenant_id: str | None, status: ExtractionStatus
    ) -> None:
        """Update extraction status in node props."""
        self._update_node_props(node_id, tenant_id, status.to_props())

    def _update_node_props(
        self, node_id: str, tenant_id: str | None, props: dict[str, Any]
    ) -> None:
        """Merge props into node."""
        with self.repo._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE nodes
                    SET props = COALESCE(props, '{}'::jsonb) || %s::jsonb,
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (json.dumps(props), node_id),
                )

    def _trigger_reembed(self, node_id: str, tenant_id: str | None) -> None:
        """Enqueue re-embed job for node."""
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
                if not item:
                    continue

                _, payload = item
                self._process_job(payload)

            except Exception as e:
                logger.error("Worker loop error", extra={"error": str(e)})
                time.sleep(self.poll_interval)


def start_extraction_worker() -> None:
    """CLI entrypoint for extraction worker."""
    from activekg.common.env import env_str
    from activekg.common.metrics import get_redis_client

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    dsn = env_str(["ACTIVEKG_DSN", "DATABASE_URL"])
    if not dsn:
        logger.error("ACTIVEKG_DSN/DATABASE_URL not set")
        sys.exit(1)

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        logger.error("GROQ_API_KEY not set")
        sys.exit(1)

    # Start healthcheck server for Railway
    start_healthcheck_server()

    redis_client = get_redis_client()
    repo = GraphRepository(dsn)
    extraction_client = ExtractionClient(api_key=groq_key)

    worker = ExtractionWorker(
        redis_client=redis_client,
        repo=repo,
        extraction_client=extraction_client,
        poll_interval_seconds=float(os.getenv("EXTRACTION_WORKER_POLL_INTERVAL", "1.0")),
        max_attempts=int(os.getenv("EXTRACTION_MAX_ATTEMPTS", "2")),
        retry_base_seconds=float(os.getenv("EXTRACTION_RETRY_BASE_SECONDS", "10")),
        retry_max_seconds=float(os.getenv("EXTRACTION_RETRY_MAX_SECONDS", "60")),
    )
    worker.run()


if __name__ == "__main__":
    start_extraction_worker()
