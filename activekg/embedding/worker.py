"""Redis-backed embedding worker.

Consumes embedding jobs, generates vectors, and updates node state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import redis

from activekg.embedding.queue import (
    DLQ_KEY,
    QUEUE_KEY,
    clear_pending,
    move_due_retries,
    schedule_retry,
)
from activekg.engine.embedding_provider import EmbeddingProvider
from activekg.graph.repository import GraphRepository

logger = logging.getLogger(__name__)


def _compute_drift(old: np.ndarray | None, new: np.ndarray) -> float:
    if old is None:
        return 0.0
    denom = (float((old**2).sum()) ** 0.5) * (float((new**2).sum()) ** 0.5)
    if denom == 0:
        return 0.0
    return 1.0 - float((old @ new) / denom)


def _compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingWorker:
    def __init__(
        self,
        redis_client: redis.Redis,
        repo: GraphRepository,
        embedder: EmbeddingProvider,
        *,
        poll_interval_seconds: float = 1.0,
        max_attempts: int = 5,
        retry_base_seconds: float = 10.0,
        retry_max_seconds: float = 300.0,
    ):
        self.redis_client = redis_client
        self.repo = repo
        self.embedder = embedder
        self.poll_interval = poll_interval_seconds
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds
        self.running = True

        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        logger.info(
            "Embedding worker initialized",
            extra={
                "poll_interval": poll_interval_seconds,
                "max_attempts": max_attempts,
                "retry_base": retry_base_seconds,
                "retry_max": retry_max_seconds,
            },
        )

    def _shutdown_handler(self, signum, frame):
        logger.info("Embedding worker shutting down", extra={"signal": signum})
        self.running = False

    def _process_job(self, raw: bytes | str) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            job = cast(dict[str, Any], json.loads(raw))
        except Exception as e:
            logger.error("Invalid job payload", extra={"error": str(e)})
            return

        node_id = cast(str | None, job.get("node_id"))
        tenant_id = cast(str | None, job.get("tenant_id"))
        action = cast(str, job.get("action") or "embed")
        if not node_id:
            logger.error("Job missing node_id")
            return

        attempts = self.repo.mark_embedding_processing(node_id, tenant_id=tenant_id)
        job["attempts"] = attempts

        try:
            node = self.repo.get_node(node_id, tenant_id=tenant_id)
            if not node:
                self.repo.mark_embedding_failed(node_id, "node_not_found", tenant_id=tenant_id)
                clear_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            current_version = os.getenv("EXTRACTION_VERSION", "1.0.0")
            node_version = (node.props or {}).get("extraction_version")
            if action == "embed" and node.embedding is not None and node_version == current_version:
                # Already embedded with current extraction version
                self.repo.mark_embedding_ready(node_id, tenant_id=tenant_id)
                clear_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            text = self.repo.build_embedding_text(node)
            if not text:
                self.repo.mark_embedding_skipped(node_id, "empty_text", tenant_id=tenant_id)
                clear_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            content_hash = None
            if not (node.props or {}).get("content_hash"):
                content_hash = _compute_content_hash(text)

            new = self.embedder.encode([text])[0]
            drift = _compute_drift(node.embedding, new)
            ts = datetime.now(timezone.utc).isoformat()

            self.repo.update_node_embedding(
                node_id,
                new,
                drift,
                ts,
                tenant_id=tenant_id,
                content_hash=content_hash,
                extraction_version=current_version,
            )
            self.repo.write_embedding_history(
                node_id, drift, embedding_ref=node.payload_ref, tenant_id=tenant_id
            )

            drift_threshold = (
                node.refresh_policy.get("drift_threshold", 0.1) if node.refresh_policy else 0.1
            )
            if drift > drift_threshold:
                self.repo.append_event(
                    node_id,
                    "refreshed",
                    {"drift_score": drift, "last_refreshed": ts, "auto_embed": True},
                    tenant_id=tenant_id,
                    actor_id="embedding_worker",
                    actor_type="system",
                )

            clear_pending(self.redis_client, node_id, tenant_id=tenant_id)
        except Exception as e:
            error_msg = str(e)
            logger.error(
                "Embedding job failed",
                extra={"node_id": node_id, "tenant_id": tenant_id, "error": error_msg},
            )
            if attempts >= self.max_attempts:
                self.repo.mark_embedding_failed(node_id, error_msg, tenant_id=tenant_id)
                job["error"] = error_msg
                job["failed_at"] = time.time()
                self.redis_client.lpush(DLQ_KEY, json.dumps(job))
                clear_pending(self.redis_client, node_id, tenant_id=tenant_id)
                return

            delay = min(
                self.retry_base_seconds * (2 ** max(0, attempts - 1)), self.retry_max_seconds
            )
            self.repo.mark_embedding_queued(node_id, tenant_id=tenant_id)
            job["error"] = error_msg
            schedule_retry(self.redis_client, job, delay_seconds=delay)

    def run(self) -> None:
        while self.running:
            try:
                move_due_retries(self.redis_client, limit=200)
                item = self.redis_client.brpop(QUEUE_KEY, timeout=int(self.poll_interval))
                if not item:
                    continue
                _, payload = item
                self._process_job(payload)
            except Exception as e:
                logger.error("Worker loop error", extra={"error": str(e)})
                time.sleep(self.poll_interval)


def start_worker() -> None:
    """CLI entrypoint for embedding worker."""
    from activekg.common.env import env_str
    from activekg.common.metrics import get_redis_client

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    dsn = env_str(["ACTIVEKG_DSN", "DATABASE_URL"])
    if not dsn:
        logger.error("ACTIVEKG_DSN/DATABASE_URL not set")
        sys.exit(1)

    redis_client = get_redis_client()
    repo = GraphRepository(dsn)
    embedder = EmbeddingProvider(
        backend=os.getenv("EMBEDDING_BACKEND", "sentence-transformers"),
        model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
    )

    worker = EmbeddingWorker(
        redis_client=redis_client,
        repo=repo,
        embedder=embedder,
        poll_interval_seconds=float(os.getenv("EMBEDDING_WORKER_POLL_INTERVAL", "1.0")),
        max_attempts=int(os.getenv("EMBEDDING_MAX_ATTEMPTS", "5")),
        retry_base_seconds=float(os.getenv("EMBEDDING_RETRY_BASE_SECONDS", "10")),
        retry_max_seconds=float(os.getenv("EMBEDDING_RETRY_MAX_SECONDS", "300")),
    )
    worker.run()


if __name__ == "__main__":
    start_worker()
