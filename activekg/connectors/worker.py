"""Queue worker for processing S3 connector events.

Polls Redis queue for change events and processes them via IngestionProcessor.
Supports graceful shutdown and observability.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

import redis
from prometheus_client import Counter, Gauge, Histogram

from activekg.connectors.base import ChangeItem
from activekg.connectors.ingest import IngestionProcessor
from activekg.connectors.providers.s3 import S3Connector
from activekg.graph.repository import GraphRepository

if TYPE_CHECKING:
    from activekg.connectors.config_store import ConnectorConfigStore

logger = logging.getLogger(__name__)

# Prometheus metrics
worker_processed = Counter(
    "connector_worker_processed_total",
    "Total events processed by worker",
    ["tenant", "provider", "result"],
)
worker_errors = Counter(
    "connector_worker_errors_total", "Total worker errors", ["tenant", "provider", "error_type"]
)
worker_queue_depth = Gauge(
    "connector_worker_queue_depth", "Current queue depth per tenant", ["tenant", "provider"]
)
worker_batch_latency = Histogram(
    "connector_worker_batch_latency_seconds", "Time to process one batch", ["tenant", "provider"]
)


class ConnectorWorker:
    """Background worker that processes connector events from Redis queue."""

    def __init__(
        self,
        redis_client: redis.Redis,
        repo: GraphRepository,
        config_store: ConnectorConfigStore | None = None,
        batch_size: int = 10,
        poll_interval_seconds: float = 1.0,
    ):
        """Initialize worker.

        Args:
            redis_client: Redis client for queue
            repo: Graph repository
            config_store: Optional connector config store (uses global if not provided)
            batch_size: Max events to process per batch
            poll_interval_seconds: Sleep time between polls
        """
        self.redis_client = redis_client
        self.repo = repo
        self.batch_size = batch_size
        self.poll_interval = poll_interval_seconds
        self.running = True

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)

        # Config store (lazy-loaded if not provided)
        self._config_store = config_store

        logger.info(
            f"Worker initialized: batch_size={batch_size}, poll_interval={poll_interval_seconds}s"
        )

    def _shutdown_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False

    @property
    def config_store(self):
        """Get config store (lazy-loaded)."""
        if self._config_store is None:
            from activekg.connectors.config_store import get_config_store

            self._config_store = get_config_store()
        return self._config_store

    def load_s3_config(self, tenant_id: str) -> dict[str, Any] | None:
        """Load S3 connector config for tenant from database.

        Uses config store with automatic caching.

        Args:
            tenant_id: Tenant ID

        Returns:
            Decrypted config dict or None if not found/disabled
        """
        try:
            config = self.config_store.get(tenant_id, "s3")
            if config:
                logger.debug(f"Loaded S3 config for tenant {tenant_id}")
            else:
                logger.debug(f"No S3 config found for tenant {tenant_id}")
            return config
        except Exception as e:
            logger.error(f"Failed to load S3 config for {tenant_id}: {e}")
            return None

    def process_batch(self, tenant_id: str, provider: str = "s3") -> int:
        """Process one batch of events for tenant.

        Args:
            tenant_id: Tenant ID
            provider: Provider name (default: s3)

        Returns:
            Number of events processed
        """
        queue_key = f"connector:{provider}:{tenant_id}:queue"

        # Check queue depth
        depth = self.redis_client.llen(queue_key)
        worker_queue_depth.labels(tenant=tenant_id, provider=provider).set(depth)

        if depth == 0:
            return 0

        with worker_batch_latency.labels(tenant=tenant_id, provider=provider).time():
            # Pop batch
            batch = []
            for _ in range(min(self.batch_size, depth)):
                item_json = self.redis_client.rpop(queue_key)
                if item_json:
                    try:
                        batch.append(json.loads(item_json))
                    except Exception as e:
                        logger.error(f"Failed to parse queue item: {e}")
                        worker_errors.labels(
                            tenant=tenant_id, provider=provider, error_type="parse"
                        ).inc()

            if not batch:
                return 0

            logger.info(f"Processing batch: {len(batch)} events (tenant={tenant_id})")

            # Load connector config
            config = self.load_s3_config(tenant_id)
            if not config:
                logger.error(f"No S3 config found for tenant {tenant_id}")
                worker_errors.labels(
                    tenant=tenant_id, provider=provider, error_type="no_config"
                ).inc()
                return 0

            # Create connector + processor
            connector = S3Connector(tenant_id=tenant_id, config=config)
            processor = IngestionProcessor(
                connector=connector, repo=self.repo, redis_client=self.redis_client
            )

            # Convert to ChangeItems
            changes = []
            for item in batch:
                try:
                    changes.append(
                        ChangeItem(
                            uri=item["uri"],
                            operation=item.get("operation", "upsert"),
                            etag=item.get("etag"),
                            modified_at=datetime.fromisoformat(item["modified_at"])
                            if item.get("modified_at")
                            else None,
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to create ChangeItem: {e}")
                    worker_errors.labels(
                        tenant=tenant_id, provider=provider, error_type="change_item"
                    ).inc()

            # Process batch
            try:
                summary = processor.process_changes(changes)
                logger.info(f"Batch complete: {summary}")

                # Update metrics
                for result_type in ["created", "updated", "skipped", "deleted"]:
                    count = summary.get(result_type, 0)
                    if count > 0:
                        worker_processed.labels(
                            tenant=tenant_id, provider=provider, result=result_type
                        ).inc(count)

                if summary.get("errors", 0) > 0:
                    worker_errors.labels(
                        tenant=tenant_id, provider=provider, error_type="processing"
                    ).inc(summary["errors"])

                return len(batch)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}", exc_info=True)
                worker_errors.labels(
                    tenant=tenant_id, provider=provider, error_type="batch_failure"
                ).inc()
                return 0

    def run(self):
        """Main worker loop.

        Polls all known tenant queues and processes batches.
        Runs until shutdown signal received.
        """
        logger.info("Worker started")

        while self.running:
            try:
                # Get all tenant queues (scan for connector:s3:*:queue)
                cursor = 0
                tenant_queues = set()
                while True:
                    cursor, keys = self.redis_client.scan(
                        cursor=cursor, match="connector:s3:*:queue", count=100
                    )
                    for key in keys:
                        # Extract tenant_id from key
                        # Format: connector:s3:{tenant_id}:queue
                        parts = key.decode("utf-8").split(":")
                        if len(parts) == 4:
                            tenant_queues.add(parts[2])
                    if cursor == 0:
                        break

                # Process each tenant queue
                total_processed = 0
                for tenant_id in tenant_queues:
                    try:
                        count = self.process_batch(tenant_id)
                        total_processed += count
                    except Exception as e:
                        logger.error(f"Error processing tenant {tenant_id}: {e}", exc_info=True)

                # If no work done, sleep
                if total_processed == 0:
                    time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                time.sleep(self.poll_interval)

        logger.info("Worker stopped")


def start_worker():
    """CLI entry point for starting worker.

    Usage:
        python -m activekg.connectors.worker
    """
    import os

    from activekg.common.metrics import get_redis_client
    from activekg.graph.repository import GraphRepository

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Get dependencies
    redis_client = get_redis_client()
    from activekg.common.env import env_str
    dsn = env_str(["ACTIVEKG_DSN", "DATABASE_URL"])  # empty string if not set
    if not dsn:
        logger.error("ACTIVEKG_DSN/DATABASE_URL not set")
        sys.exit(1)

    repo = GraphRepository(dsn)

    # Create and run worker
    worker = ConnectorWorker(
        redis_client=redis_client,
        repo=repo,
        batch_size=int(os.getenv("CONNECTOR_WORKER_BATCH_SIZE", "10")),
        poll_interval_seconds=float(os.getenv("CONNECTOR_WORKER_POLL_INTERVAL", "1.0")),
    )

    worker.run()


if __name__ == "__main__":
    start_worker()
