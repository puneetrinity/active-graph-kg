"""Queue worker for processing connector events (S3, GCS, Drive).

Polls Redis queues for change events and processes them via IngestionProcessor.
Supports graceful shutdown and observability.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import redis
from prometheus_client import Counter, Gauge, Histogram

from activekg.connectors.base import ChangeItem
from activekg.connectors.ingest import IngestionProcessor
from activekg.connectors.providers.drive import DriveConnector
from activekg.connectors.providers.gcs import GCSConnector
from activekg.connectors.providers.s3 import S3Connector
from activekg.connectors.types import ConnectorProvider
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

SUPPORTED_PROVIDERS: tuple[ConnectorProvider, ...] = ("s3", "gcs", "drive")


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
        self.supported_providers = SUPPORTED_PROVIDERS

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

    def load_config(self, tenant_id: str, provider: ConnectorProvider) -> dict[str, Any] | None:
        """Load connector config for tenant from database.

        Uses config store with automatic caching.

        Args:
            tenant_id: Tenant ID
            provider: Connector provider

        Returns:
            Decrypted config dict or None if not found/disabled
        """
        try:
            config = self.config_store.get(tenant_id, provider)
            if config:
                logger.debug(f"Loaded {provider} config for tenant {tenant_id}")
            else:
                logger.debug(f"No {provider} config found for tenant {tenant_id}")
            return config
        except Exception as e:
            logger.error(f"Failed to load {provider} config for {tenant_id}: {e}")
            return None

    def _build_connector(self, provider: ConnectorProvider, tenant_id: str, config: dict[str, Any]):
        """Instantiate the right connector for the provider."""
        if provider == "s3":
            return S3Connector(tenant_id=tenant_id, config=config)
        if provider == "gcs":
            return GCSConnector(tenant_id=tenant_id, config=config)
        if provider == "drive":
            return DriveConnector(tenant_id=tenant_id, config=config)

        raise ValueError(f"Unsupported provider: {provider}")

    def process_batch(self, tenant_id: str, provider: ConnectorProvider = "s3") -> int:
        """Process one batch of events for tenant.

        Args:
            tenant_id: Tenant ID
            provider: Provider name (default: s3)

        Returns:
            Number of events processed
        """
        queue_key = f"connector:{provider}:{tenant_id}:queue"

        # Check queue depth
        depth = cast(int, self.redis_client.llen(queue_key))
        worker_queue_depth.labels(tenant=tenant_id, provider=provider).set(float(depth))

        if depth == 0:
            return 0

        with worker_batch_latency.labels(tenant=tenant_id, provider=provider).time():
            # Pop batch
            batch = []
            for _ in range(min(self.batch_size, depth)):
                item_json = cast(bytes | None, self.redis_client.rpop(queue_key))
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
            config = self.load_config(tenant_id, provider)
            if not config:
                logger.error(f"No {provider} config found for tenant {tenant_id}")
                worker_errors.labels(
                    tenant=tenant_id, provider=provider, error_type="no_config"
                ).inc()
                return 0

            # Create connector + processor
            try:
                connector = self._build_connector(provider, tenant_id, config)
            except Exception as e:
                logger.error(f"Failed to build connector for {provider}: {e}")
                worker_errors.labels(
                    tenant=tenant_id, provider=provider, error_type="build_connector"
                ).inc()
                return 0

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
                # Get all tenant queues (scan per provider)
                provider_queues: dict[str, set[str]] = {p: set() for p in self.supported_providers}
                for provider in self.supported_providers:
                    cursor = 0
                    pattern = f"connector:{provider}:*:queue"
                    while True:
                        cursor, keys = self.redis_client.scan(
                            cursor=cursor, match=pattern, count=100
                        )
                        for key in keys:
                            # Extract tenant_id from key
                            # Format: connector:{provider}:{tenant_id}:queue
                            parts = key.decode("utf-8").split(":")
                            if len(parts) == 4:
                                provider_queues[provider].add(parts[2])
                        if cursor == 0:
                            break

                # Process each tenant queue per provider
                total_processed = 0
                for provider, tenants in provider_queues.items():
                    for tenant_id in tenants:
                        try:
                            count = self.process_batch(tenant_id, provider)
                            total_processed += count
                        except Exception as e:
                            logger.error(
                                f"Error processing tenant {tenant_id} for {provider}: {e}",
                                exc_info=True,
                            )

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
