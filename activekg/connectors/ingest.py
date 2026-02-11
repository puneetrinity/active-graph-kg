"""Ingest processor: stat → fetch → chunk → embed with throttle + DLQ."""

import logging

import redis
from prometheus_client import Counter, Histogram

from activekg.connectors.base import BaseConnector, ChangeItem
from activekg.connectors.chunker import create_chunk_nodes
from activekg.connectors.retry import PermanentError, TransientError, with_retry_and_dlq
from activekg.connectors.throttle import IngestionThrottle
from activekg.embedding.queue import enqueue_embedding_job
from activekg.graph.repository import GraphRepository

logger = logging.getLogger(__name__)

# Prometheus metrics
ingest_total = Counter("connector_ingest_total", "Total documents ingested", ["provider", "tenant"])
ingest_errors = Counter(
    "connector_ingest_errors_total", "Total ingestion errors", ["provider", "tenant", "reason"]
)
ingest_latency = Histogram(
    "connector_ingest_latency_seconds", "Time to ingest one document", ["provider", "tenant"]
)


class IngestionProcessor:
    """Process connector changes: stat → fetch → chunk → embed."""

    def __init__(
        self,
        connector: BaseConnector,
        repo: GraphRepository,
        redis_client: redis.Redis,
        throttle_max_per_sec: int = 50,
    ):
        """Initialize processor.

        Args:
            connector: Connector instance (S3, GCS, etc.)
            repo: Graph repository for DB operations
            redis_client: Redis client for throttle + DLQ
            throttle_max_per_sec: Max documents per second per tenant
        """
        self.connector = connector
        self.repo = repo
        self.redis_client = redis_client
        self.throttle = IngestionThrottle(redis_client, throttle_max_per_sec)
        self.provider_name = connector.provider_name
        self.tenant_id = connector.tenant_id

    def process_changes(self, changes: list[ChangeItem]) -> dict:
        """Process a batch of changes.

        Args:
            changes: List of ChangeItem from list_changes()

        Returns:
            Summary dict with counts
        """
        summary = {"processed": 0, "created": 0, "updated": 0, "skipped": 0, "errors": 0}

        for change in changes:
            try:
                result = self.process_one(change)
                summary[result] += 1
                summary["processed"] += 1
            except Exception as e:
                logger.error(f"Failed to process {change.uri}: {e}")
                summary["errors"] += 1
                ingest_errors.labels(
                    provider=self.provider_name, tenant=self.tenant_id, reason="exception"
                ).inc()

        return summary

    def process_one(self, change: ChangeItem) -> str:
        """Process a single change item.

        Args:
            change: ChangeItem (uri, operation, etag, modified_at)

        Returns:
            "created", "updated", "skipped", or "deleted"
        """
        # Throttle
        self.throttle.acquire(self.tenant_id, self.provider_name)

        # Handle deletions
        if change.operation == "deleted":
            return self._handle_deletion(change.uri)

        # stat → check if refresh needed
        with ingest_latency.labels(provider=self.provider_name, tenant=self.tenant_id).time():
            stats = self._stat_with_retry(change.uri)

            if not stats.exists:
                logger.warning(f"Resource no longer exists: {change.uri}")
                return "skipped"

            # Check if node already exists
            external_id = self.connector.to_external_id(change.uri)
            existing = self._get_existing_node(external_id)

            # Dedup: ETag check first
            if existing and stats.etag:
                if existing.get("props", {}).get("etag") == stats.etag:
                    logger.info(f"Skipping {change.uri} - ETag unchanged")
                    return "skipped"

            # fetch_text → content
            fetch_result = self._fetch_with_retry(change.uri)

            # Dedup: content hash fallback (if ETag changed but content identical)
            content_hash = self.connector.compute_content_hash(fetch_result.text)
            if existing and existing.get("props", {}).get("content_hash") == content_hash:
                # Update metadata only (ETag, modified_at)
                self._update_metadata_only(external_id, stats, content_hash)
                logger.info(f"Skipping {change.uri} - content hash unchanged")
                return "skipped"

            # Ingest: create parent + chunks
            self._ingest_document(
                external_id=external_id,
                title=fetch_result.title or change.uri.split("/")[-1],
                text=fetch_result.text,
                metadata={
                    **fetch_result.metadata,
                    "etag": stats.etag,
                    "modified_at": stats.modified_at.isoformat() if stats.modified_at else None,
                    "content_hash": content_hash,
                    "mime_type": stats.mime_type,
                    "size": stats.size,
                },
            )

            ingest_total.labels(provider=self.provider_name, tenant=self.tenant_id).inc()

            return "created" if not existing else "updated"

    def _stat_with_retry(self, uri: str):
        """Stat with retry wrapper."""

        @with_retry_and_dlq(self.redis_client, self.provider_name, self.tenant_id, max_attempts=3)
        def _stat():
            try:
                return self.connector.stat(uri)
            except Exception as e:
                # Classify error
                if "404" in str(e) or "NoSuchKey" in str(e):
                    raise PermanentError(f"Not found: {uri}")
                elif "403" in str(e) or "Access Denied" in str(e):
                    raise PermanentError(f"Access denied: {uri}")
                else:
                    raise TransientError(f"Transient error: {e}")

        return _stat()

    def _fetch_with_retry(self, uri: str):
        """Fetch with retry wrapper."""

        @with_retry_and_dlq(self.redis_client, self.provider_name, self.tenant_id, max_attempts=3)
        def _fetch():
            try:
                return self.connector.fetch_text(uri)
            except Exception as e:
                if "404" in str(e) or "NoSuchKey" in str(e):
                    raise PermanentError(f"Not found: {uri}")
                else:
                    raise TransientError(f"Transient error: {e}")

        return _fetch()

    def _get_existing_node(self, external_id: str) -> dict | None:
        """Get existing node by external_id."""
        # Query nodes table for external_id
        with self.repo._conn(tenant_id=self.tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, props FROM nodes WHERE props->>'external_id' = %s LIMIT 1",
                    (external_id,),
                )
                row = cur.fetchone()
                return {"id": row[0], "props": row[1]} if row else None

    def _update_metadata_only(self, external_id: str, stats, content_hash: str):
        """Update metadata without re-embedding."""
        with self.repo._conn(tenant_id=self.tenant_id) as conn:
            with conn.cursor() as cur:
                # Update parent node metadata
                cur.execute(
                    """UPDATE nodes SET props = props || %s::jsonb
                       WHERE props->>'external_id' = %s AND props->>'is_parent' = 'true'""",
                    (
                        {
                            "etag": stats.etag,
                            "modified_at": stats.modified_at.isoformat()
                            if stats.modified_at
                            else None,
                            "content_hash": content_hash,
                        },
                        external_id,
                    ),
                )
                conn.commit()

    def _ingest_document(self, external_id: str, title: str, text: str, metadata: dict):
        """Create parent + chunk nodes."""
        # Delete old chunks if updating
        self._delete_existing_chunks(external_id)

        # Create parent + chunks
        chunk_ids = create_chunk_nodes(
            parent_node_id=external_id,
            parent_title=title,
            parent_classes=["Document"],
            text=text,
            parent_metadata=metadata,
            repo=self.repo,
            tenant_id=self.tenant_id,
        )

        # Enqueue embedding jobs for each chunk
        enqueued = 0
        for chunk_uuid in chunk_ids:
            try:
                job_id = enqueue_embedding_job(self.redis_client, chunk_uuid, self.tenant_id)
                if job_id:
                    enqueued += 1
            except Exception as e:
                logger.warning(f"Failed to enqueue embedding for {chunk_uuid}: {e}")

        logger.info(f"Ingested {external_id}: {len(chunk_ids)} chunks created, {enqueued} embedding jobs enqueued")

    def _delete_existing_chunks(self, parent_external_id: str):
        """Delete existing chunk nodes for parent."""
        with self.repo._conn(tenant_id=self.tenant_id) as conn:
            with conn.cursor() as cur:
                # Delete chunks (external_id pattern: parent#chunk0, parent#chunk1, ...)
                cur.execute(
                    """DELETE FROM nodes
                       WHERE props->>'external_id' LIKE %s AND props->>'parent_id' = %s""",
                    (f"{parent_external_id}#chunk%", parent_external_id),
                )
                conn.commit()

    def _handle_deletion(self, uri: str) -> str:
        """Handle deleted resource (soft delete with grace period)."""
        external_id = self.connector.to_external_id(uri)

        with self.repo._conn(tenant_id=self.tenant_id) as conn:
            with conn.cursor() as cur:
                # Add "Deleted" class + deletion timestamp
                from datetime import datetime, timedelta

                grace_until = (datetime.utcnow() + timedelta(days=30)).isoformat()

                cur.execute(
                    """UPDATE nodes
                       SET classes = ARRAY_APPEND(classes, 'Deleted'),
                           props = props || %s::jsonb
                       WHERE props->>'external_id' = %s AND props->>'is_parent' = 'true'""",
                    (
                        {
                            "deleted_at": datetime.utcnow().isoformat(),
                            "deletion_grace_until": grace_until,
                        },
                        external_id,
                    ),
                )
                conn.commit()

        logger.info(f"Soft-deleted {uri} (grace until {grace_until})")
        return "deleted"
