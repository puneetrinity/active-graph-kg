"""Google Cloud Storage connector implementation."""

import json
import os
from typing import Any

from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage

from activekg.common.logger import get_enhanced_logger
from activekg.connectors.base import BaseConnector, ChangeItem, ConnectorStats, FetchResult
from activekg.connectors.extract import extract_text

logger = get_enhanced_logger(__name__)


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/object into (bucket, object_name)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {uri}")
    parts = uri[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"GCS URI must be gs://bucket/object: {uri}")
    return parts[0], parts[1]


class GCSConnector(BaseConnector):
    """Google Cloud Storage implementation of BaseConnector.

    Config keys (validated by GCSConnectorConfig):
      - bucket: GCS bucket name
      - prefix: Optional prefix to filter objects (default: "")
      - project: GCP project ID (optional, can use GOOGLE_CLOUD_PROJECT env)
      - credentials_json: Inline service account JSON (for production/PaaS)
      - service_account_json_path: Path to service account JSON file

    Authentication (checked in order):
      1. credentials_json in config (inline JSON string)
      2. GOOGLE_CREDENTIALS_JSON env var (inline JSON string)
      3. service_account_json_path in config (file path)
      4. GOOGLE_APPLICATION_CREDENTIALS env var (file path)
      5. Default credentials (gcloud CLI, workload identity)
    """

    def __init__(self, tenant_id: str, config: dict[str, Any]):
        super().__init__(tenant_id, config)

        # Get credentials (priority: file path > inline JSON > default)
        # File path is more reliable than inline JSON with dotenv
        credentials_path = config.get("service_account_json_path") or os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        )
        credentials_json = config.get("credentials_json") or os.getenv("GOOGLE_CREDENTIALS_JSON")
        project = config.get("project") or os.getenv("GOOGLE_CLOUD_PROJECT")

        # Debug logging
        logger.info(f"GCS credentials_path: {credentials_path}")
        logger.info(f"GCS project: {project}")

        # Initialize GCS client (prefer file path over inline JSON)
        if credentials_path:
            self.client = storage.Client.from_service_account_json(
                credentials_path, project=project
            )
            logger.info("GCS connector using credentials from file path")
        else:
            # Use default credentials (gcloud, workload identity, etc.)
            self.client = storage.Client(project=project)
            logger.info("GCS connector using default credentials")

        logger.info(
            "GCS connector initialized",
            extra_fields={
                "tenant_id": tenant_id,
                "bucket": config.get("bucket"),
                "prefix": config.get("prefix", ""),
                "project": project or "default",
            },
        )

    def stat(self, uri: str) -> ConnectorStats:
        """Get metadata about a GCS object without downloading it.

        Args:
            uri: GCS URI in format gs://bucket/object

        Returns:
            ConnectorStats with metadata (uses 'generation' field for GCS versioning)
        """
        bucket_name, object_name = _parse_gcs_uri(uri)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.get_blob(object_name)

            if blob is None:
                return ConnectorStats(exists=False)

            # GCS uses etag and generation for versioning
            # generation is a unique version identifier
            etag = blob.etag
            generation = str(blob.generation) if blob.generation else None
            size = blob.size
            mime_type = blob.content_type
            updated = blob.updated  # datetime object
            owner = blob.owner.get("entity") if blob.owner else None

            return ConnectorStats(
                exists=True,
                etag=etag,
                generation=generation,  # GCS-specific versioning
                modified_at=updated,
                size=size,
                mime_type=mime_type,
                owner=owner,
            )
        except gcp_exceptions.NotFound:
            return ConnectorStats(exists=False)
        except Exception as e:
            logger.error(
                f"GCS stat failed for {uri}: {e}",
                extra_fields={"tenant_id": self.tenant_id, "uri": uri},
            )
            raise

    def fetch_text(self, uri: str) -> FetchResult:
        """Download and extract text from a GCS object.

        Args:
            uri: GCS URI in format gs://bucket/object

        Returns:
            FetchResult with extracted text and metadata
        """
        bucket_name, object_name = _parse_gcs_uri(uri)

        try:
            bucket = self.client.bucket(bucket_name)
            blob = bucket.blob(object_name)

            # Download as bytes
            data = blob.download_as_bytes()
            content_type = blob.content_type or ""

            # Extract text based on content type
            text = extract_text(data, content_type)

            # Use object name as title (last component of path)
            title = object_name.split("/")[-1] if object_name else None

            metadata = {
                "bucket": bucket_name,
                "object": object_name,
                "content_type": content_type,
                "size": len(data),
                "generation": str(blob.generation) if blob.generation else None,
            }

            logger.info(
                "GCS fetch successful",
                extra_fields={
                    "tenant_id": self.tenant_id,
                    "uri": uri,
                    "text_length": len(text),
                    "content_type": content_type,
                },
            )

            return FetchResult(text=text, title=title, metadata=metadata)

        except gcp_exceptions.NotFound as e:
            logger.error(
                f"GCS object not found: {uri}",
                extra_fields={"tenant_id": self.tenant_id, "uri": uri},
            )
            raise FileNotFoundError(f"GCS object not found: {uri}") from e
        except Exception as e:
            logger.error(
                f"GCS fetch failed for {uri}: {e}",
                extra_fields={"tenant_id": self.tenant_id, "uri": uri},
            )
            raise

    def list_changes(self, cursor: str | None = None) -> tuple[list[ChangeItem], str | None]:
        """List new/modified objects in the configured bucket/prefix.

        Uses a high-water mark timestamp to avoid re-queuing objects that have
        already been ingested.  On the very first call (cursor=None) a full
        backfill is performed.  Once all pages are consumed the maximum
        ``modified_at`` seen is persisted in the cursor so that subsequent
        polls only return objects modified after that point.

        Cursor format (JSON):
            - First run / backfill in progress:
              ``{"page_token": "...", "high_water": "..."}``
            - Incremental (backfill done):
              ``{"high_water": "<ISO-8601>"}``

        Args:
            cursor: JSON cursor string or None (first run).

        Returns:
            Tuple of (list of ChangeItems, next_cursor).
        """
        from datetime import datetime, timezone

        bucket_name = self.config["bucket"]
        prefix = self.config.get("prefix", "")
        max_results = 1000  # GCS default page size

        # Parse cursor
        page_token = None
        high_water = None  # datetime – only include blobs modified *after* this
        if cursor:
            try:
                data = json.loads(cursor)
                page_token = data.get("page_token")
                hw_raw = data.get("high_water")
                if hw_raw:
                    high_water = datetime.fromisoformat(hw_raw)
                    # Ensure timezone-aware for comparison
                    if high_water.tzinfo is None:
                        high_water = high_water.replace(tzinfo=timezone.utc)
            except Exception as e:
                logger.warning(
                    f"Invalid cursor format: {e}",
                    extra_fields={"tenant_id": self.tenant_id, "cursor": cursor},
                )

        try:
            bucket = self.client.bucket(bucket_name)

            # List blobs with pagination
            iterator = bucket.list_blobs(
                prefix=prefix, max_results=max_results, page_token=page_token
            )

            # Get current page
            page = next(iterator.pages)

            changes: list[ChangeItem] = []
            max_modified = high_water  # track the max across all pages
            for blob in page:
                # Skip directory markers (objects ending with /)
                if blob.name.endswith("/"):
                    continue

                updated = blob.updated  # datetime (tz-aware from GCS)

                # Incremental filter: skip objects not modified since last scan
                if high_water and updated and updated <= high_water:
                    continue

                uri = f"gs://{bucket_name}/{blob.name}"
                etag = blob.etag

                changes.append(
                    ChangeItem(
                        uri=uri,
                        operation="upsert",
                        etag=etag,
                        modified_at=updated,
                    )
                )

                # Track high-water mark
                if updated and (max_modified is None or updated > max_modified):
                    max_modified = updated

            # Build next cursor
            next_page_token = iterator.next_page_token
            if next_page_token:
                # More pages to go – carry high_water through pagination
                cursor_data: dict = {"page_token": next_page_token}
                if max_modified:
                    cursor_data["high_water"] = max_modified.isoformat()
                next_cursor = json.dumps(cursor_data)
            else:
                # All pages consumed – persist high-water mark for incremental
                if max_modified:
                    next_cursor = json.dumps({"high_water": max_modified.isoformat()})
                else:
                    # Nothing seen at all (empty bucket / prefix)
                    next_cursor = cursor  # keep previous cursor as-is

            logger.info(
                "GCS list_changes complete",
                extra_fields={
                    "tenant_id": self.tenant_id,
                    "bucket": bucket_name,
                    "prefix": prefix,
                    "changes_count": len(changes),
                    "has_more": next_page_token is not None,
                    "high_water": max_modified.isoformat() if max_modified else None,
                },
            )

            return changes, next_cursor

        except Exception as e:
            logger.error(
                f"GCS list_changes failed: {e}",
                extra_fields={"tenant_id": self.tenant_id, "bucket": bucket_name, "prefix": prefix},
            )
            raise

