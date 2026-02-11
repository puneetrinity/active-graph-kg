"""S3 connector: stat, fetch_text, list_changes for S3 URIs.

URIs: s3://bucket/key

Dependencies:
  - boto3 (already listed in requirements.txt)
  - pdfplumber / PyPDF2 / python-docx / bs4 for parsing (already present)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import boto3

from activekg.connectors.base import BaseConnector, ChangeItem, ConnectorStats, FetchResult
from activekg.connectors.extract import extract_text


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {uri}")
    path = uri[len("s3://") :]
    parts = path.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


class S3Connector(BaseConnector):
    """S3 implementation of BaseConnector.

    Config keys (validated by S3ConnectorConfig):
      - bucket, prefix, region, access_key_id, secret_access_key
    """

    def __init__(self, tenant_id: str, config: dict):
        super().__init__(tenant_id, config)
        self.s3 = boto3.client(
            "s3",
            region_name=config.get("region", os.getenv("AWS_REGION", "us-east-1")),
            aws_access_key_id=config.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=config.get("secret_access_key")
            or os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

    def stat(self, uri: str) -> ConnectorStats:
        bucket, key = _parse_s3_uri(uri)
        try:
            head = self.s3.head_object(Bucket=bucket, Key=key)
            etag = head.get("ETag", "").strip('"')
            size = int(head.get("ContentLength")) if head.get("ContentLength") is not None else None
            mime = head.get("ContentType")
            lm = head.get("LastModified")
            modified_at = lm if isinstance(lm, datetime) else None
            return ConnectorStats(
                exists=True,
                etag=etag,
                generation=None,
                modified_at=modified_at,
                size=size,
                mime_type=mime,
                owner=None,
            )
        except self.s3.exceptions.NoSuchKey:
            return ConnectorStats(exists=False)
        except Exception:
            # For permission or transient issues, surface minimal info
            return ConnectorStats(exists=True)

    def fetch_text(self, uri: str) -> FetchResult:
        bucket, key = _parse_s3_uri(uri)
        obj = self.s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read()
        content_type = obj.get("ContentType") or "application/octet-stream"
        title = os.path.basename(key)

        text = extract_text(body, content_type)
        return FetchResult(
            text=text,
            title=title,
            metadata={
                "etag": obj.get("ETag", "").strip('"'),
                "modified_at": (
                    obj.get("LastModified").astimezone(timezone.utc).isoformat()
                    if isinstance(obj.get("LastModified"), datetime)
                    else None
                ),
                "bucket": bucket,
                "key": key,
                "mime_type": content_type,
            },
        )

    def list_changes(self, cursor: str | None = None) -> tuple[list[ChangeItem], str | None]:
        """Backfill listing using ListObjectsV2.

        Cursor format: JSON string with {"ContinuationToken": "..."} or None
        """
        import json

        bucket = self.config["bucket"]
        prefix = self.config.get("prefix", "")

        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if cursor:
            try:
                data = json.loads(cursor)
                if data.get("ContinuationToken"):
                    kwargs["ContinuationToken"] = data["ContinuationToken"]
            except Exception:
                pass

        resp = self.s3.list_objects_v2(**kwargs)
        changes: list[ChangeItem] = []
        for obj in resp.get("Contents", []):
            key = obj.get("Key")
            if not key or key.endswith("/"):
                continue
            uri = f"s3://{bucket}/{key}"
            lm = obj.get("LastModified")
            etag = obj.get("ETag", "").strip('"')
            changes.append(ChangeItem(uri=uri, operation="upsert", etag=etag, modified_at=lm))

        next_cursor = None
        if resp.get("IsTruncated") and resp.get("NextContinuationToken"):
            next_cursor = json.dumps({"ContinuationToken": resp["NextContinuationToken"]})

        return changes, next_cursor

