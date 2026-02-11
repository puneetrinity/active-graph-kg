from __future__ import annotations

"""Admin endpoints to manage connectors with database-backed persistence.

Provides:
- POST /s3/register - Register/update S3 connector config (encrypted in DB)
- POST /gcs/register - Register/update GCS connector config (encrypted in DB)
- POST /drive/register - Register/update Drive connector config (encrypted in DB)
- GET / - List all connector configs
- GET /{provider} - Get specific connector config metadata
- POST /{provider}/enable - Enable connector
- POST /{provider}/disable - Disable connector
- POST /{provider}/backfill - List files without queuing (dry run)
- POST /{provider}/ingest - Queue files for ingestion (async processing)
- GET /{provider}/queue-status - Monitor ingestion queue depth
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from activekg.api.auth import JWT_ENABLED, JWTClaims, get_jwt_claims
from activekg.connectors import (
    DriveConnectorConfig,
    GCSConnectorConfig,
    S3Connector,
    S3ConnectorConfig,
)

try:
    from activekg.connectors.providers.gcs import GCSConnector
except Exception:
    GCSConnector = None  # type: ignore
try:
    from activekg.connectors.providers.drive import DriveConnector
except Exception:
    DriveConnector = None  # type: ignore
from activekg.connectors.config_store import get_config_store

router = APIRouter(prefix="/_admin/connectors", tags=["connectors-admin"])


class RegisterS3Request(BaseModel):
    tenant_id: str | None = None
    config: S3ConnectorConfig


class EnableDisableRequest(BaseModel):
    tenant_id: str


@router.post("/s3/register")
def register_s3(req: RegisterS3Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    """Register or update S3 connector config (persisted to DB with encryption)."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    tenant_id = req.tenant_id or (claims.tenant_id if claims else None) or "default"

    # Save to database with encryption
    store = get_config_store()
    success = store.upsert(
        tenant_id=tenant_id,
        provider="s3",
        config=req.config.model_dump(),
        enabled=req.config.enabled,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save connector config")

    return {"status": "registered", "tenant_id": tenant_id, "provider": "s3"}


@router.post("/gcs/register")
def register_gcs(req: RegisterGCSRequest, claims: JWTClaims | None = Depends(get_jwt_claims)):
    """Register or update GCS connector config (persisted to DB with encryption)."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    tenant_id = req.tenant_id or (claims.tenant_id if claims else None) or "default"

    store = get_config_store()
    success = store.upsert(
        tenant_id=tenant_id,
        provider="gcs",
        config=req.config.model_dump(),
        enabled=req.config.enabled,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save connector config")

    return {"status": "registered", "tenant_id": tenant_id, "provider": "gcs"}


class RegisterDriveRequest(BaseModel):
    tenant_id: str | None = None
    config: DriveConnectorConfig


@router.post("/drive/register")
def register_drive(req: RegisterDriveRequest, claims: JWTClaims | None = Depends(get_jwt_claims)):
    """Register or update Drive connector config (persisted to DB with encryption)."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    tenant_id = req.tenant_id or (claims.tenant_id if claims else None) or "default"

    store = get_config_store()
    success = store.upsert(
        tenant_id=tenant_id,
        provider="drive",
        config=req.config.model_dump(),
        enabled=req.config.enabled,
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to save connector config")

    return {"status": "registered", "tenant_id": tenant_id, "provider": "drive"}


@router.get("/")
def list_connectors(claims: JWTClaims | None = Depends(get_jwt_claims)):
    """List all connector configs (metadata only, no secrets)."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    store = get_config_store()
    configs = store.list_all()
    return {"connectors": configs}


@router.get("/{provider}")
def get_connector(
    provider: str, tenant_id: str, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    """Get connector config metadata (no secrets) for specific tenant+provider."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    store = get_config_store()
    config = store.get(tenant_id, provider)

    if not config:
        raise HTTPException(status_code=404, detail=f"Connector not found: {tenant_id}/{provider}")

    # Return metadata only (sanitize secrets)
    from activekg.connectors.encryption import sanitize_config_for_logging

    return {
        "tenant_id": tenant_id,
        "provider": provider,
        "config": sanitize_config_for_logging(config),
    }


@router.post("/{provider}/enable")
def enable_connector(
    provider: str, req: EnableDisableRequest, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    """Enable connector for tenant."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    store = get_config_store()
    success = store.set_enabled(req.tenant_id, provider, enabled=True)

    if not success:
        raise HTTPException(
            status_code=404, detail=f"Connector not found: {req.tenant_id}/{provider}"
        )

    return {"status": "enabled", "tenant_id": req.tenant_id, "provider": provider}


@router.post("/{provider}/disable")
def disable_connector(
    provider: str, req: EnableDisableRequest, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    """Disable connector for tenant."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    store = get_config_store()
    success = store.set_enabled(req.tenant_id, provider, enabled=False)

    if not success:
        raise HTTPException(
            status_code=404, detail=f"Connector not found: {req.tenant_id}/{provider}"
        )

    return {"status": "disabled", "tenant_id": req.tenant_id, "provider": provider}


class BackfillRequest(BaseModel):
    tenant_id: str | None = None
    limit: int = 200


@router.post("/{provider}/backfill")
def backfill(
    provider: str, req: BackfillRequest, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    """Trigger manual backfill for a connector provider (s3|gcs)."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    tenant_id = req.tenant_id or (claims.tenant_id if claims else None) or "default"

    # Load config from database
    store = get_config_store()
    cfg = store.get(tenant_id, provider)

    if not cfg:
        raise HTTPException(
            status_code=400, detail=f"{provider} connector not registered for tenant"
        )

    # Run a single list_changes page as a lightweight backfill
    if provider == "s3":
        connector = S3Connector(tenant_id=tenant_id, config=cfg)
    elif provider == "gcs" and GCSConnector is not None:
        connector = GCSConnector(tenant_id=tenant_id, config=cfg)  # type: ignore
    elif provider == "drive" and DriveConnector is not None:
        connector = DriveConnector(tenant_id=tenant_id, config=cfg)  # type: ignore
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")
    changes, next_cursor = connector.list_changes(cursor=None)
    return {"status": "ok", "found": len(changes), "next_cursor": next_cursor}


@router.get("/drive/cursor")
def get_drive_cursor(tenant_id: str, claims: JWTClaims | None = Depends(get_jwt_claims)):
    """Get current Drive connector cursor status for a tenant."""
    if JWT_ENABLED and (not claims or "admin:refresh" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="admin scope required")

    import psycopg
    from psycopg.rows import dict_row

    # Get updated_at timestamp from database
    from activekg.common.env import env_str

    dsn = env_str(["ACTIVEKG_DSN", "DATABASE_URL"])  # empty string if not set
    if not dsn:
        raise HTTPException(status_code=500, detail="ACTIVEKG_DSN/DATABASE_URL not configured")

    try:
        with psycopg.connect(dsn, row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT cursor, updated_at
                    FROM connector_cursors
                    WHERE tenant_id = %s AND provider = %s
                    """,
                    (tenant_id, "drive"),
                )
                row = cur.fetchone()

                if not row:
                    return {
                        "tenant_id": tenant_id,
                        "provider": "drive",
                        "cursor": None,
                        "updated_at": None,
                    }

                return {
                    "tenant_id": tenant_id,
                    "provider": "drive",
                    "cursor": row["cursor"],
                    "updated_at": row["updated_at"].isoformat() if row["updated_at"] else None,
                }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}") from e


class RegisterGCSRequest(BaseModel):
    tenant_id: str | None = None
    config: GCSConnectorConfig


class IngestRequest(BaseModel):
    tenant_id: str | None = None
    max_items: int = Field(default=1000, ge=1, le=10000, description="Max total items to process")
    batch_size: int = Field(default=100, ge=1, le=500, description="Items per batch")
    cursor: str | None = None  # For pagination through large buckets
    dry_run: bool = Field(default=True, description="Preview only, don't actually queue")
    skip_existing: bool = Field(default=True, description="Skip already processed/queued items")


@router.post("/{provider}/ingest")
def ingest(
    provider: str, req: IngestRequest, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    """List files from connector and queue them for ingestion.

    This endpoint lists files from the configured bucket/folder and pushes them
    to the Redis queue for async processing by the ConnectorWorker.

    Safeguards:
    - dry_run=True (default): Preview what would be queued without actually queuing
    - max_items: Cap total items per request (default 1000, max 10000)
    - batch_size: Process in chunks (default 100, max 500)
    - skip_existing: Dedupe against already queued/processed items
    - Requires super_admin scope

    Returns:
        job_id: Unique identifier for this ingest job
        queued_count: Number of items actually queued
        skipped_count: Number of items skipped (already processed/queued)
        total_found: Total items found in source
        dry_run: Whether this was a preview only
        next_cursor: Cursor for next page (null if done)
    """
    import json
    import uuid

    import psycopg
    from psycopg.rows import dict_row

    from activekg.common.env import env_str
    from activekg.common.metrics import get_redis_client

    # Require super_admin scope
    if JWT_ENABLED and (not claims or "super_admin" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="super_admin scope required")

    tenant_id = req.tenant_id or (claims.tenant_id if claims else None) or "default"
    job_id = str(uuid.uuid4())

    # Load config from database
    store = get_config_store()
    cfg = store.get(tenant_id, provider)

    if not cfg:
        raise HTTPException(
            status_code=400, detail=f"{provider} connector not registered for tenant"
        )

    # Build connector
    if provider == "s3":
        connector = S3Connector(tenant_id=tenant_id, config=cfg)
    elif provider == "gcs" and GCSConnector is not None:
        connector = GCSConnector(tenant_id=tenant_id, config=cfg)  # type: ignore
    elif provider == "drive" and DriveConnector is not None:
        connector = DriveConnector(tenant_id=tenant_id, config=cfg)  # type: ignore
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    # List changes (paginated)
    changes, next_cursor = connector.list_changes(cursor=req.cursor)
    total_found = len(changes)

    # Apply max_items limit
    if len(changes) > req.max_items:
        changes = changes[: req.max_items]

    # Get Redis client and queue key
    redis_client = get_redis_client()
    queue_key = f"connector:{provider}:{tenant_id}:queue"

    # Dedupe: get already queued URIs from Redis
    already_queued: set[str] = set()
    if req.skip_existing:
        # Scan current queue for existing URIs
        queue_items = redis_client.lrange(queue_key, 0, -1)
        for item_bytes in queue_items:
            try:
                item_data = json.loads(item_bytes)
                if "uri" in item_data:
                    already_queued.add(item_data["uri"])
            except Exception:
                pass

    # Dedupe: get already processed URIs from database
    already_processed: set[str] = set()
    if req.skip_existing:
        dsn = env_str(["ACTIVEKG_DSN", "DATABASE_URL"])
        if dsn:
            try:
                with psycopg.connect(dsn, row_factory=dict_row) as conn:
                    with conn.cursor() as cur:
                        # Get external_ids of existing nodes for this provider
                        # External ID format: provider:uri (e.g., "gcs:gs://bucket/file.pdf")
                        uris = [change.uri for change in changes]
                        if uris:
                            cur.execute(
                                """
                                SELECT props->>'external_id' as external_id
                                FROM nodes
                                WHERE tenant_id = %s
                                  AND props->>'external_id' = ANY(%s)
                                """,
                                (tenant_id, uris),
                            )
                            for row in cur.fetchall():
                                if row["external_id"]:
                                    already_processed.add(row["external_id"])
            except Exception:
                pass  # Continue without DB dedupe if connection fails

    # Filter and count
    queued_count = 0
    skipped_count = 0
    items_to_queue = []

    for change in changes:
        # Check if already queued or processed
        if change.uri in already_queued or change.uri in already_processed:
            skipped_count += 1
            continue

        items_to_queue.append(change)

    # If dry_run, don't actually queue
    if req.dry_run:
        return {
            "status": "dry_run",
            "job_id": job_id,
            "dry_run": True,
            "would_queue": len(items_to_queue),
            "skipped_count": skipped_count,
            "total_found": total_found,
            "queue_key": queue_key,
            "next_cursor": next_cursor,
        }

    # Queue in batches
    for i in range(0, len(items_to_queue), req.batch_size):
        batch = items_to_queue[i : i + req.batch_size]
        for change in batch:
            item = {
                "uri": change.uri,
                "operation": change.operation,
                "etag": change.etag,
                "modified_at": change.modified_at.isoformat() if change.modified_at else None,
                "job_id": job_id,
            }
            redis_client.lpush(queue_key, json.dumps(item))
            queued_count += 1

    return {
        "status": "queued",
        "job_id": job_id,
        "dry_run": False,
        "queued_count": queued_count,
        "skipped_count": skipped_count,
        "total_found": total_found,
        "queue_key": queue_key,
        "next_cursor": next_cursor,
    }


@router.get("/{provider}/queue-status")
def queue_status(
    provider: str, tenant_id: str, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    """Get the current queue depth for a connector.

    Use this to monitor ingestion progress after calling /ingest.
    Requires super_admin scope.
    """
    from activekg.common.metrics import get_redis_client

    if JWT_ENABLED and (not claims or "super_admin" not in (claims.scopes or [])):
        raise HTTPException(status_code=403, detail="super_admin scope required")

    redis_client = get_redis_client()
    queue_key = f"connector:{provider}:{tenant_id}:queue"
    depth = redis_client.llen(queue_key)

    return {
        "tenant_id": tenant_id,
        "provider": provider,
        "queue_key": queue_key,
        "pending": depth,
    }
