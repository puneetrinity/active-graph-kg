from __future__ import annotations

"""Admin endpoints to manage connectors with database-backed persistence.

Provides:
- POST /s3/register - Register/update S3 connector config (encrypted in DB)
- GET / - List all connector configs
- GET /{provider} - Get specific connector config metadata
- POST /{provider}/enable - Enable connector
- POST /{provider}/disable - Disable connector
- POST /s3/backfill - Trigger manual backfill
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

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

    import os

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
