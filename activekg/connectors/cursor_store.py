"""Simple cursor store for connector incremental sync.

Uses the `connector_cursors` table added by migration 008 to persist per-tenant,
per-provider cursors (e.g., Drive Changes API pageToken).
"""

from __future__ import annotations

import os

import psycopg
from psycopg.rows import dict_row


def _get_dsn() -> str:
    from activekg.common.env import env_str
    dsn = env_str(["ACTIVEKG_DSN", "DATABASE_URL"])  # empty string if not set
    if not dsn:
        raise RuntimeError("ACTIVEKG_DSN/DATABASE_URL not set for cursor store")
    return dsn


def get_cursor(tenant_id: str, provider: str) -> str | None:
    """Fetch stored cursor for tenant/provider."""
    dsn = _get_dsn()
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT cursor
                FROM connector_cursors
                WHERE tenant_id = %s AND provider = %s
                """,
                (tenant_id, provider),
            )
            row = cur.fetchone()
            return row["cursor"] if row else None


def set_cursor(tenant_id: str, provider: str, cursor: str) -> None:
    """Upsert cursor for tenant/provider."""
    dsn = _get_dsn()
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO connector_cursors (tenant_id, provider, cursor)
                VALUES (%s, %s, %s)
                ON CONFLICT (tenant_id, provider)
                DO UPDATE SET cursor = EXCLUDED.cursor, updated_at = NOW()
                """,
                (tenant_id, provider, cursor),
            )
            conn.commit()
