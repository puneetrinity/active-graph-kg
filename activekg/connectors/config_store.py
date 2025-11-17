"""Database-backed connector config store with encryption and caching.

Provides:
- Persistent storage of connector configs
- Automatic encryption/decryption of secrets
- In-memory caching for performance
- Enable/disable per tenant
- Redis pub/sub for multi-worker cache invalidation
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any

import psycopg
from prometheus_client import Counter
from psycopg.rows import dict_row

from activekg.connectors.encryption import get_encryption, sanitize_config_for_logging
from activekg.connectors.types import ConnectorConfigTD

logger = logging.getLogger(__name__)

# Redis pub/sub (optional - graceful degradation if not available)
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - cache invalidation will be local only")

# Prometheus metrics
connector_config_cache_hits_total = Counter(
    "connector_config_cache_hits_total", "Total connector config cache hits", ["provider"]
)
connector_config_cache_misses_total = Counter(
    "connector_config_cache_misses_total", "Total connector config cache misses", ["provider"]
)
connector_config_decrypt_failures_total = Counter(
    "connector_config_decrypt_failures_total",
    "Total connector config decryption failures",
    ["provider", "field"],
)
connector_config_invalidate_total = Counter(
    "connector_config_invalidate_total",
    "Total connector config cache invalidations published",
    ["operation"],  # upsert, delete, enable
)
connector_pubsub_publish_failures_total = Counter(
    "connector_pubsub_publish_failures_total", "Total Redis pub/sub publish failures"
)
connector_rotation_total = Counter(
    "connector_rotation_total",
    "Total connector config key rotations",
    ["result"],  # rotated, skipped, error
)
connector_rotation_batch_latency_seconds = Counter(
    "connector_rotation_batch_latency_seconds", "Total latency for key rotation batches in seconds"
)


class ConnectorConfigStore:
    """Database store for connector configurations with encryption."""

    def __init__(self, dsn: str, cache_ttl_seconds: int = 300, redis_url: str | None = None):
        """Initialize config store.

        Args:
            dsn: PostgreSQL connection string
            cache_ttl_seconds: How long to cache configs in memory (default: 5 minutes)
            redis_url: Optional Redis URL for pub/sub cache invalidation
        """
        self.dsn = dsn
        self.cache_ttl = cache_ttl_seconds
        self.encryption = get_encryption()

        # In-memory cache: (tenant_id, provider) -> (config, timestamp)
        self._cache: dict[tuple, tuple[dict[str, Any], datetime]] = {}

        # Redis pub/sub for multi-worker cache invalidation (optional)
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info(f"Redis pub/sub enabled for cache invalidation: {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis for pub/sub: {e}")
                self.redis_client = None

    def _get_connection(self) -> psycopg.Connection:
        """Get database connection.

        Returns:
            psycopg connection
        """
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def _is_cache_valid(self, cache_entry: tuple[dict[str, Any], datetime]) -> bool:
        """Check if cache entry is still valid.

        Args:
            cache_entry: Tuple of (config, timestamp)

        Returns:
            True if cache still valid
        """
        config, timestamp = cache_entry
        age = (datetime.utcnow() - timestamp).total_seconds()
        return age < self.cache_ttl

    def _publish_invalidation(self, tenant_id: str, provider: str, operation: str) -> None:
        """Publish cache invalidation message via Redis pub/sub.

        Args:
            tenant_id: Tenant ID
            provider: Provider name
            operation: Operation type (upsert, delete, enable)
        """
        if not self.redis_client:
            logger.debug("Redis pub/sub not available, skipping invalidation publish")
            return

        try:
            message = json.dumps(
                {"tenant_id": tenant_id, "provider": provider, "operation": operation}
            )
            self.redis_client.publish("connector:config:changed", message)
            connector_config_invalidate_total.labels(operation=operation).inc()
            logger.debug(f"Published cache invalidation: {tenant_id}/{provider} (op={operation})")
        except Exception as e:
            connector_pubsub_publish_failures_total.inc()
            logger.warning(f"Failed to publish cache invalidation: {e}")

    def get(self, tenant_id: str, provider: str = "s3") -> dict[str, Any] | None:
        """Get connector config for tenant.

        Checks cache first, falls back to DB.
        Automatically decrypts secrets.

        Args:
            tenant_id: Tenant ID
            provider: Provider name (default: s3)

        Returns:
            Decrypted config dict or None if not found
        """
        cache_key = (tenant_id, provider)

        # Check cache
        if cache_key in self._cache:
            cache_entry = self._cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.debug(f"Config cache hit: {tenant_id}/{provider}")
                connector_config_cache_hits_total.labels(provider=provider).inc()
                return cache_entry[0]
            else:
                # Expired
                del self._cache[cache_key]

        # Cache miss
        connector_config_cache_misses_total.labels(provider=provider).inc()

        # Fetch from DB
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT config_json, enabled, key_version
                        FROM connector_configs
                        WHERE tenant_id = %s AND provider = %s
                        """,
                        (tenant_id, provider),
                    )
                    row = cur.fetchone()

                    if not row:
                        logger.debug(f"Config not found: {tenant_id}/{provider}")
                        return None

                    if not row["enabled"]:
                        logger.warning(f"Config disabled: {tenant_id}/{provider}")
                        return None

                    # Decrypt secrets using stored key_version (fallback to all KEKs if None)
                    encrypted_config = row["config_json"]
                    key_version = row.get("key_version")
                    decrypted_config = self.encryption.decrypt_config(
                        encrypted_config, key_version=key_version
                    )

                    # Cache it
                    self._cache[cache_key] = (decrypted_config, datetime.utcnow())

                    logger.info(f"Config loaded from DB: {tenant_id}/{provider}")
                    return decrypted_config

        except Exception as e:
            logger.error(f"Failed to load config from DB: {e}")
            return None

    def upsert(
        self, tenant_id: str, provider: str, config: dict[str, Any], enabled: bool = True
    ) -> bool:
        """Insert or update connector config.

        Automatically encrypts secrets before storing.

        Args:
            tenant_id: Tenant ID
            provider: Provider name
            config: Plain text config dict
            enabled: Whether config is enabled

        Returns:
            True if successful
        """
        try:
            # Encrypt secrets using active KEK
            encrypted_config = self.encryption.encrypt_config(config)
            active_key_version = self.encryption.active_version

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO connector_configs (tenant_id, provider, config_json, enabled, key_version)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (tenant_id, provider)
                        DO UPDATE SET
                            config_json = EXCLUDED.config_json,
                            enabled = EXCLUDED.enabled,
                            key_version = EXCLUDED.key_version,
                            updated_at = NOW()
                        """,
                        (
                            tenant_id,
                            provider,
                            json.dumps(encrypted_config),
                            enabled,
                            active_key_version,
                        ),
                    )
                    conn.commit()

            # Invalidate cache
            cache_key = (tenant_id, provider)
            if cache_key in self._cache:
                del self._cache[cache_key]

            # Publish invalidation message
            self._publish_invalidation(tenant_id, provider, "upsert")

            logger.info(f"Config saved: {tenant_id}/{provider} (enabled={enabled})")
            logger.debug(f"Config: {sanitize_config_for_logging(config)}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def set_enabled(self, tenant_id: str, provider: str, enabled: bool) -> bool:
        """Enable or disable connector config.

        Args:
            tenant_id: Tenant ID
            provider: Provider name
            enabled: True to enable, False to disable

        Returns:
            True if successful
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE connector_configs
                        SET enabled = %s, updated_at = NOW()
                        WHERE tenant_id = %s AND provider = %s
                        """,
                        (enabled, tenant_id, provider),
                    )
                    updated = cur.rowcount > 0
                    conn.commit()

            if updated:
                # Invalidate cache
                cache_key = (tenant_id, provider)
                if cache_key in self._cache:
                    del self._cache[cache_key]

                # Publish invalidation message
                self._publish_invalidation(tenant_id, provider, "enable")

                logger.info(
                    f"Config {'enabled' if enabled else 'disabled'}: {tenant_id}/{provider}"
                )
            else:
                logger.warning(f"Config not found for enable/disable: {tenant_id}/{provider}")

            return updated

        except Exception as e:
            logger.error(f"Failed to set enabled: {e}")
            return False

    def list_all(self, provider: str | None = None) -> list[ConnectorConfigTD]:
        """List all connector configs.

        Does NOT decrypt secrets. Returns metadata only.

        Args:
            provider: Optional filter by provider

        Returns:
            List of config metadata dicts (without decrypted secrets)
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    if provider:
                        cur.execute(
                            """
                            SELECT tenant_id, provider, enabled, created_at, updated_at
                            FROM connector_configs
                            WHERE provider = %s
                            ORDER BY tenant_id
                            """,
                            (provider,),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT tenant_id, provider, enabled, created_at, updated_at
                            FROM connector_configs
                            ORDER BY tenant_id, provider
                            """
                        )

                    rows = cur.fetchall()

                    # Convert to typed records (ISO timestamps)
                    configs: list[ConnectorConfigTD] = []
                    for row in rows:
                        created = row["created_at"]
                        updated = row["updated_at"]
                        configs.append(
                            {
                                "tenant_id": row["tenant_id"],
                                "provider": row["provider"],
                                "config": {},  # metadata listing excludes secrets
                                "created_at": created.isoformat() if created else None,
                                "updated_at": updated.isoformat() if updated else None,
                            }
                        )

                    return configs

        except Exception as e:
            logger.error(f"Failed to list configs: {e}")
            return []

    def delete(self, tenant_id: str, provider: str) -> bool:
        """Delete connector config.

        Args:
            tenant_id: Tenant ID
            provider: Provider name

        Returns:
            True if successful
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        DELETE FROM connector_configs
                        WHERE tenant_id = %s AND provider = %s
                        """,
                        (tenant_id, provider),
                    )
                    deleted = cur.rowcount > 0
                    conn.commit()

            if deleted:
                # Invalidate cache
                cache_key = (tenant_id, provider)
                if cache_key in self._cache:
                    del self._cache[cache_key]

                # Publish invalidation message
                self._publish_invalidation(tenant_id, provider, "delete")

                logger.info(f"Config deleted: {tenant_id}/{provider}")
            else:
                logger.warning(f"Config not found for deletion: {tenant_id}/{provider}")

            return deleted

        except Exception as e:
            logger.error(f"Failed to delete config: {e}")
            return False

    def rotate_keys(
        self,
        providers: list[str] | None = None,
        tenants: list[str] | None = None,
        batch_size: int = 100,
        dry_run: bool = False,
    ) -> RotationBatchResultTD:
        """Rotate encryption keys for connector configs.

        Selects rows where key_version != ACTIVE_VERSION, decrypts with old key,
        re-encrypts with active key, and updates key_version.

        Args:
            providers: Optional list of providers to filter
            tenants: Optional list of tenants to filter
            batch_size: Number of rows to process per batch
            dry_run: If True, only count rows without making changes

        Returns:
            Summary dict with counts: {rotated, skipped, errors, dry_run}
        """
        import time

        start_time = time.time()

        rotated = 0
        skipped = 0
        errors = 0

        try:
            active_version = self.encryption.active_version

            # Build query with filters
            query = """
                SELECT tenant_id, provider, config_json, key_version
                FROM connector_configs
                WHERE key_version IS NULL OR key_version != %s
            """
            params: list[Any] = [active_version]

            if providers:
                placeholders = ",".join(["%s"] * len(providers))
                query += f" AND provider IN ({placeholders})"
                params.extend(providers)

            if tenants:
                placeholders = ",".join(["%s"] * len(tenants))
                query += f" AND tenant_id IN ({placeholders})"
                params.extend(tenants)

            query += " ORDER BY tenant_id, provider LIMIT %s"
            params.append(batch_size)

            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()

                    if dry_run:
                        # Dry run: just count
                        return {
                            "rotated": 0,
                            "skipped": 0,
                            "errors": 0,
                            "candidates": len(rows),
                            "dry_run": True,
                        }

                    # Process each row
                    for row in rows:
                        tenant_id = row["tenant_id"]
                        provider = row["provider"]
                        encrypted_config = row["config_json"]
                        old_key_version = row.get("key_version")

                        try:
                            # Decrypt with old key (fallback allowed)
                            decrypted_config = self.encryption.decrypt_config(
                                encrypted_config, key_version=old_key_version
                            )

                            # Re-encrypt with active key
                            re_encrypted_config = self.encryption.encrypt_config(decrypted_config)

                            # Update row
                            cur.execute(
                                """
                                UPDATE connector_configs
                                SET config_json = %s, key_version = %s, updated_at = NOW()
                                WHERE tenant_id = %s AND provider = %s
                                """,
                                (
                                    json.dumps(re_encrypted_config),
                                    active_version,
                                    tenant_id,
                                    provider,
                                ),
                            )

                            # Invalidate cache
                            cache_key = (tenant_id, provider)
                            if cache_key in self._cache:
                                del self._cache[cache_key]

                            # Publish invalidation
                            self._publish_invalidation(tenant_id, provider, "rotate")

                            rotated += 1
                            connector_rotation_total.labels(result="rotated").inc()
                            logger.info(
                                f"Key rotated: {tenant_id}/{provider} (v{old_key_version} -> v{active_version})"
                            )

                        except Exception as e:
                            errors += 1
                            connector_rotation_total.labels(result="error").inc()
                            logger.error(f"Failed to rotate key for {tenant_id}/{provider}: {e}")

                    conn.commit()

            # Track latency
            elapsed = time.time() - start_time
            connector_rotation_batch_latency_seconds.inc(elapsed)

            logger.info(
                f"Key rotation complete: {rotated} rotated, {skipped} skipped, {errors} errors ({elapsed:.2f}s)"
            )

            return {"rotated": rotated, "skipped": skipped, "errors": errors, "dry_run": False}

        except Exception as e:
            logger.error(f"Key rotation batch failed: {e}")
            return {
                "rotated": rotated,
                "skipped": skipped,
                "errors": errors + 1,
                "dry_run": False,
                "error": str(e),
            }


# Global singleton instance (lazy-loaded)
_config_store: ConnectorConfigStore | None = None


def get_config_store(dsn: str | None = None, redis_url: str | None = None) -> ConnectorConfigStore:
    """Get global config store instance.

    Args:
        dsn: PostgreSQL DSN (if not provided, uses existing instance or raises)
        redis_url: Optional Redis URL for pub/sub (if not provided, reads from REDIS_URL env)

    Returns:
        ConnectorConfigStore instance

    Raises:
        ValueError: If DSN not provided and no instance exists
    """
    global _config_store

    if _config_store is None:
        if dsn is None:
            # Accept Railway/Heroku style DATABASE_URL as fallback
            from activekg.common.env import env_str
            dsn = env_str(["ACTIVEKG_DSN", "DATABASE_URL"])  # empty string if not set
            if not dsn:
                raise ValueError("ACTIVEKG_DSN/DATABASE_URL not set")

        if redis_url is None:
            redis_url = os.getenv("REDIS_URL")

        _config_store = ConnectorConfigStore(dsn, redis_url=redis_url)

    return _config_store
