"""Per-tenant throttling to prevent ingestion overload."""

import time
from typing import TYPE_CHECKING, cast

import redis

if TYPE_CHECKING:
    from redis import Redis


class IngestionThrottle:
    """Redis-based token bucket throttle for connector ingestion.

    Limits ingestion rate per tenant to prevent DB/embedding service overload.
    """

    def __init__(self, redis_client: "Redis[bytes]", max_per_sec: int = 50):
        """Initialize throttle.

        Args:
            redis_client: Redis client instance
            max_per_sec: Maximum documents per second per tenant
        """
        self.redis: "Redis[bytes]" = redis_client
        self.max_per_sec = max_per_sec

    def acquire(self, tenant_id: str, provider: str) -> bool:
        """Acquire ingestion token (blocking until available).

        Args:
            tenant_id: Tenant ID
            provider: Provider name (s3, gcs, etc.)

        Returns:
            True when token acquired
        """
        key = f"throttle:ingest:{provider}:{tenant_id}"

        while True:
            # Try to increment counter
            current = cast(int, self.redis.incr(key))

            if current == 1:
                # First request in this second - set expiry
                self.redis.expire(key, 1)

            if current <= self.max_per_sec:
                # Token acquired
                return True

            # Over limit - decrement and wait
            self.redis.decr(key)
            time.sleep(0.1)

    def try_acquire(self, tenant_id: str, provider: str) -> bool:
        """Try to acquire token without blocking.

        Args:
            tenant_id: Tenant ID
            provider: Provider name

        Returns:
            True if acquired, False if would block
        """
        key = f"throttle:ingest:{provider}:{tenant_id}"

        current = cast(int, self.redis.incr(key))

        if current == 1:
            self.redis.expire(key, 1)

        if current <= self.max_per_sec:
            return True

        # Over limit
        self.redis.decr(key)
        return False
