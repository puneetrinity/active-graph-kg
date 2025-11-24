"""Retry logic with DLQ for failed connector operations."""

import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import redis
from prometheus_client import Counter, Gauge
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Prometheus metrics
dlq_total = Counter(
    "connector_dlq_total", "Total items sent to DLQ", ["provider", "tenant", "reason"]
)
dlq_depth = Gauge("connector_dlq_depth", "Current DLQ depth", ["provider", "tenant"])


class TransientError(Exception):
    """Retryable error (network, rate limit, etc.)."""

    pass


class PermanentError(Exception):
    """Non-retryable error (validation, not found, etc.)."""

    pass


def with_retry_and_dlq(
    redis_client: redis.Redis, provider: str, tenant_id: str, max_attempts: int = 3
):
    """Decorator for connector operations with retry + DLQ.

    Args:
        redis_client: Redis client for DLQ
        provider: Provider name (s3, gcs, etc.)
        tenant_id: Tenant ID
        max_attempts: Maximum retry attempts

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            retry=retry_if_exception_type(TransientError),
            reraise=True,
        )
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except TransientError:
                logger.warning(f"Transient error in {func.__name__}, will retry")
                raise
            except PermanentError as e:
                logger.error(f"Permanent error in {func.__name__}: {e}")
                _send_to_dlq(
                    redis_client,
                    provider,
                    tenant_id,
                    func.__name__,
                    args,
                    kwargs,
                    str(e),
                    "permanent",
                )
                raise
            except Exception as e:
                # Unknown error - after retries exhausted, send to DLQ
                logger.error(f"Unknown error in {func.__name__} after retries: {e}")
                _send_to_dlq(
                    redis_client,
                    provider,
                    tenant_id,
                    func.__name__,
                    args,
                    kwargs,
                    str(e),
                    "unknown",
                )
                raise

        return wrapper

    return decorator


def _send_to_dlq(
    redis_client: redis.Redis,
    provider: str,
    tenant_id: str,
    operation: str,
    args: tuple,
    kwargs: dict,
    error: str,
    reason: str,
):
    """Send failed operation to DLQ.

    Args:
        redis_client: Redis client
        provider: Provider name
        tenant_id: Tenant ID
        operation: Operation name
        args: Function args
        kwargs: Function kwargs
        error: Error message
        reason: Failure reason (permanent, unknown, etc.)
    """
    dlq_key = f"dlq:{provider}:{tenant_id}"

    dlq_item = {
        "operation": operation,
        "args": str(args),  # Simplified - could serialize URIs
        "kwargs": str(kwargs),
        "error": error,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
        "provider": provider,
        "tenant_id": tenant_id,
    }

    # Push to Redis list
    redis_client.lpush(dlq_key, json.dumps(dlq_item))

    # Update metrics
    dlq_total.labels(provider=provider, tenant=tenant_id, reason=reason).inc()
    dlq_depth.labels(provider=provider, tenant=tenant_id).set(float(cast(int, redis_client.llen(dlq_key))))

    logger.error(f"Sent to DLQ: {dlq_key} - {operation} - {error}")


def inspect_dlq(redis_client: redis.Redis, provider: str, tenant_id: str, limit: int = 100) -> list:
    """Inspect DLQ contents.

    Args:
        redis_client: Redis client
        provider: Provider name
        tenant_id: Tenant ID
        limit: Max items to return

    Returns:
        List of DLQ items
    """
    dlq_key = f"dlq:{provider}:{tenant_id}"
    items = cast(list[bytes], redis_client.lrange(dlq_key, 0, limit - 1))
    return [json.loads(item) for item in items]


def clear_dlq(redis_client: redis.Redis, provider: str, tenant_id: str):
    """Clear DLQ for provider/tenant.

    Args:
        redis_client: Redis client
        provider: Provider name
        tenant_id: Tenant ID
    """
    dlq_key = f"dlq:{provider}:{tenant_id}"
    redis_client.delete(dlq_key)
    dlq_depth.labels(provider=provider, tenant=tenant_id).set(0)
    logger.info(f"Cleared DLQ: {dlq_key}")
