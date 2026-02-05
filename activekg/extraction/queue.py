"""Redis-backed extraction queue for async resume parsing."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import redis

# Queue keys (separate from embedding queue)
EXTRACTION_QUEUE_KEY = os.getenv("EXTRACTION_QUEUE_KEY", "extraction:queue")
EXTRACTION_RETRY_KEY = os.getenv("EXTRACTION_RETRY_KEY", "extraction:retry")
EXTRACTION_DLQ_KEY = os.getenv("EXTRACTION_DLQ_KEY", "extraction:dlq")
EXTRACTION_PENDING_PREFIX = os.getenv("EXTRACTION_PENDING_PREFIX", "extraction:pending")
EXTRACTION_PENDING_TTL_SECONDS = int(os.getenv("EXTRACTION_PENDING_TTL_SECONDS", "3600"))
EXTRACTION_TENANT_PENDING_PREFIX = os.getenv(
    "EXTRACTION_TENANT_PENDING_PREFIX", "extraction:tenant:pending"
)
EXTRACTION_TENANT_PENDING_TTL_SECONDS = int(
    os.getenv("EXTRACTION_TENANT_PENDING_TTL_SECONDS", "3600")
)


def _pending_key(node_id: str) -> str:
    return f"{EXTRACTION_PENDING_PREFIX}:{node_id}"


def _tenant_pending_key(tenant_id: str | None) -> str:
    return f"{EXTRACTION_TENANT_PENDING_PREFIX}:{tenant_id or 'default'}"


def get_extraction_pending_count(redis_client: redis.Redis, tenant_id: str | None) -> int:
    """Get count of pending extraction jobs for a tenant."""
    value = redis_client.get(_tenant_pending_key(tenant_id))
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def increment_extraction_pending(
    redis_client: redis.Redis, tenant_id: str | None, count: int = 1
) -> int:
    """Increment pending extraction count for tenant."""
    key = _tenant_pending_key(tenant_id)
    pipe = redis_client.pipeline()
    pipe.incrby(key, int(count))
    pipe.expire(key, EXTRACTION_TENANT_PENDING_TTL_SECONDS)
    result = pipe.execute()
    return int(result[0]) if result else 0


def decrement_extraction_pending(
    redis_client: redis.Redis, tenant_id: str | None, count: int = 1
) -> int:
    """Decrement pending extraction count for tenant."""
    key = _tenant_pending_key(tenant_id)
    value = int(redis_client.decrby(key, int(count)))
    if value < 0:
        redis_client.set(key, 0, ex=EXTRACTION_TENANT_PENDING_TTL_SECONDS)
        return 0
    return value


def enqueue_extraction_job(
    redis_client: redis.Redis,
    node_id: str,
    tenant_id: str | None,
    *,
    force: bool = False,
    priority: str = "normal",
) -> str | None:
    """Enqueue an extraction job.

    Args:
        redis_client: Redis client
        node_id: Node to extract from
        tenant_id: Tenant ID
        force: If True, bypass dedup check
        priority: Job priority ("high" for bulk sync mode, "normal" for async)

    Returns:
        job_id if enqueued, None if deduped
    """
    pending_key = _pending_key(node_id)
    if force:
        clear_extraction_pending(redis_client, node_id, tenant_id=tenant_id)

    # Prevent duplicate enqueues for the same node
    if not redis_client.set(pending_key, "1", nx=True, ex=EXTRACTION_PENDING_TTL_SECONDS):
        return None

    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "node_id": node_id,
        "tenant_id": tenant_id,
        "priority": priority,
        "attempts": 0,
        "enqueued_at": time.time(),
    }
    redis_client.lpush(EXTRACTION_QUEUE_KEY, json.dumps(payload))
    increment_extraction_pending(redis_client, tenant_id, count=1)
    return job_id


def schedule_extraction_retry(
    redis_client: redis.Redis, job: dict[str, Any], delay_seconds: float
) -> None:
    """Schedule an extraction retry by placing the job in a delayed ZSET."""
    run_at = time.time() + max(0.0, delay_seconds)
    redis_client.zadd(EXTRACTION_RETRY_KEY, {json.dumps(job): run_at})


def move_due_extraction_retries(redis_client: redis.Redis, *, limit: int = 200) -> int:
    """Move due extraction retries from ZSET to main queue. Returns number moved."""
    now = time.time()
    due = redis_client.zrangebyscore(EXTRACTION_RETRY_KEY, 0, now, start=0, num=limit)
    if not due:
        return 0

    pipe = redis_client.pipeline()
    moved = 0
    for raw in due:
        pipe.zrem(EXTRACTION_RETRY_KEY, raw)
        pipe.lpush(EXTRACTION_QUEUE_KEY, raw)
        moved += 1
    pipe.execute()
    return moved


def extraction_queue_depth(redis_client: redis.Redis) -> dict[str, int]:
    """Return extraction queue depth across main, retry, and DLQ."""
    return {
        "queue": int(redis_client.llen(EXTRACTION_QUEUE_KEY)),
        "retry": int(redis_client.zcard(EXTRACTION_RETRY_KEY)),
        "dlq": int(redis_client.llen(EXTRACTION_DLQ_KEY)),
    }


def clear_extraction_pending(
    redis_client: redis.Redis, node_id: str, tenant_id: str | None = None
) -> None:
    """Clear pending status for a node."""
    redis_client.delete(_pending_key(node_id))
    decrement_extraction_pending(redis_client, tenant_id, count=1)


def extraction_pending_exists(redis_client: redis.Redis, node_id: str) -> bool:
    """Check if extraction job is already pending for node."""
    return bool(redis_client.exists(_pending_key(node_id)))
