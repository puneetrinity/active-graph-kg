from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any

import redis

QUEUE_KEY = os.getenv("EMBEDDING_QUEUE_KEY", "embedding:queue")
RETRY_KEY = os.getenv("EMBEDDING_RETRY_KEY", "embedding:retry")
DLQ_KEY = os.getenv("EMBEDDING_DLQ_KEY", "embedding:dlq")
PENDING_PREFIX = os.getenv("EMBEDDING_PENDING_PREFIX", "embedding:pending")
PENDING_TTL_SECONDS = int(os.getenv("EMBEDDING_PENDING_TTL_SECONDS", "3600"))
TENANT_PENDING_PREFIX = os.getenv("EMBEDDING_TENANT_PENDING_PREFIX", "embedding:tenant:pending")
TENANT_PENDING_TTL_SECONDS = int(os.getenv("EMBEDDING_TENANT_PENDING_TTL_SECONDS", "3600"))


def _pending_key(node_id: str) -> str:
    return f"{PENDING_PREFIX}:{node_id}"


def _tenant_pending_key(tenant_id: str | None) -> str:
    return f"{TENANT_PENDING_PREFIX}:{tenant_id or 'default'}"


def get_pending_count(redis_client: redis.Redis, tenant_id: str | None) -> int:
    value = redis_client.get(_tenant_pending_key(tenant_id))
    if value is None:
        return 0
    try:
        return int(value)
    except Exception:
        return 0


def increment_pending(redis_client: redis.Redis, tenant_id: str | None, count: int = 1) -> int:
    key = _tenant_pending_key(tenant_id)
    pipe = redis_client.pipeline()
    pipe.incrby(key, int(count))
    pipe.expire(key, TENANT_PENDING_TTL_SECONDS)
    result = pipe.execute()
    return int(result[0]) if result else 0


def decrement_pending(redis_client: redis.Redis, tenant_id: str | None, count: int = 1) -> int:
    key = _tenant_pending_key(tenant_id)
    value = int(redis_client.decrby(key, int(count)))
    if value < 0:
        redis_client.set(key, 0, ex=TENANT_PENDING_TTL_SECONDS)
        return 0
    return value


def enqueue_embedding_job(
    redis_client: redis.Redis,
    node_id: str,
    tenant_id: str | None,
    *,
    action: str = "embed",
    force: bool = False,
) -> str | None:
    """Enqueue an embedding job. Returns job_id if enqueued, else None (deduped)."""
    pending_key = _pending_key(node_id)
    if force:
        clear_pending(redis_client, node_id, tenant_id=tenant_id)

    # Prevent duplicate enqueues for the same node
    if not redis_client.set(pending_key, "1", nx=True, ex=PENDING_TTL_SECONDS):
        return None

    job_id = str(uuid.uuid4())
    payload = {
        "job_id": job_id,
        "node_id": node_id,
        "tenant_id": tenant_id,
        "action": action,
        "attempts": 0,
        "enqueued_at": time.time(),
    }
    redis_client.lpush(QUEUE_KEY, json.dumps(payload))
    increment_pending(redis_client, tenant_id, count=1)
    return job_id


def schedule_retry(redis_client: redis.Redis, job: dict[str, Any], delay_seconds: float) -> None:
    """Schedule a retry by placing the job in a delayed ZSET."""
    run_at = time.time() + max(0.0, delay_seconds)
    redis_client.zadd(RETRY_KEY, {json.dumps(job): run_at})


def move_due_retries(redis_client: redis.Redis, *, limit: int = 200) -> int:
    """Move due retries from ZSET to main queue. Returns number moved."""
    now = time.time()
    due = redis_client.zrangebyscore(RETRY_KEY, 0, now, start=0, num=limit)
    if not due:
        return 0

    pipe = redis_client.pipeline()
    moved = 0
    for raw in due:
        pipe.zrem(RETRY_KEY, raw)
        pipe.lpush(QUEUE_KEY, raw)
        moved += 1
    pipe.execute()
    return moved


def queue_depth(redis_client: redis.Redis) -> dict[str, int]:
    """Return queue depth across main, retry, and DLQ."""
    return {
        "queue": int(redis_client.llen(QUEUE_KEY)),
        "retry": int(redis_client.zcard(RETRY_KEY)),
        "dlq": int(redis_client.llen(DLQ_KEY)),
    }


def clear_pending(redis_client: redis.Redis, node_id: str, tenant_id: str | None = None) -> None:
    redis_client.delete(_pending_key(node_id))
    decrement_pending(redis_client, tenant_id, count=1)


def pending_exists(redis_client: redis.Redis, node_id: str) -> bool:
    return bool(redis_client.exists(_pending_key(node_id)))
