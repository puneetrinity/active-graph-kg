# Embedding Queue Operations

**Last Updated**: 2026-02-03  
**Audience**: SREs, DevOps, Support

This runbook covers the Redis‑backed async embedding queue and worker.

---

## Overview

Async embedding decouples ingestion from embedding generation:

```
POST /nodes or /nodes/batch
  → enqueue Redis (embedding:queue)
  → embedding worker generates vector
  → DB update (embedding_status=ready)
```

---

## Redis Keys

- `embedding:queue` — main queue (LPUSH / BRPOP)
- `embedding:retry` — delayed retry ZSET (score = run_at)
- `embedding:dlq` — failed jobs after max attempts
- `embedding:pending:{node_id}` — per‑node dedupe key
- `embedding:tenant:pending:{tenant}` — per‑tenant pending counter

---

## Node Status Lifecycle

`queued` → `processing` → `ready`  
`failed` when max attempts exceeded  
`skipped` when text is empty/unavailable

---

## Configuration

**API service**
- `EMBEDDING_ASYNC=true`
- `REDIS_URL=...`
- `EMBEDDING_QUEUE_MAX_DEPTH=5000`
- `EMBEDDING_TENANT_MAX_PENDING=2000`
- `NODE_BATCH_MAX=200`

**Worker service**
- `ACTIVEKG_DSN` (same DB as API)
- `REDIS_URL`
- `EMBEDDING_BACKEND`, `EMBEDDING_MODEL`
- `EMBEDDING_MAX_ATTEMPTS=5`
- `EMBEDDING_RETRY_BASE_SECONDS=10`
- `EMBEDDING_RETRY_MAX_SECONDS=300`
- `EMBEDDING_WORKER_POLL_INTERVAL=1.0`

---

## Operational Endpoints

### Queue status
```
GET /admin/embedding/status
```
Returns DB counts by status and Redis queue depth.

### Requeue + backfill
```
POST /admin/embedding/requeue
```
Example:
```bash
curl -X POST http://localhost:8000/admin/embedding/requeue \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"default","status":"queued","only_missing_embedding":true,"backfill_ready":true,"limit":2000}'
```

---

## Common Issues

### Many nodes `queued` but Redis queue empty
**Cause**: nodes created before async queue or Redis unavailable at ingest.  
**Fix**: run requeue/backfill:
```bash
curl -X POST http://localhost:8000/admin/embedding/requeue \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"default","status":"queued","only_missing_embedding":true,"backfill_ready":true,"limit":2000}'
```

### Nodes `ready` but `has_embedding=false`
**Cause**: embedding failed or was never generated.  
**Fix**: check worker logs + DLQ, then requeue.

### Large backlog
**Cause**: ingest rate exceeds worker throughput.  
**Fix**: scale workers or reduce ingest rate; tune `EMBEDDING_TENANT_MAX_PENDING`.

---

## Maintenance

### Requeue all failed nodes
```bash
curl -X POST http://localhost:8000/admin/embedding/requeue \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"default","status":"failed","only_missing_embedding":true,"backfill_ready":true,"limit":2000}'
```

### Inspect Redis queue depth
```bash
redis-cli LLEN embedding:queue
redis-cli ZCARD embedding:retry
redis-cli LLEN embedding:dlq
```

### Clear DLQ (use with caution)
```bash
redis-cli DEL embedding:dlq
```

---

## Post‑Incident Checklist

- [ ] Redis queue depth back to normal
- [ ] `embedding_status` counts stable
- [ ] No growth in DLQ
- [ ] Worker logs clean for 30 minutes
