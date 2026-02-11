# Extraction Queue Operations

**Last Updated**: 2026-02-05
**Audience**: SREs, DevOps, Support

This runbook covers the Redis-backed async extraction queue and worker for LLM-based resume field extraction.

---

## Overview

The extraction pipeline uses Groq LLMs to parse structured fields from resume text:

```
POST /nodes (extract_before_embed=true)
  → enqueue Redis (extraction:queue)
  → extraction worker calls Groq API
  → DB update (props + extraction_status=ready)
  → trigger re-embed if extraction_version changed
```

**Two-tier model fallback**:
1. Primary: `llama-3.1-8b-instant` (fast, cheap)
2. Fallback: `llama-3.3-70b-versatile` (if JSON invalid, missing fields, or confidence < 0.65)

---

## Extracted Fields

| Field | Type | Description |
|-------|------|-------------|
| `primary_skills` | `list[str]` | Top 8-12 technical/professional skills |
| `recent_job_titles` | `list[str]` | 1-3 most recent job titles |
| `years_experience_total` | `int\|str\|null` | Total years (number or range like "5-7") |
| `certifications` | `list[str]\|null` | Professional certs (AWS, PMP, etc.) |
| `industries` | `list[str]\|null` | Industries worked in |

**Metadata fields** (stored in props):
- `extraction_status`: queued | processing | ready | failed | skipped
- `extraction_confidence`: 0.0-1.0
- `extraction_version`: prompt version (e.g., "2026-02-05.1")
- `extraction_model`: model used (primary or fallback)
- `extracted_at`: ISO timestamp

---

## Redis Keys

- `extraction:queue` — main queue (LPUSH / BRPOP)
- `extraction:retry` — delayed retry ZSET (score = run_at)
- `extraction:dlq` — failed jobs after max attempts
- `extraction:pending:{node_id}` — per-node dedupe key
- `extraction:tenant:pending:{tenant}` — per-tenant pending counter

---

## Node Status Lifecycle

```
queued → processing → ready
                   → failed (after max attempts)
                   → skipped (text < 100 chars)
```

---

## Configuration

### API Service

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTRACTION_ENABLED` | `false` | Enable extraction pipeline |
| `EXTRACTION_MODE` | `async` | `async` (embed first) or `sync` (extract first) |
| `EXTRACTION_VERSION` | `1.0.0` | Prompt version; bump to trigger re-extraction |
| `REDIS_URL` | — | Required for async queue |

### Worker Service

| Variable | Default | Description |
|----------|---------|-------------|
| `ACTIVEKG_DSN` | — | Database connection (same as API) |
| `REDIS_URL` | — | Redis connection |
| `GROQ_API_KEY` | — | Groq API key (required) |
| `EXTRACTION_PRIMARY_MODEL` | `llama-3.1-8b-instant` | Primary model |
| `EXTRACTION_FALLBACK_MODEL` | `llama-3.3-70b-versatile` | Fallback model |
| `EXTRACTION_CONFIDENCE_THRESHOLD` | `0.65` | Below this triggers fallback |
| `EXTRACTION_MAX_TOKENS` | `1024` | Max response tokens |
| `EXTRACTION_MAX_INPUT_CHARS` | `12000` | Truncate input text |
| `EXTRACTION_MAX_ATTEMPTS` | `2` | Primary + fallback = 2 |
| `EXTRACTION_RETRY_BASE_SECONDS` | `10` | Retry delay base |
| `EXTRACTION_RETRY_MAX_SECONDS` | `60` | Retry delay cap |
| `EXTRACTION_WORKER_POLL_INTERVAL` | `1.0` | Queue poll interval |
| `EXTRACTION_HEALTHCHECK_PORT` | `8080` | HTTP healthcheck port |

### Field Caps (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `EXTRACTION_MAX_PRIMARY_SKILLS` | `12` | Max skills to store |
| `EXTRACTION_MAX_RECENT_TITLES` | `3` | Max job titles |
| `EXTRACTION_MAX_CERTIFICATIONS` | `10` | Max certs |
| `EXTRACTION_MAX_INDUSTRIES` | `5` | Max industries |

---

## Operational Endpoints

### Queue Status
```bash
GET /admin/extraction/status?tenant_id=<tenant>
```
Returns:
```json
{
  "enabled": true,
  "mode": "async",
  "tenant_id": "my_tenant",
  "status_counts": {"ready": 67, "none": 100},
  "queue": {"queue": 0, "retry": 0, "dlq": 0}
}
```

### Requeue Extraction Jobs
```bash
POST /admin/extraction/requeue
```
Example — requeue nodes that never had extraction:
```bash
curl -X POST https://api.example.com/admin/extraction/requeue \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"my_tenant","only_null_status":true,"limit":2000}'
```

Example — requeue failed extractions:
```bash
curl -X POST https://api.example.com/admin/extraction/requeue \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"my_tenant","status":"failed","limit":500}'
```

---

## Common Issues

### Nodes have `extraction_status: null`
**Cause**: Nodes created before extraction was enabled, or Redis unavailable at ingest.
**Fix**: Use requeue endpoint:
```bash
curl -X POST https://api.example.com/admin/extraction/requeue \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"my_tenant","only_null_status":true}'
```

### Extraction failing with low confidence
**Cause**: Resume text is malformed, too short, or non-standard format.
**Fix**: Check worker logs for specific errors. Nodes with text < 100 chars are auto-skipped.

### High fallback rate
**Cause**: Primary model struggling with certain resume formats.
**Fix**: Monitor `extraction_model` field in props. If > 30% use fallback, consider:
- Adjusting `EXTRACTION_CONFIDENCE_THRESHOLD`
- Reviewing prompt in `activekg/extraction/prompt.py`

### Worker not processing
**Cause**: Missing env vars or Redis connection issue.
**Fix**: Check worker logs for startup errors. Verify:
- `GROQ_API_KEY` is set
- `REDIS_URL` matches API service
- `ACTIVEKG_DSN` matches API service

---

## Maintenance

### Requeue all failed nodes
```bash
curl -X POST https://api.example.com/admin/extraction/requeue \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"my_tenant","status":"failed","limit":2000}'
```

### Inspect Redis queue depth
```bash
redis-cli LLEN extraction:queue
redis-cli ZCARD extraction:retry
redis-cli LLEN extraction:dlq
```

### Clear DLQ (use with caution)
```bash
redis-cli DEL extraction:dlq
```

### Force re-extraction (bump version)
Update `EXTRACTION_VERSION` env var on both API and worker. Nodes with older `extraction_version` will be re-extracted when next processed.

---

## Monitoring

### Key Metrics
- Queue depth: `extraction:queue` length
- Processing rate: extractions completed per minute
- Fallback rate: % using fallback model
- Error rate: failed extractions / total

### Health Checks
- Worker exposes `/health` on `EXTRACTION_HEALTHCHECK_PORT` (default 8080)
- Returns `{"status":"healthy","service":"extraction-worker"}`

### Log Patterns
```
# Successful extraction
INFO - Extraction succeeded with primary model
INFO - Extraction completed

# Fallback triggered
INFO - Falling back to larger model

# Re-embed triggered
INFO - Triggered re-embed after extraction
```

---

## Post-Incident Checklist

- [ ] Redis queue depth back to normal
- [ ] `extraction_status` counts stable
- [ ] No growth in DLQ
- [ ] Worker logs clean for 30 minutes
- [ ] Fallback rate within acceptable range (< 30%)
