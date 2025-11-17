# Active Graph KG - Operations Guide

**Last Updated**: 2025-11-17

## Table of Contents

- [Runtime & Config](#runtime--config)
- [Database & Indexing](#database--indexing)
- [Admin API Reference](#admin-api-reference)
- [Operational Runbook](#operational-runbook)
- [Monitoring & Metrics](#monitoring--metrics)
- [Troubleshooting](#troubleshooting)

---

## Runtime & Config

### Core Application

- **API Entry Point**: `activekg/api/main.py` (uvicorn, FastAPI)
- **Standard Port**: 5432 (Postgres), 8000 (API)

### DSN Fallback

DSN is resolved using `ACTIVEKG_DSN` or `DATABASE_URL` fallback for PaaS compatibility:

```python
# Resolved in:
# - activekg/api/main.py:58
# - activekg/api/admin_connectors.py:241
# - activekg/connectors/config_store.py:562
# - activekg/connectors/cursor_store.py:16
# - activekg/connectors/worker.py:270
```

### Key Environment Variables

```bash
# Database (fallback: DATABASE_URL for Railway/Heroku)
export ACTIVEKG_DSN='postgresql://activekg:activekg@localhost:5432/activekg'

# Embedding
export EMBEDDING_BACKEND='sentence-transformers'
export EMBEDDING_MODEL='all-MiniLM-L6-v2'

# Scheduler (run on exactly one instance)
export RUN_SCHEDULER=true  # false on replicas

# JWT Auth (HS256 for dev, RS256 in prod via JWT_PUBLIC_KEY)
export JWT_ENABLED=true
export JWT_SECRET_KEY='your-32-char-secret'
export JWT_ALGORITHM=HS256
export JWT_AUDIENCE=activekg
export JWT_ISSUER=https://auth.yourcompany.com

# Rate Limiting (optional)
export RATE_LIMIT_ENABLED=true
export REDIS_URL='redis://localhost:6379/0'

# ANN Configuration (dual-mode supported)
export PGVECTOR_INDEX=ivfflat            # or hnsw
export PGVECTOR_INDEXES=ivfflat,hnsw     # both present
export SEARCH_DISTANCE=cosine            # must match index opclass
export IVFFLAT_LISTS=100
export IVFFLAT_PROBES=4
export HNSW_M=16
export HNSW_EF_CONSTRUCTION=128
export HNSW_EF_SEARCH=80
```

---

## Database & Indexing

### Schema Initialization

```bash
# Schema init
psql $ACTIVEKG_DSN -f db/init.sql

# RLS policies
psql $ACTIVEKG_DSN -f enable_rls_policies.sql

# Hybrid text search (BM25)
psql $ACTIVEKG_DSN -f db/migrations/add_text_search.sql

# Vector index (IVFFLAT default; HNSW optional)
psql $ACTIVEKG_DSN -f enable_vector_index.sql
```

### Admin Index Management

```bash
# List indexes
curl -X POST $API/admin/indexes -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"action":"list"}'

# Ensure indexes exist (no auto-drop; concurrent builds)
curl -X POST $API/admin/indexes -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"action":"ensure","types":["ivfflat","hnsw"],"metric":"cosine"}'

# Rebuild
curl -X POST $API/admin/indexes -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"action":"rebuild","types":["ivfflat"],"metric":"cosine"}'

# Drop (careful!)
curl -X POST $API/admin/indexes -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"action":"drop","types":["hnsw"]}'
```

---

## Admin API Reference

### Connector Cache Health

**Endpoint**: `GET /_admin/connectors/cache/health`

**Description**: Check the health status of the connector config cache subscriber (Redis pub/sub).

**Authentication**: JWT required when `JWT_ENABLED=true`

**Response**:
```json
{
  "status": "ok",
  "subscriber": {
    "connected": true,
    "last_message_ts": "2025-11-11T15:08:16.411197Z",
    "reconnects": 0
  }
}
```

**Status Values**:
- `ok`: Subscriber connected and operational
- `degraded`: Subscriber not running or disconnected

**Usage**:
```bash
# Check cache subscriber health
curl http://localhost:8000/_admin/connectors/cache/health

# With JWT authentication
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/_admin/connectors/cache/health
```

---

### Connector Key Rotation

**Endpoint**: `POST /_admin/connectors/rotate_keys`

**Description**: Rotate encryption keys for connector configurations. Selects rows where `key_version != ACTIVE_VERSION`, decrypts with old key, re-encrypts with active key.

**Authentication**: JWT required when `JWT_ENABLED=true`

**Request Body**:
```json
{
  "providers": ["s3", "gcs"],     // Optional: filter by providers
  "tenants": ["acme", "corp"],    // Optional: filter by tenants
  "batch_size": 100,              // Optional: rows per batch (default: 100)
  "dry_run": false                // Optional: count only (default: false)
}
```

**Response**:
```json
{
  "rotated": 42,
  "skipped": 0,
  "errors": 0,
  "dry_run": false
}
```

**Dry-Run Response** (when `dry_run: true`):
```json
{
  "rotated": 0,
  "skipped": 0,
  "errors": 0,
  "candidates": 42,
  "dry_run": true
}
```

**Usage Examples**:

```bash
# 1. Dry-run to preview rotation candidates
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'

# 2. Rotate all configs
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'

# 3. Rotate specific provider
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"providers": ["s3"], "dry_run": false}'

# 4. Rotate specific tenant
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"tenants": ["acme"], "dry_run": false}'

# 5. Rotate with custom batch size
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"batch_size": 200, "dry_run": false}'

# 6. With JWT authentication
curl -X POST http://localhost:8000/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

---

## Operational Runbook

### Key Rotation Procedure

**When to Rotate Keys**:
- Regular security hygiene (quarterly/annually)
- Key compromise suspected
- Compliance requirements
- Before decommissioning old KEK versions

**Pre-Rotation Checklist**:
1. Generate new KEK:
   ```bash
   python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
   ```
2. Add new KEK to environment:
   ```bash
   export CONNECTOR_KEK_V2="<new-key>"
   ```
3. Keep old KEK(s) available:
   ```bash
   export CONNECTOR_KEK_V1="<existing-key>"
   ```
4. Verify both keys are loaded:
   ```bash
   # Check logs for "KEK validation passed" with correct version count
   ```

**Rotation Steps**:

#### 1. Staging Dry-Run
```bash
# Preview rotation candidates
curl -X POST http://staging.example.com/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'

# Expected: {"candidates": N, "dry_run": true}
```

#### 2. Staging Single Tenant Test
```bash
# Rotate one tenant to verify end-to-end
curl -X POST http://staging.example.com/_admin/connectors/rotate_keys \
  -H "Content-Type: application/json" \
  -d '{"tenants": ["test_tenant"], "dry_run": false}'

# Verify:
# - key_version updated in database
# - Configs decrypt correctly with new key
# - Cache invalidation fires (check pub/sub metrics)
# - Workers/API continue functioning
```

#### 3. Production Rotation

**Step 1: Update Environment**
```bash
# Set new ACTIVE_VERSION
export CONNECTOR_KEK_ACTIVE_VERSION="2"

# Ensure both keys present
export CONNECTOR_KEK_V1="<old-key>"
export CONNECTOR_KEK_V2="<new-key>"

# Restart API servers (rolling restart for zero downtime)
```

**Step 2: Dry-Run in Production**
```bash
# Get candidate counts
curl -X POST http://prod.example.com/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'
```

**Step 3: Rotate in Batches**
```bash
# For large datasets, rotate in provider batches
for provider in s3 gcs azure; do
  echo "Rotating $provider..."
  curl -X POST http://prod.example.com/_admin/connectors/rotate_keys \
    -H "Authorization: Bearer $ADMIN_TOKEN" \
    -H "Content-Type: application/json" \
    -d "{\"providers\": [\"$provider\"], \"batch_size\": 100, \"dry_run\": false}"

  echo "Sleeping 30s between batches..."
  sleep 30
done
```

**Step 4: Verify Completion**
```bash
# Confirm all rows rotated
curl -X POST http://prod.example.com/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true}'

# Expected: {"candidates": 0, "dry_run": true}

# Check rotation status via SQL
psql $ACTIVEKG_DSN -c "
  SELECT
    COUNT(*) FILTER (WHERE key_version != '$CONNECTOR_KEK_ACTIVE_VERSION') AS pending,
    COUNT(*) AS total
  FROM connector_configs
"
# Expected: pending=0, total=N
```

**Step 5: Monitor Metrics**
```promql
# Check rotation success rate
sum(connector_rotation_total{result="rotated"}) by (result)
sum(connector_rotation_total{result="error"}) by (result)

# Check decryption failures (should remain zero)
connector_config_decrypt_failures_total

# Check cache performance
rate(connector_config_cache_hits_total[5m])
rate(connector_config_cache_misses_total[5m])
```

**Step 6: Prune Old KEKs** (schedule for later)
```bash
# After confirming all configs on new version, remove old KEK
# DO NOT do this immediately - wait 24-48 hours to ensure stability
unset CONNECTOR_KEK_V1
# Or remove from deployment config and redeploy
```

---

### Failure Handling

#### Rotation Errors

**Symptom**: `errors > 0` in rotation response

**Investigation**:
```bash
# Check application logs
grep "Failed to rotate key" /var/log/app.log

# Check Prometheus metrics
connector_rotation_total{result="error"}

# Identify failed tenant/provider pairs from logs
```

**Resolution**:
```bash
# Retry specific tenant/provider
curl -X POST http://prod.example.com/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenants": ["failed_tenant"], "providers": ["failed_provider"], "dry_run": false}'
```

**Root Causes**:
- Database connectivity issues → Check connection pool
- KEK not available → Verify environment variables
- Corrupt ciphertext → Manual intervention required (re-encrypt from backup)

#### Cache Subscriber Down

**Symptom**: Health endpoint returns `"status": "degraded"`

**Investigation**:
```bash
# Check subscriber health
curl http://prod.example.com/_admin/connectors/cache/health

# Check Redis connectivity
redis-cli -h $REDIS_HOST ping

# Check logs
grep "cache subscriber" /var/log/app.log
```

**Resolution**:
```bash
# Restart API servers (subscriber auto-starts)
systemctl restart activekg-api

# Verify reconnection
curl http://prod.example.com/_admin/connectors/cache/health
# Expected: "connected": true, "reconnects": N
```

**Impact**: Cache invalidations not propagated to all workers (stale cache possible)

**Workaround**: Manual cache TTL expiration (5 minutes default)

#### Decryption Failures

**Symptom**: `connector_config_decrypt_failures_total` increasing

**Investigation**:
```bash
# Check which fields/providers failing
connector_config_decrypt_failures_total{provider="s3", field="secret_key"}

# Check KEK configuration
echo $CONNECTOR_KEK_V1
echo $CONNECTOR_KEK_V2
echo $CONNECTOR_KEK_ACTIVE_VERSION
```

**Resolution**:
```bash
# Ensure all KEK versions present
# If missing, add to environment and restart

# If ciphertext corrupt, restore from backup:
# 1. Get plaintext config from backup
# 2. Re-encrypt and upsert via admin API
```

---

## Monitoring & Metrics

### Prometheus Metrics

#### Rotation Metrics
```promql
# Total rotations by result
connector_rotation_total{result="rotated"}
connector_rotation_total{result="error"}

# Rotation latency
connector_rotation_batch_latency_seconds

# Rotation rate (per minute)
rate(connector_rotation_total{result="rotated"}[1m])
```

#### Cache Metrics
```promql
# Cache hit rate
rate(connector_config_cache_hits_total[5m])
  / (rate(connector_config_cache_hits_total[5m])
     + rate(connector_config_cache_misses_total[5m]))

# Decryption failures
connector_config_decrypt_failures_total

# Cache invalidations
connector_config_invalidate_total{operation="upsert"}
connector_config_invalidate_total{operation="rotate"}
```

#### Pub/Sub Metrics
```promql
# Message throughput
rate(connector_pubsub_messages_total[1m])

# Invalid messages
rate(connector_pubsub_invalid_msg_total[1m])

# Reconnection rate
rate(connector_pubsub_reconnect_total[5m])

# Graceful shutdowns
connector_pubsub_shutdown_total
```

### Alerts

**Recommended Prometheus Alerts**:

```yaml
groups:
- name: connector_alerts
  rules:
  # Cache subscriber down
  - alert: ConnectorCacheSubscriberDown
    expr: up{job="activekg-api"} == 1 and absent(connector_pubsub_messages_total)
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Connector cache subscriber not running"

  # High decryption failure rate
  - alert: ConnectorDecryptionFailures
    expr: rate(connector_config_decrypt_failures_total[5m]) > 0
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Connector config decryption failures detected"

  # Rotation errors
  - alert: ConnectorRotationErrors
    expr: rate(connector_rotation_total{result="error"}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Connector key rotation errors detected"

  # Cache hit rate too low
  - alert: ConnectorCacheLowHitRate
    expr: |
      rate(connector_config_cache_hits_total[5m])
      / (rate(connector_config_cache_hits_total[5m])
         + rate(connector_config_cache_misses_total[5m])) < 0.8
    for: 10m
    labels:
      severity: info
    annotations:
      summary: "Connector cache hit rate below 80%"
```

### Grafana Dashboard

**Recommended Dashboard Panels**:

1. **Cache Performance**
   - Cache hit rate (percentage)
   - Cache hits/misses over time
   - Cache invalidations by operation

2. **Pub/Sub Health**
   - Subscriber connection status
   - Messages processed per minute
   - Invalid messages rate
   - Reconnection events

3. **Rotation Operations**
   - Rotations completed (total)
   - Rotation errors (total)
   - Rotation latency (heatmap)
   - Rotation rate over time

4. **Security**
   - Decryption failures by provider/field
   - Key version distribution (ensure migration progress)

---

## Troubleshooting

### Common Issues

#### Issue: Configs not refreshing across workers

**Symptoms**:
- Config changes not visible on all API instances
- Stale data served after update

**Diagnosis**:
```bash
# Check cache subscriber health
curl http://localhost:8000/_admin/connectors/cache/health

# Check Redis connectivity
redis-cli ping

# Check pub/sub channel
redis-cli PUBSUB CHANNELS
# Expected: connector:config:changed
```

**Solutions**:
1. Restart API servers (subscriber auto-connects)
2. Check `REDIS_URL` environment variable
3. Verify Redis pub/sub not blocked by firewall

---

#### Issue: Rotation stuck at partial completion

**Symptoms**:
- Dry-run shows candidates but rotation makes no progress
- `rotated: 0` in response despite candidates

**Diagnosis**:
```bash
# Check database key_version distribution
psql $ACTIVEKG_DSN -c "
  SELECT key_version, COUNT(*)
  FROM connector_configs
  GROUP BY key_version
"

# Check active KEK version
echo $CONNECTOR_KEK_ACTIVE_VERSION

# Verify KEK availability
python3 -c "
from activekg.connectors.encryption import get_encryption
enc = get_encryption()
print(f'Active version: {enc.active_version}')
print(f'Available versions: {list(enc.keks.keys())}')
"
```

**Solutions**:
1. Ensure `CONNECTOR_KEK_ACTIVE_VERSION` set correctly
2. Verify new KEK in environment (`CONNECTOR_KEK_V2`)
3. Restart API servers to reload KEK config

---

#### Issue: High cache miss rate

**Symptoms**:
- `connector_config_cache_misses_total` high
- Increased database load

**Diagnosis**:
```bash
# Check cache TTL setting
# Default: 300 seconds (5 minutes)

# Check invalidation rate
# High invalidation = short-lived cache entries
```

**Solutions**:
1. Increase cache TTL (if configs rarely change):
   ```python
   # In config_store initialization
   ConnectorConfigStore(dsn, cache_ttl_seconds=900)  # 15 min
   ```
2. Reduce config update frequency
3. Ensure pub/sub working (invalidations should be selective, not frequent)

---

#### Issue: JWT authentication failing for admin endpoints

**Symptoms**:
- 401 Unauthorized on `/_admin/*` endpoints
- Health/rotation requests rejected

**Diagnosis**:
```bash
# Check JWT configuration
echo $JWT_ENABLED
echo $JWT_SECRET_KEY
echo $JWT_ALGORITHM

# Test token generation
python3 -c "
import jwt
token = jwt.encode({'sub': 'admin'}, '$JWT_SECRET_KEY', algorithm='$JWT_ALGORITHM')
print(token)
"
```

**Solutions**:
1. Verify `JWT_ENABLED=true` in production
2. Ensure `JWT_SECRET_KEY` matches token issuer
3. Use correct Authorization header: `Bearer <token>`
4. For dev/testing: Set `JWT_ENABLED=false`

---

### Secrets Hygiene

**CRITICAL**: Never log KEKs or ciphertext

**Safe Practices**:
- ✅ Log: `key_version`, rotation counts, tenant/provider IDs
- ✅ Metrics: Aggregated counts, latencies
- ❌ NEVER log: `CONNECTOR_KEK_*`, `secret_key`, `config_json` ciphertext

**Code Review Checklist**:
- No `print(config)` or `logger.debug(config)` with raw secrets
- Use `sanitize_config_for_logging()` when logging configs
- Mask secrets in error messages

---

### Audit Trail

**Key Version Tracking**:
```sql
-- View key version distribution
SELECT key_version, COUNT(*) as count
FROM connector_configs
GROUP BY key_version
ORDER BY key_version;

-- Audit recent rotations (via updated_at)
SELECT tenant_id, provider, key_version, updated_at
FROM connector_configs
WHERE updated_at > NOW() - INTERVAL '1 hour'
ORDER BY updated_at DESC;

-- Find configs needing rotation
SELECT tenant_id, provider, key_version
FROM connector_configs
WHERE key_version IS NULL OR key_version != '2'
ORDER BY tenant_id, provider;
```

**Structured Logging**:
```python
# Log rotation events
logger.info(f"Key rotation complete: {rotated} rotated, {errors} errors")

# Log with context
logger.info(
    "Config rotated",
    extra={
        "tenant_id": tenant_id,
        "provider": provider,
        "old_version": old_key_version,
        "new_version": active_version
    }
)
```

---

## Environment Variables Reference

### Core Configuration
```bash
# Database
ACTIVEKG_DSN="postgresql:///activekg?host=/var/run/postgresql"

# Redis (for pub/sub)
REDIS_URL="redis://localhost:6379/0"

# Encryption Keys
CONNECTOR_KEK_V1="<base64-key>"
CONNECTOR_KEK_V2="<base64-key>"
CONNECTOR_KEK_ACTIVE_VERSION="2"

# Authentication
JWT_ENABLED="true"
JWT_SECRET_KEY="<secret>"
JWT_ALGORITHM="HS256"

# Feature Flags
RUN_SCHEDULER="false"
RATE_LIMIT_ENABLED="true"
```

### Cache Configuration
```bash
# Cache TTL (default: 300 seconds)
# Increase for stable configs, decrease for frequently changing
CONNECTOR_CACHE_TTL="300"
```

---

## Support & Escalation

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL="DEBUG"

# Restart API
systemctl restart activekg-api

# Tail logs
tail -f /var/log/app.log | grep "connector"
```

### Emergency Procedures

**Complete Cache Flush** (use sparingly):
```bash
# Flush Redis (affects all workers)
redis-cli FLUSHDB

# All workers will cache-miss and reload from DB
```

**Rollback Rotation**:
```bash
# 1. Revert ACTIVE_VERSION
export CONNECTOR_KEK_ACTIVE_VERSION="1"

# 2. Rotate back (if needed)
curl -X POST http://prod.example.com/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'
```

---

## Additional Resources

- [PHASE_3_ROTATION_COMPLETE.md](PHASE_3_ROTATION_COMPLETE.md) - Implementation details
- [activekg/connectors/README.md](activekg/connectors/README.md) - Connector architecture
- [tests/test_phase3_rotation.py](tests/test_phase3_rotation.py) - Rotation test suite
- [tests/test_phase2_hardening.py](tests/test_phase2_hardening.py) - Pub/sub test suite
- [docs/operations/SELF_SERVE_RAILWAY.md](docs/operations/SELF_SERVE_RAILWAY.md) - Railway deployment guide
- [scripts/README.md](scripts/README.md) - Validation scripts catalog

### Core Admin Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/refresh` | POST | Trigger on-demand refresh |
| `/admin/indexes` | POST | List/ensure/rebuild/drop ANN indexes |
| `/_admin/embed_info` | GET | Embedding health |
| `/_admin/embed_class_coverage` | GET | Per-class embedding coverage |
| `/_admin/metrics_summary` | GET | Scheduler/trigger snapshots |
| `/_admin/drift_histogram` | GET | Drift distribution (bucketed) |
| `/_admin/connectors/cache/health` | GET | Cache subscriber health |
| `/_admin/connectors/rotate_keys` | POST | Key rotation |

### Debug Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/debug/embed_info` | GET | Embedding backend status |
| `/debug/search_explain` | GET | Query plan analysis |
| `/debug/search_sanity` | GET | Sanity checks |
| `/debug/dbinfo` | GET | Database metadata |

### Governance Metrics

New metrics for auth/tenant governance:
- `activekg_access_violations_total{type="missing_token"}` - Requests without JWT
- `activekg_access_violations_total{type="scope_denied"}` - Insufficient scopes
- `activekg_access_violations_total{type="cross_tenant_query"}` - RLS violations

### Observability

- **JSON metrics**: `GET /metrics`
- **Prometheus format**: `GET /prometheus`
- **Grafana dashboard**: `http://localhost:3000/d/activekg-ops`
- **Make targets**: `make open-grafana`

New metrics include:
- Trigger latency/fired
- Scheduler inter-run timing
- Refresh per node/cycle
- Vector index build histogram (type/metric/result)
- Ask first-token SSE latency
- Retrieval uplift gauge (from nightly eval publish)
