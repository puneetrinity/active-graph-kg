# Active Graph KG Operations Guide

**Last Updated**: 2026-02-03
**Target Audience**: SREs, DevOps, Production Support

This guide covers operational procedures for Active Graph KG connector infrastructure including monitoring, troubleshooting, and maintenance tasks.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Monitoring & Alerts](#monitoring--alerts)
- [Webhook Troubleshooting](#webhook-troubleshooting)
- [Worker Troubleshooting](#worker-troubleshooting)
- [Ingestion Troubleshooting](#ingestion-troubleshooting)
- [Embedding Queue](#embedding-queue)
- [Purger Operations](#purger)
- [Cache Subscriber](#cache-subscriber)
- [Key Rotation](#key-rotation)
- [Common Operations](#common-operations)
- [Incident Response](#incident-response)

---

## Architecture Overview

Active Graph KG connector system consists of:

- **API Server**: Hosts webhook endpoints (`/_webhooks/s3`, `/_webhooks/gcs`)
- **Queue (Redis)**: Stores connector events per tenant/provider (`connector:{provider}:{tenant}:queue`)
- **Worker**: Background process that polls queues and processes changes
- **Embedding Queue (Redis)**: Stores async embedding jobs (`embedding:queue`, `embedding:retry`, `embedding:dlq`)
- **Embedding Worker**: Background process that generates embeddings and updates node status
- **Scheduler**: APScheduler-based cron tasks (purger runs daily at 02:00 UTC)
- **Config Store**: Encrypted connector configurations in PostgreSQL

### Data Flow

```
Cloud Storage Event → Webhook (SNS/Pub/Sub) → API Server → Redis Queue → Worker → Graph DB
```

### Async Embedding Flow

```
POST /nodes or /nodes/batch
  → enqueue Redis (embedding:queue)
  → embedding worker generates vector
  → DB update (embedding_status=ready)
```

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| `webhook_pubsub_verify_total{result}` | Webhook signature verifications | Failure rate >10% |
| `connector_worker_queue_depth{provider,tenant}` | Queue backlog per tenant | >1000 items for 10m |
| `connector_ingest_total` | Successful ingestions | 0 for 30m (stalled) |
| `connector_ingest_errors_total` | Ingestion failures | Error rate >1% |
| `connector_purger_total{result}` | Purger executions | Any errors |
| `connector_rotation_total{result}` | Key rotation results | Any errors |

---

## Monitoring & Alerts

### Prometheus Setup

1. **Add alert rules** to Prometheus:
```yaml
# prometheus.yml
rule_files:
  - "observability/alerts/connector_alerts.yml"
```

2. **Configure Alertmanager** routes:
```yaml
route:
  group_by: ['alertname', 'tenant', 'provider']
  receiver: 'connector-oncall'

receivers:
  - name: 'connector-oncall'
    pagerduty_configs:
      - service_key: '<your-key>'
```

3. **Scrape targets**:
```yaml
scrape_configs:
  - job_name: 'activekg-api'
    static_configs:
      - targets: ['api:8000']

  - job_name: 'activekg-worker'
    static_configs:
      - targets: ['worker:9090']  # If worker exposes metrics endpoint
```

### Grafana Dashboards

Import `observability/dashboards/connector_overview.json` for:
- Ingestion rate/errors by provider
- Queue depth heatmap
- Webhook verification success rate
- Purger execution history
- P50/P95/P99 latency

**Quick metrics queries**:
```promql
# Ingestion rate by provider
sum(rate(connector_ingest_total[5m])) by (provider)

# Error rate percentage
(sum(rate(connector_ingest_errors_total[5m])) / sum(rate(connector_ingest_total[5m]))) * 100

# Queue depth (current)
connector_worker_queue_depth{provider="gcs",tenant="default"}

# Webhook verify failures
sum(rate(webhook_pubsub_verify_total{result!~"secret_ok|oidc_ok|skipped"}[5m]))
```

---

## Webhook Troubleshooting

### Problem: High Webhook Verification Failures

**Alert**: `WebhookVerificationFailuresHigh`

**Symptoms**:
- >10% of webhook requests failing signature verification
- `webhook_pubsub_verify_total{result="signature_invalid"}` increasing

**Diagnosis**:
1. Check webhook logs:
```bash
kubectl logs -l app=activekg-api --tail=100 | grep "webhook verification failed"
```

2. Verify environment variables:
```bash
# For secret-based verification
echo $PUBSUB_VERIFY_SECRET  # Should match subscription push endpoint token

# For OIDC verification
echo $PUBSUB_VERIFY_OIDC  # Should be "true"
echo $PUBSUB_OIDC_AUDIENCE  # Should match subscription audience
```

3. Test manual webhook:
```bash
curl -X POST https://$HOST/_webhooks/gcs \
  -H "Content-Type: application/json" \
  -H "X-PubSub-Token: $PUBSUB_VERIFY_SECRET" \
  -d '{"message":{"data":"...","attributes":{...}}}'
```

**Resolution**:
- **Secret mismatch**: Rotate subscription token and update `PUBSUB_VERIFY_SECRET`
- **OIDC audience mismatch**: Update `PUBSUB_OIDC_AUDIENCE` to match subscription config
- **Missing headers**: Ensure Pub/Sub subscription configured with correct auth method

**Prevention**:
- Store secrets in secure vault (AWS Secrets Manager, GCP Secret Manager)
- Automate secret rotation (90-day cycle)
- Add integration tests for webhook auth

---

### Problem: Webhook Topic Rejected

**Alert**: `WebhookTopicRejected`

**Symptoms**:
- `webhook_topic_rejected_total` > 0
- Logs show "topic not in allowlist"

**Diagnosis**:
1. Check tenant's topic allowlist:
```sql
SELECT tenant_id, provider, config->>'topic_allowlist'
FROM connector_configs
WHERE tenant_id = 'default' AND provider = 'gcs';
```

2. Compare with incoming topic:
```bash
# Check recent webhook logs
kubectl logs -l app=activekg-api --tail=50 | grep "topic rejected"
```

**Resolution**:
```bash
# Update allowlist via admin API
curl -X PATCH https://$HOST/_admin/connectors/gcs/config \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "default",
    "config": {
      "topic_allowlist": ["projects/my-proj/topics/activekg-gcs-prod"]
    }
  }'
```

**Prevention**:
- Document topic naming conventions
- Validate topic on connector registration
- Alert on topic rejections (already configured)

---

## Worker Troubleshooting

### Problem: High Queue Depth

**Alert**: `ConnectorQueueDepthHigh`

**Symptoms**:
- `connector_worker_queue_depth{provider,tenant}` > 1000 for >10m
- Ingestion lag increasing

**Diagnosis**:
1. Check queue depth directly:
```bash
redis-cli LLEN "connector:gcs:default:queue"
```

2. Verify worker is running:
```bash
ps aux | grep "activekg.connectors.worker"
# or
kubectl get pods -l app=activekg-worker
```

3. Check worker logs for errors:
```bash
kubectl logs -l app=activekg-worker --tail=100
```

**Common Causes**:
- Worker not running (crashed, not deployed)
- Worker misconfigured (wrong REDIS_URL, missing credentials)
- Downstream bottleneck (database connection pool exhausted)
- High error rate (items failing and not being retried)

**Resolution**:

**Scenario 1: Worker not running**
```bash
# Restart worker
python -m activekg.connectors.worker
# or
kubectl rollout restart deployment/activekg-worker
```

**Scenario 2: Database bottleneck**
```bash
# Check active connections
psql $ACTIVEKG_DSN -c "SELECT count(*) FROM pg_stat_activity WHERE application_name = 'activekg';"

# Increase connection pool if needed
export ACTIVEKG_POOL_SIZE=20  # Default: 10
```

**Scenario 3: Scaling**
```bash
# Scale worker horizontally (multiple workers poll same queue)
kubectl scale deployment/activekg-worker --replicas=3
```

**Scenario 4: Purge stale items**
```bash
# Check oldest item in queue (LINDEX -1 gets last item)
redis-cli --raw LINDEX "connector:gcs:default:queue" -1 | jq '.modified_at'

# If >7 days old, consider manual purge (DANGER: only after investigation)
redis-cli LTRIM "connector:gcs:default:queue" 0 0  # Keeps only 1 item (effectively clears)
```

**Prevention**:
- Set up worker autoscaling based on queue depth
- Monitor worker health endpoints
- Add dead-letter queue for permanently failed items

---

### Problem: Worker Processing Errors

**Symptoms**:
- `connector_worker_errors_total` increasing
- Queue depth not decreasing despite worker running

**Diagnosis**:
1. Check error types:
```promql
sum(rate(connector_worker_errors_total[5m])) by (error_type)
```

2. Common error types:
   - `no_config`: Connector config not found/disabled
   - `parse`: Invalid queue item JSON
   - `processing`: Ingestion failure (network, permissions, etc.)
   - `unsupported_provider`: Unknown provider

3. Examine specific errors:
```bash
kubectl logs -l app=activekg-worker --tail=200 | grep ERROR
```

**Resolution by Error Type**:

| Error Type | Cause | Fix |
|------------|-------|-----|
| `no_config` | Config deleted/disabled | Re-register connector |
| `parse` | Corrupted queue item | Clear queue, investigate webhook format changes |
| `processing` | Network/permissions | Check GCS/S3 permissions, retry manually |
| `unsupported_provider` | Code bug or typo | Fix provider name in config |

---

## Ingestion Troubleshooting

### Problem: High Ingestion Error Rate

**Alert**: `IngestErrorRateHigh`

**Symptoms**:
- Error rate >1% for >10m
- `connector_ingest_errors_total` increasing

**Diagnosis**:
1. Check error distribution:
```promql
sum(rate(connector_ingest_errors_total[10m])) by (provider, tenant)
```

2. Sample failed items:
```bash
# Check Redis DLQ (dead letter queue) if configured
redis-cli LRANGE "connector:gcs:default:dlq" 0 10

# Or check worker logs
kubectl logs -l app=activekg-worker | grep "ingestion failed"
```

3. Test single item manually:
```python
from activekg.connectors.providers.gcs import GCSConnector
from activekg.connectors.ingest import IngestionProcessor

config = {...}  # From DB
connector = GCSConnector(tenant_id="default", config=config)
processor = IngestionProcessor(connector=connector, repo=repo, redis_client=redis)

# Test single URI
changes = [ChangeItem(uri="gs://bucket/docs/sample.pdf", operation="upsert")]
result = processor.process_changes(changes)
```

**Common Issues**:
- **Permissions**: GCS/S3 bucket not accessible (403/404 errors)
- **Format**: Unsupported file type or corrupted file
- **Size**: File too large (timeout, OOM)
- **Embedding**: Embedding service unavailable

**Resolution**:
1. **Permission errors**:
```bash
# Test bucket access
gsutil ls gs://bucket/docs/
# or
aws s3 ls s3://bucket/docs/
```

2. **Format errors**:
```bash
# Check file type
gsutil cat gs://bucket/docs/sample.pdf | file -
```

3. **Size errors**:
```bash
# Check file size
gsutil du -h gs://bucket/docs/sample.pdf

# Adjust max file size if needed
export CONNECTOR_MAX_FILE_SIZE_MB=50  # Default: 10
```

---

### Problem: Ingestion Stalled

**Alert**: `IngestStalled`

**Symptoms**:
- `connector_ingest_total` rate = 0 for >30m
- No activity despite webhook events arriving

**Diagnosis**:
1. Verify webhooks arriving:
```bash
redis-cli LLEN "connector:gcs:default:queue"  # Should be > 0 if webhooks arriving
```

2. Check worker status:
```bash
kubectl get pods -l app=activekg-worker
kubectl logs -l app=activekg-worker --tail=50
```

3. Check database connectivity:
```bash
psql $ACTIVEKG_DSN -c "SELECT 1;"
```

**Resolution**:
- If queue empty but expecting traffic → Check webhook configuration
- If queue full but worker idle → Restart worker
- If database down → Fix database connectivity first
- If all healthy but still stalled → Check for deadlock/blocking queries

```sql
-- Check for blocking queries
SELECT pid, wait_event_type, wait_event, state, query
FROM pg_stat_activity
WHERE application_name = 'activekg' AND state != 'idle';
```

---

## Embedding Queue

### Queue Keys

- `embedding:queue` — main queue (LPUSH/BRPOP)
- `embedding:retry` — delayed retry ZSET (score = run_at)
- `embedding:dlq` — failed jobs after max attempts
- `embedding:pending:{node_id}` — dedupe key per node
- `embedding:tenant:pending:{tenant}` — per-tenant pending count

### Node Status Lifecycle

`queued` → `processing` → `ready`  
`failed` on repeated errors  
`skipped` if no text payload

### Operational Endpoints

- `GET /admin/embedding/status`  
  Returns DB counts by status and Redis queue depth.

- `POST /admin/embedding/requeue`  
  Requeue failed/queued nodes, optionally backfill ready statuses.

Example:
```bash
curl -X POST http://localhost:8000/admin/embedding/requeue \
  -H "Content-Type: application/json" \
  -d '{"tenant_id":"default","status":"queued","only_missing_embedding":true,"backfill_ready":true,"limit":2000}'
```

### Common Issues

- **Many nodes `queued` but Redis queue empty**  
  Nodes were created before async queue or when Redis unavailable.  
  Fix: run `/admin/embedding/requeue` with `status="queued"` + `only_missing_embedding=true`.

- **Nodes `ready` but `has_embedding=false`**  
  Indicates embedding failed; check worker logs and DLQ.

- **Large backlog**  
  Increase worker replicas or reduce ingest rate; check `EMBEDDING_TENANT_MAX_PENDING`.

---

## Purger

### Daily Purge Schedule

Purger runs daily at 02:00 UTC via APScheduler cron job.

**Configuration**:
```bash
export RUN_SCHEDULER=true  # Enable scheduler
export PURGER_BATCH_SIZE=500  # Items per batch (default: 500)
export PURGER_RETENTION_DAYS=30  # Grace period before hard delete (default: 30)
```

### Problem: Purger Errors

**Alert**: `PurgerErrors`

**Symptoms**:
- `connector_purger_total{result="error"}` > 0

**Diagnosis**:
```bash
# Check scheduler logs
kubectl logs -l app=activekg-api | grep "purge cycle"

# Manual purge dry-run
curl -X POST http://$HOST/_admin/connectors/purge_deleted \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true, "tenant_id": "default"}'
```

**Common Errors**:
- Database connection lost during purge
- Transaction timeout (purging too many items)
- Permission denied on connector_configs table

**Resolution**:
1. **Transaction timeout**:
```bash
# Reduce batch size
export PURGER_BATCH_SIZE=100
```

2. **Manual purge** (if scheduler broken):
```bash
curl -X POST http://$HOST/_admin/connectors/purge_deleted \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false, "tenant_id": "default", "batch_size": 100}'
```

---

## Cache Subscriber

GCS Pub/Sub subscriber maintains long-lived connection to receive real-time notifications.

### Problem: High Reconnect Rate

**Alert**: `PubSubReconnectsHigh`

**Symptoms**:
- `connector_pubsub_reconnect_total` > 5 in 15m
- Intermittent webhook delivery delays

**Diagnosis**:
```bash
# Check subscriber logs
kubectl logs -l app=activekg-api | grep "subscriber"

# Check network connectivity to Pub/Sub
curl -I https://pubsub.googleapis.com/v1/projects/$PROJECT/topics
```

**Common Causes**:
- Network instability (firewall, NAT timeout)
- Pub/Sub service disruption
- Client credential expiration
- Resource limits (file descriptors, memory)

**Resolution**:
1. **Network issues**:
```bash
# Increase TCP keepalive
export PUBSUB_TCP_KEEPALIVE=60  # Seconds
```

2. **Credential issues**:
```bash
# Refresh service account key
gcloud iam service-accounts keys create new-key.json \
  --iam-account=svc@$PROJECT.iam.gserviceaccount.com

# Update secret
kubectl create secret generic gcs-creds --from-file=key.json=new-key.json --dry-run=client -o yaml | kubectl apply -f -
```

3. **Resource limits**:
```bash
# Increase file descriptor limit
ulimit -n 4096

# or in Kubernetes
kubectl patch deployment activekg-api -p '{"spec":{"template":{"spec":{"containers":[{"name":"api","resources":{"limits":{"cpu":"2","memory":"2Gi"}}}]}}}}'
```

---

## Key Rotation

### KEK (Key Encryption Key) Rotation

**Frequency**: Every 90 days (recommended)

**Procedure**:

1. **Generate new KEK**:
```bash
python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
```

2. **Add to environment** (non-breaking):
```bash
export CONNECTOR_KEK_V2=<new-key>
export CONNECTOR_KEK_ACTIVE_VERSION=2
```

3. **Rotate configs** (re-encrypt with new key):
```bash
curl -X POST http://$HOST/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": null, "batch_size": 100}'
```

4. **Monitor rotation**:
```promql
sum(rate(connector_rotation_total[5m])) by (result)
```

5. **Verify** all configs rotated:
```sql
SELECT tenant_id, provider, key_version
FROM connector_configs
WHERE key_version < 2;
```

6. **Remove old KEK** (after all configs rotated):
```bash
unset CONNECTOR_KEK_V1
```

### Problem: Rotation Errors

**Alert**: `RotationErrors`

**Diagnosis**:
```bash
# Check rotation logs
kubectl logs -l app=activekg-api | grep "rotation"

# Check failed rows
curl -X POST http://$HOST/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"dry_run": true}'
```

**Resolution**:
```bash
# Retry failed rows only
curl -X POST http://$HOST/_admin/connectors/rotate_keys \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"tenant_id": "default", "batch_size": 10}'
```

### Problem: Config Decryption Failures

**Alert**: `ConfigDecryptFailures`

**Symptoms**:
- Worker can't load connector config
- `connector_config_decrypt_failures_total` > 0

**Diagnosis**:
```bash
# Check key versions
echo $CONNECTOR_KEK_ACTIVE_VERSION
echo $CONNECTOR_KEK_V1
echo $CONNECTOR_KEK_V2

# Check DB key versions
psql $ACTIVEKG_DSN -c "SELECT DISTINCT key_version FROM connector_configs;"
```

**Resolution**:
1. **Missing KEK**: Re-add old KEK temporarily
```bash
export CONNECTOR_KEK_V1=<old-key>
```

2. **Corrupted config**: Delete and re-register
```bash
curl -X DELETE http://$HOST/_admin/connectors/gcs/config?tenant_id=default \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Then re-register
curl -X POST http://$HOST/_admin/connectors/gcs/register ...
```

---

## Common Operations

### Register New Connector

**GCS Example**:
```bash
curl -X POST http://$HOST/_admin/connectors/gcs/register \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme-corp",
    "config": {
      "bucket": "acme-docs",
      "prefix": "knowledge-base/",
      "project": "acme-prod",
      "credentials_path": "/secrets/gcs-key.json",
      "enabled": true,
      "topic_allowlist": ["projects/acme-prod/topics/activekg-gcs"]
    }
  }'
```

### Backfill Historical Data

```bash
curl -X POST http://$HOST/_admin/connectors/gcs/backfill \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme-corp",
    "limit": 500,
    "prefix": "knowledge-base/q4-reports/"
  }'
```

### Manual Purge

```bash
# Dry-run first
curl -X POST http://$HOST/_admin/connectors/purge_deleted \
  -H "Content-Type: application/json" \
  -d '{"dry_run": true, "tenant_id": "acme-corp"}'

# Execute
curl -X POST http://$HOST/_admin/connectors/purge_deleted \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false, "tenant_id": "acme-corp", "batch_size": 500}'
```

### Check Health

```bash
# API health
curl http://$HOST/health

# Webhook health
curl http://$HOST/_webhooks/gcs/health

# Metrics
curl http://$HOST/metrics
```

---

## Incident Response

### Severity Levels

| Severity | Response Time | Example |
|----------|--------------|---------|
| **Critical** | 15 min | Ingestion completely down, data loss risk |
| **Warning** | 1 hour | High error rate, queue backlog |
| **Info** | Next business day | Single tenant issue, minor config problem |

### Incident Checklist

1. **Acknowledge alert** in PagerDuty/OpsGenie
2. **Check dashboards** for context (Grafana connector overview)
3. **Review recent changes** (deployments, config updates, traffic spikes)
4. **Gather logs**:
   ```bash
   kubectl logs -l app=activekg-api --tail=500 > api.log
   kubectl logs -l app=activekg-worker --tail=500 > worker.log
   ```
5. **Identify root cause** using runbooks above
6. **Mitigate** (restart service, scale up, disable problematic tenant)
7. **Monitor recovery** (metrics return to normal)
8. **Document incident** (postmortem, runbook updates)

### Emergency Contacts

- **On-Call Engineer**: Check PagerDuty schedule
- **Connector Team Lead**: [Contact info]
- **Database Team**: [Contact info] (for PostgreSQL issues)
- **Cloud Provider Support**: [Escalation path for GCS/S3 issues]

---

## Appendix

### Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `ACTIVEKG_DSN` | PostgreSQL connection | Required |
| `RUN_SCHEDULER` | Enable APScheduler | `true` |
| `PUBSUB_VERIFY_SECRET` | Pub/Sub push endpoint token | None |
| `PUBSUB_VERIFY_OIDC` | Enable OIDC verification | `false` |
| `CONNECTOR_KEK_V1` | Key Encryption Key v1 | Required |
| `CONNECTOR_KEK_ACTIVE_VERSION` | Active KEK version | `1` |
| `CONNECTOR_WORKER_BATCH_SIZE` | Worker batch size | `10` |
| `CONNECTOR_WORKER_POLL_INTERVAL` | Worker poll interval (seconds) | `1.0` |
| `PURGER_BATCH_SIZE` | Purger batch size | `500` |
| `PURGER_RETENTION_DAYS` | Soft-delete retention | `30` |

### Useful Queries

```sql
-- Check connector configs
SELECT tenant_id, provider, config->>'bucket', enabled, created_at, updated_at
FROM connector_configs
ORDER BY updated_at DESC;

-- Check ingestion stats
SELECT
  DATE_TRUNC('hour', created_at) AS hour,
  COUNT(*) AS nodes_created
FROM nodes
WHERE entity_type = 'Document'
  AND created_at > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour DESC;

-- Check soft-deleted nodes
SELECT COUNT(*), entity_type
FROM nodes
WHERE deleted_at IS NOT NULL
  AND deleted_at < NOW() - INTERVAL '30 days'
GROUP BY entity_type;
```

---

**Document Version**: 1.0
**Last Reviewed**: 2025-11-12
**Next Review**: 2025-12-12
