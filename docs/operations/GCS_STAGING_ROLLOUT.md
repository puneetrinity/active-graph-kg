# GCS Connector Staging Rollout Guide

> **ARCHIVED — DO NOT RUN.** GCS connectors are unavailable. This rollout plan is retained as future-design
> history and does not describe a deployable staging or production capability.

Complete validation and rollout procedure for GCS connector infrastructure in staging environment before production deployment.

## Prerequisites

Before starting the staging rollout, ensure:

- [ ] GCP project with billing enabled
- [ ] GCS bucket created for testing (e.g., `activekg-staging-test`)
- [ ] Pub/Sub topic and subscription configured
- [ ] Service account with appropriate permissions
- [ ] Staging Active Graph KG instance deployed
- [ ] Redis instance available
- [ ] PostgreSQL database initialized
- [ ] Prometheus and Grafana monitoring configured

## Environment Setup

### 1. GCP Service Account

Create a service account with necessary permissions:

```bash
# Set project
export GCP_PROJECT="your-staging-project"
gcloud config set project $GCP_PROJECT

# Create service account
gcloud iam service-accounts create activekg-connector-staging \
  --display-name="Active Graph KG Connector (Staging)" \
  --description="Service account for staging GCS connector"

# Grant bucket permissions
gsutil iam ch serviceAccount:activekg-connector-staging@${GCP_PROJECT}.iam.gserviceaccount.com:objectViewer \
  gs://activekg-staging-test

# Grant Pub/Sub permissions
gcloud pubsub topics add-iam-policy-binding activekg-gcs-staging \
  --member="serviceAccount:activekg-connector-staging@${GCP_PROJECT}.iam.gserviceaccount.com" \
  --role="roles/pubsub.subscriber"

# Create and download key
gcloud iam service-accounts keys create staging-gcs-key.json \
  --iam-account=activekg-connector-staging@${GCP_PROJECT}.iam.gserviceaccount.com
```

### 2. GCS Bucket and Pub/Sub

Create test bucket and notification subscription:

```bash
# Create bucket
gsutil mb -l us-central1 gs://activekg-staging-test

# Create Pub/Sub topic
gcloud pubsub topics create activekg-gcs-staging

# Create notification on bucket
gsutil notification create -f json -t activekg-gcs-staging gs://activekg-staging-test

# Create pull subscription
gcloud pubsub subscriptions create activekg-gcs-staging-sub \
  --topic=activekg-gcs-staging \
  --ack-deadline=60 \
  --message-retention-duration=7d
```

Verify notification is configured:

```bash
gsutil notification list gs://activekg-staging-test
```

Expected output:
```
projects/_/buckets/activekg-staging-test/notificationConfigs/1
Cloud Pub/Sub topic: projects/your-staging-project/topics/activekg-gcs-staging
```

### 3. Active Graph KG Configuration

Set environment variables on staging instance:

```bash
# Database
export ACTIVEKG_DSN="postgresql://user:pass@staging-db:5432/activekg"

# Redis
export REDIS_URL="redis://staging-redis:6379/0"

# GCS connector
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/staging-gcs-key.json"
export GCS_BUCKET="activekg-staging-test"

# Pub/Sub webhook verification (choose one)
export PUBSUB_VERIFY_SECRET="your-staging-secret-token"  # OR
export PUBSUB_OIDC_AUDIENCE="https://staging.activekg.example.com/_webhooks/gcs"

# Rate limiting
export RATE_LIMIT_ENABLED=true
export RATE_LIMIT_WEBHOOK_GCS_RATE=100
export RATE_LIMIT_WEBHOOK_GCS_BURST=200

# Scheduler (disable for initial testing)
export RUN_SCHEDULER=false

# Monitoring
export METRICS_ENABLED=true
```

## Stage 1: Connector Registration

### Register GCS Connector

```bash
# Using curl
curl -X POST http://staging.activekg.example.com/_admin/connectors/register \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gcs",
    "tenant_id": "staging-tenant",
    "config": {
      "bucket": "activekg-staging-test",
      "credentials_json": "<SERVICE_ACCOUNT_JSON>",
      "allowed_paths": ["test-docs/"]
    }
  }'
```

Expected response:
```json
{
  "connector_id": "conn_abc123",
  "provider": "gcs",
  "tenant_id": "staging-tenant",
  "status": "active"
}
```

### Verify Registration

Check connector config is encrypted and stored:

```bash
# Query database
psql $ACTIVEKG_DSN -c "
  SELECT connector_id, provider, tenant_id, key_version, created_at
  FROM connector_configs
  WHERE tenant_id = 'staging-tenant' AND provider = 'gcs';
"
```

Expected output:
```
 connector_id | provider | tenant_id      | key_version | created_at
--------------+----------+----------------+-------------+---------------------------
 conn_abc123  | gcs      | staging-tenant | 1           | 2025-11-12 10:30:00+00
```

Check Redis registry:

```bash
redis-cli -h staging-redis SMEMBERS connector:active_tenants
```

Should NOT show entry yet (registry populated on first webhook).

## Stage 2: Webhook Verification

### Test Webhook Endpoint

**Option A: Manual Pub/Sub Message**

```python
from google.cloud import pubsub_v1
import base64
import json

publisher = pubsub_v1.PublisherClient()
topic_path = "projects/your-staging-project/topics/activekg-gcs-staging"

# Simulate GCS notification
payload = {
    "name": "test-docs/sample.txt",
    "bucket": "activekg-staging-test",
    "size": "1024",
    "contentType": "text/plain"
}

attributes = {
    "bucketId": "activekg-staging-test",
    "objectId": "test-docs/sample.txt",
    "eventType": "OBJECT_FINALIZE"
}

data = base64.b64encode(json.dumps(payload).encode('utf-8')).decode('utf-8')

# Publish
future = publisher.publish(topic_path, data=data.encode('utf-8'), **attributes)
print(f"Published message ID: {future.result()}")
```

**Option B: Actual File Upload**

```bash
# Upload test file
echo "This is a staging test document" > staging-test.txt
gsutil cp staging-test.txt gs://activekg-staging-test/test-docs/staging-test.txt

# Wait 5-10 seconds for notification to propagate
```

### Verify Webhook Processing

**Check logs:**

```bash
# Kubernetes
kubectl logs -l app=activekg-api --tail=50 | grep webhook

# Docker
docker logs activekg-staging | grep webhook
```

Expected log entries:
```
INFO  Webhook received provider=gcs tenant=staging-tenant
INFO  Webhook verification result=secret_ok provider=gcs
INFO  Enqueued for ingestion uri=gs://activekg-staging-test/test-docs/staging-test.txt
```

**Check Redis queue:**

```bash
redis-cli -h staging-redis LLEN "connector:gcs:staging-tenant:queue"
```

Should return `1` (or more if multiple files uploaded).

**Check registry:**

```bash
redis-cli -h staging-redis SMEMBERS connector:active_tenants
```

Should return:
```
"{\"provider\":\"gcs\",\"tenant_id\":\"staging-tenant\"}"
```

**Check Prometheus metrics:**

```bash
curl -s http://staging.activekg.example.com/metrics | grep webhook_pubsub_verify_total
```

Should show:
```
webhook_pubsub_verify_total{provider="gcs",result="secret_ok"} 1.0
```

## Stage 3: Worker Processing

### Start Worker

```bash
# Kubernetes
kubectl scale deployment/activekg-worker --replicas=1

# Docker
docker run -d \
  --name activekg-worker-staging \
  --env-file staging.env \
  activekg:latest \
  python -m activekg.connectors.worker

# Direct execution
source venv/bin/activate
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/staging-gcs-key.json"
python -m activekg.connectors.worker
```

### Monitor Worker Activity

**Check logs:**

```bash
# Worker should discover queue via registry
# Expected log entries:
# INFO  Worker started providers=['gcs'] poll_interval=30
# INFO  Discovered queues via registry count=1
# INFO  Processing queue provider=gcs tenant=staging-tenant depth=1
# INFO  Fetched document uri=gs://activekg-staging-test/test-docs/staging-test.txt size=32
# INFO  Ingested document uri=... chunk_count=1
```

**Check queue depth:**

```bash
redis-cli -h staging-redis LLEN "connector:gcs:staging-tenant:queue"
```

Should return `0` (queue drained).

**Check database:**

```sql
-- Verify document chunks ingested
SELECT
  uri,
  tenant_id,
  embedding IS NOT NULL as has_embedding,
  created_at
FROM document_chunks
WHERE tenant_id = 'staging-tenant' AND uri LIKE '%staging-test.txt%';
```

Expected output:
```
 uri                                                | tenant_id      | has_embedding | created_at
----------------------------------------------------+----------------+---------------+---------------------------
 gs://activekg-staging-test/test-docs/staging-test.txt#0 | staging-tenant | t             | 2025-11-12 10:35:00+00
```

**Check Prometheus metrics:**

```bash
curl -s http://staging.activekg.example.com/metrics | grep connector_ingest_total
```

Should show:
```
connector_ingest_total{provider="gcs",tenant="staging-tenant"} 1.0
```

## Stage 4: Search Validation

### Test Search Endpoint

```bash
curl -X POST http://staging.activekg.example.com/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "staging test document",
    "tenant_id": "staging-tenant",
    "limit": 5
  }' | jq .
```

Expected response:
```json
{
  "results": [
    {
      "uri": "gs://activekg-staging-test/test-docs/staging-test.txt#0",
      "text": "This is a staging test document",
      "score": 0.95,
      "metadata": {
        "bucket": "activekg-staging-test",
        "object": "test-docs/staging-test.txt"
      }
    }
  ],
  "total": 1
}
```

## Stage 5: Backfill Validation

### Trigger Historical Backfill

```bash
curl -X POST http://staging.activekg.example.com/_admin/connectors/backfill \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gcs",
    "tenant_id": "staging-tenant",
    "connector_id": "conn_abc123"
  }'
```

Expected response:
```json
{
  "job_id": "backfill_xyz789",
  "status": "running",
  "objects_enqueued": 15
}
```

### Monitor Backfill Progress

**Check queue depth:**

```bash
watch -n 5 "redis-cli -h staging-redis LLEN 'connector:gcs:staging-tenant:queue'"
```

**Check ingestion rate:**

```promql
# In Grafana or Prometheus UI
rate(connector_ingest_total{provider="gcs",tenant="staging-tenant"}[5m])
```

**Check backfill completion:**

```bash
curl -X GET "http://staging.activekg.example.com/_admin/connectors/backfill/backfill_xyz789" | jq .
```

Expected response (when complete):
```json
{
  "job_id": "backfill_xyz789",
  "status": "completed",
  "objects_processed": 15,
  "errors": 0,
  "duration_seconds": 45
}
```

## Stage 6: Update and Delete Operations

### Test Document Update

```bash
# Update existing file
echo "This is an UPDATED staging test document" > staging-test.txt
gsutil cp staging-test.txt gs://activekg-staging-test/test-docs/staging-test.txt

# Wait for webhook + worker processing (30-60 seconds)

# Verify update in database
psql $ACTIVEKG_DSN -c "
  SELECT uri, text, updated_at
  FROM document_chunks
  WHERE tenant_id = 'staging-tenant' AND uri LIKE '%staging-test.txt%';
"
```

Expected: `text` should contain "UPDATED", `updated_at` should be recent.

### Test Document Deletion

```bash
# Delete file
gsutil rm gs://activekg-staging-test/test-docs/staging-test.txt

# Wait for webhook processing

# Verify soft-delete in database
psql $ACTIVEKG_DSN -c "
  SELECT uri, deleted_at, purge_after
  FROM document_chunks
  WHERE tenant_id = 'staging-tenant' AND uri LIKE '%staging-test.txt%';
"
```

Expected:
```
 uri                                                | deleted_at                | purge_after
----------------------------------------------------+---------------------------+---------------------------
 gs://activekg-staging-test/test-docs/staging-test.txt#0 | 2025-11-12 10:40:00+00    | 2025-12-12 10:40:00+00
```

### Test Manual Purge

```bash
curl -X POST http://staging.activekg.example.com/_admin/connectors/purge_deleted \
  -H "Content-Type: application/json" \
  -d '{
    "dry_run": false,
    "tenant_id": "staging-tenant"
  }' | jq .
```

Expected response:
```json
{
  "purged_chunks": 1,
  "purged_parents": 1,
  "dry_run": false
}
```

Verify deletion:

```bash
psql $ACTIVEKG_DSN -c "
  SELECT COUNT(*) FROM document_chunks
  WHERE tenant_id = 'staging-tenant' AND uri LIKE '%staging-test.txt%';
"
```

Should return `0`.

## Stage 7: Error Handling and Recovery

### Test Invalid Credentials

```bash
# Temporarily corrupt credentials
curl -X POST http://staging.activekg.example.com/_admin/connectors/update \
  -H "Content-Type: application/json" \
  -d '{
    "connector_id": "conn_abc123",
    "config": {
      "credentials_json": "{\"type\": \"invalid\"}"
    }
  }'

# Upload file and observe error
gsutil cp test.txt gs://activekg-staging-test/test-docs/error-test.txt
```

**Expected behavior:**
- Webhook succeeds (enqueues item)
- Worker fails to fetch document
- Metrics: `connector_ingest_errors_total` increments
- Alert: `IngestErrorRateHigh` fires if error rate >1%

**Verify metrics:**

```bash
curl -s http://staging.activekg.example.com/metrics | grep connector_ingest_errors_total
```

**Restore credentials:**

```bash
curl -X POST http://staging.activekg.example.com/_admin/connectors/update \
  -H "Content-Type: application/json" \
  -d '{
    "connector_id": "conn_abc123",
    "config": {
      "credentials_json": "<VALID_SERVICE_ACCOUNT_JSON>"
    }
  }'
```

### Test Worker Restart

```bash
# Stop worker
kubectl scale deployment/activekg-worker --replicas=0

# Upload files (queue will build up)
for i in {1..5}; do
  echo "Test document $i" > test$i.txt
  gsutil cp test$i.txt gs://activekg-staging-test/test-docs/test$i.txt
done

# Check queue depth
redis-cli -h staging-redis LLEN "connector:gcs:staging-tenant:queue"
# Should return 5

# Restart worker
kubectl scale deployment/activekg-worker --replicas=1

# Monitor queue drain
watch -n 2 "redis-cli -h staging-redis LLEN 'connector:gcs:staging-tenant:queue'"
# Should decrease to 0
```

## Stage 8: Monitoring and Alerts

### Verify Prometheus Metrics

All connector metrics should be available:

```bash
curl -s http://staging.activekg.example.com/metrics | grep -E '^connector_|^webhook_'
```

Expected metrics:
- `webhook_pubsub_verify_total`
- `webhook_topic_rejected_total`
- `connector_worker_queue_depth`
- `connector_ingest_total`
- `connector_ingest_errors_total`
- `connector_worker_process_duration_bucket`
- `connector_purger_total`
- `connector_rotation_total`
- `connector_config_decrypt_failures_total`

### Verify Grafana Dashboards

1. Navigate to Grafana staging instance
2. Import dashboards from `observability/dashboards/`
3. Verify panels populate with data:
   - Connector Overview: Shows webhook activity, queue depth, ingestion rate
   - Connector Queues: Shows detailed queue metrics, latency heatmap

### Trigger Test Alerts

**Test WebhookVerificationFailuresHigh:**

```bash
# Send 20 requests with invalid secret
for i in {1..20}; do
  curl -X POST http://staging.activekg.example.com/_webhooks/gcs \
    -H "X-PubSub-Token: INVALID_SECRET" \
    -d '{"message":{"data":"e30="}}'
done
```

Alert should fire after 5 minutes if >10% fail.

**Test ConnectorQueueDepthHigh:**

```bash
# Stop worker
kubectl scale deployment/activekg-worker --replicas=0

# Upload 1500 files
for i in {1..1500}; do
  echo "Load test $i" | gsutil cp - gs://activekg-staging-test/test-docs/load-test-$i.txt
done

# Wait 10 minutes
# Alert should fire when queue depth >1000 for 10m
```

## Stage 9: Load Testing

### Sustained Webhook Load

```bash
# Run in background
for i in {1..1000}; do
  echo "Load test document $i" > load-$i.txt
  gsutil cp load-$i.txt gs://activekg-staging-test/test-docs/load-$i.txt &

  # Rate limit: 10 uploads/sec
  if [ $((i % 10)) -eq 0 ]; then
    sleep 1
  fi
done
wait
```

**Monitor:**
- Queue depth (Grafana dashboard)
- Worker processing rate: `rate(connector_ingest_total[5m])`
- Latency: `histogram_quantile(0.95, rate(connector_worker_process_duration_bucket[5m]))`
- Error rate: Should remain <1%

### Verify Rate Limiting

```bash
# Fire 300 rapid webhook requests (exceeds 200 burst)
for i in {1..300}; do
  curl -s -X POST http://staging.activekg.example.com/_webhooks/gcs \
    -H "X-PubSub-Token: $PUBSUB_VERIFY_SECRET" \
    -H "Content-Type: application/json" \
    -d '{"message":{"data":"e30=","attributes":{"bucketId":"test","objectId":"test.txt","eventType":"OBJECT_FINALIZE"}}}' &
done
wait

# Check 429 responses in logs
kubectl logs -l app=activekg-api --tail=100 | grep "429"
```

Expected: Some requests receive HTTP 429 after burst limit exceeded.

## Stage 10: Scheduler Validation

### Enable Scheduler

```bash
# Update environment
export RUN_SCHEDULER=true

# Restart API server
kubectl rollout restart deployment/activekg-api
```

### Verify Scheduler Startup

```bash
kubectl logs -l app=activekg-api --tail=50 | grep scheduler
```

Expected log:
```
INFO  RefreshScheduler started has_triggers=True purge_enabled=True
INFO  Next purge cycle scheduled for 2025-11-13 02:00:00 UTC
```

### Test Manual Purge Cycle

```bash
# Upload and delete test file
echo "Purge test" > purge-test.txt
gsutil cp purge-test.txt gs://activekg-staging-test/test-docs/purge-test.txt
sleep 60  # Wait for ingestion
gsutil rm gs://activekg-staging-test/test-docs/purge-test.txt
sleep 30  # Wait for soft-delete

# Check soft-deleted count
psql $ACTIVEKG_DSN -c "
  SELECT COUNT(*) FROM document_chunks
  WHERE tenant_id = 'staging-tenant' AND deleted_at IS NOT NULL;
"

# Trigger manual purge (don't wait for 02:00 UTC)
curl -X POST http://staging.activekg.example.com/_admin/connectors/purge_deleted \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false, "tenant_id": "staging-tenant"}'

# Verify purge
psql $ACTIVEKG_DSN -c "
  SELECT COUNT(*) FROM document_chunks
  WHERE tenant_id = 'staging-tenant' AND deleted_at IS NOT NULL;
"
# Should return 0
```

## Rollout Acceptance Criteria

Before promoting to production, all criteria must be met:

### Functionality
- [ ] Connector registration succeeds
- [ ] Webhooks authenticate correctly (secret or OIDC)
- [ ] Worker discovers queues via registry (SMEMBERS)
- [ ] Documents ingest successfully
- [ ] Search returns ingested documents
- [ ] Updates replace existing chunks
- [ ] Deletes soft-delete chunks
- [ ] Purge removes soft-deleted chunks
- [ ] Backfill processes all historical objects

### Performance
- [ ] Queue depth remains <1000 under normal load
- [ ] Ingestion rate >10 docs/sec per worker
- [ ] p95 processing latency <5 seconds
- [ ] Error rate <1%
- [ ] Webhook rate limiting prevents abuse (429s at burst limit)

### Reliability
- [ ] Worker restarts gracefully (queue persists in Redis)
- [ ] Invalid credentials fail gracefully (errors logged, no crashes)
- [ ] Network interruptions recover (Pub/Sub reconnects)
- [ ] Registry fallback to SCAN works if Redis SET corrupted

### Monitoring
- [ ] All Prometheus metrics emit correctly
- [ ] Grafana dashboards populate with data
- [ ] Alerts fire at correct thresholds
- [ ] Runbooks accessible and accurate

### Security
- [ ] Credentials encrypted at rest (KEK)
- [ ] Webhook signatures verified
- [ ] Topic allowlists enforced
- [ ] Rate limiting protects against abuse

## Production Deployment

Once all acceptance criteria met:

1. **Update production environment variables**:
   ```bash
   export GCS_BUCKET="activekg-production"
   export PUBSUB_VERIFY_SECRET="<PRODUCTION_SECRET>"
   export CONNECTOR_KEK_V1="<PRODUCTION_KEK>"
   export RATE_LIMIT_ENABLED=true
   export RUN_SCHEDULER=true
   ```

2. **Deploy infrastructure**:
   - Create production GCS bucket
   - Configure Pub/Sub notifications
   - Create service account with minimal permissions

3. **Deploy application**:
   ```bash
   kubectl apply -f k8s/production/
   kubectl rollout status deployment/activekg-api
   kubectl rollout status deployment/activekg-worker
   ```

4. **Smoke test**:
   - Register one connector
   - Upload test file
   - Verify ingestion
   - Delete test data

5. **Monitor for 24 hours**:
   - Watch Grafana dashboards
   - Verify alerts don't fire
   - Check error logs

6. **Scale up**:
   ```bash
   kubectl scale deployment/activekg-worker --replicas=3
   ```

## Rollback Procedure

If issues arise in production:

1. **Disable webhook processing**:
   ```bash
   # Scale down workers
   kubectl scale deployment/activekg-worker --replicas=0
   ```

2. **Queue will buffer in Redis** (up to memory limits)

3. **Investigate issue** using logs, metrics, dashboards

4. **Deploy fix** to staging first

5. **Re-validate** using this guide

6. **Re-enable production**:
   ```bash
   kubectl scale deployment/activekg-worker --replicas=3
   ```

## Troubleshooting

For detailed troubleshooting procedures, see:
- `docs/operations/OPERATIONS.md#webhook-troubleshooting`
- `docs/operations/OPERATIONS.md#worker-troubleshooting`
- `docs/operations/OPERATIONS.md#ingestion-troubleshooting`

## Support

- **Alerts**: `observability/alerts/connector_alerts.yml`
- **Dashboards**: `observability/dashboards/*.json`
- **Runbooks**: `docs/operations/OPERATIONS.md`
- **Metrics**: http://staging.activekg.example.com/metrics
