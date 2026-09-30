# Observability Production Verification Checklist

Quick, practical verification steps before rolling observability to production/staging.

---

## 1. Load Rules and Verify State

### Load Prometheus Alert Rules
```bash
# Verify alert group loaded
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name=="connector-alerts")'

# Check current alert state
curl -s http://localhost:9090/api/v1/alerts

# Verify rule count (should be 15 active + 3 optional commented)
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name=="connector-alerts") | .rules | length'
```

---

## 2. Sanity-Check PromQL Against Live Data

Run these queries in Prometheus UI or via API to verify metrics exist and queries work:

### Webhook Verification (Pub/Sub)
```promql
sum by (result) (rate(webhook_pubsub_verify_total[5m]))
```
**Expected**: Labels like `secret_ok`, `oidc_ok`, `skipped`, or error states

### Webhook Verification (SNS)
```promql
sum by (result) (rate(webhook_sns_verify_total[5m]))
```
**Expected**: Labels like `success`, `skipped`, or error states

### Queue Depth (Per Provider/Tenant)
```promql
max by (provider, tenant) (max_over_time(connector_worker_queue_depth[10m]))
```
**Expected**: One series per `{provider="gcs", tenant="default"}` combination

### Ingestion Error Rate (Per Provider)
```promql
sum by (provider) (rate(connector_ingest_errors_total[10m])) /
clamp_min(sum by (provider) (rate(connector_ingest_total[10m])), 1)
```
**Expected**: Ratio between 0.0 and 1.0 (0% to 100% error rate)

### DLQ Depth
```promql
max(connector_dlq_depth)
```
**Expected**: Gauge value (0 if no persistent failures)

### DLQ Flow
```promql
increase(connector_dlq_total[10m])
```
**Expected**: Counter (0 if no new DLQ items)

### Rate-Limited Webhooks
```promql
increase(api_rate_limited_total{endpoint=~"webhook_.*"}[10m])
```
**Expected**: 0 under normal load, >0 during burst scenarios

---

## 3. Dry-Fire Tests (Staging Environment)

**Purpose**: Verify alerts fire correctly under known failure conditions

### Test 1: Webhook Verification Failure
```bash
# Send webhook with invalid token
curl -X POST http://localhost:8000/webhooks/pubsub/gcs \
  -H "Authorization: Bearer INVALID_TOKEN" \
  -d '{"message": {"data": "..."}}'

# Wait 5 minutes, check Prometheus alerts
curl -s http://localhost:9090/api/v1/alerts | jq '.data.alerts[] | select(.labels.alertname=="WebhookPubSubVerificationFailuresHigh")'
```
**Expected**: Alert fires if >10% failure rate over 5m

### Test 2: Queue Depth Warning
```bash
# Push >1000 fake items to connector queue (use test script or load generator)
# Monitor queue metric:
watch -n 1 'curl -s http://localhost:8000/metrics | grep connector_worker_queue_depth'

# Alert should fire after 10 minutes if depth stays >1000
```
**Expected**: `ConnectorQueueDepthHigh` fires, `ConnectorQueueDepthCritical` fires at >5000

### Test 3: DLQ Flow Detection
```bash
# Force a 403 provider error (e.g., revoke GCS credentials)
# Trigger webhook that attempts to fetch from GCS
# After max retries, item should land in DLQ

# Check DLQ metric:
curl -s http://localhost:8000/metrics | grep connector_dlq_total
```
**Expected**: `ConnectorDLQFlowDetected` fires when items added to DLQ

### Test 4: Subscriber Reconnects
```bash
# Simulate network partition (iptables drop or restart Pub/Sub emulator)
# Monitor reconnect metric:
curl -s http://localhost:8000/metrics | grep connector_pubsub_reconnect_total
```
**Expected**: `PubSubReconnectsHigh` fires if >5 reconnects in 15m

---

## 4. Alertmanager Configuration

### Grouping and Rate Limiting
```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'provider']  # Avoid per-tenant flood
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h
  receiver: 'connector-team'

  routes:
    # Route by provider
    - match:
        provider: gcs
      receiver: gcs-oncall
    - match:
        provider: s3
      receiver: s3-oncall

    # Route critical alerts to PagerDuty
    - match:
        severity: critical
      receiver: pagerduty-critical
      continue: true  # Also send to default receiver

receivers:
  - name: 'connector-team'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#connector-alerts'

  - name: 'gcs-oncall'
    slack_configs:
      - api_url: 'https://hooks.slack.com/services/...'
        channel: '#gcs-team'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: 'YOUR_PAGERDUTY_KEY'
```

### Silencing Patterns
```bash
# Silence all alerts for a specific provider during maintenance
amtool silence add alertname=~".*" provider="gcs" --duration=2h --comment="GCS maintenance window"

# Silence specific tenant
amtool silence add tenant="acme-corp" --duration=1h --comment="Acme planned downgrade"
```

---

## 5. Cardinality and Noise Control

### Verify Label Cardinality
```bash
# Check unique provider/tenant combinations
curl -s http://localhost:8000/metrics | grep connector_worker_queue_depth | cut -d'{' -f2 | cut -d'}' -f1 | sort -u

# Expected: Reasonable number (e.g., <100 unique combinations)
# If >1000, consider alert aggregation changes
```

### Alert Grouping Best Practices
- **Group by provider**: Prevents notification storm when one provider fails across many tenants
- **Tiered thresholds**: Warning at 1000, Critical at 5000 (already implemented)
- **Longer evaluation windows**: 60m for `IngestStalled` reduces flapping in low-traffic environments

---

## 6. Dashboard Quick Checks

### Import and Verify Dashboards
```bash
# Import via Grafana API
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @observability/dashboards/connector_overview.json

curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @observability/dashboards/connector_queues.json
```

### Template Variables Check
- Navigate to dashboard settings → Variables
- Verify `$provider` and `$tenant` are populated from Prometheus
- Test filtering: Select specific provider, ensure panels update

### Panel Spot Checks
1. **Queue Depth Over Time**: Should show color-coded lines with legend showing last/max/mean
2. **Current Queue Depth by Provider/Tenant**: Table with green (<500), yellow (<1000), red (≥1000)
3. **Webhook Verification Rate**: Should show breakdown by `secret_ok`/`oidc_ok`/`failed`
4. **Annotations**: Firing alerts should appear as vertical lines on time-series panels

### Add DLQ Panels (Optional Enhancement)
```json
{
  "id": 10,
  "title": "DLQ Depth",
  "type": "timeseries",
  "targets": [{
    "expr": "max(connector_dlq_depth)",
    "legendFormat": "DLQ Depth"
  }]
}
```

---

## 7. Runbook Links Validation

All alerts include `runbook_url` pointing to `docs/operations/OPERATIONS.md#<section>`.

### Verify Links Accessible
```bash
# Check runbook sections exist
grep -E "^## (Webhook|Worker|Ingestion|Cache|Key)" docs/operations/OPERATIONS.md
```

**Expected sections**:
- `## Webhook Troubleshooting`
- `## Worker Troubleshooting`
- `## Ingestion Troubleshooting`
- `## Cache Subscriber`
- `## Key Rotation`

### Add First-Checks to Runbooks
For each section, include 1-liner quick diagnostics:

**Example (Webhook section)**:
```markdown
### First Checks
- Verify webhook secret: `echo $GCS_WEBHOOK_SECRET | wc -c` (should be >32)
- Check recent webhook logs: `kubectl logs -l app=activekg --tail=100 | grep webhook`
- Test OIDC endpoint: `curl https://oauth2.googleapis.com/tokeninfo?id_token=...`
```

---

## 8. Production Rollout Acceptance Criteria

- [ ] All 15 alerts load successfully in Prometheus
- [ ] At least 3 dry-fire tests pass (webhook failure, queue depth, DLQ)
- [ ] Alertmanager routes configured (Slack + PagerDuty for critical)
- [ ] Dashboards import without errors
- [ ] Template variables (`$provider`, `$tenant`) populated
- [ ] Runbook URLs accessible to on-call team
- [ ] Baseline metrics observed for 24h in staging
- [ ] Alert thresholds tuned based on baseline (if needed)
- [ ] Silencing patterns documented for planned maintenance

---

## 9. Baseline Observation Period (Staging)

**Duration**: 24-48 hours before production rollout

### Metrics to Observe
1. **Queue depth peaks**: Note max queue depth during peak traffic
   - If consistently >1000, consider raising warning threshold to 2500
2. **Webhook verification failure rate**: Establish normal error rate
   - If baseline is 2%, lower alert threshold from 10% to 5%
3. **Ingestion error rate**: Measure typical transient error rate
   - If <0.1%, consider lowering critical threshold from 1% to 0.5%
4. **DLQ flow**: Verify DLQ is normally empty
   - Any baseline DLQ activity indicates underlying issue to fix first

### Tuning Based on Baseline
```yaml
# Example: High-traffic environment
- alert: ConnectorQueueDepthHigh
  expr: max by (provider, tenant) (max_over_time(connector_worker_queue_depth[10m])) > 2500  # Was 1000

# Example: Strict SLA environment
- alert: IngestErrorRateHigh
  expr: ... > 0.005  # Was 0.01 (0.5% instead of 1%)
```

---

## 10. Future Enhancements (Optional)

### SLO Burn-Rate Alerts
```yaml
# Fast burn (1% error budget in 1 hour)
- alert: IngestErrorBudgetBurnFast
  expr: |
    (
      sum(rate(connector_ingest_errors_total[1h]))
      / sum(rate(connector_ingest_total[1h]))
    ) > 0.01
  for: 5m

# Slow burn (10% error budget in 6 hours)
- alert: IngestErrorBudgetBurnSlow
  expr: |
    (
      sum(rate(connector_ingest_errors_total[6h]))
      / sum(rate(connector_ingest_total[6h]))
    ) > 0.001
  for: 30m
```

### Cache Subscriber Age Alert
```yaml
# Uncomment in connector_alerts.yml when metric available
- alert: CacheSubscriberStale
  expr: max(connector_last_message_age_seconds) > 300
  for: 5m
  annotations:
    summary: "No cache invalidation messages in 5 minutes"
```

---

## Green Light to Production

✅ **Proceed to production rollout** if:
- All verification steps pass
- Baseline metrics observed for 24h in staging
- At least 2 dry-fire tests successful
- Alertmanager routes configured and tested
- On-call team has access to runbooks

🚀 **Next Step**: Follow `docs/operations/GCS_STAGING_ROLLOUT.md` for step-by-step connector deployment with observability enabled.

---

## Support

- **Alerts file**: `observability/alerts/connector_alerts.yml`
- **README**: `observability/README.md`
- **Dashboards**: `observability/dashboards/*.json`
- **Operations runbook**: `docs/operations/OPERATIONS.md`
- **Staging rollout**: `docs/operations/GCS_STAGING_ROLLOUT.md`
