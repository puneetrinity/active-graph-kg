# Active Graph KG Connector Observability Bundle

Complete observability package for Active Graph KG connector infrastructure, including Prometheus alerts, Grafana dashboards, and operational runbooks.

## Contents

```
observability/
├── alerts/
│   └── connector_alerts.yml          # Prometheus alerting rules
├── dashboards/
│   ├── connector_overview.json       # High-level connector metrics
│   └── connector_queues.json         # Detailed queue and performance
├── alertmanager.yml                  # Alertmanager configuration (Slack/PagerDuty)
├── README.md                          # This file
├── PRODUCTION_VERIFICATION.md        # Production rollout checklist
└── DASHBOARD_IMPROVEMENTS.md         # Dashboard optimization guide
```

## Quick Start

### 1. Prometheus Setup

Load the connector alerts into Prometheus:

**Option A: prometheus.yml reference**
```yaml
# prometheus.yml
rule_files:
  - "observability/alerts/connector_alerts.yml"
```

**Option B: Docker Compose**
```yaml
# docker-compose.yml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./observability/alerts:/etc/prometheus/alerts
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
```

**Verify alerts loaded:**
```bash
curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name=="connector-alerts")'
```

### 2. Grafana Dashboard Import

**Import via UI:**
1. Navigate to Grafana → Dashboards → Import
2. Upload `observability/dashboards/connector_overview.json`
3. Upload `observability/dashboards/connector_queues.json`
4. Select your Prometheus data source

**Import via API:**
```bash
# Connector Overview
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @observability/dashboards/connector_overview.json

# Connector Queues
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @observability/dashboards/connector_queues.json
```

**Import via provisioning:**
```yaml
# grafana/provisioning/dashboards/dashboards.yml
apiVersion: 1
providers:
  - name: 'Active Graph KG'
    folder: 'Connectors'
    type: file
    options:
      path: /etc/grafana/dashboards
```

Place dashboard JSONs in `/etc/grafana/dashboards/`.

### Nightly Retrieval Uplift Publishing (Core API)

The core API publishes retrieval MRR uplift (vs vector baseline) to a gauge for Grafana.

- Metric: `activekg_retrieval_uplift_mrr_percent{mode="hybrid|weighted"}`
- Dashboard Panels (core ops dashboard):
  - "Hybrid MRR Uplift (stat)" – shows latest uplift value
  - "Access Violations by Type (rate)" – governance signals over time
- CI workflow: `.github/workflows/nightly-proof.yml`
  - Seeds ground truth, runs triple retrieval quality, publishes uplift via `/_admin/metrics/retrieval_uplift`

Manual publish:
```bash
export API=http://localhost:8000
export TOKEN='<admin JWT>'
make retrieval-quality           # produce evaluation/weighted_search_results.json
make publish-retrieval-uplift    # post uplift to gauge
```

### 3. Verify Metrics Collection

Check that Active Graph KG is emitting connector metrics:

```bash
# Quick check: List all connector/webhook metrics (one-liner)
curl -s http://localhost:8000/prometheus | grep -E '^(connector_|webhook_)' | cut -d' ' -f1 | sort -u

# Verify Prometheus alert rules loaded
curl -s http://localhost:9090/api/v1/rules | jq '.data.groups[].name'

# Check specific metrics
curl -s http://localhost:8000/prometheus | grep webhook_pubsub_verify_total
curl -s http://localhost:8000/prometheus | grep webhook_sns_verify_total
curl -s http://localhost:8000/prometheus | grep connector_ingest_total
curl -s http://localhost:8000/prometheus | grep connector_worker_queue_depth
curl -s http://localhost:8000/prometheus | grep connector_dlq_depth
```

**Expected metrics** (minimum set for alerts):
- `webhook_pubsub_verify_total` - GCS Pub/Sub webhook verification results
- `webhook_sns_verify_total` - AWS SNS webhook verification results
- `webhook_topic_rejected_total` - Topic allowlist rejections
- `api_rate_limited_total` - Rate limiting (429) events
- `connector_worker_queue_depth` - Queue backlog gauge
- `connector_ingest_total` - Documents ingested counter
- `connector_ingest_errors_total` - Ingestion errors counter
- `connector_worker_batch_latency_seconds_bucket` - Processing latency histogram
- `connector_dlq_depth` - Dead letter queue depth gauge
- `connector_dlq_total` - DLQ items counter
- `connector_pubsub_reconnect_total` - Subscriber reconnect counter
- `connector_rotation_total` - Key rotation operations counter
- `connector_config_decrypt_failures_total` - Config decryption failures

If metrics are missing, ensure:
- `METRICS_ENABLED=true` environment variable is set
- Prometheus client library is installed: `pip install prometheus-client`
- `/prometheus` endpoint is accessible
- Active Graph KG server is running and has processed some connector activity

## Alert Reference

### Critical Alerts

| Alert | Threshold | Duration | Description |
|-------|-----------|----------|-------------|
| **IngestErrorRateHigh** | >1% errors | 10m | Connector ingestion errors exceed acceptable rate |
| **ConfigDecryptFailures** | >0 failures | 30m | KEK decryption failures (check key rotation) |

### Warning Alerts

| Alert | Threshold | Duration | Description |
|-------|-----------|----------|-------------|
| **WebhookVerificationFailuresHigh** | >10% failures | 5m | Webhook auth failures (check secrets/OIDC) |
| **WebhookTopicRejected** | >0 rejections | 1m | Topic not in allowlist |
| **ConnectorQueueDepthHigh** | >1000 items | 10m | Worker queue backlog growing |
| **IngestStalled** | 0 docs/sec | 30m | No ingestion activity detected |
| **PubSubReconnectsHigh** | >5 reconnects | 15m | Subscriber connection instability |
| **RotationErrors** | >0 errors | 30m | Key rotation failures |

## Dashboard Guide

### Connector Overview

**Purpose**: High-level health and activity monitoring

**Key Panels**:
- **Webhook Verification Rate**: Auth success/failure breakdown
- **Webhook Verification Failure Rate**: Percentage gauge (thresholds: 5%/10%)
- **Webhook Topic Rejections**: Count of allowlist mismatches
- **Connector Queue Depth**: Per-provider/tenant queue backlog
- **Ingestion Rate**: Documents processed per second
- **Ingestion Error Rate**: Percentage of failed ingests
- **Worker Processing Latency**: p50/p95/p99 latencies
- **Pub/Sub Reconnects**: Subscriber stability metric
- **Key Rotation Errors**: KEK rotation failures
- **Config Decryption Failures**: Credential decryption issues
- **Total Ingested Documents (24h)**: Daily throughput summary

**Recommended for**: NOC dashboards, SRE on-call rotation

### Connector Queue Details

**Purpose**: Deep-dive into queue performance and troubleshooting

**Key Panels**:
- **Queue Depth Over Time**: Trend analysis with max/mean/last calcs
- **Current Queue Depth by Provider/Tenant**: Table with color coding (green <500, yellow <1000, red ≥1000)
- **Worker Processing Rate**: Throughput by provider
- **Ingestion Errors by Provider**: Error breakdown by type
- **Processing Latency Heatmap**: Visual distribution of processing times
- **Recent Webhook Events**: Real-time webhook activity
- **Documents Ingested (24h Rolling)**: Volume tracker
- **Errors (Last Hour)**: Recent error count
- **Max Queue Depth (Last Hour)**: Peak backlog indicator

**Template Variables**:
- `$provider`: Filter by connector provider (s3, gcs, drive, etc.)
- `$tenant`: Filter by tenant ID

**Recommended for**: Incident investigation, capacity planning

## Alert Label Scopes

Understanding label scoping in alerts helps with filtering and troubleshooting:

| Alert | Scope | Labels Available |
|-------|-------|------------------|
| WebhookPubSubVerificationFailuresHigh | Aggregate | - |
| WebhookSnsVerificationFailuresHigh | Aggregate | - |
| WebhookTopicRejected | Per instance | `provider`, `tenant` |
| WebhookRateLimitingHigh | Per endpoint | `endpoint` |
| ConnectorQueueDepthHigh | **Per provider/tenant** | `provider`, `tenant` |
| ConnectorQueueDepthCritical | **Per provider/tenant** | `provider`, `tenant` |
| IngestErrorRateHigh | **Per provider** | `provider` |
| IngestStalled | **Per provider** | `provider` |
| ConnectorDLQDepthHigh | Aggregate | - |
| ConnectorDLQFlowDetected | Aggregate | - |

**Use in dashboards**: Filter panels by `{provider="gcs", tenant="acme-corp"}` to isolate specific connector instances.

**Use in Alertmanager**: Route alerts by label to different teams:
```yaml
# alertmanager.yml
routes:
  - match:
      provider: gcs
    receiver: gcs-team
  - match:
      provider: s3
    receiver: s3-team
```

## Metrics Reference

### Webhook Metrics

```promql
# Verification rate by result
rate(webhook_pubsub_verify_total[5m])
rate(webhook_sns_verify_total[5m])

# Topic rejections
increase(webhook_topic_rejected_total[10m])

# Rate limiting (429s)
increase(api_rate_limited_total{endpoint=~"webhook_.*"}[10m])
```

### Queue Metrics

```promql
# Queue depth by provider/tenant
connector_worker_queue_depth{provider="gcs", tenant="default"}

# Processing latency (histogram)
histogram_quantile(0.95, rate(connector_worker_batch_latency_seconds_bucket[5m]))
```

### Ingestion Metrics

```promql
# Ingestion rate
rate(connector_ingest_total[5m])

# Error rate
rate(connector_ingest_errors_total[5m]) / rate(connector_ingest_total[5m])
```

### DLQ Metrics

```promql
# Dead letter queue depth
connector_dlq_depth

# DLQ flow rate
rate(connector_dlq_total[10m])

# Total items added to DLQ
increase(connector_dlq_total[1h])
```

### Rotation Metrics

```promql
# Key rotation errors
increase(connector_rotation_total{result="error"}[30m])

# Decryption failures
increase(connector_config_decrypt_failures_total[30m])
```

## Troubleshooting

### No Metrics Available

**Symptoms**: Dashboards show "No data", Prometheus scrape failures

**Resolution**:
1. Verify `/prometheus` endpoint:
   ```bash
   curl http://localhost:8000/prometheus
   ```
2. Check Prometheus scrape config:
   ```yaml
   scrape_configs:
     - job_name: 'activekg'
       static_configs:
         - targets: ['localhost:8000']
   ```
3. Check Prometheus targets: http://localhost:9090/targets
4. Ensure `METRICS_ENABLED=true` in Active Graph KG environment

### Alerts Not Firing

**Symptoms**: Known issues (e.g., high error rate) not triggering alerts

**Resolution**:
1. Verify alert rule loaded:
   ```bash
   curl http://localhost:9090/api/v1/rules | jq '.data.groups[] | select(.name=="connector-alerts")'
   ```
2. Check alert state in Prometheus UI: http://localhost:9090/alerts
3. Verify Alertmanager configuration (if using)
4. Check PromQL expression evaluates correctly:
   ```bash
   curl -G http://localhost:9090/api/v1/query \
     --data-urlencode 'query=sum(rate(connector_ingest_errors_total[10m])) / clamp_min(sum(rate(connector_ingest_total[10m])), 1)'
   ```

### Dashboard Import Fails

**Symptoms**: "Invalid dashboard JSON", "Dashboard validation failed"

**Resolution**:
1. Ensure Prometheus data source is configured in Grafana
2. Update data source references in JSON:
   ```json
   {
     "datasource": "Prometheus"  // Change to your data source name
   }
   ```
3. Import via API with correct headers:
   ```bash
   curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
     -H "Content-Type: application/json" \
     -d @observability/dashboards/connector_overview.json
   ```

### Missing Panels or Data

**Symptoms**: Some panels show "No data" while others work

**Resolution**:
1. Verify metric exists in Prometheus:
   ```bash
   curl -G http://localhost:9090/api/v1/label/__name__/values | grep connector_
   ```
2. Check time range (some metrics may not have historical data)
3. Verify label filters match your environment:
   - Dashboard expects `provider` and `tenant` labels
   - Adjust template variables if using different label names

## Runbook Integration

All alerts include `runbook_url` annotations pointing to:

```
docs/operations/OPERATIONS.md#<section>
```

**Sections**:
- `#webhook-troubleshooting` - Webhook verification and topic issues
- `#worker-troubleshooting` - Queue depth and processing errors
- `#ingestion-troubleshooting` - Error rates and stalled ingestion
- `#cache-subscriber` - Pub/Sub reconnection issues
- `#key-rotation` - KEK rotation and decryption failures

See `docs/operations/OPERATIONS.md` for detailed incident response procedures.

## Production Deployment Checklist

**Quick checklist** for production rollout:

- [ ] Prometheus alerts loaded and active
- [ ] Grafana dashboards imported and accessible
- [ ] Alertmanager configured (PagerDuty, Slack, etc.)
- [ ] `/prometheus` endpoint exposed (ensure no auth blocks scraping)
- [ ] Scrape interval set (recommended: 15s-30s)
- [ ] Retention configured (recommended: 30d minimum)
- [ ] Runbook URLs accessible to on-call team
- [ ] Dashboard URLs added to incident response playbooks
- [ ] SLOs defined for critical metrics (ingestion rate, error rate, queue depth)
- [ ] Baseline established for alert threshold tuning

**For detailed verification steps**, see: `observability/PRODUCTION_VERIFICATION.md`

This includes:
- PromQL sanity checks against live data
- Dry-fire alert tests (webhook failures, queue depth, DLQ)
- Alertmanager routing configuration examples
- 24h baseline observation guidelines
- Threshold tuning recommendations based on scale

## Tuning Alert Thresholds

Default thresholds are production-ready but may need adjustment based on your scale:

### High-Volume Environments (>1M docs/day)

```yaml
# Increase queue depth threshold
- alert: ConnectorQueueDepthHigh
  expr: max_over_time(connector_worker_queue_depth[10m]) > 5000  # Was 1000
```

### Strict SLAs (<0.1% error rate)

```yaml
# Lower error rate threshold
- alert: IngestErrorRateHigh
  expr: |
    sum(rate(connector_ingest_errors_total[10m]))
    /
    clamp_min(sum(rate(connector_ingest_total[10m])), 1) > 0.001  # Was 0.01
```

### Development/Staging

```yaml
# Increase duration before firing
- alert: IngestStalled
  expr: sum(rate(connector_ingest_total[30m])) == 0
  for: 2h  # Was 30m
```

## Maintenance

### Adding New Connector Providers

When adding new providers (Azure, etc.), no changes needed to alerts or dashboards - they automatically detect new `provider` label values.

**Verify new provider appears**:
```bash
# Prometheus
curl -G http://localhost:9090/api/v1/label/provider/values

# Grafana template variable will auto-populate
```

### Custom Alerts

Add provider-specific alerts to `connector_alerts.yml`:

```yaml
- alert: DriveQuotaExceeded
  expr: increase(connector_drive_quota_errors_total[1h]) > 0
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Google Drive API quota exceeded"
    description: "Drive connector hit quota limits."
    runbook_url: "docs/operations/OPERATIONS.md#drive-quota"
```

Reload Prometheus:
```bash
curl -X POST http://localhost:9090/-/reload
```

## Alertmanager Setup

### Configuration

Use the provided `observability/alertmanager.yml` for production-ready alert routing.

**Features**:
- Grouping by alertname/provider to prevent per-tenant notification storms
- Critical alerts → PagerDuty (1h repeat interval)
- Warning alerts → Slack (2h repeat interval)
- Inhibition rules (suppress warnings when critical alerts fire)
- Provider/tenant labels included in all notifications

### Quick Start

```bash
# Set environment variables
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
export PAGERDUTY_ROUTING_KEY="YOUR_PAGERDUTY_INTEGRATION_KEY"

# Validate configuration
amtool check-config observability/alertmanager.yml

# Start Alertmanager
alertmanager --config.file=observability/alertmanager.yml
```

### Test Alert Routing

```bash
# Test Slack receiver (warning alert)
amtool alert add test_warning severity=warning provider=gcs tenant=default \
  --alertmanager.url=http://localhost:9093

# Test PagerDuty receiver (critical alert)
amtool alert add test_critical severity=critical provider=s3 tenant=acme \
  --alertmanager.url=http://localhost:9093
```

### Silence Alerts During Maintenance

```bash
# Silence all alerts for a specific provider (2h)
amtool silence add provider="gcs" --duration=2h \
  --comment="GCS maintenance window" \
  --alertmanager.url=http://localhost:9093

# Silence specific alert for a tenant
amtool silence add alertname="ConnectorQueueDepthHigh" tenant="default" \
  --duration=1h --comment="Planned backfill" \
  --alertmanager.url=http://localhost:9093
```

For detailed configuration examples, see `observability/alertmanager.yml`.

---

## Dashboard Improvements

The provided Grafana dashboards are immediately usable, but can be enhanced for better incident triage and operational visibility.

See `observability/DASHBOARD_IMPROVEMENTS.md` for:
- Adding provider/tenant template variables to overview dashboard
- Verifying metric names match your environment
- Using `$__rate_interval` for adaptive rate windows
- Adding deployment annotations
- Best practices for color schemes and thresholds

---

## Support

- **Alerts**: See `observability/alerts/connector_alerts.yml` for alert definitions
- **Dashboards**: See `observability/dashboards/*.json` for panel queries
- **Alertmanager**: See `observability/alertmanager.yml` for routing configuration
- **Dashboard improvements**: See `observability/DASHBOARD_IMPROVEMENTS.md`
- **Production verification**: See `observability/PRODUCTION_VERIFICATION.md`
- **Runbooks**: See `docs/operations/OPERATIONS.md` for troubleshooting procedures
- **Metrics**: Active Graph KG emits Prometheus metrics at `/prometheus` endpoint

For issues or questions, consult the operations guide at `docs/operations/OPERATIONS.md`.
