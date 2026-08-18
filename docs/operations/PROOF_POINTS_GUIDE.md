# Proof Points & Observability Guide

## Overview

This guide covers the Active Graph KG proof points framework and observability stack, including automated validation scripts, metrics collection, and visualization.

> Current boundary: metrics require `ACTIVEKG_CONTROL_PLANE_TOKEN`; raw tenant/org-labelled samples are omitted.
> Ask, trigger and connector series are dormant definitions and are not launch-product evidence.

## Proof Points Report

### Quick Start

Generate a comprehensive proof points report:

```bash
export TOKEN='<your-admin-jwt>'
export API=http://localhost:8000
# Load ACTIVEKG_CONTROL_PLANE_TOKEN into the environment from your secret store.
test -n "$ACTIVEKG_CONTROL_PLANE_TOKEN"

# Basic report (metrics only)
make proof-report

# With live timing tests (creates test nodes)
export RUN_PROOFS=1
make proof-report

# View report
cat evaluation/PROOF_POINTS_REPORT.md
```

### Report Sections

The proof points report includes:

1. **Environment** - API URL and database connection
2. **Health** - System status, LLM backend configuration
3. **Embedding Health** - Coverage percentage and staleness metrics
4. **Search/Ask Activity** - Request counts by mode and score type
5. **Latency Snapshot** - p50/p95 search latency percentiles
6. **ANN Snapshot** - Index configuration and top similarity scores
7. **Embedding Coverage by Class** - Per-class breakdown (top 5)
8. **Scheduler Summary** - Last run timestamps for scheduled jobs
9. **Trigger Effectiveness** - Total triggers fired, pattern matching status
10. **Proof Metrics** - Live DX timing and ingestion E2E latency (if `RUN_PROOFS=1`)

### Live Proof Tests

When `RUN_PROOFS=1` is set, the report executes:

- **DX Timing Test** (`scripts/dx_timing.sh`)
  - Creates a test node
  - Refreshes to generate embedding
  - Searches until result appears
  - Reports time-to-searchable metric

- **Ingestion Pipeline Test** (`scripts/ingestion_pipeline.sh`)
  - Creates a document
  - Forces refresh
  - Waits for searchability
  - Reports end-to-end latency

**Note**: Live tests create temporary test nodes. Use with caution in production.

## Validation Scripts

### Core Validation Suite

| Script | Purpose | Token Required | Creates Test Data |
|--------|---------|----------------|-------------------|
| `metrics-probe` | Scrape Prometheus metrics | No | No |
| `live-smoke` | CRUD + search validation | Yes | Yes (temporary) |
| `live-extended` | Lineage, drift, events | Yes | Yes (temporary) |
| `proof-report` | Generate markdown report | Yes | Optional |

### Proof Point Scripts

| Script | Metric | Token Required | Creates Test Data |
|--------|--------|----------------|-------------------|
| `dx-timing` | Time to first searchable answer | Yes | Yes |
| `ingestion-pipeline` | End-to-end ingestion latency | Yes | Yes |
| `scheduler-sla` | Scheduler inter-run intervals | Yes | No |
| `trigger-effectiveness` | Pattern matching and firing | Yes | Yes |
| `governance-audit` | RLS cross-tenant isolation | Yes (2 tokens) | Yes |
| `failure-recovery` | Graceful degradation checks | Yes | No |

### Running Scripts

```bash
export TOKEN='<your-admin-jwt>'
export API=http://localhost:8000

# Individual scripts
make dx-timing
make ingestion-pipeline
make scheduler-sla
make trigger-effectiveness

# For governance audit (requires second tenant)
export SECOND_TOKEN='<other-tenant-jwt>'
make governance-audit

# Full validation flow
make live-smoke && make live-extended && make proof-report
```

## Admin Endpoints

### `/_admin/metrics_summary`

Returns runtime configuration and scheduler/trigger snapshots:

```json
{
  "version": "1.0.0",
  "rate_limit_enabled": false,
  "scheduler_enabled": true,
  "triggers": {
    "last_run": "2025-11-14T10:30:00Z",
    "patterns_registered": 5
  },
  "scheduler": {
    "jobs": {
      "refresh_cycle": "2025-11-14T10:29:45Z",
      "trigger_cycle": "2025-11-14T10:29:50Z"
    }
  }
}
```

**Usage**:
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/_admin/metrics_summary | jq .
```

### `/_admin/embed_class_coverage`

Returns embedding coverage breakdown by node class (top 50):

```json
{
  "classes": [
    {
      "class": "Document",
      "total": 1000,
      "with_embedding": 950,
      "coverage_pct": 95.0
    },
    {
      "class": "Paper",
      "total": 500,
      "with_embedding": 500,
      "coverage_pct": 100.0
    }
  ],
  "count": 2
}
```

**Usage**:
```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/_admin/embed_class_coverage | jq .
```

### `/_admin/embed_info`

Alias to `/debug/embed_info` for consistency. Returns embedding statistics:

```json
{
  "counts": {
    "total_nodes": 1500,
    "with_embedding": 1450,
    "without_embedding": 50
  },
  "last_refreshed": {
    "age_seconds": {
      "min": 10.5,
      "max": 600.2,
      "avg": 150.3
    }
  }
}
```

## Grafana Dashboard

### Import Dashboard

1. **Copy dashboard JSON**:
   ```bash
   cat observability/grafana-dashboard.json
   ```

2. **In Grafana UI**:
   - Navigate to **Dashboards** → **Import**
   - Paste JSON or upload `observability/grafana-dashboard.json`
   - Select Prometheus datasource
   - Click **Import**

### Dashboard Panels

The **Active Graph KG - Operations Dashboard** includes:

1. **Search Requests Rate** (Bar Gauge)
   - Rate of vector/hybrid/text searches
   - Grouped by mode and score type

2. **Search Latency p50/p95** (Time Series)
   - 50th and 95th percentile latencies
   - Split by search mode

3. **Embedding Coverage** (Gauge)
   - Percentage of nodes with embeddings
   - Per-tenant view

4. **Max Embedding Staleness** (Gauge)
   - Time since least-recently-refreshed node
   - Thresholds: green (<300s), yellow (<600s), red (>600s)

5. **Triggers Fired** (Time Series Bars)
   - Count of triggers fired per pattern
   - 1-hour rolling window

6. **Trigger Run Latency p50/p95** (Time Series)
   - Trigger execution time percentiles
   - By trigger mode

7. **Scheduler Runs** (Time Series Bars)
   - Count of scheduled job executions
   - Grouped by job_id and kind

8. **Scheduler Inter-Run Interval p50/p95** (Time Series)
   - Time between consecutive job runs
   - Per-job view for SLA monitoring

9. **Node Refresh Latency by Result p50/p95** (Time Series)
   - Embedding refresh time distribution
   - Split by ok/error outcomes

10. **Ask Requests** (Time Series Bars)
    - LLM ask endpoint usage
    - Grouped by rejection status

### Prometheus Queries

Key non-tenant-labelled metrics exposed at authenticated `/prometheus`:

- `activekg_search_requests_total{mode, score_type}`
- `activekg_search_latency_seconds_bucket{mode, score_type}`
- `activekg_schedule_runs_total{job_id, kind}`
- `activekg_schedule_inter_run_seconds_bucket{job_id}`
- `activekg_node_refresh_latency_seconds_bucket{result}`

### Custom Alerting Rules

Example Prometheus alert rules:

```yaml
groups:
  - name: activekg_slas
    interval: 30s
    rules:
      - alert: HighSearchLatency
        expr: histogram_quantile(0.95, rate(activekg_search_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Search p95 latency above 500ms"

      - alert: LowEmbeddingCoverage
        expr: activekg_embedding_coverage_ratio < 0.8
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Embedding coverage below 80%"

      - alert: SchedulerStalled
        expr: time() - activekg_schedule_last_run_timestamp > 600
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Scheduler has not run in 10+ minutes"
```

## CI/CD Integration

### GitHub Actions Workflows

#### Nightly Proof Report

Located at `.github/workflows/nightly-proof.yml`:

- **Schedule**: Daily at 3 AM UTC
- **Trigger**: Manual dispatch also available
- **Steps**:
  1. Spin up pgvector PostgreSQL service
  2. Start API with LLM disabled
  3. Run `metrics-probe`
  4. Run `proof-report`
  5. Upload report artifact

**Setup**:
1. Add `E2E_ADMIN_TOKEN` secret to GitHub repo
2. Workflow will auto-run nightly
3. Download artifacts from Actions tab

#### Live Validation (Manual)

Located at `.github/workflows/live-validation.yml`:

- **Trigger**: Manual dispatch with optional inputs
- **Inputs**:
  - `run_smoke`: Execute live_smoke.sh
  - `run_extended`: Execute live_extended.sh
- **Steps**: Same as nightly, plus optional smoke/extended tests

**Usage**:
1. Go to **Actions** → **Live Validation (Manual)**
2. Click **Run workflow**
3. Select desired test suites
4. Download proof report artifact

### Fetching Artifacts

**From GitHub UI**:
1. Navigate to **Actions** tab
2. Select workflow run
3. Download `proof-points-report` or `nightly-proof-points-report` artifact

**From CLI** (using `gh` tool):
```bash
# List recent workflow runs
gh run list --workflow=nightly-proof.yml

# Download latest artifact
gh run download --name nightly-proof-points-report

# View report
cat PROOF_POINTS_REPORT.md
```

## Production Recommendations

### Metrics Retention

Configure Prometheus retention based on usage:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# Retention (adjust based on disk space)
storage:
  tsdb:
    retention.time: 30d
    retention.size: 50GB
```

### Dashboard Variables

Add tenant/environment template variables for multi-tenant deployments:

```json
{
  "templating": {
    "list": [
      {
        "name": "tenant_id",
        "type": "query",
        "datasource": "Prometheus",
        "query": "label_values(activekg_search_requests_total, tenant_id)"
      }
    ]
  }
}
```

### Proof Cadence

Recommended validation frequency:

- **Nightly**: Automated proof report (via GitHub Actions)
- **Weekly**: Manual `live-extended` run with `RUN_PROOFS=1`
- **Pre-release**: Full suite including `governance-audit` and `failure-recovery`
- **On-demand**: After infrastructure changes or schema migrations

### Security Considerations

- **JWT Tokens**: Store as GitHub secrets, rotate regularly
- **Test Data**: Live proof tests create nodes; ensure cleanup in scripts
- **Rate Limits**: Proof scripts can generate burst traffic; disable rate limiting for test tokens
- **Multi-Tenancy**: Use `SECOND_TOKEN` for governance audit only in isolated test environments

## Troubleshooting

### Common Issues

**"Missing Authorization header"**
- Ensure `TOKEN` is exported and valid
- Check JWT expiration (`exp` claim)
- Verify `JWT_ENABLED=true` in API config

**"Proof report shows zero metrics"**
- Run `make live-smoke` first to populate metrics
- Check authenticated Prometheus access with `scripts/metrics_probe.sh`
- Verify scheduler is running if checking `schedule_runs_total`

**"RUN_PROOFS=1 doesn't execute tests"**
- Export variable: `export RUN_PROOFS=1`
- Don't use `RUN_PROOFS=1 make proof-report` (shell expansion issue)
- Check script permissions: `chmod +x scripts/*.sh`

**"Grafana dashboard shows no data"**
- Verify Prometheus datasource is configured
- Check time range (default: last 1 hour)
- Ensure the scraper sends the API control-plane bearer from its secret store

## Next Steps

1. **Extend Metrics**: Add connector-specific metrics (GCS, S3, Drive) using same pattern
2. **Custom Proofs**: Create domain-specific validation scripts in `scripts/`
3. **Alerting**: Set up Prometheus AlertManager for proactive monitoring
4. **Tracing**: Integrate OpenTelemetry for distributed tracing
5. **Chaos Testing**: Add `/_admin/simulate_failure` endpoint (opt-in, guarded)

## References

- [Prometheus Query Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
- [Grafana Provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/)
- [GitHub Actions Artifacts](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts)
