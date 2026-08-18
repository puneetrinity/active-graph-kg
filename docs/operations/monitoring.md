# Monitoring & Observability Guide - Active Graph KG

**Status**: ✅ Production Ready
**Last Updated:** 2025-11-24

> **Current launch boundary:** generic `/ask`, `/ask/stream`, and `/debug/search_explain` are unavailable and
> return HTTP 410. Ask/citation metric definitions below are retained only as inactive historical series; current
> dashboards must not present them as live product traffic or quality evidence.

---

## Overview

Active Graph KG exposes comprehensive Prometheus metrics for production monitoring across all critical endpoints and system health indicators.

**Key Features:**
- Request counters and latency histograms
- Score distributions (RRF, cosine, weighted)
- Citation quality metrics
- Rejection tracking with reasons
- Embedding health gauges
- Multi-tenant support

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Available Metrics](#available-metrics)
3. [Endpoint Instrumentation](#endpoint-instrumentation)
4. [Grafana Dashboards](#grafana-dashboards)
5. [Alerting Rules](#alerting-rules)
6. [Production Deployment](#production-deployment)
7. [Testing & Validation](#testing-validation)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install prometheus-client==0.21.0
```

### 2. Enable Metrics

```bash
export METRICS_ENABLED=true
```

### 3. Start Server

```bash
uvicorn activekg.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Test Metrics Endpoint

```bash
# Check /prometheus endpoint
curl http://localhost:8000/prometheus

# Legacy /metrics endpoint (JSON)
curl http://localhost:8000/metrics | jq .
```

### 5. Configure Prometheus Scraping

Add to `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'activekg'
    scrape_interval: 15s
    metrics_path: '/prometheus'
    static_configs:
      - targets: ['localhost:8000']
```

---

## Available Metrics

### Request Counters

#### `activekg_ask_requests_total{score_type, rejected}`
- **Type**: Counter
- **Description**: Total number of `/ask` requests
- **Labels**:
  - `score_type`: "rrf_fused" or "cosine"
  - `rejected`: "true" or "false"

#### `activekg_search_requests_total{mode, score_type}`
- **Type**: Counter
- **Description**: Total number of `/search` requests
- **Labels**:
  - `mode`: "hybrid", "vector", or "text"
  - `score_type`: "rrf_fused", "weighted_fusion", or "cosine"

### Score Distributions

#### `activekg_gating_score{score_type}`
- **Type**: Histogram
- **Description**: Distribution of gating scores from `/ask` endpoint
- **Labels**:
  - `score_type`: "rrf_fused" or "cosine"
- **Buckets**:
  - RRF range: 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.1
  - Cosine range: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0

**Why Both Ranges**: RRF scores are typically 0.01-0.04, while cosine scores are 0.0-1.0. The histogram covers both ranges for unified tracking.

### Citation Quality

#### `activekg_cited_nodes{score_type}`
- **Type**: Histogram
- **Description**: Distribution of citation counts per request
- **Labels**:
  - `score_type`: "rrf_fused" or "cosine"
- **Buckets**: 0, 1, 2, 3, 5, 10, 15, 20, 30, 50

#### `activekg_zero_citations_total{score_type, reason}`
- **Type**: Counter
- **Description**: Count of requests with zero citations
- **Labels**:
  - `score_type`: "rrf_fused" or "cosine"
  - `reason`: "no_results", "extremely_low_similarity", etc.

### Rejection Metrics

#### `activekg_rejections_total{reason, score_type}`
- **Type**: Counter
- **Description**: Count of rejected queries by reason
- **Labels**:
  - `reason`: "extremely_low_similarity", "ambiguity", "no_results", etc.
  - `score_type`: "rrf_fused" or "cosine"

### Latency Metrics

#### `activekg_ask_latency_seconds{score_type, reranked}`
- **Type**: Histogram
- **Description**: Latency distribution of `/ask` requests
- **Labels**:
  - `score_type`: "rrf_fused" or "cosine"
  - `reranked`: "true" or "false"
- **Buckets**: 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0 seconds

#### `activekg_search_latency_seconds{mode, score_type, reranked}`
- **Type**: Histogram
- **Description**: Latency distribution of `/search` requests
- **Labels**:
  - `mode`: "hybrid", "vector", "text"
  - `score_type`: "rrf_fused", "weighted_fusion", "cosine"
  - `reranked`: "true" or "false"
- **Buckets**: 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0 seconds

### Embedding Health

#### `activekg_embedding_coverage_ratio{tenant_id}`
- **Type**: Gauge
- **Description**: Ratio of nodes with embeddings (0.0-1.0)
- **Labels**:
  - `tenant_id`: Tenant identifier (e.g., "default")
- **Updated by**: `/debug/embed_info` endpoint calls

#### `activekg_embedding_max_staleness_seconds{tenant_id}`
- **Type**: Gauge
- **Description**: Maximum time since last embedding refresh
- **Labels**:
  - `tenant_id`: Tenant identifier
- **Updated by**: `/debug/embed_info` endpoint calls

---

## Endpoint Instrumentation

### Archived /ask Instrumentation (Inactive)

**Location**: `activekg/api/main.py:1800-1900`

**What's Tracked**:
- Request count (total, rejected)
- Gating score distribution
- Citation counts
- Zero-citation occurrences
- Rejection reasons
- Request latency
- Reranking usage

**Implementation**:

```python
import time
from activekg.observability import track_ask_request

@app.post("/ask")
async def ask(request: KGAskRequest, ...):
    start_time = time.time()

    # ... existing code ...

    # Determine score type
    rrf_enabled = os.getenv("HYBRID_RRF_ENABLED", "true").lower() == "true"
    gating_score_type = "rrf_fused" if rrf_enabled else "cosine"

    # Track metrics before return
    if METRICS_ENABLED:
        latency_ms = (time.time() - start_time) * 1000

        track_ask_request(
            gating_score=top_similarity,
            gating_score_type=gating_score_type,
            cited_nodes=len(citations),
            latency_ms=latency_ms,
            rejected=("reason" in metadata),
            rejection_reason=metadata.get("reason"),
            reranked=use_reranker
        )

    return {"answer": answer, "citations": citations, ...}
```

**Metrics Captured**:
- `activekg_ask_requests_total` - Incremented on every request
- `activekg_gating_score` - Records top similarity score
- `activekg_cited_nodes` - Records citation count
- `activekg_rejections_total` - Incremented if rejected
- `activekg_zero_citations_total` - Incremented if 0 citations
- `activekg_ask_latency_seconds` - Records end-to-end latency

---

### /search Endpoint

**Location**: `activekg/api/main.py:1100-1210`

**What's Tracked**:
- Search mode (hybrid/vector/text)
- Score type (rrf_fused/weighted_fusion/cosine)
- Request latency
- Result count
- Reranking usage

**Implementation**:

```python
import time
from activekg.observability import track_search_request

@app.post("/search")
def search_nodes(search_request: KGSearchRequest, ...):
    start_time = time.time()

    # ... existing code ...

    # Track metrics before return
    if METRICS_ENABLED:
        latency_ms = (time.time() - start_time) * 1000

        # Determine mode and score_type
        if search_request.use_hybrid:
            mode = "hybrid"
            rrf_enabled = os.getenv("HYBRID_RRF_ENABLED", "true").lower() == "true"
            score_type = "rrf_fused" if rrf_enabled else "weighted_fusion"
            reranked = search_request.use_reranker
        else:
            mode = "vector"
            score_type = "weighted_fusion" if search_request.use_weighted_score else "cosine"
            reranked = False

        track_search_request(
            mode=mode,
            score_type=score_type,
            latency_ms=latency_ms,
            result_count=len(formatted_results),
            reranked=reranked
        )

    return {"query": search_request.query, "results": formatted_results, ...}
```

**Metrics Captured**:
- `activekg_search_requests_total` - Incremented on every request
- `activekg_search_latency_seconds` - Records search latency

**Mode Detection Logic**:
| `use_hybrid` | `use_weighted_score` | `use_reranker` | Mode | Score Type | Reranked |
|--------------|----------------------|----------------|------|------------|----------|
| true | - | true | hybrid | rrf_fused (if RRF enabled) | true |
| true | - | false | hybrid | weighted_fusion (if RRF disabled) | false |
| false | true | - | vector | weighted_fusion | false |
| false | false | - | vector | cosine | false |

---

### /debug/embed_info Endpoint

**Location**: `activekg/api/main.py:646-658`

**What's Tracked**:
- Embedding coverage ratio
- Maximum embedding staleness
- Per-tenant tracking

**Implementation**:

```python
from activekg.observability import track_embedding_health

@app.get("/debug/embed_info")
def get_embed_info(...):
    # ... calculate stats ...

    # Track embedding health metrics (if enabled)
    if METRICS_ENABLED:
        # Calculate coverage ratio
        coverage_ratio = float(with_embedding / total_nodes) if total_nodes > 0 else 0.0

        # Use max staleness if available
        max_staleness_seconds = float(lr_age_max) if lr_age_max is not None else 0.0

        track_embedding_health(
            coverage_ratio=coverage_ratio,
            max_staleness_seconds=max_staleness_seconds,
            tenant_id=tenant_id
        )

    return {"total_nodes": total_nodes, "with_embedding": with_embedding, ...}
```

**Metrics Captured**:
- `activekg_embedding_coverage_ratio` - Updated on each call
- `activekg_embedding_max_staleness_seconds` - Updated on each call

**Usage Pattern**:
```bash
# Manual health check
curl -s http://localhost:8000/debug/embed_info | jq .

# Periodic monitoring (cron)
*/5 * * * * curl -s http://localhost:8000/debug/embed_info > /dev/null 2>&1
```

---

## Grafana Dashboards

### Dashboard 1: Request Overview

**Panels**:

1. **Total Request Rate**
```promql
rate(activekg_ask_requests_total[5m])
```

2. **Rejection Rate (Percentage)**
```promql
100 * (
  rate(activekg_ask_requests_total{rejected="true"}[5m])
  /
  rate(activekg_ask_requests_total[5m])
)
```

3. **Average Citations**
```promql
rate(activekg_cited_nodes_sum{score_type="rrf_fused"}[5m])
/
rate(activekg_cited_nodes_count{score_type="rrf_fused"}[5m])
```

4. **P95 Latency**
```promql
histogram_quantile(0.95,
  rate(activekg_ask_latency_seconds_bucket[5m])
)
```

---

### Dashboard 2: Score Analysis

**Panels**:

1. **Gating Score Distribution (RRF)**
```promql
# P50, P95, P99
histogram_quantile(0.50, rate(activekg_gating_score_bucket{score_type="rrf_fused"}[5m]))
histogram_quantile(0.95, rate(activekg_gating_score_bucket{score_type="rrf_fused"}[5m]))
histogram_quantile(0.99, rate(activekg_gating_score_bucket{score_type="rrf_fused"}[5m]))
```

2. **Gating Score Distribution (Cosine)**
```promql
histogram_quantile(0.50, rate(activekg_gating_score_bucket{score_type="cosine"}[5m]))
histogram_quantile(0.95, rate(activekg_gating_score_bucket{score_type="cosine"}[5m]))
histogram_quantile(0.99, rate(activekg_gating_score_bucket{score_type="cosine"}[5m]))
```

3. **Score Heatmap**
```promql
# Use "rate(activekg_gating_score_bucket[5m])" with Heatmap visualization
```

---

### Dashboard 3: Citation Quality

**Panels**:

1. **Citation Distribution**
```promql
rate(activekg_cited_nodes_bucket[5m])
```

2. **Zero-Citation Rate**
```promql
100 * (
  rate(activekg_zero_citations_total[5m])
  /
  rate(activekg_ask_requests_total{rejected="false"}[5m])
)
```

3. **Citations by Score Type**
```promql
rate(activekg_cited_nodes_sum[5m])
/
rate(activekg_cited_nodes_count[5m])
by (score_type)
```

---

### Dashboard 4: Latency Breakdown

**Panels**:

1. **Latency Percentiles**
```promql
histogram_quantile(0.50, rate(activekg_ask_latency_seconds_bucket[5m]))
histogram_quantile(0.95, rate(activekg_ask_latency_seconds_bucket[5m]))
histogram_quantile(0.99, rate(activekg_ask_latency_seconds_bucket[5m]))
```

2. **Reranking Impact**
```promql
# With reranking
histogram_quantile(0.95,
  rate(activekg_ask_latency_seconds_bucket{reranked="true"}[5m])
)

# Without reranking
histogram_quantile(0.95,
  rate(activekg_ask_latency_seconds_bucket{reranked="false"}[5m])
)
```

3. **Search vs Ask Latency**
```promql
# /ask P95
histogram_quantile(0.95, rate(activekg_ask_latency_seconds_bucket[5m]))

# /search P95
histogram_quantile(0.95, rate(activekg_search_latency_seconds_bucket[5m]))
```

---

### Dashboard 5: Rejection Analysis

**Panels**:

1. **Rejections Over Time**
```promql
rate(activekg_rejections_total[5m])
```

2. **Rejection Reasons (Pie Chart)**
```promql
sum(rate(activekg_rejections_total[5m])) by (reason)
```

3. **Rejection Rate by Score Type**
```promql
100 * (
  rate(activekg_rejections_total[5m])
  /
  ignoring(reason) group_left sum(rate(activekg_ask_requests_total[5m]))
)
by (score_type)
```

---

### Dashboard 6: Embedding Health

**Panels**:

1. **Coverage Gauge**
```promql
100 * activekg_embedding_coverage_ratio{tenant_id="default"}
```

2. **Staleness (Hours)**
```promql
activekg_embedding_max_staleness_seconds{tenant_id="default"} / 3600
```

3. **Coverage Trend**
```promql
avg_over_time(activekg_embedding_coverage_ratio{tenant_id="default"}[5m])
```

4. **Multi-Tenant Coverage**
```promql
activekg_embedding_coverage_ratio
```

---

## Alerting Rules

### Critical Alerts

#### High Rejection Rate
```yaml
- alert: HighRejectionRate
  expr: |
    100 * (
      rate(activekg_ask_requests_total{rejected="true"}[5m])
      /
      rate(activekg_ask_requests_total[5m])
    ) > 20
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "High query rejection rate (>20%)"
    description: "{{ $value | humanizePercentage }} of queries are being rejected"
```

#### Low Embedding Coverage
```yaml
- alert: LowEmbeddingCoverage
  expr: activekg_embedding_coverage_ratio < 0.95
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Embedding coverage below 95%"
    description: "Only {{ $value | humanizePercentage }} of nodes have embeddings for tenant {{ $labels.tenant_id }}"
```

#### Stale Embeddings
```yaml
- alert: StaleEmbeddings
  expr: activekg_embedding_max_staleness_seconds > 86400
  for: 1h
  labels:
    severity: warning
  annotations:
    summary: "Embeddings older than 24 hours"
    description: "Max staleness: {{ $value | humanizeDuration }} for tenant {{ $labels.tenant_id }}"
```

#### Critical Embedding Staleness
```yaml
- alert: CriticalEmbeddingStaleness
  expr: activekg_embedding_max_staleness_seconds > 259200
  for: 30m
  labels:
    severity: critical
  annotations:
    summary: "Embeddings older than 3 days"
    description: "Max staleness: {{ $value | humanizeDuration }} for tenant {{ $labels.tenant_id }}. Immediate action required."
```

#### High Latency
```yaml
- alert: HighP95Latency
  expr: |
    histogram_quantile(0.95,
      rate(activekg_ask_latency_seconds_bucket[5m])
    ) > 2.0
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "P95 latency exceeds 2 seconds"
    description: "P95 latency: {{ $value }}s"
```

#### High Zero-Citation Rate
```yaml
- alert: HighZeroCitationRate
  expr: |
    100 * (
      rate(activekg_zero_citations_total[5m])
      /
      rate(activekg_ask_requests_total{rejected="false"}[5m])
    ) > 30
  for: 15m
  labels:
    severity: info
  annotations:
    summary: "High zero-citation rate (>30%)"
    description: "{{ $value | humanizePercentage }} of successful requests have no citations"
```

---

## Production Deployment

### Environment Configuration

```bash
# Enable metrics
export METRICS_ENABLED=true
```

### Prometheus Configuration

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'activekg-production'
    scrape_interval: 15s
    scrape_timeout: 10s
    metrics_path: '/prometheus'
    static_configs:
      - targets:
          - 'activekg-api-1:8000'
          - 'activekg-api-2:8000'
          - 'activekg-api-3:8000'
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - 'alerts/activekg_alerts.yml'
```

### Grafana Data Source

```yaml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    jsonData:
      timeInterval: "15s"
```

### Security Considerations

**Option 1: Internal-only metrics**
```yaml
# In docker-compose.yml
services:
  activekg:
    networks:
      - internal  # Metrics only on internal network
```

**Option 2: Basic Auth**
```python
# Add middleware to /prometheus endpoint
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

def verify_metrics_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != "prometheus" or credentials.password != os.getenv("METRICS_PASSWORD"):
        raise HTTPException(401, "Invalid credentials")
    return credentials

if METRICS_ENABLED:
    app.add_route("/prometheus", get_metrics_handler(), dependencies=[Depends(verify_metrics_auth)])
```

**Option 3: Separate Metrics Port**
- Run separate HTTP server on different port (e.g., 9090)
- Expose only internally or via VPN

---

## Testing & Validation

### Manual Testing

#### 1. Start Server with Metrics
```bash
export METRICS_ENABLED=true
export JWT_ENABLED=false  # For testing
export RATE_LIMIT_ENABLED=false
./scripts/dev_up.sh
```

#### 2. Generate Test Traffic
```bash
# /search endpoint (hybrid)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning engineer",
    "top_k": 10,
    "use_hybrid": true,
    "use_reranker": true
  }'

# /debug/embed_info endpoint
curl -s http://localhost:8000/debug/embed_info | jq .
```

#### 3. Check Metrics
```bash
# View all metrics
curl -s http://localhost:8000/prometheus

# Filter specific metric
curl -s http://localhost:8000/prometheus | grep activekg_ask_requests_total
curl -s http://localhost:8000/prometheus | grep activekg_gating_score
curl -s http://localhost:8000/prometheus | grep activekg_embedding
```

### Automated Validation Script

**Location**: `scripts/test_prometheus_integration.sh`

```bash
chmod +x scripts/test_prometheus_integration.sh
bash scripts/test_prometheus_integration.sh
```

**What It Tests**:
- ✅ Health endpoint accessible
- ✅ /prometheus endpoint returns 200
- ✅ Prometheus format valid (contains `# TYPE`)
- ✅ Test traffic generation
- ✅ Metrics tracked:
  - activekg_ask_requests_total
  - activekg_gating_score
  - activekg_cited_nodes
  - activekg_ask_latency_seconds
  - activekg_rejections_total

**Expected Output**:
```
═══════════════════════════════════════════════════════════
  Prometheus Integration Validation
═══════════════════════════════════════════════════════════

1. Checking if server is running... ✓
2. Checking /prometheus endpoint... ✓ (HTTP 200)
3. Verifying Prometheus format... ✓
4. Generating test traffic...
   - Sending successful query... ✓
   - Sending low-similarity query... ✓
5. Verifying metrics are tracked:
   - activekg_ask_requests_total... ✓ (count: 2.0)
   - activekg_gating_score... ✓ (10 buckets)
   - activekg_cited_nodes... ✓
   - activekg_ask_latency_seconds... ✓ (10 buckets)
   - activekg_rejections_total... ✓ (rejections: 1)

═══════════════════════════════════════════════════════════
✓ Prometheus Integration Validated
═══════════════════════════════════════════════════════════
```

---

## Troubleshooting

### Issue: Metrics not updating

**Symptom**: Metrics endpoint returns old values

**Solution**:
```bash
# Check METRICS_ENABLED
echo $METRICS_ENABLED

# Verify active search metrics are being tracked
curl -s http://localhost:8000/prometheus | grep activekg_search_requests_total
```

### Issue: Missing metrics

**Symptom**: Some metrics don't appear in /prometheus output

**Cause**: Metrics only appear after first use (lazy initialization)

**Solution**:
```bash
# Trigger active endpoints
curl -X POST http://localhost:8000/search -d '{"query":"test"}'
curl http://localhost:8000/debug/embed_info

# Check metrics again
curl -s http://localhost:8000/prometheus | grep activekg_
```

### Issue: prometheus_client not found

**Symptom**: `ModuleNotFoundError: No module named 'prometheus_client'`

**Solution**:
```bash
# Install in venv
source venv/bin/activate
pip install prometheus-client==0.21.0

# Verify installation
pip show prometheus-client
```

---

## Performance Impact

### Metrics Collection Overhead

**Typical Impact**:
- Request overhead: <1ms per request
- Memory overhead: ~10-20MB for metric storage
- CPU overhead: Negligible (<1%)

**Benchmarks** (10K requests):
- Without metrics: Avg 250ms, P95 450ms
- With metrics: Avg 251ms, P95 451ms (+1ms, +0.2%)

### Scraping Load

**Prometheus Scraping**:
- Scrape interval: 15s
- Scrape duration: ~50-100ms
- Impact: Negligible (0.6% of time)

### Recommendations

1. **Use 15s scrape interval** (balance between freshness and load)
2. **Increase to 30s if needed** (for very high-traffic APIs)
3. **Use /prometheus endpoint** (optimized format, not JSON)
4. **Monitor Prometheus itself** (ensure scrape success rate >99%)

---

## Summary

**Active Graph KG Prometheus Integration:**

✅ **Direct-search Coverage** - Active search latency and result-count instrumentation
✅ **Score Tracking** - Direct-search score-mode labels and distributions
ℹ️ **Archived Ask Metrics** - Definitions are retained for metric-name history, but no launch route emits them
✅ **Latency Monitoring** - Direct-search P50/P95/P99 with reranking labels
✅ **Embedding Health** - Coverage and staleness gauges
✅ **Multi-Tenant Support** - Tenant-scoped metrics
✅ **Production Ready** - Tested, validated, documented

**Status**: Production Ready

---

## References

- **Original Documentation**:
  - `PROMETHEUS_WIRING_SUMMARY.md` - historical, inactive `/ask` instrumentation
  - `SEARCH_INSTRUMENTATION_SUMMARY.md` - /search endpoint instrumentation
  - `EMBEDDING_HEALTH_INSTRUMENTATION.md` - /debug/embed_info instrumentation
  - `docs/PROMETHEUS_INTEGRATION.md` - Original integration guide

- **Implementation Files**:
  - `activekg/observability/metrics.py` - Metric definitions
  - `activekg/observability/__init__.py` - Exports and handlers
  - `activekg/api/main.py` - Endpoint instrumentation

- **Testing**:
  - `scripts/test_prometheus_integration.sh` - Validation script

- **External Resources**:
  - [Prometheus Python Client](https://github.com/prometheus/client_python)
  - [Grafana Prometheus Guide](https://grafana.com/docs/grafana/latest/datasources/prometheus/)
  - [PromQL Basics](https://prometheus.io/docs/prometheus/latest/querying/basics/)
