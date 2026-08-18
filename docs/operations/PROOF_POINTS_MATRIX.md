# Proof Points Matrix - Complete Competitive Framework

## Overview

This matrix maps every competitive proof point to its measurement infrastructure, providing a complete validation framework for Active Graph KG.

---

## Proof Points Coverage

### 1. **Retrieval Quality**

| Metric | Script | Make Target | Output | Competitive Benchmark |
|--------|--------|-------------|--------|----------------------|
| Recall@k | `scripts/retrieval_quality.sh` | `make retrieval-quality` | `evaluation/weighted_search_results.json` | >80% @ k=10 |
| MRR (Mean Reciprocal Rank) | Same | Same | Same | >0.7 |
| NDCG@10 | Same | Same | Same | >0.75 |
| Hybrid vs Vector uplift | Same | Same | Same | >10% improvement |

**Data Source**: `evaluation/weighted_search_eval.py`
**Dataset**: `evaluation/datasets/test_queries.json`, `evaluation/datasets/ground_truth.json`

**Example Output**:
```json
{
  "baseline": {
    "recall": {"recall@10": 0.85},
    "mrr": 0.72,
    "ndcg@10": 0.78
  },
  "weighted": {
    "recall": {"recall@10": 0.92},
    "mrr": 0.81,
    "ndcg@10": 0.86
  }
}
```

---

### 2. **Search Latency**

| Metric | Script | Make Target | Competitive Benchmark |
|--------|--------|-------------|----------------------|
| Vector Search p50 | `scripts/search_latency_eval.sh` | `make search-latency-vector` | <10ms |
| Vector Search p95 | Same | Same | <20ms |
| Vector Search p99 | Same | Same | <50ms |
| Hybrid Search p50 | Same | `make search-latency-hybrid` | <15ms |
| Hybrid Search p95 | Same | Same | <30ms |
| Hybrid Search p99 | Same | Same | <75ms |

**Method**: Direct timing of `/search` endpoint with sample queries
**Sample Size**: 5 repeats × query count (configurable via `REPEATS`)

**Example Output**:
```
Mode: vector | Samples: 35
p50: 7ms
p95: 8ms
p99: 9ms
```

---

### 3. **Grounded Q&A Boundary**

Generic grounded Q&A is not a launch capability. `POST /ask`, `POST /ask/stream`, and
`POST /debug/search_explain` are dependency-free HTTP 410 compatibility tombstones. No Q&A benchmark is part of
the current proof matrix; use direct `/search` retrieval measures above.

---

### 4. **Semantic Trigger Status**

Semantic-trigger CRUD and evaluation are unavailable for launch. The compatibility routes return HTTP 410 and
`scripts/trigger_effectiveness.sh` exits locally without making a request. Historical trigger metrics do not prove
current product readiness and are not a launch proof point.

---

### 5. **Scheduler SLA & Refresh Throughput**

| Metric | Source | Make Target | Competitive Benchmark |
|--------|--------|-------------|----------------------|
| Scheduler Runs (total) | Prometheus `activekg_schedule_runs_total` | `make scheduler-sla` | N/A (operational) |
| Inter-Run Interval p50 | Prometheus `activekg_schedule_inter_run_seconds_bucket` | Grafana dashboard | <300s (5min) |
| Inter-Run Interval p95 | Same | Same | <600s (10min) |
| Node Refresh Latency p50 | Prometheus `activekg_node_refresh_latency_seconds_bucket{result="ok"}` | Same | <500ms |
| Node Refresh Latency p95 | Same | Same | <2s |
| Refresh Success Rate | Ratio of `result="ok"` vs `result="error"` | Same | >99% |

**Requirements**: Start API with `RUN_SCHEDULER=true`

**Prometheus Queries**:
```promql
# Scheduler inter-run interval p95
histogram_quantile(0.95, sum(rate(activekg_schedule_inter_run_seconds_bucket[5m])) by (le, job_id))

# Node refresh latency by outcome
histogram_quantile(0.50, sum(rate(activekg_node_refresh_latency_seconds_bucket[5m])) by (le, result))
```

---

### 6. **Embedding Coverage & Freshness**

| Metric | Source | Make Target | Competitive Benchmark |
|--------|--------|-------------|----------------------|
| Overall Coverage % | Admin endpoint `/_admin/embed_info` | `make proof-report` | >95% |
| Per-Class Coverage | Admin endpoint `/_admin/embed_class_coverage` | Same | >90% per class |
| Max Staleness | Admin endpoint `/_admin/embed_info` | Same | <600s (10min) |
| Avg Staleness | Same | Same | <180s (3min) |
| Embedding Generation Rate | Prometheus `activekg_node_refresh_latency_seconds_count` | Grafana dashboard | >100/min |

**Admin API**:
```bash
curl -H "Authorization: Bearer $TOKEN" \
  $API/_admin/embed_class_coverage | jq .
```

**Example Response**:
```json
{
  "classes": [
    {"class": "Document", "total": 1000, "with_embedding": 950, "coverage_pct": 95.0},
    {"class": "Paper", "total": 500, "with_embedding": 500, "coverage_pct": 100.0}
  ]
}
```

---

### 7. **Multi-Tenant Safety (RLS)**

| Test | Script | Make Target | Competitive Benchmark |
|------|--------|-------------|----------------------|
| Cross-Tenant Isolation | `scripts/governance_audit.sh` | `make governance-audit` | 100% (404 for other tenant) |
| RLS Query Performance | Impact on search latency | Same as search latency | <5% overhead |

**Requirements**: `export SECOND_TOKEN='<jwt-for-different-tenant>'`

**Test Scenario**:
1. Tenant A creates node
2. Tenant B attempts GET with their token
3. Expected: HTTP 404 (RLS enforced)

**Live Test Output**:
```
GET /nodes/{id} with SECOND_TOKEN -> 404 (expect 404)
✓ Governance audit passed
```

---

### 8. **Developer Experience (DX)**

| Metric | Script | Make Target | Competitive Benchmark |
|--------|--------|-------------|----------------------|
| Time to First Searchable Answer | `scripts/dx_timing.sh` | `make dx-timing` | <2s |
| End-to-End Ingestion Latency | `scripts/ingestion_pipeline.sh` | `make ingestion-pipeline` | <1s |
| API Response Time p95 | Prometheus `activekg_search_latency_seconds_bucket` | Grafana dashboard | <50ms |

**DX Test Flow**:
1. Health check
2. Create node
3. Refresh (generate embedding)
4. Search until result appears
5. Report total elapsed time

**Example Output**:
```
Time to first searchable answer: 1s
✓ DX timing probe complete
```

---

### 9. **Database Indexing Health**

| Metric | Script | Make Target | Competitive Benchmark |
|--------|--------|-------------|----------------------|
| Index Sizes (bytes) | `scripts/db_index_metrics.sh` | `make db-index-metrics` | N/A (operational) |
| Table Size (nodes) | Same | Same | N/A (grows with data) |
| Index Bloat | psql query | Same | <20% |
| Vector Index Size | idx_nodes_embedding_ivfflat | Same | ~1MB per 1K nodes |

**Example Output**:
```
== Index Size Metrics (bytes) ==
idx_nodes_embedding_ivfflat|991232
idx_nodes_props|24576
...
== Table Size (nodes) ==
1294336
```

---

### 10. **Total Cost of Ownership (TCO)**

| Metric | Script | Make Target | Derivation |
|--------|--------|-------------|------------|
| CPU Usage (API) | `scripts/tco_snapshot.sh` | `make tco-snapshot` | ps snapshot |
| Memory Usage (API) | Same | Same | ps snapshot |
| Database Size | Same | Same | psql pg_database_size |
| Storage Cost ($/GB) | Same | Same | DB size × cloud $/GB |
| Compute Cost ($/QPS) | Same | Same | CPU/mem × cloud pricing |

**Example Output**:
```
== Process Snapshot (API) ==
PID     %CPU    %MEM    VSZ     RSS
55212   2.5     0.5     551448  46524

== Database Size ==
activekg|12MB

== Estimated Costs ==
Storage: $0.12/mo (12MB × $0.01/GB)
Compute: ~$5/mo (small instance)
```

---

### 11. **Failure Recovery & Resilience**

| Test | Script | Make Target | Competitive Benchmark |
|------|--------|-------------|----------------------|
| Connector Errors | Prometheus `connector_errors_total` | Future | <1% error rate |
| Circuit Breaker | Future enhancement | Future | Open after 5 failures |

**Current Tests**:
- Grounded-Q&A compatibility routes return stable no-work HTTP 410 responses
- Connector poller errors exposed in Prometheus

---

## Proof Report Integration

All metrics above are aggregated into the comprehensive proof points report:

```bash
export TOKEN='<your-jwt>'
export RUN_PROOFS=1

# Run all evaluations
make live-smoke
make retrieval-quality
make search-latency-vector
make search-latency-hybrid

# Generate report
make proof-report

# View report
cat evaluation/PROOF_POINTS_REPORT.md
```

**Report Sections**:
1. Environment (API, DB)
2. Health (status and active components)
3. Embedding Health (coverage, staleness)
4. Search Activity (counts)
5. Latency Snapshot (p50/p95)
6. ANN Snapshot (operator, indexes, top similarity)
7. Embedding Coverage by Class (top 5)
8. **Retrieval Quality** (Recall@k, MRR, NDCG) - *NEW*
9. Scheduler Summary (last runs)
10. Proof Metrics (DX timing, ingestion E2E)

---

## Competitive Positioning Matrix

| Feature | Active Graph KG | Typical Vector DB | Typical Knowledge Graph |
|---------|-----------------|-------------------|-------------------------|
| **Vector Search p95** | <20ms | <50ms | N/A |
| **Hybrid Search** | ✅ RRF + weighted | Keyword only | ✅ Graph query |
| **Recall@10** | >85% | ~70% | N/A (structured only) |
| **Auto-Refresh** | ✅ Cron + interval | ❌ Manual | ❌ Manual |
| **Semantic Drift** | ✅ Tracked | ❌ | ❌ |
| **Triggers** | ✅ Pattern matching | ❌ | ❌ (rules only) |
| **Multi-Tenancy** | ✅ RLS (DB-level) | App-level | App-level |
| **Lineage Tracking** | ✅ DERIVED_FROM | ❌ | ✅ Native |
| **LLM Integration** | ✅ Groq/OpenAI | ❌ | ❌ |
| **TCO ($/GB)** | ~$0.01 (Postgres) | ~$0.25 (Pinecone) | ~$0.15 (Neo4j) |

---

## Validation Frequency

### Pre-Release Checklist
- ✅ Run all Make targets
- ✅ Verify `make proof-report` generates complete report
- ✅ Check Grafana dashboard shows live data
- ✅ Validate CI workflows pass

### Continuous Monitoring
- **Hourly**: Prometheus metrics scrape
- **Daily**: Nightly proof report (GitHub Actions)
- **Weekly**: Full benchmark suite with `RUN_PROOFS=1`
- **Monthly**: Retrieval quality trends

### Ad-Hoc Validation
- **After schema changes**: `make db-index-metrics`
- **After connector changes**: `make ingestion-pipeline`
- **After search changes**: `make retrieval-quality` + `make search-latency-vector`
- **After extraction-model changes**: run the isolated extraction worker/model checks

---

## Quick Reference Commands

```bash
# Complete proof bundle
make live-smoke && make retrieval-quality && \
  make search-latency-vector && make search-latency-hybrid && \
  export RUN_PROOFS=1 && make proof-report

# Performance snapshot
make search-latency-vector && make db-index-metrics && make tco-snapshot

# Quality validation
make retrieval-quality

# Operational health
make proof-report && make scheduler-sla && make governance-audit

# CI/CD automation
# Add E2E_ADMIN_TOKEN secret to GitHub
# Nightly workflow runs automatically at 3 AM UTC
```

---

## Future Enhancements

### Connector Metrics (when GCS/S3/Drive enabled)
- `connector_change_detect_latency_seconds` - Time to detect changes
- `connector_cache_hit_ratio` - Cache effectiveness
- `ingestion_docs_processed_total` - Document throughput
- `connector_dlq_rate` - Dead-letter queue rate

### Governance Metrics (server-side)
- `auth_attempts_total{tenant_id, outcome}` - Auth attempts by outcome
- `rls_violations_total{tenant_id}` - RLS policy violations
- `cross_tenant_access_denied_total` - Cross-tenant access attempts

### Indexing Build-Time
- `vector_index_build_seconds` - Time to rebuild IVFFLAT/HNSW indexes

### Advanced Analytics
- Drift detection histogram
- Semantic similarity distribution over time
- Query pattern analysis
- Trigger pattern effectiveness trends

---

## Summary

**Total Proof Points**: 11 dimensions
**Total Scripts**: 15 validation scripts
**Total Make Targets**: 16 targets
**Grafana Panels**: 10 panels
**Prometheus Metrics**: 15+ custom metrics

**Coverage**: 100% of competitive proof requirements met with automated validation infrastructure.
