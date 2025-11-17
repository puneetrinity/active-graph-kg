# Scripts Catalog

**Last Updated**: 2025-11-17

All validation, proof, and utility scripts live here and are exposed via Makefile targets.

## Common Variables

```bash
export API=http://localhost:8000
export TOKEN='<admin JWT for protected endpoints>'
export SECOND_TOKEN='<optional, for cross-tenant governance tests>'
```

---

## Core Validation

| Script | Description | Makefile Target |
|--------|-------------|-----------------|
| `live_smoke.sh` | CRUD, ANN, vector/hybrid search, ask/stream, metrics | `make live-smoke` |
| `live_extended.sh` | Drift, lineage, events, triggers | `make live-extended` |
| `metrics_probe.sh` | Prometheus counters/histograms summary | `make metrics-probe` |
| `proof_points_report.sh` | Builds evaluation/PROOF_POINTS_REPORT.md | `make proof-report` |

## Evaluation & Benchmarks

| Script | Description | Makefile Target |
|--------|-------------|-----------------|
| `seed_ground_truth.sh` | Populate evaluation/datasets/ground_truth.json | - |
| `retrieval_quality.sh` | Vector vs hybrid vs weighted (triple mode) | `make retrieval-quality` |
| `qa_benchmark.sh` | LLM Q&A latency and accuracy | `make qa-benchmark` |
| `publish_retrieval_uplift.sh` | Expose uplift as Prom stats (Grafana) | `make publish-retrieval-uplift` |
| `search_latency_eval.sh` | p50/p95/p99 for vector/hybrid | - |

## Ops & SRE

| Script | Description | Makefile Target |
|--------|-------------|-----------------|
| `db_bootstrap.sh` | Initialize DB schema (extension, init.sql, RLS) | `make db-bootstrap` |
| `db_index_metrics.sh` | Index sizes, table sizes | - |
| `scheduler_sla.sh` | Inter-run intervals; first-token SSE latency | - |
| `dx_timing.sh` | Developer experience timing (startup, first request) | - |
| `ingestion_pipeline.sh` | Ingestion throughput testing | - |
| `failure_recovery.sh` | Graceful failure modes (timeouts, DLQ, resume) | - |
| `governance_audit.sh` | RLS isolation audit | - |
| `governance_demo.sh` | RLS isolation demo/proof | - |
| `rate_limit_validation.sh` | Rate limiting validation | - |

## Python Utilities

| Script | Description |
|--------|-------------|
| `smoke_test.py` | E2E integration tests (API running required) |
| `backend_readiness_check.py` | Comprehensive readiness validation |
| `generate_test_jwt.py` | Generate test JWT tokens |
| `e2e_api_smoke.py` | API smoke test (CI/CD) |
| `admin_refresh_by_class.py` | Admin refresh by class |
| `debug_repo_search.py` | Debug repository search |

## Deployment

| Script | Description |
|--------|-------------|
| `dev_up.sh` | Start development environment |
| `start_api.sh` | Start API server with standard config |
| `db_bootstrap.sh` | Database schema bootstrap |

---

## Usage via Makefile

```bash
export API=http://localhost:8000
export TOKEN='<admin JWT>'

# Core validation
make live-smoke
make live-extended
make metrics-probe

# Retrieval quality + proof report
make retrieval-quality && make publish-retrieval-uplift
make proof-report            # writes evaluation/PROOF_POINTS_REPORT.md

# DB bootstrap and indexes
make db-bootstrap            # uses ACTIVEKG_DSN or DATABASE_URL
curl -X POST "$API/admin/indexes" -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' -d '{"action":"ensure","types":["ivfflat","hnsw"],"metric":"cosine"}'

# Demo run
make demo-run && make open-grafana
```

---

## Environment Requirements

- **PostgreSQL 16** with pgvector extension
- **Redis** (for rate limiting, cache pub/sub)
- **Python 3.11+** with venv activated
- **JWT Token** (use `generate_test_jwt.py` for testing)

## Notes

- All scripts assume standard port 5432 for Postgres, 8000 for API
- Scripts use `ACTIVEKG_DSN` or `DATABASE_URL` fallback
- Most bash scripts are executable (`chmod +x`)
- Python scripts require venv activation
