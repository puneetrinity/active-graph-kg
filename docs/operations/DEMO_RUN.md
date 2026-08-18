# Active Graph KG — Demo Run (End-to-End)

This guide runs a complete proof bundle on a fresh or running system: seed ground truth → run triple retrieval comparison → publish uplift metrics → generate proof report → open Grafana.

Prerequisites:
- API running at `http://localhost:8000`
- Postgres running and reachable by the API
- Admin JWT token (single-line), export as `TOKEN`
- Grafana reachable at `http://localhost:3000` (optional)

Quick env:
```bash
export API=http://localhost:8000
export TOKEN='<admin JWT with admin:refresh>'
```

1) Seed Ground Truth (from current corpus)
```bash
make seed-ground-truth THRESH=0.10 TOPK=20
# Writes: evaluation/datasets/ground_truth.json
```

2) Triple Retrieval Quality (vector vs hybrid vs weighted)
```bash
make retrieval-quality
# Writes: evaluation/weighted_search_results.json
```

3) Publish Uplift Metrics to Prometheus (Grafana-ready)
```bash
make publish-retrieval-uplift
```

4) Generate Proof Points Report
```bash
make proof-report
# Writes: evaluation/PROOF_POINTS_REPORT.md
```

5) Open Grafana (optional)
- Vector Index Build Latency, Index Build Success Rate, search latency and refresh metrics
- URL: http://localhost:3000/d/activekg-ops or simply:
```bash
make open-grafana
```

Notes
- Scheduler metrics require API started with `RUN_SCHEDULER=true`.
- For governance tests, set a second token: `export SECOND_TOKEN='<jwt for other tenant>'` and run `make governance-audit`.
- All scripts reside under `scripts/` and are exposed as Makefile targets.
