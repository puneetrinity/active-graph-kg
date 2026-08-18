#!/usr/bin/env bash
set -euo pipefail

# Summarize key Prometheus metrics exposed by the API

API_URL=${API_URL:-${API:-http://localhost:8000}}
CONTROL_PLANE_TOKEN=${ACTIVEKG_CONTROL_PLANE_TOKEN:-}

if [[ -z "${CONTROL_PLANE_TOKEN}" ]]; then
  echo "ERROR: ACTIVEKG_CONTROL_PLANE_TOKEN must be set before metrics access." >&2
  exit 1
fi

METRICS=$(curl -sS -H "Authorization: Bearer ${CONTROL_PLANE_TOKEN}" "${API_URL}/prometheus")

echo "== Request Counters =="
echo "$METRICS" | grep '^activekg_search_requests_total' || true
echo "$METRICS" | grep '^activekg_ask_requests_total' || true
echo

echo "== Gating Score Histogram Buckets =="
echo "$METRICS" | grep '^activekg_gating_score_bucket' | head -n 10 || true
echo

echo "== Search Latency Histogram =="
echo "$METRICS" | grep '^activekg_search_latency_seconds_bucket' | head -n 10 || true
echo

echo "== Ask Latency Histogram =="
echo "$METRICS" | grep '^activekg_ask_latency_seconds_bucket' | head -n 10 || true
echo

echo "== Embedding Health Gauges =="
echo "$METRICS" | grep '^activekg_embedding_coverage_ratio' || true
echo "$METRICS" | grep '^activekg_embedding_max_staleness_seconds' || true
echo

echo "== Zero-citation / Rejections (if any) =="
echo "$METRICS" | grep '^activekg_zero_citations_total' || true
echo "$METRICS" | grep '^activekg_rejections_total' || true
echo

echo "✓ Metrics probe complete"
