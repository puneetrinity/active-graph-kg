#!/usr/bin/env bash
set -euo pipefail

# Generate a Proof Points report from live API metrics and status
# Output: evaluation/PROOF_POINTS_REPORT.md
# Set RUN_PROOFS=1 to execute live proof scripts (dx_timing, ingestion_pipeline)

API_URL=${API_URL:-${API:-http://localhost:8000}}
TOKEN=${TOKEN:-${E2E_ADMIN_TOKEN:-}}
OUT=${OUT:-evaluation/PROOF_POINTS_REPORT.md}
RUN_PROOFS=${RUN_PROOFS:-0}

if [[ -z "${TOKEN}" ]]; then
  echo "ERROR: TOKEN env var not set. Export a single-line JWT into TOKEN." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

hdr_auth=( -H "Authorization: Bearer ${TOKEN}" )

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Collect core status
HEALTH=$(curl -sS "${API_URL}/health" || echo '{}')
DBINFO=$(curl -sS "${API_URL}/debug/dbinfo" "${hdr_auth[@]}" || echo '{}')
# Prefer admin path if available, fallback to debug path for older builds.
# Avoid pipefail aborting when status != 200 by evaluating in a variable.
_status=$(curl -sS -o /dev/null -w '%{http_code}' "${API_URL}/_admin/embed_info" "${hdr_auth[@]}" || echo "")
if [[ "${_status}" == "200" ]]; then
  EMBED=$(curl -sS "${API_URL}/_admin/embed_info" "${hdr_auth[@]}" || echo '{}')
else
  EMBED=$(curl -sS "${API_URL}/debug/embed_info" "${hdr_auth[@]}" || echo '{}')
fi
METRICS=$(curl -sS "${API_URL}/prometheus" || echo '')

# Class coverage (admin)
CLASS_COV=$(curl -sS "${API_URL}/_admin/embed_class_coverage" "${hdr_auth[@]}" || echo '{}')

# Metrics summary (scheduler)
MET_SUM=$(curl -sS "${API_URL}/_admin/metrics_summary" "${hdr_auth[@]}" || echo '{}')

# Helpers to extract numbers from Prometheus exposition
get_metric_val() {
  # $1: metric name regex, $2..$: substring filters
  local pattern="$1"; shift || true
  local filters=("$@")
  awk -v pat="$pattern" -v n=NF -v ORS="\n" '
    $0 ~ pat { 
      ok=1;
      for (i=1; i<ARGC; i++) { 
        if (ARGV[i] != "" && index($0, ARGV[i])==0) { ok=0; break } 
      }
      if (ok) print $NF 
    }' "$@" <<< "$METRICS" 2>/dev/null | tail -n 1
}

get_search_count() {
  # $1: mode (vector|hybrid|text), $2: score_type (cosine|rrf_fused|weighted_fusion)
  awk -v mode="$1" -v sc="$2" '
    /^activekg_search_requests_total/ && index($0, "mode=\""mode"\"") && index($0, "score_type=\""sc"\"") {
      s+=$NF
    }
    END { if (s=="") s=0; print s }
  ' <<< "$METRICS" 2>/dev/null
}

get_latency_bucket() {
  # $1: metric base name (without _bucket), $2: label selector (mode/score_type), $3: le cutoff (e.g., 0.05)
  local base="$1"; local selector="$2"; local le="$3"
  awk -v base="$base" -v sel="$selector" -v le="le=\""le"\"" '
    $0 ~ ("^"base"_bucket") && index($0, sel) && index($0, le) { v=$NF }
    END { if (v=="") v=0; print v }
  ' <<< "$METRICS" 2>/dev/null
}

get_latency_count() {
  # $1: metric base name, $2: label selector
  local base="$1"; local selector="$2"
  awk -v base="$base" -v sel="$selector" '
    $0 ~ ("^"base"_count") && index($0, sel) { v=$NF }
    END { if (v=="") v=0; print v }
  ' <<< "$METRICS" 2>/dev/null
}

# Compute vector p<=50ms fraction if available
VEC_LE_50=$(get_latency_bucket activekg_search_latency_seconds_bucket 'mode="vector".*score_type="cosine"' '0.05')
VEC_CNT=$(get_latency_count activekg_search_latency_seconds 'mode="vector".*score_type="cosine"')
if [[ -z "${VEC_LE_50}" || -z "${VEC_CNT}" || "${VEC_CNT}" == "" || "${VEC_CNT}" == 0 ]]; then
  VEC_FRAC="n/a"
else
  # multiply by 100 for percent, keep integer if possible
  VEC_FRAC=$(awk -v a="$VEC_LE_50" -v b="$VEC_CNT" 'BEGIN{ if(b==0){print "n/a"} else {printf "%.1f%%", (a/b)*100}}')
fi

# Extract embedding coverage/staleness
COVER=$(echo "$EMBED" | jq -r '.counts.with_embedding as $w | .counts.total_nodes as $t | if ($t//0)>0 then ((($w//0)/($t//0))*100) else 0 end' 2>/dev/null || echo 0)
STALENESS=$(echo "$EMBED" | jq -r '.last_refreshed.age_seconds.max // 0' 2>/dev/null || echo 0)

# Format coverage safely
if [[ "${COVER:-}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  COVER_FMT=$(printf '%.1f' "$COVER")
else
  COVER_FMT="0.0"
fi

# Search request counts
VEC_CNT_REQ=$(get_search_count vector cosine)
HYB_CNT_REQ=$(get_search_count hybrid rrf_fused)

# DB host/port
DB_HOST=$(echo "$DBINFO" | jq -r '.server_host // ""' 2>/dev/null || echo "")
DB_PORT=$(echo "$DBINFO" | jq -r '.server_port // ""' 2>/dev/null || echo "")

# Health summary lines (robust against jq failure)
HEALTH_SUMMARY=$(echo "$HEALTH" | jq -r '. | {status, llm_backend, llm_model} | to_entries[] | "- " + .key + ": " + ( .value|tostring )' 2>/dev/null || true)

# Format class coverage (top 5)
CLASS_TOP5=$(echo "$CLASS_COV" | jq -r '.classes[:5][] | "- \(.class): total=\(.total), with_embedding=\(.with_embedding), coverage=\(.coverage_pct)%"' 2>/dev/null || true)
SCHED_SUM=$(echo "$MET_SUM" | jq -r '.scheduler.jobs // {}' 2>/dev/null || true)

# Extract scheduler metrics from Prometheus
SCHED_RUNS=$(awk '/^activekg_schedule_runs_total/ {s+=$NF} END {if(s=="")s=0; print s}' <<< "$METRICS" 2>/dev/null || echo 0)

# Optional: Run live proof scripts if RUN_PROOFS=1
DX_TIMING=""
INGESTION_LATENCY=""
if [[ "${RUN_PROOFS}" == "1" ]]; then
  echo "Running live proof scripts..." >&2
  # DX timing test (capture timing line)
  DX_OUT=$(bash "$(dirname "$0")/dx_timing.sh" 2>&1 || true)
  DX_TIMING=$(echo "$DX_OUT" | grep -oP 'Time to first searchable answer: \K[0-9]+s' || echo "n/a")

  # Ingestion pipeline test (capture latency line)
  ING_OUT=$(bash "$(dirname "$0")/ingestion_pipeline.sh" 2>&1 || true)
  INGESTION_LATENCY=$(echo "$ING_OUT" | grep -oP 'End-to-end latency: \K[0-9]+s' || echo "n/a")
fi

cat > "$OUT" <<EOF
# Proof Points Report

Generated: $(ts)

## Environment
- API: ${API_URL}
- DB: ${DB_HOST}:${DB_PORT}

## Health

${HEALTH_SUMMARY}

## Embedding Health
- Coverage: ${COVER_FMT}%
- Max staleness: ${STALENESS}s

## Search Activity
- Vector searches (cosine): ${VEC_CNT_REQ}
- Hybrid searches (RRF): ${HYB_CNT_REQ}

## Latency Snapshot
- Vector search <= 50ms: ${VEC_FRAC}

## Embedding Coverage by Class (top 5)
${CLASS_TOP5}

## Scheduler Summary (last runs)
$(echo "$SCHED_SUM" | jq -r 'to_entries[] | "- " + .key + ": " + ( .value|tostring )' 2>/dev/null)

## Retrieval Quality (from evaluation/weighted_search_results.json)
$(
  if [[ -f "evaluation/weighted_search_results.json" ]]; then
    # Prefer triple-run format if present, else fallback to baseline/weighted
    if jq -e '.vector and .hybrid and .weighted' evaluation/weighted_search_results.json >/dev/null 2>&1; then
      jq -r '
        . as $r |
        ($r.vector.metrics.recall["recall@10"]) as $rv |
        ($r.hybrid.metrics.recall["recall@10"]) as $rh |
        ($r.weighted.metrics.recall["recall@10"]) as $rw |
        ($r.vector.metrics.mrr) as $mv |
        ($r.hybrid.metrics.mrr) as $mh |
        ($r.weighted.metrics.mrr) as $mw |
        def upl(n;b): if (b!=0) then ((n-b)/b*100.0) else 0 end; 
        "Vector:  recall@10=" + ($rv|tostring) + ", MRR=" + ($mv|tostring) + ", NDCG@10=" + ($r.vector.metrics["ndcg@10"]|tostring) + "\n"
        + "Hybrid:  recall@10=" + ($rh|tostring) + ", MRR=" + ($mh|tostring)
        + ", Uplift (Recall)=" + (upl($rh;$rv)|tostring) + "%"
        + ", Uplift (MRR)=" + (upl($mh;$mv)|tostring) + "%\n"
        + "Weighted: recall@10=" + ($rw|tostring) + ", MRR=" + ($mw|tostring)
        + ", Uplift (Recall)=" + (upl($rw;$rv)|tostring) + "%"
        + ", Uplift (MRR)=" + (upl($mw;$mv)|tostring) + "%"
      ' evaluation/weighted_search_results.json 2>/dev/null || true
      echo
      echo "Retrieval Uplift Summary"
      echo "Mode       | Recall@10 | MRR  | Uplift Recall | Uplift MRR"
      echo "-----------|-----------|------|---------------|-----------"
      jq -r '
        . as $r |
        ($r.vector.metrics.recall["recall@10"]) as $rv |
        ($r.hybrid.metrics.recall["recall@10"]) as $rh |
        ($r.weighted.metrics.recall["recall@10"]) as $rw |
        ($r.vector.metrics.mrr) as $mv |
        ($r.hybrid.metrics.mrr) as $mh |
        ($r.weighted.metrics.mrr) as $mw |
        def upl(n;b): if (b!=0) then ((n-b)/b*100.0) else 0 end; 
        "Hybrid     | " + ($rh|tostring) + "      | " + ($mh|tostring) + " | " + (upl($rh;$rv)|tostring) + "%          | " + (upl($mh;$mv)|tostring) + "%\n"
        + "Weighted   | " + ($rw|tostring) + "      | " + ($mw|tostring) + " | " + (upl($rw;$rv)|tostring) + "%          | " + (upl($mw;$mv)|tostring) + "%"
      ' evaluation/weighted_search_results.json 2>/dev/null || true
    else
      jq -r '
        "Baseline: recall@10=" + (.baseline.recall["recall@10"]|tostring) + ", MRR=" + (.baseline.mrr|tostring) + ", NDCG@10=" + (.baseline["ndcg@10"]|tostring) + "\n"
        + "Weighted: recall@10=" + (.weighted.recall["recall@10"]|tostring) + ", MRR=" + (.weighted.mrr|tostring) + ", NDCG@10=" + (.weighted["ndcg@10"]|tostring)
      ' evaluation/weighted_search_results.json 2>/dev/null || true
    fi
  else
    echo "(No retrieval results file found)"
  fi
)

## Drift Histogram (/_admin/drift_histogram)
$(
  DH=$(curl -sS "${API_URL}/_admin/drift_histogram" "${hdr_auth[@]}" || echo '')
  if [[ -n "$DH" ]]; then
    echo "$DH" | jq -r '.buckets | map("[" + ( .lower|tostring ) + "," + ( .upper|tostring ) + "]: " + ( .count|tostring )) | .[]' 2>/dev/null || true
  else
    echo "(No drift histogram data)"
  fi
)

## Governance (access violations)
$(
  PROM=$(curl -sS "${API_URL}/prometheus" || echo '')
  if [[ -n "$PROM" ]]; then
    echo "$PROM" | awk '/^activekg_access_violations_total/ { \
      if (match($0, /type=\"([^\"]+)\"/, a)) { printf("- %s: %s\n", a[1], $NF) } \
    }' 2>/dev/null || true
  else
    echo "(No governance metrics found)"
  fi
)

## Semantic Triggers
- Unavailable for launch; the compatibility routes return HTTP 410 and no evaluation job is scheduled.

## Proof Metrics (live tests)
EOF

if [[ "${RUN_PROOFS}" == "1" ]]; then
cat >> "$OUT" <<EOF
- DX timing (time to searchable): ${DX_TIMING}
- Ingestion E2E latency: ${INGESTION_LATENCY}
EOF
else
cat >> "$OUT" <<EOF
- Run with RUN_PROOFS=1 to execute live timing tests
EOF
fi

cat >> "$OUT" <<EOF

## Notes
- Values derived from Prometheus histogram/counter series exposed at /prometheus.
- Re-run scripts/live_smoke.sh and scripts/live_extended.sh to generate fresh activity before this report.
- Set RUN_PROOFS=1 to include live DX timing and ingestion latency measurements.

EOF

echo "Wrote $OUT"
