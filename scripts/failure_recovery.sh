#!/usr/bin/env bash
set -euo pipefail

# Placeholder failure recovery probes (non-destructive)

API_URL=${API_URL:-${API:-http://localhost:8000}}
TOKEN=${TOKEN:-${E2E_ADMIN_TOKEN:-}}

if [[ -z "${TOKEN}" ]]; then
  echo "ERROR: TOKEN env var not set (admin JWT)." >&2
  exit 1
fi

HDR=( -H "Authorization: Bearer ${TOKEN}" )

echo "== LLM disabled fallback (expect 503 when LLM_ENABLED=false) =="
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "${API_URL}/ask/stream" \
  "${HDR[@]}" -H 'Accept: text/event-stream' -H 'Content-Type: application/json' \
  --data '{"question":"resilience test","stream":true}')
echo "POST /ask/stream -> ${code}"

echo "== Connector poller errors (if any) exposed in Prometheus =="

echo "(For full chaos tests, add guarded /_admin/simulate_failure endpoints)"
echo "✓ Failure recovery placeholder probe complete"
