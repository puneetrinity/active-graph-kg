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

echo "== Connector poller errors (if any) exposed in Prometheus =="

echo "(For full chaos tests, add guarded /_admin/simulate_failure endpoints)"
echo "✓ Failure recovery placeholder probe complete"
