#!/usr/bin/env bash
set -euo pipefail

# One-click governance demo:
# - Triggers missing_token, scope_denied (optional), cross_tenant_query signals
# - Rebuilds the proof report to include governance section

API_URL=${API_URL:-${API:-http://localhost:8000}}
TOKEN=${TOKEN:-${E2E_ADMIN_TOKEN:-}}
SECOND_TOKEN=${SECOND_TOKEN:-}
TOKEN_NO_ADMIN=${TOKEN_NO_ADMIN:-}
CONTROL_PLANE_TOKEN=${ACTIVEKG_CONTROL_PLANE_TOKEN:-}

if [[ -z "${CONTROL_PLANE_TOKEN}" ]]; then
  echo "ERROR: ACTIVEKG_CONTROL_PLANE_TOKEN must be set before metrics access." >&2
  exit 1
fi

echo "== Governance Demo =="

echo "-- Missing token (/_admin/metrics_summary)"
curl -s -o /dev/null -w "HTTP %{http_code}\n" "$API_URL/_admin/metrics_summary" || true

if [[ -n "$TOKEN_NO_ADMIN" ]]; then
  echo "-- Scope denied using TOKEN_NO_ADMIN (/_admin/metrics_summary)"
  curl -s -o /dev/null -w "HTTP %{http_code}\n" \
    -H "Authorization: Bearer $TOKEN_NO_ADMIN" \
    "$API_URL/_admin/metrics_summary" || true
else
  echo "(skip) TOKEN_NO_ADMIN not set; cannot demo scope_denied"
fi

if [[ -n "$TOKEN" ]]; then
  echo "-- Cross-tenant query param with TOKEN (GET /nodes?tenant_id=evil)"
  curl -s -o /dev/null -w "HTTP %{http_code}\n" \
    -H "Authorization: Bearer $TOKEN" \
    "$API_URL/nodes?tenant_id=evil&limit=1" || true
else
  echo "(skip) TOKEN not set; cannot demo cross_tenant_query"
fi

echo "-- Scrape governance metrics"
curl -s -H "Authorization: Bearer ${CONTROL_PLANE_TOKEN}" \
  "$API_URL/prometheus" | grep '^activekg_access_violations_total' || true

echo "-- Rebuilding proof report"
API=$API_URL TOKEN=$TOKEN ACTIVEKG_CONTROL_PLANE_TOKEN=$CONTROL_PLANE_TOKEN \
  bash scripts/proof_points_report.sh || true
echo "✓ Governance demo complete. See evaluation/PROOF_POINTS_REPORT.md"
