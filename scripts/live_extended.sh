#!/usr/bin/env bash
set -euo pipefail

# Extended live validation suite for Active Graph KG
# Covers: CRUD, drift detection, lineage/edges, search (vector+hybrid), ask streaming,
# ANN admin, optional cross-tenant RLS if SECOND_TOKEN is provided.

API_URL=${API_URL:-${API:-http://localhost:8000}}
TOKEN=${TOKEN:-${E2E_ADMIN_TOKEN:-}}
SECOND_TOKEN=${SECOND_TOKEN:-}

if [[ -z "${TOKEN}" ]]; then
  echo "ERROR: TOKEN env var not set. Export a single-line JWT into TOKEN." >&2
  exit 1
fi

HDR=( -H "Authorization: Bearer ${TOKEN}" )
json() { jq -r "$1" 2>/dev/null || true; }

echo "== Health =="
curl -sS "${API_URL}/health" | jq . || true

echo "== Ensure ANN Indexes =="
curl -sS -X POST "${API_URL}/admin/indexes" "${HDR[@]}" -H 'Content-Type: application/json' -d '{"action":"ensure"}' | jq . || true

echo "== Create Parent Node A =="
RES_A=$(curl -sS -X POST "${API_URL}/nodes" "${HDR[@]}" -H 'Content-Type: application/json' -d '{"classes":["TestDoc"],"props":{"title":"Parent A","text":"Parent content for lineage"}}')
ID_A=$(echo "$RES_A" | json .id)
echo "A=${ID_A}"

echo "== Create Child Node B (DERIVED_FROM A) =="
B_TEXT="Child derived content for lineage"
RES_B=$(curl -sS -X POST "${API_URL}/nodes" "${HDR[@]}" -H 'Content-Type: application/json' -d "{\"classes\":[\"TestDoc\"],\"props\":{\"title\":\"Child B\",\"text\":\"$B_TEXT\"}}")
ID_B=$(echo "$RES_B" | json .id)
echo "B=${ID_B}"

echo "== Create Edge B -[DERIVED_FROM]-> A =="
curl -sS -X POST "${API_URL}/edges" "${HDR[@]}" -H 'Content-Type: application/json' -d "{\"src\":\"$ID_B\",\"rel\":\"DERIVED_FROM\",\"dst\":\"$ID_A\",\"props\":{}}" | jq .

echo "== Lineage for B =="
curl -sS "${API_URL}/lineage/${ID_B}" "${HDR[@]}" | jq .

echo "== Drift Test: Create Node C, refresh, change text, refresh again =="
RES_C=$(curl -sS -X POST "${API_URL}/nodes" "${HDR[@]}" -H 'Content-Type: application/json' -d '{"classes":["TestDoc"],"props":{"title":"Drift C","text":"alpha beta gamma"},"refresh_policy":{"drift_threshold":0.1}}')
ID_C=$(echo "$RES_C" | json .id)
echo "C=${ID_C}"
curl -sS -X POST "${API_URL}/nodes/${ID_C}/refresh" "${HDR[@]}" | jq .

# Change text significantly to induce high drift
NEW_TEXT="zzzz brand new unrelated content with totally different vocabulary $RANDOM"
curl -sS -X PUT "${API_URL}/nodes/${ID_C}" "${HDR[@]}" -H 'Content-Type: application/json' -d "{\"props\":{\"title\":\"Drift C\",\"text\":\"$NEW_TEXT\"}}" | jq .

REF2=$(curl -sS -X POST "${API_URL}/nodes/${ID_C}/refresh" "${HDR[@]}")
echo "$REF2" | jq .
DRIFT=$(echo "$REF2" | json .drift_score)
echo "Observed drift: ${DRIFT}"

echo "== Events for C (refreshed) =="
curl -sS "${API_URL}/events?node_id=${ID_C}&event_type=refreshed&limit=10" "${HDR[@]}" | jq .

echo "== Search (vector & hybrid) =="
curl -sS -X POST "${API_URL}/search" "${HDR[@]}" -H 'Content-Type: application/json' -d '{"query":"lineage content","top_k":5,"use_hybrid":false}' | jq '.results | length'
curl -sS -X POST "${API_URL}/search" "${HDR[@]}" -H 'Content-Type: application/json' -d '{"query":"lineage content","top_k":5,"use_hybrid":true}' | jq '.results | length'

echo "== Ask stream (short run) =="
curl -N -m 20 -X POST "${API_URL}/ask/stream" "${HDR[@]}" -H 'Accept: text/event-stream' -H 'Content-Type: application/json' --data '{"question":"Summarize the lineage of B","stream":true}' | head -n 10 || true

if [[ -n "${SECOND_TOKEN}" ]]; then
  echo "== Optional RLS test with SECOND_TOKEN =="
  HDR2=( -H "Authorization: Bearer ${SECOND_TOKEN}" )
  # Node B should be 404 for another tenant
  code=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/nodes/${ID_B}" "${HDR2[@]}")
  echo "GET /nodes/${ID_B} with SECOND_TOKEN -> ${code} (expect 404)"
fi

echo "== Cleanup test nodes (hard delete) =="
for id in "$ID_B" "$ID_A" "$ID_C"; do
  curl -sS -X DELETE "${API_URL}/nodes/${id}?hard=true" "${HDR[@]}" | jq . || true
done

echo "✓ Extended live suite complete"
