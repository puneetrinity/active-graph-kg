#!/bin/bash
# Test the three fixes

TENANT_A_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyX2EiLCJ0ZW5hbnRfaWQiOiJ0ZW5hbnRfYSIsImFjdG9yX3R5cGUiOiJ1c2VyIiwic2NvcGVzIjpbInNlYXJjaDpyZWFkIiwibm9kZXM6d3JpdGUiXSwiYXVkIjoiYWN0aXZla2ciLCJpc3MiOiJodHRwczovL3N0YWdpbmctYXV0aC55b3VyY29tcGFueS5jb20iLCJpYXQiOjE3NjI1MjA1MDksIm5iZiI6MTc2MjUyMDUwOSwiZXhwIjoxNzYyNjA2OTA5fQ.Ak8cyiAKYxYFcfH-qK-z6zDz5CSAb5-m0ZVJTnBl0Ps"
ADMIN_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbl91c2VyIiwidGVuYW50X2lkIjoidGVuYW50X2EiLCJhY3Rvcl90eXBlIjoidXNlciIsInNjb3BlcyI6WyJhZG1pbjpyZWZyZXNoIiwic2VhcmNoOnJlYWQiLCJub2Rlczp3cml0ZSJdLCJhdWQiOiJhY3RpdmVrZyIsImlzcyI6Imh0dHBzOi8vc3RhZ2luZy1hdXRoLnlvdXJjb21wYW55LmNvbSIsImlhdCI6MTc2MjUyMDUyNywibmJmIjoxNzYyNTIwNTI3LCJleHAiOjE3NjI2MDY5Mjd9.KBcC9e_HeKjnJAkYCFcRs2Yj0nerkv9_CTUObbZ95qo"

echo "======================================"
echo "Testing Three Fixes"
echo "======================================"
echo ""

# Test 1: Admin refresh with both formats
echo "[Test 1] Admin refresh with raw array"
curl -s -w "\nStatus: %{http_code}\n" -X POST http://localhost:8000/admin/refresh \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '[]'
echo ""

echo "[Test 2] Admin refresh with wrapped object"
curl -s -w "\nStatus: %{http_code}\n" -X POST http://localhost:8000/admin/refresh \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"node_ids":[]}'
echo ""

# Test 3: Manual refresh with RLS
echo "[Test 3] Create node and test manual refresh"
NODE_RESPONSE=$(curl -s -X POST http://localhost:8000/nodes \
  -H "Authorization: Bearer $TENANT_A_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"classes":["TestNode"],"props":{"name":"refresh_test"}}')
NODE_ID=$(echo "$NODE_RESPONSE" | jq -r '.id')
echo "Created node: $NODE_ID"

# Wait for background embedding
sleep 5

echo "Testing manual refresh..."
curl -s -w "\nStatus: %{http_code}\n" -X POST "http://localhost:8000/nodes/$NODE_ID/refresh" \
  -H "Authorization: Bearer $TENANT_A_TOKEN"

echo ""
echo "======================================"
echo "All tests complete"
echo "======================================"
