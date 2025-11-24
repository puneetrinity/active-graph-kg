# Drive Connector E2E Validation (Staging)

Use this checklist to validate the Drive poller → Redis → worker → DB path in a staging environment.

## Prerequisites
- Staging Postgres and Redis reachable by the API/worker.
- Google service account JSON with access to a staging Drive folder (shared with the SA).
- Environment configured for Drive connector (service account path, folder ID).
- Worker and poller running (or ability to trigger manually).

## Steps
1) Create Drive connector config
```bash
curl -X POST http://$HOST/_admin/connectors/configs \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "staging-tenant",
    "provider": "drive",
    "config": {
      "service_account_json_path": "/path/to/key.json",
      "folder_id": "staging-folder-id",
      "poll_interval_seconds": 300,
      "enabled": true
    }
  }'
```

2) Seed a file in the Drive folder
- Upload a small test file (PDF or DOCX) to the configured folder.
- Note the file ID for verification.

3) Trigger poller (optional manual kick)
- Wait for the scheduled poller (default 60s) **or** trigger a manual sync if exposed:
```bash
curl -X POST http://$HOST/_admin/connectors/configs/{config_id}/sync \
  -H "Authorization: Bearer $ADMIN_JWT"
```

4) Check Redis queue
```bash
redis-cli LLEN "connector:drive:staging-tenant:queue"
redis-cli --raw LINDEX "connector:drive:staging-tenant:queue" -1 | jq
```
- Expect at least one enqueued change item with the Drive file URI and modified_at.

5) Run worker
- Ensure the worker is running and scanning drive queues.
- Confirm it processes the queued item (logs/metrics).

6) Verify ingestion in DB
```bash
psql $ACTIVEKG_DSN -c "SELECT id, props->>'title', tenant_id FROM nodes WHERE tenant_id='staging-tenant' ORDER BY created_at DESC LIMIT 5;"
```
- Expect a node for the uploaded file with Drive metadata/ETag.

7) Metrics/Logs
- Check poller metrics: `connector_poller_runs_total{provider="drive",tenant="staging-tenant"}`
- Check worker metrics: `connector_worker_processed_total{provider="drive",tenant="staging-tenant"}`
- Verify no errors in poller/worker logs.

## Pass/Fail
- PASS: File enqueued from Drive, worker ingests it, node appears in DB, no errors in logs/metrics.
- FAIL: Capture logs/Redis queue contents and investigate connector config, service account access, or poller/worker errors.
