# Google Cloud Storage (GCS) Connector

Complete guide for integrating ActiveKG with Google Cloud Storage to automatically sync and embed documents.

## Overview

The GCS connector monitors Google Cloud Storage buckets for new or updated objects and automatically ingests them into Active Graph KG. It supports:

- ✅ Automatic polling for new/updated files
- ✅ Incremental sync with cursor-based pagination
- ✅ Multi-format support (PDF, DOCX, HTML, TXT)
- ✅ Generation-based versioning
- ✅ Service account & workload identity authentication
- ✅ Idempotent ingestion (no duplicates)

---

## Prerequisites

1. **Google Cloud Project** with Cloud Storage enabled
2. **Service Account** with appropriate permissions
3. **ActiveKG instance** running with database initialized

---

## Setup

### 1. Create Service Account

Create a service account with Storage Object Viewer role:

```bash
# Set project
export PROJECT_ID=your-project-id
gcloud config set project $PROJECT_ID

# Create service account
gcloud iam service-accounts create activekg-connector \
  --display-name="ActiveKG GCS Connector" \
  --description="Service account for ActiveKG GCS connector"

# Grant Storage Object Viewer role
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:activekg-connector@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"

# Grant bucket-specific access (optional, more restrictive)
gsutil iam ch \
  serviceAccount:activekg-connector@${PROJECT_ID}.iam.gserviceaccount.com:objectViewer \
  gs://your-bucket-name
```

### 2. Create Service Account Key

```bash
# Generate key file
gcloud iam service-accounts keys create activekg-gcs-key.json \
  --iam-account=activekg-connector@${PROJECT_ID}.iam.gserviceaccount.com

# Verify key file
cat activekg-gcs-key.json | jq .type  # Should output "service_account"
```

### 3. Configure Connector

Register the GCS connector via the admin API:

```bash
curl -X POST http://localhost:8000/_admin/connectors/gcs/register \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme_corp",
    "config": {
      "bucket": "my-documents-bucket",
      "prefix": "documents/",
      "project": "my-gcp-project",
      "service_account_json_path": "/path/to/activekg-gcs-key.json",
      "poll_interval_seconds": 900,
      "enabled": true
    }
  }'
```

### 4. Test Connection

Test the connector by listing files (dry run):

```bash
curl -X POST http://localhost:8000/_admin/connectors/gcs/backfill \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "acme_corp"}'
```

Expected response:
```json
{
  "status": "ok",
  "found": 37,
  "next_cursor": null
}
```

---

## Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bucket` | string | ✅ | GCS bucket name |
| `prefix` | string | | Prefix filter (e.g., "documents/") |
| `project` | string | | GCP project ID (can use GOOGLE_CLOUD_PROJECT env) |
| `service_account_json_path` | string | ✅ | Path to service account JSON key file |
| `poll_interval_seconds` | int | | Poll frequency (60-3600, default: 900) |
| `enabled` | bool | | Enable/disable connector (default: true) |

---

## Authentication Methods

### 1. Service Account Key File (Recommended for Development)

```json
{
  "config": {
    "service_account_json_path": "/path/to/key.json",
    "project": "my-project"
  }
}
```

### 2. Workload Identity (Recommended for Production)

For GKE/Cloud Run:
```bash
# Don't specify credentials_path - uses default credentials
export GOOGLE_APPLICATION_CREDENTIALS=/var/secrets/gcs-key.json
# Or use workload identity
```

### 3. gcloud Default Credentials (Development Only)

```bash
# Authenticate with user account
gcloud auth application-default login
# Connector will use these credentials automatically
```

---

## Supported File Formats

The GCS connector automatically detects and extracts text from:

| Format | Extension | Content-Type | Extraction Method |
|--------|-----------|--------------|-------------------|
| PDF | `.pdf` | `application/pdf` | pdfplumber |
| Word | `.docx` | `application/vnd.openxmlformats...` | python-docx |
| HTML | `.html` | `text/html` | BeautifulSoup |
| Text | `.txt, .md` | `text/plain` | UTF-8 decode |

---

## How It Works

### 1. Polling Cycle

Every `poll_interval_seconds`:
1. List objects in bucket with prefix filter
2. Check generations/ETags against database
3. Download changed objects
4. Extract text based on content type
5. Create/update nodes with embeddings

### 2. Change Detection

GCS uses **generations** for versioning:

```python
# GCS generation = unique version identifier
if node.props.generation == blob.generation:
    # Skip - no changes
else:
    # Download and re-embed
```

### 3. URI Format

Objects are referenced as:
```
gs://bucket-name/path/to/document.pdf
```

---

## Operations

### Trigger Ingestion (Queue Files for Processing)

Queue files from the GCS bucket for async ingestion. Requires `super_admin` scope.

**Step 1: Preview (dry run, default)**
```bash
curl -X POST http://localhost:8000/_admin/connectors/gcs/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "default",
    "dry_run": true
  }'
```

Response:
```json
{
  "status": "dry_run",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "dry_run": true,
  "would_queue": 42,
  "skipped_count": 8,
  "total_found": 50,
  "queue_key": "connector:gcs:default:queue",
  "next_cursor": null
}
```

**Step 2: Actually queue files**
```bash
curl -X POST http://localhost:8000/_admin/connectors/gcs/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "default",
    "dry_run": false,
    "max_items": 1000,
    "batch_size": 100
  }'
```

Response:
```json
{
  "status": "queued",
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "dry_run": false,
  "queued_count": 42,
  "skipped_count": 8,
  "total_found": 50,
  "queue_key": "connector:gcs:default:queue",
  "next_cursor": null
}
```

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `dry_run` | `true` | Preview only, don't actually queue |
| `max_items` | 1000 | Max total items (1-10000) |
| `batch_size` | 100 | Items per batch (1-500) |
| `skip_existing` | `true` | Skip already queued/processed items |
| `cursor` | null | Pagination cursor for large buckets |

For large buckets, use pagination:
```bash
# Continue with cursor from previous response
curl -X POST http://localhost:8000/_admin/connectors/gcs/ingest \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default", "dry_run": false, "cursor": "eyJwYWdlX3Rva2VuIjogIkNnPT0ifQ=="}'
```

### Monitor Queue Status

Check ingestion progress:

```bash
curl "http://localhost:8000/_admin/connectors/gcs/queue-status?tenant_id=default"
```

Response:
```json
{
  "tenant_id": "default",
  "provider": "gcs",
  "queue_key": "connector:gcs:default:queue",
  "pending": 15
}
```

When `pending` reaches `0`, all files have been processed.

### Start Workers

The ingestion pipeline requires background workers to process queued files:

```bash
# Terminal 1: Start the connector worker (fetches files, extracts text, creates nodes)
python -m activekg.connectors.worker

# Terminal 2: Start the embedding worker (generates embeddings for nodes)
# Required if EMBEDDING_ASYNC=true
python -m activekg.embedding.worker
```

### List Files (Dry Run)

Preview files without queuing them:

```bash
curl -X POST http://localhost:8000/_admin/connectors/gcs/backfill \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default"}'
```

### Enable/Disable Connector

```bash
# Enable
curl -X POST http://localhost:8000/_admin/connectors/gcs/enable \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default"}'

# Disable
curl -X POST http://localhost:8000/_admin/connectors/gcs/disable \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default"}'
```

---

## Monitoring

### Prometheus Metrics

```
# Connector run metrics
connector_run_duration_seconds{tenant="acme_corp", provider="gcs"}
connector_run_items_total{tenant="acme_corp", provider="gcs", status="success"}

# Worker metrics
connector_worker_errors_total{tenant="acme_corp", provider="gcs", error_type="fetch_failed"}
```

### Logs

Check connector logs:

```bash
# Successful ingest
INFO: GCS connector synced 23 objects for tenant acme_corp

# Generation skipped (no changes)
DEBUG: Skipping gs://bucket/doc.pdf - generation unchanged

# Fetch error
ERROR: Failed to fetch gs://bucket/doc.pdf: 403 Forbidden
```

---

## Troubleshooting

### Issue: "403 Forbidden" Error

**Cause:** Service account lacks permissions

**Fix:**
1. Verify service account has `storage.objectViewer` role
2. Check bucket IAM policy
3. Test with `gsutil ls gs://bucket-name` using service account

```bash
# Test service account access
gcloud auth activate-service-account \
  --key-file=activekg-gcs-key.json
gsutil ls gs://your-bucket-name/
```

### Issue: Objects Not Syncing

**Cause:** Prefix filter too restrictive or generation unchanged

**Fix:**
1. Check `prefix` matches object paths
2. Verify objects were actually modified (check GCS console)
3. Check logs for "generation unchanged" messages

### Issue: "Default credentials not found"

**Cause:** Neither service account JSON nor default credentials available

**Fix:**
1. Specify `service_account_json_path` in config, OR
2. Set `GOOGLE_APPLICATION_CREDENTIALS` environment variable, OR
3. Run `gcloud auth application-default login` for development

---

## Best Practices

### 1. Security
- ✅ Use service accounts (not user accounts) for production
- ✅ Grant minimum required permissions (storage.objectViewer only)
- ✅ Rotate service account keys every 90 days
- ✅ Use workload identity on GKE (no key files needed)

### 2. Performance
- ✅ Set appropriate poll intervals (default: 15 minutes)
- ✅ Use prefix filters to limit objects scanned
- ✅ Monitor worker queue depth

### 3. Cost Optimization
- ✅ Enable generation-based skipping (automatic)
- ✅ Use Cloud Storage lifecycle policies for archiving
- ✅ Set poll intervals based on update frequency

---

## GCS-Specific Features

### Object Versioning

GCS tracks object versions via `generation`:

```json
{
  "uri": "gs://bucket/doc.pdf",
  "metadata": {
    "generation": "1637612345678000",
    "etag": "CMa2vPr1xPACEAE=",
    "updated": "2025-11-24T10:15:30Z"
  }
}
```

### Cloud Storage Events (Optional)

For real-time sync, configure Cloud Storage notifications:

```bash
# Create Pub/Sub topic
gcloud pubsub topics create activekg-gcs-events

# Configure bucket notifications
gsutil notification create \
  -t activekg-gcs-events \
  -f json \
  gs://your-bucket-name
```

Then configure ActiveKG webhook endpoint to receive events.

---

## Integration Examples

### Python Client

```python
import requests
import time

ADMIN_API = "http://localhost:8000"

# 1. Register GCS connector
response = requests.post(
    f"{ADMIN_API}/_admin/connectors/gcs/register",
    json={
        "tenant_id": "acme_corp",
        "config": {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "project": "my-project-123",
            "service_account_json_path": "/secrets/gcs-key.json",
            "poll_interval_seconds": 600,
            "enabled": True
        }
    }
)
print(f"Registered: {response.json()}")

# 2. Preview what would be queued (dry run)
response = requests.post(
    f"{ADMIN_API}/_admin/connectors/gcs/ingest",
    json={"tenant_id": "acme_corp", "dry_run": True}
)
preview = response.json()
print(f"Would queue: {preview['would_queue']}, skip: {preview['skipped_count']}")

# 3. Actually queue files
response = requests.post(
    f"{ADMIN_API}/_admin/connectors/gcs/ingest",
    json={"tenant_id": "acme_corp", "dry_run": False, "max_items": 500}
)
result = response.json()
print(f"Job {result['job_id']}: queued {result['queued_count']}, skipped {result['skipped_count']}")

# 4. Monitor progress
while True:
    status = requests.get(
        f"{ADMIN_API}/_admin/connectors/gcs/queue-status",
        params={"tenant_id": "acme_corp"}
    ).json()
    print(f"Pending: {status['pending']}")
    if status['pending'] == 0:
        break
    time.sleep(5)

print("Ingestion complete!")
```

### Terraform Configuration

```hcl
# Create service account
resource "google_service_account" "activekg_connector" {
  account_id   = "activekg-connector"
  display_name = "ActiveKG GCS Connector"
  project      = var.project_id
}

# Grant storage viewer role
resource "google_project_iam_member" "activekg_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.activekg_connector.email}"
}

# Create key
resource "google_service_account_key" "activekg_key" {
  service_account_id = google_service_account.activekg_connector.name
}
```

---

## Limitations

- **File Size:** Max 100MB per object (configurable via `ACTIVEKG_MAX_FILE_BYTES`)
- **Formats:** Only text-extractable formats supported
- **Polling:** Not real-time (use Cloud Storage events for real-time)
- **Versioning:** Tracks latest version only (not full version history)

---

## See Also

- [Connector Operations Guide](operations/connectors.md) - Idempotency, cursors, key rotation
- [S3 Connector](S3_CONNECTOR.md) - AWS S3 integration
- [Drive Connector](DRIVE_CONNECTOR.md) - Google Drive integration
- [GCS Staging Rollout](operations/GCS_STAGING_ROLLOUT.md) - Production deployment guide
- [API Reference](api-reference.md) - Admin connector endpoints

---

**Status:** ✅ Production Ready
**Last Updated:** 2025-11-24
