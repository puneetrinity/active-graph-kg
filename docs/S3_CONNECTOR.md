# AWS S3 Connector

> **DORMANT FUTURE DESIGN — NOT AVAILABLE.** Do not follow the commands in this document. The API accepts no S3
> configuration or webhook and returns HTTP 410 on every connector compatibility route.

Complete guide for integrating ActiveKG with AWS S3 to automatically sync and embed documents.

## Overview

The S3 connector monitors S3 buckets for new or updated objects and automatically ingests them into Active Graph KG. It supports:

- ✅ Automatic polling for new/updated files
- ✅ Incremental sync with cursor-based pagination
- ✅ Multi-format support (PDF, DOCX, HTML, TXT)
- ✅ ETag-based change detection
- ✅ Idempotent ingestion (no duplicates)
- ✅ Retry and DLQ support

---

## Prerequisites

1. **AWS Account** with S3 access
2. **IAM User** with appropriate permissions
3. **ActiveKG instance** running with database initialized

---

## Setup

### 1. Create IAM User

Create an IAM user with the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:GetObjectVersion"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name",
        "arn:aws:s3:::your-bucket-name/*"
      ]
    }
  ]
}
```

**Generate Access Keys:**
- Go to IAM → Users → Security Credentials
- Create access key
- Save `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

### 2. Configure Connector

Register the S3 connector via the admin API:

```bash
curl -X POST http://localhost:8000/_admin/connectors/configs \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "acme_corp",
    "provider": "s3",
    "config": {
      "bucket": "my-documents-bucket",
      "prefix": "documents/",
      "region": "us-east-1",
      "access_key_id": "AKIAIOSFODNN7EXAMPLE",
      "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
      "poll_interval_seconds": 900,
      "enabled": true
    }
  }'
```

### 3. Test Connection

Test the connector configuration:

```bash
curl -X POST http://localhost:8000/_admin/connectors/configs/{config_id}/test \
  -H "Authorization: Bearer $ADMIN_JWT"
```

Expected response:
```json
{
  "status": "success",
  "message": "Successfully connected to S3 bucket",
  "objects_found": 42
}
```

---

## Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `bucket` | string | ✅ | S3 bucket name |
| `prefix` | string | | Prefix filter (e.g., "documents/") |
| `region` | string | | AWS region (default: us-east-1) |
| `access_key_id` | string | ✅ | AWS access key (min 16 chars) |
| `secret_access_key` | string | ✅ | AWS secret key (min 32 chars) |
| `poll_interval_seconds` | int | | Poll frequency (60-3600, default: 900) |
| `enabled` | bool | | Enable/disable connector (default: true) |

---

## Supported File Formats

The S3 connector automatically detects and extracts text from:

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
2. Check ETags against database
3. Download changed objects
4. Extract text based on content type
5. Create/update nodes with embeddings

### 2. Change Detection

The connector uses **ETags** for efficient change detection:

```python
# S3 ETag = MD5 hash of object
if node.props.etag == s3_object.ETag:
    # Skip - no changes
else:
    # Download and re-embed
```

### 3. URI Format

Objects are referenced as:
```
s3://bucket-name/path/to/document.pdf
```

---

## Operations

### Trigger Manual Sync

Force an immediate sync:

```bash
curl -X POST http://localhost:8000/_admin/connectors/configs/{config_id}/sync \
  -H "Authorization: Bearer $ADMIN_JWT"
```

### List Connector Runs

View sync history:

```bash
curl http://localhost:8000/_admin/connectors/runs?config_id={config_id} \
  -H "Authorization: Bearer $ADMIN_JWT"
```

### Update Configuration

Update connector settings:

```bash
curl -X PUT http://localhost:8000/_admin/connectors/configs/{config_id} \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "poll_interval_seconds": 600,
      "enabled": true
    }
  }'
```

---

## Monitoring

### Prometheus Metrics

```
# Connector run metrics
connector_run_duration_seconds{tenant="acme_corp", provider="s3"}
connector_run_items_total{tenant="acme_corp", provider="s3", status="success"}

# Worker metrics
connector_worker_errors_total{tenant="acme_corp", provider="s3", error_type="fetch_failed"}
```

### Logs

Check connector logs:

```bash
# Successful ingest
INFO: S3 connector synced 15 objects for tenant acme_corp

# ETag skipped (no changes)
DEBUG: Skipping s3://bucket/doc.pdf - ETag unchanged

# Fetch error
ERROR: Failed to fetch s3://bucket/doc.pdf: NoSuchKey
```

---

## Troubleshooting

### Issue: "Access Denied" Error

**Cause:** IAM permissions insufficient

**Fix:**
1. Verify IAM policy includes `s3:GetObject` and `s3:ListBucket`
2. Check bucket policy doesn't deny access
3. Verify credentials are correct

### Issue: Objects Not Syncing

**Cause:** Prefix filter too restrictive or ETag unchanged

**Fix:**
1. Check `prefix` matches object paths
2. Verify objects were actually modified (check S3 console)
3. Check logs for "ETag unchanged" messages

### Issue: "Invalid Credentials"

**Cause:** Access keys expired or incorrect

**Fix:**
1. Regenerate access keys in IAM console
2. Update connector config with new keys
3. Test connection endpoint

---

## Best Practices

### 1. Security
- ✅ Use dedicated IAM user per tenant
- ✅ Store credentials encrypted (handled by ActiveKG)
- ✅ Rotate access keys every 90 days
- ✅ Use least-privilege IAM policies

### 2. Performance
- ✅ Set appropriate poll intervals (don't poll too frequently)
- ✅ Use prefix filters to limit objects scanned
- ✅ Monitor worker queue depth

### 3. Cost Optimization
- ✅ Enable ETag-based skipping (automatic)
- ✅ Use S3 Intelligent-Tiering for infrequently accessed objects
- ✅ Set poll intervals based on update frequency

---

## Integration Examples

### Python Client

```python
import requests

ADMIN_API = "http://localhost:8000"
ADMIN_JWT = "your-admin-token"

# Register S3 connector
response = requests.post(
    f"{ADMIN_API}/_admin/connectors/configs",
    headers={"Authorization": f"Bearer {ADMIN_JWT}"},
    json={
        "tenant_id": "acme_corp",
        "provider": "s3",
        "config": {
            "bucket": "my-bucket",
            "prefix": "docs/",
            "region": "us-west-2",
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "poll_interval_seconds": 600,
            "enabled": True
        }
    }
)

config_id = response.json()["id"]
print(f"Connector created: {config_id}")
```

---

## Limitations

- **File Size:** Max 100MB per object (configurable via `ACTIVEKG_MAX_FILE_BYTES`)
- **Formats:** Only text-extractable formats supported
- **Polling:** Not real-time (use S3 events + webhooks for real-time)
- **Versioning:** Tracks latest version only (not full version history)

---

## See Also

- [Connector Operations Guide](operations/connectors.md) - Idempotency, cursors, key rotation
- [GCS Connector](GCS_CONNECTOR.md) - Google Cloud Storage integration
- [Drive Connector](DRIVE_CONNECTOR.md) - Google Drive integration
- [API Reference](api-reference.md) - Admin connector endpoints

---

**Status:** Dormant future design; unavailable.
**Last Updated:** 2025-11-24
