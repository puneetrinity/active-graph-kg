# Google Drive Connector

> **DORMANT FUTURE DESIGN — NOT AVAILABLE.** Do not follow the commands in this document. The API accepts no Drive
> configuration, runs no Drive poller/worker and returns HTTP 410 on every connector compatibility route.

Complete guide for integrating ActiveKG with Google Drive to automatically sync and embed documents from My Drive and Shared Drives.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Service Account Setup](#service-account-setup)
- [Configuration Reference](#configuration-reference)
- [API Endpoints](#api-endpoints)
- [Operational Guide](#operational-guide)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## Overview

The Drive connector provides:
- Incremental sync using the Drive Changes API
- Support for My Drive and Shared Drives
- Google Workspace export (Docs→HTML, Sheets→CSV, Slides→text)
- Folder-based filtering and MIME type filtering
- Automatic deduplication using ETags
- Cursor-based resumable sync

## Quick Start

### 1. Create Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing project
3. Enable Google Drive API
4. Create Service Account:
   - Navigate to IAM & Admin → Service Accounts
   - Create Service Account
   - Download JSON key file

### 2. Configure Drive Access

**Option A: Domain-Wide Delegation (recommended for org-wide access)**
1. In GCP Console → Service Account → Edit → Enable domain-wide delegation
2. In Google Workspace Admin Console → Security → API Controls → Domain-wide Delegation
3. Add Client ID with scope: `https://www.googleapis.com/auth/drive.readonly`
4. Set `subject_email` in connector config to impersonate user

**Option B: Direct Sharing (per-folder access)**
1. Share specific folders/files with service account email (xxx@xxx.iam.gserviceaccount.com)
2. Grant "Viewer" or "Reader" permissions
3. Service account can only access explicitly shared content

### 3. Register Connector

```bash
curl -X POST http://localhost:8000/_admin/connectors/drive/register \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "default",
    "credentials": "<SERVICE_ACCOUNT_JSON>",
    "root_folders": ["<FOLDER_ID_1>", "<FOLDER_ID_2>"],
    "poll_interval_seconds": 300,
    "enabled": true
  }'
```

### 4. Initial Backfill

```bash
curl -X POST http://localhost:8000/_admin/connectors/drive/backfill \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "default"}'
```

The backfill will discover all files and queue them for processing. The scheduler (60s poll) will then incrementally sync changes.

## Service Account Setup

### Required OAuth Scopes

```
https://www.googleapis.com/auth/drive.readonly
```

### Obtaining Folder IDs

Folder ID is the last part of the Google Drive URL:
```
https://drive.google.com/drive/folders/<FOLDER_ID>
```

### Shared Drives

To access Shared Drives:
1. Grant service account access to Shared Drive (Reader role)
2. Add Shared Drive IDs to `shared_drives` config field
3. Use `supportsAllDrives=true` in Drive API calls (handled automatically)

**Note:** Current v1 implementation uses a single global cursor. For complete Shared Drive coverage, consider maintaining per-drive cursors (see [Known Limitations](#known-limitations)).

## Configuration Reference

### DriveConnectorConfig Schema

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `credentials` | string | Yes | - | Service account JSON (will be encrypted at rest) |
| `subject_email` | string | No | null | Email to impersonate (requires domain-wide delegation) |
| `project` | string | No | null | GCP project ID |
| `shared_drives` | list[string] | No | [] | Shared Drive IDs to sync |
| `root_folders` | list[string] | No | [] | My Drive root folder IDs to sync |
| `include_folders` | list[string] | No | [] | Folder ID allowlist (empty = all folders) |
| `exclude_folders` | list[string] | No | [] | Folder ID blocklist |
| `include_mime_types` | list[string] | No | [] | MIME type allowlist (empty = all types) |
| `export_formats` | dict | No | See below | Google Workspace export mappings |
| `poll_interval_seconds` | int | No | 300 | Polling frequency (60-3600s) |
| `page_size` | int | No | 100 | Drive API page size (1-1000) |
| `use_changes_feed` | bool | No | true | Enable Changes API incremental sync |
| `max_file_size_bytes` | int | No | 100MB | Skip files larger than this |
| `webhook_secret` | string | No | null | Reserved for future webhook mode |
| `enabled` | bool | No | true | Enable/disable connector |

### Default Export Formats

```json
{
  "application/vnd.google-apps.document": "text/html",
  "application/vnd.google-apps.spreadsheet": "text/csv",
  "application/vnd.google-apps.presentation": "text/plain"
}
```

### Example Configurations

**Minimal (single folder):**
```json
{
  "tenant_id": "default",
  "credentials": "{...service_account_json...}",
  "root_folders": ["1A2B3C4D5E6F"],
  "enabled": true
}
```

**Full (domain-wide with filtering):**
```json
{
  "tenant_id": "acme-corp",
  "credentials": "{...}",
  "subject_email": "admin@acme.com",
  "shared_drives": ["0AxByCzD"],
  "root_folders": ["1A2B3C", "4D5E6F"],
  "exclude_folders": ["7G8H9I"],
  "include_mime_types": [
    "application/pdf",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet"
  ],
  "poll_interval_seconds": 300,
  "max_file_size_bytes": 52428800,
  "enabled": true
}
```

## API Endpoints

### POST /_admin/connectors/drive/register

Register or update Drive connector configuration.

**Request:**
```json
{
  "tenant_id": "string",
  "credentials": "string (service account JSON)",
  "subject_email": "string (optional)",
  "shared_drives": ["string"],
  "root_folders": ["string"],
  "include_folders": ["string"],
  "exclude_folders": ["string"],
  "include_mime_types": ["string"],
  "poll_interval_seconds": 300,
  "enabled": true
}
```

**Response:**
```json
{
  "status": "registered",
  "tenant_id": "default",
  "provider": "drive",
  "poll_interval_seconds": 300
}
```

### POST /_admin/connectors/drive/backfill

Trigger initial backfill (full sync) or re-sync all files.

**Request:**
```json
{
  "tenant_id": "string"
}
```

**Response:**
```json
{
  "status": "queued",
  "tenant_id": "default",
  "files_discovered": 142,
  "queue_name": "connector:drive:default:queue"
}
```

## Operational Guide

### Polling Behavior

1. **Scheduler runs every 60 seconds** (configurable in scheduler.py)
2. **Per-tenant Redis NX lock** prevents concurrent polls
3. **Lock TTL = poll_interval_seconds** (default 300s)
4. Each poll:
   - Fetches Drive Changes API with stored cursor
   - Filters files by folder/MIME type rules
   - Pushes changed files to Redis queue (connector:drive:{tenant}:queue)
   - Persists new cursor to connector_cursors table
   - Worker processes queue asynchronously

### Cursor Management

Cursors are stored in the `connector_cursors` table:

```sql
CREATE TABLE connector_cursors (
    tenant_id TEXT NOT NULL,
    provider TEXT NOT NULL,  -- 'drive'
    cursor TEXT NOT NULL,     -- Drive pageToken
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tenant_id, provider)
);
```

**Operations:**
- `get_cursor(tenant_id, 'drive')` → Returns pageToken or None (initial sync)
- `set_cursor(tenant_id, 'drive', new_token)` → Updates cursor after successful poll

**Cursor lifecycle:**
1. **No cursor (initial)** → Calls `changes().getStartPageToken()` to get baseline
2. **Has cursor** → Calls `changes().list(pageToken=cursor)` for incremental changes
3. **After each poll** → Stores `newStartPageToken` from response

### ETag Handling

- **Binary files (PDF, DOCX, etc.):** Use `md5Checksum` as ETag
- **Google Workspace docs:** Use `modifiedTime:version` as ETag (md5Checksum is empty)

ETags enable deduplication: identical ETag → skip processing.

### File Processing Flow

```
Drive Changes API → Poller (scheduler.py)
  → Redis Queue (connector:drive:{tenant}:queue)
  → Worker (worker.py)
  → Document Parser
  → Embedding Model
  → PostgreSQL (chunks table)
```

### Quota Limits

**Google Drive API Quotas (per project):**
- 20,000 queries per 100 seconds
- 1 billion queries per day

**Connector quotas (configurable):**
- max_docs_per_day: 10,000 (default)
- max_storage_bytes: 10GB (default)
- max_api_calls_per_hour: 5,000 (default)

**Rate limiting:**
- Implemented via Redis-based rate limiting
- Prometheus metrics: connector_api_calls_total, connector_quota_exceeded_total

## Known Limitations

### 1. Shared Drives Cursor Scope (v1)

**Issue:** Single global Changes API cursor may not capture all Shared Drive changes.

**Workaround (v2):**
- Enumerate `driveIds` using `drives().list()`
- Maintain cursor per drive: `connector_cursors` with `scope` column (e.g., `drive:{driveId}`)
- Poll each drive separately

**Current behavior:** Best-effort coverage of "My Drive + shared items". May miss some Shared Drive changes.

### 2. Folder Filtering Ancestry

**Issue:** `_should_include_file()` only checks direct parents, missing nested items under `root_folders`.

**Workaround (v2):**
- Precompute allowed folder ancestry during initial backfill
- Build folder tree: `allowed_folders = set(root_folders) ∪ {descendants}`
- Check: `any(parent in allowed_folders for parent in file_info["parents"])`

**Current behavior:** Only files with direct parent in `root_folders` are included.

### 3. ETag Version Changes

**Issue:** Google Workspace doc `version` field may increment without content changes (e.g., permission changes).

**Impact:** May trigger unnecessary re-processing.

**Mitigation:** Monitor `connector_duplicate_skipped_total` metric for skip rate.

## Troubleshooting

### Authentication Errors

**Error:** `401 Unauthorized` or `403 Forbidden`

**Solutions:**
1. Verify service account JSON is valid (not truncated)
2. Check Drive API is enabled in GCP project
3. For domain-wide delegation: verify Client ID and scopes in Workspace Admin Console
4. For direct sharing: verify service account email has access to folders

### No Files Discovered

**Checklist:**
1. Verify folder IDs are correct (check Drive URLs)
2. Check service account has access (try opening folder with service account email)
3. Review `include_folders` / `exclude_folders` / `include_mime_types` filters
4. Check connector is enabled: `enabled: true`
5. Inspect logs for filter warnings: `grep "Skipping file" /var/log/activekg.log`

### Cursor Not Advancing

**Symptoms:** Repeated processing of same files

**Solutions:**
1. Check `connector_cursors` table: `SELECT * FROM connector_cursors WHERE provider='drive';`
2. Verify cursor is being updated: watch `updated_at` timestamp
3. Check for errors in scheduler logs
4. Ensure Redis NX lock is releasing properly

### Files Not Embedding

**Debug steps:**
1. Verify files are in Redis queue: `redis-cli LLEN connector:drive:default:queue`
2. Check worker is running and consuming queue
3. Inspect worker logs for parsing errors
4. Verify file size < max_file_size_bytes
5. Check MIME type is supported by document parser

### Performance Tuning

**Slow sync:**
- Increase `page_size` (max 1000) for faster API calls
- Reduce `poll_interval_seconds` for more frequent checks (min 60s)
- Scale workers horizontally to process queue faster

**High API usage:**
- Increase `poll_interval_seconds` to reduce API calls
- Add `include_mime_types` filter to skip unwanted files
- Set `max_file_size_bytes` lower to skip large files

## Monitoring

### Prometheus Metrics

```
connector_api_calls_total{provider="drive", tenant="default"}
connector_files_processed_total{provider="drive", tenant="default"}
connector_decrypt_failures_total{field="credentials"}
connector_duplicate_skipped_total{provider="drive"}
```

### Health Checks

```bash
# Check connector registration
curl http://localhost:8000/_admin/connectors | jq '.[] | select(.provider=="drive")'

# Check cursor status
psql -c "SELECT * FROM connector_cursors WHERE provider='drive';"

# Check queue length
redis-cli LLEN connector:drive:default:queue

# Check recent file changes
curl http://localhost:8000/search?query="source:drive" | jq '.results[].uri'
```

## Additional Resources

- [Google Drive API Documentation](https://developers.google.com/drive/api/v3/reference)
- [Service Account Authentication](https://developers.google.com/identity/protocols/oauth2/service-account)
- [Domain-Wide Delegation](https://support.google.com/a/answer/162106)
- [Drive API Quotas](https://developers.google.com/drive/api/guides/limits)

---

**Support:** For issues or questions, open an issue on GitHub or contact support.
