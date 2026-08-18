# Connectors – Idempotency, Cursors, and Rotation

> **DORMANT FUTURE DESIGN — NOT AN OPERATIONS RUNBOOK.** Connectors are unavailable; do not execute the procedures
> below. The API accepts no connector configuration/credential and all compatibility routes return HTTP 410.

This document explains how connector ingest achieves idempotency and how to operate cursors and key rotation safely.

## Idempotency Keys

Idempotency is enforced using multiple signals in the ingestion flow to avoid reprocessing unchanged content:

- Primary: (tenant_id, provider, uri, etag)
- Fallback: (tenant_id, provider, uri, content_hash)

Workflow (see activekg/connectors/ingest.py):
- If an existing node’s props.etag matches the incoming ETag, the change is skipped (no content fetch).
- If no ETag or mismatch, compute content hash and compare with existing props.content_hash. If equal, update metadata only and skip re-embed.

This prevents duplicate upserts when providers emit unchanged updates or when polling replays a window.

## Cursors

The `connector_cursors` table stores per-tenant cursors for incremental sync (e.g., Drive Changes API pageToken).
- Functions: `get_cursor(tenant_id, provider)`, `set_cursor(tenant_id, provider, cursor)`
- Provider type is constrained via ConnectorProvider Literal.

## Key Rotation

The config store supports key rotation with batch processing and metrics:
- Endpoint: `/_admin/connectors/rotate_keys`
- Store API: `ConnectorConfigStore.rotate_keys()` returns a typed batch result.

Metrics:
- `connector_rotation_total{result}`
- `connector_rotation_batch_latency_seconds`

## Operations Guidance

- Dedup windows: Keep change windows modest (5–15 minutes) to minimize duplicates.
- DLQ: When enabled, inspect queue depths and DLQ metrics for spikes.
- Idempotency logging: Look for “Skipping … ETag unchanged” or “content hash unchanged”.
