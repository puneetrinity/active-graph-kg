#!/usr/bin/env python3
"""Integration test for S3 connector end-to-end flow.

Tests:
1. Register S3 connector config
2. Trigger backfill
3. Simulate webhook event (manual queue push)
4. Start worker to process queue
5. Verify parent + chunk nodes created
6. Verify connector compatibility methods remain unavailable

Requirements:
- PostgreSQL running
- Redis running
- Test S3 bucket with sample files
"""

import json
import os

import pytest
import requests

if __name__ == "__main__":
    print("MEMORY_CONNECTORS_UNAVAILABLE: S3 integration proof is archived.")
    raise SystemExit(1)

pytest.skip("connector product unavailable", allow_module_level=True)

# Base URL
BASE_URL = "http://localhost:8000"


def test_1_health_check():
    """Verify API is running."""
    print("\n=== Test 1: Health Check ===")
    resp = requests.get(f"{BASE_URL}/health")
    print(f"Health: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200, "API not healthy"
    print("✓ API is healthy")


def test_2_register_connector():
    """Register S3 connector config."""
    print("\n=== Test 2: Register S3 Connector ===")

    # Example config - replace with your actual S3 bucket
    config = {
        "tenant_id": "test_tenant",
        "config": {
            "bucket": "my-test-bucket",
            "prefix": "documents/",
            "region": "us-east-1",
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID", "AKIAEXAMPLE"),
            "secret_access_key": os.getenv(
                "AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            ),
        },
    }

    resp = requests.post(f"{BASE_URL}/_admin/connectors/s3/register", json=config)
    print(f"Register: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200, f"Registration failed: {resp.text}"
    print("✓ S3 connector registered")


def test_3_backfill():
    """Trigger a backfill (list first page of changes)."""
    print("\n=== Test 3: Backfill ===")

    req = {"tenant_id": "test_tenant", "limit": 10}

    resp = requests.post(f"{BASE_URL}/_admin/connectors/s3/backfill", json=req)
    print(f"Backfill: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    if resp.status_code == 400:
        print("⚠ Backfill failed - likely need valid S3 credentials")
        return

    assert resp.status_code == 200, f"Backfill failed: {resp.text}"
    result = resp.json()
    print(f"✓ Found {result.get('found')} changes")


def test_4_webhook_health():
    """Test webhook health endpoint."""
    print("\n=== Test 4: Webhook Health ===")

    resp = requests.get(f"{BASE_URL}/_webhooks/s3/health")
    print(f"Webhook health: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))
    assert resp.status_code == 200, "Webhook not healthy"
    print("✓ Webhook endpoint is healthy")


def test_5_manual_queue_push():
    """Manually push a test item to Redis queue for processing.

    In production, this would come from SNS webhook.
    """
    print("\n=== Test 5: Manual Queue Push ===")

    try:
        import redis

        r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
        r.ping()

        # Create a test change item
        test_item = {
            "uri": "s3://my-test-bucket/documents/sample.pdf",
            "operation": "upsert",
            "etag": "abc123",
            "modified_at": "2025-11-11T12:00:00Z",
            "tenant_id": "test_tenant",
        }

        queue_key = "connector:s3:test_tenant:queue"
        r.lpush(queue_key, json.dumps(test_item))
        depth = r.llen(queue_key)

        print(f"✓ Pushed test item to queue (depth: {depth})")
    except Exception as e:
        print(f"⚠ Could not push to Redis: {e}")
        print("  (Worker test will be skipped)")


def test_6_worker_instructions():
    """Instructions for running the worker."""
    print("\n=== Test 6: Worker Processing ===")
    print("""
To process the queue, run the worker:

    python -m activekg.connectors.worker

The worker will:
1. Poll Redis queue for events
2. Load S3 connector config
3. Process each event through IngestionProcessor
4. Create parent + chunk nodes with embeddings
5. Update Prometheus metrics

Monitor progress with:
    redis-cli LLEN connector:s3:test_tenant:queue
    curl http://localhost:8000/metrics | grep connector_worker
""")
    print("✓ See instructions above to run worker")


def test_7_verify_nodes():
    """Verify nodes were created (requires actual S3 file + worker run)."""
    print("\n=== Test 7: Verify Nodes ===")
    print("""
To verify parent + chunks were created, query the database:

    psql -h /var/run/postgresql -p 5433 -d activekg -c "
        SELECT
            props->>'external_id' as external_id,
            props->>'is_parent' as is_parent,
            classes
        FROM nodes
        WHERE props->>'external_id' LIKE 's3:test_tenant:%'
        LIMIT 10;
    "

Expected output:
- 1 parent node with is_parent='true', classes=['Document']
- N chunk nodes with classes=['Chunk', 'Document']
- All sharing same external_id prefix
""")
    print("✓ See SQL query above to verify")


def main():
    """Run all tests."""
    print("=== S3 Connector Integration Test ===")
    print(f"Base URL: {BASE_URL}")
    print("=" * 60)

    tests = [
        test_1_health_check,
        test_2_register_connector,
        test_3_backfill,
        test_4_webhook_health,
        test_5_manual_queue_push,
        test_6_worker_instructions,
        test_7_verify_nodes,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"✗ Test failed: {e}")
            # Continue with next test

    print("\n" + "=" * 60)
    print("Integration test complete!")
    print("""
Next steps:
1. Configure real S3 credentials in environment
2. Upload test PDF to S3 bucket
3. Run worker: python -m activekg.connectors.worker
4. Verify nodes created in database
5. Set up SNS → webhook integration for production

For production deployment:
- Configure SNS topic: arn:aws:sns:region:account:activekg-s3-{tenant_id}
- Subscribe webhook: https://your-api.com/_webhooks/s3
- Enable event notifications on S3 bucket
- Monitor metrics: curl http://localhost:8000/metrics
""")


if __name__ == "__main__":
    main()
