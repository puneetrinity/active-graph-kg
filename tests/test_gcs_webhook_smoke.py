"""Archived smoke test for the unavailable GCS connector design.

The connector product has no GCS webhook registration. This historical test is
retained as future-design evidence and must not exercise an API.
"""

import base64
import json
import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skip(reason="connector product unavailable")


class FakeRedis:
    """Minimal Redis stub with list + key ops used by webhooks."""

    def __init__(self):
        self.store = {}

    # List operations
    def lpush(self, key, value):
        lst = self.store.setdefault(key, [])
        lst.insert(0, value)
        return len(lst)

    def rpop(self, key):
        lst = self.store.get(key, [])
        if not lst:
            return None
        return lst.pop()

    def llen(self, key):
        return len(self.store.get(key, []))

    # Key/value ops for dedup
    def set(self, key, value, ex=None, nx=False):
        # nx=True: only set if key doesn't exist (returns True if set, None if key exists)
        if nx:
            if key in self.store:
                return None
            self.store[key] = value
            return True
        # Normal set
        self.store[key] = value
        return True

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        # Ignore TTL in fake
        self.store[key] = value
        return True

    def exists(self, key):
        return 1 if key in self.store else 0

    # SET operations
    def sadd(self, key, *values):
        """Add members to a set."""
        s = self.store.get(key)
        if s is None or not isinstance(s, set):
            s = set()
            self.store[key] = s
        for value in values:
            s.add(value)
        return len(values)

    def smembers(self, key):
        """Get all members of a set."""
        s = self.store.get(key, set())
        if isinstance(s, set):
            return s
        return set()

    # Ping
    def ping(self):
        return True


def test_gcs_webhook_enqueues_one_item(monkeypatch):
    # Arrange
    # Force secret verification path
    os.environ["PUBSUB_VERIFY_SECRET"] = "test-secret"

    # Monkeypatch get_redis_client to return FakeRedis
    fake = FakeRedis()

    def _fake_get_redis_client():
        return fake

    from activekg.common import metrics as common_metrics

    monkeypatch.setattr(common_metrics, "get_redis_client", _fake_get_redis_client, raising=True)

    # Build app client
    from activekg.api.main import app

    client = TestClient(app)

    # Build Pub/Sub envelope
    attributes = {
        "bucketId": "test-bucket",
        "objectId": "docs/sample.pdf",
        "eventType": "OBJECT_FINALIZE",
    }
    payload = {"name": attributes["objectId"], "bucket": attributes["bucketId"]}
    envelope = {
        "message": {
            "messageId": "msg-123",
            "data": base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8"),
            "attributes": attributes,
        },
        "subscription": "projects/proj/subscriptions/sub",
    }

    headers = {
        "Content-Type": "application/json",
        "X-PubSub-Token": "test-secret",
        "X-Goog-Topic": "projects/proj/topics/activekg-gcs-default",
    }

    # Act: call webhook
    resp = client.post("/_webhooks/gcs", data=json.dumps(envelope), headers=headers)

    # Assert
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("status") == "queued"
    assert body.get("tenant_id") == "default"

    # Queue should have one item
    qkey = "connector:gcs:default:queue"
    assert fake.llen(qkey) == 1

    # Simulate worker consumption
    raw = fake.rpop(qkey)
    assert raw is not None
    item = json.loads(raw)
    assert item["uri"].startswith("gs://test-bucket/docs/sample.pdf")
    assert item["operation"] == "upsert"
