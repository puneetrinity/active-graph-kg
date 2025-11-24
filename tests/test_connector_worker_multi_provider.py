"""Tests for ConnectorWorker multi-provider processing."""

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from activekg.connectors.worker import ConnectorWorker


class FakeRedis:
    """Minimal Redis stub supporting list ops + scan."""

    def __init__(self):
        self.store = {}

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

    def scan(self, cursor=0, match=None, count=None):
        """Naive scan: return all keys matching prefix with wildcards."""
        keys = []
        if match is None:
            keys = list(self.store.keys())
        else:
            prefix = match.split("*", 1)[0]
            keys = [k for k in self.store.keys() if k.startswith(prefix)]
        # Return bytes to mirror redis-py
        return 0, [k.encode("utf-8") for k in keys]


class DummyConfigStore:
    """Config store stub that tracks calls."""

    def __init__(self):
        self.calls = []

    def get(self, tenant_id, provider):
        self.calls.append((tenant_id, provider))
        return {"provider": provider}


class FakeProcessor:
    """Stub ingestion processor that reports created count only."""

    def __init__(self, connector, repo, redis_client, throttle_max_per_sec=50):
        self.connector = connector

    def process_changes(self, changes):
        return {"created": len(changes), "updated": 0, "skipped": 0, "deleted": 0, "errors": 0}


@pytest.mark.parametrize("provider", ["s3", "gcs", "drive"])
def test_process_batch_handles_multiple_providers(monkeypatch, provider):
    # Arrange queues
    redis_client = FakeRedis()
    tenant_id = "tenant-123"
    queue_key = f"connector:{provider}:{tenant_id}:queue"
    uri_map = {
        "s3": "s3://bucket/key",
        "gcs": "gs://bucket/key",
        "drive": "drive://file-id",
    }
    redis_client.lpush(
        queue_key,
        json.dumps(
            {
                "uri": uri_map[provider],
                "operation": "upsert",
                "etag": "etag-1",
                "modified_at": datetime.utcnow().isoformat(),
            }
        ),
    )

    config_store = DummyConfigStore()
    built_connectors = []

    def _fake_build(self, provider_name, tenant, config):
        built_connectors.append((provider_name, tenant, config))
        return SimpleNamespace(provider_name=provider_name, tenant_id=tenant)

    # Patch processor + connector factory
    monkeypatch.setattr("activekg.connectors.worker.IngestionProcessor", FakeProcessor)

    worker = ConnectorWorker(
        redis_client=redis_client,
        repo=MagicMock(),
        config_store=config_store,
        batch_size=5,
        poll_interval_seconds=0.01,
    )
    monkeypatch.setattr(worker, "_build_connector", _fake_build.__get__(worker, ConnectorWorker))

    # Act
    processed = worker.process_batch(tenant_id, provider)

    # Assert
    assert processed == 1
    assert built_connectors == [(provider, tenant_id, {"provider": provider})]
    assert (tenant_id, provider) in config_store.calls
    assert redis_client.llen(queue_key) == 0
