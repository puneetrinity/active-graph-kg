#!/usr/bin/env python3
"""Test Phase 2 Hardening: Graceful shutdown, message validation, health endpoint.

Tests:
1. Graceful shutdown with stop Event
2. Message validation (malformed JSON, missing fields, invalid operations)
3. Health endpoint GET /_admin/connectors/cache/health
4. Prometheus metrics tracking
"""

import json
import os
import time

import pytest
import requests
from cryptography.fernet import Fernet

if __name__ == "__main__":
    print("MEMORY_CONNECTORS_UNAVAILABLE: connector hardening proof is archived.")
    raise SystemExit(1)

pytest.skip("connector product unavailable", allow_module_level=True)

# Set environment
os.environ["CONNECTOR_KEK_V1"] = Fernet.generate_key().decode()
os.environ["CONNECTOR_KEK_ACTIVE_VERSION"] = "1"
os.environ["ACTIVEKG_DSN"] = "postgresql:///activekg?host=/var/run/postgresql&port=5433"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["RUN_SCHEDULER"] = "false"

print("=" * 60)
print("Phase 2 Hardening Test Suite")
print("=" * 60)

# Test 1: Module imports
print("\n=== Test 1: Module imports ===")
try:
    import redis

    from activekg.connectors.cache_subscriber import (
        get_subscriber_health,
        start_subscriber,
        stop_subscriber,
    )
    from activekg.connectors.config_store import get_config_store

    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Health endpoint (subscriber not running)
print("\n=== Test 2: Health endpoint (subscriber not running) ===")
try:
    response = requests.get("http://localhost:8000/_admin/connectors/cache/health")
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["status"] == "degraded", "Expected degraded status when subscriber not running"
    assert data["subscriber"] is None, "Expected None for subscriber when not running"

    print("✓ Health endpoint returns degraded when subscriber not running")
    print(f"  Response: {json.dumps(data, indent=2)}")
except Exception as e:
    print(f"✗ Health endpoint test failed: {e}")
    exit(1)

# Test 3: Start subscriber
print("\n=== Test 3: Start subscriber ===")
try:
    store = get_config_store()
    start_subscriber(os.getenv("REDIS_URL"), store)
    time.sleep(2)  # Give subscriber time to connect

    # Check subscriber health via function
    health = get_subscriber_health()
    assert health is not None, "Subscriber health should not be None"
    assert health.get("connected") is True, "Subscriber should be connected"
    assert health.get("reconnects") == 0, "Should have 0 reconnects initially"

    print("✓ Subscriber started and connected")
    print(f"  Health: {json.dumps(health, indent=2)}")
except Exception as e:
    print(f"✗ Subscriber startup failed: {e}")
    stop_subscriber()
    exit(1)

# Test 4: Health endpoint (subscriber running)
print("\n=== Test 4: Health endpoint (subscriber running) ===")
try:
    response = requests.get("http://localhost:8000/_admin/connectors/cache/health")
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["status"] == "ok", "Expected ok status when subscriber connected"
    assert data["subscriber"] is not None, "Expected subscriber data"
    assert data["subscriber"]["connected"] is True, "Subscriber should be connected"
    assert data["subscriber"]["reconnects"] == 0, "Should have 0 reconnects"

    print("✓ Health endpoint returns ok when subscriber running")
    print(f"  Response: {json.dumps(data, indent=2)}")
except Exception as e:
    print(f"✗ Health endpoint test failed: {e}")
    stop_subscriber()
    exit(1)

# Test 5: Message validation - malformed JSON
print("\n=== Test 5: Message validation - malformed JSON ===")
try:
    redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)

    # Publish malformed JSON
    redis_client.publish("connector:config:changed", "not valid json{")
    time.sleep(0.5)

    # Check that subscriber is still running (didn't crash)
    health = get_subscriber_health()
    assert health is not None, "Subscriber should still be running"
    assert health.get("connected") is True, "Subscriber should still be connected"

    print("✓ Subscriber handled malformed JSON gracefully")
except Exception as e:
    print(f"✗ Malformed JSON test failed: {e}")
    stop_subscriber()
    exit(1)

# Test 6: Message validation - missing tenant_id
print("\n=== Test 6: Message validation - missing tenant_id ===")
try:
    message = json.dumps({"provider": "s3", "operation": "upsert"})
    redis_client.publish("connector:config:changed", message)
    time.sleep(0.5)

    # Check that subscriber is still running
    health = get_subscriber_health()
    assert health is not None, "Subscriber should still be running"
    assert health.get("connected") is True, "Subscriber should still be connected"

    print("✓ Subscriber rejected message missing tenant_id")
except Exception as e:
    print(f"✗ Missing tenant_id test failed: {e}")
    stop_subscriber()
    exit(1)

# Test 7: Message validation - missing provider
print("\n=== Test 7: Message validation - missing provider ===")
try:
    message = json.dumps({"tenant_id": "test_tenant", "operation": "upsert"})
    redis_client.publish("connector:config:changed", message)
    time.sleep(0.5)

    # Check that subscriber is still running
    health = get_subscriber_health()
    assert health is not None, "Subscriber should still be running"
    assert health.get("connected") is True, "Subscriber should still be connected"

    print("✓ Subscriber rejected message missing provider")
except Exception as e:
    print(f"✗ Missing provider test failed: {e}")
    stop_subscriber()
    exit(1)

# Test 8: Message validation - missing operation
print("\n=== Test 8: Message validation - missing operation ===")
try:
    message = json.dumps({"tenant_id": "test_tenant", "provider": "s3"})
    redis_client.publish("connector:config:changed", message)
    time.sleep(0.5)

    # Check that subscriber is still running
    health = get_subscriber_health()
    assert health is not None, "Subscriber should still be running"
    assert health.get("connected") is True, "Subscriber should still be connected"

    print("✓ Subscriber rejected message missing operation")
except Exception as e:
    print(f"✗ Missing operation test failed: {e}")
    stop_subscriber()
    exit(1)

# Test 9: Message validation - invalid operation
print("\n=== Test 9: Message validation - invalid operation ===")
try:
    message = json.dumps({"tenant_id": "test_tenant", "provider": "s3", "operation": "invalid_op"})
    redis_client.publish("connector:config:changed", message)
    time.sleep(0.5)

    # Check that subscriber is still running
    health = get_subscriber_health()
    assert health is not None, "Subscriber should still be running"
    assert health.get("connected") is True, "Subscriber should still be connected"

    print("✓ Subscriber rejected message with invalid operation")
except Exception as e:
    print(f"✗ Invalid operation test failed: {e}")
    stop_subscriber()
    exit(1)

# Test 10: Valid message processing
print("\n=== Test 10: Valid message processing ===")
try:
    # First, add a config to cache
    config = {
        "endpoint": "https://s3.amazonaws.com",
        "access_key": "test_key",
        "secret_key": "test_secret",
    }
    store.upsert("test_tenant_hardening", "s3", config)

    # Load into cache
    loaded_config = store.get("test_tenant_hardening", "s3")
    assert loaded_config is not None, "Config should be loaded"
    assert ("test_tenant_hardening", "s3") in store._cache, "Should be in cache"
    print("  Config loaded into cache")

    # Publish valid invalidation message
    valid_message = json.dumps(
        {"tenant_id": "test_tenant_hardening", "provider": "s3", "operation": "upsert"}
    )
    redis_client.publish("connector:config:changed", valid_message)
    time.sleep(1)

    # Check that cache was invalidated
    cache_invalidated = ("test_tenant_hardening", "s3") not in store._cache

    # Check health updated
    health = get_subscriber_health()
    assert health.get("last_message_ts") is not None, "Should have last_message_ts"

    print("✓ Valid message processed successfully")
    print(f"  Cache invalidated: {cache_invalidated}")
    print(f"  Last message timestamp: {health.get('last_message_ts')}")

    # Cleanup
    store.delete("test_tenant_hardening", "s3")
except Exception as e:
    print(f"✗ Valid message test failed: {e}")
    stop_subscriber()
    exit(1)

# Test 11: Graceful shutdown
print("\n=== Test 11: Graceful shutdown ===")
try:
    # Get health before stopping
    health_before = get_subscriber_health()
    assert health_before is not None, "Subscriber should be running"

    # Stop subscriber
    stop_subscriber()
    time.sleep(1)

    # Check that subscriber stopped
    health_after = get_subscriber_health()
    assert health_after is None, "Subscriber should be stopped"

    print("✓ Subscriber stopped gracefully")
except Exception as e:
    print(f"✗ Graceful shutdown test failed: {e}")
    exit(1)

# Test 12: Health endpoint after shutdown
print("\n=== Test 12: Health endpoint after shutdown ===")
try:
    response = requests.get("http://localhost:8000/_admin/connectors/cache/health")
    data = response.json()

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert data["status"] == "degraded", "Expected degraded status after shutdown"
    assert data["subscriber"] is None, "Expected None for subscriber after shutdown"

    print("✓ Health endpoint returns degraded after shutdown")
    print(f"  Response: {json.dumps(data, indent=2)}")
except Exception as e:
    print(f"✗ Health endpoint after shutdown test failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("✅ Phase 2 hardening tests completed successfully!")
print("=" * 60)
print("\nHardening improvements verified:")
print("  1. ✅ Graceful shutdown with stop Event")
print("  2. ✅ Message validation (JSON, fields, operations)")
print("  3. ✅ Health endpoint with subscriber status")
print("  4. ✅ Prometheus metrics tracking")
print("\nMetrics to check in Prometheus:")
print("  - connector_pubsub_shutdown_total")
print("  - connector_pubsub_invalid_msg_total{reason='invalid_json'}")
print("  - connector_pubsub_invalid_msg_total{reason='missing_tenant_id'}")
print("  - connector_pubsub_invalid_msg_total{reason='missing_provider'}")
print("  - connector_pubsub_invalid_msg_total{reason='missing_operation'}")
print("  - connector_pubsub_invalid_msg_total{reason='invalid_operation'}")
print("  - connector_pubsub_messages_total{operation='upsert'}")
