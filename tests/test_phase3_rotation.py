#!/usr/bin/env python3
"""Test Phase 3: Key Rotation Endpoint.

Tests:
1. Create configs encrypted with V1
2. Dry-run rotation (should count candidates without changing)
3. Switch to V2 and rotate keys
4. Verify configs still work after rotation
5. Test provider/tenant filtering
6. Verify metrics tracking
"""

import os

import pytest
import requests
from cryptography.fernet import Fernet

if __name__ == "__main__":
    print("MEMORY_CONNECTORS_UNAVAILABLE: connector rotation proof is archived.")
    raise SystemExit(1)

pytest.skip("connector product unavailable", allow_module_level=True)

# Generate two different KEKs for testing rotation
KEK_V1 = Fernet.generate_key().decode()
KEK_V2 = Fernet.generate_key().decode()

print("=" * 60)
print("Phase 3 Rotation Test Suite")
print("=" * 60)
print(f"V1 KEK: {KEK_V1[:20]}...")
print(f"V2 KEK: {KEK_V2[:20]}...")

# Set V1 as active initially
os.environ["CONNECTOR_KEK_V1"] = KEK_V1
os.environ["CONNECTOR_KEK_V2"] = KEK_V2
os.environ["CONNECTOR_KEK_ACTIVE_VERSION"] = "1"
os.environ["ACTIVEKG_DSN"] = "postgresql:///activekg?host=/var/run/postgresql&port=5433"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["RUN_SCHEDULER"] = "false"

# Test 1: Module imports
print("\n=== Test 1: Module imports ===")
try:
    from activekg.connectors.config_store import get_config_store

    print("✓ Modules imported successfully")
except Exception as e:
    print(f"✗ Import failed: {e}")
    exit(1)

# Test 2: Create configs with V1 encryption
print("\n=== Test 2: Create configs with V1 encryption ===")
try:
    store = get_config_store()

    # Create configs for different tenants and providers
    configs = [
        (
            "tenant_rotation_1",
            "s3",
            {"endpoint": "https://s3.amazonaws.com", "access_key": "key1", "secret_key": "secret1"},
        ),
        (
            "tenant_rotation_1",
            "gcs",
            {
                "endpoint": "https://storage.googleapis.com",
                "access_key": "key2",
                "secret_key": "secret2",
            },
        ),
        (
            "tenant_rotation_2",
            "s3",
            {"endpoint": "https://s3.amazonaws.com", "access_key": "key3", "secret_key": "secret3"},
        ),
        (
            "tenant_rotation_3",
            "s3",
            {"endpoint": "https://s3.amazonaws.com", "access_key": "key4", "secret_key": "secret4"},
        ),
    ]

    for tenant_id, provider, config in configs:
        result = store.upsert(tenant_id, provider, config)
        assert result, f"Failed to create config for {tenant_id}/{provider}"

    print(f"✓ Created {len(configs)} configs encrypted with V1")
    print(f"  Active KEK version: {store.encryption.active_version}")

    # Verify all configs are using V1
    all_configs = store.list_all()
    print(f"  Total configs in DB: {len(all_configs)}")

except Exception as e:
    print(f"✗ Config creation failed: {e}")
    exit(1)

# Test 3: Dry-run rotation (should only count candidates)
print("\n=== Test 3: Dry-run rotation (V1 active) ===")
try:
    response = requests.post(
        "http://localhost:8000/_admin/connectors/rotate_keys", json={"dry_run": True}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    result = response.json()

    assert result["dry_run"] is True, "Expected dry_run=true"
    assert result["rotated"] == 0, "Expected 0 rotations in dry-run"
    assert result["candidates"] == 0, "Expected 0 candidates when active version matches"

    print("✓ Dry-run completed")
    print(f"  Candidates: {result.get('candidates', 0)} (expected 0 since all are V1)")

except Exception as e:
    print(f"✗ Dry-run test failed: {e}")
    exit(1)

# Test 4: Switch to V2 and test dry-run
print("\n=== Test 4: Switch to V2 and dry-run rotation ===")
try:
    # Switch active version to V2
    os.environ["CONNECTOR_KEK_ACTIVE_VERSION"] = "2"

    # Re-initialize store to pick up new active version
    import activekg.connectors.config_store as config_module

    config_module._config_store = None  # Clear singleton
    store = get_config_store()

    print(f"  Active KEK version: {store.encryption.active_version}")
    assert store.encryption.active_version == "2", "Expected active version 2"

    # Dry-run should now find candidates
    response = requests.post(
        "http://localhost:8000/_admin/connectors/rotate_keys", json={"dry_run": True}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    result = response.json()

    assert result["dry_run"] is True, "Expected dry_run=true"
    assert result["rotated"] == 0, "Expected 0 rotations in dry-run"
    assert result["candidates"] == 4, f"Expected 4 candidates, got {result['candidates']}"

    print("✓ Dry-run with V2 active")
    print(f"  Candidates: {result['candidates']} (expected 4)")

except Exception as e:
    print(f"✗ V2 dry-run test failed: {e}")
    exit(1)

# Test 5: Actual rotation (V1 → V2)
print("\n=== Test 5: Actual rotation (V1 → V2) ===")
try:
    response = requests.post(
        "http://localhost:8000/_admin/connectors/rotate_keys", json={"dry_run": False}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    result = response.json()

    assert result["dry_run"] is False, "Expected dry_run=false"
    assert result["rotated"] == 4, f"Expected 4 rotations, got {result['rotated']}"
    assert result["errors"] == 0, f"Expected 0 errors, got {result['errors']}"

    print("✓ Rotation completed")
    print(f"  Rotated: {result['rotated']}")
    print(f"  Errors: {result['errors']}")

except Exception as e:
    print(f"✗ Rotation test failed: {e}")
    exit(1)

# Test 6: Verify configs still work after rotation
print("\n=== Test 6: Verify configs work after rotation ===")
try:
    # Reload configs and verify they decrypt correctly
    for tenant_id, provider, expected_config in configs:
        loaded_config = store.get(tenant_id, provider)
        assert loaded_config is not None, f"Failed to load {tenant_id}/{provider}"
        assert loaded_config["access_key"] == expected_config["access_key"], (
            f"Decryption failed for {tenant_id}/{provider}"
        )
        print(f"  ✓ {tenant_id}/{provider}: Decrypted correctly")

    print("✓ All configs decrypt correctly with V2")

except Exception as e:
    print(f"✗ Post-rotation verification failed: {e}")
    exit(1)

# Test 7: Test provider filtering
print("\n=== Test 7: Test provider filtering ===")
try:
    # Create one more config with V1 for a different provider
    os.environ["CONNECTOR_KEK_ACTIVE_VERSION"] = "1"
    _config_store = None
    store_v1 = get_config_store()

    store_v1.upsert(
        "tenant_filter_test",
        "azure",
        {
            "endpoint": "https://blob.azure.com",
            "access_key": "azure_key",
            "secret_key": "azure_secret",
        },
    )

    # Switch back to V2
    os.environ["CONNECTOR_KEK_ACTIVE_VERSION"] = "2"
    _config_store = None
    store = get_config_store()

    # Rotate only s3 provider
    response = requests.post(
        "http://localhost:8000/_admin/connectors/rotate_keys",
        json={"providers": ["s3"], "dry_run": False},
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    result = response.json()

    # Should rotate 0 (all s3 configs already on V2)
    assert result["rotated"] == 0, (
        f"Expected 0 rotations (s3 already on V2), got {result['rotated']}"
    )

    # Now rotate azure
    response = requests.post(
        "http://localhost:8000/_admin/connectors/rotate_keys",
        json={"providers": ["azure"], "dry_run": False},
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    result = response.json()

    assert result["rotated"] == 1, f"Expected 1 rotation (azure), got {result['rotated']}"

    print("✓ Provider filtering works correctly")
    print(f"  S3 filter: {0} rotated (expected 0, already on V2)")
    print(f"  Azure filter: {1} rotated (expected 1)")

    # Cleanup
    store.delete("tenant_filter_test", "azure")

except Exception as e:
    print(f"✗ Provider filtering test failed: {e}")
    exit(1)

# Test 8: Test tenant filtering
print("\n=== Test 8: Test tenant filtering ===")
try:
    # Create config for specific tenant with V1
    os.environ["CONNECTOR_KEK_ACTIVE_VERSION"] = "1"
    _config_store = None
    store_v1 = get_config_store()

    store_v1.upsert(
        "tenant_specific_test",
        "s3",
        {
            "endpoint": "https://s3.amazonaws.com",
            "access_key": "specific_key",
            "secret_key": "specific_secret",
        },
    )

    # Switch to V2
    os.environ["CONNECTOR_KEK_ACTIVE_VERSION"] = "2"
    _config_store = None
    store = get_config_store()

    # Rotate only for specific tenant
    response = requests.post(
        "http://localhost:8000/_admin/connectors/rotate_keys",
        json={"tenants": ["tenant_specific_test"], "dry_run": False},
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    result = response.json()

    assert result["rotated"] == 1, f"Expected 1 rotation, got {result['rotated']}"

    print("✓ Tenant filtering works correctly")
    print(f"  Filtered rotation: {result['rotated']} configs")

    # Cleanup
    store.delete("tenant_specific_test", "s3")

except Exception as e:
    print(f"✗ Tenant filtering test failed: {e}")
    exit(1)

# Test 9: Cleanup test configs
print("\n=== Test 9: Cleanup ===")
try:
    for tenant_id, provider, _ in configs:
        store.delete(tenant_id, provider)
    print("✓ Test configs cleaned up")
except Exception as e:
    print(f"⚠ Cleanup warning: {e}")

print("\n" + "=" * 60)
print("✅ Phase 3 rotation tests completed successfully!")
print("=" * 60)
print("\nRotation endpoint verified:")
print("  1. ✅ Dry-run mode counts candidates without changes")
print("  2. ✅ Actual rotation (V1 → V2) succeeds")
print("  3. ✅ Configs decrypt correctly after rotation")
print("  4. ✅ Provider filtering works")
print("  5. ✅ Tenant filtering works")
print("\nMetrics to check in Prometheus:")
print("  - connector_rotation_total{result='rotated'}")
print("  - connector_rotation_total{result='error'}")
print("  - connector_rotation_batch_latency_seconds")
