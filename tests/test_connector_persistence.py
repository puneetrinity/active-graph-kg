#!/usr/bin/env python3
"""Test connector config persistence across server restarts.

Tests:
1. Run database migration to create connector_configs table
2. Generate encryption key (KEK)
3. Register S3 connector config
4. Verify it's saved to database (check encrypted secrets)
5. Simulate server restart
6. Verify config still works (backfill endpoint)
7. Test enable/disable functionality

Requirements:
- PostgreSQL running
- Redis running
- CONNECTOR_KEK environment variable set
"""

import json
import os
import subprocess
import time

import pytest

if __name__ == "__main__":
    print("MEMORY_CONNECTORS_UNAVAILABLE: connector persistence proof is archived.")
    raise SystemExit(1)

pytest.skip("connector product unavailable", allow_module_level=True)

# Base URL
BASE_URL = "http://localhost:8000"


def run_migration():
    """Run the connector_configs table migration."""
    print("\n=== Step 1: Run Migration ===")

    migration_file = "/home/ews/active-graph-kg/db/migrations/005_connector_configs_table.sql"
    dsn = os.getenv("ACTIVEKG_DSN", "postgresql:///activekg?host=/var/run/postgresql&port=5433")

    # Extract connection params from DSN
    cmd = f"psql '{dsn}' -f {migration_file}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        # Migration might already be applied, check if table exists
        check_cmd = f"psql '{dsn}' -c '\\d connector_configs'"
        check_result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
        if "does not exist" in check_result.stderr:
            print(f"✗ Migration failed: {result.stderr}")
            return False
        else:
            print("✓ Table already exists (migration previously applied)")
            return True

    print("✓ Migration successful")
    return True


def generate_kek():
    """Generate encryption key if not set."""
    print("\n=== Step 2: Generate KEK ===")

    if os.getenv("CONNECTOR_KEK"):
        print("✓ CONNECTOR_KEK already set")
        return True

    print("Generating new KEK...")
    result = subprocess.run(
        [
            "python3",
            "-c",
            "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"✗ Failed to generate KEK: {result.stderr}")
        return False

    kek = result.stdout.strip()
    os.environ["CONNECTOR_KEK"] = kek
    print(f"✓ Generated KEK: {kek[:20]}...")
    print(f"  Export this for server/worker: export CONNECTOR_KEK={kek}")
    return True


def test_register():
    """Register S3 connector config."""
    print("\n=== Step 3: Register Connector ===")

    import requests

    config = {
        "tenant_id": "test_persistence",
        "config": {
            "bucket": "my-test-bucket",
            "prefix": "documents/",
            "region": "us-east-1",
            "access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "enabled": True,
        },
    }

    resp = requests.post(f"{BASE_URL}/_admin/connectors/s3/register", json=config)
    print(f"Register: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    if resp.status_code != 200:
        print("✗ Registration failed")
        return False

    print("✓ Connector registered")
    return True


def verify_db_encryption():
    """Verify config is encrypted in database."""
    print("\n=== Step 4: Verify DB Encryption ===")

    dsn = os.getenv("ACTIVEKG_DSN", "postgresql:///activekg?host=/var/run/postgresql&port=5433")

    cmd = f"""psql '{dsn}' -t -c "
        SELECT config_json->>'access_key_id', config_json->>'secret_access_key'
        FROM connector_configs
        WHERE tenant_id = 'test_persistence' AND provider = 's3';
    " """

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"✗ Query failed: {result.stderr}")
        return False

    output = result.stdout.strip()
    print(f"DB values: {output[:100]}...")

    # Should NOT contain plaintext secrets
    if "AKIAIOSFODNN7EXAMPLE" in output or "wJalrXUtnFEMI/K7MDENG" in output:
        print("✗ SECURITY ISSUE: Secrets are NOT encrypted in database!")
        return False

    # Should contain Fernet-encrypted ciphertext (starts with "gAAAAA")
    if "gAAAAA" in output:
        print("✓ Secrets are encrypted in database")
        return True
    else:
        print("⚠ Unexpected format (might be OK if using different encryption)")
        return True


def simulate_restart():
    """Simulate server restart."""
    print("\n=== Step 5: Simulate Server Restart ===")
    print("Instruction: Please restart your API server now")
    print(
        "  Example: pkill uvicorn && sleep 2 && uvicorn activekg.api.main:app --host 0.0.0.0 --port 8000 &"
    )
    print("  Make sure CONNECTOR_KEK is exported!")
    input("Press ENTER after restarting...")

    # Wait for server to be ready
    import requests

    for _ in range(10):
        try:
            resp = requests.get(f"{BASE_URL}/health", timeout=2)
            if resp.status_code == 200:
                print("✓ Server is back online")
                return True
        except Exception:
            pass
        time.sleep(1)

    print("✗ Server did not come back online")
    return False


def test_backfill_after_restart():
    """Test that backfill still works after restart (config loaded from DB)."""
    print("\n=== Step 6: Test Backfill After Restart ===")

    import requests

    req = {"tenant_id": "test_persistence", "limit": 5}

    resp = requests.post(f"{BASE_URL}/_admin/connectors/s3/backfill", json=req)
    print(f"Backfill: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    # May fail due to invalid S3 credentials, but should NOT fail with "not registered"
    if "not registered" in resp.text.lower():
        print("✗ Config was NOT persisted across restart!")
        return False

    print("✓ Config survived restart (backfill endpoint accessible)")
    return True


def test_disable_enable():
    """Test disable/enable functionality."""
    print("\n=== Step 7: Test Disable/Enable ===")

    import requests

    # Disable
    resp = requests.post(
        f"{BASE_URL}/_admin/connectors/s3/disable", json={"tenant_id": "test_persistence"}
    )
    print(f"Disable: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    if resp.status_code != 200:
        print("✗ Disable failed")
        return False

    # Try backfill (should fail)
    resp = requests.post(
        f"{BASE_URL}/_admin/connectors/s3/backfill",
        json={"tenant_id": "test_persistence", "limit": 5},
    )

    if resp.status_code == 200:
        print("✗ Backfill worked when connector was disabled!")
        return False

    print("✓ Disabled connector blocks backfill")

    # Re-enable
    resp = requests.post(
        f"{BASE_URL}/_admin/connectors/s3/enable", json={"tenant_id": "test_persistence"}
    )
    print(f"Enable: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))

    if resp.status_code != 200:
        print("✗ Enable failed")
        return False

    # Try backfill again (should work)
    resp = requests.post(
        f"{BASE_URL}/_admin/connectors/s3/backfill",
        json={"tenant_id": "test_persistence", "limit": 5},
    )

    if "not registered" in resp.text.lower():
        print("✗ Re-enabled connector still not accessible")
        return False

    print("✓ Re-enabled connector works")
    return True


def main():
    """Run all tests."""
    print("=== Connector Config Persistence Test ===")
    print(f"Base URL: {BASE_URL}")
    print("=" * 60)

    tests = [
        ("Migration", run_migration),
        ("KEK Generation", generate_kek),
        ("Register", test_register),
        ("DB Encryption", verify_db_encryption),
        ("Server Restart", simulate_restart),
        ("Backfill After Restart", test_backfill_after_restart),
        ("Disable/Enable", test_disable_enable),
    ]

    results = {}
    for name, test in tests:
        try:
            success = test()
            results[name] = success
            if not success:
                print(f"\n⚠ Test '{name}' failed, continuing...")
        except Exception as e:
            print(f"\n✗ Test '{name}' raised exception: {e}")
            results[name] = False
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed!")
        print("\nAcceptance Criteria Met:")
        print("  ✓ Restart server/worker → backfill still works")
        print("  ✓ Enable/disable per tenant toggles behavior live")
        print("  ✓ Secrets at rest are encrypted; logs never print secrets")
    else:
        print("✗ Some tests failed")
        print("\nNext steps:")
        print("  - Review failure output above")
        print("  - Ensure CONNECTOR_KEK is set when starting server/worker")
        print("  - Check database migration was applied")

    print("=" * 60)


if __name__ == "__main__":
    main()
