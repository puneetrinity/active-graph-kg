#!/usr/bin/env python3
"""Comprehensive JWT + Rate Limiting + RLS Test Suite"""

import sys

import requests

API_URL = "http://localhost:8000"

# JWT tokens (valid for 24 hours)
TENANT_A_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyX2EiLCJ0ZW5hbnRfaWQiOiJ0ZW5hbnRfYSIsImFjdG9yX3R5cGUiOiJ1c2VyIiwic2NvcGVzIjpbInNlYXJjaDpyZWFkIiwibm9kZXM6d3JpdGUiXSwiYXVkIjoiYWN0aXZla2ciLCJpc3MiOiJodHRwczovL3N0YWdpbmctYXV0aC55b3VyY29tcGFueS5jb20iLCJpYXQiOjE3NjI1MjA1MDksIm5iZiI6MTc2MjUyMDUwOSwiZXhwIjoxNzYyNjA2OTA5fQ.Ak8cyiAKYxYFcfH-qK-z6zDz5CSAb5-m0ZVJTnBl0Ps"
TENANT_B_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyX2IiLCJ0ZW5hbnRfaWQiOiJ0ZW5hbnRfYiIsImFjdG9yX3R5cGUiOiJ1c2VyIiwic2NvcGVzIjpbInNlYXJjaDpyZWFkIiwibm9kZXM6d3JpdGUiXSwiYXVkIjoiYWN0aXZla2ciLCJpc3MiOiJodHRwczovL3N0YWdpbmctYXV0aC55b3VyY29tcGFueS5jb20iLCJpYXQiOjE3NjI1MjA1MTgsIm5iZiI6MTc2MjUyMDUxOCwiZXhwIjoxNzYyNjA2OTE4fQ.XUGZ2qbmBpY5xIDp4gOfb_inJh0Tdqyxl7hO0wCEm4Y"
ADMIN_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbl91c2VyIiwidGVuYW50X2lkIjoidGVuYW50X2EiLCJhY3Rvcl90eXBlIjoidXNlciIsInNjb3BlcyI6WyJhZG1pbjpyZWZyZXNoIiwic2VhcmNoOnJlYWQiLCJub2Rlczp3cml0ZSJdLCJhdWQiOiJhY3RpdmVrZyIsImlzcyI6Imh0dHBzOi8vc3RhZ2luZy1hdXRoLnlvdXJjb21wYW55LmNvbSIsImlhdCI6MTc2MjUyMDUyNywibmJmIjoxNzYyNTIwNTI3LCJleHAiOjE3NjI2MDY5Mjd9.KBcC9e_HeKjnJAkYCFcRs2Yj0nerkv9_CTUObbZ95qo"

# Test results
passed = 0
failed = 0
test_results = []


def test(name, condition, details=""):
    """Record test result"""
    global passed, failed
    if condition:
        passed += 1
        test_results.append(("✅ PASS", name, details))
        print(f"✅ PASS: {name}")
        if details:
            print(f"   {details}")
    else:
        failed += 1
        test_results.append(("❌ FAIL", name, details))
        print(f"❌ FAIL: {name}")
        if details:
            print(f"   {details}")


print("=" * 80)
print("JWT + Rate Limiting + RLS Test Suite")
print("=" * 80)
print()

# Test 1: Health check
print("[Test 1] Health Check")
try:
    r = requests.get(f"{API_URL}/health")
    test("Health endpoint accessible", r.status_code == 200, f"Status: {r.status_code}")
except Exception as e:
    test("Health endpoint accessible", False, str(e))
print()

# Test 2: JWT authentication on protected endpoints
print("[Test 2] JWT Authentication")
try:
    # Without JWT
    r = requests.post(f"{API_URL}/search", json={"query": "test", "top_k": 1}, timeout=5)
    test(
        "Search endpoint rejects unauthenticated requests",
        r.status_code == 401,
        f"Status: {r.status_code}",
    )
except requests.Timeout:
    test(
        "Search endpoint rejects unauthenticated requests",
        False,
        "Timeout while checking authentication",
    )
except Exception as e:
    test("Search endpoint rejects unauthenticated requests", False, str(e))

try:
    # With JWT
    r = requests.post(
        f"{API_URL}/search",
        headers={"Authorization": f"Bearer {TENANT_A_TOKEN}"},
        json={"query": "test", "limit": 5},
    )
    test("Authenticated search works", r.status_code == 200, f"Status: {r.status_code}")
except Exception as e:
    test("Authenticated search works", False, str(e))
print()

# Test 3: Write endpoint protection (tenant_id from JWT)
print("[Test 3] Write Endpoint Protection")
try:
    # Create node - attempt to set tenant_b but JWT says tenant_a
    r = requests.post(
        f"{API_URL}/nodes",
        headers={"Authorization": f"Bearer {TENANT_A_TOKEN}"},
        json={
            "classes": ["TestNode"],
            "props": {"name": "rls_test_node", "test_id": "rls_test_1"},
            "tenant_id": "tenant_b",  # This should be IGNORED
        },
    )

    if r.status_code == 200:
        node_id = r.json().get("id")
        test("Node creation successful", node_id is not None, f"Node ID: {node_id}")

        # Verify tenant_id is tenant_a (from JWT), not tenant_b (from body)
        # We can't check tenant_id directly since GET doesn't return it,
        # but we can verify tenant B can't access it
        if node_id:
            # Test 4: RLS Tenant Isolation
            print()
            print("[Test 4] RLS Tenant Isolation")

            # Tenant A (owner) should be able to access
            r_a = requests.get(
                f"{API_URL}/nodes/{node_id}", headers={"Authorization": f"Bearer {TENANT_A_TOKEN}"}
            )
            test(
                "Tenant A can access own node", r_a.status_code == 200, f"Status: {r_a.status_code}"
            )

            # Tenant B (different tenant) should NOT be able to access
            r_b = requests.get(
                f"{API_URL}/nodes/{node_id}", headers={"Authorization": f"Bearer {TENANT_B_TOKEN}"}
            )
            test(
                "Tenant B CANNOT access tenant A node (RLS working)",
                r_b.status_code == 404,
                f"Status: {r_b.status_code} (expected 404)",
            )

            # Test RLS on query param override attempt
            r_override = requests.get(
                f"{API_URL}/nodes/{node_id}?tenant_id=tenant_b",
                headers={"Authorization": f"Bearer {TENANT_A_TOKEN}"},
            )
            test(
                "Read endpoint ignores query param tenant_id",
                r_override.status_code == 200,
                "JWT tenant_id used, query param ignored",
            )
    else:
        test("Node creation successful", False, f"Status: {r.status_code}, Body: {r.text}")
except Exception as e:
    test("Write endpoint protection", False, str(e))
print()

# Test 5: Scope-based authorization
print("[Test 5] Scope-Based Authorization")
try:
    # Without admin scope
    r = requests.post(
        f"{API_URL}/admin/refresh", headers={"Authorization": f"Bearer {TENANT_A_TOKEN}"}, json={}
    )
    test(
        "Admin endpoint rejects non-admin user",
        r.status_code == 403,
        f"Status: {r.status_code} (expected 403)",
    )
except Exception as e:
    test("Admin endpoint rejects non-admin user", False, str(e))

try:
    # With admin scope
    r = requests.post(
        f"{API_URL}/admin/refresh", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}, json={}
    )
    test(
        "Admin endpoint accepts admin user", r.status_code in [200, 202], f"Status: {r.status_code}"
    )
except Exception as e:
    test("Admin endpoint accepts admin user", False, str(e))
print()

# Test 6: Rate limiting
print("[Test 6] Rate Limiting")
try:
    success_count = 0
    rate_limited_count = 0

    # Send 60 rapid requests (burst limit is 100)
    for _ in range(60):
        r = requests.post(
            f"{API_URL}/search",
            headers={"Authorization": f"Bearer {TENANT_A_TOKEN}"},
            json={"query": "test", "limit": 1},
            timeout=5,
        )
        if r.status_code == 200:
            success_count += 1
        elif r.status_code == 429:
            rate_limited_count += 1

    test(
        "Rate limiting allows reasonable burst",
        50 <= success_count <= 100,
        f"Allowed {success_count}/60 requests, rate limited {rate_limited_count}",
    )

    # Check for rate limit headers
    r = requests.post(
        f"{API_URL}/search",
        headers={"Authorization": f"Bearer {TENANT_A_TOKEN}"},
        json={"query": "test", "limit": 1},
    )
    has_headers = "X-RateLimit-Limit" in r.headers
    test("X-RateLimit headers present", has_headers, f"Headers: {list(r.headers.keys())[:10]}")
except Exception as e:
    test("Rate limiting", False, str(e))
print()

# Test 7: Manual refresh endpoint protection
print("[Test 7] Manual Refresh Endpoint Protection")
if "node_id" in locals():
    try:
        # Without auth
        r = requests.post(f"{API_URL}/nodes/{node_id}/refresh")
        test(
            "Manual refresh requires authentication",
            r.status_code == 401,
            f"Status: {r.status_code}",
        )
    except Exception as e:
        test("Manual refresh requires authentication", False, str(e))

    try:
        # With auth
        r = requests.post(
            f"{API_URL}/nodes/{node_id}/refresh",
            headers={"Authorization": f"Bearer {TENANT_A_TOKEN}"},
        )
        test("Manual refresh works with auth", r.status_code == 200, f"Status: {r.status_code}")
    except Exception as e:
        test("Manual refresh works with auth", False, str(e))
else:
    print("   Skipped (no node created)")
print()

# Summary
print("=" * 80)
print("Test Summary")
print("=" * 80)
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")
print(f"Total: {passed + failed}")
print()

if failed == 0:
    print("🎉 All tests passed!")
    sys.exit(0)
else:
    print("⚠️  Some tests failed:")
    for status, name, details in test_results:
        if status.startswith("❌"):
            print(f"  {status}: {name}")
            if details:
                print(f"     {details}")
    sys.exit(1)
