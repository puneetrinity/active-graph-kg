#!/usr/bin/env python3
"""Quick validation script for corrected endpoints."""

import json
import subprocess
import sys

import requests

API_URL = "http://localhost:8000"

# Generate tokens
print("Generating JWT tokens...")
result = subprocess.run(["python3", "scripts/generate_test_jwt.py"], capture_output=True, text=True)
lines = result.stdout.split("\n")
ADMIN_TOKEN = None
for line in lines:
    if "Admin Token:" in line:
        ADMIN_TOKEN = line.split("Admin Token:")[1].strip()
        break

if not ADMIN_TOKEN:
    print("Failed to generate token!")
    sys.exit(1)


def test(name, func):
    """Run a test and print result."""
    try:
        result = func()
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
        return result
    except Exception as e:
        print(f"✗ FAIL - {name}: {e}")
        return False


print("\n" + "=" * 60)
print("QUICK VALIDATION - Corrected Endpoints")
print("=" * 60 + "\n")


# 1. Schema/indexes/RLS
def check_migrate():
    resp = requests.post(
        f"{API_URL}/admin/migrate", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}, timeout=30
    )
    data = resp.json()
    print(f"  Migrate response: {json.dumps(data, indent=2)}")
    return resp.status_code == 200 and data.get("status") == "ok"


def check_db_status():
    resp = requests.get(
        f"{API_URL}/admin/db_status", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}, timeout=10
    )
    data = resp.json()
    print(f"  DB Status: {json.dumps(data, indent=2)}")
    return resp.status_code == 200


test("POST /admin/migrate", check_migrate)
test("GET /admin/db_status", check_db_status)

# 2. Node CRUD with corrected hard delete
print("\nNode CRUD Tests:")
node_id = None


def create_node():
    global node_id
    resp = requests.post(
        f"{API_URL}/nodes",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
        json={"class_name": "QuickTest", "text": "Quick validation test node"},
        timeout=10,
    )
    data = resp.json()
    node_id = data.get("id")
    print(f"  Created node: {node_id}")
    return resp.status_code == 200 and node_id is not None


def get_node():
    resp = requests.get(
        f"{API_URL}/nodes/{node_id}", headers={"Authorization": f"Bearer {ADMIN_TOKEN}"}, timeout=10
    )
    return resp.status_code == 200


def update_node():
    resp = requests.put(
        f"{API_URL}/nodes/{node_id}",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
        json={"properties": {"validated": True}},
        timeout=10,
    )
    return resp.status_code == 200


def delete_node_hard():
    # Create another node for hard delete
    resp = requests.post(
        f"{API_URL}/nodes",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
        json={"class_name": "ToDelete", "text": "Will be hard deleted"},
        timeout=10,
    )
    if resp.status_code != 200:
        return False
    delete_id = resp.json().get("id")

    # Hard delete with correct param
    resp2 = requests.delete(
        f"{API_URL}/nodes/{delete_id}?hard=true",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
        timeout=10,
    )
    print(f"  Hard delete node {delete_id}: {resp2.status_code}")
    return resp2.status_code == 200


test("POST /nodes", create_node)
if node_id:
    test("GET /nodes/{id}", get_node)
    test("PUT /nodes/{id}", update_node)
test("DELETE /nodes/{id}?hard=true", delete_node_hard)

# 3. Search
print("\nSearch Tests:")


def test_search():
    resp = requests.post(
        f"{API_URL}/search",
        headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
        json={"query": "validation test", "mode": "weighted", "top_k": 10},
        timeout=15,
    )
    data = resp.json()
    print(f"  Search results: {len(data.get('results', []))}")
    return resp.status_code == 200


test("POST /search", test_search)

# 5. Closed connector product
print("\nConnector Retirement Tests:")


def test_connectors_unavailable():
    resp = requests.get(
        f"{API_URL}/_admin/connectors/",
        timeout=10,
    )
    expected = {
        "detail": {
            "code": "MEMORY_CONNECTORS_UNAVAILABLE",
            "message": "Connectors are not available.",
        }
    }
    print(f"  Connector boundary: HTTP {resp.status_code}")
    return (
        resp.status_code == 410
        and resp.headers.get("cache-control") == "no-store"
        and resp.json() == expected
    )


test("Connector product unavailable", test_connectors_unavailable)

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)
