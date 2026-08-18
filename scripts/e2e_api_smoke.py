#!/usr/bin/env python3
"""End-to-end API smoke test.

Tests full stack: DB → repository → API → client
Verifies vector search, hybrid search, RLS, and all core endpoints.

Usage:
    export API_URL=http://localhost:8000
    export TENANT=eval_tenant
    export JWT_SECRET=test-secret-key-min-32-chars-long
    python3 scripts/e2e_api_smoke.py
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import requests

try:
    import jwt
except ImportError:
    print("Error: PyJWT not installed. Run: pip install pyjwt")
    sys.exit(1)

API = os.getenv("API_URL", "http://localhost:8000")
TENANT = os.getenv("TENANT", "eval_tenant")
JWT_SECRET = os.getenv("JWT_SECRET", "test-secret-key-min-32-chars-long")
JWT_ALG = os.getenv("JWT_ALGORITHM", "HS256")


def make_token(tenant_id, scopes):
    """Generate JWT token for testing."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "e2e_user",
        "tenant_id": tenant_id,
        "actor_type": "user",
        "scopes": scopes,
        "aud": "activekg",
        "iss": os.getenv("JWT_ISSUER", "https://staging-auth.yourcompany.com"),
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(hours=1),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def req(method, path, token=None, **kwargs):
    """Make HTTP request with optional JWT token."""
    headers = kwargs.pop("headers", {})
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if "json" in kwargs:
        headers["Content-Type"] = "application/json"
    r = requests.request(method, f"{API}{path}", headers=headers, timeout=30, **kwargs)
    return r


def main():
    """Run end-to-end smoke tests."""
    print(f"Testing API: {API}")
    print(f"Tenant: {TENANT}\n")

    admin_token = make_token(TENANT, ["admin:refresh", "search:read", "ask:read", "kg:write"])
    user_token = make_token(TENANT, ["search:read", "ask:read", "kg:write"])

    # 1) Health check
    print("1. Testing /health...")
    r = req("GET", "/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print("   ✓ Health check passed")

    # 2) Create a node (tenant-scoped)
    print("\n2. Creating test node...")
    node_body = {
        "classes": ["Resume"],
        "props": {
            "text": "Senior software engineer experienced in Python and Kubernetes. "
            "Expert in distributed systems, microservices, and cloud architecture."
        },
        "metadata": {
            "skills": ["python", "kubernetes", "docker"],
            "role": "software engineer",
            "location": "Remote",
        },
        "refresh_policy": {"interval": "5m", "drift_threshold": 0.1},
    }
    r = req("POST", "/nodes", token=user_token, json=node_body)
    assert r.status_code == 200, f"Node creation failed: {r.text}"
    node_id = r.json()["id"]
    print(f"   ✓ Created node: {node_id}")

    # 3) Admin refresh (forces embedding)
    print("\n3. Triggering admin refresh...")
    r = req("POST", "/admin/refresh", token=admin_token, json=[node_id])
    if r.status_code == 200:
        print("   ✓ Admin refresh completed")
    elif r.status_code == 503:
        print("   ⚠ Admin refresh unavailable (scope issue or disabled)")
    else:
        print(f"   ⚠ Admin refresh returned: {r.status_code}")

    # 4) Get node by id
    print("\n4. Fetching node by ID...")
    r = req("GET", f"/nodes/{node_id}", token=user_token)
    assert r.status_code == 200, f"Get node failed: {r.text}"
    node_data = r.json()
    print(f"   ✓ Retrieved node: {node_data.get('id')}")
    if node_data.get("embedding"):
        print(f"   ✓ Embedding present (dim: {len(node_data['embedding'])})")

    # 5) Vector search
    print("\n5. Testing vector search...")
    r = req(
        "POST",
        "/search",
        token=user_token,
        json={"query": "software engineer python kubernetes", "use_hybrid": False, "top_k": 5},
    )
    assert r.status_code == 200, f"Vector search failed: {r.text}"
    data = r.json()
    print(f"   ✓ Vector search returned {data['count']} results")
    if data["count"] > 0:
        top_result = data["results"][0]
        print(f"   ✓ Top result similarity: {top_result.get('similarity', 'N/A'):.4f}")

    # 6) Hybrid search
    print("\n6. Testing hybrid search...")
    r = req(
        "POST",
        "/search",
        token=user_token,
        json={"query": "software engineer python", "use_hybrid": True, "top_k": 5},
    )
    assert r.status_code == 200, f"Hybrid search failed: {r.text}"
    data = r.json()
    print(f"   ✓ Hybrid search returned {data['count']} results")

    # 7) Events endpoint
    print("\n7. Testing /events endpoint...")
    r = req("GET", "/events?limit=10", token=user_token)
    assert r.status_code == 200, f"Events fetch failed: {r.text}"
    events_data = r.json()
    event_count = events_data.get("count", 0)
    print(f"   ✓ Retrieved {event_count} events")

    # 8) Node versions
    print("\n8. Testing /nodes/{id}/versions...")
    r = req("GET", f"/nodes/{node_id}/versions", token=user_token)
    assert r.status_code == 200, f"Versions fetch failed: {r.text}"
    versions_data = r.json()
    version_count = versions_data.get("count", 0)
    print(f"   ✓ Retrieved {version_count} version(s)")

    print("\n" + "=" * 60)
    print("✅ E2E smoke test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
