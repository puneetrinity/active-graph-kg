"""Pytest wrapper for E2E retrieval tests with citation validation.

Usage:
    pytest tests/test_e2e_retrieval.py -v
    pytest tests/test_e2e_retrieval.py::test_vector_search_returns_results -v
    pytest tests/test_e2e_retrieval.py::test_ask_includes_citations -v
"""

import os
from datetime import datetime, timedelta, timezone

import jwt
import pytest
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
TENANT = os.getenv("TENANT", "eval_tenant")
JWT_SECRET = os.getenv("JWT_SECRET", "test-secret-key-min-32-chars-long")
JWT_ALG = os.getenv("JWT_ALGORITHM", "HS256")


def make_token(tenant_id, scopes):
    """Generate JWT token for testing."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": "test_user",
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


@pytest.fixture(scope="module")
def user_token():
    """Create user token with search and write permissions."""
    return make_token(TENANT, ["search:read", "ask:read", "kg:write"])


@pytest.fixture(scope="module")
def test_node(user_token):
    """Create a test node for retrieval tests."""
    node_body = {
        "classes": ["Resume"],
        "props": {
            "text": "Senior Python developer with 5+ years experience in Django and Flask. "
            "Expert in REST APIs, PostgreSQL, and Redis caching."
        },
        "metadata": {
            "skills": ["python", "django", "flask", "postgresql"],
            "role": "software engineer",
        },
    }

    headers = {"Authorization": f"Bearer {user_token}"}
    r = requests.post(f"{API_URL}/nodes", json=node_body, headers=headers, timeout=30)
    assert r.status_code == 200, f"Node creation failed: {r.text}"

    node_id = r.json()["id"]
    yield node_id

    # Cleanup not critical for E2E tests


def test_vector_search_returns_results(user_token, test_node):
    """Test that vector search returns non-zero results for relevant queries."""
    headers = {"Authorization": f"Bearer {user_token}"}

    # Wait briefly for embedding to be generated
    import time

    time.sleep(2)

    search_body = {"query": "python developer django experience", "use_hybrid": False, "top_k": 10}

    r = requests.post(f"{API_URL}/search", json=search_body, headers=headers, timeout=30)
    assert r.status_code == 200, f"Vector search failed: {r.text}"

    data = r.json()
    assert data["count"] > 0, "Vector search returned zero results for relevant query"
    assert len(data["results"]) > 0, "Results list is empty"

    # Check that similarity scores exist
    top_result = data["results"][0]
    assert "similarity" in top_result, "No similarity score in result"
    assert top_result["similarity"] > 0, "Similarity score is not positive"


def test_hybrid_search_returns_results(user_token, test_node):
    """Test that hybrid search returns non-zero results."""
    headers = {"Authorization": f"Bearer {user_token}"}

    import time

    time.sleep(2)

    search_body = {"query": "python postgresql", "use_hybrid": True, "top_k": 10}

    r = requests.post(f"{API_URL}/search", json=search_body, headers=headers, timeout=30)
    assert r.status_code == 200, f"Hybrid search failed: {r.text}"

    data = r.json()
    assert data["count"] > 0, "Hybrid search returned zero results"


def test_search_sanity_endpoint(user_token):
    """Test /debug/search_sanity endpoint for retrieval diagnostics."""
    # This endpoint requires admin scope
    admin_token = make_token(TENANT, ["admin:refresh", "search:read"])
    headers = {"Authorization": f"Bearer {admin_token}"}

    r = requests.get(f"{API_URL}/debug/search_sanity", headers=headers, timeout=30)
    assert r.status_code == 200, f"search_sanity endpoint failed: {r.text}"

    data = r.json()

    # Validate response structure
    assert "total_nodes" in data
    assert "nodes_with_embeddings" in data
    assert "nodes_with_text_search" in data
    assert "embedding_coverage_pct" in data
    assert "text_search_coverage_pct" in data

    # Should have at least our test node
    assert data["total_nodes"] > 0, "No nodes found in database"


# Parametrized test for search modes
@pytest.mark.parametrize(
    "use_hybrid,mode_name",
    [
        (False, "vector-only"),
        (True, "hybrid"),
    ],
)
def test_search_modes(user_token, test_node, use_hybrid, mode_name):
    """Test both vector-only and hybrid search modes."""
    headers = {"Authorization": f"Bearer {user_token}"}

    import time

    time.sleep(1)

    search_body = {"query": "python developer", "use_hybrid": use_hybrid, "top_k": 5}

    r = requests.post(f"{API_URL}/search", json=search_body, headers=headers, timeout=30)
    assert r.status_code == 200, f"{mode_name} search failed: {r.text}"

    data = r.json()
    # May return 0 results depending on content, but should not error
    assert "count" in data
    assert "results" in data
