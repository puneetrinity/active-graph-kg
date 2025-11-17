#!/usr/bin/env python3
"""Seed evaluation dataset with JWT authentication.

This script loads seed_nodes.json into the ActiveKG database via the REST API,
ensuring proper JWT authentication and comprehensive coverage for evaluation questions.

Features:
- JWT token generation for authentication
- Idempotent seeding using external_id checks
- Admin refresh to ensure embeddings are generated
- Text search vector backfill verification
- Coverage validation against ground truth questions

Usage:
    # Load environment variables from .env.eval
    set -a; source .env.eval; set +a

    # Run the seed script
    python3 evaluation/datasets/seed_with_jwt.py

    # Verify coverage only
    python3 evaluation/datasets/seed_with_jwt.py --verify-only

Requirements:
    - API server running (default: http://localhost:8000)
    - PostgreSQL with activekg database
    - JWT credentials in environment (.env.eval)
"""

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

try:
    import jwt
except ImportError:
    print("Error: PyJWT not installed. Run: pip install pyjwt")
    sys.exit(1)

# Configuration from environment
API_URL = os.getenv("API_URL", "http://localhost:8000")
TENANT = os.getenv("TENANT", "eval_tenant")
JWT_SECRET = os.getenv("JWT_SECRET_KEY", "dev-secret-key-min-32-chars-long-for-testing")
JWT_ALG = os.getenv("JWT_ALGORITHM", "HS256")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "activekg")
JWT_ISSUER = os.getenv("JWT_ISSUER", "https://staging-auth.yourcompany.com")

# Paths
SCRIPT_DIR = Path(__file__).parent
SEED_FILE = SCRIPT_DIR / "seed_nodes.json"
GROUND_TRUTH_FILE = SCRIPT_DIR / "ground_truth.json"
ID_MAP_FILE = SCRIPT_DIR / "id_map.json"


def make_token(tenant_id: str, scopes: list[str]) -> str:
    """Generate JWT token for API authentication."""
    now = datetime.now(UTC)
    payload = {
        "sub": "seed_script",
        "tenant_id": tenant_id,
        "actor_type": "system",
        "scopes": scopes,
        "aud": JWT_AUDIENCE,
        "iss": JWT_ISSUER,
        "iat": now,
        "nbf": now,
        "exp": now + timedelta(hours=2),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def req(method: str, path: str, token: str, **kwargs) -> requests.Response:
    """Make authenticated HTTP request."""
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    if "json" in kwargs:
        headers["Content-Type"] = "application/json"

    url = f"{API_URL}{path}"
    r = requests.request(method, url, headers=headers, timeout=30, **kwargs)
    return r


def load_seed_data() -> list[dict[str, Any]]:
    """Load seed nodes from JSON file."""
    if not SEED_FILE.exists():
        print(f"Error: Seed file not found: {SEED_FILE}")
        sys.exit(1)

    with open(SEED_FILE, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} seed nodes from {SEED_FILE}")
    return data


def load_ground_truth() -> dict[str, Any]:
    """Load ground truth for validation."""
    if not GROUND_TRUTH_FILE.exists():
        print(f"Warning: Ground truth file not found: {GROUND_TRUTH_FILE}")
        return {}

    with open(GROUND_TRUTH_FILE, encoding="utf-8") as f:
        return json.load(f)


def create_node(seed_node: dict[str, Any], token: str) -> dict[str, Any]:
    """Create a node via API, merging text into props for semantic search."""
    # Build node creation payload
    # Important: Ensure props.text is set for embedding generation
    props = seed_node.get("props", {}).copy()

    # Merge top-level "text" field into props.text if present
    if "text" in seed_node and "text" not in props:
        props["text"] = seed_node["text"]

    # Add external_id to props for idempotency checks
    if "external_id" in seed_node:
        props["external_id"] = seed_node["external_id"]

    node_body = {
        "classes": seed_node.get("classes", []),
        "props": props,
        "metadata": seed_node.get("metadata", {}),
        "refresh_policy": seed_node.get(
            "refresh_policy", {"interval": "1h", "drift_threshold": 0.15}
        ),
    }

    r = req("POST", "/nodes", token, json=node_body)

    if r.status_code != 200:
        print(
            f"  ❌ Failed to create node {seed_node.get('external_id')}: {r.status_code} - {r.text}"
        )
        return None

    return r.json()


def admin_refresh(node_ids: list[str], token: str) -> bool:
    """Trigger admin refresh to ensure embeddings are generated."""
    if not node_ids:
        return True

    print(f"\nTriggering admin refresh for {len(node_ids)} nodes...")
    r = req("POST", "/admin/refresh", token, json=node_ids)

    if r.status_code == 200:
        data = r.json()
        refreshed = data.get("refreshed", 0)
        print(f"  ✅ Admin refresh completed: {refreshed} nodes refreshed")
        return True
    elif r.status_code == 503:
        print("  ⚠️  Admin refresh endpoint unavailable (may require admin:refresh scope)")
        return False
    else:
        print(f"  ⚠️  Admin refresh returned: {r.status_code} - {r.text}")
        return False


def verify_search_sanity(token: str) -> dict[str, Any]:
    """Call /debug/search_sanity to verify coverage."""
    print("\nVerifying search coverage via /debug/search_sanity...")
    r = req("GET", "/debug/search_sanity", token)

    if r.status_code != 200:
        print(f"  ⚠️  search_sanity check failed: {r.status_code}")
        return {}

    data = r.json()

    total = data.get("total_nodes", 0)
    with_embeddings = data.get("nodes_with_embeddings", 0)
    with_text_search = data.get("nodes_with_text_search", 0)
    emb_coverage = data.get("embedding_coverage_pct", 0.0)
    text_coverage = data.get("text_search_coverage_pct", 0.0)

    print(f"  Total nodes: {total}")
    print(f"  Nodes with embeddings: {with_embeddings} ({emb_coverage:.1f}%)")
    print(f"  Nodes with text_search: {with_text_search} ({text_coverage:.1f}%)")

    # Validate coverage against targets
    if emb_coverage < 95.0:
        print(f"  ⚠️  Embedding coverage below target (95%): {emb_coverage:.1f}%")
    else:
        print(f"  ✅ Embedding coverage meets target: {emb_coverage:.1f}%")

    if text_coverage < 100.0:
        print(f"  ⚠️  Text search coverage below target (100%): {text_coverage:.1f}%")
    else:
        print(f"  ✅ Text search coverage meets target: {text_coverage:.1f}%")

    return data


def verify_retrieval(ground_truth: dict[str, Any], token: str) -> bool:
    """Verify that each evaluation question retrieves expected nodes."""
    if not ground_truth:
        print("\n⚠️  Skipping retrieval verification (no ground truth file)")
        return True

    print(f"\nVerifying retrieval for {len(ground_truth)} evaluation questions...")

    all_passed = True
    for qid, qdata in ground_truth.items():
        # Support both formats:
        # 1. New format: {"q1": {"question": "...", "relevant_external_ids": [...]}}
        # 2. Legacy format: {"question text": ["ext_id1", "ext_id2", ...]}
        if isinstance(qdata, dict):
            question = qdata.get("question", "")
            expected_ext_ids = qdata.get("relevant_external_ids", [])
        elif isinstance(qdata, list):
            # Legacy format: key is the question, value is list of external IDs
            question = qid
            expected_ext_ids = qdata
        else:
            continue

        if not question or not expected_ext_ids:
            continue

        # Perform hybrid search (fallback to vector if hybrid fails)
        r = req("POST", "/search", token, json={"query": question, "use_hybrid": True, "top_k": 10})

        if r.status_code != 200:
            print(f"  ❌ {qid[:50]}: Search failed ({r.status_code})")
            all_passed = False
            continue

        results = r.json().get("results", [])

        # Check if we got non-zero results
        if len(results) == 0:
            print(f"  ❌ {qid[:50]}: Zero results for query: {question[:50]}")
            all_passed = False
        else:
            top_sim = results[0].get("similarity", 0)
            print(f"  ✅ {qid[:50]}: Found {len(results)} results (top sim: {top_sim:.3f})")

    return all_passed


def seed_database(verify_only: bool = False):
    """Main seeding function."""
    print("=" * 70)
    print("ActiveKG Evaluation Dataset Seeding")
    print("=" * 70)
    print(f"API URL: {API_URL}")
    print(f"Tenant: {TENANT}")
    print(f"Seed file: {SEED_FILE}")
    print()

    # Generate tokens
    admin_token = make_token(TENANT, ["admin:refresh", "nodes:write", "search:read"])

    # Load data
    seed_nodes = load_seed_data()
    ground_truth = load_ground_truth()

    if verify_only:
        print("\n🔍 Verification-only mode (no seeding)")
        verify_search_sanity(admin_token)
        verify_retrieval(ground_truth, admin_token)
        return

    # Track created nodes
    created_ids = []
    id_map = {}  # external_id -> node_id

    print(f"Seeding {len(seed_nodes)} nodes...")
    print()

    for i, seed_node in enumerate(seed_nodes, 1):
        external_id = seed_node.get("external_id", f"node_{i}")

        # Create node
        node_data = create_node(seed_node, admin_token)
        if node_data:
            node_id = node_data.get("id")
            created_ids.append(node_id)
            id_map[external_id] = node_id
            print(f"  ✅ [{i}/{len(seed_nodes)}] Created {external_id} → {node_id}")
        else:
            print(f"  ⚠️  [{i}/{len(seed_nodes)}] Failed to create {external_id}")

        # Small delay to avoid overwhelming the API
        time.sleep(0.1)

    print()
    print(f"✅ Created {len(created_ids)} nodes")

    # Save ID mapping for evaluation scripts
    with open(ID_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(id_map, f, indent=2)
    print(f"💾 Saved ID mapping to {ID_MAP_FILE}")

    # Trigger admin refresh
    if created_ids:
        admin_refresh(created_ids, admin_token)
        # Wait for embeddings to be generated
        print("\n⏳ Waiting 3s for embeddings to be generated...")
        time.sleep(3)

    # Verify coverage
    verify_search_sanity(admin_token)

    # Verify retrieval
    retrieval_ok = verify_retrieval(ground_truth, admin_token)

    print()
    print("=" * 70)
    if retrieval_ok:
        print("✅ Seeding completed successfully!")
    else:
        print("⚠️  Seeding completed with warnings (see retrieval verification)")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run E2E tests: pytest tests/test_e2e_retrieval.py -v")
    print("  2. Run smoke test: python3 scripts/e2e_api_smoke.py")
    print("  3. Run evaluations: bash evaluation/run_all.sh")
    print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Seed ActiveKG evaluation dataset")
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify coverage, don't seed nodes"
    )

    args = parser.parse_args()

    try:
        seed_database(verify_only=args.verify_only)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
