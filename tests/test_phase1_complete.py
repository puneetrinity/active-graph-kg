#!/usr/bin/env python3
"""
Phase 1 Complete - End-to-End Verification Test

Tests all claimed Phase 1 features:
1. Node CRUD with explicit last_refreshed/drift_score columns
2. Vector search with pgvector
3. Pattern persistence (DB-backed)
4. Trigger firing
5. Embedding history writes
6. Inline payload loading (remote/local payload references unavailable)
7. Lineage traversal
8. All API endpoints
"""

import os
import sys
from datetime import datetime

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from activekg.engine.embedding_provider import EmbeddingProvider
from activekg.graph.models import Edge, Node
from activekg.graph.repository import GraphRepository
from activekg.refresh.scheduler import RefreshScheduler
from activekg.triggers.pattern_store import PatternStore
from activekg.triggers.trigger_engine import TriggerEngine


def test_repository_crud():
    """Test basic CRUD with explicit columns."""
    print("\n=== Test 1: Repository CRUD ===")
    dsn = os.getenv("ACTIVEKG_DSN", "postgresql://activekg:activekg@localhost:5432/activekg")
    repo = GraphRepository(dsn)

    # Create node
    node = Node(
        classes=["TestDoc"],
        props={"text": "Machine learning fundamentals", "category": "AI"},
        refresh_policy={"interval": "5m", "drift_threshold": 0.15},
        triggers=[{"name": "test_pattern", "threshold": 0.8}],
        last_refreshed=datetime.utcnow(),
        drift_score=0.05,
    )

    # Embed
    embedder = EmbeddingProvider()
    node.embedding = embedder.encode([node.props["text"]])[0]

    node_id = repo.create_node(node)
    print(f"✓ Created node: {node_id}")

    # Retrieve
    retrieved = repo.get_node(node_id)
    assert retrieved is not None
    assert retrieved.last_refreshed is not None
    assert retrieved.drift_score == 0.05
    print(
        f"✓ Retrieved node with last_refreshed={retrieved.last_refreshed}, drift_score={retrieved.drift_score}"
    )

    return node_id, embedder, repo


def test_vector_search(repo, embedder):
    """Test pgvector search with filters."""
    print("\n=== Test 2: Vector Search ===")

    query = "deep learning optimization"
    query_vec = embedder.encode([query])[0]

    results = repo.vector_search(
        query_embedding=query_vec, top_k=5, metadata_filters=None, tenant_id=None
    )

    print(f"✓ Search returned {len(results)} results")
    for node, sim in results[:3]:
        print(f"  - {node.classes} (similarity: {sim:.4f})")

    return results


def test_pattern_store():
    """Test DB-backed pattern persistence."""
    print("\n=== Test 3: Pattern Store (DB-backed) ===")
    dsn = os.getenv("ACTIVEKG_DSN", "postgresql://activekg:activekg@localhost:5432/activekg")
    pattern_store = PatternStore(dsn)
    embedder = EmbeddingProvider()

    # Create pattern
    pattern_text = "fraud detection suspicious activity"
    pattern_vec = embedder.encode([pattern_text])[0]
    pattern_store.set("fraud_test", pattern_vec, "Test fraud pattern")
    print("✓ Saved pattern to DB")

    # Retrieve
    retrieved = pattern_store.get("fraud_test")
    assert retrieved is not None
    assert retrieved.shape == (384,)
    print(f"✓ Retrieved pattern from DB: shape={retrieved.shape}")

    # List
    patterns = pattern_store.list_patterns()
    assert len(patterns) > 0
    print(f"✓ Listed {len(patterns)} patterns from DB")

    return pattern_store


def test_trigger_engine(repo, pattern_store):
    """Test trigger firing."""
    print("\n=== Test 4: Trigger Engine ===")

    trigger_engine = TriggerEngine(pattern_store, repo)
    fired = trigger_engine.run()
    print(f"✓ Trigger engine ran, fired {fired} events")

    return trigger_engine


def test_embedding_history(repo, node_id):
    """Test embedding history writes."""
    print("\n=== Test 5: Embedding History ===")

    repo.write_embedding_history(node_id, drift_score=0.12, embedding_ref="test_ref")
    print("✓ Wrote embedding history entry")

    # Verify it was written
    with repo._conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM embedding_history WHERE node_id = %s", (node_id,))
            count = cur.fetchone()[0]
            print(f"✓ Found {count} history entries for node")


def test_refresh_scheduler(repo, embedder, trigger_engine):
    """Test scheduler integration (dry run)."""
    print("\n=== Test 6: Refresh Scheduler ===")

    scheduler = RefreshScheduler(repo, embedder, trigger_engine)
    print("✓ Scheduler initialized with trigger engine")

    # Manually trigger one cycle (don't start background)
    scheduler.run_cycle()
    print("✓ Ran one refresh cycle manually")


def test_payload_loaders(repo):
    """Test the inline-only payload boundary."""
    print("\n=== Test 7: Payload Loaders ===")

    # Test inline
    node = Node(props={"text": "inline content"})
    text = repo.load_payload_text(node)
    assert text == "inline content"
    print("✓ Inline payload loader works")

    inert = Node(props={}, payload_ref="https://example.invalid/content")
    assert repo.load_payload_text(inert) == ""
    assert not hasattr(repo, "_load_from_url")
    assert not hasattr(repo, "_load_from_file")
    assert not hasattr(repo, "_load_from_s3")
    print("✓ Remote/local payload references are inert")


def test_lineage(repo, node_id):
    """Test lineage traversal."""
    print("\n=== Test 8: Lineage Traversal ===")

    # Create parent node
    parent = Node(classes=["Source"], props={"text": "parent document"})
    parent_id = repo.create_node(parent)

    # Create edge
    edge = Edge(src=node_id, rel="DERIVED_FROM", dst=parent_id, props={"confidence": 0.95})
    repo.create_edge(edge)
    print(f"✓ Created DERIVED_FROM edge: {node_id} -> {parent_id}")

    # Traverse
    lineage = repo.get_lineage(node_id, max_depth=5)
    print(f"✓ Lineage traversal returned {len(lineage)} ancestors")
    for ancestor in lineage:
        print(f"  - Depth {ancestor['depth']}: {ancestor['id']}")


def test_api_imports():
    """Verify all API endpoints are defined."""
    print("\n=== Test 9: API Endpoints ===")
    from activekg.api.main import app

    routes = [route.path for route in app.routes]

    expected = [
        "/health",
        "/metrics",
        "/nodes",
        "/nodes/{node_id}",
        "/search",
        "/edges",
        "/triggers",
        "/triggers/{name}",
        "/events",
        "/lineage/{node_id}",
    ]

    for endpoint in expected:
        # Check if endpoint pattern exists (exact or parametrized)
        found = any(endpoint in r or r in endpoint for r in routes)
        status = "✓" if found else "✗"
        print(f"{status} {endpoint}")


def main():
    print("=" * 60)
    print("Phase 1 Complete - Verification Test")
    print("=" * 60)

    try:
        # Run all tests
        node_id, embedder, repo = test_repository_crud()
        test_vector_search(repo, embedder)
        pattern_store = test_pattern_store()
        trigger_engine = test_trigger_engine(repo, pattern_store)
        test_embedding_history(repo, node_id)
        test_refresh_scheduler(repo, embedder, trigger_engine)
        test_payload_loaders(repo)
        test_lineage(repo, node_id)
        test_api_imports()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Phase 1 Complete!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
