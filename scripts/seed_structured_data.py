#!/usr/bin/env python3
"""
Seed structured test data for intent-based queries.

Seeds:
- Open positions (Q5: "What ML engineer positions are open?")
- Performance issues (Q8: "What are the main performance issues reported?")
"""

import time

import requests

API_URL = "http://localhost:8000"

# Open positions data
OPEN_POSITIONS = [
    {
        "tenant_id": "default",
        "classes": ["Job"],
        "props": {
            "title": "Senior ML Engineer",
            "status": "open",
            "text": "Senior Machine Learning Engineer position open. 5+ years experience with PyTorch, TensorFlow, and MLOps required. Work on cutting-edge NLP and computer vision models.",
        },
        "metadata": {
            "department": "AI Research",
            "location": "Remote",
            "posted_date": "2025-11-01",
        },
    },
    {
        "tenant_id": "default",
        "classes": ["Job"],
        "props": {
            "title": "ML Engineer - Computer Vision",
            "status": "open",
            "text": "Machine Learning Engineer specializing in computer vision. Experience with PyTorch, OpenCV, and object detection models. Will work on autonomous driving perception systems.",
        },
        "metadata": {
            "department": "Autonomous Systems",
            "location": "San Francisco",
            "posted_date": "2025-11-02",
        },
    },
    {
        "tenant_id": "default",
        "classes": ["Job"],
        "props": {
            "title": "Staff ML Engineer",
            "status": "open",
            "text": "Staff-level ML Engineer for recommendation systems. Expert knowledge of deep learning, large-scale training, and A/B testing required. Lead ML infrastructure initiatives.",
        },
        "metadata": {
            "department": "Recommendations",
            "location": "New York",
            "posted_date": "2025-10-28",
        },
    },
]

# Performance issues data
PERFORMANCE_ISSUES = [
    {
        "tenant_id": "default",
        "classes": ["Ticket", "Incident"],
        "props": {
            "title": "Search API latency spike",
            "text": "Performance issue: Search API p95 latency increased from 200ms to 2.5s. Users experiencing slow search results. Affects /search endpoint with high concurrency.",
        },
        "metadata": {
            "severity": "high",
            "status": "open",
            "reported_by": "monitoring-system",
            "tags": ["performance", "latency", "search"],
        },
    },
    {
        "tenant_id": "default",
        "classes": ["Bug", "Ticket"],
        "props": {
            "title": "Database query timeout",
            "text": "Performance problem with database queries timing out after 30s. Complex joins on nodes table causing bottleneck. Query planner not using vector index efficiently.",
        },
        "metadata": {
            "severity": "critical",
            "status": "open",
            "reported_by": "alice@example.com",
            "tags": ["performance", "database", "timeout"],
        },
    },
    {
        "tenant_id": "default",
        "classes": ["Incident"],
        "props": {
            "title": "LLM generation bottleneck",
            "text": "Performance degradation in semantic search. Retrieval latency is above the target p95 and needs profiling.",
        },
        "metadata": {
            "severity": "medium",
            "status": "investigating",
            "reported_by": "bob@example.com",
            "tags": ["performance", "llm", "latency", "slow"],
        },
    },
    {
        "tenant_id": "default",
        "classes": ["Ticket"],
        "props": {
            "title": "Embedding computation slowness",
            "text": "Slow embedding generation for new nodes. Taking 500ms per node, blocking node creation. Performance issue with sentence-transformers model loading.",
        },
        "metadata": {
            "severity": "medium",
            "status": "open",
            "reported_by": "charlie@example.com",
            "tags": ["performance", "embedding", "slow"],
        },
    },
]


def seed_data():
    """Seed test data for structured queries."""
    print(f"Seeding structured test data to {API_URL}")
    print("=" * 70)

    # Check API health
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        if resp.status_code != 200:
            print(f"❌ API health check failed: HTTP {resp.status_code}")
            return
        print("✓ API is healthy\n")
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return

    # Seed open positions
    print(f"Seeding {len(OPEN_POSITIONS)} open positions...")
    for i, position in enumerate(OPEN_POSITIONS, 1):
        try:
            resp = requests.post(f"{API_URL}/nodes", json=position, timeout=10)
            if resp.status_code == 200:
                node_id = resp.json().get("id")
                print(
                    f"  [{i}/{len(OPEN_POSITIONS)}] ✓ Created position: {position['props']['title']} (ID: {node_id})"
                )
            else:
                print(f"  [{i}/{len(OPEN_POSITIONS)}] ❌ Failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [{i}/{len(OPEN_POSITIONS)}] ❌ Error: {e}")
        time.sleep(0.5)  # Rate limiting

    print()

    # Seed performance issues
    print(f"Seeding {len(PERFORMANCE_ISSUES)} performance issues...")
    for i, issue in enumerate(PERFORMANCE_ISSUES, 1):
        try:
            resp = requests.post(f"{API_URL}/nodes", json=issue, timeout=10)
            if resp.status_code == 200:
                node_id = resp.json().get("id")
                print(
                    f"  [{i}/{len(PERFORMANCE_ISSUES)}] ✓ Created issue: {issue['props']['title']} (ID: {node_id})"
                )
            else:
                print(f"  [{i}/{len(PERFORMANCE_ISSUES)}] ❌ Failed: HTTP {resp.status_code}")
        except Exception as e:
            print(f"  [{i}/{len(PERFORMANCE_ISSUES)}] ❌ Error: {e}")
        time.sleep(0.5)  # Rate limiting

    print()
    print("=" * 70)
    print("✓ Seeding complete!")
    print(f"  - {len(OPEN_POSITIONS)} open positions")
    print(f"  - {len(PERFORMANCE_ISSUES)} performance issues")
    print()
    print("Wait 5-10 seconds for embeddings to be generated, then test with direct /search.")


if __name__ == "__main__":
    seed_data()
