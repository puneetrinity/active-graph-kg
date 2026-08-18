#!/usr/bin/env python3
"""
Seed Demo Data for actvgraph-kg

Creates a small dataset of synthetic resumes and job posts with a default
refresh_policy so the scheduler (or auto-embed) makes them searchable quickly.

Usage:
  # Random seeding (legacy):
  python3 examples/seed_demo.py --api-url http://localhost:8000 --tenant demo

  # Canonical seeding with ID mapping:
  python3 examples/seed_demo.py --api-url http://localhost:8000 --from-file evaluation/datasets/seed_nodes.json

Notes:
  - Safe to run multiple times; IDs are randomized (or from file).
  - With --from-file: generates evaluation/datasets/id_map.json mapping external_id → UUID
"""

import argparse
import json
import random
import string
from pathlib import Path

import requests


def rand_id(prefix: str) -> str:
    return f"{prefix}_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


RESUMES: list[str] = [
    "Senior Java engineer with 8 years experience in Spring, Hibernate, AWS.",
    "Data scientist skilled in Python, PyTorch, NLP, MLOps on Kubernetes.",
    "Backend engineer with Go, Postgres, Redis, event-driven architectures.",
    "Site Reliability Engineer focused on observability, Prometheus, Grafana.",
    "Machine learning engineer experienced with vector databases and pgvector.",
]

JOBS: list[str] = [
    "Hiring Senior Java Engineer to build microservices with Spring Boot.",
    "Looking for ML Engineer to improve NLP pipelines and LLM scoring.",
    "Seeking Backend Engineer to design APIs with Postgres and caching.",
    "SRE position to build metrics and alerting with Prometheus and Grafana.",
    "Data Platform Engineer with experience in vector search and embeddings.",
]

DEFAULT_POLICY = {"interval": "5m", "drift_threshold": 0.15}


def post_node(api: str, text: str, classes: list[str], tenant: str | None) -> str:
    payload = {
        "classes": classes,
        "props": {"text": text},
        "refresh_policy": DEFAULT_POLICY,
    }
    if tenant:
        payload["tenant_id"] = tenant
    r = requests.post(f"{api}/nodes", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["id"]


def load_seed_file(file_path: str) -> list[dict]:
    """Load seed data from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def seed_from_file(api: str, file_path: str, tenant: str | None) -> dict[str, str]:
    """Seed nodes from file and return external_id -> UUID mapping."""
    nodes = load_seed_file(file_path)
    id_map = {}
    created_ids = []

    print(f"Loading {len(nodes)} nodes from {file_path}...")

    for node in nodes:
        external_id = node["external_id"]
        text = node["text"]
        classes = node["classes"]

        uuid = post_node(api, text, classes, tenant)
        id_map[external_id] = uuid
        created_ids.append(uuid)
        print(f"  ✓ {external_id} → {uuid[:8]}...")

    print(f"✓ Created {len(created_ids)} nodes")

    # Save id_map
    id_map_path = Path(file_path).parent / "id_map.json"
    with open(id_map_path, "w") as f:
        json.dump(id_map, f, indent=2)
    print(f"✓ Saved ID mapping to {id_map_path}")

    return id_map, created_ids


def seed_random(api: str, tenant: str | None, num_resumes: int, num_jobs: int) -> list[str]:
    """Legacy random seeding."""
    created_ids = []

    # Create resumes
    for _ in range(num_resumes):
        text = random.choice(RESUMES)
        text = f"{text} #{rand_id('cv')}"
        nid = post_node(api, text, ["Resume"], tenant)
        created_ids.append(nid)

    # Create jobs
    for _ in range(num_jobs):
        text = random.choice(JOBS)
        text = f"{text} #{rand_id('job')}"
        nid = post_node(api, text, ["Job"], tenant)
        created_ids.append(nid)

    print(f"✓ Created {len(created_ids)} nodes")
    return created_ids


def main():
    ap = argparse.ArgumentParser(description="Seed demo data for actvgraph-kg")
    ap.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    ap.add_argument("--tenant", default=None, help="Optional tenant_id for seeded data")
    ap.add_argument("--from-file", default=None, help="Load canonical seed data from JSON file")
    ap.add_argument(
        "--resumes", type=int, default=45, help="How many resume nodes to create (random mode)"
    )
    ap.add_argument(
        "--jobs", type=int, default=45, help="How many job nodes to create (random mode)"
    )
    args = ap.parse_args()

    api = args.api_url.rstrip("/")
    tenant = args.tenant

    print(f"Seeding to {api} (tenant={tenant or 'None'})...")

    if args.from_file:
        # Canonical seeding with ID mapping
        id_map, created_ids = seed_from_file(api, args.from_file, tenant)
    else:
        # Legacy random seeding
        created_ids = seed_random(api, tenant, args.resumes, args.jobs)

    # Force background refresh for quick searchability
    try:
        r = requests.post(f"{api}/admin/refresh", json=created_ids, timeout=60)
        if r.ok:
            data = r.json()
            print(f"✓ Forced refresh for {data.get('refreshed', 'N/A')} nodes (admin refresh)")
        else:
            print(f"⚠ Admin refresh HTTP {r.status_code}")
    except Exception as e:
        print(f"⚠ Could not run admin refresh: {e}")

    print("Done. Try /search or /ask with your queries.")


if __name__ == "__main__":
    main()
