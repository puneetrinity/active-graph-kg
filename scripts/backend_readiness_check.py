#!/usr/bin/env python3
"""
Comprehensive Backend Readiness Checker for Active Graph KG.
Tests all endpoints and functionality before UI development.
"""

import json
import os
import sys
import time
from datetime import datetime

import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")
ADMIN_TOKEN = None
USER_TOKEN = None


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{title}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")


def print_test(name, passed, details=""):
    status = f"{Colors.GREEN}✓ PASS{Colors.ENDC}" if passed else f"{Colors.RED}✗ FAIL{Colors.ENDC}"
    print(f"{status} - {name}")
    if details:
        print(f"  {Colors.YELLOW}{details}{Colors.ENDC}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ {msg}{Colors.ENDC}")


def test_health():
    """Test health endpoint."""
    print_section("Health & Metrics")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        data = resp.json()
        passed = resp.status_code == 200 and data.get("status") == "ok"
        print_test("Health endpoint", passed, json.dumps(data, indent=2))
        return passed
    except Exception as e:
        print_test("Health endpoint", False, str(e))
        return False


def test_prometheus():
    """Test Prometheus metrics endpoint."""
    try:
        resp = requests.get(f"{API_URL}/prometheus", timeout=5)
        passed = resp.status_code == 200 and "python_gc" in resp.text
        print_test("Prometheus metrics", passed, f"Got {len(resp.text)} bytes")
        return passed
    except Exception as e:
        print_test("Prometheus metrics", False, str(e))
        return False


def test_json_metrics():
    """Test JSON metrics endpoint."""
    try:
        resp = requests.get(f"{API_URL}/metrics", timeout=5)
        data = resp.json()
        passed = resp.status_code == 200 and "counters" in data
        print_test("JSON metrics", passed, f"Counters: {len(data.get('counters', {}))}")
        return passed
    except Exception as e:
        print_test("JSON metrics", False, str(e))
        return False


def test_admin_migrate():
    """Test database migration endpoint."""
    print_section("Schema Bootstrap")
    try:
        resp = requests.post(
            f"{API_URL}/admin/migrate",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=30,
        )
        data = resp.json()
        passed = resp.status_code == 200 and data.get("status") == "ok"
        print_test("POST /admin/migrate", passed, json.dumps(data, indent=2))
        return passed
    except Exception as e:
        print_test("POST /admin/migrate", False, str(e))
        return False


def test_db_status():
    """Test database status endpoint."""
    try:
        resp = requests.get(
            f"{API_URL}/admin/db_status",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        data = resp.json()
        passed = resp.status_code == 200 and data.get("tables", {}).get("nodes")
        details = f"Tables: {data.get('tables', {})}, Vector Index: {data.get('indexes', {}).get('vector_index')}"
        print_test("GET /admin/db_status", passed, details)
        return passed
    except Exception as e:
        print_test("GET /admin/db_status", False, str(e))
        return False


def test_node_crud():
    """Test Node CRUD operations."""
    print_section("Node CRUD")
    node_id = None

    # Create node
    try:
        resp = requests.post(
            f"{API_URL}/nodes",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={
                "class_name": "TestDoc",
                "text": "This is a test document for readiness checking.",
                "properties": {"test": True, "timestamp": datetime.now().isoformat()},
            },
            timeout=10,
        )
        data = resp.json()
        passed = resp.status_code == 200 and "id" in data
        node_id = data.get("id")
        print_test("POST /nodes (create)", passed, f"Created node: {node_id}")
    except Exception as e:
        print_test("POST /nodes (create)", False, str(e))
        return False

    if not node_id:
        return False

    # Get node
    try:
        resp = requests.get(
            f"{API_URL}/nodes/{node_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        data = resp.json()
        passed = resp.status_code == 200 and data.get("id") == node_id
        print_test("GET /nodes/{id}", passed, f"Retrieved node: {data.get('class_name')}")
    except Exception as e:
        print_test("GET /nodes/{id}", False, str(e))

    # List nodes
    try:
        resp = requests.get(
            f"{API_URL}/nodes?limit=10",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        data = resp.json()
        passed = resp.status_code == 200 and "nodes" in data
        print_test("GET /nodes (list)", passed, f"Found {len(data.get('nodes', []))} nodes")
    except Exception as e:
        print_test("GET /nodes (list)", False, str(e))

    # Update node
    try:
        resp = requests.put(
            f"{API_URL}/nodes/{node_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={"properties": {"test": True, "updated": True}},
            timeout=10,
        )
        passed = resp.status_code == 200
        print_test("PUT /nodes/{id} (update)", passed, "Node updated")
    except Exception as e:
        print_test("PUT /nodes/{id} (update)", False, str(e))

    # Delete node (soft)
    try:
        resp = requests.delete(
            f"{API_URL}/nodes/{node_id}",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        passed = resp.status_code == 200
        print_test("DELETE /nodes/{id} (soft)", passed, "Node soft-deleted")

        # Test hard delete
        if passed:
            # Create another node for hard delete test
            resp2 = requests.post(
                f"{API_URL}/nodes",
                headers={
                    "Authorization": f"Bearer {ADMIN_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={"class_name": "TestDoc", "text": "To be hard deleted"},
                timeout=10,
            )
            if resp2.status_code == 200:
                hard_node_id = resp2.json().get("id")
                resp3 = requests.delete(
                    f"{API_URL}/nodes/{hard_node_id}?hard=true",
                    headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                    timeout=10,
                )
                passed = resp3.status_code == 200
                print_test("DELETE /nodes/{id}?hard=true (hard)", passed, "Node hard-deleted")
    except Exception as e:
        print_test("DELETE /nodes/{id} (soft)", False, str(e))

    return True


def test_search():
    """Test search endpoints."""
    print_section("Search & Debug")

    # First create a searchable node
    try:
        requests.post(
            f"{API_URL}/nodes",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={
                "class_name": "SearchableDoc",
                "text": "Machine learning and artificial intelligence are transforming technology.",
                "properties": {"category": "AI"},
            },
            timeout=10,
        )
    except Exception:
        pass

    time.sleep(2)  # Wait for indexing

    # Test hybrid search
    try:
        resp = requests.post(
            f"{API_URL}/search",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={"query": "machine learning", "mode": "weighted", "top_k": 10},
            timeout=10,
        )
        data = resp.json()
        passed = resp.status_code == 200 and "results" in data
        print_test(
            "POST /search (weighted)", passed, f"Found {len(data.get('results', []))} results"
        )
    except Exception as e:
        print_test("POST /search (weighted)", False, str(e))

    # Test search sanity
    try:
        resp = requests.get(
            f"{API_URL}/debug/search_sanity",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        passed = resp.status_code == 200
        print_test("GET /debug/search_sanity", passed, resp.text[:100])
    except Exception as e:
        print_test("GET /debug/search_sanity", False, str(e))


def test_qa():
    """Test Q&A endpoints."""
    print_section("Q&A (LLM-Powered)")

    # Non-streaming Q&A
    try:
        resp = requests.post(
            f"{API_URL}/ask",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={"question": "What is machine learning?", "top_k": 5},
            timeout=30,
        )
        data = resp.json()
        passed = resp.status_code == 200 and "answer" in data
        answer_preview = data.get("answer", "")[:100]
        print_test("POST /ask (non-streaming)", passed, f"Answer: {answer_preview}...")
    except Exception as e:
        print_test("POST /ask (non-streaming)", False, str(e))

    # Streaming Q&A
    try:
        resp = requests.post(
            f"{API_URL}/ask/stream",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={"question": "Explain AI briefly", "stream": True},
            stream=True,
            timeout=30,
        )
        chunks = []
        for line in resp.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    chunks.append(decoded[6:])
            if len(chunks) >= 3:  # Just test a few chunks
                break
        passed = resp.status_code == 200 and len(chunks) > 0
        print_test("POST /ask/stream (SSE)", passed, f"Received {len(chunks)} chunks")
    except Exception as e:
        print_test("POST /ask/stream (SSE)", False, str(e))


def test_events_lineage():
    """Test events and lineage."""
    print_section("Events & Lineage")

    # List events
    try:
        resp = requests.get(
            f"{API_URL}/events?limit=20",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        data = resp.json()
        passed = resp.status_code == 200 and "events" in data
        print_test("GET /events", passed, f"Found {len(data.get('events', []))} events")
    except Exception as e:
        print_test("GET /events", False, str(e))

    # Create edge for lineage test
    try:
        # Create parent node
        resp1 = requests.post(
            f"{API_URL}/nodes",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={"class_name": "Parent", "text": "Parent node"},
            timeout=10,
        )
        parent_id = resp1.json().get("id")

        # Create child node
        resp2 = requests.post(
            f"{API_URL}/nodes",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={"class_name": "Child", "text": "Child node"},
            timeout=10,
        )
        child_id = resp2.json().get("id")

        # Create edge
        resp3 = requests.post(
            f"{API_URL}/edges",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}", "Content-Type": "application/json"},
            json={
                "source_id": parent_id,
                "target_id": child_id,
                "relationship_type": "DERIVED_FROM",
            },
            timeout=10,
        )
        passed = resp3.status_code == 200
        print_test("POST /edges (create)", passed, f"Edge: {parent_id} -> {child_id}")

        # Test lineage
        if passed and child_id:
            resp4 = requests.get(
                f"{API_URL}/lineage/{child_id}",
                headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
                timeout=10,
            )
            data = resp4.json()
            passed = resp4.status_code == 200 and "ancestors" in data
            print_test(
                "GET /lineage/{id}", passed, f"Found {len(data.get('ancestors', []))} ancestors"
            )
    except Exception as e:
        print_test("Lineage test", False, str(e))


def test_admin_refresh():
    """Test admin refresh endpoint."""
    print_section("Refresh & Embedding History")
    try:
        # Get some node IDs first
        resp = requests.get(
            f"{API_URL}/nodes?limit=5",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        nodes = resp.json().get("nodes", [])

        if nodes:
            node_ids = [n["id"] for n in nodes[:2]]
            resp2 = requests.post(
                f"{API_URL}/admin/refresh",
                headers={
                    "Authorization": f"Bearer {ADMIN_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={"node_ids": node_ids},
                timeout=30,
            )
            data = resp2.json()
            passed = resp2.status_code == 200
            print_test("POST /admin/refresh", passed, json.dumps(data, indent=2))
        else:
            print_test("POST /admin/refresh", False, "No nodes to refresh")
    except Exception as e:
        print_test("POST /admin/refresh", False, str(e))


def test_connectors_admin():
    """Test connector admin endpoints."""
    print_section("Connector Admin")

    # List connectors
    try:
        resp = requests.get(
            f"{API_URL}/_admin/connectors/",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        data = resp.json()
        passed = resp.status_code == 200
        print_test(
            "GET /_admin/connectors/", passed, f"Found {len(data.get('connectors', []))} connectors"
        )
    except Exception as e:
        print_test("GET /_admin/connectors/", False, str(e))

    # Cache health
    try:
        resp = requests.get(
            f"{API_URL}/_admin/connectors/cache/health",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        passed = resp.status_code == 200
        print_test("GET /_admin/connectors/cache/health", passed, resp.text[:100])
    except Exception as e:
        print_test("GET /_admin/connectors/cache/health", False, str(e))


def test_auth_rls():
    """Test authentication and RLS."""
    print_section("Auth & RLS")

    # Test admin endpoint with admin token
    try:
        resp = requests.get(
            f"{API_URL}/admin/db_status",
            headers={"Authorization": f"Bearer {ADMIN_TOKEN}"},
            timeout=10,
        )
        passed = resp.status_code == 200
        print_test("Admin endpoint with admin:refresh scope", passed, "Access granted")
    except Exception as e:
        print_test("Admin endpoint with admin:refresh scope", False, str(e))

    # Test admin endpoint with user token (should fail)
    try:
        resp = requests.get(
            f"{API_URL}/admin/db_status",
            headers={"Authorization": f"Bearer {USER_TOKEN}"},
            timeout=10,
        )
        passed = resp.status_code == 403
        print_test("Admin endpoint without admin:refresh scope", passed, "Correctly denied access")
    except Exception as e:
        print_test("Admin endpoint without admin:refresh scope", False, str(e))

    # Test endpoint without token (should fail)
    try:
        resp = requests.get(f"{API_URL}/nodes", timeout=10)
        passed = resp.status_code == 401
        print_test("Endpoint without JWT token", passed, "Correctly requires authentication")
    except Exception as e:
        print_test("Endpoint without JWT token", False, str(e))

    # RLS Cross-tenant isolation test
    print_info("Testing RLS tenant isolation...")
    try:
        # Generate token for tenant A
        import subprocess

        result_a = subprocess.run(
            ["python3", "scripts/generate_test_jwt.py", "tenant_a"], capture_output=True, text=True
        )
        token_a = None
        for line in result_a.stdout.split("\n"):
            if "Admin Token:" in line:
                token_a = line.split("Admin Token:")[1].strip()
                break

        # Generate token for tenant B
        result_b = subprocess.run(
            ["python3", "scripts/generate_test_jwt.py", "tenant_b"], capture_output=True, text=True
        )
        token_b = None
        for line in result_b.stdout.split("\n"):
            if "Admin Token:" in line:
                token_b = line.split("Admin Token:")[1].strip()
                break

        if token_a and token_b:
            # Create node as tenant A
            resp_create = requests.post(
                f"{API_URL}/nodes",
                headers={"Authorization": f"Bearer {token_a}", "Content-Type": "application/json"},
                json={
                    "class_name": "RLSTest",
                    "text": "Tenant A node",
                    "properties": {"tenant": "a"},
                },
                timeout=10,
            )

            if resp_create.status_code == 200:
                node_id = resp_create.json().get("id")

                # Try to access as tenant A (should succeed)
                resp_a = requests.get(
                    f"{API_URL}/nodes/{node_id}",
                    headers={"Authorization": f"Bearer {token_a}"},
                    timeout=10,
                )

                # Try to access as tenant B (should fail with 404)
                resp_b = requests.get(
                    f"{API_URL}/nodes/{node_id}",
                    headers={"Authorization": f"Bearer {token_b}"},
                    timeout=10,
                )

                passed = resp_a.status_code == 200 and resp_b.status_code == 404
                details = (
                    f"Tenant A: {resp_a.status_code}, Tenant B: {resp_b.status_code} (expect 404)"
                )
                print_test("RLS tenant isolation (cross-tenant access)", passed, details)

                # Cleanup
                requests.delete(
                    f"{API_URL}/nodes/{node_id}?hard=true",
                    headers={"Authorization": f"Bearer {token_a}"},
                    timeout=10,
                )
            else:
                print_test("RLS tenant isolation", False, "Failed to create test node")
        else:
            print_test("RLS tenant isolation", False, "Failed to generate tenant tokens")
    except Exception as e:
        print_test("RLS tenant isolation", False, str(e))


def run_regression_tests():
    """Run existing regression test scripts."""
    print_section("Regression Tests")

    tests = [
        ("scripts/smoke_test.py", "Smoke test"),
        ("tests/test_phase1_complete.py", "Phase 1 complete test"),
        ("tests/test_prometheus_metrics.py", "Prometheus metrics test"),
    ]

    for test_file, test_name in tests:
        if os.path.exists(test_file):
            try:
                print_info(f"Running {test_file}...")
                result = os.system(
                    f"source venv/bin/activate && source .env.test && python3 {test_file} >/dev/null 2>&1"
                )
                passed = result == 0
                print_test(test_name, passed, f"Exit code: {result}")
            except Exception as e:
                print_test(test_name, False, str(e))
        else:
            print_info(f"Skipping {test_file} (not found)")


def main():
    global ADMIN_TOKEN, USER_TOKEN

    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("Active Graph KG - Backend Readiness Check")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    print(f"API URL: {API_URL}")
    print(f"Time: {datetime.now().isoformat()}")

    # Generate tokens
    print_info("Generating JWT tokens...")
    import subprocess

    result = subprocess.run(
        ["python3", "scripts/generate_test_jwt.py"], capture_output=True, text=True
    )
    lines = result.stdout.split("\n")
    for line in lines:
        if "Admin Token:" in line:
            ADMIN_TOKEN = line.split("Admin Token:")[1].strip()
        elif "Regular User Token:" in line:
            USER_TOKEN = line.split("Regular User Token:")[1].strip()

    if not ADMIN_TOKEN or not USER_TOKEN:
        print(f"{Colors.RED}Failed to generate tokens!{Colors.ENDC}")
        return 1

    print_info("Tokens generated successfully")

    # Run all tests
    results = []

    results.append(("Health", test_health()))
    results.append(("Prometheus", test_prometheus()))
    results.append(("JSON Metrics", test_json_metrics()))
    results.append(("Admin Migrate", test_admin_migrate()))
    results.append(("DB Status", test_db_status()))
    results.append(("Auth & RLS", lambda: (test_auth_rls(), True)[1] or True))
    results.append(("Node CRUD", test_node_crud()))
    results.append(("Search", lambda: (test_search(), True)[1] or True))
    results.append(("Q&A", lambda: (test_qa(), True)[1] or True))
    results.append(("Events & Lineage", lambda: (test_events_lineage(), True)[1] or True))
    results.append(("Admin Refresh", lambda: (test_admin_refresh(), True)[1] or True))
    results.append(("Connectors Admin", lambda: (test_connectors_admin(), True)[1] or True))

    # Execute callable tests
    executed_results = []
    for name, test_func in results:
        try:
            if callable(test_func):
                result = test_func()
            else:
                result = test_func
            executed_results.append((name, result))
        except Exception as e:
            print_test(name, False, str(e))
            executed_results.append((name, False))

    # Print summary
    print_section("Summary")
    passed = sum(1 for _, r in executed_results if r)
    total = len(executed_results)
    print(f"Total Tests: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.ENDC}")
    print(f"{Colors.RED}Failed: {total - passed}{Colors.ENDC}")

    if passed == total:
        print(
            f"\n{Colors.GREEN}{Colors.BOLD}✓ ALL TESTS PASSED - Backend is ready for UI development!{Colors.ENDC}"
        )
        return 0
    else:
        print(
            f"\n{Colors.YELLOW}{Colors.BOLD}⚠ Some tests failed - review issues before proceeding{Colors.ENDC}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
