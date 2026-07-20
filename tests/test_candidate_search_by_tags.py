"""API tests for POST /candidates/search/by-tags."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient

MIGRATION_012 = (
    Path(__file__).resolve().parents[1] / "db" / "migrations" / "012_candidate_identity.sql"
)
MIGRATION_013 = (
    Path(__file__).resolve().parents[1] / "db" / "migrations" / "013_vantahire_provenance.sql"
)
MIGRATION_014 = (
    Path(__file__).resolve().parents[1] / "db" / "migrations" / "014_signal_job_tags.sql"
)

DSN = os.getenv("ACTIVEKG_DSN", "postgresql://activekg:activekg@localhost:5432/activekg")


def _db_reachable() -> bool:
    try:
        with psycopg.connect(DSN, connect_timeout=2):
            return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_reachable(), reason="ACTIVEKG_DSN not reachable; skipping DB-backed tests"
)


@pytest.fixture(scope="module", autouse=True)
def _migrated_db() -> None:
    with psycopg.connect(DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(MIGRATION_012.read_text())
            cur.execute(MIGRATION_013.read_text())
            cur.execute(MIGRATION_014.read_text())


@pytest.fixture(scope="module")
def client() -> TestClient:
    os.environ["JWT_ENABLED"] = "false"
    from activekg.api.main import app

    return TestClient(app)


@pytest.fixture()
def tenant() -> str:
    return f"test-{uuid.uuid4()}"


def _ingest(client: TestClient, *, sig_id: str, tags: list[str], tenant: str) -> dict:
    r = client.post(
        "/candidates/resolve/signal/candidate",
        json={"signal_candidate_id": sig_id, "tags": tags, "tenant_id": tenant},
    )
    assert r.status_code == 200, r.text
    return r.json()


def _search(client: TestClient, *, tags: list[str], tenant: str, limit: int = 100) -> dict:
    r = client.post(
        "/candidates/search/by-tags",
        json={"tags": tags, "tenant_id": tenant, "limit": limit},
    )
    assert r.status_code == 200, r.text
    return r.json()


def test_search_returns_matching_candidate(client: TestClient, tenant: str):
    sig_id = f"SIG-{uuid.uuid4()}"
    ingest = _ingest(client, sig_id=sig_id, tags=["python", "go", "kafka"], tenant=tenant)
    cid = ingest["candidate_id"]

    resp = _search(client, tags=["python", "go", "kafka"], tenant=tenant)

    assert resp["query_tags"] == ["python", "go", "kafka"]
    assert resp["total"] >= 1
    match = next((r for r in resp["results"] if r["candidate_id"] == cid), None)
    assert match is not None
    assert match["overlap_count"] == 3
    assert match["overlap_ratio"] == pytest.approx(1.0)
    assert set(match["matched_tags"]) == {"python", "go", "kafka"}
    assert match["signal_candidate_id"] == sig_id


def test_search_normalizes_query_tags(client: TestClient, tenant: str):
    sig_id = f"SIG-{uuid.uuid4()}"
    ingest = _ingest(client, sig_id=sig_id, tags=["Python", "Go"], tenant=tenant)
    cid = ingest["candidate_id"]

    # Query with different casing and extra whitespace.
    resp = _search(client, tags=["  PYTHON  ", "go"], tenant=tenant)
    assert any(r["candidate_id"] == cid for r in resp["results"])
    assert resp["query_tags"] == ["python", "go"]


def test_search_70_percent_threshold_included(client: TestClient, tenant: str):
    stored_tags = [f"tag{i}" for i in range(10)]
    sig_id = f"SIG-{uuid.uuid4()}"
    ingest = _ingest(client, sig_id=sig_id, tags=stored_tags, tenant=tenant)
    cid = ingest["candidate_id"]

    # 7 out of 10 query tags match → 70 % → should be included.
    query = stored_tags[:7] + ["extra1", "extra2", "extra3"]
    resp = _search(client, tags=query, tenant=tenant)
    assert any(r["candidate_id"] == cid for r in resp["results"])


def test_search_below_threshold_excluded(client: TestClient, tenant: str):
    stored_tags = [f"unique-{uuid.uuid4()}-{i}" for i in range(10)]
    sig_id = f"SIG-{uuid.uuid4()}"
    ingest = _ingest(client, sig_id=sig_id, tags=stored_tags, tenant=tenant)
    cid = ingest["candidate_id"]

    # 6 out of 10 query tags match → 60 % → below threshold → excluded.
    query = stored_tags[:6] + [f"missing-{uuid.uuid4()}" for _ in range(4)]
    resp = _search(client, tags=query, tenant=tenant)
    assert not any(r["candidate_id"] == cid for r in resp["results"])


def test_search_empty_tags_returns_empty(client: TestClient, tenant: str):
    resp = _search(client, tags=[], tenant=tenant)
    assert resp["results"] == []
    assert resp["total"] == 0


def test_search_respects_limit(client: TestClient, tenant: str):
    shared_tag = f"limit-tag-{uuid.uuid4()}"
    for _ in range(5):
        _ingest(client, sig_id=f"SIG-{uuid.uuid4()}", tags=[shared_tag], tenant=tenant)

    resp = _search(client, tags=[shared_tag], tenant=tenant, limit=3)
    assert len(resp["results"]) <= 3


def test_search_tenant_isolation(client: TestClient):
    tenant_a = f"tenant-a-{uuid.uuid4()}"
    tenant_b = f"tenant-b-{uuid.uuid4()}"
    shared_tag = f"iso-tag-{uuid.uuid4()}"

    ingest = _ingest(client, sig_id=f"SIG-{uuid.uuid4()}", tags=[shared_tag], tenant=tenant_a)
    cid = ingest["candidate_id"]

    # Searching from tenant_b should not see tenant_a's candidate.
    resp = _search(client, tags=[shared_tag], tenant=tenant_b)
    assert not any(r["candidate_id"] == cid for r in resp["results"])


def test_truncation_contract_reports_total_matched(client: TestClient, tenant: str):
    """Slice (b) contract: the caller must be able to detect result truncation."""
    tags = ["python", "django", "backend"]
    for i in range(3):
        _ingest(client, sig_id=f"trunc-{i}-{uuid.uuid4().hex[:6]}", tags=tags, tenant=tenant)

    body = _search(client, tags=tags, tenant=tenant, limit=2)
    assert body["total"] == 2
    assert body["total_matched"] == 3
    assert body["truncated"] is True
    assert body["applied_limit"] == 2

    full = _search(client, tags=tags, tenant=tenant, limit=100)
    assert full["total"] == 3
    assert full["total_matched"] == 3
    assert full["truncated"] is False


def test_limit_above_legacy_cap_is_accepted_and_clamped(client: TestClient, tenant: str):
    """Requests above the old le=100 model cap must not 422; the server clamps
    to ACTIVEKG_TAG_SEARCH_MAX_LIMIT and reports the applied limit."""
    tags = ["golang", "kubernetes"]
    _ingest(client, sig_id=f"cap-{uuid.uuid4().hex[:6]}", tags=tags, tenant=tenant)

    body = _search(client, tags=tags, tenant=tenant, limit=300)
    assert body["applied_limit"] == 300  # default server max is 500
    assert body["total"] == 1
    assert body["truncated"] is False
