"""Provenance tests for the shared/global candidate identity system.

Verifies that:
- Canonical candidates remain globally shared (no org ownership on the candidate).
- VantaHire source records reliably preserve org/job/recruiter provenance in
  structured columns (not just JSONB payload).
- A single canonical candidate can carry VantaHire records from multiple orgs.
- A candidate discovered by both Signal and VantaHire shares one canonical record;
  only the VantaHire source record carries org provenance.
- CandidateRepository provenance query methods return the right candidates.
"""

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


@pytest.fixture(scope="module")
def client() -> TestClient:
    os.environ["JWT_ENABLED"] = "false"
    from activekg.api.main import app

    return TestClient(app)


@pytest.fixture()
def tenant() -> str:
    return f"test-{uuid.uuid4()}"


def _vh_post(client: TestClient, body: dict) -> dict:
    r = client.post("/candidates/resolve/vantahire/application", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _signal_post(client: TestClient, body: dict) -> dict:
    r = client.post("/candidates/resolve/signal/candidate", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _get_repo():
    from activekg.graph.candidate_repository import CandidateRepository

    return CandidateRepository(DSN)


# ---------------------------------------------------------------------------
# VantaHire provenance fields preserved in structured source record columns
# ---------------------------------------------------------------------------


def test_vantahire_provenance_fields_in_structured_columns(client: TestClient, tenant: str):
    app_id = f"VH-APP-{uuid.uuid4()}"
    recruiter_id = f"REC-{uuid.uuid4()}"
    uploader_id = f"USER-{uuid.uuid4()}"

    resp = _vh_post(
        client,
        {
            "application_id": app_id,
            "resume_id": f"RES-{uuid.uuid4()}",
            "job_id": "JOB-1",
            "org_id": "ORG-A",
            "effective_recruiter_id": recruiter_id,
            "created_by_user_id": uploader_id,
            "resume_source": "linkedin",
            "email": f"prov-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "created"
    cid = resp["candidate_id"]

    repo = _get_repo()
    try:
        records = repo.list_source_records(cid, tenant_id=tenant)
        assert len(records) == 1
        rec = records[0]

        # Structured columns populated
        assert rec.org_id == "ORG-A"
        assert rec.job_id == "JOB-1"
        assert rec.effective_recruiter_id == recruiter_id
        assert rec.created_by_user_id == uploader_id
        assert rec.resume_source == "linkedin"

        # JSONB payload also carries all provenance fields for raw consumers
        assert rec.payload["org_id"] == "ORG-A"
        assert rec.payload["job_id"] == "JOB-1"
        assert rec.payload["effective_recruiter_id"] == recruiter_id
        assert rec.payload["created_by_user_id"] == uploader_id
        assert rec.payload["resume_source"] == "linkedin"
        assert rec.payload["source"] == "vantahire"
        assert rec.payload["source_record_type"] == "application"
    finally:
        repo.close()


def test_org_id_not_on_canonical_candidate(client: TestClient, tenant: str):
    """org_id must NOT appear on the canonical candidate's props — it belongs on
    the source record only so the canonical candidate stays globally shared."""
    from activekg.graph.candidate_repository import CandidateRepository

    resp = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": "ORG-SHOULD-NOT-LEAK",
            "job_id": "JOB-SHOULD-NOT-LEAK",
            "email": f"noleak-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "created"

    repo = CandidateRepository(DSN)
    try:
        candidate = repo.get_candidate(resp["candidate_id"])
        assert candidate is not None
        assert "org_id" not in candidate.props
        assert "job_id" not in candidate.props
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Multi-org: one canonical candidate with VantaHire records from two orgs
# ---------------------------------------------------------------------------


def test_multi_org_same_canonical_candidate(client: TestClient, tenant: str):
    """A candidate who applies via two different orgs shares one canonical identity;
    each application produces a separate source record with its own org provenance."""
    shared_email = f"multi-org-{uuid.uuid4().hex[:6]}@example.com"

    resp_a = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": "ORG-ALPHA",
            "job_id": "JOB-ALPHA-1",
            "effective_recruiter_id": "REC-ALPHA",
            "email": shared_email,
            "tenant_id": tenant,
        },
    )
    assert resp_a["resolution_status"] == "created"
    cid = resp_a["candidate_id"]

    resp_b = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": "ORG-BETA",
            "job_id": "JOB-BETA-2",
            "effective_recruiter_id": "REC-BETA",
            "email": shared_email,
            "tenant_id": tenant,
        },
    )
    assert resp_b["resolution_status"] == "matched"
    assert resp_b["candidate_id"] == cid  # same canonical candidate

    repo = _get_repo()
    try:
        records = repo.list_source_records(cid, source="vantahire", tenant_id=tenant)
        assert len(records) == 2

        org_ids = {r.org_id for r in records}
        assert "ORG-ALPHA" in org_ids
        assert "ORG-BETA" in org_ids

        recruiter_ids = {r.effective_recruiter_id for r in records}
        assert "REC-ALPHA" in recruiter_ids
        assert "REC-BETA" in recruiter_ids
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Shared candidate matched from both Signal and VantaHire
# ---------------------------------------------------------------------------


def test_shared_canonical_candidate_signal_and_vantahire(client: TestClient, tenant: str):
    """A candidate sourced by Signal then applied through VantaHire resolves to one
    canonical candidate.  The Signal source record has no org provenance; the
    VantaHire source record does."""
    linkedin = f"https://linkedin.com/in/shared-{uuid.uuid4().hex[:8]}"
    signal_id = f"SIG-{uuid.uuid4()}"

    sig_resp = _signal_post(
        client,
        {
            "signal_candidate_id": signal_id,
            "linkedinUrl": linkedin,
            "display_name": "Sam Shared",
            "tenant_id": tenant,
        },
    )
    assert sig_resp["resolution_status"] == "created"
    cid = sig_resp["candidate_id"]

    vh_resp = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "linkedin_url": linkedin,
            "org_id": "ORG-VH",
            "job_id": "JOB-VH-1",
            "effective_recruiter_id": "REC-VH-1",
            "email": f"shared-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )
    assert vh_resp["resolution_status"] == "matched"
    assert vh_resp["candidate_id"] == cid

    repo = _get_repo()
    try:
        all_records = repo.list_source_records(cid, tenant_id=tenant)
        sources = {r.source for r in all_records}
        assert "signal" in sources
        assert "vantahire" in sources

        sig_record = next(r for r in all_records if r.source == "signal")
        assert sig_record.org_id is None
        assert sig_record.effective_recruiter_id is None

        vh_record = next(r for r in all_records if r.source == "vantahire")
        assert vh_record.org_id == "ORG-VH"
        assert vh_record.effective_recruiter_id == "REC-VH-1"
    finally:
        repo.close()


# ---------------------------------------------------------------------------
# Provenance query methods: find by org / recruiter / uploader
# ---------------------------------------------------------------------------


def test_find_candidates_by_vantahire_org(client: TestClient, tenant: str):
    org_id = f"ORG-QUERY-{uuid.uuid4().hex[:6]}"
    other_org_id = f"ORG-OTHER-{uuid.uuid4().hex[:6]}"

    r1 = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": org_id,
            "email": f"q1-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )
    r2 = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": org_id,
            "email": f"q2-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )
    _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": other_org_id,
            "email": f"q3-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )

    repo = _get_repo()
    try:
        found = repo.find_candidates_by_vantahire_org(org_id, tenant_id=tenant)
        found_ids = {c.candidate_id for c in found}
        assert r1["candidate_id"] in found_ids
        assert r2["candidate_id"] in found_ids
        # candidate from other_org must not appear
        for c in found:
            records = repo.list_source_records(c.candidate_id, tenant_id=tenant)
            assert any(rec.org_id == org_id for rec in records)
    finally:
        repo.close()


def test_find_candidates_by_vantahire_recruiter(client: TestClient, tenant: str):
    recruiter_id = f"REC-{uuid.uuid4().hex[:8]}"

    r1 = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": "ORG-X",
            "effective_recruiter_id": recruiter_id,
            "email": f"rec1-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )
    _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": "ORG-X",
            "effective_recruiter_id": "REC-OTHER",
            "email": f"rec2-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )

    repo = _get_repo()
    try:
        found = repo.find_candidates_by_vantahire_recruiter(recruiter_id, tenant_id=tenant)
        found_ids = {c.candidate_id for c in found}
        assert r1["candidate_id"] in found_ids
    finally:
        repo.close()


def test_find_candidates_by_vantahire_uploader(client: TestClient, tenant: str):
    user_id = f"USER-{uuid.uuid4().hex[:8]}"

    r1 = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": "ORG-Y",
            "created_by_user_id": user_id,
            "email": f"up1-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )
    _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": "ORG-Y",
            "created_by_user_id": "USER-OTHER",
            "email": f"up2-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant,
        },
    )

    repo = _get_repo()
    try:
        found = repo.find_candidates_by_vantahire_uploader(user_id, tenant_id=tenant)
        found_ids = {c.candidate_id for c in found}
        assert r1["candidate_id"] in found_ids
    finally:
        repo.close()


def test_provenance_query_tenant_isolation(client: TestClient):
    """find_candidates_by_vantahire_org must not cross tenant boundaries."""
    tenant_a = f"ta-{uuid.uuid4()}"
    tenant_b = f"tb-{uuid.uuid4()}"
    shared_org = f"ORG-SHARED-{uuid.uuid4().hex[:6]}"

    ra = _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": shared_org,
            "email": f"ta-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant_a,
        },
    )
    _vh_post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "org_id": shared_org,
            "email": f"tb-{uuid.uuid4().hex[:6]}@example.com",
            "tenant_id": tenant_b,
        },
    )

    repo = _get_repo()
    try:
        found_a = repo.find_candidates_by_vantahire_org(shared_org, tenant_id=tenant_a)
        assert all(c.candidate_id == ra["candidate_id"] for c in found_a)

        found_b = repo.find_candidates_by_vantahire_org(shared_org, tenant_id=tenant_b)
        assert ra["candidate_id"] not in {c.candidate_id for c in found_b}
    finally:
        repo.close()
