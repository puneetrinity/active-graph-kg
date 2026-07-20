"""API tests for POST /candidates/resolve/vantahire/application.

Exercises the VantaHire-specific translation layer that maps an application
payload onto the canonical resolve-or-create flow.
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


def _post(client: TestClient, body: dict) -> dict:
    r = client.post("/candidates/resolve/vantahire/application", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def _list_identifiers(candidate_id: str, tenant: str) -> list:
    from activekg.graph.candidate_repository import CandidateRepository

    repo = CandidateRepository(DSN)
    try:
        return repo.list_identifiers(candidate_id, tenant_id=tenant)
    finally:
        repo.close()


def _get_source_records(candidate_id: str, tenant: str) -> list:
    from activekg.graph.candidate_repository import CandidateRepository

    repo = CandidateRepository(DSN)
    try:
        return repo.list_source_records(candidate_id, tenant_id=tenant)
    finally:
        repo.close()


def test_vantahire_application_creates_candidate_with_translated_identifiers(
    client: TestClient, tenant: str
):
    app_id = f"VH-APP-{uuid.uuid4()}"
    resume_id = f"VH-RES-{uuid.uuid4()}"
    body = {
        "application_id": app_id,
        "resume_id": resume_id,
        "job_id": "JOB-7",
        "org_id": "ORG-42",
        "applicant_name": "Alice Example",
        "email": "Alice@Example.com",
        "phone": "+1 (415) 555-2671",
        "linkedin_url": "https://www.linkedin.com/in/alice-example/",
        "github_url": "https://github.com/AliceExample",
        "medium_url": "https://medium.com/@alice",
        "other_links": ["https://alice.dev/portfolio"],
        "resume_gcp_url": "https://storage.googleapis.com/vh-resumes/alice.pdf",
        "skills": ["python", "postgres"],
        "source_metadata": {"ingested_by": "vantahire-webhook"},
        "tenant_id": tenant,
    }
    resp = _post(client, body)
    assert resp["resolution_status"] == "created"
    cid = resp["candidate_id"]
    assert cid
    assert resp["source_record_id"] == app_id

    types = {i.identifier_type for i in _list_identifiers(cid, tenant)}
    assert "vantahire_application_id" in types
    assert "vantahire_resume_id" in types
    assert "linkedin_url" in types
    assert "github_url" in types
    assert "medium_url" in types
    assert "email" in types
    assert "phone" in types
    assert "website_url" in types  # other_links fallback

    records = _get_source_records(cid, tenant)
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "vantahire"
    assert rec.source_record_type == "application"
    assert rec.source_record_id == app_id
    assert rec.payload["job_id"] == "JOB-7"
    assert rec.payload["org_id"] == "ORG-42"
    assert rec.payload["resume_id"] == resume_id
    assert rec.payload["resume_gcp_url"].endswith("alice.pdf")


def test_vantahire_application_matches_existing_by_linkedin(client: TestClient, tenant: str):
    linkedin = "https://linkedin.com/in/shared-linkedin-handle"
    first = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "email": f"first-{uuid.uuid4().hex[:6]}@example.com",
            "linkedin_url": linkedin,
            "tenant_id": tenant,
        },
    )
    assert first["resolution_status"] == "created"

    # Different application id, same LinkedIn — must match.
    second = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "linkedin_url": "https://www.linkedin.com/in/Shared-LinkedIn-Handle/",
            "tenant_id": tenant,
        },
    )
    assert second["resolution_status"] == "matched"
    assert second["candidate_id"] == first["candidate_id"]
    assert second["matched_identifier"]["identifier_type"] in {
        "linkedin_url",
        "vantahire_application_id",
    }


def test_vantahire_application_matches_existing_by_github(client: TestClient, tenant: str):
    first = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "github_url": "https://github.com/sharedgh",
            "tenant_id": tenant,
        },
    )
    second = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "github_url": "https://github.com/SharedGH",
            "tenant_id": tenant,
        },
    )
    assert second["resolution_status"] == "matched"
    assert second["candidate_id"] == first["candidate_id"]


def test_vantahire_application_matches_existing_by_email(client: TestClient, tenant: str):
    email = f"match-{uuid.uuid4().hex[:6]}@example.com"
    first = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "email": email,
            "tenant_id": tenant,
        },
    )
    second = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "email": email.upper(),
            "tenant_id": tenant,
        },
    )
    assert second["resolution_status"] == "matched"
    assert second["candidate_id"] == first["candidate_id"]


def test_vantahire_application_matches_existing_by_medium(client: TestClient, tenant: str):
    first = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "medium_url": "https://medium.com/@mediumhandle",
            "tenant_id": tenant,
        },
    )
    second = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "medium_url": "https://mediumhandle.medium.com",
            "tenant_id": tenant,
        },
    )
    assert second["resolution_status"] == "matched"
    assert second["candidate_id"] == first["candidate_id"]


def test_vantahire_application_matches_existing_by_phone(client: TestClient, tenant: str):
    phone = "+14155552673"
    first = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "phone": phone,
            "tenant_id": tenant,
        },
    )
    second = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "phone": "+1 (415) 555-2673",
            "tenant_id": tenant,
        },
    )
    assert second["resolution_status"] == "matched"
    assert second["candidate_id"] == first["candidate_id"]


def test_vantahire_application_skips_invalid_optional_links(client: TestClient, tenant: str):
    resp = _post(
        client,
        {
            "application_id": f"VH-APP-{uuid.uuid4()}",
            "email": f"keep-{uuid.uuid4().hex[:6]}@example.com",
            "linkedin_url": "not-a-real-linkedin-url",
            "github_url": "",
            "medium_url": "https://medium.com/not-a-profile-path",
            "other_links": ["::::broken::::"],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "created"
    types = {i.identifier_type for i in _list_identifiers(resp["candidate_id"], tenant)}
    # bad linkedin/medium/other_links dropped; email + vantahire_application_id kept.
    assert "email" in types
    assert "vantahire_application_id" in types
    assert "linkedin_url" not in types
    assert "medium_url" not in types


def test_vantahire_application_requires_some_usable_identifier(client: TestClient, tenant: str):
    # Empty application_id → invalid opaque id and nothing else usable.
    r = client.post(
        "/candidates/resolve/vantahire/application",
        json={
            "application_id": "   ",
            "linkedin_url": "not-real",
            "tenant_id": tenant,
        },
    )
    assert r.status_code == 400


def test_vantahire_application_preserves_payload_in_source_record(client: TestClient, tenant: str):
    app_id = f"VH-APP-{uuid.uuid4()}"
    body = {
        "application_id": app_id,
        "resume_id": "RES-99",
        "job_id": "JOB-99",
        "org_id": "ORG-99",
        "email": f"preserve-{uuid.uuid4().hex[:6]}@example.com",
        "skills": ["go", "k8s"],
        "source_metadata": {"webhook_id": "WH-1"},
        "tenant_id": tenant,
    }
    resp = _post(client, body)
    records = _get_source_records(resp["candidate_id"], tenant)
    assert len(records) == 1
    payload = records[0].payload
    for key in ("application_id", "resume_id", "job_id", "org_id", "skills"):
        assert payload[key] == body[key]
