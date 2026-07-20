"""API tests for POST /candidates/resolve/signal/candidate.

Exercises the Signal-specific translation layer that maps a sourced-candidate
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


def _post(client: TestClient, body: dict) -> dict:
    r = client.post("/candidates/resolve/signal/candidate", json=body)
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


def test_signal_payload_creates_candidate_with_translated_identifiers(
    client: TestClient, tenant: str
):
    sig_id = f"SIG-{uuid.uuid4()}"
    body = {
        "signal_candidate_id": sig_id,
        "source_record_type": "sourced_candidate",
        "linkedinUrl": "https://www.linkedin.com/in/signal-alice/",
        "identities": [
            {
                "platform": "github",
                "profileUrl": "https://github.com/SignalAlice",
                "confidence": 0.92,
                "bridgeTier": "tier_a",
            },
            {
                "platform": "twitter",
                "profileUrl": "https://twitter.com/signal_alice",
                "confidence": 0.7,
                "bridgeTier": "tier_b",
            },
        ],
        "display_name": "Signal Alice",
        "headline": "Staff Engineer",
        "identitySummary": "Verified across GitHub and LinkedIn.",
        "aiSummary": "Experienced backend engineer with Python and Go.",
        "rank": 3,
        "request_id": f"REQ-{uuid.uuid4()}",
        "external_job_id": "JOB-42",
        "sourcing_context": {"search": "backend python", "location": "SF"},
        "source_metadata": {"ingested_by": "signal-webhook"},
        "tenant_id": tenant,
    }
    resp = _post(client, body)
    assert resp["resolution_status"] == "created"
    cid = resp["candidate_id"]
    assert cid
    assert resp["source_record_id"] == sig_id

    idents = _list_identifiers(cid, tenant)
    types = {i.identifier_type for i in idents}
    assert "signal_candidate_id" in types
    assert "linkedin_url" in types
    assert "github_url" in types
    assert "twitter_url" in types

    # confidence + bridge_tier preserved on identity-derived identifiers.
    gh = next(i for i in idents if i.identifier_type == "github_url")
    assert gh.confidence == pytest.approx(0.92)
    assert gh.metadata.get("bridge_tier") == "tier_a"
    assert gh.metadata.get("signal_platform") == "github"

    records = _get_source_records(cid, tenant)
    assert len(records) == 1
    rec = records[0]
    assert rec.source == "signal"
    assert rec.source_record_type == "sourced_candidate"
    assert rec.source_record_id == sig_id
    assert rec.payload["request_id"] == body["request_id"]
    assert rec.payload["external_job_id"] == "JOB-42"
    assert rec.payload["rank"] == 3
    assert rec.payload["identitySummary"] == body["identitySummary"]
    assert rec.payload["aiSummary"] == body["aiSummary"]


def test_signal_payload_matches_existing_candidate_by_linkedin(client: TestClient, tenant: str):
    linkedin = "https://linkedin.com/in/signal-shared"
    first = _post(
        client,
        {
            "signal_candidate_id": f"SIG-{uuid.uuid4()}",
            "linkedinUrl": linkedin,
            "tenant_id": tenant,
        },
    )
    assert first["resolution_status"] == "created"

    # Different Signal id, same LinkedIn profile (different casing/trailing slash).
    second = _post(
        client,
        {
            "signal_candidate_id": f"SIG-{uuid.uuid4()}",
            "linkedinUrl": "https://www.linkedin.com/in/Signal-Shared/",
            "tenant_id": tenant,
        },
    )
    assert second["resolution_status"] == "matched"
    assert second["candidate_id"] == first["candidate_id"]


def test_signal_identities_create_multiple_identifiers(client: TestClient, tenant: str):
    resp = _post(
        client,
        {
            "signal_candidate_id": f"SIG-{uuid.uuid4()}",
            "identities": [
                {"platform": "linkedin", "profileUrl": "https://linkedin.com/in/multi-id"},
                {"platform": "github", "profileUrl": "https://github.com/multiid"},
                {"platform": "medium", "profileUrl": "https://medium.com/@multiid"},
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "created"
    types = {i.identifier_type for i in _list_identifiers(resp["candidate_id"], tenant)}
    assert {"signal_candidate_id", "linkedin_url", "github_url", "medium_url"} <= types


def test_signal_payload_rejects_unknown_record_type(client: TestClient, tenant: str):
    r = client.post(
        "/candidates/resolve/signal/candidate",
        json={
            "signal_candidate_id": f"SIG-{uuid.uuid4()}",
            "source_record_type": "not_a_real_type",
            "tenant_id": tenant,
        },
    )
    assert r.status_code == 400


def test_signal_payload_skips_invalid_identity_urls(client: TestClient, tenant: str):
    resp = _post(
        client,
        {
            "signal_candidate_id": f"SIG-{uuid.uuid4()}",
            "linkedinUrl": "not-a-real-linkedin-url",
            "identities": [
                {"platform": "github", "profileUrl": ""},
                {"platform": "medium", "profileUrl": "https://medium.com/not-a-profile-path"},
                {"platform": "github", "profileUrl": "https://github.com/validgh"},
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "created"
    types = {i.identifier_type for i in _list_identifiers(resp["candidate_id"], tenant)}
    assert "signal_candidate_id" in types
    assert "github_url" in types
    assert "linkedin_url" not in types
    assert "medium_url" not in types


def test_signal_profile_record_type_is_accepted(client: TestClient, tenant: str):
    resp = _post(
        client,
        {
            "signal_candidate_id": f"SIG-{uuid.uuid4()}",
            "source_record_type": "profile",
            "linkedinUrl": "https://linkedin.com/in/profile-record",
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "created"
    records = _get_source_records(resp["candidate_id"], tenant)
    assert records[0].source_record_type == "profile"


def test_signal_tags_are_accepted_and_stored(client: TestClient, tenant: str):
    sig_id = f"SIG-{uuid.uuid4()}"
    body = {
        "signal_candidate_id": sig_id,
        "linkedinUrl": "https://linkedin.com/in/tags-candidate",
        "tags": ["Python", "  Go  ", "python", "", "Distributed Systems"],
        "tenant_id": tenant,
    }
    resp = _post(client, body)
    assert resp["resolution_status"] == "created"

    records = _get_source_records(resp["candidate_id"], tenant)
    assert len(records) == 1
    rec = records[0]

    # Normalized: trimmed, lowercased, deduped, empties dropped.
    assert rec.job_tags == ["python", "go", "distributed systems"]


def test_signal_tags_preserved_in_payload(client: TestClient, tenant: str):
    raw_tags = ["Machine Learning", "PyTorch"]
    body = {
        "signal_candidate_id": f"SIG-{uuid.uuid4()}",
        "tags": raw_tags,
        "tenant_id": tenant,
    }
    resp = _post(client, body)
    assert resp["resolution_status"] == "created"

    records = _get_source_records(resp["candidate_id"], tenant)
    # Raw tags must be in the verbatim payload.
    assert records[0].payload.get("tags") == raw_tags


def test_signal_empty_tags_list_is_accepted(client: TestClient, tenant: str):
    body = {
        "signal_candidate_id": f"SIG-{uuid.uuid4()}",
        "tags": [],
        "tenant_id": tenant,
    }
    resp = _post(client, body)
    assert resp["resolution_status"] == "created"

    records = _get_source_records(resp["candidate_id"], tenant)
    assert records[0].job_tags == []
