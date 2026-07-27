"""API tests for POST /candidates/resolve/signal/candidate.

Exercises the Signal-specific translation layer that maps a sourced-candidate
payload onto the canonical resolve-or-create flow.
"""

from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier

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


# ---------------------------------------------------------------------------
# Canonical-freshness invariant (#Stage-1): "whatever is in Memory is the
# freshest representation of this person."
#   (a) a fuller payload always wins — enrichment fields are overwrite-latest
#   (b) an emptier payload never wins — {} / [] / missing blob is a no-op
# ---------------------------------------------------------------------------


def _get_candidate(candidate_id: str, tenant: str):
    from activekg.graph.candidate_repository import CandidateRepository

    repo = CandidateRepository(DSN)
    try:
        return repo.get_candidate(candidate_id, tenant_id=tenant)
    finally:
        repo.close()


def _signal_body(sig_id: str, url: str, crustdata: dict | None, headline: str | None = None):
    body = {
        "signal_candidate_id": sig_id,
        "source_record_type": "sourced_candidate",
        "linkedinUrl": url,
        "display_name": "Freshness Probe",
        "request_id": f"REQ-{uuid.uuid4()}",
        "tags": ["backend"],
    }
    if crustdata is not None:
        body["crustdata"] = crustdata
    if headline is not None:
        body["headline"] = headline
    return body


_BLOB_V1 = {
    "basic_profile": {
        "name": "Freshness Probe",
        "headline": "Backend Engineer at OldCo",
        "location": {"raw": "Bengaluru, India", "full_location": "Bengaluru, Karnataka, India"},
    },
    "skills": {"professional_network_skills": ["python", "postgresql"]},
    "experience": {
        "employment_details": {"current": [{"title": "Backend Engineer", "seniority_level": "Mid"}]}
    },
}

_BLOB_V2 = {
    "basic_profile": {
        "name": "Freshness Probe",
        "headline": "Senior Backend Engineer at NewCo",
        "location": {"raw": "Bengaluru, India", "full_location": "Bengaluru, Karnataka, India"},
    },
    "skills": {"professional_network_skills": ["python", "postgresql", "kubernetes"]},
    "experience": {
        "employment_details": {
            "current": [{"title": "Senior Backend Engineer", "seniority_level": "Senior"}]
        }
    },
}


def test_fresher_fuller_payload_always_wins(client: TestClient, tenant: str):
    """(a) Re-ingest with a fuller blob must destructively refresh enrichment fields."""
    url = f"https://www.linkedin.com/in/fresh-{uuid.uuid4().hex[:10]}/"
    sig_id = f"SIG-{uuid.uuid4()}"

    first = _post(client, {**_signal_body(sig_id, url, _BLOB_V1), "tenant_id": tenant})
    cid = first["candidate_id"]

    second = _post(client, {**_signal_body(sig_id, url, _BLOB_V2), "tenant_id": tenant})
    assert second["candidate_id"] == cid
    assert second["resolution_status"] == "matched"

    cand = _get_candidate(cid, tenant)
    assert cand is not None
    assert cand.profile == _BLOB_V2, "profile must hold the freshest blob"
    assert cand.skills == ["python", "postgresql", "kubernetes"]
    assert cand.seniority_level == "Senior"
    assert cand.headline == "Senior Backend Engineer at NewCo"


def test_emptier_payload_never_wins(client: TestClient, tenant: str):
    """(b) A blob-less / skill-less re-ingest must be a no-op on enrichment fields.

    The Signal handler defaults crustdata to {} and skills to [] — without the
    truthiness guard these WIPED canonical profile/skills.
    """
    url = f"https://www.linkedin.com/in/fresh-{uuid.uuid4().hex[:10]}/"
    sig_id = f"SIG-{uuid.uuid4()}"

    first = _post(client, {**_signal_body(sig_id, url, _BLOB_V2), "tenant_id": tenant})
    cid = first["candidate_id"]

    # Re-ingest with NO crustdata at all (e.g. a Serper-discovered duplicate).
    second = _post(client, {**_signal_body(sig_id, url, None), "tenant_id": tenant})
    assert second["candidate_id"] == cid

    cand = _get_candidate(cid, tenant)
    assert cand is not None
    assert cand.profile == _BLOB_V2, "empty blob must not wipe canonical profile"
    assert cand.skills == ["python", "postgresql", "kubernetes"], "empty skills must not wipe"
    assert cand.seniority_level == "Senior"
    assert cand.headline == "Senior Backend Engineer at NewCo"


def test_partial_payload_updates_only_present_fields(client: TestClient, tenant: str):
    """A blob with headline but no skills block refreshes headline, keeps skills."""
    url = f"https://www.linkedin.com/in/fresh-{uuid.uuid4().hex[:10]}/"
    sig_id = f"SIG-{uuid.uuid4()}"

    first = _post(client, {**_signal_body(sig_id, url, _BLOB_V2), "tenant_id": tenant})
    cid = first["candidate_id"]

    partial = {
        "basic_profile": {
            "name": "Freshness Probe",
            "headline": "Principal Engineer at NewerCo",
            "location": {"raw": "Bengaluru, India"},
        },
        # no "skills" key at all
    }
    _post(client, {**_signal_body(sig_id, url, partial), "tenant_id": tenant})

    cand = _get_candidate(cid, tenant)
    assert cand is not None
    assert cand.profile == partial, "non-empty blob still wins (freshest)"
    assert cand.skills == ["python", "postgresql", "kubernetes"], (
        "missing skills block must preserve existing skills"
    )
    assert cand.headline == "Principal Engineer at NewerCo"


@pytest.mark.parametrize("replay_delta", [timedelta(minutes=-5), timedelta(0)])
def test_older_or_equal_replay_preserves_newer_signal_observation(
    client: TestClient,
    tenant: str,
    replay_delta: timedelta,
):
    """A paid-batch replay must not make Memory older or extend its freshness."""
    url = f"https://www.linkedin.com/in/ordered-{uuid.uuid4().hex[:10]}/"
    sig_id = f"SIG-{uuid.uuid4()}"
    observed_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)

    first = _post(
        client,
        {
            **_signal_body(sig_id, url, _BLOB_V2),
            "profile_observed_at": observed_at.isoformat(),
            "tags": ["newer"],
            "tenant_id": tenant,
        },
    )
    replay = _post(
        client,
        {
            **_signal_body(sig_id, url, _BLOB_V1),
            "profile_observed_at": (observed_at + replay_delta).isoformat(),
            "tags": ["older"],
            "tenant_id": tenant,
        },
    )

    assert replay["candidate_id"] == first["candidate_id"]
    assert any("not newer" in warning for warning in replay["warnings"])

    candidate = _get_candidate(first["candidate_id"], tenant)
    assert candidate is not None
    assert candidate.profile == _BLOB_V2
    assert candidate.headline == "Senior Backend Engineer at NewCo"

    records = _get_source_records(first["candidate_id"], tenant)
    assert len(records) == 1
    assert records[0].fetched_at == observed_at
    assert records[0].payload["crustdata"] == _BLOB_V2
    assert records[0].job_tags == ["newer"]


def test_concurrent_signal_observations_serialize_and_keep_newest(
    client: TestClient,
    tenant: str,
):
    """Concurrent delivery order cannot decide which provider observation wins."""
    url = f"https://www.linkedin.com/in/concurrent-{uuid.uuid4().hex[:10]}/"
    sig_id = f"SIG-{uuid.uuid4()}"
    older_at = datetime(2026, 7, 27, 11, 0, tzinfo=timezone.utc)
    newer_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    barrier = Barrier(2)

    def post_observation(blob: dict, observed_at: datetime, tag: str):
        barrier.wait(timeout=5)
        return client.post(
            "/candidates/resolve/signal/candidate",
            json={
                **_signal_body(sig_id, url, blob),
                "profile_observed_at": observed_at.isoformat(),
                "tags": [tag],
                "tenant_id": tenant,
            },
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        older_future = executor.submit(post_observation, _BLOB_V1, older_at, "older")
        newer_future = executor.submit(post_observation, _BLOB_V2, newer_at, "newer")
        responses = [older_future.result(timeout=15), newer_future.result(timeout=15)]

    assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
    candidate_ids = {response.json()["candidate_id"] for response in responses}
    assert len(candidate_ids) == 1
    candidate_id = candidate_ids.pop()

    candidate = _get_candidate(candidate_id, tenant)
    assert candidate is not None
    assert candidate.profile == _BLOB_V2

    records = _get_source_records(candidate_id, tenant)
    assert len(records) == 1
    assert records[0].fetched_at == newer_at
    assert records[0].payload["crustdata"] == _BLOB_V2
    assert records[0].job_tags == ["newer"]
