"""API tests for POST /candidates/resolve.

Exercise the resolve-or-create endpoint against a live Postgres (same pattern
as ``test_candidate_repository.py``). Skipped when the database at
``ACTIVEKG_DSN`` isn't reachable.
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
    r = client.post("/candidates/resolve", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_resolve_creates_new_candidate(client: TestClient, tenant: str):
    body = _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": "newperson@example.com"},
            ],
            "profile": {"display_name": "New Person"},
            "payload": {"role": "swe"},
            "tenant_id": tenant,
        },
    )
    assert body["resolution_status"] == "created"
    assert body["candidate_id"]
    assert body["matched_identifier"] is None
    assert body["source_record_id"]


def test_resolve_matches_existing_by_identifier(client: TestClient, tenant: str):
    shared_email = f"shared-{uuid.uuid4().hex[:8]}@example.com"
    first = _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": shared_email},
                {"identifier_type": "vantahire_application_id", "value": "VH-APP-1"},
            ],
            "tenant_id": tenant,
        },
    )
    assert first["resolution_status"] == "created"

    second = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                # Different casing — normalization should still match.
                {"identifier_type": "email", "value": shared_email.upper()},
                {"identifier_type": "signal_candidate_id", "value": "SIG-99"},
            ],
            "tenant_id": tenant,
        },
    )
    assert second["resolution_status"] == "matched"
    assert second["candidate_id"] == first["candidate_id"]
    assert second["matched_identifier"]["identifier_type"] == "email"
    assert second["matched_identifier"]["value_normalized"] == shared_email.lower()


def test_resolve_is_idempotent_for_repeated_source_record(client: TestClient, tenant: str):
    record_id = f"VH-{uuid.uuid4()}"
    body = {
        "source": "vantahire",
        "source_record_type": "application",
        "source_record_id": record_id,
        "identifiers": [
            {"identifier_type": "email", "value": f"idem-{uuid.uuid4().hex[:6]}@example.com"},
        ],
        "tenant_id": tenant,
    }
    first = _post(client, body)
    second = _post(client, body)
    assert first["candidate_id"] == second["candidate_id"]
    assert second["resolution_status"] == "matched"


def test_resolve_flags_review_required_on_conflicting_identifiers(client: TestClient, tenant: str):
    # Seed two distinct candidates.
    a = _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-A-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": f"a-{uuid.uuid4().hex[:6]}@example.com"},
            ],
            "tenant_id": tenant,
        },
    )
    b = _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-B-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "linkedin_url", "value": "https://linkedin.com/in/person-b"},
            ],
            "tenant_id": tenant,
        },
    )
    assert a["candidate_id"] != b["candidate_id"]

    # A third record references identifiers from BOTH — must flag for review,
    # not silently merge the two canonical candidates.
    conflict = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": _first_email_for(a, client, tenant)},
                {"identifier_type": "linkedin_url", "value": "https://linkedin.com/in/person-b"},
            ],
            "tenant_id": tenant,
        },
    )
    assert conflict["resolution_status"] == "review_required"
    assert conflict["candidate_id"] is None
    assert conflict["source_record_id"] is None
    assert conflict["conflicts"] and len(conflict["conflicts"]) == 2


def _first_email_for(resolve_response: dict, client: TestClient, tenant: str) -> str:
    """Look up the raw email identifier attached to a candidate so the
    conflict test can reuse it without re-plumbing fixtures."""
    from activekg.graph.candidate_repository import CandidateRepository

    repo = CandidateRepository(DSN)
    try:
        idents = repo.list_identifiers(resolve_response["candidate_id"], tenant_id=tenant)
    finally:
        repo.close()
    for i in idents:
        if i.identifier_type == "email":
            return i.value_raw or i.value_normalized
    raise AssertionError("expected an email identifier on the seed candidate")


def test_response_shape_on_created(client: TestClient, tenant: str):
    resp = _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": f"shape-{uuid.uuid4().hex[:6]}@ex.com"},
                {
                    "identifier_type": "linkedin_url",
                    "value": f"https://linkedin.com/in/shape-{uuid.uuid4().hex[:6]}",
                },
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "created"
    assert resp["matched_identifier"] is None
    attached_types = {a["identifier_type"] for a in resp["attached_identifiers"]}
    assert attached_types == {"email", "linkedin_url"}
    for a in resp["attached_identifiers"]:
        assert a["value_normalized"]
    assert resp["skipped_identifiers"] == []
    assert resp["conflicts"] == []
    assert resp["warnings"] == []
    assert resp["source_record_id"]


def test_response_shape_on_matched(client: TestClient, tenant: str):
    email = f"match-{uuid.uuid4().hex[:6]}@ex.com"
    _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-{uuid.uuid4()}",
            "identifiers": [{"identifier_type": "email", "value": email}],
            "profile": {"display_name": "Canonical Name"},
            "tenant_id": tenant,
        },
    )
    resp = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": email.upper()},
                {"identifier_type": "signal_candidate_id", "value": "SIG-ABC"},
            ],
            "profile": {"display_name": "Upstream Drift Name"},
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "matched"
    assert resp["matched_identifier"]["identifier_type"] == "email"
    attached_types = {a["identifier_type"] for a in resp["attached_identifiers"]}
    assert "signal_candidate_id" in attached_types
    assert resp["conflicts"] == []
    assert any("display_name" in w for w in resp["warnings"])


def test_response_shape_on_duplicate_identifiers_in_request(client: TestClient, tenant: str):
    email = f"dup-{uuid.uuid4().hex[:6]}@ex.com"
    resp = _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": email},
                {"identifier_type": "email", "value": email.upper()},
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "created"
    assert len(resp["attached_identifiers"]) == 1
    assert len(resp["skipped_identifiers"]) == 1
    assert resp["skipped_identifiers"][0]["reason"] == "duplicate identifier in request"


def test_response_shape_on_conflict(client: TestClient, tenant: str):
    a = _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-A-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": f"a-{uuid.uuid4().hex[:6]}@ex.com"},
            ],
            "tenant_id": tenant,
        },
    )
    b_linkedin = f"https://linkedin.com/in/b-{uuid.uuid4().hex[:6]}"
    b = _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-B-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "linkedin_url", "value": b_linkedin},
            ],
            "tenant_id": tenant,
        },
    )
    resp = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": _first_email_for(a, client, tenant)},
                {"identifier_type": "linkedin_url", "value": b_linkedin},
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "review_required"
    assert resp["candidate_id"] is None
    assert resp["source_record_id"] is None
    assert len(resp["conflicts"]) == 2
    owners = {c["candidate_id"] for c in resp["conflicts"]}
    assert owners == {a["candidate_id"], b["candidate_id"]}
    for c in resp["conflicts"]:
        assert c["identifier_type"] and c["value_normalized"] and c["reason"]
    assert resp["attached_identifiers"] == []
    assert resp["warnings"]


def test_resolve_rejects_invalid_identifier(client: TestClient, tenant: str):
    r = client.post(
        "/candidates/resolve",
        json={
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": "not-an-email"},
            ],
            "tenant_id": tenant,
        },
    )
    assert r.status_code == 400
    assert "invalid email" in r.text.lower()


def test_resolve_rejects_unknown_identifier_type(client: TestClient, tenant: str):
    r = client.post(
        "/candidates/resolve",
        json={
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "passport_number", "value": "ABC123"},
            ],
            "tenant_id": tenant,
        },
    )
    assert r.status_code == 400


def test_resolve_requires_at_least_one_identifier(client: TestClient, tenant: str):
    r = client.post(
        "/candidates/resolve",
        json={
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"VH-{uuid.uuid4()}",
            "identifiers": [],
            "tenant_id": tenant,
        },
    )
    assert r.status_code == 400


# ------------------------------------------------------------------
# Strong-signal mismatch tests
# ------------------------------------------------------------------


def _seed_candidate(client: TestClient, tenant: str, identifiers: list[dict]) -> dict:
    """Create a canonical candidate with the given identifiers."""
    return _post(
        client,
        {
            "source": "vantahire",
            "source_record_type": "application",
            "source_record_id": f"SEED-{uuid.uuid4()}",
            "identifiers": identifiers,
            "tenant_id": tenant,
        },
    )


def test_strong_signal_one_match_no_mismatch_accepted(client: TestClient, tenant: str):
    """One strong-signal match and no mismatch → accepted as matched."""
    email = f"ss-ok-{uuid.uuid4().hex[:6]}@example.com"
    seed = _seed_candidate(client, tenant, [{"identifier_type": "email", "value": email}])
    assert seed["resolution_status"] == "created"

    resp = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [{"identifier_type": "email", "value": email}],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "matched"
    assert resp["candidate_id"] == seed["candidate_id"]
    assert resp["conflicts"] == []


def test_strong_signal_one_match_one_mismatch_review_required(client: TestClient, tenant: str):
    """One strong-signal match + one strong-signal mismatch → review_required."""
    email = f"ss-1m1c-{uuid.uuid4().hex[:6]}@example.com"
    canonical_linkedin = f"https://linkedin.com/in/canonical-{uuid.uuid4().hex[:6]}"
    seed = _seed_candidate(
        client,
        tenant,
        [
            {"identifier_type": "email", "value": email},
            {"identifier_type": "linkedin_url", "value": canonical_linkedin},
        ],
    )
    assert seed["resolution_status"] == "created"

    conflicting_linkedin = f"https://linkedin.com/in/different-{uuid.uuid4().hex[:6]}"
    resp = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": email},
                {"identifier_type": "linkedin_url", "value": conflicting_linkedin},
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "review_required"
    assert resp["candidate_id"] is None
    assert resp["source_record_id"] is None
    conflict_types = {c["identifier_type"] for c in resp["conflicts"]}
    assert "linkedin_url" in conflict_types
    assert any("mismatch" in c["reason"].lower() for c in resp["conflicts"])


def test_strong_signal_two_matches_one_mismatch_accepted(client: TestClient, tenant: str):
    """Two strong-signal matches + one mismatch → accepted (tolerated), mismatch not attached."""
    email = f"ss-2m1c-{uuid.uuid4().hex[:6]}@example.com"
    gh = f"https://github.com/user2m1c-{uuid.uuid4().hex[:6]}"
    canonical_linkedin = f"https://linkedin.com/in/user2m1c-{uuid.uuid4().hex[:6]}"
    seed = _seed_candidate(
        client,
        tenant,
        [
            {"identifier_type": "email", "value": email},
            {"identifier_type": "github_url", "value": gh},
            {"identifier_type": "linkedin_url", "value": canonical_linkedin},
        ],
    )
    assert seed["resolution_status"] == "created"

    conflicting_linkedin = f"https://linkedin.com/in/wrong-{uuid.uuid4().hex[:6]}"
    resp = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": email},
                {"identifier_type": "github_url", "value": gh},
                {"identifier_type": "linkedin_url", "value": conflicting_linkedin},
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "matched"
    assert resp["candidate_id"] == seed["candidate_id"]
    assert resp["conflicts"] == []
    # The mismatching linkedin should not be attached
    attached_types = {a["identifier_type"] for a in resp["attached_identifiers"]}
    assert "linkedin_url" not in attached_types
    # It should appear in skipped with a mismatch reason
    skipped_types = {
        s["identifier_type"] for s in resp["skipped_identifiers"] if s.get("identifier_type")
    }
    assert "linkedin_url" in skipped_types
    assert any("mismatch" in w.lower() for w in resp["warnings"])


def test_strong_signal_two_matches_two_mismatches_review_required(client: TestClient, tenant: str):
    """Two strong-signal matches + two mismatches → conservative review_required."""
    email = f"ss-2m2c-{uuid.uuid4().hex[:6]}@example.com"
    gh = f"https://github.com/user2m2c-{uuid.uuid4().hex[:6]}"
    canonical_linkedin = f"https://linkedin.com/in/user2m2c-{uuid.uuid4().hex[:6]}"
    canonical_medium = f"https://medium.com/@user2m2c-{uuid.uuid4().hex[:6]}"
    seed = _seed_candidate(
        client,
        tenant,
        [
            {"identifier_type": "email", "value": email},
            {"identifier_type": "github_url", "value": gh},
            {"identifier_type": "linkedin_url", "value": canonical_linkedin},
            {"identifier_type": "medium_url", "value": canonical_medium},
        ],
    )
    assert seed["resolution_status"] == "created"

    resp = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": email},
                {"identifier_type": "github_url", "value": gh},
                {
                    "identifier_type": "linkedin_url",
                    "value": f"https://linkedin.com/in/wrong-{uuid.uuid4().hex[:6]}",
                },
                {
                    "identifier_type": "medium_url",
                    "value": f"https://medium.com/@wrong-{uuid.uuid4().hex[:6]}",
                },
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "review_required"
    assert resp["candidate_id"] is None
    conflict_types = {c["identifier_type"] for c in resp["conflicts"]}
    assert "linkedin_url" in conflict_types
    assert "medium_url" in conflict_types


def test_strong_signal_type_absent_on_canonical_not_a_mismatch(client: TestClient, tenant: str):
    """Incoming strong signal of a type absent on the canonical candidate is not a mismatch."""
    email = f"ss-absent-{uuid.uuid4().hex[:6]}@example.com"
    # Canonical has only email; no linkedin_url
    seed = _seed_candidate(
        client,
        tenant,
        [{"identifier_type": "email", "value": email}],
    )
    assert seed["resolution_status"] == "created"

    # Incoming adds a linkedin_url (absent on canonical) → should match, not review_required
    resp = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": email},
                {
                    "identifier_type": "linkedin_url",
                    "value": f"https://linkedin.com/in/new-{uuid.uuid4().hex[:6]}",
                },
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "matched"
    assert resp["candidate_id"] == seed["candidate_id"]
    assert resp["conflicts"] == []
    attached_types = {a["identifier_type"] for a in resp["attached_identifiers"]}
    assert "linkedin_url" in attached_types


def test_multi_candidate_conflict_still_returns_review_required(client: TestClient, tenant: str):
    """Existing multi-candidate conflict behavior is preserved after mismatch logic is added."""
    email_a = f"multi-a-{uuid.uuid4().hex[:6]}@example.com"
    linkedin_b = f"https://linkedin.com/in/multi-b-{uuid.uuid4().hex[:6]}"
    a = _seed_candidate(client, tenant, [{"identifier_type": "email", "value": email_a}])
    b = _seed_candidate(client, tenant, [{"identifier_type": "linkedin_url", "value": linkedin_b}])
    assert a["candidate_id"] != b["candidate_id"]

    resp = _post(
        client,
        {
            "source": "signal",
            "source_record_type": "profile",
            "source_record_id": f"SIG-{uuid.uuid4()}",
            "identifiers": [
                {"identifier_type": "email", "value": email_a},
                {"identifier_type": "linkedin_url", "value": linkedin_b},
            ],
            "tenant_id": tenant,
        },
    )
    assert resp["resolution_status"] == "review_required"
    assert resp["candidate_id"] is None
    owners = {c["candidate_id"] for c in resp["conflicts"]}
    assert owners == {a["candidate_id"], b["candidate_id"]}
