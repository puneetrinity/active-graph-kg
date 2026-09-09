from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import psycopg
import pytest

from activekg.api import sourced_candidates as receiver
from activekg.api.auth import JWTClaims

OWNER_DSN = os.getenv("ACTIVEKG_SOURCED_CANDIDATE_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_SOURCED_CANDIDATE_TEST_RUNTIME_DSN")

pytestmark = pytest.mark.skipif(
    not OWNER_DSN or not RUNTIME_DSN,
    reason="disposable sourced-candidate PostgreSQL DSNs are not configured",
)


def _claims() -> JWTClaims:
    return JWTClaims(
        tenant_id="org_970001",
        actor_id="signal-service",
        actor_type="service",
        scopes=["candidate-source:write"],
        issuer="signal",
    )


def _payload(
    provider_id: int,
    receipt: str,
    observed_at: datetime,
    *,
    linkedin_slug: str | None = None,
    expected: str | None = None,
    headline: str = "Backend Engineer",
) -> receiver.SourcedCandidateIngestRequest:
    body: dict[str, object] = {
        "schema_version": 1,
        "provider_namespace": "crustdata",
        "record_type": "person",
        "adapter_version": "crustdata_person_v1",
        "provider_record_id": str(provider_id),
        "linkedin_url": f"https://linkedin.com/in/{linkedin_slug or f'wave4a-{provider_id}'}",
        "expected_global_candidate_id": expected,
        "acquisition_receipt_id": receipt,
        "acquisition_generation": 1,
        "acquisition_slot": "exact",
        "acquired_at": observed_at.isoformat(),
        "provider_observed_at": observed_at.isoformat(),
        "idempotency_key": "0" * 64,
        "normalized_profile": {
            "display_name": f"Synthetic {provider_id}",
            "headline": headline,
            "languages": ["English"],
            "skills": ["Python", "PostgreSQL"],
            "current_employment": [],
            "past_employment": [],
            "education": [],
            "certifications": [],
            "honors": [],
            "public_profiles": [
                {
                    "platform": "linkedin",
                    "profile_url": (
                        f"https://linkedin.com/in/{linkedin_slug or f'wave4a-{provider_id}'}"
                    ),
                }
            ],
        },
    }
    components = (
        "v1",
        str(body["provider_namespace"]),
        str(body["record_type"]),
        str(body["provider_record_id"]),
        str(body["acquisition_receipt_id"]),
        str(body["acquisition_generation"]),
        str(body["acquisition_slot"]),
    )
    body["idempotency_key"] = hashlib.sha256("\0".join(components).encode()).hexdigest()
    return receiver.SourcedCandidateIngestRequest.model_validate_json(json.dumps(body))


def _counts() -> tuple[int, int, int, int]:
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT "
            "(SELECT count(*) FROM global_candidates),"
            "(SELECT count(*) FROM global_candidate_source_identities),"
            "(SELECT count(*) FROM global_candidate_source_observations),"
            "(SELECT count(*) FROM global_candidate_ingest_receipts)"
        )
        return tuple(int(value) for value in cur.fetchone())


def _store(payload: receiver.SourcedCandidateIngestRequest) -> dict[str, object]:
    return receiver._store(payload, _claims())


def test_identity_evidence_replay_conflict_atomicity_acl_and_append_only() -> None:
    assert OWNER_DSN and RUNTIME_DSN
    os.environ["ACTIVEKG_DSN"] = RUNTIME_DSN
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "TRUNCATE global_candidate_ingest_receipts,"
            "global_candidate_source_observations,"
            "global_candidate_source_identities"
        )

    with psycopg.connect(RUNTIME_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user")
        assert cur.fetchone() == (False, False)
        for table in (
            "global_candidate_source_identities",
            "global_candidate_source_observations",
            "global_candidate_ingest_receipts",
        ):
            cur.execute(
                "SELECT has_table_privilege(current_user,%s,'SELECT'),"
                "has_table_privilege(current_user,%s,'INSERT'),"
                "has_table_privilege(current_user,%s,'UPDATE'),"
                "has_table_privilege(current_user,%s,'DELETE'),"
                "has_table_privilege(current_user,%s,'TRUNCATE'),"
                "has_table_privilege(current_user,%s,'REFERENCES'),"
                "has_table_privilege(current_user,%s,'TRIGGER')",
                (table,) * 7,
            )
            assert cur.fetchone() == (True, True, False, False, False, False, False)

    before = _counts()
    observed = datetime.now(timezone.utc) - timedelta(minutes=5)
    first = _payload(970001, "wave4a-receipt-one", observed)
    created = _store(first)
    assert created["delivery_status"] == "recorded"
    assert created["resolution"] == "created"
    assert created["global_candidate_id"]
    assert created["source_identity_id"]
    after_created = _counts()
    assert tuple(after_created[index] - before[index] for index in range(4)) == (1, 1, 1, 1)

    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT created_at FROM global_candidate_ingest_receipts WHERE idempotency_key=%s",
            (first.idempotency_key,),
        )
        receipt_created_at = cur.fetchone()[0]
    replay = _store(first)
    assert replay == {**created, "delivery_status": "replayed"}
    assert _counts() == after_created
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT created_at FROM global_candidate_ingest_receipts WHERE idempotency_key=%s",
            (first.idempotency_key,),
        )
        assert cur.fetchone()[0] == receipt_created_at

    changed = _payload(
        970001,
        "wave4a-receipt-one",
        observed,
        headline="Changed bytes",
    )
    with pytest.raises(Exception) as idempotency_conflict:
        _store(changed)
    assert getattr(idempotency_conflict.value, "status_code", None) == 409
    assert _counts() == after_created

    refreshed = _payload(
        970001,
        "wave4a-receipt-two",
        observed + timedelta(minutes=2),
        headline="Principal Engineer",
    )
    assert _store(refreshed)["resolution"] == "refreshed"
    stale = _payload(
        970001,
        "wave4a-receipt-three",
        observed - timedelta(days=1),
        headline="Old title",
    )
    assert _store(stale)["resolution"] == "stale"
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT public_headline FROM global_candidates WHERE id=%s",
            (created["global_candidate_id"],),
        )
        assert cur.fetchone()[0] == "Principal Engineer"

        other_id = str(uuid4())
        cur.execute(
            "INSERT INTO global_candidates(id,linkedin_id,linkedin_url) VALUES (%s,%s,%s)",
            (
                other_id,
                f"wave4a-other-{other_id}",
                f"https://linkedin.com/in/wave4a-other-{other_id}",
            ),
        )
    conflict = _payload(
        970001,
        "wave4a-receipt-conflict",
        observed + timedelta(minutes=3),
        expected=other_id,
    )
    conflict_result = _store(conflict)
    assert conflict_result["resolution"] == "conflict_review_required"
    assert conflict_result["global_candidate_id"] is None
    assert conflict_result["source_identity_id"] is None
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT source_identity_id,global_candidate_id,conflict_code "
            "FROM global_candidate_source_observations WHERE idempotency_key=%s",
            (conflict.idempotency_key,),
        )
        assert cur.fetchone() == (None, None, "provider_expected_mismatch")

    # A previously unseen provider id cannot use an arbitrary expected UUID to
    # bind itself to a candidate whose exact LinkedIn anchor disagrees.
    expected_anchor_conflict = _payload(
        970004,
        "wave4a-expected-anchor-conflict",
        observed + timedelta(minutes=4),
        expected=other_id,
    )
    expected_conflict_result = _store(expected_anchor_conflict)
    assert expected_conflict_result["resolution"] == "conflict_review_required"
    assert expected_conflict_result["global_candidate_id"] is None
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT conflict_code FROM global_candidate_source_observations "
            "WHERE idempotency_key=%s",
            (expected_anchor_conflict.idempotency_key,),
        )
        assert cur.fetchone()[0] == "linkedin_expected_mismatch"

    concurrent = _payload(970002, "wave4a-concurrent", observed)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _index: _store(concurrent), range(2)))
    assert sorted(item["delivery_status"] for item in outcomes) == ["recorded", "replayed"]
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM global_candidate_source_observations WHERE idempotency_key=%s",
            (concurrent.idempotency_key,),
        )
        assert cur.fetchone()[0] == 1

    aliases = [
        _payload(970005, "wave4a-alias-a", observed, linkedin_slug="wave4a-alias"),
        _payload(970006, "wave4a-alias-b", observed, linkedin_slug="wave4a-alias"),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        alias_outcomes = list(pool.map(_store, aliases))
    assert len({item["global_candidate_id"] for item in alias_outcomes}) == 1
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(DISTINCT global_candidate_id),count(*) "
            "FROM global_candidate_source_identities WHERE provider_record_id IN ('970005','970006')"
        )
        assert cur.fetchone() == (1, 2)

    failing = _payload(970003, "wave4a-atomic-failure", observed)
    atomic_before = _counts()
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "CREATE FUNCTION wave4a_test_receipt_fail() RETURNS trigger LANGUAGE plpgsql "
            "AS $$ BEGIN RAISE EXCEPTION 'wave4a test'; END $$"
        )
        cur.execute(
            "CREATE TRIGGER wave4a_test_receipt_fail BEFORE INSERT "
            "ON global_candidate_ingest_receipts FOR EACH ROW "
            "EXECUTE FUNCTION wave4a_test_receipt_fail()"
        )
    try:
        with pytest.raises(Exception) as unavailable:
            _store(failing)
        assert getattr(unavailable.value, "status_code", None) == 503
        assert _counts() == atomic_before
    finally:
        with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
            cur.execute("DROP TRIGGER wave4a_test_receipt_fail ON global_candidate_ingest_receipts")
            cur.execute("DROP FUNCTION wave4a_test_receipt_fail()")

    for dsn in (OWNER_DSN, RUNTIME_DSN):
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            for statement in (
                "UPDATE global_candidate_source_identities SET provider_record_id=provider_record_id",
                "DELETE FROM global_candidate_source_observations",
                "TRUNCATE global_candidate_ingest_receipts",
            ):
                with pytest.raises(psycopg.Error):
                    cur.execute(statement)
                conn.rollback()

    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.columns WHERE table_schema='public' "
            "AND table_name LIKE 'global_candidate_source_%' "
            "AND column_name ~ '(tenant|organization|job|query|rank|email|phone|contact|resume)'"
        )
        assert cur.fetchone()[0] == 0
