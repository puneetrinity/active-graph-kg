"""Database-backed privacy contracts for the public-memory surface."""

from __future__ import annotations

import json
import os
import threading
import uuid
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import psycopg
import pytest
from fastapi import HTTPException

OWNER_DSN = os.getenv("ACTIVEKG_RLS_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_RLS_TEST_RUNTIME_DSN")

pytestmark = pytest.mark.skipif(
    not (OWNER_DSN and RUNTIME_DSN),
    reason="public-memory integration DSNs are not configured",
)

TENANT_A = f"public_test_a_{uuid.uuid4().hex[:8]}"
TENANT_B = f"public_test_b_{uuid.uuid4().hex[:8]}"
VECTOR = "[" + ",".join(["1"] + ["0"] * 383) + "]"


def _public_market(
    *,
    role_family: str = "backend",
    location_city: str = "bengaluru",
    country_code: str = "IN",
    seniority_band: str = "senior",
) -> dict[str, object]:
    dimensions = {
        "version": 1,
        "roleFamily": role_family,
        "locationCity": location_city,
        "locationCountryCode": country_code,
        "seniorityBand": seniority_band,
    }
    coarse_key = (
        "public-market:v1:"
        + sha256(json.dumps(dimensions, separators=(",", ":")).encode("utf-8")).hexdigest()
    )
    return {
        "version": 1,
        "coarse_market_key": coarse_key,
        "role_family": role_family,
        "location_city": location_city,
        "location_country_code": country_code,
        "seniority_band": seniority_band,
    }


class _StaticEmbedder:
    def encode(self, _texts):
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1.0
        return np.asarray([vector])


@pytest.fixture(scope="module")
def public_memory_rows():
    public_id = str(uuid.uuid4())
    private_a_id = str(uuid.uuid4())
    private_b_id = str(uuid.uuid4())
    candidate_a_id = str(uuid.uuid4())
    candidate_b_id = str(uuid.uuid4())
    public_slug = f"public-{uuid.uuid4().hex[:10]}"
    private_a_slug = f"private-a-{uuid.uuid4().hex[:8]}"
    private_b_slug = f"private-b-{uuid.uuid4().hex[:8]}"
    private_a_marker = f"PRIVATE_A_{uuid.uuid4().hex}"
    private_b_marker = f"PRIVATE_B_{uuid.uuid4().hex}"

    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO global_candidates
                    (id, linkedin_id, linkedin_url, email_hash, name, headline,
                     embedding, embedding_status, embed_version,
                     public_profile, public_profile_observed_at,
                     public_crustdata_person_id, public_headline,
                     public_embedding, public_embedding_status, public_embed_version)
                VALUES
                    (%s, %s, %s, %s, 'PRIVATE_CANONICAL_NAME', 'PRIVATE_CANONICAL_HEADLINE',
                     %s::vector, 'ready', 3,
                     %s::jsonb, now(), 991001, 'Public backend engineer',
                     %s::vector, 'ready', 1),
                    (%s, %s, %s, NULL, 'Private A', %s,
                     %s::vector, 'ready', 3, '{}'::jsonb, NULL, NULL, NULL,
                     NULL, 'queued', 0),
                    (%s, %s, %s, NULL, 'Private B', %s,
                     %s::vector, 'ready', 3, '{}'::jsonb, NULL, NULL, NULL,
                     NULL, 'queued', 0)
                """,
                (
                    public_id,
                    public_slug,
                    f"https://linkedin.com/in/{public_slug}",
                    f"private-hash-{public_slug}",
                    VECTOR,
                    '{"basic_profile":{"name":"Public Person","headline":"Public backend engineer"}}',
                    VECTOR,
                    private_a_id,
                    private_a_slug,
                    f"https://linkedin.com/in/{private_a_slug}",
                    private_a_marker,
                    VECTOR,
                    private_b_id,
                    private_b_slug,
                    f"https://linkedin.com/in/{private_b_slug}",
                    private_b_marker,
                    VECTOR,
                ),
            )
            cur.execute(
                """
                INSERT INTO candidate_provenance
                    (global_candidate_id, source_type, tenant_id, source_detail)
                VALUES (%s, 'signal_sourced', NULL, '{}'::jsonb)
                """,
                (public_id,),
            )
            cur.execute(
                """
                INSERT INTO tenant_candidate_access
                    (tenant_id, global_candidate_id, visibility, consent_state, access_reason)
                VALUES (%s, %s, 'private', 'opted_out', 'org_upload'),
                       (%s, %s, 'private', 'opted_out', 'org_upload')
                """,
                (TENANT_A, private_a_id, TENANT_B, private_b_id),
            )
            cur.execute(
                """
                INSERT INTO candidates
                    (candidate_id, tenant_id, display_name, profile, global_candidate_id)
                VALUES (%s, %s, 'Private A', %s::jsonb, %s),
                       (%s, %s, 'Private B', %s::jsonb, %s)
                """,
                (
                    candidate_a_id,
                    TENANT_A,
                    f'{{"private_marker":"{private_a_marker}"}}',
                    private_a_id,
                    candidate_b_id,
                    TENANT_B,
                    f'{{"private_marker":"{private_b_marker}"}}',
                    private_b_id,
                ),
            )

    yield {
        "public_id": public_id,
        "public_slug": public_slug,
        "private_a_id": private_a_id,
        "private_a_slug": private_a_slug,
        "private_b_id": private_b_id,
        "private_b_slug": private_b_slug,
        "private_a_marker": private_a_marker,
        "private_b_marker": private_b_marker,
    }

    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM candidates WHERE tenant_id IN (%s, %s)", (TENANT_A, TENANT_B))
            cur.execute(
                "DELETE FROM tenant_candidate_access WHERE tenant_id IN (%s, %s)",
                (TENANT_A, TENANT_B),
            )
            cur.execute(
                "DELETE FROM contact_suppression_tombstones WHERE global_candidate_id = %s",
                (public_id,),
            )
            cur.execute(
                "DELETE FROM candidate_provenance WHERE global_candidate_id = %s", (public_id,)
            )
            cur.execute(
                "DELETE FROM global_candidates WHERE id = ANY(%s::uuid[])",
                ([public_id, private_a_id, private_b_id],),
            )


def _enable_public_api(monkeypatch):
    from activekg.api import global_memory

    monkeypatch.setattr(global_memory, "_DSN", RUNTIME_DSN)
    monkeypatch.setattr(global_memory, "GLOBAL_MEMORY_ENABLED", True)
    monkeypatch.setattr(global_memory, "PUBLIC_PROFILE_SEARCH_ENABLED", True)
    monkeypatch.setattr(global_memory, "_embedder", _StaticEmbedder())
    return global_memory


def test_public_search_never_reads_private_canonical_or_tenant_rows(
    monkeypatch, public_memory_rows
):
    global_memory = _enable_public_api(monkeypatch)
    body = global_memory.GlobalCandidateSearchRequest(
        query_text="backend engineer",
        surface="public_v1",
        limit=10,
    )

    result_a = global_memory.search_global_candidates(
        body,
        claims=SimpleNamespace(tenant_id=TENANT_A),
    )
    result_b = global_memory.search_global_candidates(
        body,
        claims=SimpleNamespace(tenant_id=TENANT_B),
    )

    rows_a = {str(row["id"]): row for row in result_a["results"]}
    rows_b = {str(row["id"]): row for row in result_b["results"]}
    assert set(rows_a) == {public_memory_rows["public_id"]}
    assert set(rows_b) == {public_memory_rows["public_id"]}
    assert rows_a[public_memory_rows["public_id"]]["evidence_surface"] == "public"
    assert "PRIVATE_CANONICAL" not in repr(result_a)
    assert "PRIVATE_CANONICAL" not in repr(result_b)
    assert public_memory_rows["private_a_marker"] not in repr(result_b)
    assert public_memory_rows["private_b_marker"] not in repr(result_a)


def test_legacy_shared_canonical_search_can_be_disabled(monkeypatch):
    global_memory = _enable_public_api(monkeypatch)
    monkeypatch.setattr(global_memory, "LEGACY_GLOBAL_SEARCH_ENABLED", False)

    with pytest.raises(HTTPException) as excinfo:
        global_memory.search_global_candidates(
            global_memory.GlobalCandidateSearchRequest(
                query_text="backend engineer",
                surface="legacy_v0",
            ),
            claims=SimpleNamespace(tenant_id=TENANT_A),
        )
    assert excinfo.value.status_code == 410


def test_public_anchor_never_returns_private_canonical_fields(monkeypatch, public_memory_rows):
    global_memory = _enable_public_api(monkeypatch)
    public_result = global_memory.get_by_anchor(
        linkedin_id=public_memory_rows["public_slug"],
        github_id=None,
        email_hash=None,
        claims=SimpleNamespace(tenant_id=TENANT_B),
    )
    assert public_result["surface"] == "public_v1"
    assert public_result["name"] == "Public Person"
    assert "email_hash" not in public_result
    assert "PRIVATE_CANONICAL" not in repr(public_result)

    with pytest.raises(HTTPException) as excinfo:
        global_memory.get_by_anchor(
            linkedin_id=None,
            github_id=None,
            email_hash=f"private-hash-{public_memory_rows['public_slug']}",
            claims=SimpleNamespace(tenant_id=TENANT_B),
        )
    assert excinfo.value.status_code == 404


def test_tenant_anchor_is_identity_only_even_for_its_owner(monkeypatch, public_memory_rows):
    global_memory = _enable_public_api(monkeypatch)
    result = global_memory.get_by_anchor(
        linkedin_id=public_memory_rows["private_a_slug"],
        github_id=None,
        email_hash=None,
        claims=SimpleNamespace(tenant_id=TENANT_A),
    )

    assert result == {
        "id": public_memory_rows["private_a_id"],
        "linkedin_id": public_memory_rows["private_a_slug"],
        "linkedin_url": (f"https://linkedin.com/in/{public_memory_rows['private_a_slug']}"),
        "surface": "tenant_identity_v1",
    }
    assert public_memory_rows["private_a_marker"] not in repr(result)
    assert "skills_normalized" not in result
    assert "email_hash" not in result


def test_tenant_private_anchor_never_discloses_unsupplied_canonical_anchors(
    monkeypatch, public_memory_rows
):
    global_memory = _enable_public_api(monkeypatch)
    private_email_hash = f"tenant-owned-{uuid.uuid4().hex}"
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE global_candidates SET email_hash = %s WHERE id = %s",
                (private_email_hash, public_memory_rows["private_a_id"]),
            )
    try:
        result = global_memory.get_by_anchor(
            linkedin_id=None,
            github_id=None,
            email_hash=private_email_hash,
            claims=SimpleNamespace(tenant_id=TENANT_A),
        )
    finally:
        with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE global_candidates SET email_hash = NULL WHERE id = %s",
                    (public_memory_rows["private_a_id"],),
                )

    assert result == {
        "id": public_memory_rows["private_a_id"],
        "surface": "tenant_identity_v1",
        "email_hash": private_email_hash,
    }
    assert "linkedin_id" not in result
    assert "linkedin_url" not in result


def test_public_identity_resolution_is_id_only_and_rejects_lookalike_hosts(
    monkeypatch, public_memory_rows
):
    global_memory = _enable_public_api(monkeypatch)
    result = global_memory.resolve_public_identities(
        global_memory.PublicIdentityLookupRequest(
            linkedin_urls=[
                f"https://www.linkedin.com/in/{public_memory_rows['public_slug']}",
                f"https://evillinkedin.com/in/{public_memory_rows['public_slug']}",
            ]
        ),
        claims=SimpleNamespace(tenant_id=TENANT_B),
    )
    assert result["surface"] == "public_v1"
    assert result["results"] == [
        {
            "linkedin_url": f"https://www.linkedin.com/in/{public_memory_rows['public_slug']}",
            "normalized_linkedin_url": (
                f"https://linkedin.com/in/{public_memory_rows['public_slug']}"
            ),
            "global_candidate_id": public_memory_rows["public_id"],
        }
    ]


def test_platform_exclusions_include_fresh_pre_membership_public_rows(
    monkeypatch, public_memory_rows
):
    global_memory = _enable_public_api(monkeypatch)
    result = global_memory.public_candidate_exclusions(
        global_memory.PublicMarketExclusionRequest(
            coarse_market_key=f"public-market:v1:{'a' * 64}",
            fresh_days=14,
            limit=2000,
        ),
        claims=SimpleNamespace(tenant_id=TENANT_B),
    )

    assert 991001 in result["crustdata_person_ids"]
    assert result["classified_matched"] == 0
    assert result["unclassified_matched"] >= 1
    assert result["unclassified_returned"] >= 1


def test_hard_bounce_suppresses_the_address_not_the_person(monkeypatch, public_memory_rows):
    global_memory = _enable_public_api(monkeypatch)
    email = f"hard-bounce-{uuid.uuid4().hex}@example.com"
    candidate_id = public_memory_rows["public_id"]
    claims_a = SimpleNamespace(tenant_id=TENANT_A)
    claims_b = SimpleNamespace(tenant_id=TENANT_B)

    found = global_memory.record_contact_evidence(
        global_memory.ContactEvidenceRecord(
            global_candidate_id=candidate_id,
            email=email,
            provider="fullenrich",
            provider_record_id=f"fe-{uuid.uuid4().hex}",
            confidence=0.9,
            status="verified",
        ),
        claims=claims_a,
    )
    assert found["state"] == "found"

    bounced = global_memory.record_contact_evidence(
        global_memory.ContactEvidenceRecord(
            global_candidate_id=candidate_id,
            email=email,
            provider="fullenrich",
            provider_record_id=f"fe-{uuid.uuid4().hex}",
            confidence=0.9,
            status="hard_bounce",
            bounce_reason="provider_hard_bounce",
        ),
        claims=claims_a,
    )
    assert bounced["state"] == "suppressed"
    assert email not in repr(bounced)

    # Cross-org contact reuse is still release-gated: B may seek an alternate
    # address. One bad address must not suppress the entire person.
    before_provider = global_memory.lookup_contact_evidence(
        global_memory.ContactEvidenceLookup(global_candidate_ids=[candidate_id]),
        claims=claims_b,
    )
    assert before_provider["results"][0]["state"] == "miss"

    # If B's provider returns the same bad address, the platform hash tombstone
    # rejects it without exposing A's evidence.
    repeated = global_memory.record_contact_evidence(
        global_memory.ContactEvidenceRecord(
            global_candidate_id=candidate_id,
            email=email,
            provider="enrichlayer",
            confidence=0.4,
            status="found",
        ),
        claims=claims_b,
    )
    assert repeated["state"] == "suppressed"
    assert email not in repr(repeated)


def test_cross_tenant_tombstone_reselects_the_callers_alternate(monkeypatch, public_memory_rows):
    global_memory = _enable_public_api(monkeypatch)
    candidate_id = public_memory_rows["public_id"]
    bad_email = f"shared-bad-{uuid.uuid4().hex}@example.com"
    alternate = f"tenant-b-alternate-{uuid.uuid4().hex}@example.com"
    claims_a = SimpleNamespace(tenant_id=TENANT_A)
    claims_b = SimpleNamespace(tenant_id=TENANT_B)

    global_memory.record_contact_evidence(
        global_memory.ContactEvidenceRecord(
            global_candidate_id=candidate_id,
            email=alternate,
            provider="enrichlayer",
            confidence=0.5,
            status="found",
        ),
        claims=claims_b,
    )
    selected_bad = global_memory.record_contact_evidence(
        global_memory.ContactEvidenceRecord(
            global_candidate_id=candidate_id,
            email=bad_email,
            provider="fullenrich",
            confidence=0.9,
            status="verified",
        ),
        claims=claims_b,
    )
    assert selected_bad["contact"]["email"] == bad_email

    global_memory.record_contact_evidence(
        global_memory.ContactEvidenceRecord(
            global_candidate_id=candidate_id,
            email=bad_email,
            provider="fullenrich",
            confidence=0.9,
            status="hard_bounce",
        ),
        claims=claims_a,
    )

    after_bounce = global_memory.lookup_contact_evidence(
        global_memory.ContactEvidenceLookup(global_candidate_ids=[candidate_id]),
        claims=claims_b,
    )
    assert after_bounce["results"][0]["state"] == "found"
    assert after_bounce["results"][0]["contact"]["email"] == alternate


def test_one_address_tombstone_reselects_alternates_for_every_candidate(
    monkeypatch, public_memory_rows
):
    global_memory = _enable_public_api(monkeypatch)
    claims = SimpleNamespace(tenant_id=TENANT_A)
    first_id = public_memory_rows["public_id"]
    second_id = str(uuid.uuid4())
    second_slug = f"contact-second-{uuid.uuid4().hex[:10]}"
    shared_bad = f"multi-candidate-bad-{uuid.uuid4().hex}@example.com"
    first_alternate = f"first-alternate-{uuid.uuid4().hex}@example.com"
    second_alternate = f"second-alternate-{uuid.uuid4().hex}@example.com"

    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO global_candidates
                    (id, linkedin_id, linkedin_url, public_profile,
                     public_profile_observed_at)
                VALUES (%s, %s, %s, %s::jsonb, now())
                """,
                (
                    second_id,
                    second_slug,
                    f"https://linkedin.com/in/{second_slug}",
                    json.dumps({"basic_profile": {"name": "Second Public"}}),
                ),
            )
            cur.execute(
                """
                INSERT INTO candidate_provenance
                    (global_candidate_id, source_type, tenant_id, source_detail)
                VALUES (%s, 'signal_sourced', NULL, '{}'::jsonb)
                """,
                (second_id,),
            )

    try:
        for candidate_id, alternate in (
            (first_id, first_alternate),
            (second_id, second_alternate),
        ):
            global_memory.record_contact_evidence(
                global_memory.ContactEvidenceRecord(
                    global_candidate_id=candidate_id,
                    email=alternate,
                    provider="enrichlayer",
                    confidence=0.5,
                    status="found",
                ),
                claims=claims,
            )
            global_memory.record_contact_evidence(
                global_memory.ContactEvidenceRecord(
                    global_candidate_id=candidate_id,
                    email=shared_bad,
                    provider="fullenrich",
                    confidence=0.9,
                    status="verified",
                ),
                claims=claims,
            )

        global_memory.suppress_contact_evidence(
            global_memory.ContactSuppressionRecord(
                email=shared_bad,
                reason="hard_bounce",
                provider_event_id=f"brevo-{uuid.uuid4().hex}",
            ),
            claims=claims,
        )
        result = global_memory.lookup_contact_evidence(
            global_memory.ContactEvidenceLookup(global_candidate_ids=[first_id, second_id]),
            claims=claims,
        )
        contacts = {
            row["global_candidate_id"]: row["contact"]["email"] for row in result["results"]
        }
        assert contacts == {
            first_id: first_alternate,
            second_id: second_alternate,
        }
    finally:
        with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM contact_suppression_tombstones WHERE email_hash = %s",
                    (sha256(shared_bad.lower().encode()).hexdigest(),),
                )
                cur.execute(
                    "DELETE FROM candidate_provenance WHERE global_candidate_id = %s",
                    (second_id,),
                )
                cur.execute("DELETE FROM global_candidates WHERE id = %s", (second_id,))


def test_sql_and_python_public_projections_reject_type_confusion():
    from activekg.api.global_memory import sanitize_public_profile

    payload = {
        "crustdata_person_id": 771122,
        "basic_profile": {
            "name": "Public Person",
            "summary": "Email sql-private@example.com or +1-415-555-0123",
            "languages": ["English", {"email": "SQL_LANGUAGE_SENTINEL"}],
        },
        "professional_network": {
            "connections": {"email": "SQL_CONNECTION_SENTINEL"},
            "open_to_cards": ["open_to_work", {"email": "SQL_CARD_SENTINEL"}],
        },
        "experience": {
            "employment_details": {
                "current": [
                    {
                        "title": "Engineer",
                        "description": "Call 9876543210",
                        "company_industries": [
                            "Software",
                            {"email": "SQL_INDUSTRY_SENTINEL"},
                        ],
                    }
                ]
            }
        },
        "skills": {
            "professional_network_skills": [
                "Python",
                {"email": "SQL_SKILL_SENTINEL"},
            ]
        },
    }
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT activekg_public_crustdata_projection(%s::jsonb)",
                (json.dumps(payload),),
            )
            sql_projection = cur.fetchone()[0]

    python_projection = sanitize_public_profile(payload)
    assert sql_projection == python_projection
    assert "SENTINEL" not in repr(sql_projection)
    assert "sql-private@example.com" not in repr(sql_projection)
    assert "+1-415-555-0123" not in repr(sql_projection)
    assert "9876543210" not in repr(sql_projection)
    assert "[redacted]" in repr(sql_projection)


def test_public_mirror_never_promotes_flat_signal_hints():
    from activekg.api.global_memory import upsert_signal_candidate_to_global

    slug = f"flat-hint-boundary-{uuid.uuid4().hex[:8]}"
    person_id = uuid.uuid4().int % 9_000_000_000 + 1_000_000_000
    with psycopg.connect(OWNER_DSN) as conn:
        with conn.transaction(force_rollback=True):
            with conn.cursor() as cur:
                candidate_id = upsert_signal_candidate_to_global(
                    cur,
                    tenant_id=TENANT_A,
                    linkedin_url=f"https://linkedin.com/in/{slug}",
                    name="Private Hint Name",
                    headline="Leak headline@example.com or +1-415-555-0123",
                    location_city="Call 9876543210",
                    location_country="IN",
                    seniority_band="senior",
                    skills=["Python", "skill@example.com"],
                    signal_candidate_id=f"https://linkedin.com/in/{slug}",
                    public_profile={
                        "crustdata_person_id": person_id,
                        "basic_profile": {"name": "Public Person"},
                    },
                )
                cur.execute(
                    """
                    SELECT public_headline, public_location_city,
                           public_seniority_band, public_skills_normalized
                    FROM global_candidates
                    WHERE id = %s
                    """,
                    (candidate_id,),
                )
                assert cur.fetchone() == (None, None, None, None)


def test_public_headline_must_equal_projected_profile_headline():
    with psycopg.connect(OWNER_DSN) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO global_candidates
                            (id, public_profile, public_headline)
                        VALUES (%s, '{"basic_profile":{"headline":"Public"}}'::jsonb, 'Other')
                        """,
                        (str(uuid.uuid4()),),
                    )


def test_migration_preflight_refuses_duplicate_public_provider_ids():
    first_id = str(uuid.uuid4())
    second_id = str(uuid.uuid4())
    with psycopg.connect(OWNER_DSN) as conn:
        with pytest.raises(psycopg.errors.UniqueViolation) as excinfo:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("DROP INDEX idx_gc_public_crustdata_person_id")
                    cur.execute(
                        """
                        INSERT INTO global_candidates
                            (id, public_profile, public_crustdata_person_id)
                        VALUES (%s, '{"crustdata_person_id":778899}'::jsonb, 778899),
                               (%s, '{"crustdata_person_id":778899}'::jsonb, 778899)
                        """,
                        (first_id, second_id),
                    )
                    cur.execute("SELECT activekg_assert_public_crustdata_backfill_safe()")
    assert excinfo.value.sqlstate == "23505"
    assert "maps to multiple global candidates" in str(excinfo.value)


def test_live_crustdata_id_conflict_is_queued_without_publishing_duplicate():
    from activekg.api.global_memory import upsert_signal_candidate_to_global

    existing_id = str(uuid.uuid4())
    existing_slug = f"conflict-existing-{uuid.uuid4().hex[:8]}"
    incoming_slug = f"conflict-incoming-{uuid.uuid4().hex[:8]}"
    with psycopg.connect(OWNER_DSN) as conn:
        with conn.transaction(force_rollback=True):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO global_candidates
                        (id, linkedin_id, linkedin_url, public_profile,
                         public_crustdata_person_id, merge_status)
                    VALUES (%s, %s, %s, %s::jsonb, 889900, 'single')
                    """,
                    (
                        existing_id,
                        existing_slug,
                        f"https://linkedin.com/in/{existing_slug}",
                        '{"crustdata_person_id":889900,"basic_profile":{"name":"Existing"}}',
                    ),
                )
                incoming_id = upsert_signal_candidate_to_global(
                    cur,
                    tenant_id=TENANT_A,
                    linkedin_url=f"https://linkedin.com/in/{incoming_slug}",
                    name="Incoming",
                    headline="Engineer",
                    location_city="Bengaluru",
                    location_country="IN",
                    seniority_band="senior",
                    skills=["Python"],
                    signal_candidate_id=f"https://linkedin.com/in/{incoming_slug}",
                    public_profile={
                        "crustdata_person_id": 889900,
                        "basic_profile": {"name": "Incoming", "headline": "Engineer"},
                    },
                )
                assert incoming_id and incoming_id != existing_id
                cur.execute(
                    """
                    SELECT public_crustdata_person_id, public_profile, merge_status
                    FROM global_candidates WHERE id = %s
                    """,
                    (incoming_id,),
                )
                public_id, profile, merge_status = cur.fetchone()
                assert public_id is None
                assert profile == {}
                assert merge_status == "needs_merge"
                cur.execute(
                    """
                    SELECT reason, details
                    FROM candidate_merge_queue
                    WHERE global_candidate_id_a = %s
                      AND global_candidate_id_b = %s
                    """,
                    (incoming_id, existing_id),
                )
                reason, details = cur.fetchone()
                assert reason == "review_required"
                assert details["anchor"] == "crustdata_person_id"


def test_same_linkedin_cannot_silently_switch_crustdata_person_id():
    from activekg.api.global_memory import upsert_signal_candidate_to_global

    global_id = str(uuid.uuid4())
    slug = f"provider-switch-{uuid.uuid4().hex[:8]}"
    with psycopg.connect(OWNER_DSN) as conn:
        with conn.transaction(force_rollback=True):
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO global_candidates
                        (id, linkedin_id, linkedin_url, public_profile,
                         public_crustdata_person_id, merge_status)
                    VALUES (%s, %s, %s, %s::jsonb, 111001, 'single')
                    """,
                    (
                        global_id,
                        slug,
                        f"https://linkedin.com/in/{slug}",
                        '{"crustdata_person_id":111001,"basic_profile":{"name":"Original"}}',
                    ),
                )
                resolved_id = upsert_signal_candidate_to_global(
                    cur,
                    tenant_id=TENANT_A,
                    linkedin_url=f"https://linkedin.com/in/{slug}",
                    name="Incoming",
                    headline="Different profile",
                    location_city="Bengaluru",
                    location_country="IN",
                    seniority_band="senior",
                    skills=["Python"],
                    signal_candidate_id=f"https://linkedin.com/in/{slug}",
                    public_profile={
                        "crustdata_person_id": 222002,
                        "basic_profile": {
                            "name": "Wrong Incoming",
                            "headline": "Different profile",
                        },
                    },
                    public_market=_public_market(
                        role_family="wrong-market",
                        location_city="mumbai",
                    ),
                )
                assert resolved_id == global_id
                cur.execute(
                    """
                    SELECT public_crustdata_person_id, public_profile, merge_status
                    FROM global_candidates WHERE id = %s
                    """,
                    (global_id,),
                )
                person_id, profile, merge_status = cur.fetchone()
                assert person_id == 111001
                assert profile["basic_profile"]["name"] == "Original"
                assert merge_status == "needs_merge"
                cur.execute(
                    """
                    SELECT global_candidate_id_b, reason, details
                    FROM candidate_merge_queue
                    WHERE global_candidate_id_a = %s
                      AND reason = 'review_required'
                    """,
                    (global_id,),
                )
                other_id, reason, details = cur.fetchone()
                assert other_id is None
                assert reason == "review_required"
                assert details["anchor"] == "crustdata_person_id_switch"
                assert details["existing_crustdata_person_id"] == 111001
                assert details["incoming_crustdata_person_id"] == 222002
                cur.execute(
                    """
                    SELECT count(*)
                    FROM public_candidate_market_memberships
                    WHERE global_candidate_id = %s
                    """,
                    (global_id,),
                )
                assert cur.fetchone()[0] == 0


def test_concurrent_public_observations_choose_one_provider_identity(monkeypatch):
    from activekg.api import global_memory

    global_id = str(uuid.uuid4())
    slug = f"provider-race-{uuid.uuid4().hex[:8]}"
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO global_candidates
                    (id, linkedin_id, linkedin_url, public_profile, merge_status)
                VALUES (%s, %s, %s, '{}'::jsonb, 'single')
                """,
                (global_id, slug, f"https://linkedin.com/in/{slug}"),
            )

    barrier = threading.Barrier(2)
    original_find = global_memory._find_existing_all

    def synchronized_find(cur, linkedin_id, github_id, email_hash):
        result = original_find(cur, linkedin_id, github_id, email_hash)
        barrier.wait(timeout=5)
        return result

    monkeypatch.setattr(global_memory, "_find_existing_all", synchronized_find)
    first_person_id = 1_000_000_000 + uuid.uuid4().int % 3_000_000_000
    second_person_id = 4_000_000_000 + uuid.uuid4().int % 3_000_000_000
    observations = [
        (first_person_id, _public_market(role_family="race-a")),
        (second_person_id, _public_market(role_family="race-b")),
    ]
    failures: list[BaseException] = []

    def ingest(person_id: int, market: dict[str, object]) -> None:
        try:
            with psycopg.connect(OWNER_DSN) as conn:
                with conn.cursor() as cur:
                    global_memory.upsert_signal_candidate_to_global(
                        cur,
                        tenant_id=TENANT_A,
                        linkedin_url=f"https://linkedin.com/in/{slug}",
                        name=f"Race {person_id}",
                        headline="Backend engineer",
                        location_city="Bengaluru",
                        location_country="IN",
                        seniority_band="senior",
                        skills=["Python"],
                        signal_candidate_id=f"race-{person_id}",
                        public_profile={
                            "crustdata_person_id": person_id,
                            "basic_profile": {
                                "name": f"Race {person_id}",
                                "headline": "Backend engineer",
                            },
                        },
                        public_role_family=str(market["role_family"]),
                        public_market=market,
                    )
                conn.commit()
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    threads = [
        threading.Thread(target=ingest, args=observation, daemon=True)
        for observation in observations
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert failures == []

    expected_market_by_person = {
        person_id: market["coarse_market_key"] for person_id, market in observations
    }
    with psycopg.connect(OWNER_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT public_crustdata_person_id, public_profile, merge_status
                FROM global_candidates WHERE id = %s
                """,
                (global_id,),
            )
            accepted_person_id, public_profile, merge_status = cur.fetchone()
            assert accepted_person_id in expected_market_by_person
            assert public_profile["crustdata_person_id"] == accepted_person_id
            assert merge_status == "needs_merge"
            cur.execute(
                """
                SELECT coarse_market_key
                FROM public_candidate_market_memberships
                WHERE global_candidate_id = %s
                """,
                (global_id,),
            )
            memberships = [row[0] for row in cur.fetchall()]
            assert memberships == [expected_market_by_person[accepted_person_id]]
            cur.execute(
                """
                SELECT details
                FROM candidate_merge_queue
                WHERE global_candidate_id_a = %s
                  AND global_candidate_id_b IS NULL
                  AND reason = 'review_required'
                """,
                (global_id,),
            )
            conflict = cur.fetchone()[0]
            assert conflict["anchor"] == "crustdata_person_id_switch"
            assert {
                conflict["existing_crustdata_person_id"],
                conflict["incoming_crustdata_person_id"],
            } == {first_person_id, second_person_id}


def test_signal_public_ingest_records_candidate_role_and_coarse_market():
    from activekg.api.global_memory import upsert_signal_candidate_to_global

    slug = f"public-market-{uuid.uuid4().hex[:8]}"
    market_dimensions = {
        "version": 1,
        "roleFamily": "backend",
        "locationCity": "bengaluru",
        "locationCountryCode": "IN",
        "seniorityBand": "senior",
    }
    coarse_key = (
        "public-market:v1:"
        + sha256(json.dumps(market_dimensions, separators=(",", ":")).encode("utf-8")).hexdigest()
    )
    with psycopg.connect(OWNER_DSN) as conn:
        with conn.transaction(force_rollback=True):
            with conn.cursor() as cur:
                global_id = upsert_signal_candidate_to_global(
                    cur,
                    tenant_id=TENANT_A,
                    linkedin_url=f"https://linkedin.com/in/{slug}",
                    name="Public Candidate",
                    headline="Backend Engineer",
                    location_city="Bengaluru",
                    location_country="IN",
                    seniority_band="senior",
                    skills=["Python"],
                    signal_candidate_id=f"https://linkedin.com/in/{slug}",
                    public_profile={
                        "crustdata_person_id": 990011,
                        "basic_profile": {
                            "name": "Public Candidate",
                            "headline": "Backend Engineer",
                            "location": {
                                "city": "Bengaluru",
                                "country_code": "IN",
                            },
                        },
                    },
                    public_role_family="backend",
                    public_market={
                        "version": 1,
                        "coarse_market_key": coarse_key,
                        "role_family": "backend",
                        "location_city": "bengaluru",
                        "location_country_code": "in",
                        "seniority_band": "senior",
                    },
                )
                cur.execute(
                    """
                    SELECT public_role_family, public_crustdata_person_id
                    FROM global_candidates WHERE id = %s
                    """,
                    (global_id,),
                )
                assert cur.fetchone() == ("backend", 990011)
                cur.execute(
                    """
                    SELECT role_family, location_city, location_country_code, seniority_band
                    FROM public_candidate_market_memberships
                    WHERE global_candidate_id = %s AND coarse_market_key = %s
                    """,
                    (global_id, coarse_key),
                )
                assert cur.fetchone() == ("backend", "bengaluru", "IN", "senior")
                cur.execute(
                    """
                    SELECT source_detail
                    FROM candidate_provenance
                    WHERE global_candidate_id = %s
                      AND source_type = 'signal_sourced'
                      AND tenant_id IS NULL
                    """,
                    (global_id,),
                )
                assert cur.fetchone()[0] == {}
