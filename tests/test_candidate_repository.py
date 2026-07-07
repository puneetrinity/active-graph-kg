"""Integration tests for CandidateRepository.

These exercise migration 012 against a live Postgres. They are skipped if the
database at ``ACTIVEKG_DSN`` is unreachable so local runs without Docker still
pass.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import psycopg
import pytest

from activekg.graph.candidate_repository import (
    CandidateRepository,
    IdentifierConflict,
)
from activekg.graph.models import Candidate, CandidateSourceRecord

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


@pytest.fixture(scope="module")
def _migrated_db() -> None:
    with psycopg.connect(DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(MIGRATION_012.read_text())
            cur.execute(MIGRATION_013.read_text())
            cur.execute(MIGRATION_014.read_text())


@pytest.fixture()
def tenant() -> str:
    # Use a unique tenant per test so uniqueness constraints don't bleed across
    # runs on a shared database.
    return f"test-{uuid.uuid4()}"


@pytest.fixture()
def candidate_repo(_migrated_db) -> CandidateRepository:
    repo = CandidateRepository(DSN)
    yield repo
    repo.close()


def test_create_and_get_candidate(candidate_repo: CandidateRepository, tenant: str):
    candidate = Candidate(tenant_id=tenant, display_name="Alice Example")
    candidate_repo.create_candidate(candidate)

    fetched = candidate_repo.get_candidate(candidate.candidate_id, tenant_id=tenant)
    assert fetched is not None
    assert fetched.display_name == "Alice Example"
    assert fetched.scope == "shared"
    assert fetched.tenant_id == tenant


def test_add_identifier_is_idempotent(candidate_repo: CandidateRepository, tenant: str):
    candidate = Candidate(tenant_id=tenant)
    candidate_repo.create_candidate(candidate)

    ident_a = candidate_repo.add_identifier(
        candidate.candidate_id,
        "email",
        "Alice@Example.com",
        tenant_id=tenant,
        source="vantahire",
    )
    ident_b = candidate_repo.add_identifier(
        candidate.candidate_id,
        "email",
        "alice@example.com",
        tenant_id=tenant,
        source="signal",
    )

    assert ident_a.value_normalized == "alice@example.com"
    assert ident_b.value_normalized == "alice@example.com"
    idents = candidate_repo.list_identifiers(candidate.candidate_id, tenant_id=tenant)
    assert len(idents) == 1


def test_find_candidate_by_identifier(candidate_repo: CandidateRepository, tenant: str):
    candidate = Candidate(tenant_id=tenant, display_name="Bob")
    candidate_repo.create_candidate(candidate)
    candidate_repo.add_identifier(
        candidate.candidate_id,
        "linkedin_url",
        "https://www.linkedin.com/in/bob/",
        tenant_id=tenant,
    )

    # Different textual shape, same person.
    fetched = candidate_repo.find_candidate_by_identifier(
        "linkedin_url",
        "linkedin.com/in/BOB?trk=foo",
        tenant_id=tenant,
    )
    assert fetched is not None
    assert fetched.candidate_id == candidate.candidate_id


def test_identifier_conflict_across_candidates(candidate_repo: CandidateRepository, tenant: str):
    a = Candidate(tenant_id=tenant)
    b = Candidate(tenant_id=tenant)
    candidate_repo.create_candidate(a)
    candidate_repo.create_candidate(b)

    candidate_repo.add_identifier(a.candidate_id, "email", "carol@example.com", tenant_id=tenant)
    with pytest.raises(IdentifierConflict):
        candidate_repo.add_identifier(
            b.candidate_id, "email", "carol@example.com", tenant_id=tenant
        )


def test_source_record_preserves_provenance(candidate_repo: CandidateRepository, tenant: str):
    candidate = Candidate(tenant_id=tenant)
    candidate_repo.create_candidate(candidate)

    record = candidate_repo.upsert_source_record(
        CandidateSourceRecord(
            candidate_id=candidate.candidate_id,
            tenant_id=tenant,
            source="vantahire",
            source_record_type="application",
            source_record_id="VH-1001",
            source_url="https://vantahire.example.com/app/1001",
            payload={"stage": "screening", "role": "staff-eng"},
        )
    )
    assert record.id

    # Upsert again with new payload — should update, not duplicate.
    candidate_repo.upsert_source_record(
        CandidateSourceRecord(
            candidate_id=candidate.candidate_id,
            tenant_id=tenant,
            source="vantahire",
            source_record_type="application",
            source_record_id="VH-1001",
            payload={"stage": "offer", "role": "staff-eng"},
        )
    )

    records = candidate_repo.list_source_records(candidate.candidate_id, tenant_id=tenant)
    assert len(records) == 1
    assert records[0].payload["stage"] == "offer"


def test_upsert_candidate_from_source_merges_across_sources(
    candidate_repo: CandidateRepository, tenant: str
):
    # VantaHire sees the candidate first, by email + vantahire application id.
    first = candidate_repo.upsert_candidate_from_source(
        source="vantahire",
        source_record_type="application",
        source_record_id="VH-42",
        identifiers=[
            ("email", "dana@example.com"),
            ("vantahire_application_id", "VH-42"),
        ],
        payload={"role": "swe"},
        tenant_id=tenant,
        display_name="Dana",
    )

    # Signal sees the same person later — only the email overlaps, but that's
    # enough to merge onto the existing canonical candidate.
    second = candidate_repo.upsert_candidate_from_source(
        source="signal",
        source_record_type="profile",
        source_record_id="SIG-99",
        identifiers=[
            ("email", "Dana@Example.com"),
            ("signal_candidate_id", "SIG-99"),
            ("github_url", "https://github.com/dana"),
        ],
        payload={"score": 0.87},
        tenant_id=tenant,
    )

    assert first.candidate_id == second.candidate_id

    idents = candidate_repo.list_identifiers(first.candidate_id, tenant_id=tenant)
    types = {i.identifier_type for i in idents}
    assert {
        "email",
        "vantahire_application_id",
        "signal_candidate_id",
        "github_url",
    } <= types

    records = candidate_repo.list_source_records(first.candidate_id, tenant_id=tenant)
    sources = {(r.source, r.source_record_type, r.source_record_id) for r in records}
    assert ("vantahire", "application", "VH-42") in sources
    assert ("signal", "profile", "SIG-99") in sources


# ---------------------------------------------------------------------------
# Signal tag search
# ---------------------------------------------------------------------------


def _make_signal_record(
    candidate_repo: CandidateRepository,
    *,
    candidate_id: str,
    tenant: str,
    tags: list[str],
    sig_id: str | None = None,
) -> None:
    candidate_repo.upsert_source_record(
        CandidateSourceRecord(
            candidate_id=candidate_id,
            tenant_id=tenant,
            source="signal",
            source_record_type="sourced_candidate",
            source_record_id=sig_id or f"SIG-{uuid.uuid4()}",
            job_tags=tags,
        )
    )


def test_signal_tag_search_exact_match(candidate_repo: CandidateRepository, tenant: str):
    c = Candidate(tenant_id=tenant, display_name="Tag Alice")
    candidate_repo.create_candidate(c)
    _make_signal_record(
        candidate_repo,
        candidate_id=c.candidate_id,
        tenant=tenant,
        tags=["python", "go", "distributed systems"],
    )

    results = candidate_repo.search_candidates_by_signal_tags(
        ["python", "go", "distributed systems"], tenant_id=tenant
    )
    assert any(r.candidate_id == c.candidate_id for r in results)
    match = next(r for r in results if r.candidate_id == c.candidate_id)
    assert match.overlap_count == 3
    assert match.overlap_ratio == pytest.approx(1.0)


def test_signal_tag_search_70_percent_threshold_match(
    candidate_repo: CandidateRepository, tenant: str
):
    c = Candidate(tenant_id=tenant)
    candidate_repo.create_candidate(c)
    # Stored: 7 tags. Query: 10 tags, 7 of which overlap → 70 % → should match.
    stored = ["a", "b", "c", "d", "e", "f", "g"]
    query = ["a", "b", "c", "d", "e", "f", "g", "x", "y", "z"]
    _make_signal_record(candidate_repo, candidate_id=c.candidate_id, tenant=tenant, tags=stored)

    results = candidate_repo.search_candidates_by_signal_tags(query, tenant_id=tenant)
    assert any(r.candidate_id == c.candidate_id for r in results)


def test_signal_tag_search_below_threshold_no_match(
    candidate_repo: CandidateRepository, tenant: str
):
    c = Candidate(tenant_id=tenant)
    candidate_repo.create_candidate(c)
    # Stored: 6 tags matching out of 10 query tags → 60 % → below threshold.
    stored = ["a", "b", "c", "d", "e", "f", "nope1", "nope2"]
    query = ["a", "b", "c", "d", "e", "f", "x", "y", "z", "w"]
    _make_signal_record(candidate_repo, candidate_id=c.candidate_id, tenant=tenant, tags=stored)

    results = candidate_repo.search_candidates_by_signal_tags(query, tenant_id=tenant)
    assert not any(r.candidate_id == c.candidate_id for r in results)


def test_signal_tag_search_deduplicates_by_candidate(
    candidate_repo: CandidateRepository, tenant: str
):
    # Same candidate with two qualifying Signal source records → appears once.
    c = Candidate(tenant_id=tenant)
    candidate_repo.create_candidate(c)
    tags = ["python", "go"]
    _make_signal_record(
        candidate_repo,
        candidate_id=c.candidate_id,
        tenant=tenant,
        tags=tags,
        sig_id=f"SIG-A-{uuid.uuid4()}",
    )
    _make_signal_record(
        candidate_repo,
        candidate_id=c.candidate_id,
        tenant=tenant,
        tags=tags + ["java"],
        sig_id=f"SIG-B-{uuid.uuid4()}",
    )

    results = candidate_repo.search_candidates_by_signal_tags(tags, tenant_id=tenant)
    matches = [r for r in results if r.candidate_id == c.candidate_id]
    assert len(matches) == 1


def test_signal_tag_search_result_limit_never_exceeds_100(
    candidate_repo: CandidateRepository, tenant: str
):
    tag = f"bulk-tag-{uuid.uuid4()}"
    for _ in range(5):
        c = Candidate(tenant_id=tenant)
        candidate_repo.create_candidate(c)
        _make_signal_record(candidate_repo, candidate_id=c.candidate_id, tenant=tenant, tags=[tag])

    results = candidate_repo.search_candidates_by_signal_tags([tag], tenant_id=tenant, limit=2)
    assert len(results) <= 2

    results_default = candidate_repo.search_candidates_by_signal_tags([tag], tenant_id=tenant)
    assert len(results_default) <= 100
