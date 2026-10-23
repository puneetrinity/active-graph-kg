"""Explicit disposable DB only; synthetic upstream evidence, no provider calls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Barrier
from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from activekg.candidate_index.contracts import IndexTunables, canonical_json, chunk_manifest
from activekg.candidate_index.recovery import ResumeRecovery
from activekg.candidate_index.repository import IndexRepository
from activekg.candidate_index.resume_policy import policy_fields
from activekg.privacy.models import CandidatePrivacyAction
from tests import test_candidate_index_postgres as base
from tests.test_candidate_index_postgres import (
    OWNER,
    POLICY,
    RUNTIME,
    directive,
    source,
)

pytestmark = pytest.mark.skipif(not OWNER or not RUNTIME, reason="disposable DSNs required")
target = base.target
isolated_coordination = base.isolated_coordination


def failed_source(scope=None):
    repo, scope, accepted, app = source(scope)
    dispatch = repo.reserve(repo.claim("extract")[0], POLICY)
    assert repo.fail(dispatch, "incomplete")
    second = repo.reserve(repo.claim("extract")[0], POLICY)
    assert repo.fail(second, "incomplete")
    return repo, scope, accepted, app


def plan(target, scope, accepted):
    policy = replace(POLICY, **policy_fields())
    pairs = [{"source_id": accepted["source_id"], "generation_id": accepted["generation_id"]}]
    return target.execute(
        "SELECT candidate_index_plan_recovery(%s,%s,%s)",
        (scope, Jsonb(pairs), canonical_json(asdict(policy))),
    ).fetchone()[0]


def apply(target, scope, value, cap=2):
    digest = target.execute(
        "SELECT encode(sha256(convert_to(%s::jsonb::text,'UTF8')),'hex')", (Jsonb(value),)
    ).fetchone()[0]
    return target.execute(
        "SELECT candidate_index_apply_recovery(%s,%s,%s,%s)", (scope, Jsonb(value), digest, cap)
    ).fetchone()[0]


def test_plan_readonly_apply_idempotent_cap_and_no_reset(target):
    repo, scope, accepted, _ = failed_source()
    with target.transaction():
        target.execute("SET TRANSACTION READ ONLY")
        value = plan(target, scope, accepted)
    assert value["max_calls"] == 2
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        apply(target, scope, value, cap=1)
    result = apply(target, scope, value)
    assert not result["replayed"]
    assert apply(target, scope, value)["replayed"]
    assert target.execute(
        "SELECT attempts FROM candidate_index_jobs WHERE generation_id=%s AND stage='extract'",
        (accepted["generation_id"],),
    ).fetchone() == (2,)
    assert target.execute(
        "SELECT count(*) FROM candidate_index_generations WHERE source_id=%s",
        (accepted["source_id"],),
    ).fetchone() == (2,)


def test_wrong_scope_stale_expired_and_runtime_apply_refused(target):
    _, scope, accepted, _ = failed_source()
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        plan(target, "org_999999", accepted)
    value = plan(target, scope, accepted)
    expired = {**value, "created_at": "2000-01-01T00:00:00Z", "expires_at": "2000-01-01T00:15:00Z"}
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        apply(target, scope, expired)
    target.execute(
        "UPDATE candidate_index_jobs SET updated_at=clock_timestamp() WHERE generation_id=%s",
        (accepted["generation_id"],),
    )
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        apply(target, scope, value)
    with psycopg.connect(RUNTIME) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("SELECT candidate_index_plan_recovery(%s,%s,%s)", (scope, Jsonb([]), "{}"))


def test_diagnostics_fenced_idempotent_closed_and_runtime_cannot_read(target):
    current = replace(POLICY, **policy_fields())
    repo, scope, accepted, args = source(policy=current)
    d = repo.reserve(repo.claim("extract")[0], current)
    repo.record_diagnostic(d, "input_checked", {"input_characters": 20})
    repo.record_diagnostic(d, "input_checked", {"input_characters": 20})
    with pytest.raises(psycopg.errors.UniqueViolation):
        repo.record_diagnostic(d, "input_checked", {"input_characters": 21})
    with pytest.raises(ValueError):
        repo.record_diagnostic({**d, "lease_token": str(uuid4())}, "response_received", {})
    assert repo.fail(d, "output_truncated")
    rows = target.execute(
        "SELECT candidate_index_read_diagnostics(%s,%s)", (scope, accepted["generation_id"])
    ).fetchall()
    assert {row[0]["event"] for row in rows} == {"reserved", "input_checked", "settled"}
    assert (
        target.execute(
            "SELECT candidate_index_read_diagnostics(%s,%s)",
            ("org_99999", accepted["generation_id"]),
        ).fetchall()
        == []
    )
    with psycopg.connect(RUNTIME) as c, pytest.raises(psycopg.errors.InsufficientPrivilege):
        c.execute("SELECT * FROM candidate_index_attempt_diagnostics")
    directive({**args, "scope": scope}, CandidatePrivacyAction.REQUEST_ERASURE)
    assert (
        target.execute(
            "SELECT candidate_index_read_diagnostics(%s,%s)", (scope, accepted["generation_id"])
        ).fetchall()
        == []
    )
    assert (
        target.execute(
            "SELECT count(*) FROM candidate_index_attempt_diagnostics WHERE generation_id=%s",
            (accepted["generation_id"],),
        ).fetchone()[0]
        == 3
    )


def test_erasure_blocks_apply_and_receipt_replay(target):
    _, scope, accepted, args = failed_source()
    value = plan(target, scope, accepted)
    directive({**args, "scope": scope}, CandidatePrivacyAction.REQUEST_ERASURE)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        apply(target, scope, value)
    _, scope, accepted, args = failed_source()
    value = plan(target, scope, accepted)
    apply(target, scope, value)
    directive({**args, "scope": scope}, CandidatePrivacyAction.REQUEST_ERASURE)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        apply(target, scope, value)


def test_concurrent_apply_and_lost_reply_replay_create_only_one_generation(target):
    _, scope, accepted, _ = failed_source()
    value = plan(target, scope, accepted)
    ready = Barrier(2)

    def concurrent_apply():
        with psycopg.connect(OWNER, autocommit=True) as connection:
            ready.wait(timeout=5)
            return apply(connection, scope, value)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: concurrent_apply(), range(2)))
    assert sorted(item["replayed"] for item in results) == [False, True]
    assert results[0]["members"] == results[1]["members"]
    # A new process/connection after a lost commit reply sees the same receipt.
    with psycopg.connect(OWNER, autocommit=True) as connection:
        assert apply(connection, scope, value)["members"] == results[0]["members"]
    assert target.execute(
        "SELECT count(*) FROM candidate_index_generations WHERE source_id=%s",
        (accepted["source_id"],),
    ).fetchone() == (2,)


def test_outer_rollback_removes_entire_recovery_and_receipt(target):
    _, scope, accepted, _ = failed_source()
    value = plan(target, scope, accepted)
    with pytest.raises(RuntimeError, match="simulated owner failure"):
        with target.transaction():
            result = apply(target, scope, value)
            assert not result["replayed"]
            raise RuntimeError("simulated owner failure")
    assert target.execute(
        "SELECT count(*) FROM candidate_index_generations WHERE source_id=%s",
        (accepted["source_id"],),
    ).fetchone() == (1,)
    assert not apply(target, scope, value)["replayed"]


def test_recovery_adapter_identity_policy_and_configured_queue_limits(target):
    _, scope, first, _ = failed_source()
    _, _, second, _ = failed_source(scope)
    identity = target.execute(
        "SELECT target_id::text FROM activekg_schema_control.target_identity"
    ).fetchone()[0]
    policy = replace(POLICY, **policy_fields())
    service = ResumeRecovery(OWNER, identity, policy, IndexTunables(pending_per_scope=1))
    pairs = [
        {"source_id": item["source_id"], "generation_id": item["generation_id"]}
        for item in (first, second)
    ]
    envelope = service.plan(scope, pairs)
    assert envelope["plan"]["max_calls"] == 4
    with pytest.raises(ValueError, match="target_mismatch"):
        ResumeRecovery(OWNER, str(uuid4()), policy).plan(scope, pairs)
    with pytest.raises(ValueError, match="approval_mismatch"):
        service.apply(envelope, "0" * 64, 4)
    with pytest.raises(ValueError, match="approval_mismatch"):
        ResumeRecovery(OWNER, identity, POLICY).apply(envelope, envelope["plan_hash"], 4)
    result = service.apply(envelope, envelope["plan_hash"], 4)
    assert not result["replayed"]
    assert service.apply(envelope, envelope["plan_hash"], 4)["replayed"]
    states = target.execute(
        "SELECT state FROM candidate_index_jobs WHERE generation_id=ANY(%s::uuid[]) ORDER BY state",
        ([member["generation_id"] for member in result["members"]],),
    ).fetchall()
    assert states == [("pending_extraction",), ("waiting_admission",)]


def test_one_erased_member_refuses_entire_batch(target):
    _, scope, first, _ = failed_source()
    _, _, second, args = failed_source(scope)
    policy = replace(POLICY, **policy_fields())
    pairs = [
        {"source_id": item["source_id"], "generation_id": item["generation_id"]}
        for item in (first, second)
    ]
    value = target.execute(
        "SELECT candidate_index_plan_recovery(%s,%s,%s)",
        (scope, Jsonb(pairs), canonical_json(asdict(policy))),
    ).fetchone()[0]
    directive({**args, "scope": scope}, CandidatePrivacyAction.REQUEST_ERASURE)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        apply(target, scope, value, cap=4)
    for member in (first, second):
        assert target.execute(
            "SELECT count(*) FROM candidate_index_generations WHERE source_id=%s",
            (member["source_id"],),
        ).fetchone() == (1,)
    assert target.execute(
        "SELECT count(*) FROM candidate_index_recovery_receipts WHERE scope_key=%s", (scope,)
    ).fetchone() == (0,)


def v3_plan(target, scope, accepted):
    from activekg.candidate_index import resume_history_policy as history

    policy = replace(POLICY, **history.policy_fields())
    pairs = [{"source_id": accepted["source_id"], "generation_id": accepted["generation_id"]}]
    value = target.execute(
        "SELECT candidate_index_plan_recovery(%s,%s,%s)",
        (scope, Jsonb(pairs), canonical_json(asdict(policy))),
    ).fetchone()[0]
    return value, policy


def job_states(target, generation_id):
    return dict(
        target.execute(
            "SELECT stage,state FROM candidate_index_jobs WHERE generation_id=%s", (generation_id,)
        ).fetchall()
    )


def test_embedding_refusal_is_recovered_reextracted_and_published(target):
    """23 Oct: extraction succeeded, embedding refused (chunk_overflow) -> migration 038 recovery."""
    repo, scope, accepted, _ = source()
    base.extract(repo)
    embed = repo.reserve(repo.claim("embed")[0], POLICY)
    assert repo.fail(embed, "chunk_overflow")
    before = job_states(target, accepted["generation_id"])
    assert before["extract"] == "pending_embedding" and before["embed"] in {
        "failed",
        "needs_review",
    }
    attempts = target.execute(
        "SELECT stage,attempts FROM candidate_index_jobs WHERE generation_id=%s ORDER BY stage",
        (accepted["generation_id"],),
    ).fetchall()

    with target.transaction():
        target.execute("SET TRANSACTION READ ONLY")
        value, v3 = v3_plan(target, scope, accepted)
    assert value["items"][0]["state"] == "pending_embedding"
    assert (
        value["new_generations"] == 1 and value["max_calls"] == 2
    )  # re-extraction, not embed-only
    result = apply(target, scope, value)
    assert not result["replayed"] and apply(target, scope, value)["replayed"]

    # The superseded generation is retired, not reset; its history and counters stay.
    after = job_states(target, accepted["generation_id"])
    assert after == {"extract": "superseded", "embed": before["embed"]}
    assert (
        target.execute(
            "SELECT stage,attempts FROM candidate_index_jobs WHERE generation_id=%s ORDER BY stage",
            (accepted["generation_id"],),
        ).fetchall()
        == attempts
    )

    # The new generation extracts and embeds under the target policy, and is searchable.
    from activekg.candidate_index.contracts import chunk_manifest

    lease = repo.reserve(repo.claim("extract")[0], v3)
    assert lease and lease["generation_id"] == result["members"][0]["generation_id"]
    assert repo.complete_extract(
        lease,
        namespace={
            "current_title": "Backend engineer",
            "skills_normalized": ["python"],
            "employment_history": [],
        },
        evidence=[{"field": "current_title", "text": "Backend engineer"}],
        confidence=0.9,
        model_id=v3.primary_model_id,
        chunks=chunk_manifest(base.TEXT, v3),
        professional_text=base.TEXT,
    )
    embed = repo.reserve(repo.claim("embed")[0], v3)
    assert repo.complete_embed(embed, [base.VECTOR for _ in embed["extraction"]["chunks"]])
    assert repo.status(scope, accepted["source_id"])["state"] == "ready"
    with repo.transaction(scope) as cur:
        cur.execute(
            "SELECT candidate_index_read_private(%s::vector,%s,%s,100)",
            (str(base.VECTOR), v3.embedding_model_id, v3.embedding_artifact_revision),
        )
        rows = cur.fetchall()
    assert len(rows) == 1 and rows[0][0]["state"] == "ready"


def test_embedding_recovery_refuses_published_and_in_flight_generations(target):
    repo, scope, accepted, _ = source()
    base.publish(repo)  # fully indexed: nothing to recover
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        v3_plan(target, scope, accepted)
    repo2, scope2, accepted2, _ = source()
    base.extract(repo2)
    repo2.reserve(repo2.claim("embed")[0], POLICY)  # embedding leased, still running
    with pytest.raises(psycopg.errors.InvalidParameterValue):
        v3_plan(target, scope2, accepted2)


# 23 Oct rollout: a worker on a new policy must settle, not endlessly re-lease, old-policy work.


def _new_policy():
    from activekg.candidate_index import resume_history_policy as history

    return replace(POLICY, **history.policy_fields())


def _job(target, generation_id, stage):
    return target.execute(
        "SELECT state,error_code,attempts,dispatch_reserved_generation,lease_token IS NULL"
        " FROM candidate_index_jobs WHERE generation_id=%s AND stage=%s",
        (generation_id, stage),
    ).fetchone()


def test_new_policy_worker_settles_old_policy_embed_job_once(target):
    repo, scope, accepted, _ = source()
    base.extract(repo)  # extracted under the old policy during the deploy window
    lease = repo.claim("embed")[0]
    before = _job(target, accepted["generation_id"], "embed")
    assert repo.reserve(lease, _new_policy()) is None  # new worker, mismatched policy
    state, error, attempts, reserved, released = _job(target, accepted["generation_id"], "embed")
    assert (state, error, released) == ("needs_review", "policy_mismatch", True)
    assert (attempts, reserved) == before[2:4]  # no attempt, no reservation consumed
    assert repo.claim("embed") == []  # never re-leased
    assert repo.status(scope, accepted["source_id"])["state"] != "ready"  # not marked indexed


def test_policy_settlement_requires_the_current_lease_holder(target):
    repo, _, accepted, _ = source()
    base.extract(repo)
    lease = repo.claim("embed")[0]
    stale = {**lease, "lease_token": str(uuid4())}
    assert repo.reserve(stale, _new_policy()) is None
    assert _job(target, accepted["generation_id"], "embed")[:2] == ("embedding", None)  # untouched
    target.execute(
        "UPDATE candidate_index_jobs SET lease_expires_at=clock_timestamp()-interval '1 second'"
        " WHERE generation_id=%s AND stage='embed'",
        (accepted["generation_id"],),
    )
    assert repo.reserve(lease, _new_policy()) is None  # expired holder cannot settle either
    assert _job(target, accepted["generation_id"], "embed")[0] == "embedding"


def test_restart_reclaims_expired_old_policy_lease_then_settles(target):
    repo, _, accepted, _ = source()
    base.extract(repo)
    repo.claim("embed")  # old worker leased it, then restarted without settling
    target.execute(
        "UPDATE candidate_index_jobs SET lease_expires_at=clock_timestamp()-interval '1 second'"
        " WHERE generation_id=%s AND stage='embed'",
        (accepted["generation_id"],),
    )
    fresh = IndexRepository(RUNTIME)  # restarted new-policy worker
    lease = fresh.claim("embed")[0]
    assert lease["generation_id"] == accepted["generation_id"]
    assert fresh.reserve(lease, _new_policy()) is None
    assert _job(target, accepted["generation_id"], "embed")[:2] == (
        "needs_review",
        "policy_mismatch",
    )


def test_same_organisation_resume_progresses_after_old_policy_job_settles(target):
    repo, scope, old, _ = source()
    base.extract(repo)  # old-policy embed job queued first
    new = _new_policy()
    _, _, current, _ = source(scope, policy=new)
    lease = repo.claim("extract")[0]
    assert lease["generation_id"] == current["generation_id"]
    d = repo.reserve(lease, new)
    assert repo.complete_extract(
        d,
        namespace={
            "current_title": "Backend engineer",
            "skills_normalized": ["python"],
            "employment_history": [],
        },
        evidence=[{"field": "current_title", "text": "Backend engineer"}],
        confidence=0.9,
        model_id=new.primary_model_id,
        chunks=chunk_manifest(base.TEXT, new),
        professional_text=base.TEXT,
    )
    first = repo.claim("embed")[0]  # one embed slot per organisation: the old job comes first
    assert first["generation_id"] == old["generation_id"]
    assert repo.reserve(first, new) is None  # settles and frees the slot
    second = repo.claim("embed")[0]
    assert second["generation_id"] == current["generation_id"]
    e = repo.reserve(second, new)
    assert repo.complete_embed(e, [base.VECTOR for _ in e["extraction"]["chunks"]])
    assert repo.status(scope, current["source_id"])["state"] == "ready"


def test_settled_policy_mismatch_is_recovered_under_new_policy(target):
    repo, scope, accepted, _ = source()
    base.extract(repo)
    assert repo.reserve(repo.claim("embed")[0], _new_policy()) is None
    with target.transaction():
        target.execute("SET TRANSACTION READ ONLY")
        value, _ = v3_plan(target, scope, accepted)
    assert value["new_generations"] == 1 and value["max_calls"] == 2
    assert not apply(target, scope, value)["replayed"]
    assert _job(target, accepted["generation_id"], "extract")[0] == "superseded"


def test_new_policy_worker_settles_old_policy_extract_job(target):
    repo, scope, accepted, _ = source()  # captured under the old policy, not yet extracted
    lease = repo.claim("extract")[0]
    assert repo.reserve(lease, _new_policy()) is None
    state, error, attempts, _, released = _job(target, accepted["generation_id"], "extract")
    assert (state, error, attempts, released) == ("needs_review", "policy_mismatch", 0, True)
    value, _ = v3_plan(target, scope, accepted)
    assert value["new_generations"] == 1
