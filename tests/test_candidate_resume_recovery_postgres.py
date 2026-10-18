"""Explicit disposable DB only; synthetic upstream evidence, no provider calls."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from threading import Barrier
from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb

from activekg.candidate_index.contracts import IndexTunables, canonical_json
from activekg.candidate_index.recovery import ResumeRecovery
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
