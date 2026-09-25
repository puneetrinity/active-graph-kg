"""Real 4B/3C/4D admission and 4E SQL on an explicitly disposable database.

No model extraction/embedding is run: pending embedding is a positive authority
control, not a test double for privacy or the history reducer.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import psycopg
import pytest
from fastapi import HTTPException
from psycopg.conninfo import conninfo_to_dict

from activekg.api import organization_candidates as intake
from activekg.api import organization_decision_events as events
from activekg.api.auth import JWTClaims
from activekg.candidate_history.contracts import TABLES, HistoryReadRequest
from activekg.candidate_history.repository import (
    HistoryRepository,
    HistoryUnavailable,
    catalog_evidence,
    catalog_ready,
)
from activekg.candidate_index.contracts import IndexPolicy, sha256
from activekg.candidate_index.repository import IndexRepository
from scripts import init_railway_db as release
from tests.test_organization_candidate_intake import _body
from tests.test_organization_decision_events import _payload

OWNER = os.getenv("ACTIVEKG_HISTORY_TEST_OWNER_DSN")
RUNTIME = os.getenv("ACTIVEKG_HISTORY_TEST_RUNTIME_DSN")
pytestmark = pytest.mark.skipif(
    not OWNER or not RUNTIME, reason="disposable history DSNs not configured"
)
TEXT = "Synthetic backend development experience."
POLICY = IndexPolicy(
    extraction_schema_sha256="1" * 64,
    extraction_prompt_sha256="2" * 64,
    primary_model_id="fixture-primary",
    fallback_model_id="fixture-fallback",
    embedding_model_id="fixture-384",
    embedding_artifact_revision="3" * 40,
)


@pytest.fixture(scope="module")
def target():
    assert os.getenv("ACTIVEKG_HISTORY_TEST_DISPOSABLE") == "1"
    for dsn in (OWNER, RUNTIME):
        info = conninfo_to_dict(dsn)
        assert (
            info["host"] == "127.0.0.1"
            and info["dbname"].endswith("_test")
            and info["user"].endswith("_test")
        )
    assert conninfo_to_dict(OWNER)["user"] != conninfo_to_dict(RUNTIME)["user"]
    with psycopg.connect(OWNER, autocommit=True) as conn:
        with conn.cursor() as cur:
            release._harden_candidate_history_runtime_privileges(
                cur, conninfo_to_dict(RUNTIME)["user"]
            )
        yield conn


@pytest.fixture(autouse=True)
def environment(target, monkeypatch):
    monkeypatch.setenv("ACTIVEKG_DSN", RUNTIME)
    monkeypatch.setenv("CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION", "1")
    monkeypatch.setenv(
        "CANDIDATE_PRIVACY_HMAC_KEY_V1", "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="
    )
    monkeypatch.setenv("CANDIDATE_PRIVACY_INTAKE_ENABLED", "false")
    monkeypatch.setenv("CANDIDATE_PRIVACY_FLOW_ISSUER", "vantahire")
    monkeypatch.setenv("CANDIDATE_PRIVACY_FLOW_ACTOR_ID", "vantahire-backend")
    monkeypatch.setenv("CANDIDATE_PRIVACY_SIGNAL_ISSUER", "signal")
    monkeypatch.setenv("CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID", "signal-service")


def source(app=None, org=None, job=None, *, capture=True):
    app = app or uuid4().int % 1000000000 + 1
    org, job = org or app, job or app
    scope = f"org_{org}"
    body = _body(
        application_id=app,
        job_id=job,
        extracted_text_sha256=sha256(TEXT),
        privacy_subject=[{"identifier_type": "vantahire_application_id", "value": str(app)}],
    )
    payload = intake.OrganizationCandidateIntakeRequest.model_validate_json(json.dumps(body))
    body["idempotency_key"] = intake.compute_idempotency_key(payload, scope)
    payload = intake.OrganizationCandidateIntakeRequest.model_validate_json(json.dumps(body))
    intake._store(
        payload,
        JWTClaims(
            tenant_id=scope,
            issuer="vantahire",
            actor_type="service",
            actor_id="vantahire-backend",
            scopes=["organization-candidate:write"],
        ),
    )
    result = {
        "org": org,
        "app": app,
        "job": job,
        "scope": scope,
        "reference": str(payload.reference_id),
    }
    if capture:
        result["source"] = capture_source(payload, scope)
    else:
        result["payload"] = payload
    return result


def capture_source(payload, scope):
    index = IndexRepository(RUNTIME)
    with index.transaction(scope) as cur:
        result = index.capture_on_cursor(
            cur,
            kind="organization_application",
            upstream_id=str(payload.resume_version_id),
            command_digest=sha256("fixture:" + str(payload.resume_version_id)),
            content_kind="pinned_text",
            content=TEXT,
            tokens=intake._privacy_tokens(payload.privacy_subject),
            key_versions=[1],
            policy=POLICY,
        )
    return result["source_id"]


def event(target, s, stage=8):
    seq = target.execute(
        "SELECT coalesce(max(source_event_sequence),0)+10 FROM organization_decision_event_inbox"
    ).fetchone()[0]
    payload = events.OrganizationDecisionEvent.model_validate_json(
        json.dumps(
            _payload(
                event_id=str(uuid4()),
                organization_id=s["org"],
                subject_id=s["app"],
                job_id=s["job"],
                delivery_sequence=seq,
                source_event_sequence=seq,
                after_state={"stage_id": stage},
            )
        )
    )
    events._store(payload, s["scope"])
    return payload


def command(s, e=None, count=0):
    return HistoryReadRequest.model_validate_json(
        json.dumps(
            {
                "schema_version": 1,
                "organization_id": s["org"],
                "application_id": s["app"],
                "job_id": s["job"],
                "reference_id": s["reference"],
                "expected": {
                    "count": str(count),
                    "event_id": str(e.event_id) if e else None,
                    "sequence": str(e.source_event_sequence) if e else None,
                },
            }
        )
    )


def drain(s, count):
    repo = HistoryRepository(RUNTIME)
    for _ in range(100):
        repo.step(50, 10)
        with psycopg.connect(OWNER) as conn:
            row = conn.execute(
                "SELECT observed_count FROM organization_candidate_history_applications WHERE tenant_id=%s AND application_id=%s",
                (s["scope"], s["app"]),
            ).fetchone()
        if row and row[0] == count:
            return
    pytest.fail("history did not reach expected count")


def test_live_reference_pending_embedding_applies_once_and_reads(target):
    s = source()
    e = event(target, s)
    repo = HistoryRepository(RUNTIME)
    before = repo.read(command(s, e, 1))
    assert before.authority_status == "eligible" and before.freshness.status == "projection_pending"
    drain(s, 1)
    result = repo.read(command(s, e, 1))
    assert result.freshness.status == "caught_up_to_observed_capture"
    assert result.summary.latest_observed_stage_id == 8
    for _ in range(3):
        repo.step(50, 10)
    assert repo.read(command(s, e, 1)).summary.observed_stage_move_count == "1"
    newer = event(target, s, 9)
    drain(s, 2)
    assert repo.read(command(s, e, 1)).freshness.status == "history_changed_retry"
    assert repo.read(command(s, newer, 2)).summary.latest_observed_stage_id == 9


def test_declared_owner_kind_and_runtime_are_real(target):
    expected = os.getenv("ACTIVEKG_HISTORY_EXPECT_SUPERUSER", "0") == "1"
    assert target.execute(
        "SELECT rolsuper FROM pg_roles WHERE rolname=current_user"
    ).fetchone() == (expected,)
    with psycopg.connect(RUNTIME) as conn:
        assert conn.execute(
            "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
        ).fetchone() == (False, False)
    rows = target.execute(
        "SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE relname=ANY(%s)",
        (list(TABLES),),
    ).fetchall()
    assert len(rows) == 5 and all(row == (True, True) for row in rows)


def test_empty_does_not_mean_complete_history_and_forged_keys_are_hidden(target):
    s = source()
    repo = HistoryRepository(RUNTIME)
    result = repo.read(command(s))
    assert (
        result.freshness.status == "no_captured_stage_events"
        and not result.coverage.historical_complete
    )
    for key in ("org", "app", "job", "reference"):
        forged = {**s, key: str(uuid4()) if key == "reference" else s[key] + 1}
        with pytest.raises(HistoryUnavailable, match="not_found"):
            repo.read(command(forged))


def test_missing_reference_waits_then_binds_and_late_older_event_counts(target):
    app = uuid4().int % 1000000000 + 1
    s = {"org": app, "app": app, "job": app, "scope": f"org_{app}"}
    older = event(target, s, 7)
    repo = HistoryRepository(RUNTIME)
    for _ in range(100):
        repo.step(50, 10)
        row = target.execute(
            "SELECT state FROM organization_candidate_history_event_state WHERE event_id=%s",
            (older.event_id,),
        ).fetchone()
        if row:
            break
    assert row == ("waiting_reference",)
    s = source(app)
    newer = event(target, s, 9)
    drain(s, 1)
    target.execute(
        "UPDATE organization_candidate_history_event_state SET next_attempt_at=clock_timestamp() WHERE event_id=%s",
        (older.event_id,),
    )
    drain(s, 2)
    result = repo.read(command(s, newer, 2))
    assert result.summary.first.event_id == older.event_id
    assert (
        result.summary.latest.event_id == newer.event_id
        and result.summary.latest_observed_stage_id == 9
    )


def test_reference_without_source_waits_then_applies_exactly_once(target):
    s = source(capture=False)
    e = event(target, s)
    repo = HistoryRepository(RUNTIME)
    for _ in range(100):
        repo.step(50, 10)
        state = target.execute(
            "SELECT state FROM organization_candidate_history_event_state WHERE event_id=%s",
            (e.event_id,),
        ).fetchone()
        if state:
            break
    assert state == ("waiting_source",)
    assert target.execute(
        "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=%s",
        (e.event_id,),
    ).fetchone() == (0,)
    s["source"] = capture_source(s.pop("payload"), s["scope"])
    target.execute(
        "UPDATE organization_candidate_history_event_state SET next_attempt_at=clock_timestamp() WHERE event_id=%s",
        (e.event_id,),
    )
    drain(s, 1)
    for _ in range(3):
        repo.step(50, 10)
    assert repo.read(command(s, e, 1)).summary.observed_stage_move_count == "1"


def test_same_organization_distinct_applications_never_merge(target):
    first = source()
    second = source(org=first["org"], job=first["job"] + 1)
    a, b = event(target, first, 8), event(target, second, 12)
    drain(first, 1)
    drain(second, 1)
    repo = HistoryRepository(RUNTIME)
    assert repo.read(command(first, a, 1)).summary.latest_observed_stage_id == 8
    assert repo.read(command(second, b, 1)).summary.latest_observed_stage_id == 12
    assert target.execute(
        "SELECT count(DISTINCT candidate_id) FROM organization_candidate_history_bindings "
        "WHERE event_id=ANY(%s)",
        ([a.event_id, b.event_id],),
    ).fetchone() == (2,)
    with pytest.raises(HistoryUnavailable, match="not_found"):
        repo.read(command({**first, "reference": second["reference"]}, a, 1))


def test_disconnect_before_commit_rolls_back_and_after_commit_does_not_recount(target):
    s = source()
    e = event(target, s)
    # Exercise the actual SQL routine under the runtime role. Disconnecting
    # without COMMIT models process death after SQL returned but before commit.
    conn = psycopg.connect(RUNTIME)
    try:
        for _ in range(100):
            conn.execute("SELECT organization_candidate_history_step(50,10)")
            view = conn.execute(
                "SELECT organization_candidate_history_read(%s,%s,%s,%s,%s,%s,1)",
                (
                    s["scope"],
                    s["app"],
                    s["job"],
                    s["reference"],
                    e.source_event_sequence,
                    e.event_id,
                ),
            ).fetchone()[0]
            if (view.get("summary") or {}).get("observed_stage_move_count") == "1":
                break
        else:
            pytest.fail("target event not reached before simulated disconnect")
    finally:
        conn.close()
    assert target.execute(
        "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=%s",
        (e.event_id,),
    ).fetchone() == (0,)
    drain(s, 1)
    # New repository/connection after commit: no process-local success state.
    restarted = HistoryRepository(RUNTIME)
    for _ in range(4):
        restarted.step(50, 10)
    assert restarted.read(command(s, e, 1)).summary.observed_stage_move_count == "1"
    assert target.execute(
        "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=%s",
        (e.event_id,),
    ).fetchone() == (1,)


@pytest.mark.parametrize(
    "committed", [False, True], ids=["kill-before-commit", "kill-after-commit"]
)
def test_real_process_death_at_projection_commit(target, committed):
    s = source()
    e = event(target, s)
    program = r"""
import json,os,sys,time,psycopg
p=json.loads(sys.stdin.readline())
c=psycopg.connect(os.environ["FIXTURE_RUNTIME"])
c.execute("SET LOCAL statement_timeout='3s'")
for _ in range(200):
 c.execute("SELECT organization_candidate_history_step(50,10)")
 r=c.execute("SELECT organization_candidate_history_read(%s,%s,%s,%s,%s,%s,1)",p["keys"]).fetchone()[0]
 if (r.get("summary") or {}).get("observed_stage_move_count")=="1":break
else:raise RuntimeError("fixture_target_not_reached")
if p["commit"]:c.commit()
print(json.dumps({"ready":True,"backend":c.info.backend_pid}),flush=True)
time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-B", "-c", program],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "FIXTURE_RUNTIME": RUNTIME,
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    try:
        child.stdin.write(
            json.dumps(
                {
                    "keys": [
                        s["scope"],
                        s["app"],
                        s["job"],
                        s["reference"],
                        e.source_event_sequence,
                        str(e.event_id),
                    ],
                    "commit": committed,
                }
            )
            + "\n"
        )
        child.stdin.flush()
        assert select.select([child.stdout], [], [], 20)[0], (
            "projection child did not reach commit boundary"
        )
        ready = json.loads(child.stdout.readline())
        assert ready["ready"] is True
        child.kill()
        child.wait(timeout=5)
        for _ in range(200):
            if target.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE pid=%s", (ready["backend"],)
            ).fetchone() == (0,):
                break
            time.sleep(0.01)
        assert target.execute(
            "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=%s",
            (e.event_id,),
        ).fetchone() == (int(committed),)
        drain(s, 1)
        assert (
            HistoryRepository(RUNTIME).read(command(s, e, 1)).summary.observed_stage_move_count
            == "1"
        )
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


def test_runtime_readonly_acl_and_catalog_names(target):
    role = conninfo_to_dict(RUNTIME)["user"]
    with target.cursor() as cur:
        digest, acl = catalog_evidence(cur, role)
        assert len(digest) == 64 and acl
    for relation in TABLES:
        with psycopg.connect(RUNTIME) as conn:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(f"SELECT 1 FROM public.{relation} LIMIT 1")
    for (name,) in target.execute(
        "SELECT conname FROM pg_constraint WHERE conrelid IN (SELECT oid FROM pg_class WHERE relname=ANY(%s))",
        (list(TABLES),),
    ):
        assert name.startswith("och_") and len(name.encode()) <= 63
    s = source()
    event(target, s)
    with psycopg.connect(RUNTIME) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            conn.execute("SELECT organization_candidate_history_step(50,10)")
    with psycopg.connect(RUNTIME) as conn:
        conn.execute("SET TRANSACTION READ ONLY")
        conn.execute("SELECT set_config('app.current_tenant_id','original-scope',true)")
        row = conn.execute(
            "SELECT organization_candidate_history_read(%s,%s,%s,%s,NULL,NULL,0)",
            (s["scope"], s["app"], s["job"], s["reference"]),
        ).fetchone()
        assert row[0]["authority_status"] == "eligible"
        assert (
            conn.execute("SELECT current_setting('app.current_tenant_id')").fetchone()[0]
            == "original-scope"
        )
        denied = conn.execute(
            "SELECT organization_candidate_history_read(%s,%s,%s,%s,NULL,NULL,0)",
            (s["scope"], s["app"], s["job"], uuid4()),
        ).fetchone()[0]
        assert denied == {"error": "not_found"}
        with pytest.raises(psycopg.errors.InvalidParameterValue), conn.transaction():
            conn.execute(
                "SELECT organization_candidate_history_read(%s,0,%s,%s,NULL,NULL,0)",
                (s["scope"], s["job"], s["reference"]),
            )
        assert conn.execute("SELECT current_setting('app.current_tenant_id')").fetchone() == (
            "original-scope",
        )
        for scope in ("", "org_2147483647"):
            conn.execute("SELECT set_config('app.current_tenant_id',%s,true)", (scope,))
            assert conn.execute(
                "SELECT candidate_index_status(%s,true)", (s["source"],)
            ).fetchone() == (None,)
        conn.execute("SELECT set_config('app.current_tenant_id',%s,true)", (s["scope"],))
        assert conn.execute("SELECT candidate_index_status(%s,true)", (s["source"],)).fetchone()[0][
            "eligible"
        ]


def test_real_lock_timeout_rolls_back_all_projection_work(target):
    s = source()
    e = event(target, s)
    with psycopg.connect(OWNER) as blocker:
        blocker.execute(
            "LOCK TABLE organization_candidate_history_bindings IN ACCESS EXCLUSIVE MODE"
        )
        with pytest.raises(HistoryUnavailable, match="temporarily_unavailable"):
            HistoryRepository(RUNTIME).step(50, 10)
    assert target.execute(
        "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=%s",
        (e.event_id,),
    ).fetchone() == (0,)
    drain(s, 1)
    assert (
        HistoryRepository(RUNTIME).read(command(s, e, 1)).summary.observed_stage_move_count == "1"
    )


def test_mutated_delivery_refuses_and_projection_conserves_all_upstream_rows(target):
    s = source()
    e = event(target, s)
    tables = (
        "organization_decision_event_inbox",
        "organization_decision_stream_state",
        "organization_candidate_references",
        "organization_candidate_resume_evidence",
        "organization_candidate_ingest_receipts",
        "candidates",
        "candidate_index_sources",
        "candidate_index_generations",
        "candidate_index_heads",
        "candidate_index_jobs",
    )

    def inventory():
        from psycopg import sql

        result = {}
        with target.transaction():
            target.execute("SELECT set_config('app.current_tenant_id',%s,true)", (s["scope"],))
            for table in tables:
                rows = target.execute(
                    sql.SQL("SELECT row_to_json(t) FROM {} t").format(sql.Identifier(table))
                ).fetchall()
                material = sorted(json.dumps(row, sort_keys=True, default=str) for row in rows)
                result[table] = (
                    len(rows),
                    hashlib.sha256(json.dumps(material).encode()).hexdigest(),
                )
        return result

    before = inventory()
    drain(s, 1)
    replay = events._store(e, s["scope"])
    assert replay is not None
    altered = e.model_copy(
        update={"after_state": e.after_state.model_copy(update={"stage_id": 99})}
    )
    with pytest.raises(HTTPException) as refused:
        events._store(altered, s["scope"])
    assert refused.value.status_code == 409
    assert inventory() == before
    assert (
        HistoryRepository(RUNTIME).read(command(s, e, 1)).summary.observed_stage_move_count == "1"
    )


def test_unresolved_and_conflicting_tenants_do_not_starve_healthy_work(target):
    missing_id = uuid4().int % 1000000000 + 1
    missing = {
        "org": missing_id,
        "app": missing_id,
        "job": missing_id,
        "scope": f"org_{missing_id}",
    }
    unresolved = [event(target, missing, stage) for stage in range(1, 8)]
    conflict = source()
    bad = event(target, {**conflict, "job": conflict["job"] + 1})
    healthy = source()
    good = event(target, healthy, 12)
    drain(healthy, 1)
    repo = HistoryRepository(RUNTIME)
    for _ in range(40):
        repo.step(50, 10)
    assert repo.read(command(healthy, good, 1)).summary.latest_observed_stage_id == 12
    assert target.execute(
        "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=ANY(%s)",
        ([e.event_id for e in unresolved] + [bad.event_id],),
    ).fetchone() == (0,)
    assert target.execute(
        "SELECT count(*) FROM organization_candidate_history_event_state WHERE event_id=ANY(%s) AND state='waiting_reference'",
        ([e.event_id for e in unresolved],),
    ).fetchone() == (7,)
    assert target.execute(
        "SELECT state FROM organization_candidate_history_event_state WHERE event_id=%s",
        (bad.event_id,),
    ).fetchone() == ("binding_conflict",)


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("missing", "waiting_source"),
        ("contended", "privacy_wait"),
        ("review", "privacy_wait"),
        ("unavailable", "privacy_wait"),
        ("restricted", "privacy_restricted"),
        ("superseded", "waiting_source"),
        ("consent", "binding_conflict"),
        ("wrong-source", "binding_conflict"),
        ("bad-boolean", "privacy_wait"),
        ("unknown", "privacy_wait"),
        ("eligible", "applied"),
    ],
    ids=[
        "missing",
        "contended",
        "review",
        "unavailable",
        "restricted",
        "superseded",
        "consent",
        "wrong-source",
        "bad-boolean",
        "unknown",
        "eligible",
    ],
)
def test_sql_status_mapping_and_scope_restoration(target, variant, expected):
    from psycopg import sql

    s = source()
    e = event(target, s)
    values = {
        "missing": None,
        "contended": {"eligible": False, "reason": "privacy_unavailable", "contended": True},
        "review": {"eligible": False, "reason": "privacy_review"},
        "unavailable": {"eligible": False, "reason": "privacy_unavailable"},
        "restricted": {"eligible": False, "reason": "privacy_restricted"},
        "superseded": {"eligible": False, "reason": "superseded"},
        "consent": {"eligible": False, "reason": "consent_changed"},
        "wrong-source": {"eligible": True, "reason": None, "source_id": str(uuid4())},
        "bad-boolean": {"eligible": "true"},
        "unknown": {"eligible": False, "reason": "unknown"},
        "eligible": {"eligible": True, "reason": None, "source_id": str(s["source"])},
    }
    original = target.execute(
        "SELECT pg_get_functiondef('candidate_index_status(uuid,boolean)'::regprocedure)"
    ).fetchone()
    # Transaction-local dependency mutation only. The real status routine is
    # exercised by all other SQL/process proofs and restored byte-for-byte.
    with target.transaction():
        target.execute("SELECT set_config('app.current_tenant_id',%s,true)", (s["scope"],))
        assert target.execute("SELECT candidate_index_status(%s,true)", (s["source"],)).fetchone()[
            0
        ]["eligible"]
        target.execute("SELECT set_config('app.current_tenant_id','original-scope',true)")
        body = (
            "BEGIN RETURN "
            + (
                "NULL"
                if values[variant] is None
                else sql.Literal(json.dumps(values[variant])).as_string(target) + "::jsonb"
            )
            + "; END"
        )
        target.execute(
            sql.SQL(
                "CREATE OR REPLACE FUNCTION public.candidate_index_status(p_source_id uuid,p_nonblocking boolean DEFAULT false) "
                "RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS {}"
            ).format(sql.Literal(body))
        )
        for _ in range(150):
            target.execute("SELECT organization_candidate_history_step(50,10)")
            row = target.execute(
                "SELECT state FROM organization_candidate_history_event_state WHERE event_id=%s",
                (e.event_id,),
            ).fetchone()
            if row:
                break
        assert row == (expected,)
        assert target.execute("SELECT current_setting('app.current_tenant_id')").fetchone() == (
            "original-scope",
        )
        raise psycopg.Rollback()
    assert (
        target.execute(
            "SELECT pg_get_functiondef('candidate_index_status(uuid,boolean)'::regprocedure)"
        ).fetchone()
        == original
    )
    drain(s, 1)


def test_common_provisioner_preserves_history_routine_only_acl(target, monkeypatch):
    role = conninfo_to_dict(RUNTIME)["user"]
    monkeypatch.setenv("ACTIVEKG_RUNTIME_ROLE", role)
    monkeypatch.setenv("ACTIVEKG_RUNTIME_PASSWORD", "history-runtime-fixture")
    with target.transaction(), target.cursor() as cur:
        for _ in range(2):
            release._provision_runtime_role(cur)
            assert catalog_ready(cur, role)
        # Roll back the older packages' blanket grants too; this test isolates
        # 4E's carve-out without pretending to run their complete reconciliation.
        raise psycopg.Rollback()


@pytest.mark.parametrize("policy", ["och_inbox_owner_read", "och_stream_owner_read"])
@pytest.mark.parametrize("variant", ["runtime-role", "public-role", "wrong-command"])
def test_owner_policy_role_and_command_drift_refuses(target, policy, variant):
    from psycopg import sql

    table = (
        "organization_decision_event_inbox"
        if policy == "och_inbox_owner_read"
        else "organization_decision_stream_state"
    )
    role = conninfo_to_dict(RUNTIME)["user"]
    with target.transaction(), target.cursor() as cur:
        assert catalog_ready(cur, role)
        cur.execute(
            sql.SQL("DROP POLICY {} ON public.{}").format(
                sql.Identifier(policy), sql.Identifier(table)
            )
        )
        subject = (
            sql.SQL("PUBLIC")
            if variant == "public-role"
            else sql.Identifier(role if variant == "runtime-role" else target.info.user)
        )
        cur.execute(
            sql.SQL("CREATE POLICY {} ON public.{} FOR {} TO {} USING (true)").format(
                sql.Identifier(policy),
                sql.Identifier(table),
                sql.SQL("ALL" if variant == "wrong-command" else "SELECT"),
                subject,
            )
        )
        assert not catalog_ready(cur, role)
        raise psycopg.Rollback()
    with target.cursor() as cur:
        assert catalog_ready(cur, role)


def test_wrong_job_event_conflicts_without_binding_or_count(target):
    s = source()
    e = event(target, {**s, "job": s["job"] + 1})
    repo = HistoryRepository(RUNTIME)
    for _ in range(100):
        repo.step(50, 10)
        row = target.execute(
            "SELECT state FROM organization_candidate_history_event_state WHERE event_id=%s",
            (e.event_id,),
        ).fetchone()
        if row:
            break
    assert row == ("binding_conflict",)
    assert target.execute(
        "SELECT count(*) FROM organization_candidate_history_bindings WHERE event_id=%s",
        (e.event_id,),
    ).fetchone() == (0,)
    with pytest.raises(HistoryUnavailable, match="binding_conflict"):
        repo.read(command(s))


def test_binding_append_only_and_nonempty_truncate(target):
    s = source()
    event(target, s)
    drain(s, 1)
    for statement in (
        "UPDATE organization_candidate_history_bindings SET job_id=job_id",
        "DELETE FROM organization_candidate_history_bindings",
        "TRUNCATE organization_candidate_history_subjects,organization_candidate_history_applications,organization_candidate_history_bindings",
    ):
        with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState):
            target.execute(statement)


def test_concurrent_projectors_count_each_event_once(target):
    s = source()
    latest = None
    for stage in range(1, 5):
        latest = event(target, s, stage)

    def run_turns():
        repo = HistoryRepository(RUNTIME)
        return [repo.step(50, 10)["outcome"] for _ in range(16)]

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: run_turns(), range(4)))
    assert all(len(result) == 16 for result in results)
    # Overlapping privacy locks legitimately defer an event. Advance only this
    # synthetic tenant's retry deadline; do not reset its attempts or outcome.
    target.execute(
        "UPDATE organization_candidate_history_event_state SET next_attempt_at=clock_timestamp() "
        "WHERE tenant_id=%s AND state='privacy_wait' AND reason='source_contended'",
        (s["scope"],),
    )
    drain(s, 4)
    result = HistoryRepository(RUNTIME).read(command(s, latest, 4))
    assert result.summary.observed_stage_move_count == "4"
    assert result.summary.latest_observed_stage_id == 4
    assert target.execute(
        "SELECT count(*) FROM organization_candidate_history_bindings WHERE tenant_id=%s AND application_id=%s",
        (s["scope"], s["app"]),
    ).fetchone() == (4,)


def test_column_grant_drift_refuses_and_provisioner_removes_it(target):
    role = conninfo_to_dict(RUNTIME)["user"]
    from psycopg import sql

    with target.cursor() as cur:
        assert catalog_ready(cur, role)
        cur.execute(
            sql.SQL(
                "GRANT SELECT(event_id) ON organization_candidate_history_bindings TO {}"
            ).format(sql.Identifier(role))
        )
        try:
            assert not catalog_ready(cur, role)
        finally:
            release._harden_candidate_history_runtime_privileges(cur, role)
        assert catalog_ready(cur, role)


def test_privacy_lock_contention_is_retryable_not_a_denial(target):
    s = source()
    e = event(target, s)
    with psycopg.connect(OWNER) as blocker:
        blocker.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended("
            "'candidate-privacy-candidate:'||tenant_id||':'||candidate_id::text,0)) "
            "FROM candidate_index_sources WHERE source_id=%s",
            (s["source"],),
        )
        repo = HistoryRepository(RUNTIME)
        for _ in range(100):
            repo.step(50, 10)
            state = target.execute(
                "SELECT state,reason FROM organization_candidate_history_event_state WHERE event_id=%s",
                (e.event_id,),
            ).fetchone()
            if state:
                break
        assert state == ("privacy_wait", "source_contended")
        with pytest.raises(HistoryUnavailable, match="temporarily_unavailable"):
            repo.read(command(s, e, 1))
    target.execute(
        "UPDATE organization_candidate_history_event_state SET next_attempt_at=clock_timestamp() WHERE event_id=%s",
        (e.event_id,),
    )
    drain(s, 1)
    assert repo.read(command(s, e, 1)).summary.observed_stage_move_count == "1"


def test_rewind_failure_restores_history_rows_policies_and_ledger(target, monkeypatch):
    from tests import test_deploy_path_guards as retained

    assert target.execute(
        "SELECT count(*) FROM schema_migrations WHERE filename='029_organization_candidate_history.sql'"
    ).fetchone() == (1,), "this proof requires the real ledger-29 release"
    s = source()
    e = event(target, s)
    drain(s, 1)
    monkeypatch.setattr(retained, "OWNER_DSN", OWNER)
    role = conninfo_to_dict(RUNTIME)["user"]
    with target.cursor() as cur:
        before = catalog_evidence(cur, role)
    ledger = target.execute(
        "SELECT filename,checksum,baselined FROM schema_migrations ORDER BY filename"
    ).fetchall()
    binding = target.execute(
        "SELECT row_to_json(b) FROM organization_candidate_history_bindings b WHERE event_id=%s",
        (e.event_id,),
    ).fetchone()
    with pytest.raises(RuntimeError, match="injected_027_rewind_failure"):
        retained._rewind_021_tail(fail_after_consent_drop=True)
    with target.cursor() as cur:
        assert catalog_evidence(cur, role) == before
    assert (
        target.execute(
            "SELECT filename,checksum,baselined FROM schema_migrations ORDER BY filename"
        ).fetchall()
        == ledger
    )
    assert (
        target.execute(
            "SELECT row_to_json(b) FROM organization_candidate_history_bindings b WHERE event_id=%s",
            (e.event_id,),
        ).fetchone()
        == binding
    )
    assert (
        HistoryRepository(RUNTIME).read(command(s, e, 1)).summary.observed_stage_move_count == "1"
    )
