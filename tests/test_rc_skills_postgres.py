"""Real migrated Memory authority. Explicit disposable targets only; no provider."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict

from activekg.api.sourced_candidates import _privacy_tokens
from activekg.rc_skills.contracts import FinishRequest, ReadRequest, content_hash
from activekg.rc_skills.repository import RcRepository, RcUnavailable, catalog_ready
from scripts.init_railway_db import _harden_rc_skills_runtime_privileges

OWNER = os.getenv("ACTIVEKG_RC_TEST_OWNER_DSN")
UTC = timezone.utc
RUNTIME = os.getenv("ACTIVEKG_RC_TEST_RUNTIME_DSN")
pytestmark = pytest.mark.skipif(
    not OWNER or not RUNTIME, reason="explicit disposable RC DSNs required"
)


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    assert os.getenv("ACTIVEKG_RC_TEST_DISPOSABLE") == "1"
    for dsn in [OWNER, RUNTIME]:
        info = conninfo_to_dict(dsn)
        assert info["host"] == "127.0.0.1"
        assert info["dbname"].endswith("_test") and info["user"].endswith("_test")
    assert conninfo_to_dict(OWNER)["user"] != conninfo_to_dict(RUNTIME)["user"]
    monkeypatch.setenv("CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION", "1")
    monkeypatch.setenv(
        "CANDIDATE_PRIVACY_HMAC_KEY_V1", "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="
    )
    monkeypatch.setenv("CANDIDATE_PRIVACY_INTAKE_ENABLED", "false")


def person(*, public=True):
    identity = str(uuid4())
    url = f"https://linkedin.com/in/rc-test-{identity}"
    with psycopg.connect(OWNER, autocommit=True) as conn:
        conn.execute(
            "INSERT INTO public.global_candidates(id,linkedin_url) VALUES(%s,%s)", (identity, url)
        )
        if public:
            conn.execute(
                """INSERT INTO public.global_candidate_source_identities
                (provider_namespace,record_type,adapter_family,adapter_version,provider_record_id,
                 canonical_linkedin_url,global_candidate_id,verified_issuer,verified_actor_id,first_observed_at)
                VALUES('crustdata','person','crustdata_person',1,%s,%s,%s,'signal','signal-service',clock_timestamp())""",
                (str(uuid4().int % 10**17 + 1), url, identity),
            )
    return identity, url


def command(subject, *, age=10, outcome="success", observed_age=0, skills=None):
    now = datetime.now(UTC) - timedelta(seconds=1, days=observed_age)
    upstream = now - timedelta(days=age)

    def iso(value):
        return value.isoformat(timespec="microseconds").replace("+00:00", "Z")

    body = {
        "version": 1,
        "outcome": outcome,
        "identityHash": hashlib.sha256(subject[1].encode()).hexdigest(),
        "skills": (skills or ["AWS", "Café"]) if outcome == "success" else [],
        "upstreamAt": iso(upstream),
        "observedAt": iso(now),
        "validUntil": iso(upstream + timedelta(days=1095)) if outcome == "success" else None,
        "timestampSource": "check_last_update",
    }
    body["contentHash"] = content_hash(body)
    return FinishRequest.model_validate(
        {
            "version": 1,
            "globalCandidateId": subject[0],
            "originRun": str(uuid4()),
            "originAttempt": str(uuid4()),
            "observation": body,
        }
    )


def read(subject):
    return (
        RcRepository(RUNTIME)
        .read(
            "signal-service",
            ReadRequest.model_validate({"version": 1, "candidateIds": [subject[0]]}),
        )
        .candidates[0]
    )


def test_clock_skew_old_refusal_bounded_acceptance_and_far_future_refusal():
    subject = person()
    root = Path(__file__).resolve().parents[1]
    original = (root / "db/migrations/030_rc_skill_observations.sql").read_text()
    original = original[original.index("CREATE FUNCTION public.memory_rc_finish(") :]
    original = original[: original.index("FROM PUBLIC;") + len("FROM PUBLIC;")]
    original = original.replace("CREATE FUNCTION", "CREATE OR REPLACE FUNCTION", 1)
    with psycopg.connect(OWNER) as conn:

        def value(seconds):
            body = command(subject).model_dump(by_alias=True)
            at = conn.execute(
                "SELECT clock_timestamp()+%s*interval '1 second'", (seconds,)
            ).fetchone()[0]
            obs = body["observation"]
            obs["observedAt"] = at.isoformat(timespec="microseconds").replace("+00:00", "Z")
            obs["contentHash"] = content_hash({k: v for k, v in obs.items() if k != "contentHash"})
            return body

        def finish(body):
            return conn.execute(
                "SELECT memory_rc_finish('signal-service',%s,%s::jsonb)",
                (body["originAttempt"], json.dumps(body)),
            ).fetchone()[0]

        def refused(body):
            with pytest.raises(psycopg.errors.RaiseException, match="RC_INVALID_OBSERVATION"):
                with conn.transaction():
                    finish(body)

        conn.execute(original)
        refused(value(2))
        conn.execute((root / "db/migrations/031_rc_clock_tolerance.sql").read_text())
        refused(value(60))
        near = value(2)
        assert finish(near)["status"] == "stored"
        assert finish(near)["status"] == "replayed"
        conn.rollback()


def withdraw(subject, *, token_only=False, connection=None):
    def apply(conn):
        return conn.execute(
            """SELECT directive_id FROM public.candidate_privacy_create_directive(
            %s,%s,'withdraw_global_matching','global_matching','verified_candidate',%s,
            'candidate_global_opt_out','signal','signal-service',%s,NULL,NULL,1,%s::jsonb,false,clock_timestamp())""",
            (
                uuid4(),
                uuid4(),
                uuid4(),
                None if token_only else subject[0],
                json.dumps(_privacy_tokens(subject[1])),
            ),
        ).fetchone()

    if connection is not None:
        return apply(connection)
    with psycopg.connect(OWNER, autocommit=True) as conn:
        return apply(conn)


def test_migrated_catalog_and_runtime_denials():
    with psycopg.connect(RUNTIME, autocommit=True) as conn, conn.cursor() as cur:
        assert catalog_ready(cur)
        assert conn.execute(
            "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
        ).fetchone() == (False, False)
        for query in [
            "SELECT * FROM rc_skill_observations",
            "DELETE FROM rc_skill_observations",
            "TRUNCATE rc_skill_observations",
            "UPDATE rc_skill_observations SET skills='[]'",
        ]:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(query)


@pytest.mark.parametrize("outcome", ["success", "no_skills", "not_found", "unusable_age"])
def test_direct_fetch_provenance_roundtrip_and_replay(outcome):
    subject = person()
    raw = command(subject, outcome=outcome).model_dump(by_alias=True)
    raw["observation"]["timestampSource"] = "fetch_updated_at"
    raw["observation"]["contentHash"] = content_hash(
        {key: value for key, value in raw["observation"].items() if key != "contentHash"}
    )
    value = FinishRequest.model_validate(raw)
    repo = RcRepository(RUNTIME)
    assert repo.finish("signal-service", value).status == "stored"
    assert repo.finish("signal-service", value).status == "replayed"
    assert read(subject).observation.model_dump(by_alias=True) == raw["observation"]
    raw["observation"]["timestampSource"] = "check_last_update"
    raw["observation"]["contentHash"] = content_hash(
        {key: value for key, value in raw["observation"].items() if key != "contentHash"}
    )
    with pytest.raises(RcUnavailable):
        repo.finish("signal-service", FinishRequest.model_validate(raw))


def test_replay_concurrent_write_and_conflicting_run():
    subject = person()
    value = command(subject)
    repo = RcRepository(RUNTIME)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: repo.finish("signal-service", value), range(2)))
    assert sorted(r.status for r in results) == ["replayed", "stored"]
    assert results[0].receipt_id == results[1].receipt_id
    altered = value.model_copy(update={"origin_run": str(uuid4())})
    with pytest.raises(RcUnavailable, match="conflict"):
        repo.finish("signal-service", altered)
    assert read(subject).observation.content_hash == value.observation.content_hash


def test_newest_upstream_wins_not_arrival_or_no_data():
    subject = person()
    repo = RcRepository(RUNTIME)
    fresh = command(subject, age=1, skills=["AWS"])
    repo.finish("signal-service", fresh)
    repo.finish("signal-service", command(subject, age=30, skills=["Java"]))
    repo.finish("signal-service", command(subject, outcome="no_skills"))
    assert read(subject).observation.content_hash == fresh.observation.content_hash


@pytest.mark.parametrize("outcome", ["no_skills", "not_found", "unusable_age"])
def test_no_data_is_sticky(outcome):
    subject = person()
    value = command(subject, outcome=outcome, observed_age=1500)
    RcRepository(RUNTIME).finish("signal-service", value)
    result = read(subject)
    assert result.state == "no_data" and result.observation.skills == []


def test_expired_success_is_not_reused():
    subject = person()
    RcRepository(RUNTIME).finish("signal-service", command(subject, observed_age=1100))
    assert read(subject).state == "miss"


def test_private_and_unknown_candidates_are_not_cache_authority():
    subject = person(public=False)
    repo = RcRepository(RUNTIME)
    assert read(subject).state == "miss"
    with pytest.raises(RcUnavailable, match="unavailable"):
        repo.finish("signal-service", command(subject))
    assert read((str(uuid4()), "unused")).state == "miss"


@pytest.mark.parametrize("token_only", [False, True])
def test_withdrawal_rechecks_reads_and_finish_replay(token_only):
    subject = person()
    repo = RcRepository(RUNTIME)
    value = command(subject)
    repo.finish("signal-service", value)
    withdraw(subject, token_only=token_only)
    assert read(subject).state == "restricted"
    with pytest.raises(RcUnavailable, match="restricted"):
        repo.finish("signal-service", value)


def test_withdrawal_lock_fails_closed_during_uncommitted_change():
    subject = person()
    repo = RcRepository(RUNTIME)
    repo.finish("signal-service", command(subject))
    with psycopg.connect(OWNER) as conn:
        withdraw(subject, connection=conn)
        with pytest.raises(RcUnavailable, match="temporarily_unavailable"):
            read(subject)
    assert read(subject).state == "restricted"


def test_identity_mismatch_and_wrong_service_are_permanent():
    subject = person()
    other = person()
    value = command(other)
    wrong = value.model_copy(update={"global_candidate_id": subject[0]})
    with pytest.raises(RcUnavailable, match="conflict"):
        RcRepository(RUNTIME).finish("signal-service", wrong)
    with pytest.raises(RcUnavailable, match="denied"):
        RcRepository(RUNTIME).finish("different-service", command(subject))


def test_provisioner_revokes_column_drift_and_append_only_applies_to_owner():
    from psycopg import sql

    role = conninfo_to_dict(RUNTIME)["user"]
    with psycopg.connect(OWNER, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("GRANT SELECT(skills) ON rc_skill_observations TO {}").format(
                sql.Identifier(role)
            )
        )
        assert not catalog_ready(cur, role)
        _harden_rc_skills_runtime_privileges(cur, role)
        assert catalog_ready(cur, role)
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            cur.execute("TRUNCATE rc_skill_observations")


@pytest.mark.parametrize(
    "patch", [{"email": "never"}, {"version": True}, {"candidateIds": ["bad"]}]
)
def test_direct_read_routine_rejects_malformed_json(patch):
    body = {"version": 1, "candidateIds": [], **patch}
    with psycopg.connect(RUNTIME, autocommit=True) as conn:
        with pytest.raises(psycopg.errors.RaiseException, match="RC_INVALID_COMMAND"):
            conn.execute("SELECT memory_rc_read('signal-service',%s::jsonb)", (json.dumps(body),))
