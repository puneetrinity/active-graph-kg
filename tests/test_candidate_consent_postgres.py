from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import psycopg
import pytest
from fastapi import HTTPException
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from activekg.api import candidate_consent as receiver
from activekg.api.auth import JWTClaims
from activekg.api.operational import ReadinessResult, bounded_readiness_check
from activekg.common.schema_control import SchemaControlError
from scripts import init_railway_db as release
from tests.test_candidate_consent import vector

OWNER_DSN = os.getenv("ACTIVEKG_CONSENT_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_CONSENT_TEST_RUNTIME_DSN")
pytestmark = pytest.mark.skipif(
    not OWNER_DSN or not RUNTIME_DSN, reason="disposable consent PostgreSQL DSNs not configured"
)


def cross_system_legacy_probe(payload):
    """Called by the real Flow HTTP proof, not a substitute receiver or privacy writer.

    Only the embedding encoder is doubled. Selection, source convergence, RLS, privacy and readers are real.
    Inputs and model text stay in process; stdout is one fixed success marker from the parent wrapper.
    """
    import hashlib
    import json
    import logging
    from unittest.mock import patch

    import numpy as np

    from activekg.api import global_memory as legacy
    from activekg.embedding import global_candidates as embedding

    dsn = os.environ["ACTIVEKG_DSN"]
    info = conninfo_to_dict(dsn)
    assert info.get("host") == "127.0.0.1" and info["dbname"].endswith("_test")
    assert payload["phase"] in {"consent_only", "bind", "withdrawn", "restricted"}
    subject = payload["subject_id"]
    tenant = payload["tenant_id"]
    approved, independent = payload["approved_skill"], payload["independent_skill"]
    logging.disable(logging.CRITICAL)

    def snapshot():
        with psycopg.connect(dsn) as conn:
            assert conn.execute(
                "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
            ).fetchone() == (False, False)
            conn.execute(
                "SELECT set_config('app.current_tenant_id',%s,true)", (f"candidate_{subject}",)
            )
            rows = {}
            for table in (
                "candidate_consent_state",
                "candidate_consent_sources",
                "candidate_consent_receipts",
            ):
                rows[table] = conn.execute(
                    sql.SQL(
                        "SELECT to_jsonb(t) FROM {} t WHERE subject_id=%s ORDER BY to_jsonb(t)::text"
                    ).format(sql.Identifier(table)),
                    (subject,),
                ).fetchall()
            gc_id = rows["candidate_consent_state"][0][0]["global_candidate_id"]
            profile = conn.execute(
                "SELECT to_jsonb(g) FROM global_candidates g WHERE id=%s", (gc_id,)
            ).fetchone()[0]
            return rows, gc_id, profile

    before, gc_id, initial = snapshot()
    assert approved in json.dumps(before)
    email_hash = hashlib.sha256(payload["email"].encode()).hexdigest()
    claims = JWTClaims(
        tenant_id=tenant,
        actor_id="vantahire-backend",
        actor_type="service",
        scopes=["kg:read"],
        issuer="vantahire",
    )
    calls = []

    class Encoder:
        def encode(self, texts):
            calls.extend(texts)
            assert not any(approved in text for text in texts)
            vectors = np.zeros((len(texts), 384), dtype=float)
            vectors[:, 0] = 1.0
            return vectors

    encoder = Encoder()
    worker = embedding.GlobalCandidateEmbedder(dsn, encoder, batch_size=1000)
    with (
        patch.object(legacy, "GLOBAL_MEMORY_ENABLED", True),
        patch.object(legacy, "_embedder", encoder),
        patch.object(
            embedding, "EMBED_VERSION", max(embedding.EMBED_VERSION, initial["embed_version"]) + 1
        ),
        patch.object(
            embedding,
            "PUBLIC_EMBED_VERSION",
            max(embedding.PUBLIC_EMBED_VERSION, initial["public_embed_version"]) + 1,
        ),
    ):
        if payload["phase"] == "consent_only":
            # Retired upsert cannot resolve even a matching anchor. Feedback appends evidence, never identity.
            with patch.object(
                legacy,
                "_find_existing_all",
                side_effect=AssertionError("identity resolver invoked"),
            ):
                with pytest.raises(HTTPException) as retired:
                    legacy.upsert_global_candidate(
                        legacy.GlobalCandidateUpsert(email_hash=email_hash), claims=claims
                    )
                assert retired.value.status_code == 410
                feedback = legacy.ingest_feedback_events(
                    legacy.FeedbackEventIngest(
                        events=[
                            legacy.FeedbackEvent(
                                tenant_id=tenant,
                                job_id="consent-fixture",
                                global_candidate_id=gc_id,
                                action="shortlisted",
                                event_id=str(uuid4()),
                            )
                        ]
                    ),
                    claims=claims,
                )
                assert feedback == {"inserted": 1, "skipped": 0}
                assert snapshot()[0] == before
        if payload["phase"] == "bind":
            legacy.sync_applicant_to_global_memory(
                node_id=str(uuid4()),
                tenant_id=tenant,
                node_props={
                    "applicant_email": payload["email"],
                    "applicant_name": "Independent fixture",
                },
                extracted_result=SimpleNamespace(
                    skills_normalized=[independent], functions=["engineering"], seniority="senior"
                ),
                metadata={
                    "provenance_type": "platform_applicant",
                    "visibility": "private",
                    "consent_state": "opted_out",
                },
            )
        worker._sweep_public_once()
        worker._sweep_once()
        after, same_id, current = snapshot()
        assert same_id == gc_id and before == after
        if payload["phase"] == "consent_only":
            assert current["embedding"] is None and current["embedding_status"] == "consent_pending"
            with pytest.raises(HTTPException) as denied:
                legacy.get_by_anchor(
                    linkedin_id=None, github_id=None, email_hash=email_hash, claims=claims
                )
            assert denied.value.status_code == 404
        else:
            assert current["skills_normalized"] == [independent]
            assert current["public_profile"] == {} and current["public_embedding"] is None
            if payload["phase"] == "restricted":
                assert not any(independent in text for text in calls)
                assert current["embedding"] == initial["embedding"]
                with pytest.raises(HTTPException) as blocked:
                    legacy.sync_applicant_to_global_memory(
                        node_id=str(uuid4()),
                        tenant_id=tenant,
                        node_props={"applicant_email": payload["email"]},
                        extracted_result=SimpleNamespace(skills_normalized=[independent]),
                        metadata={},
                    )
                assert blocked.value.status_code == 409
                assert blocked.value.detail == "candidate_privacy_restricted"
            else:
                assert any(independent in text for text in calls)
                assert current["embedding_status"] == "ready"
                assert (
                    legacy.get_by_anchor(
                        linkedin_id=None, github_id=None, email_hash=email_hash, claims=claims
                    )["id"]
                    == gc_id
                )
        # Run both shipped search predicates independently, not by weakening the API's dual-surface readiness.
        for surface in ("legacy_v0", "public_v1"):
            with (
                patch.object(legacy, "LEGACY_GLOBAL_SEARCH_ENABLED", surface == "legacy_v0"),
                patch.object(legacy, "PUBLIC_PROFILE_SEARCH_ENABLED", surface == "public_v1"),
            ):
                result = legacy.search_global_candidates(
                    legacy.GlobalCandidateSearchRequest(
                        query_text="fixture", limit=1000, surface=surface
                    ),
                    claims=claims,
                )
                matched = [row for row in result["results"] if str(row["id"]) == gc_id]
                expected = surface == "legacy_v0" and payload["phase"] in {"bind", "withdrawn"}
                assert bool(matched) == expected
                assert approved not in json.dumps(result, default=str)
        with patch.object(legacy, "PUBLIC_PROFILE_SEARCH_ENABLED", True):
            assert (
                legacy.resolve_public_identities(
                    legacy.PublicIdentityLookupRequest(
                        linkedin_urls=[payload["approved_linkedin"]]
                    ),
                    claims=claims,
                )["results"]
                == []
            )
            assert (
                legacy.public_candidate_exclusions(
                    legacy.PublicMarketExclusionRequest(coarse_market_key="fixture-only"),
                    claims=claims,
                )["crustdata_person_ids"]
                == []
            )
        slug = payload["approved_linkedin"].rsplit("/", 1)[-1]
        assert legacy.resume_refs(legacy.ResumeRefsRequest(linkedin_ids=[slug]), claims=claims) == {
            "refs": {}
        }
        with pytest.raises(HTTPException) as missing_social:
            legacy.get_by_anchor(linkedin_id=slug, github_id=None, email_hash=None, claims=claims)
        assert missing_social.value.status_code == 404
        assert snapshot()[0] == before


@pytest.fixture
def role_target(monkeypatch):
    assert OWNER_DSN and RUNTIME_DSN
    owner_info, runtime_info = map(conninfo_to_dict, (OWNER_DSN, RUNTIME_DSN))
    for info in (owner_info, runtime_info):
        assert info.get("host") in {"127.0.0.1", "::1"}
        assert info["dbname"].endswith("_test")
    assert all(owner_info.get(k) == runtime_info.get(k) for k in ("host", "port", "dbname"))
    assert owner_info["user"] != runtime_info["user"]
    monkeypatch.setenv("ACTIVEKG_RUNTIME_ROLE", runtime_info["user"])
    monkeypatch.setenv("ACTIVEKG_RUNTIME_PASSWORD", runtime_info["password"])
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        attributes = conn.execute(
            "SELECT rolsuper,rolcreaterole FROM pg_roles WHERE rolname=current_user"
        ).fetchone()
        assert attributes and any(attributes)
        if os.getenv("ACTIVEKG_CONSENT_REQUIRE_NON_SUPER_OWNER") == "1":
            assert attributes == (False, True)
        yield conn, runtime_info["user"]


def _harden_and_assert(cur, role):
    for package in (
        "candidate_privacy",
        "decision_inbox",
        "sourced_candidate",
        "organization_candidate",
        "candidate_consent",
    ):
        getattr(release, f"_harden_{package}_runtime_privileges")(cur, role)
    release._assert_runtime_role_catalog(cur, role)
    for package in (
        "candidate_privacy",
        "decision_inbox",
        "sourced_candidate",
        "organization_candidate",
        "candidate_consent",
    ):
        getattr(release, f"_assert_{package}_runtime_privileges")(cur, role)


def _privileges(conn):
    # Catalog evidence only; no profile rows or role-password catalog reads.
    return conn.execute("""
        SELECT 'relation',c.oid::text,coalesce(c.relacl::text,'') FROM pg_class c
        JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public'
        UNION ALL SELECT 'routine',p.oid::text,coalesce(p.proacl::text,'') FROM pg_proc p
        JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public'
        UNION ALL SELECT 'column',a.attrelid::text||':'||a.attnum::text,coalesce(a.attacl::text,'')
        FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
        JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND a.attnum>0
        UNION ALL SELECT 'default',oid::text,defaclacl::text FROM pg_default_acl
        ORDER BY 1,2
    """).fetchall()


def _runtime_works():
    assert RUNTIME_DSN
    with psycopg.connect(RUNTIME_DSN) as runtime:
        assert runtime.execute(
            "SELECT rolcanlogin,rolsuper,rolcreatedb,rolcreaterole,rolreplication,rolbypassrls "
            "FROM pg_roles WHERE rolname=current_user"
        ).fetchone() == (True, False, False, False, False, False)
        with runtime.cursor() as cur:
            release._assert_candidate_consent_runtime_privileges(
                cur, conninfo_to_dict(RUNTIME_DSN)["user"]
            )


def test_a2_real_existing_role_reconciliation_is_idempotent(role_target):
    conn, role = role_target
    before = _privileges(conn)
    for _ in range(2):
        with conn.transaction(), conn.cursor() as cur:
            release._provision_runtime_role(cur)
            _harden_and_assert(cur, role)
        assert _privileges(conn) == before
        _runtime_works()


def test_a2_missing_admin_option_refuses_and_rolls_back(role_target):
    conn, role = role_target
    before = _privileges(conn)
    unprivileged = "consent_role_admin_" + uuid4().hex[:12] + "_test"
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with conn.transaction(), conn.cursor() as cur:
            # Another CREATEROLE login still cannot administer this existing runtime.
            cur.execute(
                sql.SQL("CREATE ROLE {} LOGIN CREATEROLE").format(sql.Identifier(unprivileged))
            )
            cur.execute(
                sql.SQL("GRANT {} TO {} WITH SET TRUE").format(
                    sql.Identifier(unprivileged), sql.Identifier(conn.info.user)
                )
            )
            cur.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(unprivileged)))
            release._provision_runtime_role(cur)
    assert (
        conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (unprivileged,)).fetchone() is None
    )
    assert _privileges(conn) == before
    _runtime_works()


def test_a2_unsafe_existing_attributes_retain_old_refusal(role_target, monkeypatch):
    conn, _ = role_target
    before = _privileges(conn)
    elevated = "consent_elevated_" + uuid4().hex[:12] + "_test"
    actor = "consent_attribute_admin_" + uuid4().hex[:12] + "_test"
    monkeypatch.setenv("ACTIVEKG_RUNTIME_ROLE", elevated)
    with pytest.raises(psycopg.errors.InsufficientPrivilege) as refused:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE ROLE {} CREATEROLE").format(sql.Identifier(actor)))
            cur.execute(
                sql.SQL("GRANT {} TO {} WITH SET TRUE").format(
                    sql.Identifier(actor), sql.Identifier(conn.info.user)
                )
            )
            cur.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(actor)))
            cur.execute(sql.SQL("CREATE ROLE {} LOGIN CREATEROLE").format(sql.Identifier(elevated)))
            release._provision_runtime_role(cur)
    assert "SUPERUSER attribute" in refused.value.diag.message_detail
    assert conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (elevated,)).fetchone() is None
    assert _privileges(conn) == before


@pytest.mark.parametrize("reserved", ["postgres", "app_user", "admin_role", "self"])
def test_a2_reserved_runtime_names_still_refuse(role_target, monkeypatch, reserved):
    conn, _ = role_target
    before = _privileges(conn)
    monkeypatch.setenv("ACTIVEKG_RUNTIME_ROLE", conn.info.user if reserved == "self" else reserved)
    with pytest.raises(SystemExit):
        with conn.transaction(), conn.cursor() as cur:
            release._provision_runtime_role(cur)
    assert _privileges(conn) == before


def test_a2_indirect_admin_membership_refuses_and_rolls_back(role_target):
    conn, role = role_target
    before = _privileges(conn)
    intermediary = "consent_group_" + uuid4().hex[:12] + "_test"
    with pytest.raises(SystemExit):
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE ROLE {}").format(sql.Identifier(intermediary)))
            cur.execute(sql.SQL("GRANT admin_role TO {}").format(sql.Identifier(intermediary)))
            cur.execute(
                sql.SQL("GRANT {} TO {}").format(sql.Identifier(intermediary), sql.Identifier(role))
            )
            release._provision_runtime_role(cur)
    assert conn.execute("SELECT pg_has_role(%s,'admin_role','MEMBER')", (role,)).fetchone() == (
        False,
    )
    assert (
        conn.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (intermediary,)).fetchone() is None
    )
    assert _privileges(conn) == before
    _runtime_works()


def test_a2_owned_relation_still_fails_final_postconditions(role_target):
    conn, role = role_target
    before = _privileges(conn)
    with pytest.raises(SchemaControlError, match="runtime role posture"):
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                sql.SQL("GRANT {} TO {} WITH SET TRUE").format(
                    sql.Identifier(role), sql.Identifier(conn.info.user)
                )
            )
            cur.execute(sql.SQL("GRANT CREATE ON SCHEMA public TO {}").format(sql.Identifier(role)))
            cur.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role)))
            cur.execute("CREATE TABLE public.consent_owned_probe_test(id integer)")
            cur.execute("RESET ROLE")
            release._provision_runtime_role(cur)
            _harden_and_assert(cur, role)
    assert conn.execute("SELECT to_regclass('public.consent_owned_probe_test')").fetchone() == (
        None,
    )
    assert _privileges(conn) == before
    _runtime_works()


@pytest.fixture
def consent_target(role_target, monkeypatch):
    conn, role = role_target
    monkeypatch.setenv("ACTIVEKG_DSN", RUNTIME_DSN)
    monkeypatch.setenv("CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION", "1")
    monkeypatch.setenv(
        "CANDIDATE_PRIVACY_HMAC_KEY_V1", "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="
    )
    monkeypatch.setenv("CANDIDATE_PRIVACY_INTAKE_ENABLED", "false")
    monkeypatch.setenv("CANDIDATE_PRIVACY_FLOW_ISSUER", "vantahire")
    monkeypatch.setenv("CANDIDATE_PRIVACY_FLOW_ACTOR_ID", "vantahire-backend")
    monkeypatch.setenv("CANDIDATE_PRIVACY_SIGNAL_ISSUER", "signal")
    monkeypatch.setenv("CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID", "signal-service")
    _runtime_works()
    return conn, role


def _command(subject=None, version=1, action="grant", *, linkedin=None, email=None):
    body = vector()
    body.update(
        subject_id=subject or str(uuid4()), event_id=str(uuid4()), version=version, action=action
    )
    if action == "withdraw":
        body.update(source=None, proof=None)
    else:
        body["source"].update(source_id=str(uuid4()), source_version=version)
        body["source"]["profile"]["linkedin"] = linkedin
        address = email or f"{body['subject_id']}@fixture.invalid"
        body["proof"] = {
            "verified_email": address,
            "privacy_subject": [{"identifier_type": "email", "value": address}],
        }
    payload = receiver.ConsentCommand.model_validate(body)
    return payload.model_copy(update={"idempotency_key": receiver.command_identity(payload)[1]})


def _send(payload):
    claims = JWTClaims(
        issuer="vantahire",
        actor_id="vantahire-backend",
        actor_type="service",
        scopes=["candidate-consent:write"],
        tenant_id=f"candidate_{payload.subject_id}",
    )
    return receiver._store(payload, claims)


def _state(conn, subject):
    return conn.execute(
        "SELECT row_to_json(s) FROM candidate_consent_state s WHERE subject_id=%s", (subject,)
    ).fetchone()[0]


def _evidence_count(conn, subject):
    return conn.execute(
        "SELECT (SELECT count(*) FROM candidate_consent_sources WHERE subject_id=%s), "
        "(SELECT count(*) FROM candidate_consent_receipts WHERE subject_id=%s)",
        (subject, subject),
    ).fetchone()


def test_runtime_grant_replay_mutation_and_withdraw(consent_target):
    conn, _ = consent_target
    grant = _command()
    assert _send(grant)["outcome"] == "granted"
    state = _state(conn, grant.subject_id)
    assert state["effective_version"] == 1 and state["effective_action"] == "grant"
    person = conn.execute(
        "SELECT embedding_status,embedding,public_profile FROM global_candidates WHERE id=%s",
        (state["global_candidate_id"],),
    ).fetchone()
    assert person == ("consent_pending", None, {})
    assert _evidence_count(conn, grant.subject_id) == (1, 1)
    assert _send(grant)["outcome"] == "replayed"
    assert _state(conn, grant.subject_id) == state
    changed = grant.model_copy(deep=True)
    changed.source.profile.headline = "Different approval"
    with pytest.raises(HTTPException) as refused:
        _send(changed)
    assert refused.value.status_code == 409
    assert _evidence_count(conn, grant.subject_id) == (1, 1)
    withdrawal = _command(grant.subject_id, 2, "withdraw")
    assert _send(withdrawal)["outcome"] == "withdrawn"
    withdrawn = _state(conn, grant.subject_id)
    assert withdrawn["effective_version"] == 2 and withdrawn["active_source_id"] is None
    assert _send(grant)["outcome"] == "superseded"
    assert _state(conn, grant.subject_id) == withdrawn
    assert _evidence_count(conn, grant.subject_id) == (1, 2)


def test_withdraw_before_delayed_grant_never_creates_source(consent_target):
    conn, _ = consent_target
    grant = _command()
    assert _send(_command(grant.subject_id, 2, "withdraw"))["outcome"] == "withdrawn"
    before = _state(conn, grant.subject_id)
    assert _send(grant)["outcome"] == "superseded"
    assert _state(conn, grant.subject_id) == before
    assert _evidence_count(conn, grant.subject_id) == (0, 2)


def test_concurrent_exact_transport_converges_and_conflicting_version_refuses(consent_target):
    conn, _ = consent_target
    grant = _command()
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(_send, [grant, grant]))
    assert sorted(row["outcome"] for row in outcomes) == ["granted", "replayed"]
    assert _evidence_count(conn, grant.subject_id) == (1, 1)
    with pytest.raises(HTTPException) as refused:
        _send(_command(grant.subject_id))
    assert refused.value.status_code == 409
    assert _evidence_count(conn, grant.subject_id) == (1, 1)


def test_replacement_preserves_frozen_source_and_conflict_preserves_effective_state(consent_target):
    conn, _ = consent_target
    original = _command()
    assert _send(original)["outcome"] == "granted"
    approved = conn.execute(
        "SELECT row_to_json(s) FROM candidate_consent_sources s WHERE subject_id=%s",
        (original.subject_id,),
    ).fetchall()
    replacement = _command(original.subject_id, 2, email=original.proof.verified_email)
    replacement.source.profile.skills = ["Rust"]
    replacement.idempotency_key = receiver.command_identity(replacement)[1]
    assert _send(replacement)["outcome"] == "granted"
    assert (
        conn.execute(
            "SELECT row_to_json(s) FROM candidate_consent_sources s WHERE source_id=%s",
            (original.source.source_id,),
        ).fetchall()
        == approved
    )
    before = _state(conn, original.subject_id)
    slug = uuid4().hex
    conn.execute(
        "INSERT INTO global_candidates(linkedin_id,linkedin_url,embedding_status,public_profile) "
        "VALUES(%s,%s,'pending','{}')",
        (slug, f"https://www.linkedin.com/in/{slug}"),
    )
    conflict = _command(
        original.subject_id,
        3,
        email=original.proof.verified_email,
        linkedin=f"https://www.linkedin.com/in/{slug}",
    )
    assert _send(conflict)["outcome"] == "identity_review_required"
    after = _state(conn, original.subject_id)
    assert after["highest_version"] == 3
    for field in (
        "effective_version",
        "effective_action",
        "active_source_id",
        "global_candidate_id",
        "effective_at",
    ):
        assert after[field] == before[field]
    assert _evidence_count(conn, original.subject_id) == (2, 3)


def test_social_only_match_does_not_bind_canonical_identity(consent_target):
    conn, _ = consent_target
    slug = uuid4().hex
    conn.execute(
        "INSERT INTO global_candidates(linkedin_id,embedding_status,public_profile) VALUES(%s,'pending','{}')",
        (slug,),
    )
    command = _command(linkedin=f"https://www.linkedin.com/in/{slug}")
    assert _send(command)["outcome"] == "identity_review_required"
    assert _state(conn, command.subject_id)["global_candidate_id"] is None
    assert _evidence_count(conn, command.subject_id) == (0, 1)


def test_receipt_insert_failure_rolls_back_identity_source_and_state(consent_target):
    conn, role = consent_target
    payload = _command()
    try:
        conn.execute(
            sql.SQL("REVOKE INSERT ON candidate_consent_receipts FROM {}").format(
                sql.Identifier(role)
            )
        )
        with pytest.raises(HTTPException) as refused:
            _send(payload)
        assert refused.value.status_code == 503
        assert conn.execute(
            "SELECT count(*) FROM candidate_consent_state WHERE subject_id=%s",
            (payload.subject_id,),
        ).fetchone() == (0,)
        assert conn.execute(
            "SELECT count(*) FROM global_candidates WHERE email_hash=%s",
            (receiver.digest(payload.proof.verified_email),),
        ).fetchone() == (0,)
        assert _evidence_count(conn, payload.subject_id) == (0, 0)
    finally:
        conn.execute(
            sql.SQL("GRANT INSERT ON candidate_consent_receipts TO {}").format(sql.Identifier(role))
        )
    _runtime_works()


def test_force_rls_and_actual_append_only_privileges(consent_target):
    conn, _ = consent_target
    payload = _command()
    assert _send(payload)["outcome"] == "granted"
    with psycopg.connect(RUNTIME_DSN) as runtime:
        runtime.execute(
            "SELECT set_config('app.current_tenant_id',%s,true)", (f"candidate_{uuid4()}",)
        )
        for table in (
            "candidate_consent_state",
            "candidate_consent_sources",
            "candidate_consent_receipts",
        ):
            assert runtime.execute(
                sql.SQL("SELECT count(*) FROM {} WHERE subject_id=%s").format(
                    sql.Identifier(table)
                ),
                (payload.subject_id,),
            ).fetchone() == (0,)
        with pytest.raises(psycopg.errors.InsufficientPrivilege), runtime.transaction():
            other = str(uuid4())
            runtime.execute(
                "INSERT INTO candidate_consent_state(subject_id,tenant_id) VALUES(%s,%s)",
                (other, f"candidate_{other}"),
            )
        runtime.execute(
            "SELECT set_config('app.current_tenant_id',%s,true)",
            (f"candidate_{payload.subject_id}",),
        )
        for table in ("candidate_consent_sources", "candidate_consent_receipts"):
            for statement in (
                f"UPDATE {table} SET subject_id=subject_id WHERE subject_id=%s",
                f"DELETE FROM {table} WHERE subject_id=%s",
            ):
                with pytest.raises(psycopg.errors.InsufficientPrivilege), runtime.transaction():
                    runtime.execute(statement, (payload.subject_id,))
            with pytest.raises(psycopg.errors.InsufficientPrivilege), runtime.transaction():
                runtime.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(table)))
        assert (
            runtime.execute(
                "UPDATE candidate_consent_state SET highest_version=highest_version WHERE subject_id=%s",
                (payload.subject_id,),
            ).rowcount
            == 1
        )
        with pytest.raises(psycopg.errors.InsufficientPrivilege), runtime.transaction():
            runtime.execute(
                "UPDATE candidate_consent_state SET subject_id=subject_id WHERE subject_id=%s",
                (payload.subject_id,),
            )
    for table in ("candidate_consent_sources", "candidate_consent_receipts"):
        for statement in (
            f"UPDATE {table} SET subject_id=subject_id WHERE subject_id=%s",
            f"DELETE FROM {table} WHERE subject_id=%s",
        ):
            with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState), conn.transaction():
                conn.execute(statement, (payload.subject_id,))
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState), conn.transaction():
        conn.execute(
            "TRUNCATE candidate_consent_state,candidate_consent_sources,candidate_consent_receipts"
        )


def test_keyed_forged_resume_refuses_without_consent_writes_and_restores_context(consent_target):
    conn, _ = consent_target
    body = vector(True)
    body.update(subject_id=str(uuid4()), event_id=str(uuid4()))
    payload = receiver.ConsentCommand.model_validate(body)
    payload.idempotency_key = receiver.command_identity(payload)[1]
    with pytest.raises(HTTPException) as refused:
        _send(payload)
    assert refused.value.status_code == 409
    assert _evidence_count(conn, payload.subject_id) == (0, 0)
    with psycopg.connect(RUNTIME_DSN, row_factory=psycopg.rows.dict_row) as runtime:
        tenant = f"candidate_{payload.subject_id}"
        with runtime.cursor() as cur:
            cur.execute("SELECT set_config('app.current_tenant_id',%s,true)", (tenant,))
            statements = []

            class ObservedCursor:
                def execute(self, query, params):
                    statements.append((query, params))
                    return cur.execute(query, params)

                def fetchone(self):
                    return cur.fetchone()

            with pytest.raises(HTTPException):
                receiver._check_resume(ObservedCursor(), tenant, payload.source.resume)
            assert len(statements) == 3
            assert statements[0][1] == (f"org_{payload.source.resume.organization_id}",)
            assert statements[1][0].strip().startswith("SELECT EXISTS(")
            assert "v.resume_version_id=%s::uuid" in statements[1][0]
            assert statements[1][1][0] == payload.source.resume.resume_version_id
            assert statements[2][1] == (tenant,)
            assert (
                cur.execute("SELECT current_setting('app.current_tenant_id') AS tenant").fetchone()[
                    "tenant"
                ]
                == tenant
            )


def test_source_scalar_shapes_are_enforced_by_the_runtime_insert(consent_target):
    conn, _ = consent_target
    payload = _command()
    assert _send(payload)["outcome"] == "granted"
    profile = payload.source.profile.model_dump()
    resume = vector(True)["source"]["resume"]
    bad_profiles = [
        None,
        {},
        {**profile, "display_name": None},
        {**profile, "headline": []},
        {**profile, "skills": [None]},
        {**profile, "skills": ["x" * 101]},
        {**profile, "skills": [" leading"]},
        {**profile, "skills": ["line\nbreak"]},
        {**profile, "linkedin": "https://untrusted.invalid/in/fixture"},
    ]
    bad_resumes = [
        None,
        {},
        *({**resume, key: None} for key in resume),
        {**resume, "organization_id": "1"},
        {**resume, "organization_id": 1.5},
        {**resume, "byte_count": 5242881},
        {**resume, "content_sha256": "x" * 64},
        {**resume, "media_type": "text/plain"},
        {**resume, "source_observed_at": "yesterday"},
    ]
    before = _evidence_count(conn, payload.subject_id)
    with psycopg.connect(RUNTIME_DSN) as runtime:
        runtime.execute(
            "SELECT set_config('app.current_tenant_id',%s,true)",
            (f"candidate_{payload.subject_id}",),
        )
        for approved, pinned in [(p, resume) for p in bad_profiles] + [
            (profile, r) for r in bad_resumes
        ]:
            with pytest.raises(psycopg.Error), runtime.transaction():
                runtime.execute(
                    """INSERT INTO candidate_consent_sources
                  (source_id,subject_id,tenant_id,source_version,global_candidate_id,profile,profile_sha256,
                   resume,resume_sha256,purpose,purpose_version,copy_version,copy_sha256,approved_at)
                  SELECT %s,subject_id,tenant_id,2,global_candidate_id,%s,profile_sha256,%s,%s,
                    purpose,purpose_version,copy_version,copy_sha256,approved_at
                  FROM candidate_consent_sources WHERE source_id=%s""",
                    (uuid4(), Jsonb(approved), Jsonb(pinned), "a" * 64, payload.source.source_id),
                )
        with pytest.raises(psycopg.errors.CheckViolation), runtime.transaction():
            runtime.execute(
                "UPDATE candidate_consent_state SET effective_action=NULL WHERE subject_id=%s",
                (payload.subject_id,),
            )
    assert _evidence_count(conn, payload.subject_id) == before


@pytest.fixture
def consent_readiness(consent_target, monkeypatch):
    conn, role = consent_target
    target_id, environment = conn.execute(
        "SELECT target_id,environment FROM activekg_schema_control.target_identity WHERE singleton=1"
    ).fetchone()
    monkeypatch.setenv("ACTIVEKG_SCHEMA_TARGET_ID", str(target_id))
    monkeypatch.setenv("ACTIVEKG_SCHEMA_ENVIRONMENT", environment)
    monkeypatch.setenv("ACTIVEKG_READYZ_ALLOW_OWNER", "false")
    with ConnectionPool(RUNTIME_DSN, min_size=1, max_size=1, open=True) as pool:
        pool.wait(timeout=10)

        def check(enabled=True):
            return bounded_readiness_check(
                SimpleNamespace(pool=pool),
                unsafe_search_configuration=False,
                jwt_enabled=True,
                jwt_problems=[],
                privacy_problems=[],
                privacy_key_versions={1},
                decision_inbox_enabled=True,
                organization_candidate_intake_enabled=True,
                candidate_consent_intake_enabled=enabled,
                sourced_candidate_ingest_mode="canonical_only",
            )

        yield conn, role, check


def test_consent_readiness_and_disabled_flag(consent_readiness):
    _, _, check = consent_readiness
    assert check() == ReadinessResult(True)
    assert check(False) == ReadinessResult(False, ("candidate_consent_intake_disabled",))


@pytest.mark.parametrize("variant", ["missing", "renamed", "widened"])
def test_consent_readiness_pins_named_constraint_definition(consent_readiness, variant):
    conn, _, check = consent_readiness
    name = "candidate_consent_receipts_outcome_check"
    definition = conn.execute(
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint WHERE conrelid='candidate_consent_receipts'::regclass AND conname=%s",
        (name,),
    ).fetchone()[0]
    assert check() == ReadinessResult(True)
    conn.execute(
        sql.SQL("ALTER TABLE candidate_consent_receipts DROP CONSTRAINT {}").format(
            sql.Identifier(name)
        )
    )
    replacement = "consent_probe_constraint" if variant == "renamed" else name
    try:
        if variant != "missing":
            conn.execute(
                sql.SQL("ALTER TABLE candidate_consent_receipts ADD CONSTRAINT {} ").format(
                    sql.Identifier(replacement)
                )
                + sql.SQL("CHECK (true)" if variant == "widened" else definition)
            )
        assert not check().ready
    finally:
        if variant != "missing":
            conn.execute(
                sql.SQL("ALTER TABLE candidate_consent_receipts DROP CONSTRAINT {}").format(
                    sql.Identifier(replacement)
                )
            )
        conn.execute(
            sql.SQL("ALTER TABLE candidate_consent_receipts ADD CONSTRAINT {} ").format(
                sql.Identifier(name)
            )
            + sql.SQL(definition)
        )
    assert check() == ReadinessResult(True)


@pytest.mark.parametrize("variant", ["policy", "grant", "trigger"])
def test_consent_readiness_refuses_authority_drift(consent_readiness, variant):
    conn, role, check = consent_readiness
    assert check() == ReadinessResult(True)
    if variant == "policy":
        conn.execute(
            "ALTER POLICY tenant_isolation_consent_sources ON candidate_consent_sources USING (true) WITH CHECK (true)"
        )
    elif variant == "grant":
        conn.execute(
            sql.SQL("GRANT UPDATE ON candidate_consent_sources TO {}").format(sql.Identifier(role))
        )
    else:
        conn.execute(
            "ALTER TABLE candidate_consent_sources DISABLE TRIGGER consent_sources_no_mutation"
        )
    try:
        assert not check().ready
    finally:
        if variant == "policy":
            conn.execute(
                "ALTER POLICY tenant_isolation_consent_sources ON candidate_consent_sources USING (tenant_id=current_setting('app.current_tenant_id',true)) WITH CHECK (tenant_id=current_setting('app.current_tenant_id',true))"
            )
        elif variant == "grant":
            conn.execute(
                sql.SQL("REVOKE UPDATE ON candidate_consent_sources FROM {}").format(
                    sql.Identifier(role)
                )
            )
        else:
            conn.execute(
                "ALTER TABLE candidate_consent_sources ENABLE TRIGGER consent_sources_no_mutation"
            )
    assert check() == ReadinessResult(True)
