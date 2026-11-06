"""Local migration integration, deliberately gated to disposable PostgreSQL.

Real 027/040/041 with only the global-candidate prerequisite stubbed. This is
not the full historical fresh/upgrade runner or runtime-role rehearsal.
"""

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import jwt
import psycopg
import pytest
from psycopg import sql

from activekg.candidate_profiles.spark_contracts import old, parse_command, parse_response


@pytest.fixture
def consent_http(spark_db, spark_runtime_dsn, monkeypatch):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from psycopg.rows import dict_row

    from activekg.api import auth, candidate_profiles
    from activekg.candidate_profiles import consent_v2, profile_repository, spark_repository
    from activekg.candidate_profiles.spark_contracts import parse_consent_projection

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = (
        private.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    for key, value in {
        "JWT_ENABLED": True,
        "JWT_ALGORITHM": "RS256",
        "JWT_AUDIENCE": "activekg",
        "JWT_ISSUER": "vantahire",
        "JWT_PUBLIC_KEY": pem,
        "JWT_LEEWAY_SECONDS": 0,
    }.items():
        monkeypatch.setattr(auth, key, value)
    monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_CONSENT_V2_ENABLED", "true")
    monkeypatch.setattr(
        consent_v2, "_connect", lambda: psycopg.connect(spark_runtime_dsn, row_factory=dict_row)
    )
    monkeypatch.setattr(
        spark_repository,
        "_connect",
        lambda: psycopg.connect(spark_runtime_dsn, row_factory=dict_row),
    )
    monkeypatch.setattr(
        profile_repository,
        "_connect",
        lambda: psycopg.connect(spark_runtime_dsn, row_factory=dict_row),
    )
    subject, binding = str(uuid4()), str(uuid4())
    spark_db.execute(
        "SELECT candidate_owned_ensure(%s,%s,%s,1,%s)", (subject, binding, str(uuid4()), "a" * 64)
    )
    command = {
        "schemaVersion": 2,
        "subjectId": subject,
        "bindingId": binding,
        "eventId": str(uuid4()),
        "version": 1,
        "action": "withdraw",
        "copyVersion": None,
        "copyHash": None,
        "purposes": [],
    }
    app = FastAPI()
    app.include_router(candidate_profiles.router)

    def send(
        value=None,
        claims=None,
        extra_headers=None,
        raw=None,
        endpoint="/candidate-consent/v2/events",
        method="POST",
        command_id=None,
    ):
        value = command if value is None else value
        now = int(time.time())
        payload = {
            "iss": "vantahire",
            "aud": "activekg",
            "sub": "vantahire-backend",
            "actor_type": "service",
            "tenant_id": "candidate_" + subject,
            "scopes": "candidate-consent:write",
            "iat": now,
            "exp": now + 60,
        }
        payload.update(claims or {})
        headers = {
            "authorization": "Bearer " + jwt.encode(payload, private, algorithm="RS256"),
            "content-type": "application/json",
        }
        if endpoint == "/candidate-profiles/erase":
            raw = old.canonical(value) if raw is None else raw
        elif endpoint.split("?")[0] == "/candidate-profiles/spark/expiry":
            from activekg.candidate_profiles.spark_contracts import parse_expiry

            parsed = parse_expiry(json.dumps(value).encode())
            headers["x-candidate-expiry-sha256"] = parsed["hash"]
            raw = parsed["canonical"] if raw is None else raw
        elif endpoint.startswith("/candidate-profiles/spark"):
            if method == "POST":
                parsed = parse_command(json.dumps(value).encode())
                headers.update(
                    {
                        "x-candidate-command-sha256": parsed["hash"],
                        "x-candidate-binding-id": binding,
                        "x-candidate-command-id": command_id or str(uuid4()),
                    }
                )
                raw = parsed["canonical"] if raw is None else raw
        else:
            headers["x-command-sha256"] = parse_consent_projection(json.dumps(value).encode())[
                "hash"
            ]
        headers.update(extra_headers or {})
        with TestClient(app) as client:
            return client.request(
                method,
                endpoint,
                content=None if method == "GET" else json.dumps(value) if raw is None else raw,
                headers=headers,
            )

    return command, send


def test_erasure_is_not_an_app_route_and_runtime_cannot_delete(
    spark_db, spark_runtime_dsn, consent_http, monkeypatch
):
    command, send = consent_http
    monkeypatch.setenv("CANDIDATE_PROFILE_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_SPARK_ENABLED", "false")
    body = {
        "schemaVersion": 1,
        "subjectId": command["subjectId"],
        "bindingId": command["bindingId"],
        "requestId": str(uuid4()),
    }
    endpoint = "/candidate-profiles/erase"
    scope = {"scopes": "candidate-profile:erase"}
    assert send(body, endpoint=endpoint).status_code == 404
    assert (
        send(
            body, endpoint=endpoint, claims={"scopes": "candidate-profile:owner-write"}
        ).status_code
        == 404
    )
    wrong = {**body, "bindingId": str(uuid4())}
    assert send(wrong, endpoint=endpoint, claims=scope).status_code == 404
    assert send(body, endpoint=endpoint, claims=scope).status_code == 404
    with psycopg.connect(spark_runtime_dsn, autocommit=True) as runtime:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            runtime.execute(
                "SELECT candidate_profile_erase(%s,%s,%s,'candidate')",
                (body["subjectId"], body["bindingId"], body["requestId"]),
            )
    assert (
        spark_db.execute(
            "SELECT count(*) FROM candidate_profile_erasure_audit WHERE request_id=%s",
            (body["requestId"],),
        ).fetchone()[0]
        == 0
    )
    assert (
        send({**body, "subjectId": str(uuid4())}, endpoint=endpoint, claims=scope).status_code
        == 404
    )
    assert (
        send(body, endpoint=endpoint, claims=scope, raw=old.canonical(body) + " ").status_code
        == 404
    )
    assert send(body, endpoint=endpoint, claims=scope, raw=" " * 1025).status_code == 404
    for _ in range(2):
        assert spark_db.execute(
            "SELECT candidate_profile_erase(%s,%s,%s,'privacy_operator')",
            (body["subjectId"], body["bindingId"], body["requestId"]),
        ).fetchone()[0] == {"requestId": body["requestId"], "status": "erased"}


def test_product_consent_route_replay_and_disabled(spark_db, consent_http, monkeypatch):
    command, send = consent_http
    first, second = send(), send()
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.headers["cache-control"] == "no-store"
    assert (
        spark_db.execute(
            "SELECT count(*) FROM candidate_consent_receipts WHERE event_id=%s",
            (command["eventId"],),
        ).fetchone()[0]
        == 1
    )
    monkeypatch.setenv("CANDIDATE_CONSENT_V2_ENABLED", "false")
    assert send().status_code == 404


@pytest.mark.parametrize("saved", [False, True])
def test_expiry_route_uses_restricted_storage(spark_db, consent_http, monkeypatch, saved):
    from activekg.candidate_profiles.spark_contracts import parse_command
    from tests.test_spark_contracts import command as preference_command

    identity, send = consent_http
    command, cid = preference_command(), str(uuid4())
    parsed = parse_command(json.dumps(command).encode())
    value = {
        "schemaVersion": 3,
        "commandId": cid,
        "bindingId": identity["bindingId"],
        "idempotencyKey": command["idempotencyKey"],
        "commandHash": parsed["hash"],
    }
    claims = {"scopes": "candidate-profile:owner-write"}
    endpoint = "/candidate-profiles/spark/expiry"

    def expire(**kwargs):
        return send(value, endpoint=endpoint, claims=kwargs.pop("claims", claims), **kwargs)

    assert expire().status_code == 404
    monkeypatch.setenv("CANDIDATE_SPARK_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_PROFILE_ENABLED", "true")
    original = (
        send(command, endpoint="/candidate-profiles/spark/commands", claims=claims, command_id=cid)
        if saved
        else None
    )
    if saved:
        assert original.status_code == 200, original.text
    response = expire()
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["status"] == ("applied" if saved else "rejected")
    if saved:
        assert response.json() == original.json()
    else:
        assert response.json()["reason"] == "command_expired"
    assert expire().json() == response.json()
    assert (
        send(
            command, endpoint="/candidate-profiles/spark/commands", claims=claims, command_id=cid
        ).json()
        == response.json()
    )
    assert expire(claims={"scopes": "candidate-profile:owner-read"}).status_code == 403
    assert expire(claims={**claims, "tenant_id": "candidate_" + str(uuid4())}).status_code == 404
    assert expire(extra_headers={"x-candidate-expiry-sha256": "f" * 64}).status_code == 422
    assert expire(raw=parsed["canonical"]).status_code == 422
    assert expire(raw="x" * 4097).status_code == 413


def test_spark_routes_use_restricted_real_storage(spark_db, consent_http, monkeypatch):
    from tests.test_spark_contracts import command as preference_command

    identity, send = consent_http
    command = preference_command()
    cid = str(uuid4())
    endpoint = "/candidate-profiles/spark/commands"
    write_claims = {"scopes": "candidate-profile:owner-write"}

    def write(value=None, **kwargs):
        return send(
            command if value is None else value,
            endpoint=endpoint,
            claims=kwargs.pop("claims", write_claims),
            command_id=cid,
            **kwargs,
        )

    assert write().status_code == 404
    monkeypatch.setenv("CANDIDATE_SPARK_ENABLED", "true")
    assert write().status_code == 404
    monkeypatch.setenv("CANDIDATE_PROFILE_ENABLED", "true")
    first, replay = write(), write()
    assert first.status_code == replay.status_code == 200, first.text
    assert replay.json() == first.json()
    assert first.json()["preferenceDraftRevision"] == 1
    assert first.json()["publicationActive"] is False
    assert first.headers["cache-control"] == "no-store"

    read_path = "/candidate-profiles/spark?binding=" + identity["bindingId"]
    view = send(endpoint=read_path, method="GET", claims={"scopes": "candidate-profile:owner-read"})
    assert view.status_code == 200, view.text
    assert view.json()["preferenceDraft"]["sections"]["employment_types"] == {
        "answered": True,
        "value": ["gig_freelance", "contract"],
    }
    assert view.json()["preferences"] is None  # Partial answers are not promoted.
    assert view.headers["cache-control"] == "no-store"

    assert write(claims={"scopes": "search:read"}).status_code == 403
    assert (
        write(
            claims={
                "scopes": "candidate-profile:owner-write",
                "tenant_id": "candidate_" + str(uuid4()),
            }
        ).status_code
        == 404
    )
    assert write(extra_headers={"x-candidate-binding-id": str(uuid4())}).status_code == 404
    assert write(extra_headers={"x-candidate-command-sha256": "f" * 64}).status_code == 422
    assert write(raw='{"schemaVersion":3,"schemaVersion":3}').status_code == 422
    assert write(raw="x" * (384 * 1024 + 1)).status_code == 413
    assert write(extra_headers={"content-type": "text/plain"}).status_code == 400
    assert (
        send(
            endpoint=read_path + "&binding=" + str(uuid4()),
            method="GET",
            claims={"scopes": "candidate-profile:owner-read"},
        ).status_code
        == 400
    )
    changed = json.loads(json.dumps(command))
    changed["payload"]["value"] = ["permanent"]
    assert write(changed).status_code == 409
    assert (
        send(command, endpoint=endpoint, claims=write_claims, command_id=str(uuid4())).status_code
        == 409
    )
    stale = preference_command(["permanent"])
    result = send(stale, endpoint=endpoint, claims=write_claims, command_id=str(uuid4()))
    assert result.status_code == 200 and result.json()["reason"] == "stale_edit"
    assert result.json()["changed"] is False
    assert (
        spark_db.execute(
            "SELECT count(*) FROM candidate_profile_receipts WHERE subject_id=%s",
            (identity["subjectId"],),
        ).fetchone()[0]
        == 2
    )

    # Sources cannot fall through to private profile writes or trigger providers.
    unsupported = preference_command()
    unsupported.update(
        action="request_source", payload={"kind": "professional_lookup", "attemptId": str(uuid4())}
    )
    result = send(unsupported, endpoint=endpoint, claims=write_claims, command_id=str(uuid4()))
    assert result.status_code == 409 and result.json()["detail"] == "feature_unavailable"

    monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "false")
    assert write().status_code == 404


@pytest.mark.parametrize(
    "symbol,anchor",
    [
        ("save_spark", "require_owner(claims, write=True)"),
        ("read_spark", "require_owner(claims)"),
    ],
)
def test_spark_guard_requires_route_authority(monkeypatch, symbol, anchor):
    from scripts import candidate_privacy_surface_guard as guard

    reference = next(
        r for r in guard.discover() if r.key == "activekg/api/candidate_profiles.py::" + symbol
    )
    source = guard._reference_source(reference)
    row = {"test_id": "privacy-candidate-profile-owner"}
    guard._validate_fence_anchor(reference, row)
    monkeypatch.setattr(guard, "_reference_source", lambda _: source.replace(anchor, '"untrusted"'))
    with pytest.raises(guard.GuardError):
        guard._validate_fence_anchor(reference, row)


@pytest.mark.parametrize(
    "claims",
    [
        {"tenant_id": "candidate_" + str(uuid4())},
        {"scopes": "search:read"},
        {"scopes": "candidate-consent:write search:read"},
        {"sub": "different-service"},
        {"actor_type": "user"},
        {"aud": "signal"},
        {"iss": "untrusted"},
        {"exp": 1},
    ],
)
def test_product_consent_signed_wrong_authority(spark_db, consent_http, claims):
    command, send = consent_http
    assert send(claims=claims).status_code in (401, 403)
    assert (
        spark_db.execute(
            "SELECT count(*) FROM candidate_consent_receipts WHERE event_id=%s",
            (command["eventId"],),
        ).fetchone()[0]
        == 0
    )


def test_product_consent_route_strict_body(consent_http):
    _, send = consent_http
    assert send(extra_headers={"x-command-sha256": "0" * 64}).status_code == 400
    assert send(raw='{"schemaVersion":2,"schemaVersion":2}').status_code == 400
    assert send(raw=" " * 4097).status_code == 413


def test_product_consent_grant_replay_cannot_undo_withdrawal(spark_db, consent_http):
    from activekg.candidate_profiles.spark_contracts import BUNDLED_PURPOSES

    command, send = consent_http
    spark_db.execute(
        "INSERT INTO candidate_consent_copy_catalog VALUES(1,%s,true) ON CONFLICT DO NOTHING",
        ("a" * 64,),
    )
    grant = {
        **command,
        "action": "grant",
        "copyVersion": 1,
        "copyHash": "a" * 64,
        "purposes": BUNDLED_PURPOSES,
    }
    first = send(grant)
    assert first.status_code == 200
    withdrawal = {**command, "eventId": str(uuid4()), "version": 2}
    assert send(withdrawal).status_code == 200
    replay = send(grant)
    assert replay.status_code == 200 and replay.json() == first.json()
    assert spark_db.execute(
        "SELECT effective_version,effective_action FROM candidate_consent_state WHERE subject_id=%s",
        (command["subjectId"],),
    ).fetchone() == (2, "withdraw")


@pytest.fixture(scope="module")
def spark_db():
    url = os.environ.get("SPARK_TEST_ADMIN_URL")
    if not url or os.environ.get("SPARK_DISPOSABLE") != "1":
        pytest.skip("explicit disposable database required")
    parsed = urlparse(url)
    assert parsed.hostname == "127.0.0.1" and parsed.path == "/postgres"
    name = "spark_product_" + uuid4().hex[:10]
    with psycopg.connect(url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {} TEMPLATE template0").format(sql.Identifier(name)))
    with psycopg.connect(url.rsplit("/", 1)[0] + "/" + name, autocommit=True) as db:
        db.execute("CREATE TABLE public.global_candidates(id uuid PRIMARY KEY)")
        root = Path(__file__).resolve().parents[1] / "db/migrations"
        for file in [
            "027_candidate_consent.sql",
            "040_candidate_owned_foundation.sql",
            "041_candidate_profile_drafts.sql",
        ]:
            db.execute((root / file).read_text())
        print("Retained local database:", name)
        yield db


@pytest.fixture
def spark_runtime_dsn(spark_db):
    from psycopg.conninfo import make_conninfo

    from scripts.init_railway_db import (
        _harden_candidate_consent_runtime_privileges,
        _harden_candidate_foundation_runtime_privileges,
        _harden_candidate_profile_runtime_privileges,
    )

    role = "spark_consent_" + uuid4().hex[:10]
    spark_db.execute(
        sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS").format(sql.Identifier(role))
    )
    with spark_db.cursor() as cur:
        _harden_candidate_foundation_runtime_privileges(cur, role)
        _harden_candidate_consent_runtime_privileges(cur, role)
        _harden_candidate_profile_runtime_privileges(cur, role)
    dsn = make_conninfo(spark_db.info.dsn, user=role)
    with psycopg.connect(dsn, autocommit=True) as runtime:
        assert runtime.execute(
            "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
        ).fetchone() == (False, False)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            runtime.execute("SELECT * FROM candidate_profile_heads")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            runtime.execute("SELECT * FROM candidate_consent_copy_catalog")
    yield dsn
    spark_db.execute(
        sql.SQL("REVOKE EXECUTE ON ALL FUNCTIONS IN SCHEMA public FROM {}").format(
            sql.Identifier(role)
        )
    )
    spark_db.execute(sql.SQL("ALTER ROLE {} NOLOGIN").format(sql.Identifier(role)))


def test_integrated_migration_save_replay_and_read(spark_db):
    db = spark_db
    subject, binding, cmd = str(uuid4()), str(uuid4()), str(uuid4())
    db.execute(
        "SELECT candidate_owned_ensure(%s,%s,%s,1,%s)", (subject, binding, str(uuid4()), "a" * 64)
    )
    wire = old.canonical(
        {
            "schemaVersion": 3,
            "idempotencyKey": str(uuid4()),
            "expectedDraftVersion": 0,
            "expectedProfileVersion": 0,
            "expectedPreferenceVersion": 0,
            "expectedPreferenceDraftRevision": 0,
            "action": "save_preference_section",
            "payload": {
                "section": "employment_types",
                "answered": True,
                "value": ["gig_freelance"],
            },
        }
    )
    parsed = parse_command(wire.encode())
    args = (subject, binding, cmd, parsed["hash"], parsed["canonical"])
    first = db.execute("SELECT candidate_preference_draft_apply(%s,%s,%s,%s,%s)", args).fetchone()[
        0
    ]
    assert parse_response(old.canonical(first).encode(), "receipt")["value"] == first
    assert first["preferenceDraftRevision"] == 1 and first["publicationActive"] is False
    assert (
        db.execute("SELECT candidate_preference_draft_apply(%s,%s,%s,%s,%s)", args).fetchone()[0]
        == first
    )


def test_new_security_definer_routines_not_public(spark_db):
    rows = spark_db.execute("""SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
      CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
      WHERE n.nspname='public' AND p.proname LIKE 'candidate_%' AND a.grantee=0 AND a.privilege_type='EXECUTE'""").fetchall()
    assert rows == []


def test_profile_catalog_and_restricted_permissions(spark_db):
    from activekg.candidate_profiles.profile_repository import (
        PROFILE_CATALOG_MD5,
        profile_catalog_evidence,
        profile_catalog_ready,
    )
    from scripts.init_railway_db import _harden_candidate_profile_runtime_privileges

    db = spark_db
    role = "spark_runtime_" + uuid4().hex[:10]
    # Role and all grants are rolled back; the retained fixture stays unchanged.
    with db.transaction():
        db.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role)))
        with db.cursor() as cur:
            _harden_candidate_profile_runtime_privileges(cur, role)
            digest, permissions = profile_catalog_evidence(cur, role)
            assert digest == PROFILE_CATALOG_MD5, digest
            assert permissions is True
            assert profile_catalog_ready(cur, role)
            cur.execute(
                sql.SQL("GRANT SELECT ON candidate_exclusion_proposals TO {}").format(
                    sql.Identifier(role)
                )
            )
            assert not profile_catalog_ready(cur, role)
            _harden_candidate_profile_runtime_privileges(cur, role)
            assert profile_catalog_ready(cur, role)
            cur.execute(
                sql.SQL(
                    "GRANT EXECUTE ON FUNCTION candidate_profile_canonical(jsonb) TO {}"
                ).format(sql.Identifier(role))
            )
            assert not profile_catalog_ready(cur, role)
            _harden_candidate_profile_runtime_privileges(cur, role)
            cur.execute(
                "ALTER TABLE candidate_consent_state DROP CONSTRAINT consent_effective_authority_shape"
            )
            assert not profile_catalog_ready(cur, role)
        raise psycopg.Rollback()


def test_adapter_validates_receipt_and_rolls_back_late_privacy(spark_db, monkeypatch):
    from fastapi import HTTPException
    from psycopg.rows import dict_row

    from activekg.candidate_profiles import spark_repository as repo

    db = spark_db
    subject, binding = str(uuid4()), str(uuid4())
    db.execute(
        "SELECT candidate_owned_ensure(%s,%s,%s,1,%s)", (subject, binding, str(uuid4()), "a" * 64)
    )
    monkeypatch.setattr(
        repo, "_connect", lambda: psycopg.connect(db.info.dsn, row_factory=dict_row)
    )
    checks = []

    def privacy(cur, subject):
        checks.append(subject)
        if len(checks) == 2:
            raise HTTPException(451, "candidate_privacy_restricted")

    monkeypatch.setattr(repo, "check_owner_privacy", privacy)
    wire = old.canonical(
        {
            "schemaVersion": 3,
            "idempotencyKey": str(uuid4()),
            "expectedDraftVersion": 0,
            "expectedProfileVersion": 0,
            "expectedPreferenceVersion": 0,
            "expectedPreferenceDraftRevision": 0,
            "action": "save_preference_section",
            "payload": {"section": "employment_types", "answered": True, "value": ["contract"]},
        }
    )
    digest = parse_command(wire.encode())["hash"]
    cmd = str(uuid4())
    with pytest.raises(HTTPException) as error:
        repo.execute_spark(subject, binding, raw=wire.encode(), command_id=cmd, digest=digest)
    assert error.value.status_code == 451 and len(checks) == 2
    assert (
        db.execute(
            "SELECT count(*) FROM candidate_profile_receipts WHERE command_id=%s", (cmd,)
        ).fetchone()[0]
        == 0
    )
    monkeypatch.setattr(repo, "check_owner_privacy", lambda cur, subject: None)
    result = repo.execute_spark(subject, binding, raw=wire.encode(), command_id=cmd, digest=digest)
    assert result["status"] == "applied" and result["preferenceDraftRevision"] == 1
    assert (
        repo.execute_spark(subject, binding, raw=wire.encode(), command_id=cmd, digest=digest)
        == result
    )
