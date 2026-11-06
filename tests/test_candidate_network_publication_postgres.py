"""Native publication proof on an explicitly supplied full-schema disposable DB.

Every test rolls back its own changes. No provider/model calls or real identities.
"""

import hashlib
import os
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from activekg.candidate_profiles.profile_contracts import canonical
from activekg.candidate_profiles.spark_contracts import BUNDLED_PURPOSES
from tests.test_candidate_profile_drafts_postgres import FACTS

DSN = os.getenv("SPARK_NETWORK_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="explicit full-schema P7 test database required")


@pytest.fixture(scope="module")
def network_database():
    parsed = urlparse(DSN)
    assert parsed.hostname == "127.0.0.1" and parsed.path.endswith("_test")
    assert os.getenv("SPARK_DISPOSABLE") == "1"
    name = "spark_network_" + uuid4().hex[:10] + "_test"
    with psycopg.connect(make_conninfo(DSN, dbname="postgres"), autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                sql.Identifier(name), sql.Identifier(parsed.path[1:])
            )
        )
    target = make_conninfo(DSN, dbname=name)
    with psycopg.connect(target) as conn:
        already_migrated = (
            conn.execute("SELECT to_regclass('candidate_network_subjects')").fetchone()[0]
            is not None
        )
        owner = conn.execute(
            "SELECT pg_get_userbyid(relowner) FROM pg_class WHERE oid='candidate_owned_subjects'::regclass"
        ).fetchone()[0]
        conn.execute(
            sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                sql.Identifier(name), sql.Identifier(owner)
            )
        )
        conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(owner)))
        migration = (
            Path(__file__).resolve().parents[1]
            / "db/migrations/042_candidate_network_publication.sql"
        ).read_text()
        if not already_migrated:
            conn.execute(migration)
        if (
            conn.execute(
                "SELECT to_regprocedure('candidate_network_owner_status_detail(uuid,uuid)')"
            ).fetchone()[0]
            is None
        ):
            conn.execute(
                (
                    Path(__file__).resolve().parents[1]
                    / "db/migrations/043_candidate_network_owner_status.sql"
                ).read_text()
            )
        if (
            conn.execute(
                "SELECT to_regprocedure('candidate_network_result_keys(uuid[])')"
            ).fetchone()[0]
            is None
        ):
            conn.execute(
                (
                    Path(__file__).resolve().parents[1]
                    / "db/migrations/044_candidate_network_result_dedup.sql"
                ).read_text()
            )
        if (
            conn.execute(
                "SELECT to_regprocedure('candidate_network_recovery_open(text)')"
            ).fetchone()[0]
            is None
        ):
            conn.execute(
                (
                    Path(__file__).resolve().parents[1]
                    / "db/migrations/045_candidate_network_recovery.sql"
                ).read_text()
            )
        from activekg.candidate_profiles.publication import (
            NETWORK_CATALOG_MD5,
            network_catalog_evidence,
        )

        assert network_catalog_evidence(conn.cursor())[0] == NETWORK_CATALOG_MD5
    print("Retained P7 full-schema fixture", name)
    return target


@pytest.fixture
def db(network_database):
    with psycopg.connect(network_database) as conn:
        assert conn.execute(
            "SELECT to_regprocedure('candidate_privacy_decision_for(uuid,text,uuid)') IS NOT NULL"
        ).fetchone()[0]
        try:
            yield conn
        finally:
            conn.rollback()


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def consent(db, who, action="grant", revision=1):
    copy_hash = "a" * 64
    db.execute(
        "INSERT INTO candidate_consent_copy_catalog VALUES(999,%s,true) ON CONFLICT DO NOTHING",
        (copy_hash,),
    )
    value = {
        "schemaVersion": 2,
        "subjectId": who[0],
        "bindingId": who[1],
        "eventId": str(uuid4()),
        "version": revision,
        "action": action,
        "copyVersion": 999 if action == "grant" else None,
        "copyHash": copy_hash if action == "grant" else None,
        "purposes": BUNDLED_PURPOSES if action == "grant" else [],
    }
    return db.execute(
        "SELECT candidate_consent_v2_project(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            *who,
            value["eventId"],
            revision,
            action,
            value["copyVersion"],
            value["copyHash"],
            value["purposes"],
            digest(value),
        ),
    ).fetchone()[0]


def confirmed(db, overrides=None):
    who = (str(uuid4()), str(uuid4()))
    db.execute("SELECT candidate_owned_ensure(%s,%s,%s,1,%s)", (*who, str(uuid4()), "a" * 64))
    consent(db, who)
    facts = {
        **FACTS,
        "headline": "Engineer contact secret@example.test +919876543210",
        "photoUrl": None,
        **(overrides or {}),
    }
    command = {
        "schemaVersion": 2,
        "idempotencyKey": str(uuid4()),
        "action": "save_draft",
        "expectedDraftVersion": 0,
        "expectedProfileVersion": 0,
        "expectedPreferenceVersion": 0,
        "payload": {"profile": facts},
    }
    result = db.execute(
        "SELECT candidate_profile_apply(%s,%s,%s,%s,%s)",
        (*who, str(uuid4()), digest(command), Jsonb(canonical(command))),
    ).fetchone()[0]
    assert result["status"] == "applied"
    sections = {
        "roles": {"desiredRoles": ["Engineer"], "preferredSeniority": []},
        "location": {"mode": "anywhere"},
        "arrangement": {"workArrangements": ["remote"], "remoteRegions": {"mode": "worldwide"}},
        "relocation": {"mode": "no", "destinations": []},
        "employment_types": ["permanent"],
        "availability": {"mode": "unknown"},
        "compensation": {"mode": "open_to_discussion"},
    }
    revision = 0
    for section, value in sections.items():
        request = {
            "schemaVersion": 3,
            "idempotencyKey": str(uuid4()),
            "action": "save_preference_section",
            "expectedDraftVersion": 1,
            "expectedProfileVersion": 0,
            "expectedPreferenceVersion": 0,
            "expectedPreferenceDraftRevision": revision,
            "payload": {"section": section, "answered": True, "value": value},
        }
        result = db.execute(
            "SELECT candidate_preference_draft_apply(%s,%s,%s,%s,%s)",
            (*who, str(uuid4()), digest(request), canonical(request)),
        ).fetchone()[0]
        assert result["status"] == "applied"
        revision += 1
    request.update(
        idempotencyKey=str(uuid4()),
        action="promote_preferences",
        expectedPreferenceDraftRevision=revision,
        payload={"preferenceDraftHash": result["preferenceDraftHash"]},
    )
    result = db.execute(
        "SELECT candidate_preference_draft_apply(%s,%s,%s,%s,%s)",
        (*who, str(uuid4()), digest(request), canonical(request)),
    ).fetchone()[0]
    assert result["status"] == "applied"
    request.update(
        idempotencyKey=str(uuid4()),
        action="confirm",
        expectedPreferenceVersion=result["preferenceVersion"],
        payload={
            "draftHash": digest(facts),
            "preferenceDraftHash": result["preferenceDraftHash"],
            "consentRevision": 1,
        },
    )
    result = db.execute(
        "SELECT candidate_profile_confirm_v3(%s,%s,%s,%s,%s)",
        (*who, str(uuid4()), digest(request), canonical(request)),
    ).fetchone()[0]
    assert result["status"] == "applied"
    return who


VECTOR = "[" + ",".join(["1"] + ["0"] * 383) + "]"


def claim(db):
    return db.execute("SELECT candidate_network_claim()").fetchone()[0]


def finish(db, p):
    return db.execute(
        "SELECT candidate_network_finish(%s,%s,%s,%s::vector)",
        (p["publicationId"], p["leaseToken"], p["contentHash"], VECTOR),
    ).fetchone()[0]


def search(db):
    return db.execute(
        "SELECT candidate_network_search(%s::vector,10,%s)",
        (VECTOR, Jsonb({"cities": [], "skills": []})),
    ).fetchone()[0]


def test_result_comparison_key_is_confirmed_service_only_and_withdrawal_fenced(db):
    who = confirmed(db, {"linkedinUrl": "https://linkedin.com/in/John-Doe"})
    p = claim(db)
    assert finish(db, p)

    def keys():
        return db.execute(
            "SELECT candidate_network_result_keys(%s::uuid[])", ([p["publicationId"]],)
        ).fetchone()[0]

    result = keys()
    assert len(result) == 1
    assert result[0]["linkedinUrl"] == "https://linkedin.com/in/John-Doe"
    assert result[0]["contentHash"] == p["contentHash"]
    assert "linkedinUrl" not in search(db)[0]["facts"]
    consent(db, who, "withdraw", 2)
    assert keys() == []


def test_result_comparison_does_not_invent_missing_linkedin(db):
    confirmed(db, {"linkedinUrl": None})
    p = claim(db)
    assert finish(db, p)
    assert (
        db.execute(
            "SELECT candidate_network_result_keys(%s::uuid[])", ([p["publicationId"]],)
        ).fetchone()[0]
        == []
    )


@pytest.mark.skipif(
    os.getenv("P7_LOCAL_TIMING") != "1", reason="explicit local synthetic timing run"
)
def test_owner_reader_and_public_search_local_timing(db, monkeypatch):
    """Measurement only, not a production latency or concurrency acceptance gate."""
    import json
    import time

    from scripts.init_railway_db import _harden_candidate_network_runtime_privileges

    monkeypatch.setenv("ACTIVEKG_NETWORK_WORKER_PASSWORD", "synthetic-worker-only")
    owner = db.execute(
        "SELECT pg_get_userbyid(relowner) FROM pg_class WHERE oid='candidate_owned_subjects'::regclass"
    ).fetchone()[0]
    runtime = owner.replace("s2_owner_", "s2_runtime_")
    assert runtime != owner
    _harden_candidate_network_runtime_privileges(db.cursor(), runtime)
    subjects = [confirmed(db) for _ in range(100)]
    for _ in subjects:
        assert finish(db, claim(db))
    db.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(runtime)))
    samples = {"resumeReadMs": [], "search100Ms": []}
    for _ in range(30):
        start = time.perf_counter()
        result = db.execute(
            "SELECT candidate_profile_v3_storage_read(%s,%s)", subjects[0]
        ).fetchone()[0]
        assert result is not None
        samples["resumeReadMs"].append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        result = db.execute(
            "SELECT candidate_network_search(%s::vector,100,%s)",
            (VECTOR, Jsonb({"cities": [], "skills": []})),
        ).fetchone()[0]
        assert len(result) == 100
        samples["search100Ms"].append((time.perf_counter() - start) * 1000)
    db.execute("RESET ROLE")
    print(
        "LOCAL_SYNTHETIC_TIMING",
        json.dumps(
            {
                name: {
                    "n": len(values),
                    "p50": sorted(values)[15],
                    "p95": sorted(values)[28],
                    "max": max(values),
                }
                for name, values in samples.items()
            }
        ),
    )


@pytest.mark.parametrize(
    "city,expected", [("Bengaluru Urban", True), ("London", False), ("", True)]
)
def test_native_city_filter_matches_provider_alias_and_unknown_policy(db, city, expected):
    confirmed(
        db, {"currentLocation": {"city": city, "countryCode": "IN", "displayCountry": "India"}}
    )
    p = claim(db)
    assert finish(db, p)
    rows = db.execute(
        "SELECT candidate_network_search(%s::vector,10,%s)",
        (VECTOR, Jsonb({"cities": ["bengaluru"], "skills": []})),
    ).fetchone()[0]
    assert bool(rows) is expected


def test_owner_recovery_is_bounded_idempotent_and_consent_fenced(db):
    who = confirmed(db)
    for _ in range(3):
        p = claim(db)
        assert p
        db.execute(
            "UPDATE candidate_network_publications SET lease_until=clock_timestamp()-interval '1 minute' WHERE publication_id=%s",
            (p["publicationId"],),
        )
    assert claim(db) is None
    request = str(uuid4())
    args = (p["publicationId"], request, p["contentHash"])
    assert db.execute("SELECT candidate_network_recover(%s,%s,%s)", args).fetchone()[0]
    again = claim(db)
    assert again and again["leaseToken"] != p["leaseToken"]
    assert not finish(db, p)
    assert db.execute("SELECT candidate_network_recover(%s,%s,%s)", args).fetchone()[0]
    assert (
        db.execute(
            "SELECT attempts FROM candidate_network_publications WHERE publication_id=%s",
            (p["publicationId"],),
        ).fetchone()[0]
        == 1
    )
    assert (
        db.execute(
            "SELECT count(*) FROM candidate_network_recovery_audit WHERE request_id=%s", (request,)
        ).fetchone()[0]
        == 1
    )
    with pytest.raises(psycopg.errors.UniqueViolation):
        with db.transaction():
            db.execute("SELECT candidate_network_recover(%s,%s,%s)", (*args[:2], "0" * 64))
    consent(db, who, "withdraw", 2)
    assert not finish(db, again)
    assert not db.execute(
        "SELECT candidate_network_recover(%s,%s,%s)",
        (p["publicationId"], str(uuid4()), p["contentHash"]),
    ).fetchone()[0]
    # Receipt replay acknowledges history, but never resurrects the withdrawn row.
    assert db.execute("SELECT candidate_network_recover(%s,%s,%s)", args).fetchone()[0]
    assert search(db) == []
    db.execute("SELECT candidate_profile_erase(%s,%s,%s,'privacy_operator')", (*who, str(uuid4())))
    assert (
        db.execute(
            "SELECT content_hash FROM candidate_network_recovery_audit WHERE request_id=%s",
            (request,),
        ).fetchone()[0]
        is None
    )
    assert not db.execute(
        "SELECT has_function_privilege('activekg_app','candidate_network_recover(uuid,uuid,text)','EXECUTE')"
    ).fetchone()[0]


def test_confirm_stage_embed_active_and_native_privacy_identity(db):
    who = confirmed(db)
    assert search(db) == []
    p = claim(db)
    assert p["subjectId"] == who[0]
    assert "secret@example.test" not in canonical(p["facts"])
    assert "9876543210" not in canonical(p["facts"])
    assert "photoUrl" not in p["facts"] and "preferences" not in p["facts"]
    assert finish(db, p)
    hits = search(db)
    assert len(hits) == 1 and hits[0]["identity"] == "network:" + who[0]
    assert hits[0]["contentHash"] == digest(hits[0]["facts"])
    resolved = db.execute(
        "SELECT global_candidate_id FROM candidate_privacy_resolve_subject('signal_candidate_id',%s)",
        (hashlib.sha256(("network:" + who[0]).encode()).digest(),),
    ).fetchone()
    assert str(resolved[0]) == hits[0]["globalCandidateId"]
    assert db.execute("SELECT candidate_network_owner_status(%s,%s)", who).fetchone()[0]
    assert not finish(db, p)


@pytest.mark.parametrize("activate_first", [False, True])
def test_withdrawal_defeats_late_embedding_and_new_search(db, activate_first):
    who = confirmed(db)
    p = claim(db)
    if activate_first:
        assert finish(db, p)
    consent(db, who, "withdraw", 2)
    assert not finish(db, p)
    assert search(db) == []
    assert (
        db.execute(
            "SELECT candidate_network_validate(ARRAY[%s]::uuid[])", (p["publicationId"],)
        ).fetchone()[0]
        == []
    )
    # Withdrawal is not erasure, including the canonical stored projection.
    stored = db.execute(
        "SELECT state,facts FROM candidate_network_publications WHERE subject_id=%s", (who[0],)
    ).fetchone()
    assert stored[0] == "revoked" and stored[1] == p["facts"]


def test_owner_status_distinguishes_confirmation_from_publication(db):
    who = confirmed(db)

    def status():
        return db.execute("SELECT candidate_network_owner_status_detail(%s,%s)", who).fetchone()[0]

    assert status() == {
        "schemaVersion": 1,
        "publicationActive": False,
        "blockedReason": "publication_pending",
    }
    p = claim(db)
    assert status()["blockedReason"] == "publication_pending"
    assert finish(db, p)
    assert status() == {"schemaVersion": 1, "publicationActive": True, "blockedReason": None}
    db.execute(
        "UPDATE candidate_profile_heads SET preference_version=preference_version+1 WHERE subject_id=%s",
        (who[0],),
    )
    assert status()["blockedReason"] == "confirmation_required"
    consent(db, who, "withdraw", 2)
    assert status()["blockedReason"] == "consent_required"
    with pytest.raises(psycopg.errors.NoDataFound), db.transaction():
        db.execute("SELECT candidate_network_owner_status_detail(%s,%s)", (who[0], uuid4()))


def test_owner_status_reports_held_not_confirmation_required(db):
    who = confirmed(db)
    db.execute(
        "UPDATE candidate_network_publications SET state='held' WHERE subject_id=%s", (who[0],)
    )
    assert (
        db.execute("SELECT candidate_network_owner_status_detail(%s,%s)", who).fetchone()[0][
            "blockedReason"
        ]
        == "publication_held"
    )


def test_tampered_or_expired_lease_cannot_activate(db):
    confirmed(db)
    p = claim(db)
    assert not finish(db, {**p, "contentHash": "b" * 64})
    assert not finish(db, {**p, "leaseToken": str(uuid4())})
    db.execute(
        "UPDATE candidate_network_publications SET lease_until=clock_timestamp()-interval '1 minute' WHERE publication_id=%s",
        (p["publicationId"],),
    )
    assert not finish(db, p)


def test_catalogue_and_restricted_grants(db):
    from activekg.candidate_profiles.publication import (
        NETWORK_CATALOG_MD5,
        network_catalog_evidence,
    )
    from scripts.init_railway_db import _harden_candidate_network_runtime_privileges

    _harden_candidate_network_runtime_privileges(db.cursor(), "activekg_app")
    fingerprint, permissions = network_catalog_evidence(db.cursor(), "activekg_app")
    assert permissions
    assert fingerprint == NETWORK_CATALOG_MD5
    assert not db.execute(
        "SELECT has_table_privilege('activekg_app','candidate_network_publications','SELECT')"
    ).fetchone()[0]


def test_worker_provisioning_with_non_superuser_owner_and_unsafe_role_refusal(db, monkeypatch):
    from activekg.candidate_profiles import publication
    from scripts.init_railway_db import _provision_candidate_network_worker

    caller = "p7_provisioner_" + uuid4().hex[:10]
    worker = "p7_worker_" + uuid4().hex[:10]
    owner = db.execute(
        "SELECT pg_get_userbyid(relowner) FROM pg_class WHERE oid='candidate_owned_subjects'::regclass"
    ).fetchone()[0]
    db.execute(
        sql.SQL("CREATE ROLE {} NOSUPERUSER NOBYPASSRLS CREATEROLE INHERIT").format(
            sql.Identifier(caller)
        )
    )
    db.execute(sql.SQL("GRANT {} TO {}").format(sql.Identifier(owner), sql.Identifier(caller)))
    monkeypatch.setattr(publication, "NETWORK_WORKER_ROLE", worker)
    monkeypatch.setenv("ACTIVEKG_NETWORK_WORKER_PASSWORD", "synthetic-worker-only")
    db.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(caller)))
    assert db.execute(
        "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
    ).fetchone() == (False, False)
    _provision_candidate_network_worker(db.cursor())
    _provision_candidate_network_worker(db.cursor())
    assert db.execute(
        "SELECT rolsuper,rolbypassrls,rolcreatedb,rolcreaterole,rolreplication FROM pg_roles WHERE rolname=%s",
        (worker,),
    ).fetchone() == (False, False, False, False, False)
    assert db.execute(
        "SELECT has_function_privilege(%s,'candidate_network_claim()','EXECUTE'),has_table_privilege(%s,'candidate_network_publications','SELECT')",
        (worker, worker),
    ).fetchone() == (True, False)
    db.execute("RESET ROLE")
    db.execute(sql.SQL("ALTER ROLE {} CREATEDB").format(sql.Identifier(worker)))
    db.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(caller)))
    with pytest.raises(RuntimeError, match="candidate_network_worker_role_unsafe"):
        _provision_candidate_network_worker(db.cursor())
    db.execute("RESET ROLE")


def test_separate_worker_can_only_claim_and_finish(db, monkeypatch):
    from activekg.candidate_profiles.publication import network_catalog_evidence
    from scripts.init_railway_db import _harden_candidate_network_runtime_privileges

    monkeypatch.setenv("ACTIVEKG_NETWORK_WORKER_PASSWORD", "synthetic-worker-only")
    _harden_candidate_network_runtime_privileges(db.cursor(), "activekg_app")
    assert network_catalog_evidence(db.cursor(), "activekg_app")[1]
    confirmed(db)
    db.execute("SET LOCAL ROLE activekg_network_worker")
    assert db.execute(
        "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
    ).fetchone() == (False, False)
    with db.transaction():
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.transaction():
                db.execute("SELECT * FROM candidate_profile_heads")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.transaction():
                db.execute("SELECT * FROM candidate_network_publications")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with db.transaction():
                db.execute("SELECT candidate_network_validate('{}'::uuid[])")
    p = claim(db)
    assert p and finish(db, p)
    db.execute("RESET ROLE")
    assert not db.execute(
        "SELECT has_function_privilege('activekg_app','candidate_network_finish(uuid,uuid,text,vector)','EXECUTE')"
    ).fetchone()[0]


@pytest.mark.skipif(
    os.getenv("P7_SIGNED_PROOF") != "1", reason="explicit three-repository proof required"
)
def test_signed_memory_discover_ranking_flow_and_withdrawal(network_database, monkeypatch):
    """Real JWTs, HTTP, SQL publication/ranking and Flow's reader/redactor.

    Only the embedding model is synthetic. Not a full application/browser boot.
    """
    import json
    import socket
    import subprocess
    import threading
    import time
    from types import SimpleNamespace

    import uvicorn
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import FastAPI, Header, HTTPException

    from activekg.api import auth, global_memory
    from activekg.candidate_profiles.publication import sweep_publications
    from scripts.init_railway_db import _harden_candidate_network_runtime_privileges

    flow = Path(os.environ["P7_FLOW_ROOT"]).resolve()
    discover = Path(os.environ["P7_DISCOVER_ROOT"]).resolve()
    monkeypatch.setenv("ACTIVEKG_NETWORK_WORKER_PASSWORD", "synthetic-worker-only")
    with psycopg.connect(network_database) as conn:
        owner = conn.execute(
            "SELECT pg_get_userbyid(relowner) FROM pg_class WHERE oid='candidate_owned_subjects'::regclass"
        ).fetchone()[0]
        runtime = owner.replace("s2_owner_", "s2_runtime_")
        assert runtime != owner and runtime.endswith("_test")
        _harden_candidate_network_runtime_privileges(conn.cursor(), runtime)
        who = confirmed(conn, {"linkedinUrl": "https://linkedin.com/in/John-Doe"})
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    for name, value in {
        "JWT_ENABLED": True,
        "JWT_ALGORITHM": "RS256",
        "JWT_AUDIENCE": "activekg",
        "JWT_ISSUER": "vantahire",
        "JWT_PUBLIC_KEY": public,
        "SIGNAL_JWT_ISSUER": "signal",
        "SIGNAL_JWT_PUBLIC_KEY": public,
        "JWT_LEEWAY_SECONDS": 5,
    }.items():
        monkeypatch.setattr(auth, name, value)
    for flag in (
        "CANDIDATE_FOUNDATION_ENABLED",
        "CANDIDATE_PROFILE_ENABLED",
        "CANDIDATE_SPARK_ENABLED",
        "CANDIDATE_CONSENT_V2_ENABLED",
        "CANDIDATE_NETWORK_PUBLICATION_ENABLED",
    ):
        monkeypatch.setenv(flag, "true")
    monkeypatch.setenv(
        "CANDIDATE_NETWORK_WORKER_DSN",
        make_conninfo(
            network_database, user="activekg_network_worker", password="synthetic-worker-only"
        ),
    )
    assert (
        sweep_publications(
            "must-not-use-app-dsn", SimpleNamespace(encode=lambda texts: [[1] + [0] * 383])
        )
        == 1
    )
    with psycopg.connect(network_database) as conn:
        hit = search(conn)[0]
    monkeypatch.setattr(global_memory, "GLOBAL_MEMORY_ENABLED", True)
    monkeypatch.setattr(global_memory, "PUBLIC_PROFILE_SEARCH_ENABLED", True)
    monkeypatch.setattr(global_memory, "_DSN", make_conninfo(network_database, user=runtime))
    monkeypatch.setattr(
        global_memory,
        "_embedder",
        SimpleNamespace(encode=lambda texts: [SimpleNamespace(tolist=lambda: [1] + [0] * 383)]),
    )
    app = FastAPI()
    app.include_router(global_memory.router)
    control_token = uuid4().hex

    @app.post("/_proof/withdraw")
    def withdraw_fixture(x_proof_token: str = Header()):
        if x_proof_token != control_token:
            raise HTTPException(403)
        with psycopg.connect(network_database) as conn:
            consent(conn, who, "withdraw", 2)
        return {"ok": True}

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    origin = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", access_log=False, lifespan="off")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    child = {
        **os.environ,
        "NODE_ENV": "test",
        "P7_SIGNED_PROOF": "1",
        "P7_MEMORY_ORIGIN": origin,
        "ACTIVEKG_BASE_URL": origin,
        "CANDIDATE_FOUNDATION_DISPOSABLE_TEST": "1",
        "CANDIDATE_NETWORK_SOURCING_ENABLED": "true",
        "VANTAHIRE_JWT_PRIVATE_KEY": private,
        "SIGNAL_JWT_PRIVATE_KEY": private,
        "P7_HIT": json.dumps(hit),
        "P7_SUBJECT": who[0],
        "P7_BINDING": who[1],
        "P7_CONTENT_HASH": hit["contentHash"],
        "P7_CONTROL_TOKEN": control_token,
    }

    def run(root, args, extra=None):
        result = subprocess.run(
            [
                str(root / "node_modules/.bin/vitest"),
                "run",
                *args,
                "--maxWorkers=1",
                "--no-file-parallelism",
            ],
            cwd=root,
            env={**child, **(extra or {})},
            text=True,
            capture_output=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        print(result.stdout)

    try:
        deadline = time.monotonic() + 5
        while not server.started:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        run(
            flow,
            [
                "--config",
                "vitest.server.config.ts",
                "server/candidate-product/network-sourcing.test.ts",
            ],
            {"P7_EXPECT_ACTIVE": "1"},
        )
        run(discover, ["src/lib/sourcing/__tests__/candidate-network-postgres.integration.test.ts"])
        with psycopg.connect(network_database) as conn:
            assert search(conn) == []
        run(
            flow,
            [
                "--config",
                "vitest.server.config.ts",
                "server/candidate-product/network-sourcing.test.ts",
            ],
            {"P7_EXPECT_ACTIVE": "0"},
        )
    finally:
        server.should_exit = True
        thread.join(5)
        listener.close()
        assert not thread.is_alive()


# ---- B5 restore recovery gate (Claude, 5 Nov 2026) ---------------------------------
EPOCH = "b5-test-epoch-0000000001"


def test_recovery_gate_is_closed_until_this_epoch_is_attested(db):
    def is_open(epoch):
        return db.execute("SELECT candidate_network_recovery_open(%s)", (epoch,)).fetchone()[0]

    assert is_open(EPOCH) is False
    assert is_open(None) is False and is_open("short") is False
    run = str(uuid4())
    fp = db.execute("SELECT candidate_network_recovery_fingerprint()").fetchone()[0]
    db.execute(
        "SELECT candidate_network_recovery_attest(%s,%s,'memory',%s,%s)",
        (EPOCH, run, Jsonb({"subjects": 0}), fp),
    )
    assert is_open(EPOCH) is True
    # A restored database carries only older epochs: a new epoch stays closed.
    assert is_open("b5-test-epoch-0000000002") is False
    # Same run replays; a different run cannot overwrite the attestation.
    db.execute("SELECT candidate_network_recovery_attest(%s,%s,'memory','{}',%s)", (EPOCH, run, fp))
    with pytest.raises(psycopg.errors.UniqueViolation), db.transaction():
        db.execute(
            "SELECT candidate_network_recovery_attest(%s,%s,'memory','{}',%s)",
            (EPOCH, str(uuid4()), fp),
        )


def reconfirm(db, who, headline):
    """The candidate reviews the restored profile, edits it and confirms again."""
    d, p, f = db.execute(
        "SELECT draft_version,profile_version,preference_version FROM candidate_profile_heads WHERE subject_id=%s",
        (who[0],),
    ).fetchone()
    revision, pref_hash = db.execute(
        "SELECT revision,content_hash FROM candidate_preference_drafts WHERE subject_id=%s",
        (who[0],),
    ).fetchone()
    facts = {**FACTS, "headline": headline, "photoUrl": None}
    command = {
        "schemaVersion": 2,
        "idempotencyKey": str(uuid4()),
        "action": "save_draft",
        "expectedDraftVersion": d,
        "expectedProfileVersion": p,
        "expectedPreferenceVersion": f,
        "payload": {"profile": facts},
    }
    result = db.execute(
        "SELECT candidate_profile_apply(%s,%s,%s,%s,%s)",
        (*who, str(uuid4()), digest(command), Jsonb(canonical(command))),
    ).fetchone()[0]
    assert result["status"] == "applied", result
    request = {
        "schemaVersion": 3,
        "idempotencyKey": str(uuid4()),
        "action": "confirm",
        "expectedDraftVersion": result["draftVersion"],
        "expectedProfileVersion": p,
        "expectedPreferenceVersion": f,
        "expectedPreferenceDraftRevision": revision,
        "payload": {
            "draftHash": digest(facts),
            "preferenceDraftHash": pref_hash,
            "consentRevision": 1,
        },
    }
    result = db.execute(
        "SELECT candidate_profile_confirm_v3(%s,%s,%s,%s,%s)",
        (*who, str(uuid4()), digest(request), canonical(request)),
    ).fetchone()[0]
    assert result["status"] == "applied", result


def test_recovery_hold_hides_publication_until_a_new_confirmation(db):
    who = confirmed(db)
    p = claim(db)
    assert finish(db, p)
    assert search(db)
    held = db.execute(
        "SELECT candidate_network_recovery_hold(%s,%s,'profile_rewound')", (who[0], EPOCH)
    ).fetchone()[0]
    assert held == 1
    assert not search(db)
    assert (
        db.execute(
            "SELECT candidate_network_validate(%s::uuid[])", ([p["publicationId"]],)
        ).fetchone()[0]
        == []
    )
    assert (
        db.execute("SELECT candidate_network_owner_status_detail(%s,%s)", who).fetchone()[0][
            "blockedReason"
        ]
        == "publication_held"
    )
    # The worker never revives a held publication; nothing is deleted.
    assert claim(db) is None
    assert db.execute(
        "SELECT state,facts<>'{}' FROM candidate_network_publications WHERE publication_id=%s",
        (p["publicationId"],),
    ).fetchone() == ("held", True)
    # Idempotent: repeating the hold changes nothing further.
    assert (
        db.execute(
            "SELECT candidate_network_recovery_hold(%s,%s,'profile_rewound')", (who[0], EPOCH)
        ).fetchone()[0]
        == 0
    )
    # The candidate reviews and confirms again: a new version publishes normally.
    reconfirm(db, who, "Reviewed after restore")
    p2 = claim(db)
    assert p2 and p2["publicationId"] != p["publicationId"] and finish(db, p2)
    assert search(db)


def test_recovery_functions_are_owner_only_and_gate_is_runtime_readable(db):
    from scripts.init_railway_db import _harden_candidate_network_runtime_privileges

    _harden_candidate_network_runtime_privileges(db.cursor(), "activekg_app")
    can = lambda role, sig: db.execute(  # noqa: E731
        "SELECT has_function_privilege(%s,%s,'EXECUTE')", (role, sig)
    ).fetchone()[0]
    assert can("activekg_app", "candidate_network_recovery_open(text)")
    for sig in (
        "candidate_network_recovery_hold(uuid,text,text)",
        "candidate_network_recovery_attest(text,uuid,text,jsonb,text)",
        "candidate_network_recovery_fingerprint()",
    ):
        assert not can("activekg_app", sig)
    for table in ("candidate_network_recovery_attestations", "candidate_network_recovery_holds"):
        assert not db.execute(
            "SELECT has_table_privilege('activekg_app',%s,'SELECT,INSERT,UPDATE,DELETE')", (table,)
        ).fetchone()[0]


def test_reopening_is_fenced_against_changes_since_the_final_check(db):
    who = confirmed(db)
    fp = db.execute("SELECT candidate_network_recovery_fingerprint()").fetchone()[0]
    # A consent change lands after the reconciliation's final check.
    consent(db, who, "withdraw", 2)
    assert db.execute("SELECT candidate_network_recovery_fingerprint()").fetchone()[0] != fp
    with pytest.raises(psycopg.errors.SerializationFailure), db.transaction():
        db.execute(
            "SELECT candidate_network_recovery_attest(%s,%s,'memory','{}',%s)",
            ("b5-test-epoch-fence000001", str(uuid4()), fp),
        )
    assert not db.execute(
        "SELECT candidate_network_recovery_open('b5-test-epoch-fence000001')"
    ).fetchone()[0]


def test_fingerprint_refuses_a_reader_row_security_would_filter(db):
    db.execute("SET LOCAL ROLE activekg_app")
    with pytest.raises(psycopg.errors.InsufficientPrivilege), db.transaction():
        db.execute("SELECT candidate_network_recovery_fingerprint()")
    db.execute("RESET ROLE")


@pytest.mark.parametrize(
    "change",
    [
        # Same row counts, different content: each mirrors a field the reconciliation reads.
        "UPDATE candidate_consent_receipts SET command_digest=repeat('e',64) WHERE subject_id=%(s)s",
        "UPDATE candidate_owned_subjects SET flow_binding_id=gen_random_uuid() WHERE subject_id=%(s)s",
        "UPDATE candidate_profile_receipts SET command_id=gen_random_uuid() WHERE command_id="
        "(SELECT command_id FROM candidate_profile_receipts WHERE subject_id=%(s)s LIMIT 1)",
    ],
    ids=["receipt_digest", "account_binding", "profile_receipt_id"],
)
def test_fence_catches_same_row_count_content_changes(db, change):
    who = confirmed(db)
    fp = db.execute("SELECT candidate_network_recovery_fingerprint()").fetchone()[0]
    count = "SELECT (SELECT count(*) FROM candidate_consent_receipts),(SELECT count(*) FROM candidate_owned_subjects),(SELECT count(*) FROM candidate_profile_receipts)"
    before = db.execute(count).fetchone()
    # Simulates a restore-level rewrite: append-only/consistency triggers are bypassed for this test only.
    for table in (
        "candidate_consent_receipts",
        "candidate_owned_subjects",
        "candidate_profile_receipts",
    ):
        db.execute(f"ALTER TABLE {table} DISABLE TRIGGER ALL")
    for statement in change.split(";"):
        db.execute(statement, {"s": who[0]})
    assert db.execute(count).fetchone() == before
    assert db.execute("SELECT candidate_network_recovery_fingerprint()").fetchone()[0] != fp
    with pytest.raises(psycopg.errors.SerializationFailure), db.transaction():
        db.execute(
            "SELECT candidate_network_recovery_attest(%s,%s,'memory','{}',%s)",
            ("b5-test-epoch-samecount01", str(uuid4()), fp),
        )
