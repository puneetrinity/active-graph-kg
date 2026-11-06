"""Real signed product HTTP and routine-only roles on complete release schemas.

Explicitly local/opt-in. Flow's prerequisite fixture is made by its actual
release rehearsal; Memory is freshly installed by its real migration runner.
No prototype SQL, auth override, provider call, or production credentials.
"""

import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql


@pytest.mark.skipif(
    os.getenv("SPARK_CROSS_SYSTEM") != "1", reason="local full-schema pair required"
)
def test_signed_consent_lost_reply_and_withdrawal(monkeypatch, tmp_path):
    import uvicorn
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from fastapi import FastAPI

    from activekg.api import auth, candidate_profiles
    from activekg.common.migration_manifest import MIGRATIONS

    admin_url = os.environ["SPARK_TEST_ADMIN_URL"]
    parsed = urlparse(admin_url)
    assert parsed.hostname == "127.0.0.1" and parsed.path == "/postgres"
    assert os.getenv("S2_PROTOTYPE_DISPOSABLE") == "1"
    flow_root = Path(os.environ["S2_FLOW_ROOT"]).resolve()
    flow_db = os.environ["S2_FLOW_DATABASE"]
    match = re.fullmatch(r"flow_schema_control_test_([a-f0-9]{10})", flow_db)
    assert match and flow_root.is_dir()
    flow_port = int(os.environ["S2_FLOW_PORT"])
    assert 1024 < flow_port < 65536
    flow_owner = "flow_schema_control_test_owner_" + match[1]
    flow_runtime = "flow_schema_control_test_runtime_" + match[1]
    flow_owner_url = f"postgresql://{flow_owner}@127.0.0.1:{flow_port}/{flow_db}"
    flow_runtime_url = f"postgresql://{flow_runtime}@127.0.0.1:{flow_port}/{flow_db}"
    with psycopg.connect(flow_owner_url) as flow:
        applied = dict(flow.execute("SELECT version,checksum FROM schema_control.applied"))
        expected = json.loads((flow_root / "server/schema-migrations/checksums.lock").read_text())
        assert applied == expected["migrations"] and len(applied) == 31
        assert flow.execute(
            "SELECT count(*) FROM candidate_profile_outbox WHERE state IN ('queued','retry','leased')"
        ).fetchone() == (0,), "Use an isolated Flow fixture with no runnable profile commands"

    suffix = uuid4().hex[:10]
    owner, runtime, name = (
        f"s2_owner_{suffix}_test",
        f"s2_runtime_{suffix}_test",
        f"s2_memory_{suffix}_test",
    )

    def dsn(role):
        return f"postgresql://{role}@127.0.0.1:{parsed.port}/{name}"

    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(
            sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS CREATEROLE").format(
                sql.Identifier(owner)
            )
        )
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'").format(
                sql.Identifier(name), sql.Identifier(owner)
            )
        )
    with psycopg.connect(dsn(parsed.username), autocommit=True) as admin:
        admin.execute("CREATE EXTENSION vector")
        admin.execute("CREATE EXTENSION pgcrypto")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACTIVEKG_")}
    env.update(
        ACTIVEKG_MIGRATE_DSN=dsn(owner),
        ACTIVEKG_MIGRATION_APPLY="1",
        ACTIVEKG_SCHEMA_TARGET_ID=str(uuid4()),
        ACTIVEKG_SCHEMA_ENVIRONMENT="development",
        ACTIVEKG_SCHEMA_SOURCE_COMMIT="0" * 40,
        ACTIVEKG_SCHEMA_FRESH_INIT="1",
        ACTIVEKG_RUNTIME_ROLE=runtime,
        ACTIVEKG_RUNTIME_PASSWORD="synthetic-local-only",
    )
    for _ in range(2):
        release = subprocess.run(
            [sys.executable, "scripts/init_railway_db.py"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert release.returncode == 0, release.stdout + release.stderr
        env.pop("ACTIVEKG_SCHEMA_FRESH_INIT", None)
    with psycopg.connect(dsn(owner)) as memory:
        assert memory.execute(
            "SELECT count(*),count(*) FILTER (WHERE baselined) FROM schema_migrations"
        ).fetchone() == (len(MIGRATIONS), 0)
    with psycopg.connect(dsn(runtime), autocommit=True) as restricted:
        assert restricted.execute(
            "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
        ).fetchone() == (False, False)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            restricted.execute("SELECT * FROM candidate_profile_heads")

    user = 100_000_000 + int(uuid4().hex[:6], 16)
    copy_hash = "a" * 64
    with psycopg.connect(flow_owner_url) as flow:
        flow.execute(
            "INSERT INTO users(id,username,password,role,email_verified,auth_version,first_name,last_name) VALUES(%s,%s,'synthetic','candidate',true,1,'Synthetic','Consent')",
            (user, f"{uuid4()}@example.invalid"),
        )
        flow.execute(
            "SELECT flow_candidate_foundation_ensure(%s,1,encode(sha256(convert_to('synthetic','UTF8')),'hex'),clock_timestamp(),%s)",
            (user, uuid4()),
        )
        flow.execute(
            "UPDATE candidate_product_accounts SET foundation_state='linked',acknowledged_sequence=command_sequence WHERE user_id=%s",
            (user,),
        )
        subject, binding = flow.execute(
            "SELECT subject_id,binding_id FROM candidate_product_accounts WHERE user_id=%s", (user,)
        ).fetchone()
        flow.execute(
            "INSERT INTO candidate_consent_v2_copies VALUES(2,%s,true) ON CONFLICT DO NOTHING",
            (copy_hash,),
        )
    with psycopg.connect(dsn(runtime)) as memory:
        memory.execute(
            "SELECT candidate_owned_ensure(%s,%s,%s,1,%s)", (subject, binding, uuid4(), copy_hash)
        )
    with psycopg.connect(dsn(owner)) as memory:
        memory.execute("INSERT INTO candidate_consent_copy_catalog VALUES(2,%s,true)", (copy_hash,))

    def capture(action, version):
        with psycopg.connect(flow_owner_url) as flow:
            if action == "grant" and version == 0:
                # The initial grant comes from the stored signup choice, not a
                # second browser checkbox. Later grants remain explicit actions.
                flow.execute("UPDATE users SET email_verified=false WHERE id=%s", (user,))
                flow.execute(
                    "SELECT flow_consent_v2_signup_intent(%s,%s,2,%s)",
                    (user, uuid4(), copy_hash),
                )
                flow.execute("UPDATE users SET email_verified=true WHERE id=%s", (user,))
                activated = flow.execute(
                    "SELECT flow_consent_v2_signup_activate(%s,1)", (user,)
                ).fetchone()[0]
                assert activated["status"] == "activated"
                receipt = activated["receipt"]
                command = flow.execute(
                    "SELECT flow_consent_v2_wire(%s)", (receipt["eventId"],)
                ).fetchone()[0]
                return {"eventId": receipt["eventId"], "version": 1, "command": command}
            return flow.execute(
                "SELECT flow_consent_v2_capture(%s,1,%s,%s,%s,%s,%s,2,%s)",
                (user, subject, binding, uuid4(), version, action, copy_hash),
            ).fetchone()[0]

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    public = (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    for k, v in {
        "JWT_ENABLED": True,
        "JWT_ALGORITHM": "RS256",
        "JWT_AUDIENCE": "activekg",
        "JWT_ISSUER": "vantahire",
        "JWT_PUBLIC_KEY": public,
        "JWT_LEEWAY_SECONDS": 0,
    }.items():
        monkeypatch.setattr(auth, k, v)
    monkeypatch.setenv("ACTIVEKG_DSN", dsn(runtime))
    monkeypatch.setenv("CANDIDATE_FOUNDATION_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_CONSENT_V2_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_PROFILE_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_SPARK_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_SOURCE_IMPORT_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_INTERPRETATION_ENABLED", "true")
    app = FastAPI()
    app.include_router(candidate_profiles.router)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = uvicorn.Server(
        uvicorn.Config(app, log_level="error", access_log=False, lifespan="off")
    )
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()

    def run(
        mode,
        event,
        claimed=1,
        sends=1,
        receipt=None,
        command=None,
        profile_lease=None,
        restored_url=None,
    ):
        child = {
            **os.environ,
            "SPARK_CROSS_SYSTEM": "1",
            "SPARK_FLOW_RUNTIME_URL": flow_runtime_url,
            "SPARK_MEMORY_ORIGIN": f"http://127.0.0.1:{listener.getsockname()[1]}",
            "VANTAHIRE_JWT_PRIVATE_KEY": private,
            "SPARK_SCENARIO": mode,
            "SPARK_EVENT_ID": event["eventId"],
            "SPARK_EXPECT_CLAIMED": str(claimed),
            "SPARK_EXPECT_SENDS": str(sends),
            "SPARK_COMMAND": json.dumps(command or event["command"]),
            "SPARK_EXPECT_RECEIPT": json.dumps(receipt),
            "SPARK_PROFILE_LEASE": json.dumps(profile_lease),
        }
        result = subprocess.run(
            [
                str(flow_root / "node_modules/.bin/vitest"),
                "run",
                "--config",
                "vitest.server.config.ts",
                "server/candidate-product/spark-cross-system.test.ts",
            ],
            cwd=flow_root,
            env={
                **child,
                **(
                    {
                        "SPARK_TEST_FLOW_OWNER_URL": flow_owner_url,
                        "SPARK_TEST_MEMORY_OWNER_URL": restored_url or dsn(owner),
                    }
                    if mode in ("operator_erase", "operator_erase_restored")
                    else {}
                ),
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 and mode == "lookup_lost_reply":
            with psycopg.connect(flow_owner_url) as diagnostic:
                print(
                    "Synthetic active budget diagnostic",
                    diagnostic.execute(
                        "SELECT period,stopped,credit_limit,(SELECT coalesce(sum(coalesce(charged_credits,reserved_credits)),0) "
                        "FROM candidate_source_lookups WHERE day_window=w.window_id OR month_window=w.window_id) "
                        "FROM candidate_source_lookup_budgets w WHERE starts_at<=clock_timestamp() AND ends_at>clock_timestamp()"
                    ).fetchall(),
                )
        assert result.returncode == 0, result.stdout + result.stderr

    def flow_row(event):
        with psycopg.connect(flow_owner_url) as flow:
            return flow.execute(
                "SELECT state,attempts FROM candidate_consent_outbox WHERE event_id=%s",
                (event["eventId"],),
            ).fetchone()

    def due(event):
        with psycopg.connect(flow_owner_url) as flow:
            flow.execute(
                "UPDATE candidate_consent_outbox SET next_attempt_at=clock_timestamp() WHERE event_id=%s",
                (event["eventId"],),
            )

    source_grants = []
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            assert thread.is_alive() and time.monotonic() < deadline
            time.sleep(0.01)
        grant = capture("grant", 0)
        for mode in ("wrong_tenant", "wrong_scope"):
            run(mode, grant)
        run("wrong_binding", grant, command={**grant["command"], "bindingId": str(uuid4())})
        run("wrong_subject", grant, command={**grant["command"], "subjectId": str(uuid4())})
        with psycopg.connect(dsn(owner)) as memory:
            memory.execute(
                "SELECT set_config('app.current_tenant_id',%s,true)", (f"candidate_{subject}",)
            )
            assert memory.execute(
                "SELECT count(*) FROM candidate_consent_receipts WHERE event_id=%s",
                (grant["eventId"],),
            ).fetchone() == (0,)
        run("privacy_unavailable", grant, sends=0)
        assert flow_row(grant) == ("retry_wait", 1)
        due(grant)
        run("lost_reply", grant)
        assert flow_row(grant) == ("retry_wait", 2)
        with psycopg.connect(dsn(owner)) as memory:
            memory.execute(
                "SELECT set_config('app.current_tenant_id',%s,true)", (f"candidate_{subject}",)
            )
            assert memory.execute(
                "SELECT outcome FROM candidate_consent_receipts WHERE event_id=%s",
                (grant["eventId"],),
            ).fetchone() == ("onboarding_granted",)
        due(grant)
        run("worker", grant)
        assert flow_row(grant) == ("acknowledged", 3)
        run("worker", grant, claimed=0, sends=0)
        assert flow_row(grant) == ("acknowledged", 3)
        withdrawal = capture("withdraw", 1)
        run("privacy_unavailable", withdrawal)  # Withdrawals must not depend on a privacy read.
        assert flow_row(withdrawal) == ("acknowledged", 1)
        original = {
            "eventId": grant["eventId"],
            "version": 1,
            "effectiveVersion": 1,
            "outcome": "onboarding_granted",
        }
        run("replay", grant, receipt=original)
        with psycopg.connect(dsn(owner)) as memory:
            memory.execute(
                "SELECT set_config('app.current_tenant_id',%s,true)", (f"candidate_{subject}",)
            )
            assert memory.execute(
                "SELECT effective_version,effective_action FROM candidate_consent_state WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (2, "withdraw")
            assert memory.execute(
                "SELECT count(*) FROM candidate_consent_receipts WHERE event_id=%s",
                (grant["eventId"],),
            ).fetchone() == (1,)
        with psycopg.connect(flow_owner_url) as flow:
            assert flow.execute(
                "SELECT acknowledged_version,acknowledged_action FROM candidate_consent_subjects WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (2, "withdraw")
        # A withdrawal also supersedes an unacknowledged grant whose response
        # was lost. Neither a queued retry nor a direct old replay reactivates it.
        pending = capture("grant", 2)
        run("lost_reply", pending)
        assert flow_row(pending) == ("retry_wait", 1)
        latest = capture("withdraw", 3)
        run("privacy_unavailable", latest)
        assert flow_row(pending) == ("superseded", 1)
        due(pending)
        run("worker", pending, claimed=0, sends=0)
        run(
            "replay",
            pending,
            receipt={
                "eventId": pending["eventId"],
                "version": 3,
                "effectiveVersion": 3,
                "outcome": "onboarding_granted",
            },
        )
        with psycopg.connect(dsn(owner)) as memory:
            memory.execute(
                "SELECT set_config('app.current_tenant_id',%s,true)", (f"candidate_{subject}",)
            )
            assert memory.execute(
                "SELECT effective_version,effective_action,global_candidate_id FROM candidate_consent_state WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (4, "withdraw", None)
            assert memory.execute(
                "SELECT count(*) FROM candidate_consent_receipts WHERE subject_id=%s", (subject,)
            ).fetchone() == (4,)
        with psycopg.connect(flow_owner_url) as flow:
            assert flow.execute(
                "SELECT acknowledged_version,acknowledged_action FROM candidate_consent_subjects WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (4, "withdraw")
        # Private preferences remain editable after network withdrawal. Exercise
        # the real Flow v3 sender and Memory API/runtime role, including a lost
        # committed reply, followed by the real durable Flow queue below.
        from activekg.candidate_profiles.spark_contracts import parse_command

        preference = {
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
        parsed_preference = parse_command(json.dumps(preference).encode())
        profile_lease = {
            "commandId": str(uuid4()),
            "subjectId": str(subject),
            "bindingId": str(binding),
            "userId": user,
            "authVersion": 1,
            "idempotencyKey": preference["idempotencyKey"],
            "leaseToken": str(uuid4()),
            "leaseGeneration": 1,
            "attemptCount": 1,
            "body": parsed_preference["canonical"],
            "hash": parsed_preference["hash"],
        }
        run("spark_lost_reply", grant, profile_lease=profile_lease)
        run("spark_save", grant, profile_lease=profile_lease)
        run("spark_save", grant, profile_lease=profile_lease)
        with psycopg.connect(dsn(owner)) as memory:
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_receipts WHERE subject_id=%s", (subject,)
            ).fetchone() == (1,)
            assert memory.execute(
                "SELECT revision FROM candidate_preference_drafts WHERE subject_id=%s", (subject,)
            ).fetchone() == (1,)
            assert memory.execute(
                "SELECT profile_version,preference_version FROM candidate_profile_heads WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (0, 0)
        preference["idempotencyKey"] = str(uuid4())
        preference["expectedPreferenceDraftRevision"] = 1
        preference["payload"]["value"] = ["permanent"]
        queued = parse_command(json.dumps(preference).encode())
        profile_lease.update(
            body=queued["canonical"],
            hash=queued["hash"],
            idempotencyKey=preference["idempotencyKey"],
        )
        run("queue_lost_reply", grant, profile_lease=profile_lease)
        with psycopg.connect(flow_owner_url) as flow:
            row = flow.execute(
                "SELECT o.state,o.receipt,o.attempts,c.command_id FROM candidate_profile_outbox o "
                "JOIN candidate_profile_commands c USING(command_id) WHERE c.user_id=%s AND c.body_hash=%s",
                (user, queued["hash"]),
            ).fetchone()
            assert row[:3] == ("retry", None, 1)
            queued_id = row[3]
            # Advance only this synthetic retry clock, not its lease or identity.
            flow.execute(
                "UPDATE candidate_profile_outbox SET next_attempt_at=clock_timestamp() WHERE command_id=%s",
                (queued_id,),
            )
        run("queue_recover", grant, profile_lease=profile_lease)
        run("queue_replay", grant, profile_lease=profile_lease)
        with psycopg.connect(flow_owner_url) as flow, psycopg.connect(dsn(owner)) as memory:
            row = flow.execute(
                "SELECT state,receipt,attempts FROM candidate_profile_outbox WHERE command_id=%s",
                (queued_id,),
            ).fetchone()
            assert row[0] == "applied" and row[2] == 2
            assert row[1]["preferenceDraftRevision"] == 2
            assert row[1]["commandId"] == str(queued_id)
            assert memory.execute(
                "SELECT result FROM candidate_profile_receipts WHERE subject_id=%s AND command_id=%s",
                (subject, queued_id),
            ).fetchone() == (row[1],)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_receipts WHERE subject_id=%s", (subject,)
            ).fetchone() == (2,)
            assert memory.execute(
                "SELECT revision FROM candidate_preference_drafts WHERE subject_id=%s", (subject,)
            ).fetchone() == (2,)
            assert memory.execute(
                "SELECT profile_version,preference_version FROM candidate_profile_heads WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (0, 0)
        profile_lease["commandId"] = str(queued_id)
        run("expiry_lost_reply", grant, profile_lease=profile_lease)
        run("expiry_recover", grant, profile_lease=profile_lease)
        preference["idempotencyKey"] = str(uuid4())
        preference["expectedPreferenceDraftRevision"] = 2
        preference["payload"]["value"] = ["contract"]
        unsent = parse_command(json.dumps(preference).encode())
        profile_lease.update(
            commandId=str(uuid4()),
            idempotencyKey=preference["idempotencyKey"],
            body=unsent["canonical"],
            hash=unsent["hash"],
        )
        run("expiry_unsent", grant, profile_lease=profile_lease)
        run("late_expired", grant, profile_lease=profile_lease)
        with psycopg.connect(dsn(owner)) as memory:
            assert memory.execute(
                "SELECT revision FROM candidate_preference_drafts WHERE subject_id=%s", (subject,)
            ).fetchone() == (2,)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_receipts WHERE subject_id=%s", (subject,)
            ).fetchone() == (3,)
        # Exercise the actual retention/recovery worker, not just its HTTP client.
        for was_delivered in (True, False):
            preference["idempotencyKey"] = str(uuid4())
            preference["expectedPreferenceDraftRevision"] = 2 if was_delivered else 3
            request = parse_command(json.dumps(preference).encode())
            profile_lease.update(
                idempotencyKey=preference["idempotencyKey"],
                body=request["canonical"],
                hash=request["hash"],
            )
            run(
                "queue_lost_reply" if was_delivered else "queue_only",
                grant,
                profile_lease=profile_lease,
            )
            with psycopg.connect(flow_owner_url) as flow:
                recovery_id = flow.execute(
                    "SELECT command_id FROM candidate_profile_commands WHERE user_id=%s AND body_hash=%s",
                    (user, request["hash"]),
                ).fetchone()[0]
                flow.execute(
                    "UPDATE candidate_profile_commands SET created_at=clock_timestamp()-interval '31 days' WHERE command_id=%s",
                    (recovery_id,),
                )
                flow.execute(
                    "UPDATE candidate_profile_outbox SET next_attempt_at=clock_timestamp() WHERE command_id=%s",
                    (recovery_id,),
                )
            profile_lease["commandId"] = str(recovery_id)
            run("recovery_worker_lost", grant, profile_lease=profile_lease)
            with psycopg.connect(flow_owner_url) as flow:
                assert flow.execute(
                    "SELECT body FROM candidate_profile_commands WHERE command_id=%s",
                    (recovery_id,),
                ).fetchone() == (None,)
                assert flow.execute(
                    "SELECT receipt,lease_token FROM candidate_profile_outbox WHERE command_id=%s",
                    (recovery_id,),
                ).fetchone() == (None, None)
                flow.execute(
                    "UPDATE candidate_profile_outbox SET next_attempt_at=clock_timestamp() WHERE command_id=%s",
                    (recovery_id,),
                )
            run(
                "recovery_worker_applied" if was_delivered else "recovery_worker_expired",
                grant,
                profile_lease=profile_lease,
            )
            with psycopg.connect(flow_owner_url) as flow, psycopg.connect(dsn(owner)) as memory:
                result = flow.execute(
                    "SELECT receipt FROM candidate_profile_outbox WHERE command_id=%s",
                    (recovery_id,),
                ).fetchone()[0]
                assert memory.execute(
                    "SELECT result FROM candidate_profile_receipts WHERE command_id=%s",
                    (recovery_id,),
                ).fetchone() == (result,)
                assert memory.execute(
                    "SELECT revision FROM candidate_preference_drafts WHERE subject_id=%s",
                    (subject,),
                ).fetchone() == (3,)
        # Confirm using the actual Flow GET response, never fixture fingerprints.
        final_grant = capture("grant", 4)
        run("worker", final_grant)
        run("confirm_from_get", final_grant, profile_lease=profile_lease)
        # Actual P2 worker + durable paid slots + signed receiver. Only file bytes,
        # text extraction and model response are synthetic; no provider is called.
        source_names = [
            "snapshot",
            "resolve",
            "enqueue",
            "status",
            "claim",
            "prepare",
            "transfer_store",
            "transfer_finish",
            "fail",
            "paid_reserve",
            "paid_dispatch",
            "paid_settle",
            "paid_hold",
            "paid_release_unsent",
            "locator",
            "register_object",
            "object_read",
            "extraction_cache",
            "retry",
            "retry_status",
            "lookup_admit",
            "lookup_status",
            "lookup_dispatch",
            "lookup_finish",
            "lookup_hold",
            "lookup_cancel_unsent",
            "lookup_delivery_read",
            "lookup_delivery_store",
            "lookup_delivery_finish",
            "lookup_claim",
            "lookup_work_prepare",
            "lookup_work_finish",
        ]
        with psycopg.connect(flow_owner_url) as flow:
            source_grants = [
                r[0]
                for r in flow.execute(
                    "SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                    "WHERE n.nspname='public' AND p.proname=ANY(%s)",
                    (["flow_candidate_source_" + n for n in source_names],),
                )
            ]
            assert len(source_grants) == len(source_names)
            # Preserve real provisioner grants; revoke only privileges added by this test.
            source_grants = [
                signature
                for signature in source_grants
                if not flow.execute(
                    "SELECT has_function_privilege(%s,%s,'EXECUTE')", (flow_runtime, signature)
                ).fetchone()[0]
            ]
            assert not source_grants, (
                "Real release provisioner must grant all required source functions"
            )
            for model in ("synthetic-model", "synthetic-no-provider"):
                flow.execute(
                    "INSERT INTO candidate_source_ai_prices VALUES(%s,%s,1,1,'2020-01-01','2099-01-01','Synthetic rehearsal price, not production') ON CONFLICT DO NOTHING",
                    ("cross-system-" + model, model),
                )
            flow.execute(
                "UPDATE candidate_source_ai_policy SET enabled=true,candidate_day_micro=250000,candidate_month_micro=2000000,global_day_micro=10000000,global_month_micro=100000000"
            )
            flow.execute(
                "INSERT INTO candidate_source_ocr_prices SELECT 'cross-system-ocr',1500,'2020-01-01'::timestamptz,'2099-01-01'::timestamptz,'Synthetic rehearsal price' WHERE NOT EXISTS(SELECT 1 FROM candidate_source_ocr_prices WHERE valid_from<=clock_timestamp() AND valid_until>clock_timestamp())"
            )
            # The release fixture's source tests leave their own budget windows.
            # End them before creating this scenario's non-overlapping windows.
            flow.execute(
                "UPDATE candidate_source_budget_windows SET ends_at=clock_timestamp()-interval '1 minute' "
                "WHERE ends_at>clock_timestamp() AND starts_at<clock_timestamp()"
            )
            flow.execute(
                "INSERT INTO candidate_source_budget_windows(kind,period,starts_at,ends_at,global_units,per_candidate_units) "
                "SELECT k,p,clock_timestamp()-interval '1 day',clock_timestamp()+interval '1 day',1000000,1000000 "
                "FROM unnest(ARRAY['day','month']) p CROSS JOIN unnest(ARRAY['ocr','semantic_model']) k"
            )
            resume_id = flow.execute(
                "INSERT INTO candidate_resumes(user_id,label,gcs_path,is_default) VALUES(%s,'Synthetic',%s,false) RETURNING id",
                (user, f"resumes/{uuid4()}.docx"),
            ).fetchone()[0]
        source_input = {
            "userId": user,
            "subjectId": str(subject),
            "bindingId": str(binding),
            "resumeId": resume_id,
            "snapshotKey": str(uuid4()),
            "enqueueKey": str(uuid4()),
        }
        run("resume_lost_reply", final_grant, profile_lease=source_input)
        with psycopg.connect(flow_owner_url) as flow, psycopg.connect(dsn(owner)) as memory:
            command_id, transfer, receipt = flow.execute(
                "SELECT command_id,transfer_document,receipt_document FROM candidate_source_jobs WHERE user_id=%s",
                (user,),
            ).fetchone()
            assert receipt is None
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_source_receipts WHERE command_id=%s",
                (command_id,),
            ).fetchone() == (1,)
            flow.execute(
                "UPDATE candidate_source_jobs SET lease_until=clock_timestamp()-interval '1 second' WHERE command_id=%s",
                (command_id,),
            )
        run("resume_recover", final_grant, profile_lease=source_input)
        with psycopg.connect(flow_owner_url) as flow, psycopg.connect(dsn(owner)) as memory:
            state, replay, receipt = flow.execute(
                "SELECT state,transfer_document,receipt_document FROM candidate_source_jobs WHERE command_id=%s",
                (command_id,),
            ).fetchone()
            assert state == "review_ready" and replay == transfer
            assert memory.execute(
                "SELECT result FROM candidate_profile_source_receipts WHERE command_id=%s",
                (command_id,),
            ).fetchall() == [(json.loads(receipt["canonical"]),)]
            assert flow.execute(
                "SELECT kind,state,count(*) FROM candidate_source_paid_operations WHERE command_id=%s GROUP BY kind,state",
                (command_id,),
            ).fetchall() == [("semantic_model", "settled", 1)]
            assert memory.execute(
                "SELECT state FROM candidate_profile_proposals WHERE subject_id=%s",
                (subject,),
            ).fetchall() == [("proposed",)]
            account_email = flow.execute(
                "SELECT username FROM users WHERE id=%s", (user,)
            ).fetchone()[0]
            assert account_email not in json.dumps(transfer)
            assert account_email not in json.dumps(receipt)
        stored_proposal = json.loads(transfer["canonical"])["proposal"]
        run(
            "source_accept",
            final_grant,
            profile_lease={
                **source_input,
                "sourceId": stored_proposal["sourceId"],
                "proposalId": stored_proposal["proposalId"],
            },
        )
        with psycopg.connect(dsn(owner)) as memory:
            assert memory.execute(
                "SELECT state FROM candidate_profile_proposals WHERE proposal_id=%s",
                (stored_proposal["proposalId"],),
            ).fetchone() == ("accepted",)
        with psycopg.connect(flow_owner_url) as flow:
            scan_id = flow.execute(
                "INSERT INTO candidate_resumes(user_id,label,gcs_path,is_default) VALUES(%s,'Synthetic scan',%s,false) RETURNING id",
                (user, f"resumes/{uuid4()}.pdf"),
            ).fetchone()[0]
        run(
            "resume_ocr_retry",
            final_grant,
            profile_lease={
                **source_input,
                "resumeId": scan_id,
                "snapshotKey": str(uuid4()),
                "enqueueKey": str(uuid4()),
            },
        )
        with psycopg.connect(flow_owner_url) as flow, psycopg.connect(dsn(owner)) as memory:
            # One settled OCR operation across both commands; the uncertain first
            # AI charge stays held, and only the explicit retry buys a new AI call.
            jobs = flow.execute(
                "SELECT j.command_id,j.previous_command_id,j.state,j.receipt_document FROM candidate_source_jobs j "
                "JOIN candidate_source_objects s USING(source_id) WHERE s.resume_id=%s ORDER BY j.created_at",
                (scan_id,),
            ).fetchall()
            assert len(jobs) == 2 and jobs[1][1] == jobs[0][0]
            assert jobs[1][2] == "review_ready"
            assert flow.execute(
                "SELECT kind,state,count(*) FROM candidate_source_paid_operations WHERE command_id=ANY(%s) GROUP BY kind,state ORDER BY kind,state",
                ([j[0] for j in jobs],),
            ).fetchall() == [
                ("ocr", "settled", 1),
                ("semantic_model", "held", 1),
                ("semantic_model", "settled", 1),
            ]
            assert memory.execute(
                "SELECT result FROM candidate_profile_source_receipts WHERE command_id=%s",
                (jobs[1][0],),
            ).fetchall() == [(json.loads(jobs[1][3]["canonical"]),)]
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_source_receipts WHERE command_id=%s",
                (jobs[0][0],),
            ).fetchone() == (0,)
        with psycopg.connect(flow_owner_url) as flow:
            flow.execute(
                "UPDATE candidate_source_lookup_budgets SET ends_at=clock_timestamp()-interval '1 minute' WHERE ends_at>clock_timestamp()-interval '1 minute'"
            )
            flow.execute(
                "INSERT INTO candidate_source_lookup_budgets(period,starts_at,ends_at,credit_limit) "
                "SELECT p,clock_timestamp()-interval '1 day',clock_timestamp()+interval '1 day',100 "
                "FROM unnest(ARRAY['day','month']) p"
            )
        lookup_input = {
            "userId": user,
            "subjectId": str(subject),
            "bindingId": str(binding),
            "requestKey": str(uuid4()),
            "consentRevision": 5,
        }
        run("lookup_lost_reply", final_grant, profile_lease=lookup_input)
        # Advance only the local retry schedule; no paid slot/result is reset.
        with psycopg.connect(flow_owner_url, autocommit=True) as flow:
            flow.execute(
                "UPDATE candidate_source_lookups SET next_work_at='-infinity' WHERE state='transferring'"
            )
        run("lookup_recover", final_grant, profile_lease=lookup_input)
        with psycopg.connect(flow_owner_url) as flow, psycopg.connect(dsn(owner)) as memory:
            attempt, receipt, credits, lookup_transfer = flow.execute(
                "SELECT attempt_id,receipt_document,charged_credits,transfer_document FROM candidate_source_lookups WHERE user_id=%s",
                (user,),
            ).fetchone()
            assert credits == 2
            assert memory.execute(
                "SELECT result FROM candidate_profile_source_receipts WHERE command_id=%s",
                (attempt,),
            ).fetchall() == [(json.loads(receipt["canonical"]),)]
        source_withdrawal = capture("withdraw", 5)
        run("worker", source_withdrawal)
        run("resume_after_withdrawal", source_withdrawal, profile_lease=transfer)
        run("lookup_after_withdrawal", source_withdrawal, profile_lease=lookup_transfer)
        with psycopg.connect(dsn(owner)) as memory:
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_proposals WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (3,)
            assert memory.execute(
                "SELECT result->>'outcome',count(*) FROM candidate_profile_source_receipts "
                "WHERE subject_id=%s GROUP BY result->>'outcome' ORDER BY 1",
                (subject,),
            ).fetchall() == [("rejected", 2), ("stored", 3)]
        with psycopg.connect(dsn(owner)) as memory:
            assert memory.execute(
                "SELECT profile_version,preference_version FROM candidate_profile_heads WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (1, 1)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_confirmations WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (1,)
        # P6 uses a fake model only; its paid journal, signed transport, Memory
        # storage/read and final acceptance through Flow's queue are real.
        ai_grant = capture("grant", source_withdrawal["version"])
        run("worker", ai_grant)
        with psycopg.connect(flow_owner_url) as flow:
            flow.execute(
                "INSERT INTO candidate_source_interpretation_budgets "
                "SELECT p,clock_timestamp()-interval '1 hour',clock_timestamp()+interval '1 hour',100000,100000,10,10 "
                "FROM unnest(ARRAY['day','month']) p"
            )
        run(
            "interpretation_lost_reply",
            ai_grant,
            profile_lease={
                "subjectId": str(subject),
                "bindingId": str(binding),
                "userId": user,
                "consentRevision": ai_grant["version"],
            },
        )
        with psycopg.connect(dsn(owner)) as memory:
            assert memory.execute(
                "SELECT count(*),min(state) FROM candidate_profile_interpretations WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (1, "accepted")
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_interpretation_receipts WHERE subject_id=%s",
                (subject,),
            ).fetchone() == (1,)
        erasure_id = str(uuid4())
        with psycopg.connect(flow_owner_url) as flow:
            flow.execute(
                "INSERT INTO candidate_privacy_requests(request_id,action,authority_type,actor_user_id,reason_code) "
                "VALUES(%s,'request_erasure','verified_candidate',%s,'candidate_erasure_request')",
                (erasure_id, user),
            )
            flow.execute(
                "INSERT INTO candidate_privacy_subject_links(link_id,request_id,subject_type,candidate_user_id) "
                "VALUES(%s,%s,'candidate_user',%s)",
                (str(uuid4()), erasure_id, user),
            )
        erasure = {"requestId": erasure_id, "subjectId": str(subject), "bindingId": str(binding)}
        run("erase_runtime_denied", source_withdrawal, profile_lease=erasure)
        with psycopg.connect(flow_owner_url) as flow:
            assert (
                flow.execute(
                    "SELECT stage FROM candidate_source_erasures WHERE request_id=%s", (erasure_id,)
                ).fetchone()
                is None
            )
        with psycopg.connect(flow_owner_url) as flow, psycopg.connect(dsn(owner)) as memory:
            assert (
                flow.execute(
                    "SELECT stage FROM candidate_source_erasures WHERE request_id=%s", (erasure_id,)
                ).fetchone()
                is None
            )
            assert (
                flow.execute(
                    "SELECT count(*) FROM candidate_source_jobs WHERE user_id=%s AND "
                    "(extraction_document IS NOT NULL OR transfer_document IS NOT NULL)",
                    (user,),
                ).fetchone()[0]
                > 0
            )
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_erasure_audit WHERE request_id=%s",
                (erasure_id,),
            ).fetchone() == (0,)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_proposals WHERE subject_id=%s AND payload IS NOT NULL",
                (subject,),
            ).fetchone() == (3,)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_versions WHERE subject_id=%s AND facts IS NOT NULL",
                (subject,),
            ).fetchone() == (1,)
        directive_id = str(uuid4())
        with psycopg.connect(dsn(owner)) as memory:
            memory.execute(
                "SELECT * FROM candidate_privacy_create_directive(%s,%s,'request_erasure','active_profile','verified_candidate',%s,'candidate_erasure_request','vantahire','synthetic',%s,NULL,NULL,1,jsonb_build_array(jsonb_build_object('identifier_type','email','key_version',1,'token',repeat('a',64))),false,clock_timestamp())",
                (directive_id, erasure_id, str(uuid4()), None),
            )
        with psycopg.connect(flow_owner_url) as flow:
            flow.execute(
                "UPDATE candidate_privacy_requests SET state='memory_active',directive_id=%s WHERE request_id=%s",
                (directive_id, erasure_id),
            )
            # Synthetic fixture paths use no bucket; bind them explicitly for
            # the operator's injected fake object store (no cloud call).
            flow.execute(
                "UPDATE candidate_resumes SET gcs_path='gs://synthetic-private/'||gcs_path WHERE user_id=%s AND gcs_path NOT LIKE 'gs://%%'",
                (user,),
            )
            flow.execute(
                "UPDATE candidate_source_objects SET gcs_path='gs://synthetic-private/'||gcs_path WHERE user_id=%s AND gcs_path NOT LIKE 'gs://%%'",
                (user,),
            )
        # Back up the quarantined but not yet erased subject. Restore into a
        # separate disposable database so the original HTTP fixture is untouched.
        backup = tmp_path / "quarantined-memory.dump"
        dumped = subprocess.run(
            [
                "/usr/lib/postgresql/16/bin/pg_dump",
                "-Fc",
                "--dbname",
                dsn(parsed.username),
                "--file",
                str(backup),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert dumped.returncode == 0, dumped.stderr
        run("operator_erase", source_withdrawal, profile_lease=erasure)
        restored_name = f"s2_restored_{suffix}_test"
        restored_url = f"postgresql://{owner}@127.0.0.1:{parsed.port}/{restored_name}"
        with psycopg.connect(admin_url, autocommit=True) as admin:
            admin.execute(
                sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'").format(
                    sql.Identifier(restored_name), sql.Identifier(owner)
                )
            )
        restored_admin = f"postgresql://{parsed.username}@127.0.0.1:{parsed.port}/{restored_name}"
        restored = subprocess.run(
            [
                "/usr/lib/postgresql/16/bin/pg_restore",
                "--exit-on-error",
                "--dbname",
                restored_admin,
                str(backup),
            ],
            capture_output=True,
            text=True,
            timeout=90,
        )
        assert restored.returncode == 0, restored.stderr
        with psycopg.connect(restored_url) as memory:
            assert memory.execute(
                "SELECT candidate_network_recovery_open(%s)", (str(uuid4()),)
            ).fetchone() == (False,)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_proposals WHERE subject_id=%s AND payload IS NOT NULL",
                (subject,),
            ).fetchone() == (3,)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_erasure_audit WHERE request_id=%s",
                (erasure_id,),
            ).fetchone() == (0,)
        run(
            "operator_erase_restored",
            source_withdrawal,
            profile_lease=erasure,
            restored_url=restored_url,
        )
        with psycopg.connect(restored_url) as memory:
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_proposals WHERE subject_id=%s AND payload IS NOT NULL",
                (subject,),
            ).fetchone() == (0,)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_versions WHERE subject_id=%s AND facts IS NOT NULL",
                (subject,),
            ).fetchone() == (0,)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_erasure_audit WHERE request_id=%s",
                (erasure_id,),
            ).fetchone() == (1,)
        print(
            "RESTORE_THEN_OPERATOR_ERASURE_PASS",
            restored_name,
            "real pg_dump/pg_restore; fake blob store; no Discover database",
        )
        with psycopg.connect(flow_owner_url) as flow, psycopg.connect(dsn(owner)) as memory:
            assert flow.execute(
                "SELECT stage FROM candidate_source_erasures WHERE request_id=%s", (erasure_id,)
            ).fetchone() == ("complete",)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_erasure_audit WHERE request_id=%s",
                (erasure_id,),
            ).fetchone() == (1,)
            assert memory.execute(
                "SELECT count(*) FROM candidate_profile_proposals WHERE subject_id=%s AND payload IS NOT NULL",
                (subject,),
            ).fetchone() == (0,)
        print(
            "Product consent and v3 preference HTTP proof passed; retained local Memory database",
            name,
        )
    finally:
        with psycopg.connect(flow_owner_url) as flow:
            for signature in source_grants:
                flow.execute(
                    sql.SQL("REVOKE EXECUTE ON FUNCTION {} FROM {}").format(
                        sql.SQL(signature), sql.Identifier(flow_runtime)
                    )
                )
        server.should_exit = True
        thread.join(timeout=5)
        listener.close()
        assert not thread.is_alive()
