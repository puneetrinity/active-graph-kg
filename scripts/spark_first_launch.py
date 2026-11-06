"""Empty-estate activation only. Not a restore/reconciliation shortcut."""

from __future__ import annotations

import json
import os
import re
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from scripts.spark_private_job import (
    UUID,
    assert_identity,
    assert_ledger,
    consent,
    load_migration_records,
    permissions,
    read_ledger,
    require,
)

FLOW_TABLES = (
    "candidate_product_accounts",
    "candidate_product_commands",
    "candidate_product_outbox",
    "candidate_network_intents",
    "candidate_profile_commands",
    "candidate_profile_outbox",
    "candidate_consent_signup_intents",
    "candidate_source_objects",
    "candidate_source_requests",
    "candidate_source_jobs",
    "candidate_source_paid_operations",
    "candidate_source_lookups",
    "candidate_source_lookup_requests",
    "candidate_source_interpretations",
    "candidate_source_interpretation_reviews",
    "candidate_source_erasures",
    "candidate_source_ai_spend",
    "candidate_source_ocr_spend",
)
MEMORY_TABLES = (
    "candidate_owned_subjects",
    "candidate_owned_command_receipts",
    "candidate_profile_heads",
    "candidate_profile_drafts",
    "candidate_profile_versions",
    "candidate_preference_versions",
    "candidate_preference_drafts",
    "candidate_profile_receipts",
    "candidate_profile_erasure_audit",
    "candidate_profile_confirmations",
    "candidate_exclusion_proposals",
    "candidate_profile_sources",
    "candidate_profile_proposals",
    "candidate_profile_source_receipts",
    "candidate_profile_interpretations",
    "candidate_profile_interpretation_receipts",
    "candidate_network_subjects",
    "candidate_network_publications",
    "candidate_network_recovery_audit",
    "candidate_network_recovery_holds",
)
SUMMARY = {"firstLaunch": True, "subjects": {"flow": 0, "memory": 0, "union": 0}}


def launch_input(env):
    epoch, run_id = (
        env.get("CANDIDATE_NETWORK_RECOVERY_EPOCH", ""),
        env.get("SPARK_RECOVERY_RUN_ID", ""),
    )
    require(re.fullmatch(r"[A-Za-z0-9_-]{16,128}", epoch))
    require(re.fullmatch(UUID, run_id))
    require(env.get("SPARK_FIRST_LAUNCH_CONFIRM") == "never_activated_no_restore")
    require(env.get("SPARK_RECOVERY_RESTORED", "none") == "none")
    require(re.fullmatch(UUID, env.get("FLOW_SCHEMA_TARGET_ID", "")))
    return epoch, run_id


def flow_tls_settings(env, ca_path, resolve=socket.getaddrinfo):
    try:
        fields = conninfo_to_dict(env.get("FLOW_RECOVERY_DATABASE_URL", ""))
    except Exception:
        raise ValueError("private_job_refused") from None
    require(set(fields) <= {"host", "port", "user", "password", "dbname", "sslmode"})
    require(all(fields.get(k) for k in ("host", "user", "password", "dbname")))
    require(fields["host"] == env.get("FLOW_RECOVERY_HOST"))
    require(fields.get("sslmode", "verify-full") == "verify-full")
    servername = env.get("FLOW_RECOVERY_TLS_SERVERNAME", "")
    require(re.fullmatch(r"[A-Za-z0-9.-]+", servername))
    port = int(fields.get("port", "5432"))
    require(0 < port < 65536)
    addresses = resolve(fields["host"], port, type=socket.SOCK_STREAM)
    require(bool(addresses))
    # libpq connects to hostaddr while verify-full authenticates host against the CA.
    # Never allow a caller-supplied hostaddr or downgrade on connection failure.
    return {
        **fields,
        "host": servername,
        "hostaddr": addresses[0][4][0],
        "sslmode": "verify-full",
        "sslrootcert": ca_path,
        "connect_timeout": 10,
    }


@contextmanager
def flow_connection(env):
    ca = env.get("FLOW_RECOVERY_CA_PEM", "")
    require("-----BEGIN CERTIFICATE-----" in ca)
    with tempfile.NamedTemporaryFile(mode="w", prefix="spark-flow-ca-", suffix=".pem") as cert:
        os.chmod(cert.name, 0o600)
        cert.write(ca)
        cert.flush()
        with psycopg.connect(**flow_tls_settings(env, cert.name)) as conn:
            yield conn


def owner_and_unfiltered(cur, table):
    cur.execute(
        "SELECT (r.rolsuper OR r.rolbypassrls) AND "
        "pg_get_userbyid(c.relowner)=current_user FROM pg_roles r,pg_class c "
        "WHERE r.rolname=current_user AND c.oid=to_regclass(%s)",
        ("public." + table,),
    )
    require(cur.fetchone() == (True,))


def lock_tables(cur, tables, mode="SHARE"):
    require(mode in {"SHARE", "SHARE ROW EXCLUSIVE"})
    cur.execute(
        sql.SQL("LOCK TABLE {} IN " + mode + " MODE").format(
            sql.SQL(",").join(sql.Identifier("public", t) for t in sorted(tables))
        )
    )


def empty(cur, tables):
    for table in tables:
        cur.execute(
            sql.SQL("SELECT EXISTS(SELECT 1 FROM {})").format(sql.Identifier("public", table))
        )
        require(cur.fetchone() == (False,))


def first_launch(memory, flow, config, env):
    epoch, run_id = launch_input(env)
    expected = json.loads(
        Path(__file__).with_name("spark_first_launch_flow_ledger.json").read_text()
    )["migrations"]
    # Memory commits while Flow's write-blocking locks are still held. No Flow rows are changed.
    with flow.transaction(), flow.cursor() as f:
        f.execute("SET LOCAL lock_timeout='3s'")
        f.execute("SET LOCAL statement_timeout='15s'")
        f.execute("SET LOCAL idle_in_transaction_session_timeout='30s'")
        f.execute("SELECT system,environment,target_id FROM schema_control.identity")
        require(f.fetchall() == [("flow", config.environment, env["FLOW_SCHEMA_TARGET_ID"])])
        f.execute("SELECT version,checksum FROM schema_control.applied")
        rows = f.fetchall()
        require(len(rows) == len(expected) and dict(rows) == expected)
        owner_and_unfiltered(f, "candidate_product_accounts")
        lock_tables(f, (*FLOW_TABLES, "candidate_consent_events", "candidate_consent_outbox"))
        with memory.transaction(), memory.cursor() as m:
            m.execute("SET LOCAL lock_timeout='3s'")
            m.execute("SET LOCAL statement_timeout='15s'")
            m.execute("SET LOCAL idle_in_transaction_session_timeout='30s'")
            assert_identity(m, config.target_id, config.environment)
            assert_ledger(read_ledger(m), load_migration_records(), allow_prefix=False)
            owner_and_unfiltered(m, "candidate_owned_subjects")
            permissions(m, "activekg_app")
            consent(m)
            lock_tables(m, ("candidate_network_recovery_attestations",), "SHARE ROW EXCLUSIVE")
            lock_tables(
                m,
                (
                    *MEMORY_TABLES,
                    "candidate_consent_state",
                    "candidate_consent_receipts",
                    "candidate_privacy_directives",
                    "candidate_privacy_directive_events",
                ),
            )
            empty(f, FLOW_TABLES)
            f.execute(
                "SELECT EXISTS(SELECT 1 FROM public.candidate_consent_events WHERE schema_version=2)"
            )
            require(f.fetchone() == (False,))
            empty(m, MEMORY_TABLES)
            m.execute(
                "SELECT EXISTS(SELECT 1 FROM public.candidate_consent_state WHERE effective_schema_version=2)"
            )
            require(m.fetchone() == (False,))
            m.execute(
                "SELECT EXISTS(SELECT 1 FROM public.candidate_consent_receipts WHERE authority_schema_version=2)"
            )
            require(m.fetchone() == (False,))
            m.execute(
                "SELECT epoch,run_id::text,restored,summary FROM public.candidate_network_recovery_attestations"
            )
            prior = m.fetchall()
            if prior:
                require(prior == [(epoch, run_id, "none", SUMMARY)])
                return  # Exact replay only; emptiness was re-proven, not assumed.
            m.execute("SELECT public.candidate_network_recovery_fingerprint()")
            fingerprint = m.fetchone()[0]
            f.execute("SELECT 1")  # Require the Flow lock-holding connection to remain usable.
            require(f.fetchone() == (1,))
            m.execute(
                "SELECT public.candidate_network_recovery_attest(%s,%s::uuid,'none',%s::jsonb,%s)",
                (epoch, run_id, json.dumps(SUMMARY), fingerprint),
            )
            require(m.fetchone() == (True,))
            m.execute("SELECT public.candidate_network_recovery_open(%s)", (epoch,))
            require(m.fetchone() == (True,))
