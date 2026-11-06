"""One-shot Spark owner operations. Never imported by API/worker startup."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import psycopg
from psycopg.conninfo import conninfo_to_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from activekg.common.schema_control import (  # noqa: E402
    assert_identity,
    assert_ledger,
    load_migration_records,
    read_ledger,
    resolve_control_environment,
)

COPY_HASH = "bb133dc6ab8120ad2e866bacec8253eb163415a7d215a80d6fc447c1ea23ac5e"
PROJECT = "5042caf8-8879-40ab-bb52-aa9f8af8e508"
ENVIRONMENT = "c3b75445-d0de-49fa-a533-b4b9a20b5f5d"
MODES = {"readback", "consent-register", "recovery", "first-launch"}
UUID = r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"


def require(value):
    if not value:
        raise ValueError("private_job_refused")


def private_dsn(dsn, expected_host):
    """Rebuild a minimal libpq configuration: no service/hostaddr/options escape."""
    try:
        fields = conninfo_to_dict(dsn)
    except Exception:
        raise ValueError("private_job_refused") from None
    require(re.fullmatch(r"[a-z0-9-]+\.railway\.internal", expected_host or ""))
    require(fields.get("host") == expected_host)
    require(set(fields) <= {"host", "port", "user", "password", "dbname", "sslmode"})
    require(all(fields.get(k) for k in ("user", "password", "dbname")))
    require(fields.get("sslmode", "disable") == "disable")
    return {**fields, "sslmode": "disable", "connect_timeout": 10}


def preflight(mode, env):
    require(mode in MODES and env.get("SPARK_PRIVATE_JOB_APPLY") == mode)
    require(env.get("RAILWAY_PROJECT_ID") == PROJECT)
    require(env.get("RAILWAY_ENVIRONMENT_ID") == ENVIRONMENT)
    service = env.get("SPARK_PRIVATE_JOB_SERVICE_ID", "")
    require(re.fullmatch(UUID, service))
    require(env.get("RAILWAY_SERVICE_ID") == service)
    require(bool(env.get("RAILWAY_DEPLOYMENT_ID")))
    require(env.get("ACTIVEKG_SCHEMA_ENVIRONMENT") == "production")
    require(not env.get("ACTIVEKG_MIGRATION_APPLY"))
    require(not env.get("ACTIVEKG_SCHEMA_FRESH_INIT"))
    require(env.get("NODE_TLS_REJECT_UNAUTHORIZED") != "0")
    require(not env.get("NODE_OPTIONS") and not env.get("PGSERVICE"))
    for key in ("PGHOSTADDR", "PGOPTIONS", "PGSERVICEFILE", "PGSSLMODE"):
        require(not env.get(key))
    return private_dsn(env.get("ACTIVEKG_MIGRATE_DSN", ""), env.get("SPARK_MEMORY_PRIVATE_HOST"))


def consent(cur, register=False):
    if register:
        cur.execute("LOCK TABLE public.candidate_consent_copy_catalog IN SHARE ROW EXCLUSIVE MODE")
        cur.execute(
            "SELECT pg_get_userbyid(relowner)=current_user FROM pg_class "
            "WHERE oid='public.candidate_consent_copy_catalog'::regclass"
        )
        require(cur.fetchone() == (True,))
    cur.execute(
        "SELECT copy_sha256,approved FROM public.candidate_consent_copy_catalog WHERE copy_version=2"
    )
    row = cur.fetchone()
    if row is None and register:
        cur.execute(
            "INSERT INTO public.candidate_consent_copy_catalog(copy_version,copy_sha256,approved) "
            "VALUES(2,%s,true)",
            (COPY_HASH,),
        )
        cur.execute(
            "SELECT copy_sha256,approved FROM public.candidate_consent_copy_catalog WHERE copy_version=2"
        )
        row = cur.fetchone()
    require(row == (COPY_HASH, True))


def permissions(cur, role):
    from activekg.candidate_profiles.profile_repository import profile_catalog_ready
    from activekg.candidate_profiles.publication import (
        NETWORK_WORKER_FUNCTIONS,
        NETWORK_WORKER_ROLE,
        network_catalog_ready,
    )
    from activekg.candidate_profiles.repository import catalog_ready

    require(role == "activekg_app")
    for selected in (role, NETWORK_WORKER_ROLE):
        cur.execute(
            "SELECT rolcanlogin,rolsuper,rolbypassrls,rolcreatedb,rolcreaterole,rolreplication "
            "FROM pg_roles WHERE rolname=%s",
            (selected,),
        )
        require(cur.fetchone() == (True, False, False, False, False, False))
    require(catalog_ready(cur, role) and profile_catalog_ready(cur, role))
    require(network_catalog_ready(cur, role))
    cur.execute(
        "SELECT rolinherit,EXISTS(SELECT 1 FROM pg_auth_members WHERE member=r.oid) "
        "FROM pg_roles r WHERE rolname=%s",
        (NETWORK_WORKER_ROLE,),
    )
    require(cur.fetchone() == (False, False))
    cur.execute(
        "SELECT p.oid::regprocedure::text FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname='public' AND has_function_privilege(%s,p.oid,'EXECUTE') "
        "AND p.proname LIKE 'candidate_%%'",
        (NETWORK_WORKER_ROLE,),
    )
    require({r[0] for r in cur.fetchall()} == set(NETWORK_WORKER_FUNCTIONS))
    cur.execute(
        "SELECT p.oid::regprocedure::text,a.is_grantable FROM pg_proc p "
        "JOIN pg_namespace n ON n.oid=p.pronamespace "
        "CROSS JOIN LATERAL aclexplode(p.proacl) a "
        "WHERE a.grantee=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        (NETWORK_WORKER_ROLE,),
    )
    require(set(cur.fetchall()) == {(name, False) for name in NETWORK_WORKER_FUNCTIONS})
    cur.execute(
        "SELECT EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' AND c.relkind IN ('r','p','v','m','f') AND "
        "(has_table_privilege(%s,c.oid,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER') "
        "OR has_any_column_privilege(%s,c.oid,'SELECT,INSERT,UPDATE,REFERENCES'))) ",
        (NETWORK_WORKER_ROLE, NETWORK_WORKER_ROLE),
    )
    require(cur.fetchone() == (False,))
    cur.execute(
        "SELECT has_schema_privilege(%s,'public','CREATE'), "
        "EXISTS(SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' AND CASE WHEN c.relkind='S' "
        "THEN has_sequence_privilege(%s,c.oid,'USAGE,SELECT,UPDATE') ELSE false END)",
        (NETWORK_WORKER_ROLE, NETWORK_WORKER_ROLE),
    )
    require(cur.fetchone() == (False, False))


def database_job(mode, config, connection, phase="post"):
    require(phase in {"pre", "post"})
    with connection.transaction(), connection.cursor() as cur:
        cur.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        if mode == "readback":
            cur.execute("SET TRANSACTION READ ONLY")
        cur.execute("SET LOCAL lock_timeout='3s'")
        cur.execute("SET LOCAL statement_timeout='15s'")
        assert_identity(cur, config.target_id, config.environment)
        records = load_migration_records()
        ledger = read_ledger(cur)
        assert_ledger(ledger, records, allow_prefix=mode == "readback" and phase == "pre")
        if mode == "consent-register":
            consent(cur, register=True)
        elif phase == "post":
            permissions(cur, os.getenv("ACTIVEKG_RUNTIME_ROLE", "activekg_app"))
            consent(cur)
    # Latest name comes from our manifest, never arbitrary DB rows.
    return len(ledger), records[len(ledger) - 1].filename if ledger else "none"


def main():
    try:
        require(len(sys.argv) == 2)
        mode = sys.argv[1]
        settings = preflight(mode, os.environ)
        config = resolve_control_environment()
        if mode == "first-launch":
            from scripts.spark_first_launch import first_launch, flow_connection, launch_input

            launch_input(os.environ)
            with flow_connection(os.environ) as flow, psycopg.connect(**settings) as memory:
                first_launch(memory, flow, config, os.environ)
            print("SPARK_PRIVATE_FIRST_LAUNCH PASS flow=0 memory=0")
        elif mode == "recovery":
            # Helper image alone contains the pinned Flow source; no download/exec from the network.
            result = subprocess.run(
                ["node", "--import", "tsx", "scripts/spark-private-recovery.ts"],
                cwd="/flow",
                capture_output=True,
                timeout=900,
                check=False,
            )
            lines = result.stdout.decode("utf-8", errors="replace").splitlines()
            require(result.returncode == 0 and len(lines) == 1)
            require(
                re.fullmatch(
                    r"SPARK_PRIVATE_RECOVERY PASS subjects=\d+ replayed=\d+ held=\d+", lines[0]
                )
            )
            print(lines[0])
        else:
            with psycopg.connect(**settings) as conn:
                count, latest = database_job(
                    mode, config, conn, os.getenv("SPARK_PRIVATE_JOB_READBACK_PHASE", "post")
                )
            phase = os.getenv("SPARK_PRIVATE_JOB_READBACK_PHASE", "post")
            print(
                f"SPARK_PRIVATE_JOB PASS mode={mode} phase={phase} ledger={count} latest={latest}"
            )
        return 0
    except Exception:
        # Never echo exception text, driver errors, SQL, subprocess logs, rows or DSNs.
        print("SPARK_PRIVATE_JOB FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
