"""Copy of rehearse_spark_browser.py for the backup/restore rehearsal (Claude, 4 Nov 2026): adds
S2_MEMORY_EXISTING_DB / S2_MEMORY_EXISTING_RUNTIME to reopen a restored disposable database."""
"""Opt-in local browser fixture. Real product routers/auth/SQL, no paid services.

Checks a disposable schema reference, then provisions a fresh empty database.
Keeps that disposable database for review.
No production credentials, deployment, or destructive cleanup.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    if os.environ.get("S2_BROWSER_DISPOSABLE") != "1":
        raise RuntimeError("disposable_only")
    template = os.environ["S2_MEMORY_TEMPLATE"]
    parsed = urlparse(template)
    if (
        parsed.hostname != "127.0.0.1"
        or not parsed.path.startswith("/s2_memory_")
        or not parsed.path.endswith("_test")
    ):
        raise RuntimeError("disposable_template_required")
    from activekg.common.migration_manifest import MIGRATIONS

    with psycopg.connect(template) as conn:
        files = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}
        if files != set(MIGRATIONS):
            raise RuntimeError("template_manifest_mismatch")
    existing = os.environ.get("S2_MEMORY_EXISTING_DB")
    if existing:
        # Restore rehearsal: reopen an existing (restored) disposable database without provisioning.
        name, runtime = existing, os.environ["S2_MEMORY_EXISTING_RUNTIME"]
        if not (name.startswith("spark_browser_") and name.endswith("_test") and runtime.startswith("s2_runtime_") and runtime.endswith("_test")):
            raise RuntimeError("disposable_existing_required")
    else:
        suffix = uuid4().hex[:10]
        owner, runtime = "s2_owner_" + suffix + "_test", "s2_runtime_" + suffix + "_test"
        name = "spark_browser_" + suffix + "_test"
        with psycopg.connect(make_conninfo(template, dbname="postgres"), autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS CREATEROLE").format(sql.Identifier(owner)))
            conn.execute(sql.SQL("GRANT activekg_network_worker TO {} WITH ADMIN OPTION").format(sql.Identifier(owner)))
            conn.execute(
                sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'").format(
                    sql.Identifier(name), sql.Identifier(owner)
                )
            )
        with psycopg.connect(make_conninfo(template, dbname=name), autocommit=True) as conn:
            conn.execute("CREATE EXTENSION vector")
            conn.execute("CREATE EXTENSION pgcrypto")
        migration_env = {**os.environ, "ACTIVEKG_MIGRATE_DSN": make_conninfo(template, user=owner, dbname=name),
            "ACTIVEKG_MIGRATION_APPLY": "1", "ACTIVEKG_SCHEMA_TARGET_ID": str(uuid4()),
            "ACTIVEKG_SCHEMA_ENVIRONMENT": "development", "ACTIVEKG_SCHEMA_SOURCE_COMMIT": "0" * 40,
            "ACTIVEKG_SCHEMA_FRESH_INIT": "1", "ACTIVEKG_RUNTIME_ROLE": runtime,
            "ACTIVEKG_RUNTIME_PASSWORD": "synthetic-local-only", "ACTIVEKG_NETWORK_WORKER_PASSWORD": "synthetic-worker-only"}
        release = subprocess.run([sys.executable, "scripts/init_railway_db.py"], env=migration_env,
            capture_output=True, text=True, timeout=120)
        if release.returncode:
            raise RuntimeError("local_release_failed: " + release.stdout + release.stderr)
        with psycopg.connect(make_conninfo(template, dbname=name)) as conn:
            conn.execute(
                "INSERT INTO candidate_consent_copy_catalog VALUES(2,%s,true) ON CONFLICT(copy_version) DO UPDATE SET approved=true,copy_sha256=excluded.copy_sha256",
                (os.environ["S2_CONSENT_COPY_HASH"],),
            )
    os.environ["ACTIVEKG_DSN"] = make_conninfo(template, user=runtime, dbname=name)
    # Block accidental external networking; actual PostgreSQL and HTTP are local.
    original_connect = socket.socket.connect

    def local_connect(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise RuntimeError("rehearsal_external_network_refused")
        return original_connect(sock, address)

    socket.socket.connect = local_connect
    import uvicorn
    from fastapi import FastAPI

    from activekg.api import candidate_privacy, candidate_profiles, global_memory
    from activekg.privacy.config import load_candidate_privacy_config
    from activekg.privacy.repository import CandidatePrivacyRepository

    repo = CandidatePrivacyRepository(
        os.environ["ACTIVEKG_DSN"], config=load_candidate_privacy_config(require_hmac=True)
    )
    candidate_privacy.set_repository(repo)
    app = FastAPI()
    for router in (candidate_profiles.router, candidate_privacy.router, global_memory.router):
        app.include_router(router)

    @app.get("/_rehearsal/ready")
    def ready():
        return {"ready": True, "fixture": name}

    print("REHEARSAL_MEMORY", name, runtime, flush=True)
    try:
        uvicorn.run(
            app,
            ssl_certfile=os.environ["S2_MEMORY_TLS_CERT"],
            ssl_keyfile=os.environ["S2_MEMORY_TLS_KEY"],
            host="127.0.0.1",
            port=int(os.environ["S2_MEMORY_PORT"]),
            log_level="warning",
            access_log=False,
        )
    finally:
        repo.close()


if __name__ == "__main__":
    main()
