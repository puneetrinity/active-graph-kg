"""Real foundation DDL and ACL tests. Separate from full-baseline/release-path tests."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from activekg.candidate_profiles.contracts import RUNTIME_FUNCTIONS, TABLES
from activekg.candidate_profiles.repository import catalog_evidence

ADMIN = os.getenv("CANDIDATE_FOUNDATION_TEST_ADMIN_DSN")
pytestmark = pytest.mark.skipif(not ADMIN, reason="isolated foundation database not configured")


@pytest.fixture(scope="module")
def database():
    info = conninfo_to_dict(ADMIN)
    assert info.get("host", "").startswith("/tmp/candidate-s1-")
    assert info.get("dbname") == "postgres" and info.get("user") == "ews"
    name = "s1_foundation_" + uuid4().hex[:12]
    owner = "s1_owner"
    runtime = "s1_runtime"
    with psycopg.connect(ADMIN, autocommit=True) as admin:
        for role in (owner, runtime):
            if not admin.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,)).fetchone():
                admin.execute(
                    sql.SQL("CREATE ROLE {} LOGIN NOSUPERUSER NOBYPASSRLS").format(
                        sql.Identifier(role)
                    )
                )
        admin.execute(
            sql.SQL("CREATE DATABASE {} OWNER {} TEMPLATE template0 ENCODING 'UTF8'").format(
                sql.Identifier(name), sql.Identifier(owner)
            )
        )
    owner_dsn = make_conninfo(ADMIN, user=owner, dbname=name)
    runtime_dsn = make_conninfo(ADMIN, user=runtime, dbname=name)
    with psycopg.connect(owner_dsn) as conn:
        conn.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES GRANT ALL ON TABLES TO {}").format(
                sql.Identifier(runtime)
            )
        )
        conn.execute(
            sql.SQL("ALTER DEFAULT PRIVILEGES GRANT EXECUTE ON FUNCTIONS TO {}").format(
                sql.Identifier(runtime)
            )
        )
        conn.execute(
            (
                Path(__file__).parents[1] / "db/migrations/040_candidate_owned_foundation.sql"
            ).read_text()
        )
    with psycopg.connect(runtime_dsn) as conn:
        for signature in RUNTIME_FUNCTIONS:
            assert conn.execute(
                "SELECT has_function_privilege(current_user,%s,'EXECUTE')", (signature,)
            ).fetchone() == (False,)
    with psycopg.connect(owner_dsn) as conn:
        for signature in RUNTIME_FUNCTIONS:
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(
                    sql.SQL(signature), sql.Identifier(runtime)
                )
            )
        with conn.cursor() as cur:
            digest, allowed = catalog_evidence(cur, runtime)
            assert allowed is True
            print("FOUNDATION_CATALOG_MD5", digest, "RETAINED_DATABASE", name)
    yield owner_dsn, runtime_dsn
    # Retain exact fixture/evidence for review; no destructive cleanup.


def ensure(conn, subject, binding, command, sequence=1, digest="a" * 64):
    return conn.execute(
        "SELECT candidate_owned_ensure(%s,%s,%s,%s,%s)",
        (subject, binding, command, sequence, digest),
    ).fetchone()[0]


def test_exact_replay_gap_and_delayed_command(database):
    _, dsn = database
    subject, binding, command, later = [uuid4() for _ in range(4)]
    with psycopg.connect(dsn, autocommit=True) as conn:
        first = ensure(conn, subject, binding, command)
        assert first.pop("_replayed") is False
        again = ensure(conn, subject, binding, command)
        assert again.pop("_replayed") is True and again == first
        assert ensure(conn, subject, binding, later, 3)["sequence"] == 3
        assert ensure(conn, subject, binding, command)["_replayed"] is True
        with pytest.raises(psycopg.errors.UniqueViolation, match="stale_sequence"):
            ensure(conn, subject, binding, uuid4(), 2)
        assert first["publicationActive"] is False and first["profileVersion"] == 0


def test_binding_command_and_hash_conflicts(database):
    _, dsn = database
    subject, binding, command = [uuid4() for _ in range(3)]
    with psycopg.connect(dsn, autocommit=True) as conn:
        ensure(conn, subject, binding, command)
        for args in (
            (subject, uuid4(), uuid4()),
            (uuid4(), binding, uuid4()),
            (subject, binding, command, 1, "b" * 64),
        ):
            with pytest.raises(psycopg.errors.UniqueViolation):
                ensure(conn, *args)
        with pytest.raises(psycopg.errors.NoDataFound):
            conn.execute("SELECT candidate_owned_status(%s,%s)", (subject, uuid4()))


def test_old_consecutive_rule_breaks_valid_successor_and_is_rolled_back(database):
    owner, _ = database
    with psycopg.connect(owner) as conn:
        original = conn.execute(
            "SELECT pg_get_functiondef('candidate_owned_ensure(uuid,uuid,uuid,bigint,text)'::regprocedure)"
        ).fetchone()[0]
        assert original.count("IF p_sequence<=maximum") == 1
        try:
            conn.execute(original.replace("IF p_sequence<=maximum", "IF p_sequence<>maximum+1"))
            with pytest.raises(psycopg.errors.UniqueViolation, match="stale_sequence"):
                ensure(conn, uuid4(), uuid4(), uuid4(), 2)
        finally:
            conn.rollback()
        assert (
            conn.execute(
                "SELECT pg_get_functiondef('candidate_owned_ensure(uuid,uuid,uuid,bigint,text)'::regprocedure)"
            ).fetchone()[0]
            == original
        )


def test_runtime_cannot_read_or_mutate_tables_or_become_owner(database):
    _, dsn = database
    with psycopg.connect(dsn, autocommit=True) as conn:
        for table in TABLES:
            for query in (f"SELECT * FROM {table}", f"DELETE FROM {table}", f"TRUNCATE {table}"):
                with pytest.raises(psycopg.errors.InsufficientPrivilege):
                    conn.execute(query)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute("SET ROLE s1_owner")


def test_catalog_detects_column_grant_and_constraint_mutation(database):
    owner, _ = database
    with psycopg.connect(owner) as conn:
        with conn.cursor() as cur:
            original, allowed = catalog_evidence(cur, "s1_runtime")
            assert allowed
            cur.execute("GRANT SELECT(subject_id) ON candidate_owned_subjects TO s1_runtime")
            changed, allowed = catalog_evidence(cur, "s1_runtime")
            assert changed == original and not allowed
            conn.rollback()
            cur.execute(
                "ALTER TABLE candidate_owned_subjects DROP CONSTRAINT candidate_owned_subjects_state_check"
            )
            changed, _ = catalog_evidence(cur, "s1_runtime")
            assert changed != original
            conn.rollback()


@pytest.mark.parametrize("upgrade", [False, True])
def test_full_release_path_and_foundation_receiver(tmp_path, upgrade):
    """Fresh complete schema through the actual release runner, restricted API role."""
    import json

    from activekg.candidate_profiles.contracts import parse_command
    from activekg.candidate_profiles.repository import execute

    info = conninfo_to_dict(ADMIN)
    assert (
        info.get("host", "").startswith("/tmp/candidate-s1-") and info.get("dbname") == "postgres"
    )
    suffix = uuid4().hex[:10]
    owner, runtime, name = (
        f"s1_owner_{suffix}_test",
        f"s1_runtime_{suffix}_test",
        f"s1_memory_{suffix}_test",
    )
    with psycopg.connect(ADMIN, autocommit=True) as admin:
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
    with psycopg.connect(make_conninfo(ADMIN, dbname=name), autocommit=True) as admin:
        admin.execute("CREATE EXTENSION vector")
        admin.execute("CREATE EXTENSION pgcrypto")
    owner_dsn = make_conninfo(ADMIN, user=owner, dbname=name)
    runtime_dsn = make_conninfo(ADMIN, user=runtime, dbname=name)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACTIVEKG_")}
    env.update(
        ACTIVEKG_MIGRATE_DSN=owner_dsn,
        ACTIVEKG_MIGRATION_APPLY="1",
        ACTIVEKG_SCHEMA_TARGET_ID=str(uuid4()),
        ACTIVEKG_SCHEMA_ENVIRONMENT="development",
        ACTIVEKG_SCHEMA_SOURCE_COMMIT="0" * 40,
        ACTIVEKG_SCHEMA_FRESH_INIT="1",
        ACTIVEKG_RUNTIME_ROLE=runtime,
        ACTIVEKG_RUNTIME_PASSWORD="synthetic-disposable-only",
    )
    root = Path(__file__).parents[1]
    if upgrade:
        previous = tmp_path / "before040"
        for directory in ("scripts", "activekg", "db"):
            shutil.copytree(
                root / directory,
                previous / directory,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        manifest = previous / "activekg/common/migration_manifest.py"
        manifest.write_text(
            manifest.read_text()
            .replace('    "040_candidate_owned_foundation.sql",\n', "")
            .replace("len(MIGRATIONS) != 40", "len(MIGRATIONS) != 39")
            .replace("contain 40 unique ordered entries", "contain 39 unique ordered entries")
        )
        baseline = subprocess.run(
            [sys.executable, "scripts/init_railway_db.py"],
            cwd=previous,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert baseline.returncode == 0, baseline.stdout + baseline.stderr
        with psycopg.connect(owner_dsn) as conn:
            assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone() == (39,)
        env.pop("ACTIVEKG_SCHEMA_FRESH_INIT")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy\ntry:\n runpy.run_path('scripts/init_railway_db.py',run_name='__main__')\nexcept SystemExit as e:\n if e.__cause__: raise e.__cause__\n raise",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    print("FULL_RELEASE_FIXTURE", name, "EXIT", result.returncode)
    assert result.returncode == 0, result.stdout + result.stderr
    env.pop("ACTIVEKG_SCHEMA_FRESH_INIT", None)
    replay = subprocess.run(
        [sys.executable, "scripts/init_railway_db.py"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert replay.returncode == 0, replay.stdout + replay.stderr
    with psycopg.connect(owner_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone() == (40,)
    command = {
        "schemaVersion": 1,
        "commandId": str(uuid4()),
        "subjectId": str(uuid4()),
        "bindingId": str(uuid4()),
        "sequence": 1,
        "action": "ensure_subject",
    }
    command, digest = parse_command(json.dumps(command, separators=(",", ":")).encode())
    from unittest.mock import patch

    with patch.dict(os.environ, {"ACTIVEKG_DSN": runtime_dsn}):
        first, replayed = execute(command["subjectId"], command["bindingId"], command, digest)
        assert replayed is False and first["publicationActive"] is False
        second, replayed = execute(command["subjectId"], command["bindingId"], command, digest)
        assert replayed is True and second == first
