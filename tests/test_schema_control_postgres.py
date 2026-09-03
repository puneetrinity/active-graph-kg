from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

OWNER_DSN = os.getenv("ACTIVEKG_SCHEMA_CONTROL_TEST_OWNER_DSN")
RUNTIME_DSN = os.getenv("ACTIVEKG_SCHEMA_CONTROL_TEST_RUNTIME_DSN")
ROOT = Path(__file__).resolve().parents[1]
TARGET_ID = "11111111-1111-4111-8111-111111111111"
SOURCE_COMMIT = "0" * 40
PRIVACY_READINESS_ENV = {
    "JWT_ISSUER": "flow-test",
    "SIGNAL_JWT_ISSUER": "signal-test",
    "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "1",
    "CANDIDATE_PRIVACY_HMAC_KEY_V1": ("AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="),
    "CANDIDATE_PRIVACY_INTAKE_ENABLED": "false",
    "CANDIDATE_PRIVACY_FLOW_ISSUER": "flow-test",
    "CANDIDATE_PRIVACY_FLOW_ACTOR_ID": "flow-service-test",
    "CANDIDATE_PRIVACY_SIGNAL_ISSUER": "signal-test",
    "CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID": "signal-service-test",
}

pytestmark = pytest.mark.skipif(
    not OWNER_DSN or not RUNTIME_DSN,
    reason="disposable Memory schema-control DSNs are not configured",
)


def _dsn(database: str) -> str:
    return make_conninfo(OWNER_DSN, dbname=database)


def _runtime_dsn(database: str) -> str:
    return make_conninfo(RUNTIME_DSN, dbname=database)


def _runtime_password() -> str:
    assert RUNTIME_DSN is not None
    password = conninfo_to_dict(RUNTIME_DSN).get("password")
    assert isinstance(password, str) and password, (
        "the disposable runtime DSN must contain a password"
    )
    return password


def _run(script: Path, dsn: str, **extra: str) -> subprocess.CompletedProcess[str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("ACTIVEKG_")}
    env.update(
        {
            "ACTIVEKG_MIGRATE_DSN": dsn,
            "ACTIVEKG_SCHEMA_TARGET_ID": extra.pop("target_id", TARGET_ID),
            "ACTIVEKG_SCHEMA_ENVIRONMENT": "development",
            "ACTIVEKG_SCHEMA_SOURCE_COMMIT": SOURCE_COMMIT,
            "ACTIVEKG_RUNTIME_ROLE": "activekg_app",
            **extra,
        }
    )
    return subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


def _maintenance() -> psycopg.Connection:
    return psycopg.connect(make_conninfo(OWNER_DSN, dbname="postgres"), autocommit=True)


def _drop_database(name: str) -> None:
    with _maintenance() as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name)))


def _clone_database(name: str) -> str:
    source = conninfo_to_dict(OWNER_DSN)["dbname"]
    _drop_database(name)
    with _maintenance() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(
                sql.Identifier(name), sql.Identifier(source)
            )
        )
    return _dsn(name)


def _copy_with_tail_migration(
    tmp_path: Path,
    migration_name: str,
    migration_sql: str,
    verifier: str,
) -> Path:
    copied = tmp_path / "repository"
    shutil.copytree(
        ROOT,
        copied,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    (copied / "db/migrations" / migration_name).write_text(migration_sql)
    manifest = copied / "activekg/common/migration_manifest.py"
    content = manifest.read_text()
    content = content.replace(
        '    "024_organization_decision_event_inbox.sql",\n)',
        f'    "024_organization_decision_event_inbox.sql",\n    "{migration_name}",\n)',
    ).replace("len(MIGRATIONS) != 24", "len(MIGRATIONS) != 25")
    manifest.write_text(content)

    runner = copied / "scripts/init_railway_db.py"
    content = runner.read_text()
    marker = "\n\ndef _normalize_sql_definition"
    content = content.replace(
        marker,
        f'\n\nBASELINE_VERIFIERS["{migration_name}"] = {verifier}\n' + marker,
        1,
    )
    runner.write_text(content)
    return copied


def test_disposable_identity_is_genuinely_local_and_allowlisted() -> None:
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute("SELECT current_database(), current_user, host(inet_server_addr())")
        database, role, host = cur.fetchone()
    assert database.endswith("_test") or database.endswith("_test_ci")
    assert role.endswith("_test")
    assert host in {None, "127.0.0.1", "::1"}

    with psycopg.connect(OWNER_DSN, autocommit=True) as conn, conn.cursor() as cur:
        with pytest.raises(psycopg.Error):
            cur.execute("UPDATE activekg_schema_control.target_identity SET environment='staging'")
        with pytest.raises(psycopg.Error):
            cur.execute("DELETE FROM activekg_schema_control.release_attempts")


def test_existing_target_adoption_is_control_only_and_second_run_refuses() -> None:
    name = "memory_schema_adoption_test"
    dsn = _clone_database(name)
    try:
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("DROP SCHEMA activekg_schema_control CASCADE")
            cur.execute(
                "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname='public' AND c.relkind IN ('r','p')"
            )
            before_relations = cur.fetchone()[0]
            cur.execute("SELECT filename, checksum, baselined FROM schema_migrations")
            before_ledger = cur.fetchall()

        result = _run(
            ROOT / "scripts/adopt_schema_control.py",
            dsn,
            ACTIVEKG_SCHEMA_ADOPT_EXISTING="1",
            target_id="22222222-2222-4222-8222-222222222222",
        )
        assert result.returncode == 0, result.stdout + result.stderr
        second = _run(
            ROOT / "scripts/adopt_schema_control.py",
            dsn,
            ACTIVEKG_SCHEMA_ADOPT_EXISTING="1",
            target_id="22222222-2222-4222-8222-222222222222",
        )
        assert second.returncode == 1

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname='public' AND c.relkind IN ('r','p')"
            )
            assert cur.fetchone()[0] == before_relations
            cur.execute("SELECT filename, checksum, baselined FROM schema_migrations")
            assert cur.fetchall() == before_ledger
            cur.execute(
                "SELECT (SELECT count(*) FROM activekg_schema_control.target_identity), "
                "(SELECT count(*) FROM activekg_schema_control.release_attempts), "
                "(SELECT count(*) FROM activekg_schema_control.release_attempts "
                " WHERE outcome <> 'success' OR finished_at IS NULL)"
            )
            assert cur.fetchone() == (1, 1, 0)
    finally:
        _drop_database(name)


def test_wrong_target_and_missing_identity_refuse_without_writes() -> None:
    name = "memory_schema_wrong_target_test"
    _drop_database(name)
    with _maintenance() as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    dsn = _dsn(name)
    try:
        adoption = _run(
            ROOT / "scripts/adopt_schema_control.py",
            dsn,
            ACTIVEKG_SCHEMA_ADOPT_EXISTING="1",
        )
        release = _run(
            ROOT / "scripts/init_railway_db.py",
            dsn,
            ACTIVEKG_MIGRATION_APPLY="1",
        )
        assert adoption.returncode == release.returncode == 1
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regnamespace('activekg_schema_control')")
            assert cur.fetchone()[0] is None
            cur.execute(
                "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                "WHERE n.nspname NOT IN ('pg_catalog','information_schema') "
                "AND n.nspname NOT LIKE 'pg_toast%' AND c.relkind IN ('r','p','v','m','S','f')"
            )
            assert cur.fetchone()[0] == 0
    finally:
        _drop_database(name)


def test_checksum_corruption_refuses_before_attempt_or_schema_write() -> None:
    name = "memory_schema_checksum_test"
    dsn = _clone_database(name)
    try:
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM activekg_schema_control.release_attempts")
            before_attempts = cur.fetchone()[0]
            cur.execute(
                "UPDATE schema_migrations SET checksum=%s WHERE filename=%s",
                ("0" * 64, "001_add_embedding_history_index.sql"),
            )
        refused = _run(
            ROOT / "scripts/init_railway_db.py",
            dsn,
            ACTIVEKG_MIGRATION_APPLY="1",
        )
        assert refused.returncode == 1
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM activekg_schema_control.release_attempts")
            assert cur.fetchone()[0] == before_attempts
            cur.execute(
                "SELECT checksum FROM schema_migrations WHERE filename=%s",
                ("001_add_embedding_history_index.sql",),
            )
            assert cur.fetchone()[0] == "0" * 64
    finally:
        _drop_database(name)


def test_partial_existing_target_refuses_adoption_without_control_write() -> None:
    name = "memory_schema_partial_target_test"
    dsn = _clone_database(name)
    try:
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute("DROP SCHEMA activekg_schema_control CASCADE")
            cur.execute("DROP INDEX idx_global_candidates_embed_version")
        refused = _run(
            ROOT / "scripts/adopt_schema_control.py",
            dsn,
            ACTIVEKG_SCHEMA_ADOPT_EXISTING="1",
            target_id="33333333-3333-4333-8333-333333333333",
        )
        assert refused.returncode == 1
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regnamespace('activekg_schema_control')")
            assert cur.fetchone()[0] is None
            cur.execute("SELECT to_regclass('public.idx_global_candidates_embed_version')")
            assert cur.fetchone()[0] is None
            cur.execute("SELECT count(*) FROM schema_migrations")
            assert cur.fetchone()[0] == 24
    finally:
        _drop_database(name)


def _assert_candidate_privacy_runtime_privileges(cur: psycopg.Cursor) -> None:
    for relation in (
        "candidate_privacy_directive_events",
        "candidate_privacy_directives",
    ):
        cur.execute(
            "SELECT has_table_privilege('activekg_app',%s,'SELECT'), "
            "has_table_privilege('activekg_app',%s,'INSERT'), "
            "has_table_privilege('activekg_app',%s,'UPDATE'), "
            "has_table_privilege('activekg_app',%s,'DELETE'), "
            "has_table_privilege('activekg_app',%s,'TRUNCATE')",
            tuple(f"public.{relation}" for _ in range(5)),
        )
        assert cur.fetchone() == (True, False, False, False, False)
    cur.execute(
        "SELECT has_table_privilege('activekg_app',"
        "'public.candidate_privacy_identity_tokens','SELECT'), "
        "has_table_privilege('activekg_app',"
        "'public.candidate_privacy_identity_tokens','INSERT'), "
        "has_table_privilege('activekg_app',"
        "'public.candidate_privacy_identity_tokens','UPDATE'), "
        "has_table_privilege('activekg_app',"
        "'public.candidate_privacy_identity_tokens','DELETE'), "
        "has_sequence_privilege('activekg_app',"
        "'public.candidate_privacy_directive_events_cursor_seq','USAGE'), "
        "has_sequence_privilege('activekg_app',"
        "'public.candidate_privacy_directive_events_cursor_seq','SELECT'), "
        "has_sequence_privilege('activekg_app',"
        "'public.candidate_privacy_directive_events_cursor_seq','UPDATE'), "
        "has_function_privilege('activekg_app',"
        "'public.candidate_privacy_create_directive(uuid,uuid,text,text,text,uuid,text,text,"
        "text,uuid,text,uuid,integer,jsonb,boolean,timestamp with time zone)','EXECUTE'), "
        "has_function_privilege('activekg_app',"
        "'public.candidate_privacy_append_only()','EXECUTE')"
    )
    assert cur.fetchone() == (False, False, False, False, False, False, False, True, False)


def test_every_release_reasserts_candidate_privacy_runtime_privileges() -> None:
    name = "memory_schema_privacy_privileges_test"
    dsn = _clone_database(name)
    try:
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            _assert_candidate_privacy_runtime_privileges(cur)
            cur.execute(
                "GRANT ALL ON candidate_privacy_directive_events, "
                "candidate_privacy_directives, candidate_privacy_identity_tokens "
                "TO activekg_app"
            )
            cur.execute(
                "GRANT ALL ON SEQUENCE candidate_privacy_directive_events_cursor_seq "
                "TO activekg_app"
            )
            cur.execute("GRANT EXECUTE ON FUNCTION candidate_privacy_append_only() TO activekg_app")

        first = _run(
            ROOT / "scripts/init_railway_db.py",
            dsn,
            ACTIVEKG_MIGRATION_APPLY="1",
        )
        assert first.returncode == 0, first.stdout + first.stderr
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            _assert_candidate_privacy_runtime_privileges(cur)

        second = _run(
            ROOT / "scripts/init_railway_db.py",
            dsn,
            ACTIVEKG_MIGRATION_APPLY="1",
        )
        assert second.returncode == 0, second.stdout + second.stderr
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            _assert_candidate_privacy_runtime_privileges(cur)
    finally:
        _drop_database(name)


def test_existing_23_migration_target_upgrades_to_024_without_product_mutation(
    tmp_path: Path,
) -> None:
    name = "memory_schema_privacy_upgrade_test"
    _drop_database(name)
    with _maintenance() as conn, conn.cursor() as cur:
        cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    dsn = _dsn(name)
    legacy = tmp_path / "legacy-23"
    shutil.copytree(
        ROOT,
        legacy,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    manifest = legacy / "activekg/common/migration_manifest.py"
    manifest.write_text(
        manifest.read_text()
        .replace('    "024_organization_decision_event_inbox.sql",\n', "")
        .replace("len(MIGRATIONS) != 24", "len(MIGRATIONS) != 23")
    )
    runner = legacy / "scripts/init_railway_db.py"
    runner.write_text(
        runner.read_text()
        .replace(
            "                _harden_decision_inbox_runtime_privileges(cur, runtime_role)\n",
            "",
        )
        .replace(
            "                _assert_decision_inbox_runtime_privileges(cur, runtime_role)\n",
            "",
        )
    )
    try:
        legacy_install = _run(
            runner,
            dsn,
            ACTIVEKG_MIGRATION_APPLY="1",
            ACTIVEKG_SCHEMA_FRESH_INIT="1",
            # PostgreSQL roles are cluster-wide. Reassert the password already
            # carried by the shared disposable runtime DSN so this cloned-DB
            # test cannot poison the tests that follow it.
            ACTIVEKG_RUNTIME_PASSWORD=_runtime_password(),
        )
        assert legacy_install.returncode == 0, legacy_install.stdout + legacy_install.stderr
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO nodes (id, tenant_id, classes, props) "
                "VALUES ('51515151-5151-4515-8515-515151515151',"
                "'privacy-upgrade-test',ARRAY['Document'],"
                '\'{"text":"unchanged sentinel"}\'::jsonb)'
            )
            cur.execute("SELECT count(*), min(props->>'text') FROM nodes")
            before_nodes = cur.fetchone()
            cur.execute("SELECT count(*) FROM schema_migrations")
            assert cur.fetchone()[0] == 23
            cur.execute("SELECT to_regclass('public.organization_decision_event_inbox')")
            assert cur.fetchone()[0] is None

        upgraded = _run(
            ROOT / "scripts/init_railway_db.py",
            dsn,
            ACTIVEKG_MIGRATION_APPLY="1",
        )
        assert upgraded.returncode == 0, upgraded.stdout + upgraded.stderr
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*), min(props->>'text') FROM nodes")
            assert cur.fetchone() == before_nodes
            cur.execute("SELECT count(*), count(*) FILTER (WHERE baselined) FROM schema_migrations")
            assert cur.fetchone() == (24, 0)
            _assert_candidate_privacy_runtime_privileges(cur)
            cur.execute(
                "SELECT has_table_privilege('activekg_app',"
                "'public.organization_decision_event_inbox','SELECT'),"
                "has_table_privilege('activekg_app',"
                "'public.organization_decision_event_inbox','INSERT'),"
                "has_table_privilege('activekg_app',"
                "'public.organization_decision_event_inbox','UPDATE'),"
                "has_table_privilege('activekg_app',"
                "'public.organization_decision_stream_state','UPDATE')"
            )
            assert cur.fetchone() == (True, True, False, True)
        with psycopg.connect(_runtime_dsn(name)) as conn, conn.cursor() as cur:
            cur.execute("SELECT current_user")
            assert cur.fetchone() == ("activekg_app",)
    finally:
        _drop_database(name)


def test_unfinished_attempt_blocks_runtime_readiness_without_recording_a_start() -> None:
    env = {
        **os.environ,
        **PRIVACY_READINESS_ENV,
        "ACTIVEKG_DSN": RUNTIME_DSN,
        "ACTIVEKG_SCHEMA_TARGET_ID": TARGET_ID,
        "ACTIVEKG_SCHEMA_ENVIRONMENT": "development",
    }
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM activekg_schema_control.release_attempts")
        before = cur.fetchone()[0]
    foreign = subprocess.run(
        [sys.executable, str(ROOT / "scripts/schema_ready.py")],
        cwd=ROOT,
        env={**env, "ACTIVEKG_SCHEMA_TARGET_ID": "44444444-4444-4444-8444-444444444444"},
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert foreign.returncode == 1
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO activekg_schema_control.release_attempts "
            "(kind, source_commit, manifest_digest, outcome) "
            "VALUES ('migration', %s, %s, 'running') RETURNING id",
            (SOURCE_COMMIT, "0" * 64),
        )
        attempt_id = cur.fetchone()[0]
    try:
        refused = subprocess.run(
            [sys.executable, str(ROOT / "scripts/schema_ready.py")],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert refused.returncode == 1
        with psycopg.connect(OWNER_DSN, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE activekg_schema_control.release_attempts "
                "SET outcome='success', finished_at=clock_timestamp() WHERE id=%s",
                (attempt_id,),
            )
        admitted = subprocess.run(
            [sys.executable, str(ROOT / "scripts/schema_ready.py")],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert admitted.returncode == 0, admitted.stdout + admitted.stderr
        with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM activekg_schema_control.release_attempts")
            assert cur.fetchone()[0] == before + 1
    finally:
        # The test finishes the attempt but deliberately does not delete it:
        # release history is append-only even for disposable proofs.
        with psycopg.connect(OWNER_DSN, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE activekg_schema_control.release_attempts "
                "SET outcome='success', finished_at=clock_timestamp() "
                "WHERE id=%s AND outcome='running'",
                (attempt_id,),
            )


def test_failed_tail_release_blocks_readiness_and_a_corrected_release_recovers(
    tmp_path: Path,
) -> None:
    name = "memory_schema_failure_test"
    dsn = _clone_database(name)
    migration_name = "025_schema_control_failure_test.sql"
    copied = _copy_with_tail_migration(
        tmp_path,
        migration_name,
        "SELECT 1 / 0;\n",
        '[("table", "schema_control_failure_probe")]',
    )
    try:
        failed = _run(
            copied / "scripts/init_railway_db.py",
            dsn,
            ACTIVEKG_MIGRATION_APPLY="1",
        )
        assert failed.returncode == 1
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT outcome, finished_at IS NOT NULL "
                "FROM activekg_schema_control.release_attempts ORDER BY id DESC LIMIT 1"
            )
            assert cur.fetchone() == ("failure", True)
            cur.execute(
                "SELECT count(*) FROM schema_migrations WHERE filename=%s", (migration_name,)
            )
            assert cur.fetchone()[0] == 0

        readiness_env = {
            key: value for key, value in os.environ.items() if not key.startswith("ACTIVEKG_")
        }
        readiness_env.update(
            {
                **PRIVACY_READINESS_ENV,
                "ACTIVEKG_DSN": _runtime_dsn(name),
                "ACTIVEKG_SCHEMA_TARGET_ID": TARGET_ID,
                "ACTIVEKG_SCHEMA_ENVIRONMENT": "development",
            }
        )
        refused = subprocess.run(
            [sys.executable, str(copied / "scripts/schema_ready.py")],
            cwd=copied,
            env=readiness_env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert refused.returncode == 1

        (copied / "db/migrations" / migration_name).write_text(
            "CREATE TABLE schema_control_failure_probe (id integer PRIMARY KEY);\n"
        )
        recovered = _run(
            copied / "scripts/init_railway_db.py",
            dsn,
            ACTIVEKG_MIGRATION_APPLY="1",
        )
        assert recovered.returncode == 0, recovered.stdout + recovered.stderr
        admitted = subprocess.run(
            [sys.executable, str(copied / "scripts/schema_ready.py")],
            cwd=copied,
            env=readiness_env,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert admitted.returncode == 0, admitted.stdout + admitted.stderr
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT outcome FROM activekg_schema_control.release_attempts "
                "ORDER BY id DESC LIMIT 2"
            )
            assert [row[0] for row in cur.fetchall()] == ["success", "failure"]
            cur.execute(
                "SELECT count(*) FROM schema_migrations WHERE filename=%s", (migration_name,)
            )
            assert cur.fetchone()[0] == 1
    finally:
        _drop_database(name)


def test_two_concurrent_tail_releases_apply_the_new_migration_exactly_once(
    tmp_path: Path,
) -> None:
    name = "memory_schema_tail_test"
    dsn = _clone_database(name)
    migration_name = "025_schema_control_test_tail.sql"
    copied = _copy_with_tail_migration(
        tmp_path,
        migration_name,
        "CREATE TABLE schema_control_tail_probe (id integer PRIMARY KEY);\n",
        '[("table", "schema_control_tail_probe")]',
    )
    try:
        runner = copied / "scripts/init_railway_db.py"
        env = {key: value for key, value in os.environ.items() if not key.startswith("ACTIVEKG_")}
        env.update(
            {
                "ACTIVEKG_MIGRATE_DSN": dsn,
                "ACTIVEKG_MIGRATION_APPLY": "1",
                "ACTIVEKG_SCHEMA_TARGET_ID": TARGET_ID,
                "ACTIVEKG_SCHEMA_ENVIRONMENT": "development",
                "ACTIVEKG_SCHEMA_SOURCE_COMMIT": SOURCE_COMMIT,
                "ACTIVEKG_RUNTIME_ROLE": "activekg_app",
            }
        )
        processes = [
            subprocess.Popen(
                [sys.executable, str(runner)],
                cwd=copied,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]
        results = [
            process.communicate(timeout=180) + (process.returncode,) for process in processes
        ]
        assert [result[2] for result in results] == [0, 0], results
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM schema_migrations WHERE filename=%s", (migration_name,)
            )
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT count(*) FROM schema_control_tail_probe")
            assert cur.fetchone()[0] == 0
    finally:
        _drop_database(name)
