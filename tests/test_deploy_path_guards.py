"""Regression tests for the deploy-path guards in scripts/init_railway_db.py.

Each test runs the real migration script as a subprocess against a live,
already-migrated database (the CI deploy-path job's), mutates the ledger or
schema to simulate a hazardous state, asserts the guard fires (or the safe
path succeeds), and restores the state it touched.

Covered: the PR#11→PR#12 checksum transition, unknown checksum drift,
reserved runtime-role rejection, partial-legacy baseline rejection (the
empirically found 006 case: duplicate-column error while the migration's
indexes are missing), and re-verification of previously baselined rows.

Gated on ``ACTIVEKG_RLS_TEST_OWNER_DSN`` (also used as the migrate DSN).
"""

import os
import subprocess
import sys

import psycopg
import pytest

OWNER_DSN = os.getenv("ACTIVEKG_RLS_TEST_OWNER_DSN")

pytestmark = pytest.mark.skipif(not OWNER_DSN, reason="ACTIVEKG_RLS_TEST_OWNER_DSN not configured")

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "init_railway_db.py")

PR11_016_CHECKSUM = "34f02ce7137003697e1a3e0a675883b5203d55150ea1a0c258892308ae344b21"


def _run_init(**extra_env: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACTIVEKG_")}
    env["ACTIVEKG_MIGRATE_DSN"] = OWNER_DSN
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, SCRIPT], env=env, capture_output=True, text=True, timeout=120
    )


def _sql(query: str, params: tuple = ()) -> list[tuple]:
    with psycopg.connect(OWNER_DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if cur.description:
                return cur.fetchall()
    return []


def test_known_checksum_transition_applies_in_place():
    (current,) = _sql(
        "SELECT checksum FROM schema_migrations WHERE filename = '016_candidate_rls.sql'"
    )[0]
    try:
        _sql(
            "UPDATE schema_migrations SET checksum = %s WHERE filename = '016_candidate_rls.sql'",
            (PR11_016_CHECKSUM,),
        )
        result = _run_init()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "known checksum transition applied" in result.stdout
        (after,) = _sql(
            "SELECT checksum FROM schema_migrations WHERE filename = '016_candidate_rls.sql'"
        )[0]
        assert after != PR11_016_CHECKSUM and after is not None
    finally:
        _sql(
            "UPDATE schema_migrations SET checksum = %s WHERE filename = '016_candidate_rls.sql'",
            (current,),
        )


def test_unknown_checksum_drift_fails_boot():
    (current,) = _sql(
        "SELECT checksum FROM schema_migrations WHERE filename = '016_candidate_rls.sql'"
    )[0]
    try:
        _sql(
            "UPDATE schema_migrations SET checksum = repeat('0', 64) "
            "WHERE filename = '016_candidate_rls.sql'"
        )
        result = _run_init()
        assert result.returncode == 1
        assert "changed since it was applied" in result.stdout
    finally:
        _sql(
            "UPDATE schema_migrations SET checksum = %s WHERE filename = '016_candidate_rls.sql'",
            (current,),
        )


def test_reserved_runtime_role_rejected():
    result = _run_init(ACTIVEKG_RUNTIME_ROLE="app_user", ACTIVEKG_RUNTIME_PASSWORD="irrelevant")
    assert result.returncode == 1
    assert "must be a dedicated role" in result.stdout


def test_partial_legacy_baseline_rejected_then_verified():
    """The empirically found 006 false positive: a duplicate-column error must
    not baseline the migration while its indexes are missing."""
    try:
        _sql("DELETE FROM schema_migrations WHERE filename = '006_add_key_version.sql'")
        _sql("DROP INDEX IF EXISTS idx_connector_configs_key_version")
        result = _run_init()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "cannot be baselined" in result.stdout
        assert "idx_connector_configs_key_version" in result.stdout
        # Ledger must NOT contain a false baseline.
        rows = _sql("SELECT 1 FROM schema_migrations WHERE filename = '006_add_key_version.sql'")
        assert rows == []

        # Restore the missing object: baselining must now verify and succeed.
        _sql(
            "CREATE INDEX IF NOT EXISTS idx_connector_configs_key_version "
            "ON connector_configs (key_version)"
        )
        result = _run_init()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "baselined (all objects verified present)" in result.stdout
    finally:
        _sql(
            "CREATE INDEX IF NOT EXISTS idx_connector_configs_key_version "
            "ON connector_configs (key_version)"
        )
        rows = _sql("SELECT 1 FROM schema_migrations WHERE filename = '006_add_key_version.sql'")
        if not rows:
            _run_init()


def test_previously_baselined_rows_are_reverified():
    """A historical baselined=true row whose objects are missing must fail boot."""
    (was_baselined,) = _sql(
        "SELECT baselined FROM schema_migrations WHERE filename = '005_connector_configs_table.sql'"
    )[0]
    try:
        _sql(
            "UPDATE schema_migrations SET baselined = true "
            "WHERE filename = '005_connector_configs_table.sql'"
        )
        _sql("DROP TRIGGER IF EXISTS connector_configs_updated_at ON connector_configs")
        result = _run_init()
        assert result.returncode == 1
        assert "fails re-verification" in result.stdout
    finally:
        _sql(
            "CREATE TRIGGER connector_configs_updated_at "
            "BEFORE UPDATE ON connector_configs FOR EACH ROW "
            "EXECUTE FUNCTION update_connector_configs_updated_at()"
        )
        _sql(
            "UPDATE schema_migrations SET baselined = %s "
            "WHERE filename = '005_connector_configs_table.sql'",
            (was_baselined,),
        )
