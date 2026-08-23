"""Regression tests for the deploy-path guards in scripts/init_railway_db.py.

Each test runs the real migration script as a subprocess against a live,
already-migrated database (the CI deploy-path job's), mutates the ledger or
schema to simulate a hazardous state, asserts the guard fires (or the safe
path succeeds), and restores the state it touched.

Covered: the PR#11→PR#12 checksum transition, unknown checksum drift,
reserved runtime-role rejection, adopted-target middle-ledger-gap refusal,
and re-verification of previously baselined rows.

Gated on ``ACTIVEKG_RLS_TEST_OWNER_DSN`` (also used as the migrate DSN).
"""

import os
import subprocess
import sys
import uuid
from hashlib import sha256

import psycopg
import pytest
from psycopg import sql

OWNER_DSN = os.getenv("ACTIVEKG_RLS_TEST_OWNER_DSN")

pytestmark = pytest.mark.skipif(not OWNER_DSN, reason="ACTIVEKG_RLS_TEST_OWNER_DSN not configured")

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "scripts", "init_railway_db.py")
TARGET_ID = "11111111-1111-4111-8111-111111111111"

PR11_016_CHECKSUM = "34f02ce7137003697e1a3e0a675883b5203d55150ea1a0c258892308ae344b21"


def _run_init(**extra_env: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("ACTIVEKG_")}
    env["ACTIVEKG_MIGRATE_DSN"] = OWNER_DSN
    env["ACTIVEKG_MIGRATION_APPLY"] = "1"
    env["ACTIVEKG_SCHEMA_TARGET_ID"] = TARGET_ID
    env["ACTIVEKG_SCHEMA_ENVIRONMENT"] = "development"
    env["ACTIVEKG_SCHEMA_SOURCE_COMMIT"] = "0" * 40
    for key in ("ACTIVEKG_RUNTIME_ROLE", "ACTIVEKG_RUNTIME_PASSWORD"):
        if value := os.environ.get(key):
            env[key] = value
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


def _assert_safe_refusal(result: subprocess.CompletedProcess) -> None:
    """Assert the release failed without depending on secret-bearing detail.

    The release CLI deliberately exposes only a safe error class at its outer
    boundary. Scenario-specific assertions therefore belong in database state,
    not in production-facing error text.
    """
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[Schema release] REFUSED (" in result.stderr
    combined = result.stdout + result.stderr
    assert OWNER_DSN not in combined
    assert TARGET_ID not in combined


def _assert_release_ok() -> subprocess.CompletedProcess:
    result = _run_init()
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _attempt_state() -> list[tuple]:
    return _sql(
        "SELECT outcome, error_class, finished_at IS NOT NULL "
        "FROM activekg_schema_control.release_attempts ORDER BY id"
    )


def _restore_runtime_posture() -> None:
    """Restore grants after a fixture drops and recreates migration-022 objects.

    The release runner deliberately provisions the runtime role only during a
    fresh install. Replaying migration 022 in these legacy-upgrade scenarios
    therefore recreates tables with the owner's default table grants and a new
    sequence without runtime USAGE. Restore the exact fresh-install posture so
    this shared CI database remains safe for later tests regardless of order.
    """
    role = os.environ.get("ACTIVEKG_RUNTIME_ROLE")
    if not role:
        return
    role_ident = sql.Identifier(role)
    with psycopg.connect(OWNER_DSN) as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "GRANT SELECT, INSERT, UPDATE, DELETE ON "
                "contact_suppression_tombstones, contact_person_suppressions, "
                "contact_suppression_receipts TO {}"
            ).format(role_ident)
        )
        cur.execute(
            sql.SQL(
                "REVOKE DELETE, TRUNCATE ON "
                "contact_suppression_tombstones, contact_person_suppressions, "
                "contact_suppression_receipts FROM {}"
            ).format(role_ident)
        )
        cur.execute(
            sql.SQL("REVOKE UPDATE ON contact_suppression_receipts FROM {}").format(role_ident)
        )
        cur.execute(
            sql.SQL(
                "GRANT USAGE, SELECT ON SEQUENCE contact_suppression_receipts_id_seq TO {}"
            ).format(role_ident)
        )


def _rewind_021_tail() -> bool:
    """Temporarily remove both tail ledger rows so 022 remains a valid prefix upgrade."""

    (privacy_was_baselined,) = _sql(
        "SELECT baselined FROM schema_migrations "
        "WHERE filename='023_candidate_privacy_directives.sql'"
    )[0]
    _sql(
        "DELETE FROM schema_migrations WHERE filename = ANY(%s)",
        (
            [
                "022_contact_suppression_person_and_audit.sql",
                "023_candidate_privacy_directives.sql",
            ],
        ),
    )
    return privacy_was_baselined


def _restore_023_ledger_posture(privacy_was_baselined: bool) -> None:
    if not _sql(
        "SELECT 1 FROM schema_migrations WHERE filename='023_candidate_privacy_directives.sql'"
    ):
        _assert_release_ok()
    _sql(
        "UPDATE schema_migrations SET baselined=%s "
        "WHERE filename='023_candidate_privacy_directives.sql'",
        (privacy_was_baselined,),
    )


def _contact_evidence_owner_sql(query: str, params: tuple = ()) -> None:
    """Run one owner-only fixture mutation while restoring FORCE RLS atomically."""
    with psycopg.connect(OWNER_DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE candidate_contact_evidence NO FORCE ROW LEVEL SECURITY")
            cur.execute(query, params)
            cur.execute("ALTER TABLE candidate_contact_evidence FORCE ROW LEVEL SECURITY")


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
        before_attempts = _attempt_state()
        result = _run_init()
        _assert_safe_refusal(result)
        # The target/ledger proof rejects unknown drift before an attempt or
        # migration body can be opened. The owner fixture remains responsible
        # for restoring its deliberate corruption.
        assert _attempt_state() == before_attempts
        assert _sql(
            "SELECT checksum FROM schema_migrations WHERE filename = '016_candidate_rls.sql'"
        ) == [("0" * 64,)]
    finally:
        _sql(
            "UPDATE schema_migrations SET checksum = %s WHERE filename = '016_candidate_rls.sql'",
            (current,),
        )


def test_reserved_runtime_role_rejected():
    before_attempts = len(_attempt_state())
    try:
        result = _run_init(ACTIVEKG_RUNTIME_ROLE="app_user", ACTIVEKG_RUNTIME_PASSWORD="irrelevant")
        _assert_safe_refusal(result)
        attempts = _attempt_state()
        assert len(attempts) == before_attempts + 1
        assert attempts[-1][0] == "failure"
        assert attempts[-1][1] == "SchemaControlError"
        assert attempts[-1][2] is True
    finally:
        # A failed release is intentionally readiness-blocking until a later
        # verified release succeeds. Leave the shared disposable CI target
        # healthy for the following scenarios.
        _assert_release_ok()


def test_adopted_target_middle_ledger_gap_refuses_before_baseline():
    """An adopted target must reject a missing middle row before migration SQL."""
    original = _sql(
        "SELECT checksum, baselined FROM schema_migrations "
        "WHERE filename = '006_add_key_version.sql'"
    )[0]
    try:
        _sql("DELETE FROM schema_migrations WHERE filename = '006_add_key_version.sql'")
        _sql("DROP INDEX IF EXISTS idx_connector_configs_key_version")
        before_attempts = _attempt_state()
        result = _run_init()
        _assert_safe_refusal(result)
        # Adopted targets may advance only through a manifest-prefix tail.
        # A missing middle ledger row is therefore rejected before the older
        # baseline machinery can bless a partial object set.
        assert _attempt_state() == before_attempts
        rows = _sql("SELECT 1 FROM schema_migrations WHERE filename = '006_add_key_version.sql'")
        assert rows == []
    finally:
        _sql(
            "CREATE INDEX IF NOT EXISTS idx_connector_configs_key_version "
            "ON connector_configs (key_version)"
        )
        rows = _sql("SELECT 1 FROM schema_migrations WHERE filename = '006_add_key_version.sql'")
        if not rows:
            _sql(
                "INSERT INTO schema_migrations (filename, checksum, baselined) "
                "VALUES ('006_add_key_version.sql', %s, %s)",
                original,
            )
        _assert_release_ok()


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
        _assert_release_ok()


def test_021_to_022_upgrade_hashes_opaque_provider_event_and_builds_guards():
    legacy_event_id = "legacy-brevo-provider-event-021"
    expected_event_hash = sha256(legacy_event_id.encode()).hexdigest()
    shared_email_hash = sha256(b"legacy-021-shared@example.com").hexdigest()
    null_event_email_hash = sha256(b"legacy-021-null-event@example.com").hexdigest()
    evidence_only_email_hash = sha256(b"legacy-021-evidence-only@example.com").hexdigest()
    candidate_a = str(uuid.uuid4())
    candidate_b = str(uuid.uuid4())
    candidate_c = str(uuid.uuid4())
    candidate_d = str(uuid.uuid4())
    evidence_b = str(uuid.uuid4())
    evidence_d = str(uuid.uuid4())
    expected_null_event_hash = sha256(
        f"legacy-021-complaint|{null_event_email_hash}|{candidate_c}".encode()
    ).hexdigest()
    expected_evidence_event_hash = sha256(
        f"legacy-021-evidence-complaint|{evidence_d}".encode()
    ).hexdigest()
    privacy_was_baselined = _rewind_021_tail()
    try:
        _sql("DROP TABLE IF EXISTS contact_person_suppressions")
        _sql("DROP TABLE IF EXISTS contact_suppression_receipts")
        _sql("DROP FUNCTION IF EXISTS contact_suppression_receipts_append_only()")
        _sql(
            "ALTER TABLE contact_suppression_tombstones "
            "DROP CONSTRAINT IF EXISTS contact_suppression_provider_event_hash"
        )
        _sql(
            "INSERT INTO global_candidates (id) VALUES (%s), (%s), (%s), (%s)",
            (candidate_a, candidate_b, candidate_c, candidate_d),
        )
        _contact_evidence_owner_sql(
            """
            INSERT INTO candidate_contact_evidence
                (id, global_candidate_id, tenant_id, email, email_hash, provider,
                 status, suppressed_at, observed_at, created_at, updated_at)
            VALUES
                (%s, %s, 'legacy-tenant', 'shared@example.com', %s,
                 'fullenrich', 'complaint', '2029-01-01T00:00:00Z',
                 '2028-01-01T00:00:00Z', '2028-01-01T00:00:00Z',
                 '2029-01-01T00:00:00Z'),
                (%s, %s, 'legacy-tenant', 'evidence-only@example.com', %s,
                 'enrichlayer', 'complaint', '2031-01-01T00:00:00Z',
                 '2030-01-01T00:00:00Z', '2030-01-01T00:00:00Z',
                 '2031-01-01T00:00:00Z')
            """,
            (
                evidence_b,
                candidate_b,
                shared_email_hash,
                evidence_d,
                candidate_d,
                evidence_only_email_hash,
            ),
        )
        _sql(
            "INSERT INTO contact_suppression_tombstones "
            "(email_hash, global_candidate_id, reason, source_evidence_id, "
            " provider_event_id, first_observed_at, last_observed_at) VALUES "
            "(%s, %s, 'complaint', %s, %s, '2027-01-01', '2030-01-01'), "
            "(%s, %s, 'complaint', NULL, NULL, '2027-02-01', '2030-02-01'), "
            "(%s, %s, 'hard_bounce', %s, NULL, '2030-01-01', '2031-01-01')",
            (
                shared_email_hash,
                candidate_a,
                evidence_b,
                legacy_event_id,
                null_event_email_hash,
                candidate_c,
                evidence_only_email_hash,
                candidate_d,
                evidence_d,
            ),
        )

        result = _run_init()
        assert result.returncode == 0, result.stdout + result.stderr
        assert "022_contact_suppression_person_and_audit.sql applied" in result.stdout
        assert _sql(
            "SELECT global_candidate_id::text, provider_event_id "
            "FROM contact_suppression_tombstones WHERE email_hash = %s",
            (shared_email_hash,),
        ) == [(candidate_b, expected_event_hash)]
        assert _sql(
            "SELECT global_candidate_id::text, provider_event_id "
            "FROM contact_person_suppressions "
            "WHERE global_candidate_id = ANY(%s::uuid[]) ORDER BY global_candidate_id",
            ([candidate_a, candidate_b, candidate_c, candidate_d],),
        ) == sorted(
            [
                (candidate_b, expected_event_hash),
                (candidate_c, expected_null_event_hash),
                (candidate_d, expected_evidence_event_hash),
            ]
        )
        assert _sql("SELECT count(*) FROM contact_suppression_receipts") == [(0,)]
        assert _sql(
            "SELECT relforcerowsecurity FROM pg_class WHERE oid = "
            "'candidate_contact_evidence'::regclass"
        ) == [(True,)]
        assert _sql(
            "SELECT to_regclass('contact_person_suppressions'), "
            "to_regclass('contact_suppression_receipts')"
        ) == [("contact_person_suppressions", "contact_suppression_receipts")]
        assert _sql(
            "SELECT 1 FROM pg_constraint "
            "WHERE conname = 'contact_suppression_provider_event_hash' AND convalidated"
        ) == [(1,)]
    finally:
        if _sql("SELECT to_regclass('contact_person_suppressions')") == [
            ("contact_person_suppressions",)
        ]:
            _sql(
                "DELETE FROM contact_person_suppressions "
                "WHERE global_candidate_id = ANY(%s::uuid[])",
                ([candidate_a, candidate_b, candidate_c, candidate_d],),
            )
        _sql(
            "DELETE FROM contact_suppression_tombstones WHERE email_hash = ANY(%s)",
            ([shared_email_hash, null_event_email_hash, evidence_only_email_hash],),
        )
        _contact_evidence_owner_sql(
            "DELETE FROM candidate_contact_evidence WHERE id = ANY(%s::uuid[])",
            ([evidence_b, evidence_d],),
        )
        _sql(
            "DELETE FROM global_candidates WHERE id = ANY(%s::uuid[])",
            ([candidate_a, candidate_b, candidate_c, candidate_d],),
        )
        if not _sql(
            "SELECT 1 FROM schema_migrations "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
        ):
            restored = _run_init()
            assert restored.returncode == 0, restored.stdout + restored.stderr
        _restore_023_ledger_posture(privacy_was_baselined)
        _restore_runtime_posture()


def test_021_to_022_upgrade_rejects_unresolved_legacy_complaint():
    email_hash = sha256(b"legacy-unresolved-complaint@example.com").hexdigest()
    privacy_was_baselined = _rewind_021_tail()
    try:
        _sql("DROP TABLE IF EXISTS contact_person_suppressions")
        _sql("DROP TABLE IF EXISTS contact_suppression_receipts")
        _sql("DROP FUNCTION IF EXISTS contact_suppression_receipts_append_only()")
        _sql(
            "ALTER TABLE contact_suppression_tombstones "
            "DROP CONSTRAINT IF EXISTS contact_suppression_provider_event_hash"
        )
        _sql(
            "INSERT INTO contact_suppression_tombstones (email_hash, reason) "
            "VALUES (%s, 'complaint')",
            (email_hash,),
        )

        result = _run_init()
        _assert_safe_refusal(result)
        assert (
            "022_contact_suppression_person_and_audit.sql failed (RaiseException)" in result.stdout
        )
        assert (
            _sql(
                "SELECT 1 FROM schema_migrations "
                "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
            )
            == []
        )
        assert _sql(
            "SELECT relforcerowsecurity FROM pg_class WHERE oid = "
            "'candidate_contact_evidence'::regclass"
        ) == [(True,)]
    finally:
        _sql("DELETE FROM contact_suppression_tombstones WHERE email_hash = %s", (email_hash,))
        if not _sql(
            "SELECT 1 FROM schema_migrations "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
        ):
            restored = _run_init()
            assert restored.returncode == 0, restored.stdout + restored.stderr
        _restore_023_ledger_posture(privacy_was_baselined)
        _restore_runtime_posture()


def test_022_partial_if_not_exists_schema_is_not_recorded():
    """A missing serial default survives CREATE TABLE IF NOT EXISTS and must fail."""
    privacy_was_baselined = _rewind_021_tail()
    try:
        _sql("ALTER TABLE contact_suppression_receipts ALTER COLUMN id DROP DEFAULT")

        result = _run_init()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "failed post-apply verification" in result.stdout
        assert "serial contact_suppression_receipts id" in result.stdout
        assert (
            _sql(
                "SELECT 1 FROM schema_migrations "
                "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
            )
            == []
        )
    finally:
        _sql(
            "ALTER TABLE contact_suppression_receipts ALTER COLUMN id "
            "SET DEFAULT nextval('contact_suppression_receipts_id_seq'::regclass)"
        )
        rows = _sql(
            "SELECT 1 FROM schema_migrations "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
        )
        if not rows:
            restored = _run_init()
            assert restored.returncode == 0, restored.stdout + restored.stderr
        _restore_023_ledger_posture(privacy_was_baselined)


def test_022_baseline_rejects_replica_only_audit_trigger():
    (was_baselined,) = _sql(
        "SELECT baselined FROM schema_migrations "
        "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
    )[0]
    try:
        _sql(
            "UPDATE schema_migrations SET baselined = true "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
        )
        _sql(
            "ALTER TABLE contact_suppression_receipts ENABLE REPLICA TRIGGER "
            "contact_suppression_receipts_no_mutation"
        )
        result = _run_init()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "fails re-verification" in result.stdout
        assert "contact_suppression_receipts_no_mutation" in result.stdout
    finally:
        _sql(
            "ALTER TABLE contact_suppression_receipts ENABLE TRIGGER "
            "contact_suppression_receipts_no_mutation"
        )
        _sql(
            "UPDATE schema_migrations SET baselined = %s "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'",
            (was_baselined,),
        )
        _assert_release_ok()


def test_022_baseline_rejects_permissive_authority_constraint():
    (was_baselined,) = _sql(
        "SELECT baselined FROM schema_migrations "
        "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
    )[0]
    try:
        _sql(
            "UPDATE schema_migrations SET baselined = true "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
        )
        _sql(
            "ALTER TABLE contact_suppression_receipts "
            "DROP CONSTRAINT contact_suppression_receipt_authority_check; "
            "ALTER TABLE contact_suppression_receipts "
            "ADD CONSTRAINT contact_suppression_receipt_authority_check CHECK (true)"
        )
        result = _run_init()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "fails re-verification" in result.stdout
        assert "contact_suppression_receipt_authority_check" in result.stdout
    finally:
        _sql(
            "ALTER TABLE contact_suppression_receipts "
            "DROP CONSTRAINT contact_suppression_receipt_authority_check; "
            "ALTER TABLE contact_suppression_receipts "
            "ADD CONSTRAINT contact_suppression_receipt_authority_check CHECK ("
            "btrim(issuer) <> '' AND btrim(actor_id) <> '' AND actor_type = 'service')"
        )
        _sql(
            "UPDATE schema_migrations SET baselined = %s "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'",
            (was_baselined,),
        )
        _assert_release_ok()


def test_022_baseline_rejects_wrong_receipt_primary_key_columns():
    (was_baselined,) = _sql(
        "SELECT baselined FROM schema_migrations "
        "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
    )[0]
    try:
        _sql(
            "UPDATE schema_migrations SET baselined = true "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
        )
        _sql(
            "ALTER TABLE contact_suppression_receipts "
            "DROP CONSTRAINT contact_suppression_receipts_pkey; "
            "ALTER TABLE contact_suppression_receipts "
            "ADD CONSTRAINT contact_suppression_receipts_pkey "
            "PRIMARY KEY (id, provider_event_id)"
        )
        result = _run_init()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "fails re-verification" in result.stdout
        assert "contact_suppression_receipts_pkey" in result.stdout
    finally:
        _sql(
            "ALTER TABLE contact_suppression_receipts "
            "DROP CONSTRAINT contact_suppression_receipts_pkey; "
            "ALTER TABLE contact_suppression_receipts "
            "ADD CONSTRAINT contact_suppression_receipts_pkey PRIMARY KEY (id)"
        )
        _sql(
            "UPDATE schema_migrations SET baselined = %s "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'",
            (was_baselined,),
        )
        _assert_release_ok()


def test_022_baseline_rejects_noop_append_only_function():
    (was_baselined,) = _sql(
        "SELECT baselined FROM schema_migrations "
        "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
    )[0]
    try:
        _sql(
            "UPDATE schema_migrations SET baselined = true "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'"
        )
        _sql(
            "CREATE OR REPLACE FUNCTION contact_suppression_receipts_append_only() "
            "RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RETURN OLD; END; $$"
        )
        result = _run_init()
        assert result.returncode == 1, result.stdout + result.stderr
        assert "fails re-verification" in result.stdout
        assert "trigger_function_body contact_suppression_receipts_append_only" in result.stdout
    finally:
        _sql(
            "CREATE OR REPLACE FUNCTION contact_suppression_receipts_append_only() "
            "RETURNS trigger LANGUAGE plpgsql AS $$ "
            "BEGIN RAISE EXCEPTION "
            "'contact_suppression_receipts is append-only (attempted %%)', TG_OP; END; $$"
        )
        _sql(
            "UPDATE schema_migrations SET baselined = %s "
            "WHERE filename = '022_contact_suppression_person_and_audit.sql'",
            (was_baselined,),
        )
        _assert_release_ok()
