#!/usr/bin/env python3
"""Initialize/migrate the database on the deploy path.

Runs before the API starts (scripts/start_railway.sh). Responsibilities:

* base schema (db/init.sql) + extensions on a fresh database
* RLS policies (enable_rls_policies.sql — contains no login roles)
* ordered migrations from activekg.common.migration_manifest, tracked in a
  ``schema_migrations`` ledger under a Postgres advisory lock
* optional provisioning of the restricted runtime role from env vars

DSNs:
    ACTIVEKG_MIGRATE_DSN   privileged/owner role used ONLY here (preferred)
    ACTIVEKG_DSN           runtime DSN; used as a fallback with a warning so
                           single-DSN dev environments keep working
    DATABASE_URL           last-resort fallback (Railway convention)

Runtime role provisioning (optional, all-or-nothing):
    ACTIVEKG_RUNTIME_ROLE      role name to create/harden (NOSUPERUSER,
                               NOBYPASSRLS, no ownership — RLS applies to it)
    ACTIVEKG_RUNTIME_PASSWORD  credential, injected via the environment;
                               never stored in this repository

Failure policy: a missing migration file, or any error that is not a
duplicate-object condition (e.g. unique-constraint violations from duplicate
data), aborts startup with a non-zero exit. Pre-ledger databases baseline
cleanly: objects that already exist are recorded as baselined, data errors
are not forgiven.
"""

import hashlib
import importlib.util
import os
import re
import sys
import time

import psycopg
from psycopg import sql

MAX_RETRIES = 10
RETRY_DELAY = 3  # seconds
ADVISORY_LOCK_KEY = 0x41435447  # 'ACTG'

# SQLSTATEs that mean "object already exists" — safe to baseline on a database
# migrated before the ledger existed. Data errors (23xxx) are deliberately
# absent: a unique violation during a migration is a real failure.
DUPLICATE_OBJECT_SQLSTATES = {
    "42P07",  # duplicate_table
    "42701",  # duplicate_column
    "42710",  # duplicate_object
    "42723",  # duplicate_function
    "42P06",  # duplicate_schema
}

# Migrations that were edited before the immutability rule took effect.
# Maps filename -> {old recorded checksum: new expected checksum}; a mismatch
# matching one of these pairs is upgraded in place instead of failing boot.
CHECKSUM_TRANSITIONS: dict[str, dict[str, str]] = {
    "016_candidate_rls.sql": {
        # PR #11 preflight -> PR #12 effective-tenant preflight rewrite
        "34f02ce7137003697e1a3e0a675883b5203d55150ea1a0c258892308ae344b21": (
            "2294ef74ce9436782dc5f3c1484939bb53edec69e963233f5ee705a3849d6a63"
        ),
    },
}

# Baseline verifiers: before a migration may be recorded as baselined off a
# duplicate-object error, EVERY listed object must already exist — one stray
# duplicate must not vouch for a partially migrated legacy database.
# Forms: ("table", name) | ("column", table, column) | ("index", name)
#        | ("constraint", name) | ("policy", table, name)
#        | ("function", name) | ("trigger", table, name)
BASELINE_VERIFIERS: dict[str, list[tuple[str, ...]]] = {
    "001_add_embedding_history_index.sql": [("index", "idx_embedding_history_created_at")],
    "004_add_external_id_index.sql": [
        ("index", "idx_nodes_external_id"),
        ("index", "idx_nodes_external_id_parent"),
    ],
    "add_text_search.sql": [
        ("column", "nodes", "text_search_vector"),
        ("trigger", "nodes", "nodes_text_search_update"),
    ],
    "005_connector_configs_table.sql": [("table", "connector_configs")],
    "006_add_key_version.sql": [("column", "connector_configs", "key_version")],
    "007_add_provider_check.sql": [("constraint", "chk_provider_valid")],
    "008_connector_cursors_table.sql": [("table", "connector_cursors")],
    "009_embedding_queue_status.sql": [
        ("column", "nodes", "embedding_status"),
        ("column", "nodes", "embedding_attempts"),
    ],
    "010_update_text_search_vector.sql": [("function", "update_text_search_vector")],
    "011_unique_tenant_external_id.sql": [("index", "idx_nodes_tenant_external_id_unique")],
    "012_global_memory.sql": [
        ("table", "global_candidates"),
        ("table", "candidate_provenance"),
    ],
    "012_candidate_identity.sql": [
        ("table", "candidates"),
        ("table", "candidate_identifiers"),
        ("table", "candidate_source_records"),
        ("index", "idx_candidate_identifiers_unique"),
    ],
    "013_vantahire_provenance.sql": [("column", "candidate_source_records", "org_id")],
    "014_signal_job_tags.sql": [("column", "candidate_source_records", "job_tags")],
    "015_candidate_profile.sql": [
        ("column", "candidates", "profile"),
        ("column", "candidates", "skills"),
    ],
    "016_candidate_rls.sql": [
        ("policy", "candidates", "tenant_isolation_candidates"),
        ("policy", "candidate_identifiers", "tenant_isolation_candidate_identifiers"),
        ("policy", "candidate_source_records", "tenant_isolation_candidate_source_records"),
        ("constraint", "candidate_identifiers_tenant_candidate_fkey"),
        ("constraint", "candidate_source_records_tenant_candidate_fkey"),
    ],
    "017_reserve_quarantine_tenant.sql": [
        ("policy", "candidates", "tenant_isolation_candidates"),
    ],
    "018_tenant_nonblank.sql": [
        ("constraint", "candidates_tenant_nonblank"),
        ("constraint", "candidate_identifiers_tenant_nonblank"),
        ("constraint", "candidate_source_records_tenant_nonblank"),
    ],
}


def _object_exists(cur: psycopg.Cursor, check: tuple[str, ...]) -> bool:
    kind = check[0]
    if kind == "table":
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s",
            (check[1],),
        )
    elif kind == "column":
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
            (check[1], check[2]),
        )
    elif kind == "index":
        cur.execute(
            "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = %s",
            (check[1],),
        )
    elif kind == "constraint":
        cur.execute("SELECT 1 FROM pg_constraint WHERE conname = %s", (check[1],))
    elif kind == "policy":
        cur.execute(
            "SELECT 1 FROM pg_policies WHERE tablename = %s AND policyname = %s",
            (check[1], check[2]),
        )
    elif kind == "function":
        cur.execute("SELECT 1 FROM pg_proc WHERE proname = %s", (check[1],))
    elif kind == "trigger":
        cur.execute(
            "SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
            "WHERE c.relname = %s AND t.tgname = %s",
            (check[1], check[2]),
        )
    else:
        return False
    return cur.fetchone() is not None


def _verify_baseline(cur: psycopg.Cursor, migration_file: str) -> tuple[bool, str]:
    """Return (ok, detail) — whether every object this migration creates exists."""
    checks = BASELINE_VERIFIERS.get(migration_file)
    if checks is None:
        return False, "no baseline verifier defined for this migration"
    missing = [" ".join(c) for c in checks if not _object_exists(cur, c)]
    if missing:
        return False, f"objects missing: {', '.join(missing)}"
    return True, ""


def _load_manifest() -> tuple[str, ...]:
    """Load the migration manifest without importing the activekg package."""
    manifest_path = os.path.join(
        os.path.dirname(__file__), "..", "activekg", "common", "migration_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("migration_manifest", manifest_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MIGRATIONS


def _connect_with_retry(dsn: str) -> psycopg.Connection:
    """Try connecting to the database with retries (Railway services start concurrently)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return psycopg.connect(dsn, autocommit=True)
        except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout) as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
            print(f"  Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    raise RuntimeError("unreachable")


def _is_duplicate_object_error(e: Exception) -> bool:
    # When a SQLSTATE is available it is authoritative: a unique-data
    # violation (23505) carries "already exists" in its DETAIL text and must
    # NOT be mistaken for a duplicate object. The string fallback only covers
    # errors that surface without a SQLSTATE (e.g. wrapped DO-block DDL).
    sqlstate = getattr(e, "sqlstate", None)
    if sqlstate is not None:
        return sqlstate in DUPLICATE_OBJECT_SQLSTATES
    return "already exists" in str(e).lower()


def _ensure_extensions_and_schema(cur: psycopg.Cursor) -> None:
    print("Checking pgvector extension availability...")
    cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")
    if not cur.fetchone():
        print("ERROR: pgvector extension is not available in this PostgreSQL instance")
        print("Deploy a PostgreSQL image that ships pgvector (e.g. pgvector/pgvector:pg16).")
        sys.exit(1)
    print("✓ pgvector extension is available")

    print("Creating extensions...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    print("✓ Extensions created")

    cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes';")
    if cur.fetchone() is not None:
        print("✓ Database schema already initialized (skipping init.sql)")
    else:
        print("Initializing database schema...")
        init_sql_path = os.path.join(os.path.dirname(__file__), "..", "db", "init.sql")
        with open(init_sql_path) as f:
            cur.execute(f.read())
        print("✓ Database schema initialized")

    rls_sql_path = os.path.join(os.path.dirname(__file__), "..", "enable_rls_policies.sql")
    if os.path.exists(rls_sql_path):
        print("Applying RLS policies...")
        with open(rls_sql_path) as f:
            rls_sql = f.read()
        try:
            cur.execute(rls_sql)
            print("✓ RLS policies applied")
        except Exception as e:
            if _is_duplicate_object_error(e):
                print("⊙ RLS policies already applied (skipped)")
            else:
                raise


def _execute_migration_sql(cur: psycopg.Cursor, sql_text: str) -> None:
    if "create index concurrently" not in sql_text.lower():
        cur.execute(sql_text)
        return
    # Execute statements one-by-one (needed for CREATE INDEX CONCURRENTLY)
    for raw_stmt in sql_text.split(";"):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        has_sql = any(
            line.strip() and not line.strip().startswith("--") for line in stmt.splitlines()
        )
        if has_sql:
            cur.execute(stmt)


def _apply_migrations(cur: psycopg.Cursor, migrations: tuple[str, ...]) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            baselined BOOLEAN NOT NULL DEFAULT false
        )
        """
    )
    cur.execute("ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum TEXT")
    cur.execute("SELECT filename, checksum FROM schema_migrations")
    ledger = {row[0]: row[1] for row in cur.fetchall()}

    applied = 0
    baselined = 0
    ledger_skipped = 0

    for migration_file in migrations:
        migration_path = os.path.join(
            os.path.dirname(__file__), "..", "db", "migrations", migration_file
        )
        if not os.path.exists(migration_path):
            print(f"ERROR: migration listed in manifest but missing on disk: {migration_file}")
            sys.exit(1)

        with open(migration_path) as f:
            migration_sql = f.read()
        checksum = hashlib.sha256(migration_sql.encode("utf-8")).hexdigest()

        if migration_file in ledger:
            recorded = ledger[migration_file]
            if recorded is None:
                # Rows recorded before the checksum column existed: adopt the
                # current file as the trusted baseline.
                cur.execute(
                    "UPDATE schema_migrations SET checksum = %s WHERE filename = %s",
                    (checksum, migration_file),
                )
                print(f"⊙ Backfilled checksum for {migration_file}")
            elif recorded != checksum:
                transition = CHECKSUM_TRANSITIONS.get(migration_file, {}).get(recorded)
                if transition == checksum:
                    cur.execute(
                        "UPDATE schema_migrations SET checksum = %s WHERE filename = %s",
                        (checksum, migration_file),
                    )
                    print(
                        f"⊙ {migration_file}: known checksum transition applied "
                        f"({recorded[:12]} → {checksum[:12]})"
                    )
                    ledger_skipped += 1
                    continue
                if os.getenv("ACTIVEKG_ALLOW_MIGRATION_DRIFT", "false").lower() == "true":
                    print(
                        f"WARNING: {migration_file} changed since it was applied "
                        f"(recorded {recorded[:12]}, on disk {checksum[:12]}); "
                        "continuing because ACTIVEKG_ALLOW_MIGRATION_DRIFT=true."
                    )
                else:
                    print(
                        f"ERROR: {migration_file} changed since it was applied "
                        f"(recorded {recorded[:12]}, on disk {checksum[:12]}). "
                        "Applied migrations are immutable; add a new migration instead, "
                        "or set ACTIVEKG_ALLOW_MIGRATION_DRIFT=true to override."
                    )
                    sys.exit(1)
            ledger_skipped += 1
            continue

        print(f"Applying migration: {migration_file}...")
        try:
            _execute_migration_sql(cur, migration_sql)
        except Exception as e:
            if _is_duplicate_object_error(e):
                # One duplicate error is not proof the whole migration is
                # present — verify every object it creates before baselining.
                ok, detail = _verify_baseline(cur, migration_file)
                if not ok:
                    print(
                        f"ERROR: migration {migration_file} hit a duplicate-object "
                        f"error but cannot be baselined: {detail}. The database "
                        "appears partially migrated; reconcile it manually."
                    )
                    sys.exit(1)
                print(f"⊙ Migration {migration_file} baselined (all objects verified present)")
                cur.execute(
                    "INSERT INTO schema_migrations (filename, baselined, checksum) "
                    "VALUES (%s, true, %s)",
                    (migration_file, checksum),
                )
                baselined += 1
                continue
            print(f"ERROR: migration {migration_file} failed: {e}")
            sys.exit(1)

        cur.execute(
            "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s)",
            (migration_file, checksum),
        )
        print(f"✓ Migration {migration_file} applied")
        applied += 1

    print(
        f"Migrations complete (applied={applied}, baselined={baselined}, "
        f"ledger_skipped={ledger_skipped})"
    )


def _provision_runtime_role(cur: psycopg.Cursor) -> None:
    role = os.environ.get("ACTIVEKG_RUNTIME_ROLE")
    if not role:
        print("⊙ ACTIVEKG_RUNTIME_ROLE not set — skipping runtime role provisioning")
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role):
        print(f"ERROR: invalid ACTIVEKG_RUNTIME_ROLE name: {role!r}")
        sys.exit(1)

    # Refuse to provision reserved or self-defeating role names: hardening the
    # migration user would demote the owner mid-flight, and app_user is set
    # NOLOGIN by the legacy remediation right after provisioning.
    cur.execute("SELECT current_user")
    migration_user = cur.fetchone()[0]
    if role in {migration_user, "postgres", "app_user", "admin_role"}:
        print(
            f"ERROR: ACTIVEKG_RUNTIME_ROLE must be a dedicated role, not {role!r} "
            f"(migration user: {migration_user!r}; reserved: postgres, app_user, admin_role)"
        )
        sys.exit(1)

    password = os.environ.get("ACTIVEKG_RUNTIME_PASSWORD")
    role_ident = sql.Identifier(role)

    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
    exists = cur.fetchone() is not None

    if not exists:
        if not password:
            print("ERROR: ACTIVEKG_RUNTIME_PASSWORD required to create the runtime role")
            sys.exit(1)
        cur.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS"
            ).format(role_ident, sql.Literal(password))
        )
        print(f"✓ Runtime role {role} created (NOSUPERUSER NOBYPASSRLS)")
    else:
        # Enforce the security posture even if the role pre-exists.
        cur.execute(
            sql.SQL("ALTER ROLE {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS").format(
                role_ident
            )
        )
        if password:
            cur.execute(
                sql.SQL("ALTER ROLE {} PASSWORD {}").format(role_ident, sql.Literal(password))
            )
        print(f"✓ Runtime role {role} hardened (NOSUPERUSER NOBYPASSRLS)")

    # Data access only — never ownership, so RLS policies apply.
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role_ident))
    cur.execute(
        sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}").format(
            role_ident
        )
    )
    cur.execute(
        sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(role_ident)
    )
    cur.execute(
        sql.SQL(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {}"
        ).format(role_ident)
    )

    # The runtime role must never hold the admin_role RLS bypass — including
    # through inherited (indirect) membership.
    cur.execute("SELECT pg_has_role(%s, 'admin_role', 'MEMBER')", (role,))
    if cur.fetchone()[0]:
        cur.execute(sql.SQL("REVOKE admin_role FROM {}").format(role_ident))
        cur.execute("SELECT pg_has_role(%s, 'admin_role', 'MEMBER')", (role,))
        if cur.fetchone()[0]:
            print(
                f"ERROR: {role} still inherits admin_role through an intermediate "
                "role; revoke that membership chain manually."
            )
            sys.exit(1)
        print(f"✓ Revoked admin_role membership from {role}")

    # The ledger is readiness-trusted state: the app may read it, never write it.
    cur.execute(
        sql.SQL("REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON schema_migrations FROM {}").format(
            role_ident
        )
    )
    print(f"✓ Runtime role {role} granted table access (no ownership; ledger read-only)")


def _remediate_legacy_app_user(cur: psycopg.Cursor) -> None:
    """Disable the app_user role older installs created with a known password."""
    cur.execute("SELECT rolcanlogin FROM pg_roles WHERE rolname = 'app_user'")
    row = cur.fetchone()
    if row is None:
        return
    if row[0]:
        cur.execute("ALTER ROLE app_user NOLOGIN")
        print(
            "✓ Legacy app_user role disabled (NOLOGIN) — it was provisioned with a "
            "known default password. Use ACTIVEKG_RUNTIME_ROLE provisioning instead."
        )


def main():
    migrate_dsn = os.environ.get("ACTIVEKG_MIGRATE_DSN")
    if not migrate_dsn:
        migrate_dsn = os.environ.get("ACTIVEKG_DSN") or os.environ.get("DATABASE_URL")
        if migrate_dsn:
            print(
                "WARNING: ACTIVEKG_MIGRATE_DSN not set — falling back to the runtime DSN. "
                "Migrations should run as a separate privileged role; the runtime role "
                "must stay non-owner so RLS applies to it."
            )
    if not migrate_dsn:
        print("ERROR: ACTIVEKG_MIGRATE_DSN (or ACTIVEKG_DSN/DATABASE_URL) not set")
        sys.exit(1)

    migrations = _load_manifest()
    print("Connecting to database...")

    try:
        with _connect_with_retry(migrate_dsn) as conn:
            with conn.cursor() as cur:
                # One migrator at a time; concurrent replicas wait here.
                cur.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
                try:
                    _ensure_extensions_and_schema(cur)
                    _apply_migrations(cur, migrations)
                    _provision_runtime_role(cur)
                    _remediate_legacy_app_user(cur)
                finally:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
        print("\n✅ Database initialization complete!")
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
