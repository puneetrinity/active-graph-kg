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
    sqlstate = getattr(e, "sqlstate", None)
    if sqlstate in DUPLICATE_OBJECT_SQLSTATES:
        return True
    # DO-block wrapped DDL can surface as a generic error; "already exists"
    # only ever comes from object DDL. Unique-violation messages say
    # "duplicate key value", which this deliberately does not match.
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
    cur.execute("SELECT filename FROM schema_migrations")
    already_applied = {row[0] for row in cur.fetchall()}

    applied = 0
    baselined = 0
    ledger_skipped = 0

    for migration_file in migrations:
        if migration_file in already_applied:
            ledger_skipped += 1
            continue

        migration_path = os.path.join(
            os.path.dirname(__file__), "..", "db", "migrations", migration_file
        )
        if not os.path.exists(migration_path):
            print(f"ERROR: migration listed in manifest but missing on disk: {migration_file}")
            sys.exit(1)

        print(f"Applying migration: {migration_file}...")
        with open(migration_path) as f:
            migration_sql = f.read()
        try:
            _execute_migration_sql(cur, migration_sql)
        except Exception as e:
            if _is_duplicate_object_error(e):
                print(f"⊙ Migration {migration_file} baselined (objects already exist)")
                cur.execute(
                    "INSERT INTO schema_migrations (filename, baselined) VALUES (%s, true)",
                    (migration_file,),
                )
                baselined += 1
                continue
            print(f"ERROR: migration {migration_file} failed: {e}")
            sys.exit(1)

        cur.execute(
            "INSERT INTO schema_migrations (filename) VALUES (%s)",
            (migration_file,),
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
    print(f"✓ Runtime role {role} granted table access (no ownership)")


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
