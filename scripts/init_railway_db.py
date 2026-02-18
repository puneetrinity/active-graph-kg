#!/usr/bin/env python3
"""
Initialize Railway database with schema and pgvector extension.
Run this from the Railway environment where ACTIVEKG_DSN is available.
"""

import os
import sys
import time

import psycopg

MAX_RETRIES = 10
RETRY_DELAY = 3  # seconds


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


def main():
    dsn = os.environ.get("ACTIVEKG_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: ACTIVEKG_DSN or DATABASE_URL environment variable not set")
        sys.exit(1)

    print("Connecting to database...")

    try:
        # Connect to database (with retry for Railway cold starts)
        with _connect_with_retry(dsn) as conn:
            with conn.cursor() as cur:
                # Check if pgvector is available
                print("Checking pgvector extension availability...")
                cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")
                if not cur.fetchone():
                    print("ERROR: pgvector extension is not available in this PostgreSQL instance")
                    print("Railway's default PostgreSQL doesn't include pgvector.")
                    print("\nPlease deploy a PostgreSQL instance with pgvector:")
                    print("1. Remove the current Postgres service")
                    print("2. Add a new service from the 'pgvector/pgvector:pg16' Docker image")
                    print(
                        "3. Set environment variables: POSTGRES_USER=activekg, POSTGRES_PASSWORD=<password>, POSTGRES_DB=activekg"
                    )
                    sys.exit(1)

                print("✓ pgvector extension is available")

                # Enable extensions
                print("Creating extensions...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
                print("✓ Extensions created")

                # Check if schema is already initialized
                cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes';")
                schema_exists = cur.fetchone() is not None

                if schema_exists:
                    print("✓ Database schema already initialized (skipping init.sql)")
                else:
                    # Read and execute init.sql
                    print("Initializing database schema...")
                    init_sql_path = os.path.join(os.path.dirname(__file__), "..", "db", "init.sql")
                    with open(init_sql_path) as f:
                        sql = f.read()

                    # Execute schema creation
                    cur.execute(sql)
                    print("✓ Database schema initialized")

                # Check if RLS policies file exists
                rls_sql_path = os.path.join(
                    os.path.dirname(__file__), "..", "enable_rls_policies.sql"
                )
                if os.path.exists(rls_sql_path):
                    print("Applying RLS policies...")
                    with open(rls_sql_path) as f:
                        sql = f.read()
                    try:
                        cur.execute(sql)
                        print("✓ RLS policies applied")
                    except Exception as e:
                        error_msg = str(e).lower()
                        if "already exists" in error_msg or "duplicate" in error_msg:
                            print("⊙ RLS policies already applied (skipped)")
                        else:
                            raise

                # Apply migrations (idempotent - safe to run multiple times)
                migrations = [
                    "001_add_embedding_history_index.sql",
                    "004_add_external_id_index.sql",
                    "add_text_search.sql",
                    "005_connector_configs_table.sql",
                    "006_add_key_version.sql",
                    "007_add_provider_check.sql",
                    "008_connector_cursors_table.sql",
                    "009_embedding_queue_status.sql",
                    "010_update_text_search_vector.sql",
                    "011_unique_tenant_external_id.sql",
                ]
                applied = 0
                skipped = 0

                def execute_sql(sql_text: str, *, split_statements: bool = False) -> None:
                    if not split_statements:
                        cur.execute(sql_text)
                        return
                    # Execute statements one-by-one (needed for CREATE INDEX CONCURRENTLY)
                    for raw_stmt in sql_text.split(";"):
                        stmt = raw_stmt.strip()
                        if not stmt:
                            continue
                        # Skip chunks that are only comments
                        has_sql = False
                        for line in stmt.splitlines():
                            stripped = line.strip()
                            if stripped and not stripped.startswith("--"):
                                has_sql = True
                                break
                        if not has_sql:
                            continue
                        cur.execute(stmt)

                for migration_file in migrations:
                    migration_path = os.path.join(
                        os.path.dirname(__file__), "..", "db", "migrations", migration_file
                    )
                    if os.path.exists(migration_path):
                        print(f"Applying migration: {migration_file}...")
                        try:
                            with open(migration_path) as f:
                                sql = f.read()
                            needs_split = "create index concurrently" in sql.lower()
                            execute_sql(sql, split_statements=needs_split)
                            print(f"✓ Migration {migration_file} applied")
                            applied += 1
                        except Exception as e:
                            # Migrations may already be applied - check for specific errors
                            error_msg = str(e).lower()
                            if "already exists" in error_msg or "duplicate" in error_msg:
                                print(f"⊙ Migration {migration_file} already applied (skipped)")
                                skipped += 1
                            else:
                                # Re-raise unexpected errors
                                raise
                    else:
                        print(f"⊙ Migration {migration_file} missing (skipped)")
                        skipped += 1
                print(
                    f"\n✅ Database initialization complete! (applied={applied}, skipped={skipped})"
                )

    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
