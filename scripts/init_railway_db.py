#!/usr/bin/env python3
"""
Initialize Railway database with schema and pgvector extension.
Run this from the Railway environment where ACTIVEKG_DSN is available.
"""

import os
import sys

import psycopg


def main():
    dsn = os.environ.get("ACTIVEKG_DSN") or os.environ.get("DATABASE_URL")
    if not dsn:
        print("ERROR: ACTIVEKG_DSN or DATABASE_URL environment variable not set")
        sys.exit(1)

    print("Connecting to database...")

    try:
        # Connect to database
        with psycopg.connect(dsn, autocommit=True) as conn:
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
                ]
                applied = 0
                skipped = 0
                for migration_file in migrations:
                    migration_path = os.path.join(
                        os.path.dirname(__file__), "..", "db", "migrations", migration_file
                    )
                    if os.path.exists(migration_path):
                        print(f"Applying migration: {migration_file}...")
                        try:
                            with open(migration_path) as f:
                                sql = f.read()
                            cur.execute(sql)
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
