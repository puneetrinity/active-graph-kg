#!/usr/bin/env python3
"""One-time metadata-only adoption of an existing Memory production target."""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import psycopg  # noqa: E402

from activekg.common.schema_control import (  # noqa: E402
    ADVISORY_LOCK_KEY,
    CONTROL_SCHEMA,
    RUNTIME_ROLE_DEFAULT,
    SchemaControlError,
    assert_identity,
    assert_ledger,
    assert_no_product_row_mutation,
    create_control_schema,
    finish_attempt,
    grant_control_read,
    load_migration_records,
    manifest_digest,
    read_identity,
    read_ledger,
    resolve_control_environment,
    safe_error_class,
    safe_target_fingerprint,
    start_attempt,
)
from scripts.init_railway_db import (  # noqa: E402
    _assert_full_baseline,
    _assert_runtime_role_catalog,
    _load_manifest,
)


def _assert_pre_adoption_target(cur: psycopg.Cursor) -> None:
    if read_identity(cur) is not None:
        raise SchemaControlError("Memory adoption refuses an existing target identity")
    cur.execute("SELECT to_regnamespace(%s)", (CONTROL_SCHEMA,))
    if cur.fetchone()[0] is not None:
        raise SchemaControlError("Memory adoption refuses partial schema-control state")
    cur.execute(
        """
        SELECT relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relkind IN ('r','p')
          AND c.relname = ANY(%s)
        """,
        (["nodes", "edges", "events", "candidates", "global_candidates", "schema_migrations"],),
    )
    present = {row[0] for row in cur.fetchall()}
    required = {"nodes", "edges", "events", "candidates", "global_candidates", "schema_migrations"}
    if present != required:
        raise SchemaControlError("database is not the expected existing Memory target")


def main() -> None:
    conn: psycopg.Connection | None = None
    try:
        control = resolve_control_environment()
        if os.getenv("ACTIVEKG_SCHEMA_ADOPT_EXISTING") != "1":
            raise SchemaControlError("ACTIVEKG_SCHEMA_ADOPT_EXISTING=1 is required")
        if os.getenv("ACTIVEKG_MIGRATION_APPLY"):
            raise SchemaControlError("migration apply and adoption flags are mutually exclusive")

        conn = psycopg.connect(control.dsn, autocommit=False, connect_timeout=10)
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET LOCAL statement_timeout = '120s'")
                cur.execute("SET LOCAL lock_timeout = '10s'")
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_KEY,))

                # Establish the product target before reading migration bodies.
                _assert_pre_adoption_target(cur)
                migrations = _load_manifest()
                records = load_migration_records()
                if tuple(record.filename for record in records) != migrations:
                    raise SchemaControlError("migration manifest/record order mismatch")
                assert_ledger(read_ledger(cur), records, allow_prefix=False)
                _assert_full_baseline(cur, migrations)
                role = os.getenv("ACTIVEKG_RUNTIME_ROLE", RUNTIME_ROLE_DEFAULT)
                _assert_runtime_role_catalog(cur, role)

                create_control_schema(cur, control.target_id, control.environment)
                grant_control_read(cur, role)
                digest = manifest_digest(records)
                attempt_id = start_attempt(cur, "adoption", control.source_commit, digest)
                finish_attempt(cur, attempt_id, "success")

                assert_identity(cur, control.target_id, control.environment)
                assert_ledger(read_ledger(cur), records, allow_prefix=False)
                _assert_full_baseline(cur, migrations)
                _assert_runtime_role_catalog(cur, role)
                assert_no_product_row_mutation(cur)

        print(
            "[Schema control] Memory target adopted "
            f"({safe_target_fingerprint(control.target_id)}; migrations={len(records)})"
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        if conn is not None:
            conn.rollback()
        print(f"[Schema control] Adoption refused ({safe_error_class(exc)})", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
