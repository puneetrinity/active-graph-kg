"""Fail-closed Memory schema-control primitives.

This module owns only target identity, release-attempt metadata and startup
admission. ``public.schema_migrations`` remains the sole business-migration
ledger and ``migration_manifest.py`` remains its sole ordered authority.
"""

from __future__ import annotations

import hashlib
import os
import re
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool

from activekg.common.migration_manifest import (
    CHECKSUM_TRANSITIONS,
    MIGRATIONS,
    SCHEMA_PRODUCT,
)

CONTROL_SCHEMA = "activekg_schema_control"
RUNTIME_ROLE_DEFAULT = "activekg_app"
ADVISORY_LOCK_KEY = 0x41435447  # existing Memory migration lock ('ACTG')
ALLOWED_ENVIRONMENTS = {"development", "staging", "production"}
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class SchemaControlError(RuntimeError):
    """A safe, operator-actionable schema-control refusal."""


@dataclass(frozen=True)
class MigrationRecord:
    filename: str
    checksum: str


@dataclass(frozen=True)
class ControlEnvironment:
    dsn: str
    target_id: str
    environment: str
    source_commit: str


CONTROL_DDL = (
    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL("REVOKE ALL ON SCHEMA {} FROM PUBLIC").format(sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL(
        """
        CREATE TABLE {}.target_identity (
            singleton SMALLINT PRIMARY KEY DEFAULT 1 CHECK (singleton = 1),
            product TEXT NOT NULL CHECK (product = 'memory'),
            environment TEXT NOT NULL CHECK (environment IN ('development','staging','production')),
            target_id UUID NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
        )
        """
    ).format(sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL(
        """
        CREATE TABLE {}.release_attempts (
            id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            kind TEXT NOT NULL CHECK (kind IN ('adoption','migration')),
            source_commit TEXT NOT NULL CHECK (source_commit ~ '^[0-9a-f]{{40}}$'),
            manifest_digest TEXT NOT NULL CHECK (manifest_digest ~ '^[0-9a-f]{{64}}$'),
            started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            finished_at TIMESTAMPTZ,
            outcome TEXT NOT NULL CHECK (outcome IN ('running','success','failure')),
            error_class TEXT NOT NULL DEFAULT '' CHECK (length(error_class) <= 120),
            CHECK (
                (outcome = 'running' AND finished_at IS NULL) OR
                (outcome IN ('success','failure') AND finished_at IS NOT NULL)
            )
        )
        """
    ).format(sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL(
        """
        CREATE FUNCTION {}.reject_immutable_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'schema-control identity is immutable';
        END;
        $$
        """
    ).format(sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL(
        """
        CREATE FUNCTION {}.finish_attempt_only() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'UPDATE'
               AND OLD.outcome = 'running' AND OLD.finished_at IS NULL
               AND NEW.id = OLD.id AND NEW.kind = OLD.kind
               AND NEW.source_commit = OLD.source_commit
               AND NEW.manifest_digest = OLD.manifest_digest
               AND NEW.started_at = OLD.started_at
               AND NEW.outcome IN ('success', 'failure')
               AND NEW.finished_at IS NOT NULL
            THEN
                RETURN NEW;
            END IF;
            RAISE EXCEPTION 'schema-control release attempts are append-only';
        END;
        $$
        """
    ).format(sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL(
        "CREATE TRIGGER target_identity_no_mutation BEFORE UPDATE OR DELETE ON "
        "{}.target_identity FOR EACH ROW EXECUTE FUNCTION {}.reject_immutable_mutation()"
    ).format(sql.Identifier(CONTROL_SCHEMA), sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL(
        "CREATE TRIGGER target_identity_no_truncate BEFORE TRUNCATE ON "
        "{}.target_identity FOR EACH STATEMENT EXECUTE FUNCTION {}.reject_immutable_mutation()"
    ).format(sql.Identifier(CONTROL_SCHEMA), sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL(
        "CREATE TRIGGER release_attempts_finish_only BEFORE UPDATE OR DELETE ON "
        "{}.release_attempts FOR EACH ROW EXECUTE FUNCTION {}.finish_attempt_only()"
    ).format(sql.Identifier(CONTROL_SCHEMA), sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL(
        "CREATE TRIGGER release_attempts_no_truncate BEFORE TRUNCATE ON "
        "{}.release_attempts FOR EACH STATEMENT EXECUTE FUNCTION {}.reject_immutable_mutation()"
    ).format(sql.Identifier(CONTROL_SCHEMA), sql.Identifier(CONTROL_SCHEMA)),
    sql.SQL("REVOKE ALL ON ALL TABLES IN SCHEMA {} FROM PUBLIC").format(
        sql.Identifier(CONTROL_SCHEMA)
    ),
    sql.SQL("REVOKE ALL ON ALL SEQUENCES IN SCHEMA {} FROM PUBLIC").format(
        sql.Identifier(CONTROL_SCHEMA)
    ),
    sql.SQL("REVOKE ALL ON ALL FUNCTIONS IN SCHEMA {} FROM PUBLIC").format(
        sql.Identifier(CONTROL_SCHEMA)
    ),
)


def safe_error_class(exc: BaseException) -> str:
    """Return a non-secret error class suitable for logs/control metadata."""

    name = type(exc).__name__
    return name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,119}", name) else "SchemaControlError"


def safe_target_fingerprint(target_id: str) -> str:
    return hashlib.sha256(target_id.encode("utf-8")).hexdigest()[:16]


def validate_target_id(value: str | None) -> str:
    try:
        parsed = uuid.UUID(value or "")
    except (ValueError, TypeError, AttributeError) as exc:
        raise SchemaControlError("ACTIVEKG_SCHEMA_TARGET_ID must be an exact UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise SchemaControlError("ACTIVEKG_SCHEMA_TARGET_ID must use canonical lowercase UUID form")
    return canonical


def validate_environment(value: str | None) -> str:
    if value not in ALLOWED_ENVIRONMENTS:
        raise SchemaControlError(
            "ACTIVEKG_SCHEMA_ENVIRONMENT must be development, staging or production"
        )
    return value


def validate_source_commit(value: str | None) -> str:
    normalized = (value or "").lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise SchemaControlError("schema release requires an exact 40-character source commit")
    return normalized


def resolve_source_commit(environ: Mapping[str, str] | None = None) -> str:
    env = environ or os.environ
    return validate_source_commit(
        env.get("ACTIVEKG_SCHEMA_SOURCE_COMMIT") or env.get("RAILWAY_GIT_COMMIT_SHA")
    )


def resolve_control_environment(environ: Mapping[str, str] | None = None) -> ControlEnvironment:
    env = environ or os.environ
    dsn = env.get("ACTIVEKG_MIGRATE_DSN", "")
    if not dsn:
        raise SchemaControlError("ACTIVEKG_MIGRATE_DSN is required; no DSN fallback is allowed")
    return ControlEnvironment(
        dsn=dsn,
        target_id=validate_target_id(env.get("ACTIVEKG_SCHEMA_TARGET_ID")),
        environment=validate_environment(env.get("ACTIVEKG_SCHEMA_ENVIRONMENT")),
        source_commit=resolve_source_commit(env),
    )


def resolve_runtime_dsn(
    environ: Mapping[str, str] | None = None,
    *,
    development_default: str | None = None,
) -> str:
    env = environ or os.environ
    environment = env.get("ACTIVEKG_SCHEMA_ENVIRONMENT")
    activekg_dsn = env.get("ACTIVEKG_DSN", "")
    if environment == "production":
        forbidden = {
            name
            for name in (
                "ACTIVEKG_MIGRATE_DSN",
                "ACTIVEKG_MIGRATION_APPLY",
                "ACTIVEKG_SCHEMA_ADOPT_EXISTING",
                "ACTIVEKG_SCHEMA_FRESH_INIT",
                "ACTIVEKG_RUNTIME_PASSWORD",
                "ACTIVEKG_ALLOW_MIGRATION_DRIFT",
            )
            if env.get(name)
        }
        if forbidden:
            raise SchemaControlError(
                "production runtime contains forbidden schema-release authority"
            )
        if not activekg_dsn:
            raise SchemaControlError(
                "production runtime requires ACTIVEKG_DSN; DATABASE_URL fallback is forbidden"
            )
        return activekg_dsn
    return activekg_dsn or env.get("DATABASE_URL", "") or development_default or ""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_migration_records(root: Path | None = None) -> tuple[MigrationRecord, ...]:
    base = root or repository_root()
    records: list[MigrationRecord] = []
    for filename in MIGRATIONS:
        path = base / "db" / "migrations" / filename
        if not path.is_file():
            raise SchemaControlError(f"migration manifest entry is missing on disk: {filename}")
        records.append(MigrationRecord(filename, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(records)


def manifest_digest(records: Sequence[MigrationRecord]) -> str:
    payload = "".join(f"{record.filename}\0{record.checksum}\n" for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_identity(cur: psycopg.Cursor[Any]) -> tuple[str, str, str] | None:
    cur.execute("SELECT to_regnamespace(%s)", (CONTROL_SCHEMA,))
    namespace = cur.fetchone()
    if namespace is None:
        raise SchemaControlError("Memory schema-control namespace probe returned no row")
    if namespace[0] is None:
        return None
    cur.execute(
        sql.SQL("SELECT product, environment, target_id::text FROM {}.target_identity").format(
            sql.Identifier(CONTROL_SCHEMA)
        )
    )
    rows = cur.fetchall()
    if len(rows) != 1:
        raise SchemaControlError(
            "Memory schema-control target identity must contain exactly one row"
        )
    return rows[0][0], rows[0][1], rows[0][2]


def assert_identity(cur: psycopg.Cursor[Any], target_id: str, environment: str) -> None:
    identity = read_identity(cur)
    if identity != (SCHEMA_PRODUCT, environment, target_id):
        raise SchemaControlError("Memory schema-control target identity mismatch")


def read_ledger(cur: psycopg.Cursor[Any]) -> list[tuple[str, str | None, bool]]:
    cur.execute("SELECT to_regclass('public.schema_migrations')")
    relation = cur.fetchone()
    if relation is None:
        raise SchemaControlError("Memory migration-ledger probe returned no row")
    if relation[0] is None:
        return []
    cur.execute("SELECT filename, checksum, baselined FROM public.schema_migrations")
    return [(str(row[0]), row[1], bool(row[2])) for row in cur.fetchall()]


def assert_ledger(
    rows: Sequence[tuple[str, str | None, bool]],
    records: Sequence[MigrationRecord],
    *,
    allow_prefix: bool,
) -> None:
    by_name = {row[0]: row for row in rows}
    if len(by_name) != len(rows):
        raise SchemaControlError("Memory migration ledger contains duplicate filenames")
    expected_names = [record.filename for record in records]
    actual_names = [name for name in expected_names if name in by_name]
    if set(by_name) - set(expected_names):
        raise SchemaControlError("Memory migration ledger contains an unknown migration")
    if actual_names != expected_names[: len(actual_names)]:
        raise SchemaControlError("Memory migration ledger is not an ordered manifest prefix")
    if not allow_prefix and len(actual_names) != len(expected_names):
        raise SchemaControlError("Memory migration ledger is incomplete")
    for record in records[: len(actual_names)]:
        recorded = by_name[record.filename][1]
        if recorded == record.checksum:
            continue
        if (
            recorded is not None
            and CHECKSUM_TRANSITIONS.get(record.filename, {}).get(recorded) == record.checksum
        ):
            continue
        raise SchemaControlError(f"Memory migration checksum mismatch: {record.filename}")


def read_release_health(cur: psycopg.Cursor[Any]) -> tuple[int, str | None]:
    cur.execute(
        sql.SQL(
            "SELECT count(*) FILTER (WHERE outcome = 'running' OR finished_at IS NULL) "
            "FROM {}.release_attempts"
        ).format(sql.Identifier(CONTROL_SCHEMA))
    )
    unfinished_row = cur.fetchone()
    if unfinished_row is None:
        raise SchemaControlError("Memory release-health probe returned no row")
    unfinished = int(unfinished_row[0])
    cur.execute(
        sql.SQL("SELECT outcome FROM {}.release_attempts ORDER BY id DESC LIMIT 1").format(
            sql.Identifier(CONTROL_SCHEMA)
        )
    )
    latest = cur.fetchone()
    return unfinished, (str(latest[0]) if latest else None)


def create_control_schema(cur: psycopg.Cursor[Any], target_id: str, environment: str) -> None:
    if read_identity(cur) is not None:
        raise SchemaControlError("Memory target adoption refuses existing control state")
    cur.execute("SELECT to_regnamespace(%s)", (CONTROL_SCHEMA,))
    namespace = cur.fetchone()
    if namespace is None:
        raise SchemaControlError("Memory schema-control namespace probe returned no row")
    if namespace[0] is not None:
        raise SchemaControlError("Memory target adoption refuses partial control state")
    for statement in CONTROL_DDL:
        cur.execute(statement)
    cur.execute(
        sql.SQL(
            "INSERT INTO {}.target_identity (singleton, product, environment, target_id) "
            "VALUES (1, %s, %s, %s::uuid)"
        ).format(sql.Identifier(CONTROL_SCHEMA)),
        (SCHEMA_PRODUCT, environment, target_id),
    )


def grant_control_read(cur: psycopg.Cursor[Any], role: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role):
        raise SchemaControlError("ACTIVEKG_RUNTIME_ROLE is invalid")
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
    if cur.fetchone() is None:
        raise SchemaControlError("restricted Memory runtime role must exist before adoption")
    role_ident = sql.Identifier(role)
    schema_ident = sql.Identifier(CONTROL_SCHEMA)
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema_ident, role_ident))
    cur.execute(
        sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(schema_ident, role_ident)
    )
    cur.execute(
        sql.SQL(
            "REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER "
            "ON ALL TABLES IN SCHEMA {} FROM {}"
        ).format(schema_ident, role_ident)
    )
    cur.execute(
        sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(
            schema_ident, role_ident
        )
    )


def start_attempt(cur: psycopg.Cursor[Any], kind: str, source_commit: str, digest: str) -> int:
    cur.execute(
        sql.SQL(
            "INSERT INTO {}.release_attempts "
            "(kind, source_commit, manifest_digest, outcome) "
            "VALUES (%s, %s, %s, 'running') RETURNING id"
        ).format(sql.Identifier(CONTROL_SCHEMA)),
        (kind, source_commit, digest),
    )
    row = cur.fetchone()
    if row is None:
        raise SchemaControlError("Memory release-attempt insert returned no row")
    return int(row[0])


def finish_attempt(
    cur: psycopg.Cursor[Any], attempt_id: int, outcome: str, error_class: str = ""
) -> None:
    cur.execute(
        sql.SQL(
            "UPDATE {}.release_attempts SET outcome = %s, error_class = %s, "
            "finished_at = clock_timestamp() "
            "WHERE id = %s AND outcome = 'running' AND finished_at IS NULL"
        ).format(sql.Identifier(CONTROL_SCHEMA)),
        (outcome, error_class, attempt_id),
    )
    if cur.rowcount != 1:
        raise SchemaControlError("Memory release-attempt finish compare-and-set failed")


def assert_no_product_row_mutation(cur: psycopg.Cursor[Any]) -> None:
    cur.execute(
        """
        SELECT schemaname, relname
        FROM pg_stat_xact_user_tables
        WHERE schemaname <> %s
          AND (n_tup_ins <> 0 OR n_tup_upd <> 0 OR n_tup_del <> 0)
        ORDER BY schemaname, relname
        """,
        (CONTROL_SCHEMA,),
    )
    if cur.fetchall():
        raise SchemaControlError("schema-control transaction modified a non-control relation")


def assert_startup_schema_ready(
    dsn: str | None = None,
    *,
    require_privacy_hmac: bool = True,
) -> str:
    """Prove schema/control/role readiness before any runtime dependency starts."""

    runtime_dsn = dsn or resolve_runtime_dsn()
    if not runtime_dsn:
        raise SchemaControlError("ACTIVEKG_DSN is not configured")
    # Local import prevents the common module from importing API machinery at
    # module-import time while still sharing the exact /readyz catalog core.
    from activekg.api.auth import JWT_ISSUER, SIGNAL_JWT_ISSUER
    from activekg.api.operational import bounded_readiness_check
    from activekg.privacy.config import (
        candidate_privacy_configuration_problems,
        candidate_privacy_key_versions,
    )

    privacy_problems = candidate_privacy_configuration_problems(
        require_hmac=require_privacy_hmac,
        trusted_flow_issuer=JWT_ISSUER or "",
        trusted_signal_issuer=SIGNAL_JWT_ISSUER or "",
    )
    privacy_versions = (
        candidate_privacy_key_versions() if require_privacy_hmac and not privacy_problems else None
    )

    pool = ConnectionPool(
        runtime_dsn,
        min_size=0,
        max_size=1,
        timeout=2.0,
        open=True,
        kwargs={"connect_timeout": 2},
    )
    try:
        repository = types.SimpleNamespace(pool=pool)
        result = bounded_readiness_check(
            repository,
            unsafe_search_configuration=False,
            jwt_enabled=True,
            jwt_problems=[],
            privacy_problems=privacy_problems,
            privacy_key_versions=privacy_versions,
        )
    finally:
        pool.close()
    if not result.ready:
        raise SchemaControlError(
            "Memory schema readiness refused: " + ",".join(result.reasons or ("unknown",))
        )
    return runtime_dsn
