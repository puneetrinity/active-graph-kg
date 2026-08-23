from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from activekg.common.migration_manifest import CHECKSUM_TRANSITIONS, MIGRATIONS
from activekg.common.schema_control import (
    MigrationRecord,
    SchemaControlError,
    assert_ledger,
    load_migration_records,
    resolve_control_environment,
    resolve_runtime_dsn,
)
from activekg.embedding import worker as embedding_worker
from activekg.extraction import worker as extraction_worker
from scripts import adopt_schema_control, init_railway_db, schema_control_guard

ROOT = Path(__file__).resolve().parents[1]
CONTROL_ENV = {
    "ACTIVEKG_MIGRATE_DSN": "postgresql://owner:secret@127.0.0.1:5432/memory_test",
    "ACTIVEKG_SCHEMA_TARGET_ID": "11111111-1111-4111-8111-111111111111",
    "ACTIVEKG_SCHEMA_ENVIRONMENT": "production",
    "ACTIVEKG_SCHEMA_SOURCE_COMMIT": "0" * 40,
}


def test_migration_files_and_historical_transition_are_frozen() -> None:
    manifest = json.loads((ROOT / "scripts/schema_control_callers.json").read_text())
    assert list(MIGRATIONS) == manifest["migration_manifest"]
    assert CHECKSUM_TRANSITIONS == manifest["checksum_transitions"]
    assert len(MIGRATIONS) == 23 == len(set(MIGRATIONS))
    assert {path.name for path in (ROOT / "db/migrations").glob("*.sql")} == set(
        manifest["migration_files"]
    )
    for name, expected in manifest["migration_files"].items():
        assert hashlib.sha256((ROOT / "db/migrations" / name).read_bytes()).hexdigest() == expected


def test_production_credentials_and_identity_fail_closed_without_fallback() -> None:
    with pytest.raises(SchemaControlError):
        resolve_control_environment({**CONTROL_ENV, "ACTIVEKG_MIGRATE_DSN": ""})
    with pytest.raises(SchemaControlError):
        resolve_runtime_dsn(
            {
                "ACTIVEKG_SCHEMA_ENVIRONMENT": "production",
                "DATABASE_URL": "postgresql://owner:secret@remote/production",
            }
        )
    with pytest.raises(SchemaControlError):
        resolve_runtime_dsn(
            {
                "ACTIVEKG_SCHEMA_ENVIRONMENT": "production",
                "ACTIVEKG_DSN": "postgresql://runtime:secret@remote/production",
                "ACTIVEKG_MIGRATE_DSN": "postgresql://owner:secret@remote/production",
            }
        )
    assert (
        resolve_runtime_dsn(
            {
                "ACTIVEKG_SCHEMA_ENVIRONMENT": "development",
                "DATABASE_URL": "postgresql://local/local_test",
            }
        )
        == "postgresql://local/local_test"
    )


def test_release_and_adoption_flags_refuse_before_connecting() -> None:
    with (
        patch.dict(os.environ, CONTROL_ENV, clear=True),
        patch.object(init_railway_db, "_connect_with_retry") as connect,
        pytest.raises(SystemExit) as release_exit,
    ):
        init_railway_db.main()
    assert release_exit.value.code == 1
    connect.assert_not_called()

    with (
        patch.dict(os.environ, CONTROL_ENV, clear=True),
        patch.object(adopt_schema_control.psycopg, "connect") as connect,
        pytest.raises(SystemExit) as adoption_exit,
    ):
        adopt_schema_control.main()
    assert adoption_exit.value.code == 1
    connect.assert_not_called()


def test_wrong_release_target_refuses_before_manifest_read_or_any_write() -> None:
    class Cursor:
        def __init__(self) -> None:
            self.statements: list[str] = []
            self.last = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, statement, _params=None) -> None:
            self.last = str(statement)
            self.statements.append(self.last)

        def fetchone(self):
            if "to_regnamespace" in self.last:
                return ("activekg_schema_control",)
            raise AssertionError(self.last)

        def fetchall(self):
            if "target_identity" in self.last:
                return [("memory", "production", "22222222-2222-4222-8222-222222222222")]
            raise AssertionError(self.last)

    cursor = Cursor()

    class Connection:
        def cursor(self):
            return cursor

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            return None

    env = {**CONTROL_ENV, "ACTIVEKG_MIGRATION_APPLY": "1"}
    with (
        patch.dict(os.environ, env, clear=True),
        patch.object(init_railway_db, "_connect_with_retry", return_value=Connection()),
        patch.object(init_railway_db, "_load_manifest") as load_manifest,
        pytest.raises(SystemExit),
    ):
        init_railway_db.main()
    load_manifest.assert_not_called()
    mutating = ("CREATE ", "ALTER ", "DROP ", "INSERT ", "UPDATE ", "DELETE ", "TRUNCATE ")
    assert not any(
        any(keyword in statement.upper() for keyword in mutating) for statement in cursor.statements
    )


def test_ledger_requires_exact_ordered_manifest_and_pinned_transition() -> None:
    records = load_migration_records(ROOT)
    rows = [(record.filename, record.checksum, False) for record in records]
    assert_ledger(rows, records, allow_prefix=False)
    historical = list(rows)
    historical_index = [record.filename for record in records].index("016_candidate_rls.sql")
    historical[historical_index] = (
        "016_candidate_rls.sql",
        next(iter(CHECKSUM_TRANSITIONS["016_candidate_rls.sql"])),
        False,
    )
    assert_ledger(historical, records, allow_prefix=False)
    with pytest.raises(SchemaControlError):
        assert_ledger(rows[:-1], records, allow_prefix=False)
    with pytest.raises(SchemaControlError):
        assert_ledger([*rows, ("999_unknown.sql", "0" * 64, False)], records, allow_prefix=True)
    changed = list(rows)
    changed[0] = (changed[0][0], "0" * 64, False)
    with pytest.raises(SchemaControlError):
        assert_ledger(changed, records, allow_prefix=False)


def test_all_runtime_entrypoints_admit_schema_before_dependencies() -> None:
    start = (ROOT / "scripts/start_railway.sh").read_text()
    assert start.index("schema_ready.py") < start.index("uvicorn")
    assert "init_railway_db.py" not in start
    assert "unset DATABASE_URL" in start

    embedding = (
        (ROOT / "activekg/embedding/worker.py").read_text().split("def start_worker()", 1)[1]
    )
    assert embedding.index(
        "dsn = assert_startup_schema_ready(require_privacy_hmac=False)"
    ) < embedding.index("redis_client = get_redis_client()")
    extraction = (
        (ROOT / "activekg/extraction/worker.py")
        .read_text()
        .split("def start_extraction_worker()", 1)[1]
    )
    readiness = extraction.index("dsn = assert_startup_schema_ready(require_privacy_hmac=False)")
    for dependency in (
        "assert_extraction_models_configured()",
        'groq_key = os.getenv("GROQ_API_KEY")',
        "redis_client = get_redis_client()",
    ):
        assert readiness < extraction.index(dependency)


def test_workers_refuse_before_queue_or_provider_construction() -> None:
    with (
        patch(
            "activekg.common.schema_control.assert_startup_schema_ready",
            side_effect=SchemaControlError("not ready"),
        ),
        patch("activekg.common.metrics.get_redis_client") as redis_client,
        pytest.raises(SystemExit) as embedding_exit,
    ):
        embedding_worker.start_worker()
    assert embedding_exit.value.code == 1
    redis_client.assert_not_called()

    with (
        patch(
            "activekg.common.schema_control.assert_startup_schema_ready",
            side_effect=SchemaControlError("not ready"),
        ),
        patch("activekg.common.metrics.get_redis_client") as redis_client,
        patch.object(extraction_worker, "assert_extraction_models_configured") as models,
        patch.object(extraction_worker, "start_healthcheck_server") as health,
        pytest.raises(SystemExit) as extraction_exit,
    ):
        extraction_worker.start_extraction_worker()
    assert extraction_exit.value.code == 1
    redis_client.assert_not_called()
    models.assert_not_called()
    health.assert_not_called()


def test_guard_is_green_and_mutations_restore_from_saved_bytes(tmp_path: Path) -> None:
    assert schema_control_guard.check(ROOT) == []
    copied = tmp_path / "repository"
    shutil.copytree(
        ROOT,
        copied,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"),
    )
    start = copied / "scripts/start_railway.sh"
    saved = start.read_bytes()
    start.write_text(start.read_text().replace("schema_ready.py", "init_railway_db.py"))
    findings = schema_control_guard.check(copied)
    assert any("startup is not readiness-only" in finding for finding in findings)
    start.write_bytes(saved)
    assert start.read_bytes() == saved

    descriptor = copied / "railway.schema-release.json"
    descriptor_saved = descriptor.read_bytes()
    descriptor.write_text(
        descriptor.read_text().replace(
            '"restartPolicyType": "NEVER"',
            '"restartPolicyType": "NEVER", "restartPolicyMaxRetries": 0',
        )
    )
    findings = schema_control_guard.check(copied)
    assert any("one-shot/no-healthcheck" in finding for finding in findings)
    descriptor.write_bytes(descriptor_saved)
    assert descriptor.read_bytes() == descriptor_saved

    workflow = copied / ".github/workflows/ci.yml"
    workflow_saved = workflow.read_bytes()
    workflow.write_text(
        workflow.read_text().replace(
            "postgresql://activekg_unit_test:activekg@127.0.0.1:5432/memory_unit_test",
            "postgresql://activekg:activekg@127.0.0.1:5432/memory_unit_test",
        )
    )
    findings = schema_control_guard.check(copied)
    assert any("CI fresh-init migration role is not disposable" in finding for finding in findings)
    workflow.write_bytes(workflow_saved)
    assert workflow.read_bytes() == workflow_saved

    workflow.write_text(workflow.read_text().replace("fetch-depth: 0", "fetch-depth: 1", 1))
    findings = schema_control_guard.check(copied)
    assert any("pinned-base tests use a shallow checkout" in finding for finding in findings)
    workflow.write_bytes(workflow_saved)
    assert workflow.read_bytes() == workflow_saved

    scoring_workflow = copied / ".github/workflows/test-scoring-modes.yml"
    scoring_saved = scoring_workflow.read_bytes()
    scoring_workflow.write_text(
        scoring_workflow.read_text().replace(
            "    services:\n",
            "    services:\n      postgres:\n        image: pgvector/pgvector:pg16\n",
            1,
        )
    )
    findings = schema_control_guard.check(copied)
    assert any("fresh-init PostgreSQL is not runner-local" in finding for finding in findings)
    scoring_workflow.write_bytes(scoring_saved)
    assert scoring_workflow.read_bytes() == scoring_saved

    retired = copied / "scripts/db_bootstrap.sh"
    retired.write_text('#!/bin/sh\npsql "$DATABASE_URL"\n')
    findings = schema_control_guard.check(copied)
    assert any("retired scripts/db_bootstrap.sh" in finding for finding in findings)


def test_migration_record_shape_does_not_duplicate_the_business_ledger() -> None:
    record = MigrationRecord("001.sql", "a" * 64)
    assert tuple(record.__dataclass_fields__) == ("filename", "checksum")
