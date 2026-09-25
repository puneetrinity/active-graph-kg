#!/usr/bin/env python3
"""Static 4E boundary checks; PostgreSQL and cross-system proofs remain required."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "db/migrations/029_organization_candidate_history.sql"
SOURCES = (
    ".github/workflows/ci.yml",
    MIGRATION,
    "scripts/schema_control_callers.json",
    "activekg/candidate_history/repository.py",
    "activekg/candidate_history/worker.py",
    "activekg/api/organization_candidate_history.py",
)


class GuardError(RuntimeError):
    pass


def validate_sources(sources: dict[str, str]) -> None:
    workflow = sources[".github/workflows/ci.yml"]
    caps = {
        "fast-unit-tests": 20,
        "unit-tests": 40,
        "deploy-path-db": 60,
        "integration-tests": 45,
        "benchmarks": 45,
        "candidate-index-image": 45,
    }
    for block in re.split(r"(?=^  [A-Za-z0-9_-]+:\s*$)", workflow, flags=re.MULTILINE):
        job = re.match(r"  ([A-Za-z0-9_-]+):\s*\n", block)
        if not job or "    runs-on:" not in block:
            continue
        timeout = re.search(r"^    timeout-minutes: ([0-9]+)$", block, re.MULTILINE)
        if not timeout or int(timeout[1]) != caps.get(job[1], 20):
            raise GuardError("history_ci_job_timeout")
    migration = sources[MIGRATION]
    manifest = json.loads(sources["scripts/schema_control_callers.json"])
    if manifest["migration_manifest"][-1] != Path(MIGRATION).name:
        raise GuardError("history_ledger_tail")
    if (
        hashlib.sha256(migration.encode()).hexdigest()
        != manifest["migration_files"][Path(MIGRATION).name]
    ):
        raise GuardError("history_migration_digest")
    required = (
        "FORCE ROW LEVEL SECURITY",
        "ON DELETE RESTRICT",
        "HISTORY_BINDING_APPEND_ONLY",
        "SET search_path=pg_catalog,public",
        "SET statement_timeout='3s'",
        "SET lock_timeout='500ms'",
        "SET idle_in_transaction_session_timeout='5s'",
        "public.candidate_index_status(s.source_id,true)",
        "set_config('app.current_tenant_id',tenant_key,true)",
        "set_config('app.current_tenant_id',p_tenant,true)",
        "set_config('app.current_tenant_id',coalesce(prior_scope,''),true)",
        "tenant_id=tenant_key AND application_id=e.subject_id",
        "r.job_id<>e.job_id",
        "s.source_kind<>'organization_application'",
        "s.reference_id IS DISTINCT FROM r.reference_id",
        "i.resume_version_id<>v.resume_version_id",
        "pg_try_advisory_xact_lock",
        "och_inbox_owner_read",
        "och_stream_owner_read",
    )
    if any(token not in migration for token in required):
        raise GuardError("history_sql_authority")
    if len(re.findall(r"CREATE TABLE public\.organization_candidate_history_", migration)) != 5:
        raise GuardError("history_table_inventory")
    write_targets = re.findall(
        r"\b(?:INSERT\s+INTO|UPDATE(?!\s+(?:SET|OR|ON|OF)\b)|DELETE\s+FROM)\s+(?:public\.)?([a-z_]+)",
        migration,
        re.IGNORECASE,
    )
    allowed_writes = {
        "organization_candidate_history_" + suffix
        for suffix in ("bindings", "event_state", "applications", "subjects", "scan_state")
    }
    if any(target.lower() not in allowed_writes for target in write_targets):
        raise GuardError("history_forbidden_write")
    if migration.count("SECURITY DEFINER") != 2 or migration.count("FROM PUBLIC;") != 3:
        raise GuardError("history_routine_inventory")
    for name in re.findall(r"(?:CONSTRAINT|CREATE INDEX|CREATE TRIGGER)\s+(\w+)", migration):
        if not name.startswith("och_") or len(name.encode()) > 63:
            raise GuardError("history_catalog_name")
    repository = sources["activekg/candidate_history/repository.py"]
    worker = sources["activekg/candidate_history/worker.py"]
    api = sources["activekg/api/organization_candidate_history.py"]
    for source in (repository, worker):
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                allowed_imports = {
                    "__future__",
                    "json",
                    "hashlib",
                    "typing",
                    "psycopg",
                    "asyncio",
                    "time",
                    "concurrent.futures",
                    "activekg.candidate_history.contracts",
                    "activekg.candidate_history.repository",
                }
                if any(name not in allowed_imports for name in modules):
                    raise GuardError("history_engine_external_io")
    for token in (
        "SET TRANSACTION READ ONLY",
        "connect_timeout=3",
        "readonly=True",
        "readonly=False",
    ):
        if token not in repository:
            raise GuardError("history_transaction_contract")
    for token in (
        "ThreadPoolExecutor(max_workers=1",
        "await asyncio.shield(future)",
        "0 <= age <= 30",
        "await self._task",
    ):
        if token not in worker:
            raise GuardError("history_worker_contract")
    for token in (
        '"decision-history:read"',
        "Depends(require_history_reader)",
        'claims.tenant_id != f"org_{command.organization_id}"',
        "MAX_BODY_BYTES",
    ):
        if token not in api:
            raise GuardError("history_api_authority")


def validate(root: Path = ROOT) -> None:
    validate_sources({path: (root / path).read_text() for path in SOURCES})
    for path in root.glob("tests/test_organization_candidate_history*.py"):
        validate_parameter_ids(path.read_text())


def validate_parameter_ids(source: str) -> None:
    tree = ast.parse(source)
    bindings = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }

    def large(node, seen=frozenset()):
        if isinstance(node, ast.Constant):
            return isinstance(node.value, bytes) or (
                isinstance(node.value, str) and len(node.value.encode()) >= 1024
            )
        if isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
            return large(bindings[node.id], seen | {node.id})
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                value, times = node.left.value, node.right.value
                if isinstance(value, (str, bytes)) and isinstance(times, int):
                    return isinstance(value, bytes) or len(value.encode()) * max(0, times) >= 1024
        return any(large(child, seen) for child in ast.iter_child_nodes(node))

    for call in ast.walk(tree):
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.attr != "parametrize"
            or len(call.args) < 2
        ):
            continue
        if not large(call.args[1]):
            continue
        if any(
            k.arg == "ids" and not (isinstance(k.value, ast.Constant) and k.value.value is None)
            for k in call.keywords
        ):
            continue
        values = call.args[1]
        if isinstance(values, (ast.List, ast.Tuple)) and all(
            isinstance(case, ast.Call)
            and isinstance(case.func, ast.Attribute)
            and case.func.attr == "param"
            and any(k.arg == "id" for k in case.keywords)
            for case in values.elts
        ):
            continue
        raise GuardError("history_large_parameter_requires_explicit_ids")


def collect_bounded_ids() -> None:
    import pytest

    class Bounds:
        count = 0

        def pytest_collection_modifyitems(self, items):
            self.count = len(items)
            for position, item in enumerate(items):
                length = len(item.nodeid.encode())
                if length > 512:
                    raise pytest.UsageError(f"node-id limit: item={position}, bytes={length}")

    bounds = Bounds()
    result = pytest.main(
        [
            "--collect-only",
            "-p",
            "no:terminal",
            "tests/test_organization_candidate_history.py",
            "tests/test_organization_candidate_history_postgres.py",
            "tests/test_organization_candidate_history_guard.py",
            "tests/test_candidate_index.py",
        ],
        plugins=[bounds],
    )
    if result != 0 or bounds.count == 0:
        raise GuardError("history_collection_failed")
    print(f"history-node-id-census: OK ({bounds.count} items, <=512 bytes)")


if __name__ == "__main__":
    try:
        validate()
        if "--collect" in sys.argv:
            collect_bounded_ids()
    except (GuardError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(
            f"organization-candidate-history-guard: REFUSED ({type(exc).__name__})"
        ) from None
    print("organization-candidate-history-guard: OK")
