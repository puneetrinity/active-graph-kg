#!/usr/bin/env python3
"""CI guard for Memory's single schema authority and complete caller census."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path("scripts/schema_control_callers.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return ast.literal_eval(node.value)
    raise ValueError(f"missing literal assignment: {name}")


def check(root: Path) -> list[str]:
    findings: list[str] = []
    manifest_file = root / MANIFEST_PATH
    if not manifest_file.is_file():
        return [f"missing caller manifest: {MANIFEST_PATH}"]
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"invalid caller manifest: {type(exc).__name__}"]

    for group, paths in manifest.get("runtime_callers", {}).items():
        for relative in paths:
            if not (root / relative).is_file():
                findings.append(f"declared {group} caller missing: {relative}")

    pinned_migrations = manifest.get("migration_files", {})
    actual_migrations = {
        path.name for path in (root / "db" / "migrations").glob("*.sql") if path.is_file()
    }
    if actual_migrations != set(pinned_migrations):
        findings.append("migration-file set differs from the frozen caller manifest")
    for filename, expected in pinned_migrations.items():
        path = root / "db" / "migrations" / filename
        if not path.is_file() or _sha256(path) != expected:
            findings.append(f"frozen migration changed: {filename}")
    for relative, expected in manifest.get("baseline_assets", {}).items():
        path = root / relative
        if not path.is_file() or _sha256(path) != expected:
            findings.append(f"frozen baseline asset changed: {relative}")

    migration_manifest_path = root / "activekg/common/migration_manifest.py"
    try:
        tree = ast.parse(migration_manifest_path.read_text(encoding="utf-8"))
        ordered = list(_assignment(tree, "MIGRATIONS"))
        transitions = _assignment(tree, "CHECKSUM_TRANSITIONS")
    except (OSError, SyntaxError, ValueError) as exc:
        findings.append(f"migration authority is not statically readable: {type(exc).__name__}")
    else:
        if ordered != manifest.get("migration_manifest"):
            findings.append("ordered migration manifest changed outside the locked authority")
        if transitions != manifest.get("checksum_transitions"):
            findings.append("checksum-transition allowlist changed outside the locked authority")

    if (root / "scripts/db_bootstrap.sh").exists():
        findings.append("retired scripts/db_bootstrap.sh authority exists")
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    if "db-bootstrap" in makefile or "db_bootstrap.sh" in makefile:
        findings.append("retired Make db-bootstrap authority exists")

    start = (root / "scripts/start_railway.sh").read_text(encoding="utf-8")
    if "schema_ready.py" not in start or "init_railway_db.py" in start:
        findings.append("API startup is not readiness-only")
    if start.find("schema_ready.py") > start.find("uvicorn"):
        findings.append("API readiness does not precede Uvicorn")
    for forbidden in (
        "ACTIVEKG_MIGRATE_DSN",
        "ACTIVEKG_RUNTIME_PASSWORD",
        "ACTIVEKG_MIGRATION_APPLY",
        "ACTIVEKG_SCHEMA_ADOPT_EXISTING",
    ):
        if forbidden not in start:
            findings.append(f"API startup does not strip {forbidden}")

    worker_requirements = {
        "activekg/embedding/worker.py": (
            "def start_worker()",
            "dsn = assert_startup_schema_ready()",
            "redis_client =",
        ),
        "activekg/extraction/worker.py": (
            "def start_extraction_worker()",
            "dsn = assert_startup_schema_ready()",
            "assert_extraction_models_configured()",
        ),
    }
    for relative, (entrypoint, readiness, first_dependency) in worker_requirements.items():
        content = (root / relative).read_text(encoding="utf-8")
        content = content[content.find(entrypoint) :]
        if readiness not in content or content.find(readiness) > content.find(first_dependency):
            findings.append(f"runtime dependency starts before readiness: {relative}")

    release = (root / "scripts/init_railway_db.py").read_text(encoding="utf-8")
    for required in (
        "resolve_control_environment()",
        'ACTIVEKG_MIGRATION_APPLY") != "1"',
        "assert_identity(cur, control.target_id, control.environment)",
        "ACTIVEKG_ALLOW_MIGRATION_DRIFT is forbidden in production",
    ):
        if required not in release:
            findings.append(f"release fail-closed contract missing: {required}")
    for fallback in (
        'os.environ.get("ACTIVEKG_DSN")',
        'os.getenv("ACTIVEKG_DSN")',
        'os.environ.get("DATABASE_URL")',
        'os.getenv("DATABASE_URL")',
    ):
        if fallback in release:
            findings.append(f"release credential fallback restored: {fallback}")

    descriptor = json.loads((root / "railway.schema-release.json").read_text(encoding="utf-8"))
    deploy = descriptor.get("deploy", {})
    if deploy.get("startCommand") != "python /app/scripts/init_railway_db.py":
        findings.append("manual release service command changed")
    if deploy.get("restartPolicyType") != "NEVER" or "healthcheckPath" in deploy:
        findings.append("manual release service is not a one-shot/no-healthcheck service")

    ignored = {
        MANIFEST_PATH.as_posix(),
        "scripts/schema_control_guard.py",
        "scripts/init_railway_db.py",
        "scripts/adopt_schema_control.py",
        "activekg/common/migration_manifest.py",
    }
    allowed_release_reference = {
        ".env.example",
        ".github/workflows/ci.yml",
        "README.md",
        "enable_rls_policies.sql",
        "railway.schema-release.json",
        "scripts/README.md",
    }
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if (
            relative in ignored
            or relative.startswith("tests/")
            or relative.startswith("docs/")
            or relative.startswith("site/")
        ):
            continue
        if relative.endswith((".pyc", ".lock")) or "__pycache__" in path.parts:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "init_railway_db.py" in content and relative not in allowed_release_reference:
            findings.append(f"undeclared release caller/reference: {relative}")
        if "db_bootstrap.sh" in content:
            findings.append(f"retired bootstrap reference: {relative}")

    ci = (root / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    if "python scripts/schema_control_guard.py" not in ci:
        findings.append("CI does not enforce the schema-control guard")
    return sorted(set(findings))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    findings = check(args.root.resolve())
    if findings:
        for finding in findings:
            print(f"SCHEMA_CONTROL_GUARD: {finding}", file=sys.stderr)
        raise SystemExit(1)
    print("schema-control guard: OK")


if __name__ == "__main__":
    main()
