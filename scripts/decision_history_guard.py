#!/usr/bin/env python3
"""Guard the bounded organization decision-history receiver surface."""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from pathlib import Path

EXPECTED_LEGACY_FEEDBACK_SHA256 = "db8daece024b6ae34d0faaee6d3498eb58dccfd15484271be1aca8d49a1facee"
EXPECTED_LEGACY_MIGRATION_SHA256 = (
    "f8ef4063e1dd69fe38ea9e0dcf6e79edaf70fc130fa78254c92e8ca66b495262"
)
RECEIVER = Path("activekg/api/organization_decision_events.py")
MAIN = Path("activekg/api/main.py")
WORKERS = (Path("activekg/embedding/worker.py"), Path("activekg/extraction/worker.py"))
FORBIDDEN_IMPORT_PREFIXES = (
    "activekg.api.global_memory",
    "activekg.api.candidate_privacy",
    "activekg.graph",
    "activekg.embedding",
    "activekg.extraction",
    "activekg.privacy",
    "activekg.providers",
)
FORBIDDEN_RECEIVER_TOKENS = (
    "canonical_candidate",
    "signal_candidate",
    "email",
    "phone",
    "resume",
    "contact",
    "embedding",
    "vector",
    "narrative",
)


class GuardError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _imports(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node) or ""
    raise GuardError(f"required frozen function missing: {name}")


def validate(root: Path) -> None:
    receiver_path = root / RECEIVER
    main_path = root / MAIN
    if not receiver_path.is_file():
        raise GuardError("decision receiver module missing")
    receiver = receiver_path.read_text(encoding="utf-8")
    receiver_tree = ast.parse(receiver, filename=str(receiver_path))

    imports = _imports(receiver_tree)
    if any(name.startswith(FORBIDDEN_IMPORT_PREFIXES) for name in imports):
        raise GuardError("decision receiver imports a forbidden candidate/provider/worker module")
    lowered = receiver.lower()
    if any(token in lowered for token in FORBIDDEN_RECEIVER_TOKENS):
        raise GuardError("decision receiver admits candidate identity or enrichment vocabulary")
    required_receiver = (
        'router.post("/organization-decision-events/ingest"',
        'os.getenv("ORG_DECISION_INBOX_ENABLED", "false") == "true"',
        '"decision-history:write"',
        "claims.issuer != auth.JWT_ISSUER",
        'claims.actor_type != "service"',
        'f"org_{payload.organization_id}"',
        'ConfigDict(extra="forbid", strict=True)',
        "request.stream()",
        "pg_advisory_xact_lock",
        "conn.commit()",
        "conn.rollback()",
        "conn.close()",
    )
    if any(marker not in receiver for marker in required_receiver):
        raise GuardError("decision receiver contract drifted")

    main = main_path.read_text(encoding="utf-8")
    import_marker = "from activekg.api.organization_decision_events import ("
    include_marker = "app.include_router(organization_decision_events_router)"
    if main.count(import_marker) != 1 or main.count(include_marker) != 1:
        raise GuardError("decision receiver must have exactly one import and registration")
    if "organization_decision_events" in (root / "activekg/api/global_memory.py").read_text():
        raise GuardError("decision receiver entered the legacy global-memory router")
    for worker in WORKERS:
        if "organization_decision_events" in (root / worker).read_text(encoding="utf-8"):
            raise GuardError("decision receiver imported by a worker")

    legacy = _function_source(root / "activekg/api/global_memory.py", "ingest_feedback_events")
    if _sha256(legacy.encode("utf-8")) != EXPECTED_LEGACY_FEEDBACK_SHA256:
        raise GuardError("legacy feedback receiver drifted")
    if _sha256((root / "db/migrations/012_global_memory.sql").read_bytes()) != (
        EXPECTED_LEGACY_MIGRATION_SHA256
    ):
        raise GuardError("legacy feedback migration drifted")

    migration = (root / "db/migrations/024_organization_decision_event_inbox.sql").read_text(
        encoding="utf-8"
    )
    required_migration = (
        "ENABLE ROW LEVEL SECURITY",
        "FORCE ROW LEVEL SECURITY",
        "current_setting('app.current_tenant_id', true)",
        "REVOKE ALL ON organization_decision_event_inbox FROM PUBLIC",
        "REVOKE ALL ON organization_decision_stream_state FROM PUBLIC",
    )
    force_lines = (
        "ALTER TABLE organization_decision_event_inbox FORCE ROW LEVEL SECURITY;",
        "ALTER TABLE organization_decision_stream_state FORCE ROW LEVEL SECURITY;",
    )
    if any(migration.count(line) != 1 for line in force_lines) or any(
        marker not in migration for marker in required_migration
    ):
        raise GuardError("decision inbox RLS contract drifted")
    if any(token in migration.lower() for token in FORBIDDEN_RECEIVER_TOKENS):
        raise GuardError("decision inbox schema admits candidate identity or enrichment vocabulary")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        validate(args.root.resolve())
    except (GuardError, OSError, SyntaxError) as exc:
        print(f"DECISION_HISTORY_GUARD_REFUSED: {exc}", file=sys.stderr)
        return 1
    print("DECISION_HISTORY_GUARD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
