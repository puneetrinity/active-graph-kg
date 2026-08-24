#!/usr/bin/env python3
"""Fail closed when a candidate/privacy surface escapes the checked-in census."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "activekg" / "privacy" / "surfaces.json"

GOVERNED_TABLES = (
    "global_candidates",
    "candidates",
    "candidate_identifiers",
    "candidate_source_records",
    "candidate_provenance",
    "tenant_candidate_access",
    "nodes",
    "edges",
    "events",
    "embedding_history",
    "candidate_privacy_directive_events",
    "candidate_privacy_directives",
    "candidate_privacy_identity_tokens",
)
GOVERNED_CALLS = (
    "CandidateRepository",
    "upsert_global_candidate",
    "upsert_signal_candidate_to_global",
    "sync_applicant_to_global_memory",
    "_mirror_signal_candidate_to_global",
    "update_node_embedding",
    "write_embedding_history",
    "append_event",
    "enqueue_embedding_job",
    "enqueue_extraction_job",
    "_update_node_props",
    "_trigger_reembed",
    "_privacy_filtered_vector_rows",
)
ROUTE_MARKERS = ("candidate", "profile", "resume")
ROUTE_DECORATORS = {"get", "post", "put", "patch", "delete"}
GOVERNED_ROUTES = {
    ("get", "/nodes"),
    ("post", "/nodes"),
    ("post", "/nodes/batch"),
    ("get", "/nodes/by-external-id"),
    ("get", "/nodes/{node_id}"),
    ("post", "/nodes/{node_id}/refresh"),
    ("get", "/nodes/{node_id}/versions"),
    ("post", "/upload"),
    ("post", "/search"),
    ("post", "/edges"),
    ("get", "/events"),
    ("get", "/lineage/{node_id}"),
    ("post", "/admin/refresh"),
    ("get", "/admin/embedding/status"),
    ("post", "/admin/embedding/requeue"),
    ("get", "/admin/extraction/status"),
    ("post", "/admin/extraction/requeue"),
    ("get", "/admin/anomalies"),
}
RAW_OUTPUT_PATTERNS = (
    re.compile(r"\bprint\s*\("),
    re.compile(r"\b(?:logger|logging)\s*\."),
)


class GuardError(RuntimeError):
    pass


@dataclass(frozen=True)
class Reference:
    file: str
    symbol: str
    tables: tuple[str, ...]
    calls: tuple[str, ...]
    routes: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.file}::{self.symbol}"

    def projection(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "symbol": self.symbol,
            "tables": list(self.tables),
            "calls": list(self.calls),
            "routes": list(self.routes),
        }


def _function_blocks(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    blocks: list[tuple[str, ast.AST]] = [("<module>", tree)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            blocks.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    blocks.append((f"{node.name}.{child.name}", child))
    return blocks


def _route_paths(node: ast.AST) -> tuple[str, ...]:
    decorators = getattr(node, "decorator_list", ())
    paths: list[str] = []
    for decorator in decorators:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            continue
        if decorator.func.attr not in ROUTE_DECORATORS or not decorator.args:
            continue
        first = decorator.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            path = first.value
            if (
                any(marker in path.lower() for marker in ROUTE_MARKERS)
                or (decorator.func.attr, path) in GOVERNED_ROUTES
            ):
                paths.append(f"{decorator.func.attr.upper()} {path}")
    return tuple(sorted(set(paths)))


def _text_for(path: Path, node: ast.AST, source: str) -> str:
    if isinstance(node, ast.Module):
        # Module scope is limited to imports and statements outside functions;
        # otherwise every reference would be duplicated under <module>.
        chunks: list[str] = []
        for child in node.body:
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                segment = ast.get_source_segment(source, child)
                if segment:
                    chunks.append(segment)
        return "\n".join(chunks)
    return ast.get_source_segment(source, node) or ""


def _reference_source(reference: Reference) -> str:
    path = ROOT / reference.file
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=reference.file)
    for symbol, node in _function_blocks(tree):
        if symbol == reference.symbol:
            return _text_for(path, node, source)
    raise GuardError(f"candidate privacy source block is missing: {reference.key}")


_FENCE_ANCHORS: dict[str, tuple[str, ...]] = {
    "privacy-api-contract": ("require_candidate_privacy_", "_repo()"),
    "privacy-candidate-fences": (
        "_require_privacy_allowed",
        "candidate_privacy_candidate_decision",
    ),
    "privacy-global-fences": (
        "_require_privacy_allowed",
        "_privacy_decision",
        "candidate_privacy_",
    ),
    "privacy-node-fences": (
        "candidate_privacy_node_decision",
        "_assert_candidate_values_allowed",
        "_require_candidate_node_write_allowed",
        "_require_candidate_ingest_allowed",
    ),
    "privacy-postgres-authority": ("candidate_privacy_directive_events",),
    "privacy-readiness-contract": ("candidate_privacy_",),
    "privacy-vector-recall": ("privacy_filtered_nodes AS MATERIALIZED",),
    "privacy-stale-worker-fences": (
        "node_decision",
        "_privacy_decision",
        "enforce_privacy",
    ),
    "privacy-surface-dependency": (
        "candidate_privacy_",
        "_require_candidate_ingest_allowed",
        "_execute_candidate_resolve",
        "privacy",
        "get_node",
        "get_node_by_external_id",
        "list_nodes",
        "vector_search",
        "hybrid_search",
        "append_event",
        "find_nodes_due_for_refresh",
        "search_candidates_by_signal_tags",
        "search_tenant_private_candidates",
        "detect_drift_spikes",
        "detect_trigger_storms",
        "detect_scheduler_lag",
    ),
}


def _validate_fence_anchor(reference: Reference, row: dict[str, Any]) -> None:
    source = _reference_source(reference)
    if reference.key.endswith("repository.py::GraphRepository._privacy_filtered_vector_rows"):
        exact_rescan_required = (
            "if len(rows) >= limit:",
            "WITH privacy_filtered_nodes AS MATERIALIZED",
            "WHERE embedding IS NOT NULL{where_sql}",
            "ORDER BY embedding {op} %s",
        )
        if any(value not in source for value in exact_rescan_required):
            raise GuardError("privacy vector exact-rescan boundary was weakened")
        exact_start = source.index("WITH privacy_filtered_nodes AS MATERIALIZED")
        if source.index("WHERE embedding IS NOT NULL{where_sql}", exact_start) > source.index(
            "LIMIT %s", exact_start
        ):
            raise GuardError("privacy vector filter moved after the exact-rescan limit")
    if reference.key.endswith("api/main.py::create_node"):
        if source.index("_require_candidate_node_write_allowed(") > source.index(
            "repo.create_node("
        ):
            raise GuardError("single-node privacy fence moved after the write")
    if reference.key.endswith("api/main.py::create_nodes_batch"):
        if source.index("_require_candidate_node_write_allowed(") > source.index(
            "repo.create_node("
        ):
            raise GuardError("batch-node privacy fence moved after the write")
    if reference.key.endswith("api/main.py::upload_files"):
        upload_required = (
            "_candidate_upload_identifiers(text)",
            "_require_candidate_ingest_allowed(",
            "chunk_ids = create_chunk_nodes(",
        )
        if any(value not in source for value in upload_required):
            raise GuardError("upload privacy fence is incomplete")
        if max(source.index(upload_required[0]), source.index(upload_required[1])) > source.index(
            upload_required[2]
        ):
            raise GuardError("upload privacy fence moved after chunk creation")
    if reference.key.endswith("global_memory.py::_find_existing_all"):
        module_source = (ROOT / reference.file).read_text(encoding="utf-8")
        if module_source.count("_find_existing_all(") != 5:
            raise GuardError("global identity lookup caller census changed")
        return
    if reference.key.endswith("::public_candidate_exclusions"):
        if "public_crustdata_person_id" not in source or "SELECT gc.*" in source:
            raise GuardError("public exclusion projection exposes candidate data")
        return
    anchors = _FENCE_ANCHORS.get(row["test_id"])
    if anchors is None or not any(anchor in source for anchor in anchors):
        raise GuardError(f"candidate privacy enforcement anchor missing: {reference.key}")
    if reference.key.endswith("embedding/worker.py::EmbeddingWorker._process_job"):
        if source.count("self.privacy_repository.node_decision") < 3:
            raise GuardError("embedding worker stale-job privacy recheck is missing")
    if reference.key.endswith("extraction/worker.py::ExtractionWorker._process_job"):
        if source.count("self._privacy_decision") < 3:
            raise GuardError("extraction worker stale-job privacy recheck is missing")


def _sql_text(node: ast.AST) -> str:
    values: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            value = child.value
            if re.search(
                r"\b(?:select|insert|update|delete|from|join|table|references|truncate)\b",
                value,
                re.IGNORECASE,
            ):
                values.append(value)
        elif isinstance(child, ast.JoinedStr):
            value = "".join(
                part.value
                for part in child.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if re.search(
                r"\b(?:select|insert|update|delete|from|join|table|references|truncate)\b",
                value,
                re.IGNORECASE,
            ):
                values.append(value)
    return "\n".join(values)


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        function = child.func
        if isinstance(function, ast.Name):
            names.add(function.id)
        elif isinstance(function, ast.Attribute):
            names.add(function.attr)
    if isinstance(node, ast.Module):
        for child in node.body:
            if isinstance(child, ast.ImportFrom):
                # Queue functions are governed at their call sites. Merely
                # importing them is not a candidate-data publication surface.
                names.update(
                    alias.name
                    for alias in child.names
                    if alias.name not in {"enqueue_embedding_job", "enqueue_extraction_job"}
                )
    return names


def discover() -> list[Reference]:
    references: list[Reference] = []
    for path in sorted((ROOT / "activekg").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        for symbol, node in _function_blocks(tree):
            scope_node = node
            if isinstance(node, ast.Module):
                scope_node = ast.Module(
                    body=[
                        child
                        for child in node.body
                        if not isinstance(
                            child,
                            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                        )
                    ],
                    type_ignores=[],
                )
            sql_text = _sql_text(scope_node)
            tables = tuple(
                table
                for table in GOVERNED_TABLES
                if re.search(
                    rf"(?<![a-zA-Z0-9_]){re.escape(table)}(?![a-zA-Z0-9_])",
                    sql_text,
                )
            )
            called_names = _called_names(scope_node)
            calls = tuple(call for call in GOVERNED_CALLS if call in called_names)
            routes = _route_paths(node)
            if tables or calls or routes:
                references.append(
                    Reference(
                        file=relative,
                        symbol=symbol,
                        tables=tuple(sorted(tables)),
                        calls=tuple(sorted(calls)),
                        routes=routes,
                    )
                )
    return references


def _load_manifest() -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GuardError(f"candidate privacy manifest contains duplicate key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            MANIFEST.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise GuardError("candidate privacy surface manifest is missing or malformed") from exc
    if not isinstance(value, dict) or value.get("version") != 1:
        raise GuardError("candidate privacy surface manifest version is invalid")
    return value


def _validate_raw_output_ban() -> None:
    paths = [
        ROOT / "activekg" / "privacy" / "config.py",
        ROOT / "activekg" / "privacy" / "identity.py",
        ROOT / "activekg" / "privacy" / "models.py",
        ROOT / "activekg" / "privacy" / "repository.py",
        ROOT / "activekg" / "api" / "candidate_privacy.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if any(pattern.search(source) for pattern in RAW_OUTPUT_PATTERNS):
            raise GuardError(f"raw-output primitive is forbidden in {path.relative_to(ROOT)}")
    api_source = paths[-1].read_text(encoding="utf-8")
    repository_source = paths[-2].read_text(encoding="utf-8")
    required_minimal_response = {
        '"request_id"',
        '"directive_id"',
        '"action"',
        '"scope"',
        '"state"',
        '"version"',
        '"effective_at"',
        '"decision"',
    }
    response_block = api_source[
        api_source.index("def _response(") : api_source.index("@router.post")
    ]
    if any(field not in response_block for field in required_minimal_response):
        raise GuardError("candidate privacy directive response projection was weakened")
    for forbidden in ('"canonical"', '"tenant_id"', '"evidence_ref"', '"identifiers"', '"token"'):
        if forbidden in response_block:
            raise GuardError("candidate privacy directive response exposes a forbidden field")
    eligibility_block = api_source[
        api_source.index("async def eligibility_batch(") : api_source.index(
            '@router.get("/candidate-privacy/changes"'
        )
    ]
    if "repository.evaluate_many(prepared)" not in eligibility_block:
        raise GuardError("candidate privacy eligibility lost its one-call batch boundary")
    evaluate_many_block = repository_source[
        repository_source.index("    def evaluate_many(") : repository_source.index(
            "    def canonical_decision("
        )
    ]
    set_based_required = (
        "jsonb_array_elements",
        "candidate_privacy_resolve_subject(",
        "candidate_privacy_resolve_canonical(",
        "candidate_privacy_match(",
    )
    if (
        evaluate_many_block.count("with self._conn()") != 1
        or evaluate_many_block.count("cur.execute(") != 3
        or "self.evaluate(" in evaluate_many_block
        or "self._evaluate_on_cursor(" in evaluate_many_block
        or any(anchor not in evaluate_many_block for anchor in set_based_required)
    ):
        raise GuardError(
            "candidate privacy eligibility batch lost its three-query set-based boundary"
        )


def validate() -> None:
    manifest = _load_manifest()
    rows = manifest.get("references")
    if not isinstance(rows, list):
        raise GuardError("candidate privacy manifest references must be a list")
    generated_contract = manifest.get("generated_contract")
    if not isinstance(generated_contract, dict) or generated_contract.get("discovered_rows") != len(
        rows
    ):
        raise GuardError("candidate privacy manifest discovered-row count drifted")
    declared: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise GuardError("candidate privacy manifest row is malformed")
        key = f"{row.get('file')}::{row.get('symbol')}"
        if key in declared:
            raise GuardError(f"duplicate candidate privacy manifest row: {key}")
        if row.get("classification") not in {"fenced", "excluded"}:
            raise GuardError(f"candidate privacy manifest row lacks classification: {key}")
        if not isinstance(row.get("reason"), str) or not row["reason"].strip():
            raise GuardError(f"candidate privacy manifest row lacks reason: {key}")
        if row["classification"] == "fenced" and (
            not row.get("enforcement") or not row.get("test_id")
        ):
            raise GuardError(f"fenced candidate privacy row lacks enforcement/test: {key}")
        declared[key] = row

    discovered = discover()
    actual_references = {reference.key: reference for reference in discovered}
    actual = {reference.key: reference.projection() for reference in discovered}
    if set(actual) != set(declared):
        missing = sorted(set(actual) - set(declared))
        stale = sorted(set(declared) - set(actual))
        raise GuardError(f"candidate privacy surface census drifted (new={missing}, stale={stale})")
    for key, projection in actual.items():
        row = declared[key]
        for field in ("file", "symbol", "tables", "calls", "routes"):
            if row.get(field) != projection[field]:
                raise GuardError(f"candidate privacy surface row drifted: {key}:{field}")
        for anchor in row.get("anchors", []):
            source = (ROOT / row["file"]).read_text(encoding="utf-8")
            if anchor not in source:
                raise GuardError(f"candidate privacy enforcement anchor missing: {key}")
        if row["classification"] == "fenced":
            _validate_fence_anchor(actual_references[key], row)

    _validate_raw_output_ban()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--dump-discovery", action="store_true")
    args = parser.parse_args()
    try:
        if args.dump_discovery:
            print(json.dumps([reference.projection() for reference in discover()], indent=2))
        else:
            validate()
            print("candidate-privacy-surface-guard: OK")
        return 0
    except GuardError as exc:
        print(f"candidate-privacy-surface-guard: REFUSED ({exc})", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
