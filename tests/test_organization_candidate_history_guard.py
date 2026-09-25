"""Mutation checks are source-only and never alter a worktree or database."""

import hashlib
import json

import pytest

from scripts import organization_candidate_history_guard as guard


def sources():
    return {path: (guard.ROOT / path).read_text() for path in guard.SOURCES}


def test_authored_history_guard():
    guard.validate()


@pytest.mark.parametrize(
    "target", ["global_candidates", "candidate_index_heads", "organization_decision_event_inbox"]
)
def test_non_history_write_refuses_even_if_digest_repinned(target):
    files = sources()
    files[guard.MIGRATION] += f"\nUPDATE public.{target} SET tenant_id=tenant_id;\n"
    manifest = json.loads(files["scripts/schema_control_callers.json"])
    manifest["migration_files"][guard.Path(guard.MIGRATION).name] = hashlib.sha256(
        files[guard.MIGRATION].encode()
    ).hexdigest()
    files["scripts/schema_control_callers.json"] = json.dumps(manifest)
    with pytest.raises(guard.GuardError, match="forbidden_write"):
        guard.validate_sources(files)


def test_large_payload_ids_are_required_without_rendering_payload():
    unsafe = '@pytest.mark.parametrize("payload", ["x" * 2048])\ndef test_payload(payload): pass'
    with pytest.raises(guard.GuardError, match="explicit_ids"):
        guard.validate_parameter_ids(unsafe)
    guard.validate_parameter_ids(unsafe.replace('["x" * 2048]', '["x" * 2048], ids=["large"]'))
    with pytest.raises(guard.GuardError, match="explicit_ids"):
        guard.validate_parameter_ids(unsafe.replace('"x" * 2048', 'b"x"'))


@pytest.mark.parametrize(
    "before,after",
    [
        ("FORCE ROW LEVEL SECURITY", "ENABLE ROW LEVEL SECURITY"),
        (
            "public.candidate_index_status(s.source_id,true)",
            "public.candidate_index_status(s.source_id,false)",
        ),
        ("SET search_path=pg_catalog,public", "SET search_path=public"),
        ("SET lock_timeout='500ms'", "SET lock_timeout='0'"),
        ("r.job_id<>e.job_id", "false"),
    ],
    ids=["rls", "blocking-status", "search-path", "lock-timeout", "job-binding"],
)
def test_sql_authority_mutations_refuse_even_when_digest_is_repinned(before, after):
    files = sources()
    assert before in files[guard.MIGRATION]
    files[guard.MIGRATION] = files[guard.MIGRATION].replace(before, after)
    manifest = json.loads(files["scripts/schema_control_callers.json"])
    manifest["migration_files"][guard.Path(guard.MIGRATION).name] = hashlib.sha256(
        files[guard.MIGRATION].encode()
    ).hexdigest()
    files["scripts/schema_control_callers.json"] = json.dumps(manifest)
    with pytest.raises(guard.GuardError):
        guard.validate_sources(files)


@pytest.mark.parametrize(
    "path,before,after",
    [
        ("worker", "max_workers=1", "max_workers=2"),
        ("repository", "SET TRANSACTION READ ONLY", "SELECT 1"),
    ],
    ids=["overlap", "read-write"],
)
def test_python_authority_mutations_refuse(path, before, after):
    files = sources()
    key = f"activekg/candidate_history/{path}.py"
    assert before in files[key]
    files[key] = files[key].replace(before, after)
    with pytest.raises(guard.GuardError):
        guard.validate_sources(files)
