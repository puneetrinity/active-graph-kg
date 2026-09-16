"""Mutation checks run on copies; no baseline/product file is changed by a test."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from scripts import candidate_index_guard as guard


def copied(tmp_path):
    root = tmp_path / "engine"
    for name in set(guard.FROZEN) | set(guard.MIXED) | set(guard.AUTHORITIES):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(guard.ROOT / name, target)
    return root


def test_current_engine_guard():
    guard.validate()


@pytest.mark.parametrize("token", guard.CATCHUP_REQUIRED)
def test_catchup_authority_tokens(tmp_path, token):
    root = copied(tmp_path)
    file = root / "activekg/candidate_index/catchup.py"
    source = file.read_text()
    assert token in source
    file.write_text(source.replace(token, "REMOVED"))
    with pytest.raises(guard.GuardError, match="index_catchup_authority"):
        guard.validate(root)


@pytest.mark.parametrize(
    "path,token,code",
    [
        (
            "activekg/api/global_memory.py",
            "if legacy_applicant_is_managed(cur, node_id, tenant_id):",
            "index_legacy_publisher_fence",
        ),
        (
            "activekg/candidate_index/repository.py",
            "s.scope_key=r.tenant_id AND s.tenant_id=r.tenant_id",
            "index_legacy_publisher_tuple",
        ),
        (
            "activekg/candidate_index/repository.py",
            "s.resume_version_id=e.resume_version_id",
            "index_legacy_publisher_tuple",
        ),
        (
            "activekg/candidate_index/repository.py",
            "r.job_id::text=n.metadata->>'job_id'",
            "index_legacy_publisher_tuple",
        ),
    ],
)
def test_legacy_publisher_ownership_guard(tmp_path, path, token, code):
    root = copied(tmp_path)
    file = root / path
    source = file.read_text()
    assert token in source
    file.write_text(source.replace(token, "if False:" if token.startswith("if ") else "REMOVED"))
    with pytest.raises(guard.GuardError, match=code):
        guard.validate(root)


@pytest.mark.parametrize(
    "path,token",
    [
        ("activekg/api/main.py", "app.include_router(candidate_index_router)"),
        ("activekg/api/main.py", "if candidate_index_api_configuration_problems():"),
        (
            "activekg/api/main.py",
            "candidate_index_problems=candidate_index_api_configuration_problems()",
        ),
        ("activekg/api/candidate_index.py", 'if require_placement("api"):'),
        ("activekg/api/candidate_index.py", "cached_embedding_path("),
        ("activekg/api/operational.py", "reasons.extend(candidate_index_problems)"),
    ],
)
def test_api_registration_and_posture_guard(tmp_path, path, token):
    root = copied(tmp_path)
    file = root / path
    source = file.read_text()
    assert token in source
    file.write_text(source.replace(token, "REMOVED_BY_TEST"))
    with pytest.raises(guard.GuardError):
        guard.validate(root)


@pytest.mark.parametrize(
    "token",
    [
        "RRF_K = 60",
        "COUNT_LIMIT = 1000",
        "tenant_id != self.tenant",
        "SET LOCAL transaction_read_only=on",
        "legacy.vector_search(**args)",
        "selected.update({hit.application_id: hit for hit in managed})",
        "use_reranker=False",
        "policy.embedding_artifact_revision",
        "hit.identity in admitted",
        "len(retained) != len(hits)",
        'raise SearchRefused("candidate_index_filter_unsupported")',
        'reranker = "fallback"',
    ],
)
def test_private_search_semantics_tripwires(tmp_path, token):
    root = copied(tmp_path)
    file = root / "activekg/candidate_index/search.py"
    source = file.read_text()
    assert token in source
    file.write_text(source.replace(token, "REMOVED_BY_TEST"))
    with pytest.raises(guard.GuardError):
        guard.validate(root)


@pytest.mark.parametrize(
    ("path", "token"),
    [
        ("providers.py", "PARSER_MEMORY_BYTES = 256 * 1024 * 1024"),
        ("providers.py", "await before_send()"),
        ("providers.py", "child.kill()"),
        ("providers.py", "socket.socket.connect = refuse"),
        ("providers.py", "follow_redirects=False"),
        ("providers.py", "len(raw) + len(chunk) > MAX_RESPONSE_BYTES"),
        ("providers.py", "snapshot.is_symlink()"),
        ("providers.py", "truncation=False, add_special_tokens=True"),
        ("processing.py", "self.repository.reserve, lease, self.policy"),
        ("processing.py", "before_send=lambda: self._authority(dispatch, deadline)"),
        ("processing.py", "validate_vectors(vectors, len(chunks))"),
        ("processing.py", 'raw["confidence"] < policy.minimum_confidence'),
        ("processing.py", "if not evidence:"),
    ],
)
def test_worker_boundary_mutation_refuses(tmp_path, path, token):
    root = copied(tmp_path)
    file = root / "activekg/candidate_index" / path
    source = file.read_text()
    assert token in source
    file.write_text(source.replace(token, "REMOVED_BY_TEST"))
    with pytest.raises(guard.GuardError):
        guard.validate(root)


@pytest.mark.parametrize(
    "path,token",
    [
        ("activekg/extraction/worker.py", 'index_config = worker_configuration("extract")'),
        ("activekg/embedding/worker.py", 'index_config = worker_configuration("embed")'),
        ("activekg/extraction/worker.py", "index_runtime.start()"),
        ("activekg/embedding/worker.py", "index_runtime.start()"),
        ("activekg/extraction/worker.py", "index_runtime.close()"),
        ("activekg/embedding/worker.py", "index_runtime.close()"),
        ("activekg/extraction/worker.py", "require_privacy_hmac=False"),
        ("activekg/embedding/worker.py", "require_privacy_hmac=False"),
        ("activekg/extraction/worker.py", 'components["candidate_index"]'),
        ("activekg/embedding/worker.py", 'service="embedding-worker"'),
        ("activekg/candidate_index/processing.py", "loop.call_soon_threadsafe(task.cancel)"),
        ("activekg/candidate_index/processing.py", "self._thread.join(timeout=10)"),
        ("activekg/candidate_index/repository.py", "connect_timeout=3"),
        ("activekg/candidate_index/admission.py", 'require_hmac=stage == "api"'),
    ],
)
def test_runtime_configuration_cancellation_health_and_keyring_tripwires(tmp_path, path, token):
    root = copied(tmp_path)
    file = root / path
    source = file.read_text()
    assert token in source
    file.write_text(source.replace(token, "REMOVED_BY_TEST"))
    with pytest.raises(guard.GuardError):
        guard.validate(root)


@pytest.mark.parametrize(
    "stage,path,start",
    [
        ("extract", "activekg/extraction/worker.py", "start_extraction_worker"),
        ("embed", "activekg/embedding/worker.py", "start_worker"),
    ],
)
def test_real_entrypoint_placement_moved_after_schema_refuses(tmp_path, stage, path, start):
    root = copied(tmp_path)
    file = root / path
    source = file.read_text()
    statement = f'    index_config = worker_configuration("{stage}")\n'
    assert source.count(statement) == 1
    source = source.replace(statement, "")
    marker = (
        "    redis_client = get_redis_client()"
        if stage == "embed"
        else "    assert_extraction_models_configured()"
    )
    assert marker in source
    file.write_text(source.replace(marker, statement + marker))
    with pytest.raises(guard.GuardError, match="index_config_before_io"):
        guard.validate(root)


@pytest.mark.parametrize(
    "old",
    [
        "FORCE ROW LEVEL SECURITY",
        "BEFORE UPDATE OR DELETE",
        "BEFORE TRUNCATE",
        "index_source_resume_tuple_fk",
        "SET search_path=pg_catalog,public",
        "SET lock_timeout='1500ms'",
        "integer=ANY(s.key_versions)",
        "canonical.needs_review IS DISTINCT FROM false",
        "c.active_source_id=s.consent_source_id",
        "c.effective_action='grant'",
        "c.effective_version=s.source_version",
        "src_row.source_kind<>'approved_provider'",
        "s.source_kind IN ('organization_application','candidate_consent')",
        "p_kind IN ('organization_application','candidate_consent')",
        "dispatch_deadline=coalesce(dispatch_deadline,",
        "j.dispatch_reserved_generation=j.lease_generation",
        "j.attempts>=(CASE p_stage WHEN 'extract' THEN 2 ELSE 3 END)",
        "concurrency BETWEEN 1 AND 2",
        "scope_concurrency=1",
        "mod(turn,iw+mw)<iw",
        "c.last_served_at NULLS FIRST",
        "allowed->>'contended'='true'",
        "FOR UPDATE SKIP LOCKED",
        "lease_token IS DISTINCT FROM p_token",
        "lease_generation IS DISTINCT FROM p_generation",
        "j.dispatch_reserved_generation<>j.lease_generation",
        "j.lease_expires_at<=clock_timestamp()",
        "p_confidence BETWEEN (g.policy->>'minimum_confidence')",
        "strpos(lower(input_text),lower(evidence->>'text'))>0",
        "assembled IS DISTINCT FROM p_professional_text",
        "jsonb_array_length(p_vectors)=jsonb_array_length(e.chunks)",
        "CHECK ((embedding <#> embedding) BETWEEN -1.001 AND -0.999)",
        "count(DISTINCT generation_id)",
        "REVOKE ALL ON FUNCTION public.candidate_index_read_public(integer) FROM PUBLIC;",
    ],
)
def test_sql_semantic_refusal_survives_checksum_repin(tmp_path, old):
    root = copied(tmp_path)
    migration = root / guard.MIGRATION
    source = migration.read_text()
    assert old in source
    migration.write_text(source.replace(old, "REMOVED_BY_TEST"))
    caller = root / "scripts/schema_control_callers.json"
    manifest = json.loads(caller.read_text())
    manifest["migration_files"][migration.name] = hashlib.sha256(migration.read_bytes()).hexdigest()
    caller.write_text(json.dumps(manifest))
    with pytest.raises(guard.GuardError):
        guard.validate(root)


@pytest.mark.parametrize(
    ("path", "old", "new"),
    [
        (
            "activekg/candidate_index/contracts.py",
            '"candidate_index_status(uuid,boolean)"',
            '"candidate_index_read_public(integer)"',
        ),
        ("scripts/init_railway_db.py", "if signature in RUNTIME_FUNCTIONS:", "if True:"),
        (
            "scripts/init_railway_db.py",
            "_harden_candidate_index_runtime_privileges(cur, runtime_role)",
            "pass",
        ),
        ("activekg/api/operational.py", "if not catalog_ready(cur):", "if False:"),
        ("activekg/candidate_index/repository.py", "'column_grant_option'", "'ignored'"),
        (
            "activekg/candidate_index/admission.py",
            "_require_private_privacy(cur, command.privacy_subject)",
            "pass",
        ),
        ("activekg/candidate_index/admission.py", "_evidence(cur, tenant, command)", "pass"),
        (
            "activekg/api/candidate_consent.py",
            "capture_consent(cur, payload, outcome, command_digest)",
            "pass",
        ),
        ("activekg/api/candidate_index.py", "claims.scopes != [scope]", "False"),
        ("activekg/api/candidate_index.py", "if len(raw) + len(chunk) > cap:", "if False:"),
        (
            "tests/test_organization_candidate_intake_postgres.py",
            'sqlstate == "55000"',
            'sqlstate == "0A000"',
        ),
        ("activekg/privacy/repository.py", "def canonical_decision(", "def drifted_decision("),
        (
            "activekg/api/global_memory.py",
            "def ingest_feedback_events(",
            "def drifted_feedback_events(",
        ),
    ],
)
def test_boundary_and_privileges_refuse(tmp_path, path, old, new):
    root = copied(tmp_path)
    target = root / path
    assert old in target.read_text()
    target.write_text(target.read_text().replace(old, new))
    with pytest.raises(guard.GuardError):
        guard.validate(root)


def test_long_declared_postgres_name_refused(tmp_path):
    root = copied(tmp_path)
    target = root / guard.MIGRATION
    target.write_text(
        target.read_text()
        + "\nCREATE INDEX "
        + "x" * 64
        + " ON public.candidate_index_sources(source_id);\n"
    )
    with pytest.raises(guard.GuardError, match="identifier_too_long"):
        guard.validate(root)


def test_historical_migration_refuses_even_when_caller_is_repinned(tmp_path):
    root = copied(tmp_path)
    path = root / "db/migrations/027_candidate_consent.sql"
    path.write_text(path.read_text() + "\n-- unauthorized\n")
    caller = root / "scripts/schema_control_callers.json"
    value = json.loads(caller.read_text())
    value["migration_files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    caller.write_text(json.dumps(value))
    with pytest.raises(guard.GuardError, match="frozen"):
        guard.validate(root)
