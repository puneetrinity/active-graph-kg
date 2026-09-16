#!/usr/bin/env python3
"""Wave 4D Memory index guard: H1 schema/ACL/engine authority plus H2 source adopters, legacy publisher fence,
private search semantics, worker boundaries, runtime configuration/health, API registration and posture, and
bounded catchup, with the deferred provider route asserted absent.

This is a static regression barrier, not a substitute for the real PostgreSQL
catalog, runtime-role and concurrency proofs. No database or provider is contacted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = "db/migrations/028_candidate_generation_publication.sql"
FROZEN = {
    "activekg/api/auth.py": "f4005bf2df5818d27fcae77cd3659d860170bdb0f4b75b8643b76876ee323071",
    "activekg/api/candidate_privacy.py": "29ba499e7efbdd8e81841b376781000bb8dd8d4022851a8291d49c6742506851",
    "activekg/api/organization_candidates.py": "311df4f1385eb1c4756cf5f61f850c50f831b1dad16411e226402e7c07cba50a",
    "activekg/api/sourced_candidates.py": "2d3342ff555c29f006df4a3aec9d607202f5f093023097b5f359740e74863d2c",
    "activekg/connectors/extract.py": "329f37393b21f9d241f71300c10575e7eb36adfd4d05ea245382a918ab0676a9",
    "activekg/embedding/global_candidates.py": "e81ce5747609c45dca980a184d83d81fe90308a25776561a356df014f651fbaf",
    "activekg/embedding/queue.py": "936836ad21fcb6597014f7f2903a380ca2ffae259f1f870427c680c1c1301401",
    "activekg/engine/embedding_provider.py": "cd8f6320aed622585b9084d18e07e3ef71e751b09b9fecddc7e79171b65bdfc5",
    "activekg/engine/llm_provider.py": "81ca5704eb535b170c9d57288fe1649820f79d944f977cbf9d3e846914c8a4c7",
    "activekg/engine/model_config.py": "20a5bdaff8629d5a6f27422c0d3de2418818b091f8379592cae351d7e13346a3",
    "activekg/extraction/client.py": "7d9c9d1bb3c569c8520ca919a36af9b81cc4988f62ab566741c706916ebc549e",
    "activekg/extraction/prompt.py": "faee64ff3fcc9af5cb53e2d1816c3f42ba22ae8f9e1cad526ba91c3a49a56393",
    "activekg/extraction/queue.py": "bc6f16473d28055d1d363225f043862408f940b891ca6159842ff17b2f9a2be6",
    "activekg/extraction/schema.py": "3ad7aa7b677838d7311508c9f361d501b5dd74514d6a3d0ecd6b554010399e1c",
    "activekg/graph/candidate_repository.py": "11307b9bb4719532667592f20940ddca5c47bda05e2ca69bdbd495d9812e60f7",
    "activekg/graph/repository.py": "fd9103046938e20de768554edaaad9ed350ce9d3c91fc46ab990a692e8a161c8",
    "activekg/privacy/config.py": "97c6bc67fafcf4953c85e7b86579fae4b537f0c264550f378cad4ea159fa9546",
    "activekg/privacy/identity.py": "3c5bc791e51d0e8b5bd01964ab0f7ddd52a53420b44e0684f7a52c2ac16fa65c",
    "activekg/privacy/models.py": "078caf25e85a7e07aa32dad47c7ed86fe7bd9b75f3aeb3852527efad4e99522f",
    "db/migrations/001_add_embedding_history_index.sql": "f109a2eb8da22cbcd1efdbadb5d9fa3a166784740bfe48f7e5c85805fc1556fa",
    "db/migrations/004_add_external_id_index.sql": "49a2fb38c7bcf1df96923abbfb3b6d913a6824c8b467e6769805371125e8fe20",
    "db/migrations/005_connector_configs_table.sql": "42f4a2a2f134dd455cafff14ae906ecb633d8ff594650b32ada7c3afa31afb51",
    "db/migrations/006_add_key_version.sql": "163c9b9442ec1659449ceb8848bb1b56b135c1050f2ba99e88ab8c92404e6092",
    "db/migrations/007_add_provider_check.sql": "9ded04ff2944d3f7966399917a2f44a29df5966ff5376ce23f688de3cf21e15d",
    "db/migrations/008_connector_cursors_table.sql": "099bf0b13bac290b4f5b64f1769d10c147d3b0f622c06cdc134696bd12da4039",
    "db/migrations/009_embedding_queue_status.sql": "fbe9c6e7bba390b03827bf18edf26507011d955c6c49bc857335fe146dfea3fe",
    "db/migrations/010_update_text_search_vector.sql": "26d56f1d806e910755617728db3129c9998ff19d6e1db2786b6a1e8d3773d511",
    "db/migrations/011_unique_tenant_external_id.sql": "b03014f3be91f49514fe32096780b6daa9c7782bb626e1fddad5a8d495b09689",
    "db/migrations/012_candidate_identity.sql": "fc9e708d7da68aee3ec9746c1eb3ff20d0228650ccb2d1ba1c71ed24ee089557",
    "db/migrations/012_global_memory.sql": "f8ef4063e1dd69fe38ea9e0dcf6e79edaf70fc130fa78254c92e8ca66b495262",
    "db/migrations/013_vantahire_provenance.sql": "df7553a55b26d267f1d8e2d151a9009adefb6251a34cb54510a9da0df3f27c17",
    "db/migrations/014_signal_job_tags.sql": "c575c4c64a3229156f20db744b6e0bc472343646142e447307056b50ef40403d",
    "db/migrations/015_candidate_profile.sql": "56553e49c7c28c7061c4ff5af9d276b9c43ca2bf7aeca0c0d546a072a5f160a0",
    "db/migrations/016_candidate_rls.sql": "2294ef74ce9436782dc5f3c1484939bb53edec69e963233f5ee705a3849d6a63",
    "db/migrations/017_reserve_quarantine_tenant.sql": "6a8602cc932c495a03cc35a5777cf2dfe61cb08c37ce72f2355fb18116932761",
    "db/migrations/018_tenant_nonblank.sql": "46c9da96191a87c3143f73516dd7dcb2f2499dfba9afa2ddce55673842d2db0c",
    "db/migrations/019_global_reconciliation.sql": "913da9f75523e8b81b1271c41232ec985b5fb0699a3ce446063bde481c5830ef",
    "db/migrations/020_embed_version.sql": "5558ed8a0e7b41afbb0f71f2435e0beb4de3000ffc91053183ed155d868eb154",
    "db/migrations/021_public_memory_contact_evidence.sql": "8ce5999f50ff36468ff75d2c95be972778a82c0b7d3c586f4258355f0636b161",
    "db/migrations/022_contact_suppression_person_and_audit.sql": "a7abfdeef3c2b6a6e36743b19ab5aae10d0fb02f5b94f237699e174d0be8d0bf",
    "db/migrations/023_candidate_privacy_directives.sql": "de179e695497c96321de2990b590b6e93702220b0071b488da18b0beffd94e1e",
    "db/migrations/024_organization_decision_event_inbox.sql": "a39bedef181f6152a5ecad1f5167afd9a266ea08be74686fe37686538665dacf",
    "db/migrations/025_approved_provider_candidate_ingest.sql": "2f75201a30493f099953c62aa077c43a8dbb5745b14fc84fb06b76190570b2fe",
    "db/migrations/026_organization_private_candidate_intake.sql": "154f14ad9eff1bf6f4b86c4944769356c9324cd5bb0a5becaa675179857eb2e6",
    "db/migrations/027_candidate_consent.sql": "d82dc142db35e1de81afc6fc9272a498a02c7e8eceb95e2b01d292b8fa50f72b",
    "db/migrations/add_text_search.sql": "52570186696d15cbbd6a12bf7ef6bc01fe9fe3a64c8b474d6ce7bdf1caabdac4",
    "db/migrations/rollback_text_search.sql": "86719635b6823b0da12c5e3c05e75295522e54f6bd0791c75d8f05b76a4c9bad",
    "Dockerfile": "0f344c67fe9921f70c1a572a462aac40295337862527460ed349b7cbb972637c",
    "pyproject.toml": "a208641d045be3aace8fa8b84851986a238e1c7b2769e4142d0837078f335d4a",
    "requirements.txt": "788182870906a59c8ebc772a71ce066228dba7dd27804c6701273cae6e2a336b",
    "tests/test_embedding_provider.py": "c6a73a33efe347aedb82bddf9b5a41112b7275b188adb0eaa3b9ceb4d083f8a8",
    "tests/test_global_candidate_embedding_text.py": "25e3252a6225fbb8587b2ea53c51ec4ac95747506384970547749a6d60f75fc4",
    "tests/test_public_profile_boundary.py": "667491349da55a899507fa4df43f083a5f28a4822775975f9232be83e1c96751",
    "tests/test_signal_global_observation_order.py": "0e5526e3b31faaf83c2d1f49472e05fe6632f6faeb86f39702ba70a1a02e8f9c",
    "tests/test_sourced_candidates.py": "d1bae94cc426ce2acd530e02e45feadc91a89bb891ada2726f8420a3276d67a4",
}
MIXED = {
    "activekg/privacy/repository.py": (
        "referenced_key_versions",
        "7bb09df1a49e704a2903cbfee181390abce369dd25f0ed4de2b21633fc2d3c55",
    ),
    "activekg/api/global_memory.py": (
        "sync_applicant_to_global_memory",
        "47b7784818a5749a1752fbedbd66a9f8644ed8ffafbcf7c14df786919768c179",
    ),
}
AUTHORITIES = (
    MIGRATION,
    "activekg/candidate_index/contracts.py",
    "activekg/candidate_index/repository.py",
    "activekg/common/migration_manifest.py",
    "scripts/schema_control_callers.json",
    "scripts/init_railway_db.py",
    "activekg/api/operational.py",
    "activekg/api/main.py",
    "tests/test_organization_candidate_intake_postgres.py",
    "activekg/candidate_index/admission.py",
    "activekg/api/candidate_index.py",
    "activekg/api/candidate_consent.py",
    "activekg/candidate_index/providers.py",
    "activekg/candidate_index/processing.py",
    "activekg/candidate_index/search.py",
    "activekg/candidate_index/catchup.py",
    "activekg/extraction/worker.py",
    "activekg/embedding/worker.py",
)
CATCHUP_REQUIRED = (
    'kind: Literal["candidate_consent_catchup"]',
    "ATTEMPTS_PER_SOURCE = 5",
    "len(plan.entries) * ATTEMPTS_PER_SOURCE > plan.max_provider_attempts",
    "plan.issued_at <= now < plan.expires_at <= plan.issued_at + 7200",
    "hmac.compare_digest(plan.seal, _mac(key, body))",
    "configuration().policy.digest != value.policy_sha256",
    'cur.execute("SET TRANSACTION READ ONLY")',
    "WHERE f.source_id>%s::uuid ORDER BY f.source_id LIMIT %s",
    'row["desired_action"] != "grant"',
    'row["delivery_status"] != "delivered"',
    'row["auth_version"] != row["account_auth_version"]',
    'sha256(email) != row["verified_email_sha256"]',
    'row["resume_pin"] != row["resume"]',
    "consent.command_identity(payload)",
    'row["outbox_digest"] != row["command_sha256"]',
    'ops.admit_flow_user(row["live_user"])',
    "ops.verify_original(original) is not True",
    "consent._require_global(cur, tokens, global_id)",
    "WHERE email_hash=%s",
    'kind="candidate_consent"',
    'priority="maintenance"',
    'raise Skip("already_managed")',
    "operator_approved is not True",
    "_posture(ops) != sealed.posture",
    "_memory(cur, payload, fresh[0], capture=False, ops=ops) != entry",
    "record.previous != chain",
    "os.O_NOFOLLOW",
    "os.O_EXCL",
    "fcntl.LOCK_EX | fcntl.LOCK_NB",
    "info.st_nlink != 1",
    "os.fsync(fd)",
    "_sync_parent(journal)",
)


class GuardError(RuntimeError):
    pass


def read(root: Path, path: str) -> str:
    try:
        return (root / path).read_text()
    except OSError as exc:
        raise GuardError("index_required_file_missing") from exc


def require(source: str, tokens: tuple[str, ...], code: str) -> None:
    if any(token not in source for token in tokens):
        raise GuardError(code)


def before(source: str, first: str, second: str, code: str) -> None:
    require(source, (first, second), code)
    if source.index(first) >= source.index(second):
        raise GuardError(code)


def literal(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                return ast.literal_eval(node.value)
    raise GuardError("index_inventory_missing")


def validate(root: Path = ROOT) -> None:
    catchup = read(root, "activekg/candidate_index/catchup.py")
    require(catchup, CATCHUP_REQUIRED, "index_catchup_authority")
    for path in (
        "activekg/api/main.py",
        "activekg/extraction/worker.py",
        "activekg/embedding/worker.py",
    ):
        if re.search(r"(?:from|import)\s+[^\n]*candidate_index\.catchup", read(root, path)):
            raise GuardError("index_catchup_startup_forbidden")
    legacy = read(root, "activekg/api/global_memory.py")
    legacy_hook = next(
        n
        for n in ast.parse(legacy).body
        if isinstance(n, ast.FunctionDef) and n.name == "sync_applicant_to_global_memory"
    )
    hook_source = ast.get_source_segment(legacy, legacy_hook)
    before(
        hook_source,
        "if legacy_applicant_is_managed(cur, node_id, tenant_id):",
        "_require_privacy_allowed(",
        "index_legacy_publisher_fence",
    )
    require(hook_source, ("conn.rollback()", "return"), "index_legacy_publisher_fence")
    require(
        read(root, "activekg/candidate_index/repository.py"),
        (
            "r.application_id::text=n.metadata->>'application_id'",
            "r.job_id::text=n.metadata->>'job_id'",
            "s.scope_key=r.tenant_id AND s.tenant_id=r.tenant_id AND s.reference_id=r.reference_id",
            "s.candidate_id=r.candidate_id AND s.resume_version_id=e.resume_version_id",
            "s.source_kind='organization_application' AND s.source_version=e.version",
            "n.id=%s::uuid AND n.tenant_id=%s",
            "(node_id, tenant_id)",
            "candidate_index_legacy_authority_unavailable",
        ),
        "index_legacy_publisher_tuple",
    )
    providers = read(root, "activekg/candidate_index/providers.py")
    processing = read(root, "activekg/candidate_index/processing.py")
    search = read(root, "activekg/candidate_index/search.py")
    require(
        search,
        (
            "RETRIEVAL_LIMIT = 100",
            "RRF_K = 60",
            "COUNT_LIMIT = 1000",
            'set(supplied) - {"source", "org_id", "job_id"}',
            'raise SearchRefused("candidate_index_filter_unsupported")',
            "class LegacyPrivateReader(GraphRepository)",
            "tenant_id != self.tenant",
            "SET LOCAL transaction_read_only=on",
            "self.index_repo.transaction(tenant_id)",
            "public.candidate_index_read_private(%s::vector,%s,%s,%s,%s)",
            "policy.embedding_artifact_revision",
            "legacy.hybrid_search(",
            "legacy.vector_search(**args)",
            "use_reranker=False",
            "selected.update({hit.application_id: hit for hit in managed})",
            "1 / (RRF_K + rank)",
            "hit.application_id",
            "await rerank_once(",
            'reranker = "fallback"',
            '"display_score_type": "cosine"',
            "hit.identity in admitted",
            "len(retained) != len(hits)",
            "professional_highlight",
            "counts",
            '"bounded": len(states) > COUNT_LIMIT',
        ),
        "index_search_boundary",
    )
    search_tree = ast.parse(search)
    adapter = next(
        n
        for n in search_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LegacyPrivateReader"
    )
    if {n.name for n in adapter.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))} != {
        "__init__",
        "_conn",
    }:
        raise GuardError("index_legacy_reader_override")
    if re.search(r"\b(?:INSERT\s+INTO|UPDATE\s+(?:public\.)?|DELETE\s+FROM)\b", search):
        raise GuardError("index_search_write_forbidden")
    require(
        providers,
        (
            "async def embed_query_once(",
            "async def rerank_once(",
            "models--cross-encoder--ms-marco-MiniLM-L-6-v2",
            "ref.is_symlink()",
            "not math.isfinite(score)",
            'mode == "rerank"',
        ),
        "index_search_model_transport",
    )
    require(
        providers,
        (
            "PARSER_SECONDS = 15",
            "PARSER_MEMORY_BYTES = 256 * 1024 * 1024",
            "MAX_RESPONSE_BYTES = 65536",
            "follow_redirects=False",
            "trust_env=False",
            "asyncio.timeout(remaining(deadline))",
            "await response.aclose()",
            "len(raw) + len(chunk) > MAX_RESPONSE_BYTES",
            "await before_send()",
            "response.status_code == 429",
            'headers.get("x-ratelimit-remaining-" + quota) != "0"',
            "len(text) > EXTRACTION_MAX_INPUT_CHARS",
            'choice.get("finish_reason") != "stop"',
            '"-I"',
            '"-B"',
            "env=_child_environment()",
            "child.kill()",
            "await child.wait()",
            "resource.setrlimit(resource.RLIMIT_AS, (PARSER_MEMORY_BYTES, PARSER_MEMORY_BYTES))",
            "socket.socket.connect = refuse",
            "socket.getaddrinfo = refuse",
            "if forbidden:",
            '"HF_HUB_OFFLINE": "1"',
            '"TRANSFORMERS_OFFLINE": "1"',
            "snapshot.is_symlink()",
            "not snapshot.is_dir()",
            "pdf.doc.encryption",
            "append(page.extract_text())",
            "truncation=False, add_special_tokens=True",
        ),
        "index_bounded_model_transport",
    )
    before(
        providers,
        "await before_send()",
        "response = await client.send(request, stream=True)",
        "index_model_privacy_after_wait",
    )
    child = providers[providers.index("def _run_child(") :]
    before(
        child,
        "socket.socket.connect = refuse",
        "from activekg.engine.embedding_provider import EmbeddingProvider",
        "index_child_network_fence",
    )
    before(
        child,
        "socket.socket.connect = refuse",
        "from sentence_transformers import CrossEncoder",
        "index_reranker_network_fence",
    )
    if any(
        token in providers
        for token in ("ExtractionClient(", "LLMProvider(", "generate_with_retry(", "verify=False")
    ):
        raise GuardError("index_nested_retry_or_tls")
    require(
        processing,
        (
            "self.repository.reserve, lease, self.policy",
            "self.repository.status, dispatch[",
            'dispatch["source_kind"] not in ACTIVE_SOURCE_KINDS',
            'dispatch["policy_sha256"] != self.policy.digest',
            'remaining_ms = dispatch.get("remaining_ms")',
            "deadline = started + remaining_ms / 1000",
            "before_send=lambda: self._authority(dispatch, deadline)",
            'raw["confidence"] < policy.minimum_confidence',
            "schema.ExtractionResult.model_validate(raw, strict=True)",
            "if not evidence:",
            "chunk_manifest(professional, policy)",
            "self.repository.complete_extract",
            "self.repository.complete_embed",
            "validate_vectors(vectors, len(chunks))",
            "self.repository.fail, dispatch, reason, retry_ms",
            "model = (",
            "self.policy.primary_model_id",
            "self.policy.fallback_model_id",
        ),
        "index_generation_worker_fences",
    )
    before(
        processing,
        "self.repository.reserve, lease, self.policy",
        "raw = await extract_once(",
        "index_reserve_before_model",
    )
    require(
        processing,
        (
            "if not require_placement(stage):",
            'os.environ.get("GROQ_API_KEY", "").strip()',
            "config.policy.embedding_model_id, config.policy.embedding_artifact_revision",
            "loop.call_soon_threadsafe(task.cancel)",
            "if self._stopping.is_set():",
            "self._thread.join(timeout=10)",
            "if self._thread.is_alive():",
            'raise IndexContractError("candidate_index_runtime_stop_timeout")',
            '"settlement_unavailable" in outcomes',
        ),
        "index_runtime_lifecycle",
    )
    for path, name, stage in (
        ("activekg/extraction/worker.py", "start_extraction_worker", "extract"),
        ("activekg/embedding/worker.py", "start_worker", "embed"),
    ):
        source = read(root, path)
        try:
            entry = next(
                ast.get_source_segment(source, node)
                for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == name
            )
        except (StopIteration, SyntaxError):
            raise GuardError("index_worker_entrypoint_missing") from None
        require(
            entry,
            (
                f'index_config = worker_configuration("{stage}")',
                "assert_startup_schema_ready(require_privacy_hmac=False)",
                "if index_config is not None",
                "if index_runtime is not None:",
                "index_runtime=index_runtime",
                "index_runtime.start()",
                "index_runtime.request_stop()",
                "index_runtime.close()",
                "server.server_close()",
                "monitor.join(timeout=5)",
            ),
            "index_worker_entrypoint_lifecycle",
        )
        before(
            entry,
            "index_config = worker_configuration",
            "dsn = assert_startup_schema_ready",
            "index_config_before_io",
        )
        before(entry, "index_runtime.start()", "worker.run()", "index_runtime_before_legacy_loop")
        require(
            source,
            (
                'components["candidate_index"]'
                if stage == "extract"
                else 'service="embedding-worker"',
            ),
            "index_worker_health",
        )
    admission = read(root, "activekg/candidate_index/admission.py")
    require(
        admission,
        ('load_candidate_privacy_config(require_hmac=stage == "api")',),
        "index_worker_no_keyring",
    )
    require(
        admission,
        (
            "_require_private_privacy(cur, command.privacy_subject)",
            "_evidence(cur, tenant, command)",
            "with repo.transaction(tenant)",
            'kind="organization_application"',
            'kind="candidate_consent"',
            'if not enabled("api") or outcome != "granted":',
            'env.get("CANDIDATE_INDEX_POLICY_SHA256") != policy.digest',
        ),
        "index_adopter_authority",
    )
    before(
        admission,
        "_require_private_privacy(cur, command.privacy_subject)",
        "result = repo.capture_on_cursor(",
        "index_source_privacy_order",
    )
    consent = read(root, "activekg/api/candidate_consent.py")
    before(
        consent,
        "capture_consent(cur, payload, outcome, command_digest)",
        "conn.commit()",
        "index_consent_atomic_capture",
    )
    api = read(root, "activekg/api/candidate_index.py")
    require(
        api,
        (
            'claims.actor_id != "vantahire-backend"',
            'claims.issuer != "vantahire"',
            'claims.actor_type != "service"',
            "claims.scopes != [scope]",
            "Depends(writer)",
            "Depends(reader)",
            "s.scope_key=%s",
            "s.reference_id=ANY(%s::uuid[])",
            "if len(raw) + len(chunk) > cap:",
            "SourceContentCommand, MAX_REQUEST_BYTES",
            'if require_placement("api"):',
            'os.environ.get("ACTIVEKG_DSN")',
            "cached_embedding_path(",
            "config.policy.embedding_model_id, config.policy.embedding_artifact_revision",
        ),
        "index_route_authority",
    )
    main = read(root, "activekg/api/main.py")
    require(
        main,
        (
            "app.include_router(candidate_index_router)",
            "if candidate_index_api_configuration_problems():",
            "candidate_index_problems=candidate_index_api_configuration_problems()",
        ),
        "index_api_registration",
    )
    before(
        main[main.index("def startup_event():") :],
        "if candidate_index_api_configuration_problems():",
        "repo.ensure_vector_index()",
        "index_api_startup_order",
    )
    require(
        read(root, "activekg/api/operational.py"),
        (
            "if candidate_index_problems:",
            "reasons.extend(candidate_index_problems)",
        ),
        "index_api_readiness",
    )
    for path, digest in FROZEN.items():
        if hashlib.sha256(read(root, path).encode()).hexdigest() != digest:
            raise GuardError("index_frozen_dependency")
    for path, (symbol, digest) in MIXED.items():
        source = read(root, path)
        matches = [
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == symbol
        ]
        if len(matches) != 1:
            raise GuardError("index_mixed_boundary")
        node = matches[0]
        lines = source.splitlines(keepends=True)
        outside = "".join(lines[: node.lineno - 1] + lines[node.end_lineno :])
        if hashlib.sha256(outside.encode()).hexdigest() != digest:
            raise GuardError("index_mixed_boundary")

    sql = read(root, MIGRATION)
    contract = read(root, "activekg/candidate_index/contracts.py")
    repository = read(root, "activekg/candidate_index/repository.py")
    release = read(root, "scripts/init_railway_db.py")
    immutable = literal(contract, "IMMUTABLE_TABLES")
    coordination = literal(contract, "COORDINATION_TABLES")
    runtime = literal(contract, "RUNTIME_FUNCTIONS")
    owner = literal(contract, "OWNER_FUNCTIONS")
    tables = re.findall(r"CREATE TABLE public\.(\w+)", sql)
    if len(tables) != 8 or set(tables) != set(immutable + coordination):
        raise GuardError("index_table_inventory")
    if (
        len(immutable) != 5
        or len(coordination) != 3
        or len(runtime) != 10
        or owner != ("candidate_index_read_public(integer)", "candidate_index_append_only()")
        or any("candidate_index_read_public(" in f for f in runtime)
    ):
        raise GuardError("index_exact_acl_inventory")

    functions = dict(re.findall(r"CREATE FUNCTION public\.(\w+)(\(.*?\$\$;)", sql, re.S))
    if len(functions) != 12 or set(functions) != {x.split("(")[0] for x in runtime + owner}:
        raise GuardError("index_routine_inventory")
    for name, body in functions.items():
        header = body.split("AS $$", 1)[0]
        require(
            header,
            ("SECURITY DEFINER", "SET search_path=pg_catalog,public"),
            "index_routine_security",
        )
        if name != "candidate_index_append_only" and name != "candidate_index_key_versions":
            require(
                header,
                ("SET lock_timeout='1500ms'", "SET statement_timeout='3s'"),
                "index_routine_deadline",
            )
    for signature in runtime + owner:
        # pgvector is explicitly qualified in DDL, but not in the regprocedure inventory.
        signature = signature.replace("(vector,", "(public.vector,")
        require(
            sql,
            (f"REVOKE ALL ON FUNCTION public.{signature} FROM PUBLIC;",),
            "index_public_execute",
        )

    names = re.findall(
        r"(?:CREATE (?:UNIQUE )?(?:TABLE|INDEX|FUNCTION|TRIGGER)|CONSTRAINT)\s+(?:public\.)?(\w+)",
        sql,
    )
    if any(len(name.encode()) > 63 for name in names):
        raise GuardError("index_identifier_too_long")
    if re.search(
        r"\b(?:DISABLE TRIGGER|BYPASSRLS|SET row_security\s*=\s*off|CASCADE)\b", sql, re.I
    ):
        raise GuardError("index_bypass")
    if re.search(
        r"\b(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+public\.(?!candidate_index_)\w+", sql, re.I
    ):
        raise GuardError("index_legacy_write")
    require(
        sql,
        (
            "FORCE ROW LEVEL SECURITY",
            "ENABLE ROW LEVEL SECURITY",
            "BEFORE UPDATE OR DELETE",
            "BEFORE TRUNCATE",
            "TO %I USING (true) WITH CHECK (true)",
            "',t,current_user)",
            "index_source_resume_tuple_fk",
            "ON DELETE RESTRICT",
            "CHECK ((embedding <#> embedding) BETWEEN -1.001 AND -0.999)",
        ),
        "index_schema_authority",
    )

    status = functions["candidate_index_status"]
    require(
        status,
        (
            "scope_key=current_setting('app.current_tenant_id',true)",
            "s.source_kind='approved_provider'",
            "pg_try_advisory_xact_lock",
            "canonical.needs_review IS DISTINCT FROM false",
            "checked.global_candidate_id IS DISTINCT FROM canonical.global_candidate_id",
            "integer=ANY(s.key_versions)",
            "public.candidate_privacy_match(",
            "('allow','block_global')",
            "s.source_kind='candidate_consent' AND decision='block_global'",
            "c.active_source_id=s.consent_source_id",
            "c.effective_action='grant'",
            "c.effective_version=s.source_version",
        ),
        "index_current_authority",
    )
    before(status, "'candidate-consent:'", "'candidate-privacy-token:'", "index_lock_order")
    before(status, "'candidate-privacy-token:'", "'candidate-privacy-global:'", "index_lock_order")
    claim = functions["candidate_index_claim"]
    require(
        claim,
        (
            "p_limit BETWEEN 1 AND 8",
            "concurrency BETWEEN 1 AND 2",
            "scope_concurrency=1",
            "lease_ms BETWEEN deadline_ms+10001",
            "j.dispatch_reserved_generation=j.lease_generation",
            "j.attempts>=(CASE p_stage WHEN 'extract' THEN 2 ELSE 3 END)",
            "dispatch_deadline=coalesce(dispatch_deadline,",
            "src_row.source_kind<>'approved_provider'",
            "mod(turn,iw+mw)<iw",
            "c.last_served_at NULLS FIRST",
            "allowed->>'contended'='true'",
            "FOR UPDATE SKIP LOCKED",
            "lease_generation=lease_generation+1",
        ),
        "index_dispatch_fencing",
    )
    before(
        claim,
        "allowed:=public.candidate_index_status(s.source_id)",
        "attempts=attempts+",
        "index_dispatch_privacy",
    )
    capture = functions["candidate_index_capture_source"]
    require(
        capture,
        (
            "p_kind IN ('organization_application','candidate_consent')",
            "public.organization_candidate_resume_evidence",
            "public.organization_candidate_ingest_receipts",
            "public.candidate_consent_sources",
            "public.candidate_index_status(",
            "count(DISTINCT generation_id)",
            "'waiting_admission'",
            "source_id=p_upstream_id AND policy_sha256=policy_hash",
        ),
        "index_durable_capture",
    )
    for name in ("candidate_index_complete_extract", "candidate_index_complete_embed"):
        body = functions[name]
        require(
            body,
            (
                "public.candidate_index_status(",
                "lease_token IS DISTINCT FROM p_token",
                "lease_generation IS DISTINCT FROM p_generation",
                "j.lease_expires_at<=clock_timestamp()",
                "j.dispatch_reserved_generation<>j.lease_generation",
                "IS DISTINCT FROM",
            ),
            "index_completion_fencing",
        )
        before(
            body,
            "public.candidate_index_status(",
            "INSERT INTO public.candidate_index_",
            "index_completion_privacy",
        )
        if not any(
            check in body
            for check in (
                "p_policy_sha256 IS DISTINCT FROM g.policy_sha256",
                "g.policy_sha256 IS DISTINCT FROM p_policy_sha256",
            )
        ):
            raise GuardError("index_completion_policy")
    extract = functions["candidate_index_complete_extract"]
    require(
        extract,
        (
            "p_confidence BETWEEN (g.policy->>'minimum_confidence')",
            "WHEN j.attempts=2 THEN g.policy->>'fallback_model_id'",
            "strpos(lower(input_text),lower(evidence->>'text'))>0",
            "assembled IS DISTINCT FROM p_professional_text",
            "INSERT INTO public.candidate_index_jobs",
        ),
        "index_complete_extraction",
    )
    embed = functions["candidate_index_complete_embed"]
    require(
        embed,
        (
            "jsonb_array_length(p_vectors)=jsonb_array_length(e.chunks)",
            "jsonb_array_length(v)=384",
            "jsonb_typeof(n)<>'number'",
            "INSERT INTO public.candidate_index_vectors",
            "INSERT INTO public.candidate_index_publication_events",
            "published_generation_id=j.generation_id",
        ),
        "index_atomic_publication",
    )
    before(
        embed,
        "INSERT INTO public.candidate_index_vectors",
        "INSERT INTO public.candidate_index_publication_events",
        "index_atomic_publication",
    )
    before(
        functions["candidate_index_read_private"],
        "public.candidate_index_status(",
        "RETURN NEXT",
        "index_read_privacy",
    )
    require(
        functions["candidate_index_read_public"],
        (
            "source_kind='candidate_consent'",
            "public.candidate_index_status(",
            "'approved_profile',item.approved_profile",
        ),
        "index_consent_only_read",
    )
    require(
        functions["candidate_index_catchup"],
        (
            "s.source_kind IN ('organization_application','candidate_consent')",
            "count(DISTINCT generation_id)",
            "public.candidate_index_status(",
        ),
        "index_bounded_catchup",
    )

    require(
        release,
        (
            "_harden_candidate_index_runtime_privileges(cur, runtime_role)",
            "_assert_candidate_index_runtime_privileges(cur, runtime_role)",
            "if signature in RUNTIME_FUNCTIONS:",
            "REVOKE SELECT({}),INSERT({}),UPDATE({}),REFERENCES({})",
            "catalog_ready(cur, role)",
        ),
        "index_role_reconciliation",
    )
    require(
        repository,
        (
            "'columns'",
            "'policies'",
            "'grant_option'",
            "'column_grant_option'",
            "'public_columns'",
            "digest == INDEX_CATALOG_SHA256",
            "with psycopg.connect(self._dsn, connect_timeout=3) as conn",
            "SET LOCAL lock_timeout='1500ms'",
        ),
        "index_catalog_or_transactions",
    )
    require(
        read(root, "activekg/api/operational.py"),
        ("catalog_ready(cur)", "candidate_index_hmac_version_missing"),
        "index_readiness",
    )
    if "/global-candidates/search-generations" in read(root, "activekg/api/main.py"):
        raise GuardError("index_deferred_route")
    for path in ("activekg/candidate_index/contracts.py", "activekg/candidate_index/repository.py"):
        if re.search(
            r"\b(?:requests|httpx|aiohttp|urllib|socket|subprocess)\b|\bprint\s*\(",
            read(root, path),
        ):
            raise GuardError("index_engine_network_or_output")
    caller = json.loads(read(root, "scripts/schema_control_callers.json"))
    if (
        caller["migration_manifest"][-1] != Path(MIGRATION).name
        or len(caller["migration_manifest"]) != 28
    ):
        raise GuardError("index_ledger")
    if hashlib.sha256(sql.encode()).hexdigest() != caller["migration_files"][Path(MIGRATION).name]:
        raise GuardError("index_migration_pin")
    retained = read(root, "tests/test_organization_candidate_intake_postgres.py")
    require(
        retained,
        (
            'assert truncation.value.sqlstate == "55000"',
            "organization_candidate_ingest_receipts contains committed evidence and cannot be truncated",
            *immutable,
            *coordination,
        ),
        "index_retained_append_only_proof",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        validate(args.root)
    except (GuardError, ValueError, KeyError, SyntaxError) as exc:
        raise SystemExit(f"candidate-index-guard: REFUSED ({exc})") from None
    print("candidate-index-guard: OK (engine/adopters/search/workers/api/catchup)")
