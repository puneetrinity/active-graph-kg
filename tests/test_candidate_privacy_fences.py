from __future__ import annotations

import ast
import base64
import inspect
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi import HTTPException

from activekg.api.global_memory import _require_hash_only_candidate_subject_verifiable
from activekg.common.migration_manifest import MIGRATIONS
from activekg.embedding.worker import EmbeddingWorker
from activekg.extraction.worker import ExtractionWorker, WorkerHealthState
from activekg.graph.candidate_repository import CandidateRepository
from activekg.graph.repository import GraphRepository
from activekg.privacy.models import CandidatePrivacyDecision
from activekg.privacy.repository import (
    CandidatePrivacyRestricted,
    CandidatePrivacyUnavailable,
    require_allowed,
)

ROOT = Path(__file__).resolve().parents[1]
BASE = "d3ed778e6b705100f0671169375848f71c992f5c"


class PrivacyAuthority:
    def __init__(self, decision: CandidatePrivacyDecision) -> None:
        self.decision = decision
        self.calls: list[dict[str, object]] = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision

    def canonical_decision(self, **kwargs):
        self.calls.append(kwargs)
        return self.decision


def _repository(decision: CandidatePrivacyDecision) -> CandidateRepository:
    repository = CandidateRepository.__new__(CandidateRepository)
    repository.privacy_repository = PrivacyAuthority(decision)
    return repository


def test_global_opt_out_preserves_private_workflow_but_blocks_global_use() -> None:
    repository = _repository(CandidatePrivacyDecision.BLOCK_GLOBAL)
    repository._require_privacy_allowed(
        tenant_id="org-a", candidate_id="11111111-1111-4111-8111-111111111111", global_use=False
    )
    with pytest.raises(CandidatePrivacyRestricted):
        repository._require_privacy_allowed(
            tenant_id="org-a",
            candidate_id="11111111-1111-4111-8111-111111111111",
            global_use=True,
        )


@pytest.mark.parametrize(
    "decision",
    [CandidatePrivacyDecision.BLOCK_ALL, CandidatePrivacyDecision.REVIEW],
)
def test_erasure_and_uncertainty_block_even_existing_private_use(
    decision: CandidatePrivacyDecision,
) -> None:
    with pytest.raises(CandidatePrivacyRestricted):
        require_allowed(decision, global_use=False)


def test_missing_authority_fails_closed_in_production() -> None:
    repository = CandidateRepository.__new__(CandidateRepository)
    repository.privacy_repository = None
    with patch.dict("os.environ", {"ACTIVEKG_SCHEMA_ENVIRONMENT": "production"}, clear=True):
        with pytest.raises(CandidatePrivacyUnavailable):
            repository._require_privacy_allowed(global_use=False)


def _privacy_env(*, intake: bool) -> dict[str, str]:
    return {
        "CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION": "1",
        "CANDIDATE_PRIVACY_HMAC_KEY_V1": base64.b64encode(b"k" * 32).decode(),
        "CANDIDATE_PRIVACY_INTAKE_ENABLED": str(intake).lower(),
        "CANDIDATE_PRIVACY_FLOW_ISSUER": "flow",
        "CANDIDATE_PRIVACY_FLOW_ACTOR_ID": "flow-service",
        "CANDIDATE_PRIVACY_SIGNAL_ISSUER": "signal",
        "CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID": "signal-service",
    }


def test_hash_only_global_ingest_is_preserved_while_intake_is_disabled() -> None:
    with patch.dict("os.environ", _privacy_env(intake=False), clear=True):
        _require_hash_only_candidate_subject_verifiable(
            existing=None,
            email_hash="legacy-one-way-hash",
            transient_identifiers=[],
        )


def test_hash_only_new_subject_fails_closed_once_intake_is_enabled() -> None:
    with patch.dict("os.environ", _privacy_env(intake=True), clear=True):
        with pytest.raises(HTTPException) as exc:
            _require_hash_only_candidate_subject_verifiable(
                existing=None,
                email_hash="legacy-one-way-hash",
                transient_identifiers=[],
            )
    assert getattr(exc.value, "status_code", None) == 503
    assert getattr(exc.value, "detail", None) == "candidate_privacy_subject_unverifiable"


def test_existing_or_strongly_identified_global_subject_skips_hash_only_gate() -> None:
    with patch.dict("os.environ", {}, clear=True):
        _require_hash_only_candidate_subject_verifiable(
            existing={"id": "opaque"},
            email_hash="legacy-one-way-hash",
            transient_identifiers=[],
        )
        _require_hash_only_candidate_subject_verifiable(
            existing=None,
            email_hash="legacy-one-way-hash",
            transient_identifiers=[("linkedin_url", "https://linkedin.com/in/example")],
        )


def test_candidate_upload_refuses_more_than_eight_distinct_strong_identifiers() -> None:
    from activekg.api.main import _candidate_upload_identifiers

    text = " ".join(f"person{index}@example.test" for index in range(9))
    with pytest.raises(HTTPException) as exc:
        _candidate_upload_identifiers(text)
    assert exc.value.status_code == 422
    assert exc.value.detail == "candidate_privacy_subject_too_many_identifiers"


def test_candidate_node_payload_checks_nested_raw_identifiers() -> None:
    from activekg.api import main

    authority = PrivacyAuthority(CandidatePrivacyDecision.BLOCK_ALL)
    with patch.object(main, "candidate_privacy_repo", authority):
        with pytest.raises(HTTPException) as exc:
            main._require_candidate_node_write_allowed(
                classes=["Resume"],
                props={"text": "Candidate can be reached at private@example.test"},
                metadata={},
                tenant_id="org-a",
            )
    assert exc.value.status_code == 409
    assert authority.calls
    identifiers = authority.calls[0]["identifiers"]
    assert [(item.identifier_type, item.normalized) for item in identifiers] == [
        ("email", "private@example.test")
    ]


@pytest.mark.parametrize(
    "method",
    [
        CandidateRepository.get_candidate,
        CandidateRepository.list_identifiers,
        CandidateRepository.get_source_record,
        CandidateRepository.get_latest_source_record,
        CandidateRepository.list_source_records,
        CandidateRepository.search_candidates_by_signal_tags,
        CandidateRepository.search_tenant_private_candidates,
    ],
)
def test_candidate_reader_filters_are_in_sql(method) -> None:
    source = inspect.getsource(method)
    assert "candidate_privacy_candidate_decision" in source
    if "ORDER BY" in source:
        assert source.index("candidate_privacy_candidate_decision") < source.rindex("ORDER BY")


@pytest.mark.parametrize(
    "method",
    [
        GraphRepository.list_nodes,
        GraphRepository.get_node,
        GraphRepository.get_node_by_external_id,
        GraphRepository.vector_search,
        GraphRepository.hybrid_search,
        GraphRepository.get_lineage,
        GraphRepository.get_node_versions,
        GraphRepository.find_nodes_due_for_refresh,
        GraphRepository.all_nodes,
        GraphRepository.detect_drift_spikes,
        GraphRepository.detect_trigger_storms,
        GraphRepository.detect_scheduler_lag,
    ],
)
def test_node_reader_and_enumerator_filters_are_in_sql(method) -> None:
    source = inspect.getsource(method)
    assert "candidate_privacy_node_decision" in source
    if method is GraphRepository.vector_search:
        assert "_privacy_filtered_vector_rows" in source
        helper = inspect.getsource(GraphRepository._privacy_filtered_vector_rows)
        exact_start = helper.index("WITH privacy_filtered_nodes AS MATERIALIZED")
        assert helper.index("WHERE embedding IS NOT NULL{where_sql}", exact_start) < helper.index(
            "LIMIT %s", exact_start
        )
        return
    if "ORDER BY" in source:
        assert source.index("candidate_privacy_node_decision") < source.rindex("ORDER BY")


def test_privacy_filtered_ann_underfill_forces_materialized_exact_rescan() -> None:
    ann_row = ("ann",)
    exact_rows = [("exact-1",), ("exact-2",)]

    class Cursor:
        def __init__(self) -> None:
            self.executions: list[tuple[str, list[object]]] = []
            self.batches = [[ann_row], exact_rows]

        def execute(self, statement: str, params: list[object]) -> None:
            self.executions.append((statement, params))

        def fetchall(self):
            return self.batches.pop(0)

    repository = GraphRepository.__new__(GraphRepository)
    repository.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    cursor = Cursor()

    rows = repository._privacy_filtered_vector_rows(
        cursor,
        query_vec_param=object(),
        where_sql=" AND candidate_privacy_node_decision(id) = %s",
        filter_params=["allow"],
        limit=2,
    )

    assert rows == exact_rows
    assert len(cursor.executions) == 2
    assert "AS MATERIALIZED" not in cursor.executions[0][0]
    assert "WITH privacy_filtered_nodes AS MATERIALIZED" in cursor.executions[1][0]
    assert cursor.executions[1][0].index("candidate_privacy_node_decision") < cursor.executions[1][
        0
    ].index("LIMIT")


@pytest.mark.parametrize(
    "method",
    [GraphRepository.find_nodes_due_for_refresh, GraphRepository.all_nodes],
)
def test_internal_enumerators_preserve_private_global_opt_out_work(method) -> None:
    source = inspect.getsource(method)
    assert "candidate_privacy_node_decision" in source
    assert "block_global" in source
    assert "tenant_id IS NOT NULL" in source


def test_lineage_filters_each_recursive_parent_before_depth_is_advanced() -> None:
    source = inspect.getsource(GraphRepository.get_lineage)
    recursive = source[source.index("WITH RECURSIVE lineage") : source.index("SELECT DISTINCT")]
    assert recursive.count("JOIN nodes parent ON parent.id = e.dst") == 2
    assert recursive.count("candidate_privacy_node_decision(parent.id)") == 4


def test_workers_recheck_privacy_before_model_and_publication() -> None:
    embedding = inspect.getsource(EmbeddingWorker._process_job)
    first_check = embedding.index("self.privacy_repository.node_decision")
    pre_model_check = embedding.index("self.privacy_repository.node_decision", first_check + 1)
    model = embedding.index("self.embedder.encode")
    pre_sink_check = embedding.rindex("self.privacy_repository.node_decision")
    sink = min(
        embedding.index("update_node_embedding"),
        embedding.index("write_embedding_history"),
        embedding.index("append_event"),
    )
    assert first_check < pre_model_check < model < pre_sink_check < sink

    extraction = inspect.getsource(ExtractionWorker._process_job)
    first_check = extraction.index("self._privacy_decision")
    pre_provider_check = extraction.index("self._privacy_decision", first_check + 1)
    provider = extraction.index("self.extraction_client.extract")
    pre_sink_check = extraction.rindex("self._privacy_decision")
    sink = extraction.index("self._update_node_props")
    assert first_check < pre_provider_check < provider < pre_sink_check < sink


class _SequencePrivacyAuthority:
    def __init__(self, *decisions: CandidatePrivacyDecision) -> None:
        self.decisions = list(decisions)

    def node_decision(self, _node_id: str) -> CandidatePrivacyDecision:
        return self.decisions.pop(0)


class _EmbeddingRepo:
    def __init__(self) -> None:
        self.skipped: list[tuple[str, str]] = []
        self.publications: list[str] = []

    def get_node(self, _node_id: str, *, tenant_id: str | None):
        return SimpleNamespace(
            id=_node_id,
            tenant_id=tenant_id,
            props={},
            embedding=None,
            payload_ref=None,
            refresh_policy={},
        )

    def mark_embedding_processing(self, *_args, **_kwargs) -> int:
        return 1

    def build_embedding_text(self, _node) -> str:
        return "synthetic candidate profile"

    def mark_embedding_skipped(self, node_id: str, reason: str, **_kwargs) -> None:
        self.skipped.append((node_id, reason))

    def update_node_embedding(self, *_args, **_kwargs) -> None:
        self.publications.append("embedding")

    def write_embedding_history(self, *_args, **_kwargs) -> None:
        self.publications.append("history")

    def append_event(self, *_args, **_kwargs) -> None:
        self.publications.append("event")


def test_embedding_race_after_model_settles_without_any_publication() -> None:
    repository = _EmbeddingRepo()
    embedder = SimpleNamespace(encode=lambda _texts: np.array([[1.0, 0.0]]))
    redis_client = SimpleNamespace()
    with (
        patch("activekg.embedding.worker.signal.signal"),
        patch("activekg.embedding.worker.clear_pending") as clear_pending,
    ):
        worker = EmbeddingWorker(
            redis_client,
            repository,
            embedder,
            privacy_repository=_SequencePrivacyAuthority(
                CandidatePrivacyDecision.ALLOW,
                CandidatePrivacyDecision.ALLOW,
                CandidatePrivacyDecision.BLOCK_ALL,
            ),
        )
        worker._process_job(json.dumps({"node_id": "node-1", "tenant_id": "org-a"}))
    assert repository.publications == []
    assert repository.skipped == [("node-1", "privacy_restricted")]
    clear_pending.assert_called_once_with(redis_client, "node-1", tenant_id="org-a")


def test_embedding_global_opt_out_blocks_public_but_not_private_work() -> None:
    assert EmbeddingWorker._privacy_blocks_node(CandidatePrivacyDecision.BLOCK_GLOBAL, None)
    assert not EmbeddingWorker._privacy_blocks_node(CandidatePrivacyDecision.BLOCK_GLOBAL, "org-a")


class _ExtractionRepo:
    def get_node(self, node_id: str, *, tenant_id: str | None):
        return SimpleNamespace(
            id=node_id,
            tenant_id=tenant_id,
            props={},
            metadata={},
        )

    def load_payload_text(self, _node) -> str:
        return "x" * 200


def test_extraction_race_after_provider_publishes_only_restricted_status() -> None:
    result = SimpleNamespace(
        confidence=0.9,
        skills_raw=[],
        primary_skills=[],
        primary_titles=[],
        recent_job_titles=[],
        to_props=lambda: {"candidate_name": "must-not-publish"},
    )
    extraction_client = SimpleNamespace(extract=lambda _text: (result, "synthetic-model"))
    updates: list[tuple[dict[str, object], bool]] = []
    with (
        patch("activekg.extraction.worker.signal.signal"),
        patch("activekg.extraction.worker.clear_extraction_pending") as clear_pending,
    ):
        worker = ExtractionWorker(
            SimpleNamespace(),
            _ExtractionRepo(),
            extraction_client,
            WorkerHealthState(1.0),
            privacy_repository=_SequencePrivacyAuthority(
                CandidatePrivacyDecision.ALLOW,
                CandidatePrivacyDecision.ALLOW,
                CandidatePrivacyDecision.REVIEW,
            ),
        )

        def capture_update(
            _node_id: str,
            _tenant_id: str | None,
            props: dict[str, object],
            *,
            enforce_privacy: bool = True,
        ) -> None:
            updates.append((props, enforce_privacy))

        with (
            patch.object(worker, "_update_node_props", side_effect=capture_update),
            patch.object(worker, "_maybe_sync_to_global_memory") as sync,
            patch.object(worker, "_trigger_reembed") as reembed,
        ):
            worker._process_job(json.dumps({"node_id": "node-2", "tenant_id": "org-a"}))
    assert len(updates) == 2
    assert updates[0][0]["extraction_status"] == "processing"
    assert updates[0][1] is True
    assert updates[1][0]["extraction_status"] == "skipped"
    assert updates[1][0]["extraction_error"] == "privacy_restricted"
    assert updates[1][1] is False
    assert all("candidate_name" not in props for props, _enforce in updates)
    sync.assert_not_called()
    reembed.assert_not_called()
    clear_pending.assert_called_once()


def test_extraction_global_opt_out_blocks_public_but_not_private_work() -> None:
    assert ExtractionWorker._privacy_blocks_node(CandidatePrivacyDecision.BLOCK_GLOBAL, None)
    assert not ExtractionWorker._privacy_blocks_node(CandidatePrivacyDecision.BLOCK_GLOBAL, "org-a")


def _function_source(source: str, qualified: str) -> str:
    tree = ast.parse(source)
    class_name, function_name = qualified.split(".", 1)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if (
                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and child.name == function_name
                ):
                    return ast.get_source_segment(source, child) or ""
    raise AssertionError(f"missing {qualified}")


@pytest.mark.parametrize(
    ("file", "qualified"),
    [
        ("activekg/graph/candidate_repository.py", "CandidateRepository.delete_candidate"),
        ("activekg/graph/repository.py", "GraphRepository.delete_node"),
        ("activekg/graph/repository.py", "GraphRepository.purge_deleted_nodes"),
    ],
)
def test_destructive_helpers_are_byte_identical_to_the_deployed_base(
    file: str, qualified: str
) -> None:
    base = subprocess.run(
        ["git", "show", f"{BASE}:{file}"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    current = (ROOT / file).read_text(encoding="utf-8")
    assert _function_source(current, qualified) == _function_source(base, qualified)


def test_increment_adds_no_destructive_candidate_data_path() -> None:
    diff = subprocess.run(
        ["git", "diff", "-U0", BASE, "--", "activekg", "scripts/*.py"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    added = "\n".join(
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    for forbidden in (
        "DELETE FROM",
        "TRUNCATE ",
        "DROP TABLE",
        ".delete_candidate(",
        ".delete_node(",
        ".purge_deleted_nodes(",
        "delete_blob(",
        "delete_file(",
    ):
        assert forbidden not in added


def test_historical_migrations_and_baseline_assets_are_byte_identical() -> None:
    for relative in (
        *(f"db/migrations/{name}" for name in MIGRATIONS[:-1]),
        "db/init.sql",
        "enable_rls_policies.sql",
    ):
        deployed = subprocess.run(
            ["git", "show", f"{BASE}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert (ROOT / relative).read_bytes() == deployed


def test_migration_023_mutates_only_its_new_authority_tables() -> None:
    migration = (ROOT / "db/migrations/023_candidate_privacy_directives.sql").read_text(
        encoding="utf-8"
    )
    targets = re.findall(
        r"^\s*(?:INSERT\s+INTO|UPDATE(?!\s+OR)|DELETE\s+FROM|"
        r"TRUNCATE(?!\s+ON)|DROP\s+TABLE)\s+"
        r"(?:public\.)?([a-z_][a-z0-9_]*)",
        migration,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    assert targets
    assert set(targets) <= {
        "candidate_privacy_directive_events",
        "candidate_privacy_directives",
        "candidate_privacy_identity_tokens",
    }
