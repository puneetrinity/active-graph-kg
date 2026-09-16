"""Real generation routines on an explicitly disposable, migrated PostgreSQL 16.

All upstream application evidence is created through the real 4B receiver, not
fake receipt rows. Model results below are synthetic: no provider is contacted.
The mutable coordination reset is test-only and never disables an evidence guard.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from activekg.api import candidate_consent as consent
from activekg.api import organization_candidates as intake
from activekg.api.auth import JWTClaims
from activekg.candidate_index.contracts import (
    COORDINATION_TABLES,
    IMMUTABLE_TABLES,
    INDEX_TABLES,
    OWNER_FUNCTIONS,
    RUNTIME_FUNCTIONS,
    IndexPolicy,
    IndexTunables,
    chunk_manifest,
    sha256,
)
from activekg.candidate_index.repository import (
    INDEX_CATALOG_SHA256,
    IndexRepository,
    catalog_evidence,
    catalog_ready,
)
from activekg.privacy.identity import normalize_privacy_identifier
from activekg.privacy.models import CandidatePrivacyAction
from activekg.privacy.repository import CandidatePrivacyRepository
from scripts import init_railway_db as release
from tests.test_candidate_consent_postgres import _command, _send
from tests.test_candidate_privacy_postgres import _config, _create
from tests.test_organization_candidate_intake import _body

OWNER = os.getenv("ACTIVEKG_INDEX_TEST_OWNER_DSN")
RUNTIME = os.getenv("ACTIVEKG_INDEX_TEST_RUNTIME_DSN")
READONLY = os.getenv("ACTIVEKG_INDEX_TEST_READONLY_DSN")
pytestmark = pytest.mark.skipif(not OWNER or not RUNTIME, reason="disposable index DSNs not set")
POLICY = IndexPolicy(
    extraction_schema_sha256="1" * 64,
    extraction_prompt_sha256="2" * 64,
    primary_model_id="fixture-primary",
    fallback_model_id="fixture-fallback",
    embedding_model_id="fixture-384",
    embedding_artifact_revision="3" * 40,
)
TEXT = "Backend engineer. Python database development and distributed systems."
VECTOR = [1.0] + [0.0] * 383


@pytest.fixture(scope="module")
def target():
    assert os.getenv("ACTIVEKG_INDEX_TEST_DISPOSABLE") == "1"
    owner_info, runtime_info = map(conninfo_to_dict, (OWNER, RUNTIME))
    for info in (owner_info, runtime_info):
        assert info["dbname"].endswith("_test") and info["user"].endswith("_test")
        assert info["host"] in {"127.0.0.1", "::1"} or info["host"].startswith("/tmp/ealana-4d-")
    assert owner_info["user"] != runtime_info["user"]
    assert all(owner_info.get(k) == runtime_info.get(k) for k in ("host", "port", "dbname"))
    with psycopg.connect(OWNER, autocommit=True) as owner:
        assert owner.execute(
            "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
        ).fetchone() == (False, False)
        assert owner.execute("SELECT to_regclass('public.candidate_index_sources')").fetchone()[0]
        with owner.cursor() as cur:
            release._harden_candidate_index_runtime_privileges(cur, runtime_info["user"])
        with psycopg.connect(RUNTIME) as runtime:
            assert runtime.execute(
                "SELECT rolsuper,rolbypassrls FROM pg_roles WHERE rolname=current_user"
            ).fetchone() == (False, False)
        yield owner


@pytest.fixture(autouse=True)
def isolated_coordination(target, monkeypatch):
    monkeypatch.setenv("ACTIVEKG_DSN", RUNTIME)
    monkeypatch.setenv("CANDIDATE_PRIVACY_HMAC_ACTIVE_VERSION", "1")
    monkeypatch.setenv(
        "CANDIDATE_PRIVACY_HMAC_KEY_V1", "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQE="
    )
    monkeypatch.setenv("CANDIDATE_PRIVACY_INTAKE_ENABLED", "false")
    monkeypatch.setenv("CANDIDATE_PRIVACY_FLOW_ISSUER", "vantahire")
    monkeypatch.setenv("CANDIDATE_PRIVACY_FLOW_ACTOR_ID", "vantahire-backend")
    monkeypatch.setenv("CANDIDATE_PRIVACY_SIGNAL_ISSUER", "signal")
    monkeypatch.setenv("CANDIDATE_PRIVACY_SIGNAL_ACTOR_ID", "signal-service")
    target.execute(
        "UPDATE candidate_index_jobs SET state='cancelled',lease_token=NULL,lease_expires_at=NULL"
    )
    target.execute(
        "UPDATE candidate_index_heads SET published_generation_id=NULL,last_complete_generation_id=NULL"
    )
    target.execute("UPDATE candidate_index_scheduler SET turns=0,last_served_at=NULL")


def source(scope=None, *, policy=POLICY, priority="interactive", capture=True, tunables=None):
    app = uuid4().int % 1_000_000_000 + 1
    scope = scope or f"org_{app}"
    body = _body(
        application_id=app,
        extracted_text_sha256=sha256(TEXT),
        privacy_subject=[
            {"identifier_type": "vantahire_application_id", "value": str(app)},
            {"identifier_type": "email", "value": f"fixture-{app}@fixture.invalid"},
        ],
    )
    payload = intake.OrganizationCandidateIntakeRequest.model_validate_json(json.dumps(body))
    body["idempotency_key"] = intake.compute_idempotency_key(payload, scope)
    payload = intake.OrganizationCandidateIntakeRequest.model_validate_json(json.dumps(body))
    receipt = intake._store(
        payload,
        JWTClaims(
            tenant_id=scope,
            issuer="vantahire",
            actor_type="service",
            actor_id="vantahire-backend",
            scopes=["organization-candidate:write"],
        ),
    )
    assert receipt["delivery_status"] == "recorded"
    repo = IndexRepository(RUNTIME, tunables=tunables)
    args = {
        "kind": "organization_application",
        "upstream_id": str(payload.resume_version_id),
        "command_digest": sha256("fixture:" + str(payload.resume_version_id)),
        "content_kind": "pinned_text",
        "content": TEXT,
        "tokens": intake._privacy_tokens(payload.privacy_subject),
        "key_versions": [1],
        "policy": policy,
        "priority": priority,
        "tunables": tunables,
    }
    result = None
    if capture:
        with repo.transaction(scope) as cur:
            result = repo.capture_on_cursor(cur, **args)
    return repo, scope, result, args


def extract(repo, policy=POLICY):
    lease = repo.claim("extract")[0]
    dispatched = repo.reserve(lease, policy)
    assert dispatched and dispatched["attempt"] == 1
    assert repo.complete_extract(
        dispatched,
        namespace={"current_title": "Backend engineer", "skills_normalized": ["python"]},
        evidence=[{"field": "current_title", "text": "Backend engineer"}],
        confidence=0.9,
        model_id=policy.primary_model_id,
        chunks=chunk_manifest(TEXT, policy),
        professional_text=TEXT,
    )
    return dispatched


def publish(repo, policy=POLICY):
    extract(repo, policy)
    dispatched = repo.reserve(repo.claim("embed")[0], policy)
    assert dispatched
    assert repo.complete_embed(dispatched, [VECTOR for _ in dispatched["extraction"]["chunks"]])
    return dispatched


def source_command(target):
    from activekg.candidate_index.contracts import SourceContentCommand, command_key

    repo, scope, _, args = source(capture=False)
    target.execute("SELECT set_config('app.current_tenant_id',%s,false)", (scope,))
    e = target.execute(
        "SELECT to_jsonb(e),r.application_id,r.job_id FROM organization_candidate_resume_evidence e "
        "JOIN organization_candidate_references r USING(tenant_id,reference_id,candidate_id) "
        "WHERE e.resume_version_id=%s",
        (args["upstream_id"],),
    ).fetchone()
    evidence, app, job = e
    data = {
        "schema_version": 1,
        "reference_id": evidence["reference_id"],
        "resume_version_id": evidence["resume_version_id"],
        "application_id": app,
        "job_id": job,
        "source_version": 1,
        "content_sha256": evidence["content_sha256"],
        "byte_count": evidence["byte_count"],
        "media_type": evidence["media_type"],
        "source_observed_at": evidence["source_observed_at"],
        "captured_at": evidence["captured_at"],
        "content_kind": "pinned_text",
        "payload_sha256": sha256(TEXT),
        "content": TEXT,
        "privacy_subject": [{"identifier_type": "vantahire_application_id", "value": str(app)}],
    }
    data["idempotency_key"] = command_key(
        scope,
        data["reference_id"],
        data["resume_version_id"],
        1,
        data["content_sha256"],
        "pinned_text",
        sha256(TEXT),
    )
    return repo, scope, SourceContentCommand.model_validate_json(json.dumps(data))


def worker_transport(calls, *, confidences=(0.9,)):
    import httpx

    class Body(httpx.AsyncByteStream):
        def __init__(self, value):
            self.value = value

        async def __aiter__(self):
            yield json.dumps(self.value).encode()

    def handler(request):
        body = json.loads(request.content)
        calls.append(body["model"])
        result = {
            "current_title": "Backend engineer",
            "skills_raw": ["Python"],
            "confidence": confidences[min(len(calls) - 1, len(confidences) - 1)],
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=Body(
                {
                    "model": body["model"],
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": json.dumps(result)},
                        }
                    ],
                }
            ),
        )

    return httpx.MockTransport(handler)


def test_generation_worker_real_restricted_roles_extract_then_atomic_publication(
    target, monkeypatch
):
    from activekg.candidate_index import processing

    repo, scope, captured, _ = source()
    calls = []
    worker = processing.GenerationWorker(
        repo, POLICY, api_key="synthetic", transport=worker_transport(calls)
    )
    assert asyncio.run(worker.tick("extract")) == ["completed"]
    assert calls == [POLICY.primary_model_id]
    assert repo.status(scope, captured["source_id"])["state"] == "pending_embedding"
    assert target.execute(
        "SELECT count(*) FROM candidate_index_vectors WHERE generation_id=%s",
        (captured["generation_id"],),
    ).fetchone() == (0,)

    async def model_boundary(texts, model, revision, *, deadline):
        assert (
            texts
            and model == POLICY.embedding_model_id
            and revision == POLICY.embedding_artifact_revision
        )
        return [VECTOR for _ in texts]

    monkeypatch.setattr(processing, "embed_once", model_boundary)
    assert asyncio.run(worker.tick("embed")) == ["completed"]
    status = repo.status(scope, captured["source_id"])
    assert (
        status["state"] == "ready"
        and status["published_generation_id"] == captured["generation_id"]
    )
    assert target.execute(
        "SELECT count(*) FROM candidate_index_extractions WHERE generation_id=%s",
        (captured["generation_id"],),
    ).fetchone() == (1,)
    with repo.transaction(scope) as cur:
        cur.execute(
            "SELECT public.candidate_index_read_private(%s::vector,%s,%s,10,NULL)",
            (str(VECTOR), POLICY.embedding_model_id, POLICY.embedding_artifact_revision),
        )
        results = cur.fetchall()
    assert len(results) == 1 and results[0][0]["generation_id"] == captured["generation_id"]


def test_two_real_runtime_loops_publish_once_without_direct_tick_calls(target, monkeypatch):
    from activekg.candidate_index import processing

    repo, scope, captured, _ = source()
    calls = []
    worker = processing.GenerationWorker(
        repo, POLICY, api_key="synthetic", transport=worker_transport(calls)
    )

    async def model_boundary(texts, model, revision, *, deadline):
        assert model == POLICY.embedding_model_id and revision == POLICY.embedding_artifact_revision
        return [VECTOR for _ in texts]

    monkeypatch.setattr(processing, "embed_once", model_boundary)
    runtimes = [processing.GenerationRuntime(worker, stage) for stage in ("extract", "embed")]
    try:
        for runtime in runtimes:
            runtime.start()
        until = time.monotonic() + 10
        while repo.status(scope, captured["source_id"])["state"] != "ready":
            assert time.monotonic() < until, "runtime publication did not finish"
            time.sleep(0.025)
    finally:
        for runtime in runtimes:
            runtime.request_stop()
        for runtime in runtimes:
            runtime.close()
    assert calls == [POLICY.primary_model_id]
    assert all(not runtime._thread.is_alive() for runtime in runtimes)
    assert target.execute(
        "SELECT stage,state,attempts,lease_token FROM candidate_index_jobs "
        "WHERE generation_id=%s ORDER BY stage",
        (captured["generation_id"],),
    ).fetchall() == [("embed", "ready", 1, None), ("extract", "ready", 1, None)]
    with repo.transaction(scope) as cur:
        cur.execute(
            "SELECT public.candidate_index_read_private(%s::vector,%s,%s,10,NULL)",
            (str(VECTOR), POLICY.embedding_model_id, POLICY.embedding_artifact_revision),
        )
        rows = cur.fetchall()
    assert len(rows) == 1 and rows[0][0]["generation_id"] == captured["generation_id"]


def test_generation_worker_primary_quality_refusal_then_reserved_fallback_after_restart(target):
    from activekg.candidate_index.processing import GenerationWorker

    repo, scope, captured, _ = source()
    calls = []
    transport = worker_transport(calls, confidences=(0.4, 0.9))
    assert asyncio.run(
        GenerationWorker(repo, POLICY, api_key="synthetic", transport=transport).tick("extract")
    ) == ["low_confidence"]
    first = target.execute(
        "SELECT attempts,dispatch_deadline FROM candidate_index_jobs WHERE generation_id=%s",
        (captured["generation_id"],),
    ).fetchone()
    assert first[0] == 1 and first[1] is not None
    # New instance models a restart: no in-memory retry or attempt counter.
    assert asyncio.run(
        GenerationWorker(repo, POLICY, api_key="synthetic", transport=transport).tick("extract")
    ) == ["completed"]
    second = target.execute(
        "SELECT attempts,dispatch_deadline FROM candidate_index_jobs WHERE generation_id=%s AND stage='extract'",
        (captured["generation_id"],),
    ).fetchone()
    assert second == (2, first[1])
    assert calls == [POLICY.primary_model_id, POLICY.fallback_model_id]
    assert target.execute(
        "SELECT model_id,attempt FROM candidate_index_extractions WHERE generation_id=%s",
        (captured["generation_id"],),
    ).fetchone() == (POLICY.fallback_model_id, 2)


def test_generation_worker_fallback_quality_never_publishes_or_resets_attempts(target):
    from activekg.candidate_index.processing import GenerationWorker

    repo, scope, captured, _ = source()
    calls = []
    transport = worker_transport(calls, confidences=(0.4,))
    for _ in range(2):
        assert asyncio.run(
            GenerationWorker(repo, POLICY, api_key="synthetic", transport=transport).tick("extract")
        ) == ["low_confidence"]
    assert repo.status(scope, captured["source_id"])["state"] == "needs_review"
    assert (
        asyncio.run(
            GenerationWorker(repo, POLICY, api_key="synthetic", transport=transport).tick("extract")
        )
        == []
    )
    assert calls == [POLICY.primary_model_id, POLICY.fallback_model_id]
    assert target.execute(
        "SELECT count(*) FROM candidate_index_extractions WHERE generation_id=%s",
        (captured["generation_id"],),
    ).fetchone() == (0,)


def test_generation_worker_profile_only_has_zero_paid_attempts(target):
    import httpx

    from activekg.candidate_index.processing import GenerationWorker

    repo, grant, scope, captured = consent_source()
    worker = GenerationWorker(
        repo,
        POLICY,
        api_key="",
        transport=httpx.MockTransport(
            lambda _: pytest.fail("deterministic consent must never dispatch HTTP")
        ),
    )
    assert asyncio.run(worker.tick("extract")) == ["completed"]
    assert target.execute(
        "SELECT attempts FROM candidate_index_jobs WHERE generation_id=%s AND stage='extract'",
        (captured["generation_id"],),
    ).fetchone() == (0,)
    assert target.execute(
        "SELECT model_id FROM candidate_index_extractions WHERE generation_id=%s",
        (captured["generation_id"],),
    ).fetchone() == ("deterministic-profile:v1",)


def test_source_adopter_binds_entire_4b_tuple_and_replays(target):
    from fastapi import HTTPException

    from activekg.candidate_index.admission import IndexConfiguration, accept_source

    repo, scope, command = source_command(target)
    config = IndexConfiguration(POLICY, IndexTunables())
    first = accept_source(command, scope, repo, config)
    assert first["outcome"] == "accepted" and "state" not in first
    assert accept_source(command, scope, repo, config) == {**first, "outcome": "replayed"}
    for field, value in [
        ("application_id", command.application_id + 1),
        ("job_id", command.job_id + 1),
        ("byte_count", command.byte_count + 1),
    ]:
        with pytest.raises(HTTPException) as refused:
            accept_source(command.model_copy(update={field: value}), scope, repo, config)
        assert refused.value.status_code == 409
    assert target.execute(
        "SELECT count(*) FROM candidate_index_sources WHERE source_id=%s", (first["source_id"],)
    ).fetchone() == (1,)


def test_source_router_strict_auth_and_no_value_errors(target, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from activekg.api import candidate_index as api
    from activekg.candidate_index.admission import IndexConfiguration

    repo, scope, command = source_command(target)
    monkeypatch.setattr(
        api, "_repository", lambda: (repo, IndexConfiguration(POLICY, IndexTunables()))
    )
    monkeypatch.setattr(api.auth, "JWT_ENABLED", True)
    claims = JWTClaims(
        tenant_id=scope,
        actor_id="vantahire-backend",
        actor_type="service",
        issuer="vantahire",
        scopes=["organization-candidate-source:write"],
    )
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_jwt_claims] = lambda: claims
    with TestClient(app) as client:
        body = command.model_dump(mode="json")
        invalid = client.post(
            "/organization-candidates/source-content",
            json={**body, "url": "https://sentinel.invalid/private"},
        )
        assert invalid.status_code == 422 and "sentinel" not in invalid.text
        for field, bad in [
            ("issuer", "signal"),
            ("actor_type", "human"),
            ("actor_id", "other"),
            ("scopes", ["kg:write"]),
        ]:
            old = getattr(claims, field)
            setattr(claims, field, bad)
            assert (
                client.post("/organization-candidates/source-content", json=body).status_code == 403
            )
            setattr(claims, field, old)
        assert client.post("/organization-candidates/source-content", json=body).status_code == 201
        assert client.post("/organization-candidates/source-content", json=body).status_code == 200


def test_enabled_consent_adopter_commits_with_grant_even_when_workers_stopped(target, monkeypatch):
    from activekg.candidate_index import admission

    monkeypatch.setenv("CANDIDATE_INDEX_API_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_INDEX_EXTRACTION_ENABLED", "false")
    monkeypatch.setenv("CANDIDATE_INDEX_EMBEDDING_ENABLED", "false")
    monkeypatch.setattr(
        admission, "configuration", lambda: admission.IndexConfiguration(POLICY, IndexTunables())
    )
    command = _command()
    result = _send(command)
    assert result["outcome"] == "granted"
    target.execute(
        "SELECT set_config('app.current_tenant_id',%s,false)",
        (f"candidate_{command.subject_id}",),
    )
    source_id = command.source.source_id
    assert target.execute(
        "SELECT count(*) FROM candidate_index_sources WHERE source_id=%s", (source_id,)
    ).fetchone() == (1,)
    assert target.execute(
        "SELECT count(*) FROM candidate_index_generations WHERE source_id=%s", (source_id,)
    ).fetchone() == (1,)
    assert _send(command)["outcome"] == "replayed"
    assert target.execute(
        "SELECT count(*) FROM candidate_index_generations WHERE source_id=%s", (source_id,)
    ).fetchone() == (1,)


def test_consent_and_index_capture_roll_back_together(target, monkeypatch):
    from fastapi import HTTPException

    from activekg.candidate_index import admission

    monkeypatch.setenv("CANDIDATE_INDEX_API_ENABLED", "true")

    def refuse():
        raise ValueError("synthetic-policy-refusal")

    monkeypatch.setattr(admission, "configuration", refuse)
    command = _command()
    with pytest.raises(HTTPException) as error:
        _send(command)
    assert error.value.status_code == 503
    target.execute(
        "SELECT set_config('app.current_tenant_id',%s,false)",
        (f"candidate_{command.subject_id}",),
    )
    assert target.execute(
        "SELECT count(*) FROM candidate_consent_receipts WHERE subject_id=%s",
        (command.subject_id,),
    ).fetchone() == (0,)
    assert target.execute(
        "SELECT count(*) FROM candidate_index_sources WHERE source_id=%s",
        (command.source.source_id,),
    ).fetchone() == (0,)


def test_capture_exact_replay_conflict_and_scope(target):
    repo, scope, accepted, args = source()
    assert accepted["outcome"] == "accepted"
    with repo.transaction(scope) as cur:
        assert repo.capture_on_cursor(cur, **args) == {**accepted, "outcome": "replayed"}
    with pytest.raises(psycopg.errors.UniqueViolation), repo.transaction(scope) as cur:
        repo.capture_on_cursor(cur, **{**args, "command_digest": "f" * 64})
    assert repo.status("org_2147483647", accepted["source_id"]) is None
    assert repo.status(scope, accepted["source_id"])["state"] == "pending_extraction"


def test_attempt_reserved_separately_and_only_once(target):
    repo, _, _, _ = source()
    lease = repo.claim("extract")[0]
    assert (
        target.execute(
            "SELECT attempts FROM candidate_index_jobs WHERE job_id=%s", (lease["job_id"],)
        ).fetchone()[0]
        == 0
    )
    assert repo.reserve({**lease, "lease_token": str(uuid4())}, POLICY) is None
    dispatched = repo.reserve(lease, POLICY)
    assert dispatched["attempt"] == 1
    assert repo.reserve(lease, POLICY) is None
    assert repo.reserve(lease, replace(POLICY, primary_model_id="other")) is None


def test_complete_publish_read_and_no_partial_replay(target):
    repo, scope, accepted, _ = source()
    dispatched = publish(repo)
    assert repo.status(scope, accepted["source_id"])["state"] == "ready"
    assert not repo.complete_embed(dispatched, [VECTOR])
    with repo.transaction(scope) as cur:
        cur.execute(
            "SELECT candidate_index_read_private(%s::vector,%s,%s,100)",
            (str(VECTOR), POLICY.embedding_model_id, POLICY.embedding_artifact_revision),
        )
        rows = cur.fetchall()
    assert len(rows) == 1 and rows[0][0]["state"] == "ready" and rows[0][0]["score"] == 1
    assert "privacy" not in json.dumps(rows) and "original_bytes" not in json.dumps(rows)
    assert (
        target.execute(
            "SELECT count(*) FROM candidate_index_publication_events WHERE generation_id=%s",
            (accepted["generation_id"],),
        ).fetchone()[0]
        == 1
    )


@pytest.mark.parametrize("variant", ["managed", "wrong_job", "wrong_org", "clone", "unadopted"])
def test_legacy_applicant_hook_excludes_only_persisted_exact_managed_sources(
    target, monkeypatch, variant
):
    from types import SimpleNamespace

    from psycopg.types.json import Jsonb

    from activekg.api import global_memory as legacy
    from activekg.candidate_index.repository import legacy_applicant_is_managed

    repo, scope, _, args = source(capture=variant != "unadopted")
    with repo.transaction(scope) as cur:
        cur.execute(
            "SELECT r.application_id,r.job_id FROM organization_candidate_references r "
            "JOIN organization_candidate_resume_evidence e ON e.reference_id=r.reference_id "
            "AND e.tenant_id=r.tenant_id WHERE e.resume_version_id=%s",
            (args["upstream_id"],),
        )
        app, job = cur.fetchone()
        node = str(uuid4())
        metadata = {
            "source": "vantahire",
            "org_id": int(scope[4:]),
            "application_id": app,
            "job_id": job,
        }
        if variant == "wrong_job":
            metadata["job_id"] += 1
        if variant == "wrong_org":
            metadata["org_id"] += 1
        if variant == "clone":
            metadata["application_id"] += 1
        cur.execute(
            "INSERT INTO nodes(id,tenant_id,classes,props,metadata) VALUES(%s,%s,%s,%s,%s)",
            (node, scope, ["Resume"], Jsonb({}), Jsonb(metadata)),
        )
        assert legacy_applicant_is_managed(cur, node, scope) is (variant == "managed")
        # A forged caller tuple cannot read the managed node in this tenant.
        assert not legacy_applicant_is_managed(cur, node, "org_999999999")

    before_counts = target.execute(
        "SELECT (SELECT count(*) FROM global_candidates), (SELECT count(*) FROM candidate_provenance), "
        "(SELECT count(*) FROM tenant_candidate_access)"
    ).fetchone()
    monkeypatch.setattr(legacy, "_DSN", RUNTIME)

    class ReachedUnchangedPrivacyGate(Exception):
        pass

    def privacy_probe(*_args, **_kwargs):
        raise ReachedUnchangedPrivacyGate()

    monkeypatch.setattr(legacy, "_require_privacy_allowed", privacy_probe)

    def call_hook():
        # Hostile in-memory task metadata cannot unmanage a persisted adopted node.
        legacy.sync_applicant_to_global_memory(
            node_id=node,
            tenant_id=scope,
            node_props={},
            extracted_result=SimpleNamespace(),
            metadata={"application_id": "wrong"},
        )

    if variant == "managed":
        call_hook()  # No extraction/global write and no dependency on publication readiness.
    else:
        with pytest.raises(ReachedUnchangedPrivacyGate):
            call_hook()  # All non-adopters still take the unchanged global-use path.
    assert (
        target.execute(
            "SELECT (SELECT count(*) FROM global_candidates), (SELECT count(*) FROM candidate_provenance), "
            "(SELECT count(*) FROM tenant_candidate_access)"
        ).fetchone()
        == before_counts
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_real_private_search_and_frozen_legacy_reader_fuse_with_exact_tenant_and_job(
    target, hybrid
):
    from psycopg.types.json import Jsonb

    from activekg.candidate_index.search import SearchQuery, retrieve

    repo, scope, accepted, _ = source()
    publish(repo)
    with repo.transaction(scope) as cur:
        cur.execute(
            "SELECT r.application_id,r.job_id FROM organization_candidate_references r "
            "JOIN candidate_index_sources s ON s.reference_id=r.reference_id AND s.tenant_id=r.tenant_id "
            "WHERE s.source_id=%s",
            (accepted["source_id"],),
        )
        app, job = cur.fetchone()
        # Synthetic retained graph chunks, not fabricated production provenance.
        # The managed hit must replace this payload while the unadopted app stays.
        for app_id in (app, app + 1):
            cur.execute(
                "INSERT INTO nodes(id,tenant_id,classes,props,metadata,embedding) "
                "VALUES(%s,%s,%s,%s,%s,%s::vector)",
                (
                    str(uuid4()),
                    scope,
                    ["Resume"],
                    Jsonb({"text": "Python legacy sentinel@example.invalid"}),
                    Jsonb(
                        {
                            "source": "vantahire",
                            "org_id": int(scope[4:]),
                            "application_id": app_id,
                            "job_id": job,
                        }
                    ),
                    str(VECTOR),
                ),
            )
    query = SearchQuery(
        query="Python", use_hybrid=hybrid, use_reranker=False, metadata_filters={"job_id": str(job)}
    )
    hits, saturated, processing = retrieve(query, scope, repo, POLICY, VECTOR)
    assert len(hits) == 2 and not saturated and processing["counts"]["ready"] == 1
    managed = next(hit for hit in hits if hit.application_id == app)
    assert managed.generation_id == accepted["generation_id"] and managed.state == "ready"
    assert "sentinel" not in json.dumps([hit.public() for hit in hits])
    assert retrieve(query, "org_2147483647", repo, POLICY, VECTOR)[0] == []
    assert (
        retrieve(
            SearchQuery(query="Python", job_id=job + 1, use_hybrid=hybrid),
            scope,
            repo,
            POLICY,
            VECTOR,
        )[0]
        == []
    )
    assert (
        retrieve(query, scope, repo, replace(POLICY, embedding_artifact_revision="f" * 40), VECTOR)[
            0
        ][0].generation
        is None
    )


@pytest.mark.parametrize("variant", ["missing", "extra", "dimension", "nonunit", "string"])
def test_invalid_vectors_roll_back_every_chunk(target, variant):
    repo, _, accepted, _ = source()
    extract(repo)
    dispatched = repo.reserve(repo.claim("embed")[0], POLICY)
    vectors = {
        "missing": [],
        "extra": [VECTOR, VECTOR],
        "dimension": [VECTOR[:-1]],
        "nonunit": [[0.0] * 384],
        "string": [["1"] + [0.0] * 383],
    }[variant]
    with pytest.raises((psycopg.errors.InvalidParameterValue, psycopg.errors.CheckViolation)):
        repo.complete_embed(dispatched, vectors)
    assert (
        target.execute(
            "SELECT count(*) FROM candidate_index_vectors WHERE generation_id=%s",
            (accepted["generation_id"],),
        ).fetchone()[0]
        == 0
    )


def test_stale_lease_and_attempt_exhaustion_survive_reclaim(target):
    repo, scope, accepted, _ = source()
    first = repo.reserve(repo.claim("extract")[0], POLICY)
    target.execute(
        "UPDATE candidate_index_jobs SET lease_expires_at=clock_timestamp()-interval '1 second' WHERE job_id=%s",
        (first["job_id"],),
    )
    second = repo.reserve(repo.claim("extract")[0], POLICY)
    assert second["attempt"] == 2 and second["lease_generation"] == first["lease_generation"] + 1
    assert second["dispatch_deadline"] == first["dispatch_deadline"]
    assert not repo.fail(first, "provider_timeout")
    assert repo.fail(second, "provider_timeout")
    assert repo.claim("extract") == []
    assert repo.status(scope, accepted["source_id"])["state"] == "failed"


def test_cross_replica_scope_and_stage_capacity(target):
    repo, scope, _, _ = source()
    source(scope)
    source()
    source()
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: repo.claim("extract"), range(2)))
    leases = [lease for batch in results for lease in batch]
    leases += repo.claim("extract")
    assert len(leases) == 2 and len({lease["scope_key"] for lease in leases}) == 2
    assert repo.claim("extract") == []


def test_runtime_evidence_is_scoped_and_cannot_mutate_coordination(target):
    repo, scope, accepted, _ = source()
    for table in INDEX_TABLES:
        with pytest.raises(psycopg.errors.InsufficientPrivilege), repo.transaction(scope) as cur:
            cur.execute(sql.SQL("DELETE FROM public.{}").format(sql.Identifier(table)))
    with repo.transaction("org_2147483647") as cur:
        for table in IMMUTABLE_TABLES:
            cur.execute(sql.SQL("SELECT count(*) FROM public.{}").format(sql.Identifier(table)))
            assert cur.fetchone()[0] == 0
    for table in COORDINATION_TABLES:
        with pytest.raises(psycopg.errors.InsufficientPrivilege), repo.transaction(scope) as cur:
            cur.execute(sql.SQL("SELECT * FROM public.{}").format(sql.Identifier(table)))
    with pytest.raises(psycopg.errors.InsufficientPrivilege), repo.transaction(scope) as cur:
        cur.execute("SELECT candidate_index_read_public(10)")
    assert repo.status(scope, accepted["source_id"])["eligible"]


def test_owner_append_only_and_complete_truncate_guard(target):
    _, _, accepted, _ = source()
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState):
        target.execute(
            "DELETE FROM candidate_index_sources WHERE source_id=%s", (accepted["source_id"],)
        )
    with pytest.raises(psycopg.errors.ObjectNotInPrerequisiteState):
        target.execute(
            sql.SQL("TRUNCATE {}").format(sql.SQL(",").join(map(sql.Identifier, INDEX_TABLES)))
        )


def test_all_relations_forced_and_routines_fixed_search_path(target):
    rows = target.execute(
        "SELECT relname,relrowsecurity,relforcerowsecurity FROM pg_class WHERE relname=ANY(%s)",
        (list(INDEX_TABLES),),
    ).fetchall()
    assert len(rows) == 8 and all(row[1:] == (True, True) for row in rows)
    rows = target.execute(
        "SELECT proname,prosecdef,proconfig FROM pg_proc WHERE proname LIKE 'candidate_index_%%'"
    ).fetchall()
    assert len(rows) == 12 and all(
        row[1] and "search_path=pg_catalog, public" in row[2] for row in rows
    )


def test_exact_catalog_and_real_provisioner(target):
    role = conninfo_to_dict(RUNTIME)["user"]
    with target.cursor() as cur:
        digest, acl = catalog_evidence(cur, role)
        assert acl
        assert digest == INDEX_CATALOG_SHA256
        release._assert_candidate_index_runtime_privileges(cur, role)
        release._harden_candidate_index_runtime_privileges(cur, role)
        assert catalog_ready(cur, role)


@pytest.mark.parametrize(
    "mutation",
    [
        "ALTER TABLE candidate_index_jobs ADD COLUMN unexpected integer",
        "ALTER TABLE candidate_index_jobs ALTER COLUMN attempts SET DEFAULT 1",
        "ALTER TABLE candidate_index_jobs DROP CONSTRAINT index_job_attempts",
        "CREATE INDEX unexpected_index ON candidate_index_jobs(state)",
        "ALTER TRIGGER index_evidence_no_mutation ON candidate_index_sources RENAME TO unexpected_trigger",
        "CREATE POLICY unexpected_read ON candidate_index_sources FOR SELECT TO PUBLIC USING (true)",
        "ALTER FUNCTION candidate_index_status(uuid,boolean) SET search_path=public",
        "GRANT SELECT ON candidate_index_sources TO PUBLIC",
        "GRANT SELECT(source_id) ON candidate_index_sources TO PUBLIC",
        "GRANT EXECUTE ON FUNCTION candidate_index_read_public(integer) TO PUBLIC",
    ],
)
def test_catalog_drift_refuses_atomically(target, mutation):
    role = conninfo_to_dict(RUNTIME)["user"]
    with target.cursor() as cur:
        assert catalog_ready(cur, role)
        with target.transaction(force_rollback=True):
            cur.execute(mutation)
            assert not catalog_ready(cur, role)
        assert catalog_ready(cur, role)


def test_provisioner_repairs_column_and_broad_grants(target):
    role = conninfo_to_dict(RUNTIME)["user"]
    with target.cursor() as cur:
        cur.execute(sql.SQL("GRANT ALL ON candidate_index_jobs TO {}").format(sql.Identifier(role)))
        cur.execute(
            sql.SQL("GRANT UPDATE(source_hash) ON candidate_index_sources TO {}").format(
                sql.Identifier(role)
            )
        )
        cur.execute(
            sql.SQL("GRANT SELECT ON candidate_index_sources TO {} WITH GRANT OPTION").format(
                sql.Identifier(role)
            )
        )
        assert not catalog_ready(cur, role)
        release._harden_candidate_index_runtime_privileges(cur, role)
        assert catalog_ready(cur, role)


def test_weighted_turns_reserve_maintenance_one_in_five(target):
    repo = IndexRepository(RUNTIME)
    for _ in range(6):
        source()
    for _ in range(2):
        source(priority="maintenance")
    classes = []
    for _ in range(6):
        lease = repo.claim("extract")[0]
        classes.append(
            target.execute(
                "SELECT priority_class FROM candidate_index_jobs WHERE job_id=%s",
                (lease["job_id"],),
            ).fetchone()[0]
        )
        assert repo.fail(lease, "unsupported_format")
    assert classes == ["interactive"] * 4 + ["maintenance", "interactive"]


@pytest.mark.parametrize("fallback_confidence", [0.5, 0.9])
def test_primary_and_fallback_share_two_attempt_and_quality_bounds(target, fallback_confidence):
    repo, scope, accepted, _ = source()
    first = repo.reserve(repo.claim("extract")[0], POLICY)
    assert repo.fail(first, "low_confidence")
    second = repo.reserve(repo.claim("extract")[0], POLICY)
    assert second["attempt"] == 2 and second["dispatch_deadline"] == first["dispatch_deadline"]
    payload = {
        "namespace": {"current_title": "Backend engineer"},
        "evidence": [{"field": "current_title", "text": "Backend engineer"}],
        "confidence": fallback_confidence,
        "model_id": POLICY.fallback_model_id,
        "chunks": chunk_manifest(TEXT, POLICY),
        "professional_text": TEXT,
    }
    if fallback_confidence < POLICY.minimum_confidence:
        with pytest.raises(psycopg.errors.InvalidParameterValue):
            repo.complete_extract(second, **payload)
        assert repo.fail(second, "low_confidence")
        assert repo.status(scope, accepted["source_id"])["state"] == "needs_review"
        assert repo.claim("extract") == []
    else:
        assert repo.complete_extract(second, **payload)
        assert repo.status(scope, accepted["source_id"])["state"] == "pending_embedding"


def test_admission_counts_generations_and_durably_defers_overflow(target):
    bounds = IndexTunables(pending_per_scope=2, pending_total=4)
    repo, scope, first, _ = source(tunables=bounds)
    extract(repo)
    _, _, second, _ = source(scope, tunables=bounds)
    _, _, third, _ = source(scope, tunables=bounds)
    assert repo.status(scope, second["source_id"])["state"] == "pending_extraction"
    assert repo.status(scope, third["source_id"])["state"] == "waiting_admission"
    assert repo.catchup() == 0
    dispatched = repo.reserve(repo.claim("embed")[0], POLICY)
    assert repo.complete_embed(dispatched, [VECTOR])
    assert repo.status(scope, first["source_id"])["state"] == "ready"
    assert repo.catchup() == 1
    assert repo.status(scope, third["source_id"])["state"] == "pending_extraction"


def test_privacy_lock_contention_spends_no_attempt_and_other_scope_progresses(target):
    repo, scope, blocked, args = source()
    _, _, other, _ = source()
    token = sorted(
        args["tokens"], key=lambda t: (t["identifier_type"], str(t["key_version"]), t["token"])
    )[0]
    lock = f"candidate-privacy-token:{token['identifier_type']}:{token['key_version']}:{token['token']}"
    with psycopg.connect(OWNER) as blocker:
        blocker.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (lock,))
        lease = repo.claim("extract")[0]
        assert lease["generation_id"] == other["generation_id"]
        assert (
            target.execute(
                "SELECT attempts FROM candidate_index_jobs WHERE generation_id=%s",
                (blocked["generation_id"],),
            ).fetchone()[0]
            == 0
        )
    assert repo.status(scope, blocked["source_id"])["state"] == "pending_extraction"


def test_two_poison_scope_locks_cannot_exhaust_statement_budget(target):
    repo, _, _, a = source()
    _, _, _, b = source()
    _, _, other, _ = source()
    with psycopg.connect(OWNER) as blocker:
        for args in (a, b):
            token = sorted(
                args["tokens"],
                key=lambda t: (t["identifier_type"], str(t["key_version"]), t["token"]),
            )[0]
            blocker.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (
                    f"candidate-privacy-token:{token['identifier_type']}:{token['key_version']}:{token['token']}",
                ),
            )
        assert repo.claim("extract")[0]["generation_id"] == other["generation_id"]


def directive(args, action):
    # Exercise the shipped Memory authority, never INSERT directive/tombstone
    # rows. The later two-system proof also covers the complete Flow request path.
    identity = next(t for t in args["tokens"] if t["identifier_type"] == "vantahire_application_id")
    with psycopg.connect(OWNER) as conn:
        conn.execute("SELECT set_config('app.current_tenant_id',%s,true)", (args["scope"],))
        app = conn.execute(
            "SELECT r.application_id FROM organization_candidate_references r "
            "JOIN organization_candidate_resume_evidence e USING(tenant_id,reference_id,candidate_id) "
            "WHERE e.resume_version_id=%s",
            (args["upstream_id"],),
        ).fetchone()[0]
    assert identity["key_version"] == 1
    privacy = CandidatePrivacyRepository(RUNTIME, config=_config(1))
    try:
        return _create(
            privacy,
            request_id=uuid4(),
            action=action,
            identifiers=[normalize_privacy_identifier("vantahire_application_id", str(app))],
        )
    finally:
        privacy.close()


def test_global_opt_out_allows_private_dispatch_publication_and_read(target):
    from activekg.candidate_index.search import SearchQuery, retrieve

    repo, scope, accepted, args = source()
    directive({**args, "scope": scope}, CandidatePrivacyAction.WITHDRAW_GLOBAL_MATCHING)
    assert repo.status(scope, accepted["source_id"])["eligible"]
    publish(repo)
    assert repo.status(scope, accepted["source_id"])["state"] == "ready"
    with repo.transaction(scope) as cur:
        cur.execute(
            "SELECT candidate_index_read_private(%s::vector,%s,%s,1)",
            (str(VECTOR), POLICY.embedding_model_id, POLICY.embedding_artifact_revision),
        )
        assert len(cur.fetchall()) == 1
    hits, saturated, counts = retrieve(SearchQuery(query="Python"), scope, repo, POLICY, VECTOR)
    assert len(hits) == 1 and hits[0].generation_id == accepted["generation_id"]
    assert not saturated and counts["counts"]["ready"] == 1
    directive({**args, "scope": scope}, CandidatePrivacyAction.REQUEST_ERASURE)
    hits, _, counts = retrieve(SearchQuery(query="Python"), scope, repo, POLICY, VECTOR)
    assert hits == [] and sum(counts["counts"].values()) == 0


def test_erasure_after_dispatch_prevents_completion_and_publication(target):
    repo, scope, accepted, args = source()
    extract(repo)
    dispatched = repo.reserve(repo.claim("embed")[0], POLICY)
    directive({**args, "scope": scope}, CandidatePrivacyAction.REQUEST_ERASURE)
    assert not repo.complete_embed(dispatched, [VECTOR])
    assert repo.status(scope, accepted["source_id"])["reason"] == "privacy_restricted"
    assert repo.invalidate(scope, accepted["source_id"])
    assert (
        target.execute(
            "SELECT count(*) FROM candidate_index_vectors WHERE generation_id=%s",
            (accepted["generation_id"],),
        ).fetchone()[0]
        == 0
    )


def consent_source(*, subject=None, version=1):
    grant = _command(subject=subject, version=version)
    assert _send(grant)["outcome"] == "granted"
    scope = f"candidate_{grant.subject_id}"
    repo = IndexRepository(RUNTIME)
    with repo.transaction(scope) as cur:
        result = repo.capture_on_cursor(
            cur,
            kind="candidate_consent",
            upstream_id=str(grant.source.source_id),
            command_digest=consent.command_identity(grant)[0],
            content_kind="approved_profile",
            content=None,
            tokens=consent._privacy_tokens(grant.proof),
            key_versions=[1],
            policy=POLICY,
        )
    return repo, grant, scope, result


def publish_consent(repo, grant):
    dispatched = repo.reserve(repo.claim("extract")[0], POLICY)
    assert dispatched["attempt"] == 0
    text = grant.source.profile.headline
    assert repo.complete_extract(
        dispatched,
        namespace={"headline": text},
        evidence=[{"field": "headline", "text": text}],
        confidence=1.0,
        model_id="deterministic-profile:v1",
        chunks=chunk_manifest(text, POLICY),
        professional_text=text,
    )
    dispatched = repo.reserve(repo.claim("embed")[0], POLICY)
    assert repo.complete_embed(dispatched, [VECTOR])
    return dispatched


@pytest.mark.parametrize(
    "variant",
    ["live", "withdrawn", "account_gone", "binding_changed", "hash_changed", "crash_after_commit"],
)
def test_consent_catchup_real_receiver_evidence_and_bounded_resume(
    target, monkeypatch, tmp_path, variant
):
    from unittest.mock import Mock

    from activekg.candidate_index import catchup
    from activekg.candidate_index.admission import IndexConfiguration
    from tests.test_candidate_index import catchup_flow_row

    # Real receiver receipt/source/state; only the Flow-side read and admission
    # adapters are doubled in this matrix. The full built two-system proof owns
    # replay of those real Flow boundaries and the cross-database census SQL.
    monkeypatch.setenv("CANDIDATE_INDEX_API_ENABLED", "false")
    grant = _command()
    assert _send(grant)["outcome"] == "granted"
    scope = f"candidate_{grant.subject_id}"
    row = catchup_flow_row(grant)
    config = IndexConfiguration(POLICY, IndexTunables())
    identity = str(
        target.execute("SELECT target_id FROM activekg_schema_control.target_identity").fetchone()[
            0
        ]
    )
    posture = catchup.Posture(
        flow_target="flow-synthetic",
        memory_target=identity,
        flow_tree="a" * 40,
        memory_tree="b" * 40,
        policy_sha256=POLICY.digest,
        running=True,
    )
    ops = catchup.Operations(
        "not-a-connection", RUNTIME, lambda: posture, Mock(), Mock(return_value=True), config
    )
    monkeypatch.setattr(catchup, "configuration", lambda: config)
    monkeypatch.setattr(catchup, "_flow", lambda *args: [row])
    key = b"c" * 32
    target.execute("SELECT set_config('app.current_tenant_id',%s,false)", (scope,))
    before = target.execute(
        "SELECT to_jsonb(c) FROM candidate_consent_sources c WHERE source_id=%s",
        (grant.source.source_id,),
    ).fetchone()
    # Exercise the read-only boundary directly too, so a regression reports the
    # local synthetic failing statement rather than only the public refusal.
    payload = catchup._flow_admit(row, ops)
    with catchup._transaction(RUNTIME, identity, memory=True, readonly=True, scope=scope) as cur:
        assert (
            catchup._memory(cur, payload, row, capture=False, ops=ops).source_id
            == grant.source.source_id
        )
    plan = catchup.census(ops, key=key, max_rows=1, max_provider_attempts=5)
    assert len(plan.entries) == 1
    assert (
        target.execute(
            "SELECT count(*) FROM candidate_index_sources WHERE source_id=%s",
            (grant.source.source_id,),
        ).fetchone()[0]
        == 0
    )
    assert grant.proof.verified_email not in plan.model_dump_json()
    if variant == "withdrawn":
        assert (
            _send(_command(subject=grant.subject_id, version=2, action="withdraw"))["outcome"]
            == "withdrawn"
        )
    elif variant == "account_gone":
        row["live_user"] = None
    elif variant == "binding_changed":
        target.execute(
            "UPDATE global_candidates SET email_hash=%s WHERE id=(SELECT global_candidate_id FROM candidate_consent_state WHERE subject_id=%s)",
            (sha256(str(uuid4())), grant.subject_id),
        )
    elif variant == "hash_changed":
        row["profile_sha256"] = "0" * 64
    journal = tmp_path / "journal"
    if variant == "crash_after_commit":
        append = catchup._append
        monkeypatch.setattr(
            catchup, "_append", Mock(side_effect=RuntimeError("synthetic crash after commit"))
        )
        with pytest.raises(catchup.CatchupRefused):
            catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True)
        monkeypatch.setattr(catchup, "_append", append)
    expected = {
        "live": "captured",
        "withdrawn": "withdrawn_or_superseded",
        "account_gone": "source_orphaned",
        "binding_changed": "binding_ambiguous",
        "hash_changed": "source_mismatch",
        "crash_after_commit": "already_managed",
    }[variant]
    assert catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True) == {
        expected: 1
    }
    assert catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True) == {
        expected: 1
    }
    expected_rows = int(variant in {"live", "crash_after_commit"})
    assert (
        target.execute(
            "SELECT count(*) FROM candidate_index_sources WHERE source_id=%s",
            (grant.source.source_id,),
        ).fetchone()[0]
        == expected_rows
    )
    assert (
        target.execute(
            "SELECT count(*) FROM candidate_index_generations WHERE source_id=%s",
            (grant.source.source_id,),
        ).fetchone()[0]
        == expected_rows
    )
    assert (
        target.execute(
            "SELECT to_jsonb(c) FROM candidate_consent_sources c WHERE source_id=%s",
            (grant.source.source_id,),
        ).fetchone()
        == before
    )
    assert target.execute(
        "SELECT count(*) FROM candidate_consent_receipts WHERE subject_id=%s", (grant.subject_id,)
    ).fetchone()[0] == (2 if variant == "withdrawn" else 1)
    if expected_rows:
        assert target.execute(
            "SELECT priority_class,state,attempts FROM candidate_index_jobs WHERE generation_id=(SELECT generation_id FROM candidate_index_generations WHERE source_id=%s)",
            (grant.source.source_id,),
        ).fetchone() == ("maintenance", "pending_extraction", 0)
    ops.verify_original.assert_not_called()  # profile-only uses no storage or model


def test_profile_consent_withdrawal_stops_read_and_has_no_legacy_writes(target):
    repo, grant, scope, accepted = consent_source()
    target.execute("SELECT set_config('app.current_tenant_id',%s,false)", (scope,))
    old = target.execute(
        "SELECT to_jsonb(g) FROM global_candidates g JOIN candidate_consent_state c "
        "ON c.global_candidate_id=g.id WHERE c.subject_id=%s",
        (str(grant.subject_id),),
    ).fetchone()[0]
    publish_consent(repo, grant)
    rows = target.execute("SELECT candidate_index_read_public(100)").fetchall()
    assert any(row[0]["generation_id"] == accepted["generation_id"] for row in rows)
    assert (
        _send(_command(subject=str(grant.subject_id), version=2, action="withdraw"))["outcome"]
        == "withdrawn"
    )
    assert repo.status(scope, accepted["source_id"])["reason"] == "consent_changed"
    assert not any(
        row[0]["generation_id"] == accepted["generation_id"]
        for row in target.execute("SELECT candidate_index_read_public(100)")
    )
    assert repo.invalidate(scope, accepted["source_id"])
    assert (
        target.execute(
            "SELECT to_jsonb(g) FROM global_candidates g JOIN candidate_consent_state c "
            "ON c.global_candidate_id=g.id WHERE c.subject_id=%s",
            (str(grant.subject_id),),
        ).fetchone()[0]
        == old
    )


def test_consent_replacement_cannot_use_prior_lkg(target):
    repo, grant, scope, old = consent_source()
    publish_consent(repo, grant)
    _, _, _, new = consent_source(subject=str(grant.subject_id), version=2)
    assert repo.status(scope, old["source_id"])["reason"] == "consent_changed"
    status = repo.status(scope, new["source_id"])
    assert status["published_generation_id"] is None
    assert status["state"] == "pending_extraction"


def test_compatible_refresh_retains_lkg_but_incompatible_vectors_do_not(target):
    repo, scope, accepted, args = source()
    publish(repo)
    refresh = replace(POLICY, extraction_prompt_sha256="7" * 64)
    with repo.transaction(scope) as cur:
        new = repo.capture_on_cursor(cur, **{**args, "policy": refresh})
    assert new["generation_id"] != accepted["generation_id"]
    assert (
        repo.status(scope, accepted["source_id"])["published_generation_id"]
        == accepted["generation_id"]
    )
    changed_space = replace(refresh, embedding_artifact_revision="8" * 40)
    with repo.transaction(scope) as cur:
        repo.capture_on_cursor(cur, **{**args, "policy": changed_space})
    assert repo.status(scope, accepted["source_id"])["published_generation_id"] is None


def test_provider_kind_and_unknown_token_version_refuse_without_artifact(target):
    repo, scope, _, args = source()
    with pytest.raises(psycopg.errors.InvalidParameterValue), repo.transaction(scope) as cur:
        repo.capture_on_cursor(cur, **{**args, "kind": "approved_provider"})
    _, second_scope, _, new_args = source(capture=False)
    with pytest.raises(psycopg.errors.InsufficientPrivilege), repo.transaction(second_scope) as cur:
        repo.capture_on_cursor(cur, **{**new_args, "key_versions": [2]})
    assert (
        target.execute(
            "SELECT count(*) FROM candidate_index_sources WHERE source_id=%s",
            (new_args["upstream_id"],),
        ).fetchone()[0]
        == 0
    )


@pytest.mark.skipif(not READONLY, reason="distinct disposable read-only verifier DSN not set")
def test_genuine_readonly_role_is_scoped_and_cannot_execute_or_write(target):
    info, owner_info = map(conninfo_to_dict, (READONLY, OWNER))
    assert info["user"].endswith("_test") and info["user"] not in {
        owner_info["user"],
        conninfo_to_dict(RUNTIME)["user"],
    }
    assert all(info.get(key) == owner_info.get(key) for key in ("host", "port", "dbname"))
    _, scope_a, accepted, _ = source()
    _, scope_b, other, _ = source()
    with psycopg.connect(READONLY) as conn:
        assert conn.execute("SHOW transaction_read_only").fetchone() == ("on",)
        assert conn.execute(
            "SELECT rolsuper,rolbypassrls,rolcreaterole,rolcreatedb,rolinherit "
            "FROM pg_roles WHERE rolname=current_user"
        ).fetchone() == (False, False, False, False, False)
        assert conn.execute(
            "SELECT has_database_privilege(current_user,current_database(),'TEMP')"
        ).fetchone() == (False,)
        assert conn.execute(
            "SELECT count(*) FROM pg_auth_members WHERE member=(SELECT oid FROM pg_roles WHERE rolname=current_user)"
        ).fetchone() == (0,)
        for table in INDEX_TABLES:
            assert conn.execute(
                "SELECT has_table_privilege(current_user,%s,'SELECT'), "
                "has_table_privilege(current_user,%s,'INSERT,UPDATE,DELETE,TRUNCATE,TRIGGER,REFERENCES')",
                (table, table),
            ).fetchone() == (True, False)
        for signature in RUNTIME_FUNCTIONS + OWNER_FUNCTIONS:
            assert conn.execute(
                "SELECT has_function_privilege(current_user,%s,'EXECUTE')", (signature,)
            ).fetchone() == (False,)
        for scope, expected, excluded in ((scope_a, accepted, other), (scope_b, other, accepted)):
            conn.execute("SELECT set_config('app.current_tenant_id',%s,true)", (scope,))
            assert conn.execute(
                "SELECT source_id::text FROM candidate_index_sources WHERE source_id=ANY(%s::uuid[])",
                ([expected["source_id"], excluded["source_id"]],),
            ).fetchall() == [(expected["source_id"],)]
        conn.rollback()
        # Read-only is privileges, not merely a default that a login can unset.
        conn.execute("SET TRANSACTION READ WRITE")
        with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
            conn.execute("UPDATE candidate_index_jobs SET attempts=attempts")
        with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
            conn.execute("SELECT candidate_index_read_public(1)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege), conn.transaction():
            conn.execute("CREATE TEMP TABLE index_verifier_must_not_create(id integer)")


def test_retained_rewind_failure_restores_index_evidence_and_catalog(target, monkeypatch):
    from tests import test_deploy_path_guards as retained

    _, _, result, _ = source()
    monkeypatch.setattr(retained, "OWNER_DSN", OWNER)
    before = target.execute(
        "SELECT row_to_json(s) FROM candidate_index_sources s WHERE source_id=%s",
        (result["source_id"],),
    ).fetchone()
    with target.cursor() as cur:
        digest_before = catalog_evidence(cur, conninfo_to_dict(RUNTIME)["user"])
    ledger = target.execute(
        "SELECT filename,checksum,baselined FROM schema_migrations ORDER BY filename"
    ).fetchall()
    with pytest.raises(RuntimeError, match="injected_027_rewind_failure"):
        retained._rewind_021_tail(fail_after_consent_drop=True)
    assert (
        target.execute(
            "SELECT row_to_json(s) FROM candidate_index_sources s WHERE source_id=%s",
            (result["source_id"],),
        ).fetchone()
        == before
    )
    with target.cursor() as cur:
        assert catalog_evidence(cur, conninfo_to_dict(RUNTIME)["user"]) == digest_before
    assert (
        target.execute(
            "SELECT filename,checksum,baselined FROM schema_migrations ORDER BY filename"
        ).fetchall()
        == ledger
    )


def test_frozen_provider_accept_and_refresh_add_no_index_intent_or_generation(target, monkeypatch):
    """Lock §10.6 no-adoption regression: the frozen 4A receiver still accepts and then refreshes an
    approved-provider observation exactly as shipped, and that path writes no candidate_index source,
    generation or job. Provider adoption is reserved for 4D-2; this is not a provider-generation proof."""
    from datetime import datetime, timedelta, timezone

    from activekg.api import sourced_candidates as receiver
    from tests.test_sourced_candidates_postgres import _claims as provider_claims
    from tests.test_sourced_candidates_postgres import _payload as provider_payload

    def index_counts():
        return target.execute(
            "SELECT (SELECT count(*) FROM candidate_index_sources),"
            "(SELECT count(*) FROM candidate_index_generations),"
            "(SELECT count(*) FROM candidate_index_jobs),"
            "(SELECT count(*) FROM candidate_index_heads)"
        ).fetchone()

    def provider_counts():
        return target.execute(
            "SELECT (SELECT count(*) FROM global_candidates),"
            "(SELECT count(*) FROM global_candidate_source_observations),"
            "(SELECT count(*) FROM global_candidate_ingest_receipts)"
        ).fetchone()

    slug = f"fixture-provider-{uuid4().hex[:12]}"
    index_before, provider_before = index_counts(), provider_counts()
    observed = datetime.now(timezone.utc) - timedelta(minutes=5)
    accepted = receiver._store(
        provider_payload(970001, f"4d-noadopt-{uuid4().hex[:8]}", observed, linkedin_slug=slug),
        provider_claims(),
    )
    assert accepted["delivery_status"] == "recorded"
    after_accept = provider_counts()
    assert after_accept[0] == provider_before[0] + 1 and after_accept[1] == provider_before[1] + 1
    refreshed = receiver._store(
        provider_payload(
            970001,
            f"4d-noadopt-{uuid4().hex[:8]}",
            observed + timedelta(minutes=1),
            linkedin_slug=slug,
        ),
        provider_claims(),
    )
    assert refreshed["delivery_status"] != "replayed"
    after_refresh = provider_counts()
    assert after_refresh[0] == after_accept[0]  # same canonical person, no new identity
    assert after_refresh[1] == after_accept[1] + 1  # a real second observation was recorded
    # Shipped 4A semantics are untouched: the refreshed row waits for the legacy public embedder.
    public = target.execute(
        "SELECT public_embedding IS NULL, public_embedding_status FROM global_candidates "
        "WHERE linkedin_url LIKE %s ORDER BY updated_at DESC LIMIT 1",
        (f"%{slug}%",),
    ).fetchone()
    assert public is not None and public[0] is True and public[1] != "ready"
    # No adoption: neither accept nor refresh created any index evidence or coordination row.
    assert index_counts() == index_before


def test_public_v1_search_never_returns_a_consent_only_shell(target, monkeypatch):
    """Lock §10.6: a consent-only canonical shell created by the real consent adopter stays invisible to
    the unchanged public_v1 search (the guard freezes ``search_global_candidates`` byte-for-byte outside
    the applicant-sync function; this proves the runtime consequence)."""
    from types import SimpleNamespace

    from activekg.api import global_memory
    from activekg.candidate_index import admission
    from tests.test_public_memory_integration import _StaticEmbedder

    monkeypatch.setenv("CANDIDATE_INDEX_API_ENABLED", "true")
    monkeypatch.setenv("CANDIDATE_INDEX_EXTRACTION_ENABLED", "false")
    monkeypatch.setenv("CANDIDATE_INDEX_EMBEDDING_ENABLED", "false")
    monkeypatch.setattr(
        admission, "configuration", lambda: admission.IndexConfiguration(POLICY, IndexTunables())
    )
    command = _command()
    assert _send(command)["outcome"] == "granted"
    target.execute(
        "SELECT set_config('app.current_tenant_id',%s,false)", (f"candidate_{command.subject_id}",)
    )
    shell = target.execute(
        "SELECT gc.id, gc.embedding_status, gc.public_crustdata_person_id, gc.public_embedding IS NULL "
        "FROM global_candidates gc JOIN candidate_consent_state s ON s.global_candidate_id = gc.id "
        "WHERE s.subject_id = %s::uuid",
        (command.subject_id,),
    ).fetchone()
    target.execute("SELECT set_config('app.current_tenant_id','',false)")
    assert (
        shell is not None
        and shell[1] == "consent_pending"
        and shell[2] is None
        and shell[3] is True
    )

    # Positive control: a provider person accepted by the frozen 4A receiver and then made public-ready
    # by the owner (the legacy public embedder's outcome) IS returned by the same query, so the shell
    # assertion below proves exclusion rather than an empty index.
    from datetime import datetime, timedelta, timezone

    from activekg.api import sourced_candidates as receiver
    from activekg.embedding.global_candidates import PUBLIC_EMBED_VERSION
    from tests.test_public_memory_integration import VECTOR
    from tests.test_sourced_candidates_postgres import _claims as provider_claims
    from tests.test_sourced_candidates_postgres import _payload as provider_payload

    control_slug = f"fixture-control-{uuid4().hex[:12]}"
    assert (
        receiver._store(
            provider_payload(
                970002,
                f"4d-control-{uuid4().hex[:8]}",
                datetime.now(timezone.utc) - timedelta(minutes=5),
                linkedin_slug=control_slug,
            ),
            provider_claims(),
        )["delivery_status"]
        == "recorded"
    )
    control = target.execute(
        "UPDATE global_candidates SET public_embedding = %s::vector, public_embedding_status = 'ready', "
        "public_embed_version = %s, public_profile_observed_at = now() "
        "WHERE linkedin_url LIKE %s RETURNING id, public_crustdata_person_id",
        (VECTOR, PUBLIC_EMBED_VERSION, f"%{control_slug}%"),
    ).fetchone()
    assert control is not None and control[1] is not None

    monkeypatch.setattr(global_memory, "_DSN", RUNTIME)
    monkeypatch.setattr(global_memory, "GLOBAL_MEMORY_ENABLED", True)
    monkeypatch.setattr(global_memory, "PUBLIC_PROFILE_SEARCH_ENABLED", True)
    monkeypatch.setattr(global_memory, "_embedder", _StaticEmbedder())
    body = global_memory.GlobalCandidateSearchRequest(
        query_text="backend engineer", surface="public_v1", limit=50
    )
    result = global_memory.search_global_candidates(
        body, claims=SimpleNamespace(tenant_id=f"org_{uuid4().int % 1_000_000 + 1}")
    )
    assert result["surface"] == "public_v1"
    assert {"surface", "results", "count", "applied_limit"} <= set(result)
    returned = {str(row["id"]) for row in result["results"]}
    assert str(control[0]) in returned  # the query does return public-ready people
    assert str(shell[0]) not in returned  # ...and never the consent-only shell
