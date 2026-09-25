"""Pure contracts: no API boot, provider request, or database connection."""

import base64
import json
from dataclasses import asdict, replace

import pytest
from pydantic import ValidationError

from activekg.candidate_index.contracts import (
    ACTIVE_SOURCE_KINDS,
    DIMENSION,
    IndexContractError,
    IndexPolicy,
    IndexTunables,
    SourceContentCommand,
    canonical_json,
    chunk_manifest,
    command_key,
    sha256,
    validate_vectors,
)

REFERENCE = "11111111-1111-4111-8111-111111111111"
RESUME = "22222222-2222-4222-8222-222222222222"
POLICY = IndexPolicy(
    extraction_schema_sha256="a" * 64,
    extraction_prompt_sha256="b" * 64,
    primary_model_id="fixture/primary",
    fallback_model_id="fixture/fallback",
    embedding_model_id="fixture/embedding",
    embedding_artifact_revision="c" * 40,
)


def catchup_flow_row(payload):
    """Synthetic Flow read shape, never a planted Memory receipt or authority."""
    from datetime import datetime
    from uuid import UUID

    from activekg.api import candidate_consent as c

    source = payload.source
    digest, key = c.command_identity(payload)
    captured = datetime.fromisoformat(payload.captured_at.replace("Z", "+00:00"))
    resume = source.resume.model_dump() if source.resume else None
    return {
        "source_id": UUID(source.source_id),
        "subject_id": UUID(payload.subject_id),
        "source_version": payload.version,
        "user_id": 101,
        "version": payload.version,
        "desired_action": "grant",
        "acknowledged_version": payload.version,
        "acknowledged_action": "grant",
        "current_source_id": UUID(source.source_id),
        "effective_source_id": UUID(source.source_id),
        "delivery_status": "delivered",
        "live_user": 101,
        "username": payload.proof.verified_email,
        "role": "candidate",
        "email_verified": True,
        "auth_version": 1,
        "event_user_id": 101,
        "verified_email_sha256": sha256(payload.proof.verified_email),
        "account_auth_version": 1,
        "event_id": UUID(payload.event_id),
        "action": "grant",
        "purpose": c.PURPOSE,
        "purpose_version": 1,
        "schema_version": 1,
        "copy_version": 1,
        "copy_sha256": c.COPY_SHA256,
        "command_sha256": digest,
        "outbox_digest": digest,
        "captured_at": captured,
        "approved_at": captured,
        "outbox_state": "acknowledged",
        "idempotency_key": key,
        "profile": source.profile.model_dump(),
        "profile_sha256": c.digest(c.compact(source.profile.model_dump())),
        "resume": resume,
        "resume_sha256": c.digest(c.compact(source.resume.ordered())) if source.resume else None,
        "resume_pin": resume,
        "resume_live": bool(resume),
        "gcs_locator": "gs://fixture/resume.pdf" if resume else None,
        "source_resume_id": None,
        "extracted_text_sha256": None,
    }


def test_catchup_exact_immutable_command_and_live_verified_proof():
    from activekg.api import candidate_consent as c
    from activekg.candidate_index.catchup import _command
    from tests.test_candidate_consent import vector

    for resume in (False, True):
        payload = c.ConsentCommand.model_validate(vector(resume))
        assert _command(catchup_flow_row(payload)) == payload


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        ({"live_user": None}, "source_orphaned"),
        ({"desired_action": "withdraw"}, "withdrawn_or_superseded"),
        ({"delivery_status": "pending"}, "withdrawn_or_superseded"),
        ({"version": 43}, "withdrawn_or_superseded"),
        ({"outbox_state": "pending"}, "intake_unacknowledged"),
        ({"username": "changed@fixture.invalid"}, "account_changed"),
        ({"email_verified": False}, "account_changed"),
        ({"auth_version": 2}, "account_changed"),
        ({"role": "recruiter"}, "account_changed"),
        ({"profile_sha256": "0" * 64}, "source_mismatch"),
        ({"command_sha256": "0" * 64}, "source_mismatch"),
        ({"outbox_digest": "0" * 64}, "source_mismatch"),
        ({"resume_live": False}, "source_orphaned"),
    ],
)
def test_catchup_refuses_unverifiable_historical_authority(patch, reason):
    from activekg.api import candidate_consent as c
    from activekg.candidate_index.catchup import Skip, _command
    from tests.test_candidate_consent import vector

    row = {**catchup_flow_row(c.ConsentCommand.model_validate(vector(True))), **patch}
    with pytest.raises(Skip) as error:
        _command(row)
    assert error.value.reason == reason


@pytest.fixture
def catchup_operations(monkeypatch):
    from unittest.mock import Mock

    from activekg.api import candidate_consent as c
    from activekg.candidate_index import catchup
    from activekg.candidate_index.admission import IndexConfiguration
    from tests.test_candidate_consent import vector

    config = IndexConfiguration(POLICY, IndexTunables())
    posture = catchup.Posture(
        flow_target="flow-synthetic",
        memory_target=REFERENCE,
        flow_tree="a" * 40,
        memory_tree="b" * 40,
        policy_sha256=POLICY.digest,
        running=True,
    )
    ops = catchup.Operations(
        "not-a-connection",
        "not-a-connection",
        Mock(return_value=posture),
        Mock(),
        Mock(return_value=True),
        config,
        now=lambda: 1000,
    )
    monkeypatch.setattr(catchup, "configuration", lambda: config)
    payload = c.ConsentCommand.model_validate(vector())
    row = catchup_flow_row(payload)
    entry = catchup.Entry(
        source_id=payload.source.source_id,
        subject_id=payload.subject_id,
        fingerprint="1" * 64,
        authority_sha256="2" * 64,
    )
    body = catchup.PlanBody(
        nonce=RESUME,
        issued_at=1000,
        expires_at=1060,
        posture=posture,
        cursor=catchup.ZERO,
        next_cursor=entry.source_id,
        max_rows=1,
        max_provider_attempts=5,
        scanned=1,
        entries=[entry],
        skipped={},
    )
    key = b"x" * 32
    plan = catchup.Plan(**body.model_dump(), seal=catchup._mac(key, body.model_dump()))
    return ops, plan, key, row


@pytest.mark.parametrize(
    "change", ["seal", "target", "ceiling", "source", "authority", "expiry", "key"]
)
def test_catchup_seal_binds_authority_and_bounds(catchup_operations, change):
    from activekg.candidate_index import catchup

    ops, plan, key, _ = catchup_operations
    data = plan.model_dump()
    if change == "seal":
        data["seal"] = "0" * 64
    elif change == "target":
        data["posture"]["flow_target"] = "other"
    elif change == "ceiling":
        data["max_provider_attempts"] = 500
    elif change == "source":
        data["entries"][0]["source_id"] = REFERENCE
    elif change == "authority":
        data["entries"][0]["authority_sha256"] = "3" * 64
    elif change == "expiry":
        ops.now = lambda: 1060
    elif change == "key":
        key = b"y" * 32
    with pytest.raises(catchup.CatchupRefused):
        catchup.verify_plan(data, key, ops.now())


def test_catchup_current_policy_and_live_posture_not_staged_flags(catchup_operations):
    from activekg.candidate_index import catchup

    ops, plan, _, _ = catchup_operations
    assert catchup._posture(ops) == plan.posture
    ops.observe_running.return_value = plan.posture.model_dump()
    with pytest.raises(catchup.CatchupRefused):
        catchup._posture(ops)
    ops.observe_running.return_value = plan.posture.model_copy(update={"policy_sha256": "0" * 64})
    with pytest.raises(catchup.CatchupRefused):
        catchup._posture(ops)


def test_catchup_journal_resume_torn_append_tamper_and_lock(
    catchup_operations, monkeypatch, tmp_path
):
    from contextlib import contextmanager
    from unittest.mock import Mock

    from activekg.candidate_index import catchup

    ops, plan, key, row = catchup_operations
    monkeypatch.setattr(catchup, "_flow", lambda *args: [row])
    capture = Mock(return_value=plan.entries[0])
    monkeypatch.setattr(catchup, "_memory", capture)

    @contextmanager
    def transaction(*args, **kwargs):
        yield None

    monkeypatch.setattr(catchup, "_transaction", transaction)
    journal = tmp_path / "journal"
    journal.write_bytes(b'{"seq":0,"source')
    journal.chmod(0o600)
    assert catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True) == {
        "captured": 1
    }
    assert capture.call_count == 2
    assert catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True) == {
        "captured": 1
    }
    assert capture.call_count == 2
    assert journal.stat().st_mode & 0o777 == 0o600
    with catchup._private_file(journal), pytest.raises(catchup.CatchupRefused):
        catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True)
    journal.write_bytes(journal.read_bytes().replace(b'"captured"', b'"source_orphaned"'))
    with pytest.raises(catchup.CatchupRefused):
        catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True)
    assert capture.call_count == 2


@pytest.mark.parametrize("restricted", [False, True])
def test_catchup_rechecks_privacy_after_storage_and_journals_refusal(
    catchup_operations, monkeypatch, tmp_path, restricted
):
    from unittest.mock import Mock

    from activekg.candidate_index import catchup

    ops, plan, key, row = catchup_operations
    refusal = RuntimeError("synthetic privacy refusal")
    refusal.code = "candidate_privacy_restricted" if restricted else "candidate_privacy_unavailable"
    ops.admit_flow_user = Mock(side_effect=[None, refusal])
    monkeypatch.setattr(catchup, "_flow", lambda *args: [row])
    capture = Mock()
    monkeypatch.setattr(catchup, "_memory", capture)
    expected = {"privacy_restricted" if restricted else "privacy_unavailable": 1}
    journal = tmp_path / "journal"
    assert (
        catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True)
        == expected
    )
    assert (
        catchup.execute(ops, plan=plan, key=key, journal=journal, operator_approved=True)
        == expected
    )
    assert ops.admit_flow_user.call_count == 2
    capture.assert_not_called()


def test_catchup_original_object_gate_uses_only_immutable_pin(catchup_operations):
    from activekg.api import candidate_consent as c
    from activekg.candidate_index import catchup
    from tests.test_candidate_consent import vector

    ops, _, _, _ = catchup_operations
    row = catchup_flow_row(c.ConsentCommand.model_validate(vector(True)))
    assert catchup._flow_admit(row, ops).source.resume is not None
    ops.verify_original.assert_called_once_with(
        {**row["resume_pin"], "gcs_locator": row["gcs_locator"]}
    )
    ops.verify_original.return_value = False
    with pytest.raises(catchup.Skip) as error:
        catchup._flow_admit(row, ops)
    assert error.value.reason == "object_unavailable"


@pytest.mark.parametrize("row", [(True,), {"managed": True}, (False,), {"managed": False}])
def test_legacy_hook_checks_only_the_persisted_node_and_scoped_source_tuple(row):
    from unittest.mock import Mock

    from activekg.candidate_index.repository import legacy_applicant_is_managed

    cur = Mock()
    cur.fetchone.return_value = row
    assert legacy_applicant_is_managed(cur, REFERENCE, "org_17") is (
        row["managed"] if isinstance(row, dict) else row[0]
    )
    query, args = cur.execute.call_args.args
    assert args == (REFERENCE, "org_17")
    assert "public.nodes n" in query
    assert "s.source_kind='organization_application'" in query
    assert "s.resume_version_id=e.resume_version_id" in query
    assert "s.scope_key=r.tenant_id" in query
    assert "r.job_id::text=n.metadata->>'job_id'" in query
    assert "UPDATE" not in query and "INSERT" not in query


@pytest.mark.parametrize("row", [None, (None,), ("false",), {"wrong": False}, {"managed": 1}])
def test_legacy_hook_unknown_authority_refuses_instead_of_publishing(row):
    from unittest.mock import Mock

    from activekg.candidate_index.repository import legacy_applicant_is_managed

    cur = Mock()
    cur.fetchone.return_value = row
    with pytest.raises(ValueError, match="candidate_index_legacy_authority_unavailable"):
        legacy_applicant_is_managed(cur, REFERENCE, "org_17")


def body():
    value = {
        "schema_version": 1,
        "reference_id": REFERENCE,
        "resume_version_id": RESUME,
        "application_id": 2001,
        "job_id": 1001,
        "source_version": 1,
        "content_sha256": "a" * 64,
        "byte_count": 11,
        "media_type": "application/pdf",
        "source_observed_at": "2026-09-14T00:00:00.000Z",
        "captured_at": "2026-09-14T00:00:00.000Z",
        "content_kind": "pinned_text",
        "payload_sha256": sha256("Platform engineering. Python."),
        "content": "Platform engineering. Python.",
        "privacy_subject": [{"identifier_type": "vantahire_application_id", "value": "2001"}],
    }
    value["idempotency_key"] = command_key(
        "org_17",
        REFERENCE,
        RESUME,
        1,
        value["content_sha256"],
        value["content_kind"],
        value["payload_sha256"],
    )
    return value


def parse(value):
    return SourceContentCommand.model_validate_json(json.dumps(value))


def test_key_literal_matches_flow_and_postgres():
    assert command_key("org_17", REFERENCE, RESUME, 1, "a" * 64, "pinned_text", "b" * 64) == (
        "77ed6de5ba676e0408944df3bf0fe9c7d9e7b5b9b964bb3a9c7c361153b397bf"
    )


def test_whole_command_digest_literal_matches_flow_client():
    value = body()
    value.update(content="Python database engineering", byte_count=12)
    value["payload_sha256"] = sha256(value["content"])
    value["idempotency_key"] = command_key(
        "org_17",
        REFERENCE,
        RESUME,
        1,
        value["content_sha256"],
        "pinned_text",
        value["payload_sha256"],
    )
    assert parse(value).persistent_digest("org_17") == (
        "014f6f92fed28bb3280424f45684e6b3ed6fbddcdaabc66a59e658c0987fb0f3"
    )


@pytest.mark.parametrize("stage", ["api", "extract", "embed"])
def test_source_admission_flags_are_explicit_and_cannot_leak_to_another_runtime(stage):
    from activekg.candidate_index.admission import FLAGS, enabled, require_placement

    assert not enabled(stage, {})
    assert not require_placement(stage, {})
    assert enabled(stage, {FLAGS[stage]: "true"})
    for wrong in ("", "1", "TRUE", "yes"):
        with pytest.raises(IndexContractError):
            enabled(stage, {FLAGS[stage]: wrong})
    for other, name in FLAGS.items():
        if other != stage:
            for configured in ("false", "true", ""):
                with pytest.raises(IndexContractError, match="flag_placement"):
                    require_placement(stage, {name: configured})


def test_admission_policy_uses_frozen_contracts_and_requires_an_explicit_digest():
    from pathlib import Path

    from activekg.candidate_index import admission
    from activekg.engine.model_config import DEFAULT_GROQ_FAST_MODEL, DEFAULT_GROQ_LARGE_MODEL
    from activekg.extraction.schema import ExtractionResult

    policy = IndexPolicy(
        extraction_schema_sha256=sha256(canonical_json(ExtractionResult.model_json_schema())),
        extraction_prompt_sha256=sha256(
            (Path(admission.__file__).parents[1] / "extraction" / "prompt.py").read_bytes()
        ),
        primary_model_id=DEFAULT_GROQ_FAST_MODEL,
        fallback_model_id=DEFAULT_GROQ_LARGE_MODEL,
        embedding_model_id="all-MiniLM-L6-v2",
        embedding_artifact_revision="1" * 40,
    )
    env = {"CANDIDATE_INDEX_EMBEDDING_ARTIFACT_REVISION": "1" * 40}
    with pytest.raises(IndexContractError, match="policy_not_pinned"):
        admission.configuration(env)
    env["CANDIDATE_INDEX_POLICY_SHA256"] = policy.digest
    config = admission.configuration(env)
    assert config.policy == policy and config.tunables == IndexTunables()
    assert config.posture() == {"policy_sha256": policy.digest, "tunables": asdict(IndexTunables())}
    for delta in (
        {"CANDIDATE_INDEX_EMBEDDING_ARTIFACT_REVISION": "2" * 40},
        {"CANDIDATE_INDEX_EMBEDDING_ARTIFACT_REVISION": "main"},
        {"EMBEDDING_BACKEND": "openai"},
        {"CANDIDATE_INDEX_CLAIM_LIMIT": "9"},
        {"CANDIDATE_INDEX_EXTRACTION_LEASE_MS": "55000"},
        {"CANDIDATE_INDEX_EMBEDDING_ATTEMPTS": "4"},
        {"CANDIDATE_INDEX_PENDING_TOTAL": "10001"},
        {"CANDIDATE_INDEX_POLL_MS": "0"},
        {"CANDIDATE_INDEX_POLL_MS": "1.5"},
        {"CANDIDATE_INDEX_POLL_MS": "-1"},
    ):
        with pytest.raises(IndexContractError):
            admission.configuration({**env, **delta})


@pytest.mark.parametrize(
    "position,value",
    [
        (0, "org_017"),
        (0, "org_2147483648"),
        (0, "public_provider"),
        (0, None),
        (1, "AAAAAAAA-1111-4111-8111-111111111111"),
        (3, True),
        (3, "1"),
        (3, 0),
        (3, 9007199254740992),
        (4, "A" * 64),
        (5, "url"),
        (5, "original_bytes"),
    ],
)
def test_key_refuses_ambiguous_encoding(position, value):
    args = ["org_17", REFERENCE, RESUME, 1, "a" * 64, "pinned_text", "b" * 64]
    args[position] = value
    with pytest.raises(IndexContractError):
        command_key(*args)


def test_source_key_content_and_persistent_replay_boundary():
    value = body()
    parsed = parse(value)
    parsed.validate_key("org_17")
    with pytest.raises(IndexContractError):
        parsed.validate_key("org_18")
    value["privacy_subject"].append(
        {"identifier_type": "email", "value": "fixture@example.invalid"}
    )
    assert parse(value).persistent_digest("org_17") == parsed.persistent_digest("org_17")
    value["job_id"] += 1
    assert parse(value).persistent_digest("org_17") != parsed.persistent_digest("org_17")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("source_version", True),
        ("application_id", "2001"),
        ("reference_id", "AAAAAAAA-1111-4111-8111-111111111111"),
        ("payload_sha256", "c" * 64),
        ("content", "different"),
        ("content", "\ud800"),
        ("captured_at", "2026-09-14T00:00:00"),
        ("captured_at", "2026-09-14T00:00:00.000001Z"),
        ("content_kind", "url"),
        ("content", "x" * (2 * 1024 * 1024 + 1)),
        ("privacy_subject", [{"identifier_type": "vantahire_application_id", "value": "2002"}]),
    ],
    # Explicit ids: the default id embeds each value, and the 2 MiB oversize-content case produced a single
    # 2 MB verbose-output line that froze the hosted CI runner's log pipeline (both CI jobs collecting this module).
    ids=[
        "schema_version-bool",
        "source_version-bool",
        "application_id-string",
        "reference_id-uppercase",
        "payload_sha256-wrong",
        "content-different",
        "content-lone-surrogate",
        "captured_at-naive",
        "captured_at-microseconds",
        "content_kind-url",
        "content-oversize-2MiB-plus-1",
        "privacy_subject-mismatch",
    ],
)
def test_strict_source_refusals(field, value):
    command = body()
    command[field] = value
    with pytest.raises((ValidationError, IndexContractError)):
        parse(command)


def test_no_locator_or_tenant_override():
    for field in ("url", "gcs_locator", "tenant_id", "candidate_id"):
        command = body()
        command[field] = "synthetic"
        with pytest.raises(ValidationError):
            parse(command)


def test_original_bytes_exact_length_digest_and_canonical_base64():
    command = body()
    raw = b"%PDF-fixture"
    command.update(
        content_kind="original_bytes",
        content=base64.b64encode(raw).decode(),
        byte_count=len(raw),
        content_sha256=sha256(raw),
        payload_sha256=sha256(raw),
    )
    assert parse(command).content_bytes() == raw
    command["byte_count"] += 1
    with pytest.raises(ValidationError):
        parse(command)
    command["byte_count"] -= 1
    command["content"] += "\n"
    with pytest.raises(ValidationError):
        parse(command)


def test_policy_digest_binds_every_manifest_field():
    assert POLICY.digest == sha256(canonical_json(asdict(POLICY)))
    assert replace(POLICY, primary_model_id="fixture/replacement").digest != POLICY.digest
    assert replace(POLICY, embedding_artifact_revision="d" * 40).digest != POLICY.digest
    assert "approved_provider" not in ACTIVE_SOURCE_KINDS


@pytest.mark.parametrize(
    "patch",
    [
        {"dimension": 768},
        {"minimum_confidence": 0.64},
        {"minimum_confidence": float("nan")},
        {"minimum_confidence": True},
        {"embedding_artifact_revision": "main"},
        {"embedding_artifact_revision": 1},
        {"max_chunks": 33},
        {"chunk_characters": 1201},
        {"extraction_schema_sha256": "bad"},
        {"primary_model_id": "fixture model\n"},
    ],
)
def test_policy_refuses_unpinned_or_unbounded_values(patch):
    with pytest.raises(IndexContractError):
        replace(POLICY, **patch)


@pytest.mark.parametrize(
    "patch",
    [
        {"claim_limit": 9},
        {"concurrent_per_stage": 3},
        {"concurrent_per_scope_stage": 2},
        {"extraction_attempts": 3},
        {"embedding_attempts": 4},
        {"pending_per_scope": 1001},
        {"pending_total": 10001},
        {"catchup_batch": 101},
        {"maintenance_batch": 11},
        {"extraction_lease_ms": 55000},
        {"embedding_lease_ms": 30000},
        {"poll_ms": True},
    ],
)
def test_tunables_refuse_overspend_and_unsafe_deadline(patch):
    with pytest.raises(IndexContractError):
        replace(IndexTunables(), **patch)


def test_defaults_preserve_reserved_inactive_provider_weight():
    config = IndexTunables()
    assert (config.interactive_weight, config.provider_weight, config.maintenance_weight) == (
        4,
        2,
        1,
    )
    assert config.extraction_lease_ms > config.extraction_dispatch_ms + 10000


def test_chunks_are_lossless_deterministic_and_never_silently_truncated():
    text = "Platform engineer.\nPython and PostgreSQL. " * 80
    chunks = chunk_manifest(text, POLICY)
    assert chunks == chunk_manifest(text, POLICY)
    assert "".join(row["text"] for row in chunks) == text
    assert all(len(row["text"]) <= 1200 and row["sha256"] == sha256(row["text"]) for row in chunks)
    assert [row["ordinal"] for row in chunks] == list(range(len(chunks)))
    with pytest.raises(IndexContractError, match="chunk_overflow"):
        chunk_manifest("x" * (1200 * 32 + 1), POLICY)
    assert len(chunk_manifest("x" * (1200 * 32), POLICY)) == 32


@pytest.mark.parametrize("value", ["", " ", "\ud800"])
def test_invalid_text_never_publishes(value):
    with pytest.raises(IndexContractError):
        chunk_manifest(value, POLICY)


def test_vectors_require_all_finite_normalized_chunks_in_the_exact_space():
    vector = [1.0] + [0.0] * (DIMENSION - 1)
    assert validate_vectors([vector], 1) == [vector]
    for invalid in [
        [],
        [vector[:-1]],
        [[float("nan")] + vector[1:]],
        [[True] + vector[1:]],
        [[0.0] * DIMENSION],
        [vector, vector],
    ]:
        with pytest.raises(IndexContractError):
            validate_vectors(invalid, 1)


@pytest.mark.parametrize(
    "delta",
    [
        {"query": " "},
        {"query": "x" * 2001},
        {"query": "x\x00y"},
        {"top_k": 0},
        {"top_k": 101},
        {"top_k": True},
        {"top_k": "10"},
        {"use_hybrid": "true"},
        {"use_reranker": 1},
        {"job_id": -1},
        {"tenant_id": "org_99"},
        {"compound_filter": {"email": "sentinel@example.invalid"}},
    ],
)
def test_private_search_query_bounds_and_no_caller_authority(delta):
    from activekg.candidate_index.search import SearchQuery

    with pytest.raises(ValidationError):
        SearchQuery.model_validate_json(json.dumps({"query": "Python", **delta}))


@pytest.mark.parametrize("value", [1001, "1001"])
def test_search_filter_parity_is_exact_string_equality_and_server_owned(value):
    from activekg.candidate_index.search import SearchQuery, filters_for

    query = SearchQuery(
        query="Python", metadata_filters={"source": "vantahire", "org_id": "17", "job_id": value}
    )
    assert filters_for(query, "org_17") == (
        1001,
        {"source": "vantahire", "org_id": 17, "job_id": 1001},
    )
    assert filters_for(SearchQuery(query="Python", job_id=1001), "org_17") == filters_for(
        query, "org_17"
    )


@pytest.mark.parametrize(
    "filters",
    [
        {"source": "other"},
        {"org_id": 18},
        {"org_id": True},
        {"org_id": "017"},
        {"job_id": "01"},
        {"job_id": "1001 OR true"},
        {"job_id": True},
        {"job_id": 2**32},
        {"email": "sentinel@example.invalid"},
        {"metadata->>org_id": "17"},
        {"submitted_by_recruiter": False},
        {"resume_id": 9},
    ],
)
def test_unmodelled_filters_never_silently_fall_through_to_an_unfiltered_private_lane(filters):
    from activekg.candidate_index.search import SearchQuery, SearchRefused, filters_for

    with pytest.raises(SearchRefused):
        filters_for(SearchQuery(query="Python", metadata_filters=filters), "org_17")


def test_search_rejects_conflicting_job_filters_and_non_org_tenants():
    from activekg.candidate_index.search import SearchQuery, SearchRefused, filters_for

    with pytest.raises(SearchRefused, match="filter_conflict"):
        filters_for(
            SearchQuery(query="Python", job_id=1001, metadata_filters={"job_id": 1002}), "org_17"
        )
    for tenant in ("default", "global", "org_0", "org_01", "org_2147483648"):
        with pytest.raises(SearchRefused):
            filters_for(SearchQuery(query="Python"), tenant)


def _search_hit(app, score=0.8, **kwargs):
    from activekg.candidate_index.search import Hit

    return Hit(app, 1001, score, ("Python",), "Python", score, **kwargs)


def test_rrf_is_per_application_deterministic_and_complete_managed_payload_wins():
    from activekg.candidate_index.search import fuse

    legacy = [_search_hit(2, 0.95), _search_hit(1, 0.9)]
    managed = [
        _search_hit(1, 0.8, generation=2, generation_id=REFERENCE, state="updating"),
        _search_hit(3, 0.7),
    ]
    rows = fuse(managed, legacy)
    assert [r.application_id for r in rows] == [1, 2, 3]
    assert rows[0].ranking == 1 / 61 + 1 / 62
    assert rows[0].generation == 2 and rows[0].state == "updating"
    assert rows[0].cosine == 0.8  # Not the legacy cosine and never the RRF value.
    tied = fuse([_search_hit(9)], [_search_hit(2)])
    assert [r.application_id for r in tied] == [2, 9]
    assert "document" not in rows[0].public() and "evidence" not in rows[0].public()


def test_legacy_serialization_discards_contact_locator_and_raw_props_and_groups_chunks():
    from types import SimpleNamespace

    from activekg.candidate_index.search import legacy_hits

    node = SimpleNamespace(
        id="legacy-node",
        tenant_id="org_17",
        metadata={
            "source": "vantahire",
            "org_id": 17,
            "application_id": 2001,
            "job_id": 1001,
            "gcs_path": "gs://sentinel/private.pdf",
        },
        props={
            "text": "Python. sentinel@example.invalid +1 (234) 567-8901 https://example.invalid/cv",
            "name": "sentinel-name",
        },
        embedding=[1.0] + [0.0] * 383,
        version=1,
    )
    rows = legacy_hits([(node, 0.7), (node, 0.8)], "org_17", [1.0] + [0.0] * 383)
    assert len(rows) == 1 and rows[0].matched_chunks == 2 and rows[0].cosine == 1.0
    public = json.dumps(rows[0].public())
    assert "Python" in public
    for hidden in ("sentinel", "@", "234", "gs://", "https://", "legacy-node"):
        assert hidden not in public
    changed = replace(rows[0], evidence=("different",))
    assert changed.identity != rows[0].identity


@pytest.mark.parametrize("delta", [{"tenant_id": "org_18"}, {"metadata": {"source": "provider"}}])
def test_legacy_reader_cannot_return_another_tenant_or_source(delta):
    from types import SimpleNamespace

    from activekg.candidate_index.search import SearchRefused, legacy_hits

    node = {
        "tenant_id": "org_17",
        "metadata": {"source": "vantahire", "org_id": 17, "application_id": 2001, "job_id": 1001},
        **delta,
    }
    with pytest.raises(SearchRefused):
        legacy_hits([(SimpleNamespace(**node), 0.8)], "org_17", [1.0] + [0.0] * 383)


@pytest.mark.parametrize("rerank_mode", ["success", "failure", "off"])
def test_search_honest_rerank_status_and_revalidation_after_model_work(monkeypatch, rerank_mode):
    import asyncio

    from activekg.candidate_index import search as module
    from activekg.candidate_index.providers import WorkRefused

    calls = []
    first = [_search_hit(2), _search_hit(1)]

    async def embed(query, model, revision, *, deadline):
        calls.append("embed")
        assert (model, revision) == (POLICY.embedding_model_id, POLICY.embedding_artifact_revision)
        return [1.0] + [0.0] * 383

    def retrieve(*args):
        calls.append("retrieve")
        return (
            first if calls.count("retrieve") == 1 else [first[1]],
            False,
            {"counts": {}, "bounded": False, "limit": 1000},
        )

    async def rerank(*args, **kwargs):
        calls.append("rerank")
        if rerank_mode == "failure":
            raise WorkRefused("provider_timeout")
        return [-1.0, 2.0]

    monkeypatch.setattr(module, "embed_query_once", embed)
    monkeypatch.setattr(module, "retrieve", retrieve)
    monkeypatch.setattr(module, "rerank_once", rerank)
    result = asyncio.run(
        module.search(
            module.SearchQuery(query="Python", use_reranker=rerank_mode != "off"),
            "org_17",
            None,
            POLICY,
        )
    )
    assert (
        result["reranker"]
        == {"success": "applied", "failure": "fallback", "off": "not_requested"}[rerank_mode]
    )
    assert result["score_type"] == ("cross_encoder" if rerank_mode == "success" else "rrf_fused")
    if rerank_mode == "off":
        assert calls == ["embed", "retrieve"] and not result["saturated"]
    else:
        assert calls == ["embed", "retrieve", "rerank", "retrieve"]
        assert [r["application_id"] for r in result["results"]] == [1]
        assert result["saturated"]  # A removed authority cannot masquerade as complete retrieval.


def test_unsupported_filter_refuses_before_even_query_embedding(monkeypatch):
    import asyncio

    from activekg.candidate_index import search as module

    async def forbidden(*args, **kwargs):
        pytest.fail("model called before filter admission")

    monkeypatch.setattr(module, "embed_query_once", forbidden)
    with pytest.raises(module.SearchRefused, match="filter_unsupported"):
        asyncio.run(
            module.search(
                module.SearchQuery(query="Python", metadata_filters={"resume_id": 1}),
                "org_17",
                None,
                POLICY,
            )
        )


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("ok", 200),
        ("user", 403),
        ("other_org", 403),
        ("extra_scope", 403),
        ("wrong_actor", 403),
        ("wrong_issuer", 403),
        ("writer_scope", 403),
        ("extra_body", 422),
        ("filter", 422),
        ("unavailable", 503),
    ],
)
def test_private_search_http_authority_and_closed_errors_before_model_work(
    monkeypatch, variant, expected
):
    import asyncio
    from types import SimpleNamespace

    import httpx
    from fastapi import FastAPI

    from activekg.api import auth
    from activekg.api import candidate_index as api
    from activekg.candidate_index.search import SearchRefused

    monkeypatch.setattr(auth, "JWT_ENABLED", True)
    monkeypatch.setattr(auth, "JWT_AUDIENCE", "activekg")
    values = {
        "tenant_id": "org_17",
        "issuer": "vantahire",
        "actor_type": "service",
        "actor_id": "vantahire-backend",
        "scopes": ["organization-candidate-index:read"],
    }
    changes = {
        "user": {"actor_type": "user"},
        "other_org": {"tenant_id": "global"},
        "extra_scope": {"scopes": ["organization-candidate-index:read", "search:read"]},
        "wrong_actor": {"actor_id": "signal-service"},
        "wrong_issuer": {"issuer": "signal"},
        "writer_scope": {"scopes": ["organization-candidate-source:write"]},
    }
    app = FastAPI()
    app.include_router(api.router)
    app.dependency_overrides[api.get_jwt_claims] = lambda: auth.JWTClaims(
        **{**values, **changes.get(variant, {})}
    )
    monkeypatch.setattr(api, "_repository", lambda: (None, SimpleNamespace(policy=POLICY)))
    calls = []

    async def search(query, tenant, repo, policy):
        calls.append(tenant)
        if variant == "filter":
            raise SearchRefused("candidate_index_filter_unsupported")
        if variant == "unavailable":
            raise RuntimeError("secret-sentinel-sql-or-identity")
        return {"results": [], "score_type": "rrf_fused"}

    monkeypatch.setattr(api, "search", search)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://local.invalid"
        ) as client:
            body = {"query": "Python"}
            if variant == "extra_body":
                body["email"] = "secret-sentinel@example.invalid"
            return await client.post("/organization-candidates/search", json=body)

    response = asyncio.run(run())
    assert response.status_code == expected
    assert "secret-sentinel" not in response.text
    assert calls == (["org_17"] if variant in {"ok", "filter", "unavailable"} else [])


@pytest.mark.parametrize(
    "key",
    [
        "application_id",
        "resume_id",
        "gcs_path",
        "resume_gcp_url",
        "resume_source",
        "effective_recruiter_id",
        "submitted_by_recruiter",
        "created_by_user_id",
        "provenance_type",
        "visibility",
        "consent_state",
        "consent_captured_at",
        "applicant_name",
        "applicant_email",
        "linkedin_url",
        "github_url",
        "medium_url",
        "other_links",
        "not_a_known_key",
    ],
)
def test_frozen_flow_filter_census_has_explicit_refusals_before_sql_or_model(key, monkeypatch):
    import asyncio

    from activekg.candidate_index import search as module

    async def forbidden(*_args, **_kwargs):
        pytest.fail("unmodelled filter reached query embedding")

    monkeypatch.setattr(module, "embed_query_once", forbidden)
    with pytest.raises(module.SearchRefused, match="filter_unsupported"):
        asyncio.run(
            module.search(
                module.SearchQuery(
                    query="Python", metadata_filters={key: "sentinel-private-value"}
                ),
                "org_17",
                None,
                POLICY,
            )
        )


@pytest.mark.parametrize(
    "variant", ["disabled", "enabled", "placement", "policy", "artifact", "dsn"]
)
def test_api_posture_requires_exact_placement_policy_and_cached_query_model(monkeypatch, variant):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from activekg.api import candidate_index as api
    from activekg.candidate_index.admission import FLAGS

    for name in FLAGS.values():
        monkeypatch.delenv(name, raising=False)
    cached = Mock()
    monkeypatch.setattr(api, "cached_embedding_path", cached)
    if variant in {"disabled", "placement"}:
        if variant == "placement":
            monkeypatch.setenv(FLAGS["extract"], "false")
    else:
        monkeypatch.setattr(api, "require_placement", lambda stage: stage == "api")
        monkeypatch.setattr(api, "configuration", lambda: SimpleNamespace(policy=POLICY))
        monkeypatch.setenv("ACTIVEKG_DSN", "not-used-by-this-no-connection-test")
        if variant == "policy":

            def broken():
                raise RuntimeError("secret-sentinel-policy")

            monkeypatch.setattr(api, "configuration", broken)
        if variant == "artifact":
            cached.side_effect = RuntimeError("secret-sentinel-cache-path")
        if variant == "dsn":
            monkeypatch.delenv("ACTIVEKG_DSN")
    result = api.api_configuration_problems()
    assert result == (
        [] if variant in {"disabled", "enabled"} else ["candidate_index_api_configuration_invalid"]
    )
    if variant == "enabled":
        cached.assert_called_once_with(
            POLICY.embedding_model_id, POLICY.embedding_artifact_revision
        )
    if variant in {"disabled", "placement", "policy", "dsn"}:
        cached.assert_not_called()


def test_full_main_registration_startup_and_readiness_bind_index_posture():
    import os
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    # The wildcard Deploy Path mixes this no-DB proof with real repository
    # tests. A TEST_MODE main import must not poison their process module cache
    # or overwrite the shared privacy router repository with None.
    program = textwrap.dedent("""
    import json
    from unittest.mock import Mock, patch
    from activekg.api import main
    from activekg.api.candidate_index import reader, writer
    from activekg.api.operational import ReadinessCoordinator, ReadinessResult

    assert len(main.app.routes) == 83
    registered = {route.path: route for route in main.app.routes}
    for path, authority in [
        ("/organization-candidates/source-content", writer),
        ("/organization-candidates/search", reader),
        ("/organization-candidates/index-status", reader),
    ]:
        route = registered[path]
        assert route.methods == {"POST"}
        assert authority in {dependency.call for dependency in route.dependant.dependencies}
    assert "/global-candidates/search-generations" not in registered
    problem = "candidate_index_api_configuration_invalid"
    with patch.object(main, "candidate_index_api_configuration_problems", lambda: [problem]):
        try:
            main.startup_event()
        except RuntimeError as error:
            assert str(error) == problem
        else:
            raise AssertionError("startup did not refuse")
    check = Mock(return_value=ReadinessResult(ready=False, reasons=(problem,)))
    with patch.object(main, "candidate_index_api_configuration_problems", lambda: [problem]), \\
         patch.object(main, "bounded_readiness_check", check), \\
         patch.object(main, "_readiness_coordinator", ReadinessCoordinator()):
        response = main.readyz(cache_control="no-cache")
    assert response.status_code == 503
    assert json.loads(response.body) == {"status": "not_ready", "reasons": [problem]}
    assert check.call_args.kwargs["candidate_index_problems"] == [problem]
    """)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-B", "-c", program],
        cwd=root,
        env={
            "PATH": os.environ["PATH"],
            "PYTHONPATH": str(root),
            "ACTIVEKG_TEST_NO_DB": "true",
            "JWT_ENABLED": "false",
            "LLM_ENABLED": "false",
        },
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr[-2000:]
