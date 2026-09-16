"""Source authority and configuration, without model or storage IO.

Adoption is an API concern; extraction/embedding enablement never controls the
source commit. A worker may be stopped without losing an accepted command.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import psycopg
from fastapi import HTTPException

from activekg.api.organization_candidates import _privacy_tokens, _require_private_privacy
from activekg.candidate_index.contracts import (
    IndexContractError,
    IndexPolicy,
    IndexTunables,
    SourceContentCommand,
    canonical_json,
    sha256,
)
from activekg.candidate_index.repository import IndexRepository
from activekg.engine.model_config import (
    DEFAULT_GROQ_FAST_MODEL,
    DEFAULT_GROQ_LARGE_MODEL,
    resolve_groq_model,
)
from activekg.extraction.schema import ExtractionResult
from activekg.privacy.config import load_candidate_privacy_config

FLAGS = {
    "api": "CANDIDATE_INDEX_API_ENABLED",
    "extract": "CANDIDATE_INDEX_EXTRACTION_ENABLED",
    "embed": "CANDIDATE_INDEX_EMBEDDING_ENABLED",
}


def enabled(stage: str, env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    value = env.get(FLAGS[stage], "false")
    if value not in {"true", "false"}:
        raise IndexContractError("candidate_index_flag_invalid")
    return value == "true"


@dataclass(frozen=True)
class IndexConfiguration:
    policy: IndexPolicy
    tunables: IndexTunables

    def posture(self) -> dict[str, Any]:
        # Neither environment values nor paths/keys are returned by readiness.
        return {"policy_sha256": self.policy.digest, "tunables": asdict(self.tunables)}


def configuration(env: Mapping[str, str] | None = None) -> IndexConfiguration:
    env = os.environ if env is None else env
    if env.get("EMBEDDING_BACKEND", "sentence-transformers") != "sentence-transformers":
        raise IndexContractError("candidate_index_embedding_backend")
    revision = env.get("CANDIDATE_INDEX_EMBEDDING_ARTIFACT_REVISION", "")
    prompt_file = Path(__file__).parents[1] / "extraction" / "prompt.py"
    policy = IndexPolicy(
        extraction_schema_sha256=sha256(canonical_json(ExtractionResult.model_json_schema())),
        extraction_prompt_sha256=sha256(prompt_file.read_bytes()),
        primary_model_id=resolve_groq_model(
            "EXTRACTION_PRIMARY_MODEL", DEFAULT_GROQ_FAST_MODEL, env
        ),
        fallback_model_id=resolve_groq_model(
            "EXTRACTION_FALLBACK_MODEL", DEFAULT_GROQ_LARGE_MODEL, env
        ),
        embedding_model_id=env.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        embedding_artifact_revision=revision,
    )
    # An environment edit is not a reindex authorisation. Both API and workers
    # must match an explicitly provisioned immutable policy identity.
    if env.get("CANDIDATE_INDEX_POLICY_SHA256") != policy.digest:
        raise IndexContractError("candidate_index_policy_not_pinned")
    values = {}
    for field in fields(IndexTunables):
        value = env.get("CANDIDATE_INDEX_" + field.name.upper())
        if value is not None:
            if re.fullmatch(r"[1-9][0-9]{0,8}", value) is None:
                raise IndexContractError("candidate_index_tunable_invalid")
            values[field.name] = int(value)
    return IndexConfiguration(policy=policy, tunables=IndexTunables(**values))


def require_placement(stage: str, env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    for other, name in FLAGS.items():
        if other != stage and name in env:
            raise IndexContractError("candidate_index_flag_placement")
    active = enabled(stage, env)
    if active:
        configuration(env)
        # Only the API hashes raw identifiers. Workers use the source's retained
        # tokens inside the definer routines and preserve the shipped no-keyring
        # worker contract, including refusal if one is accidentally staged there.
        load_candidate_privacy_config(require_hmac=stage == "api")
    return active


def _evidence(cur: psycopg.Cursor, tenant: str, command: SourceContentCommand) -> None:
    # One keyed, tenant-scoped lookup proves every immutable 026 field, including
    # its receipt and saved-resume privacy identifier. No caller tuple is trusted.
    cur.execute(
        "SELECT r.application_id,r.job_id,e.version,e.content_sha256,e.byte_count,e.media_type,"
        "e.extracted_text_sha256,e.source_observed_at,e.captured_at,e.source_resume_id "
        "FROM organization_candidate_resume_evidence e "
        "JOIN organization_candidate_references r ON r.tenant_id=e.tenant_id "
        "AND r.reference_id=e.reference_id AND r.candidate_id=e.candidate_id "
        "JOIN organization_candidate_ingest_receipts i ON i.tenant_id=e.tenant_id "
        "AND i.reference_id=e.reference_id AND i.resume_version_id=e.resume_version_id "
        "AND i.candidate_id=e.candidate_id "
        "WHERE e.tenant_id=%s AND e.reference_id=%s AND e.resume_version_id=%s",
        (tenant, command.reference_id, command.resume_version_id),
    )
    row = cur.fetchone()
    if row is None:
        raise HTTPException(409, "candidate_index_source_not_received")
    expected = (
        command.application_id,
        command.job_id,
        command.source_version,
        command.content_sha256,
        command.byte_count,
        command.media_type,
    )
    if tuple(row[:6]) != expected or row[7:9] != (
        command.source_observed_at,
        command.captured_at,
    ):
        raise HTTPException(409, "candidate_index_source_tuple_conflict")
    if (command.content_kind == "pinned_text" and row[6] != command.payload_sha256) or (
        command.content_kind == "original_bytes" and row[6] is not None
    ):
        raise HTTPException(409, "candidate_index_content_pin_conflict")
    saved = [i.value for i in command.privacy_subject if i.identifier_type == "vantahire_resume_id"]
    if saved != ([] if row[9] is None else [str(row[9])]):
        raise HTTPException(422, "candidate_index_privacy_resume_mismatch")


def accept_source(
    command: SourceContentCommand, tenant: str, repo: IndexRepository, config: IndexConfiguration
) -> dict[str, str]:
    command.validate_key(tenant)
    digest = command.persistent_digest(tenant)
    with repo.transaction(tenant) as cur:
        _require_private_privacy(cur, command.privacy_subject)
        _evidence(cur, tenant, command)
        _, keys = load_candidate_privacy_config(require_hmac=True).require_hmac()
        result = repo.capture_on_cursor(
            cur,
            kind="organization_application",
            upstream_id=str(command.resume_version_id),
            command_digest=digest,
            content_kind=command.content_kind,
            content=command.content,
            tokens=_privacy_tokens(command.privacy_subject),
            key_versions=sorted(keys),
            policy=config.policy,
            tunables=config.tunables,
        )
    return {
        "outcome": result["outcome"],
        "idempotency_key": command.idempotency_key,
        "command_digest": digest,
        "reference_id": str(command.reference_id),
        "resume_version_id": str(command.resume_version_id),
        "source_id": result["source_id"],
    }


def capture_consent(cur: psycopg.Cursor, payload: Any, outcome: str, command_digest: str) -> None:
    """Runs inside the existing consent transaction, after effective state/receipt.

    Withdrawal invalidates authority at that transaction's existing state update;
    read/dispatch/completion join it, not a delayed queue invalidation. A grant must
    not depend on either worker's processing flag or a Redis notification.
    """
    if not enabled("api") or outcome != "granted":
        return
    from activekg.api.candidate_consent import _privacy_tokens as consent_tokens

    config = configuration()
    _, keys = load_candidate_privacy_config(require_hmac=True).require_hmac()
    IndexRepository.capture_on_cursor(
        cur,
        kind="candidate_consent",
        upstream_id=str(payload.source.source_id),
        command_digest=command_digest,
        content_kind="approved_profile",
        content=None,
        tokens=consent_tokens(payload.proof),
        key_versions=sorted(keys),
        policy=config.policy,
        tunables=config.tunables,
    )
