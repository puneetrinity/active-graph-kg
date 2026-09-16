"""Pure, bounded source and policy contracts. No clients, timers or network IO."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

MAX_ORIGINAL_BYTES = 5 * 1024 * 1024
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_REQUEST_BYTES = 8 * 1024 * 1024
DIMENSION = 384
MAX_CHUNKS = 32
MAX_CHUNK_CHARACTERS = 1200
DIGEST = re.compile(r"^[0-9a-f]{64}$")
UUID_TEXT = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
ORG_TENANT = re.compile(r"^org_[1-9][0-9]{0,9}$")
SOURCE_KINDS = frozenset({"organization_application", "candidate_consent", "approved_provider"})
ACTIVE_SOURCE_KINDS = frozenset({"organization_application", "candidate_consent"})
IMMUTABLE_TABLES = (
    "candidate_index_sources",
    "candidate_index_generations",
    "candidate_index_extractions",
    "candidate_index_vectors",
    "candidate_index_publication_events",
)
COORDINATION_TABLES = (
    "candidate_index_jobs",
    "candidate_index_heads",
    "candidate_index_scheduler",
)
INDEX_TABLES = IMMUTABLE_TABLES + COORDINATION_TABLES
RUNTIME_FUNCTIONS = (
    "candidate_index_status(uuid,boolean)",
    "candidate_index_capture_source(text,uuid,text,text,text,jsonb,integer[],text,text,integer,integer)",
    "candidate_index_claim(text,integer,jsonb,uuid,uuid,bigint,text)",
    "candidate_index_complete_extract(uuid,uuid,bigint,text,text,jsonb,jsonb,double precision,text,jsonb,text,text)",
    "candidate_index_complete_embed(uuid,uuid,bigint,uuid,text,text,text,jsonb)",
    "candidate_index_fail(uuid,uuid,bigint,text,integer)",
    "candidate_index_invalidate(uuid)",
    "candidate_index_read_private(vector,text,text,integer,integer)",
    "candidate_index_catchup(integer,boolean,integer,integer)",
    "candidate_index_key_versions()",
)
OWNER_FUNCTIONS = ("candidate_index_read_public(integer)", "candidate_index_append_only()")
STATES = frozenset(
    {
        "waiting_source",
        "waiting_admission",
        "pending_extraction",
        "extracting",
        "needs_review",
        "pending_embedding",
        "embedding",
        "ready",
        "failed",
        "superseded",
        "quarantined",
        "cancelled",
    }
)


class IndexContractError(ValueError):
    """Only a closed error code is safe to propagate; never include source values."""


def canonical_json(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        encoded.encode("utf-8", errors="strict")
        return encoded
    except (TypeError, ValueError, UnicodeError) as exc:
        raise IndexContractError("candidate_index_json_invalid") from exc


def sha256(value: str | bytes) -> str:
    try:
        raw = value.encode("utf-8", errors="strict") if isinstance(value, str) else value
        return hashlib.sha256(raw).hexdigest()
    except (TypeError, UnicodeError) as exc:
        raise IndexContractError("candidate_index_content_invalid") from exc


def command_key(
    tenant: str,
    reference_id: str,
    resume_version_id: str,
    source_version: int,
    content_sha256: str,
    content_kind: str,
    payload_sha256: str,
) -> str:
    if (
        not isinstance(tenant, str)
        or ORG_TENANT.fullmatch(tenant) is None
        or int(tenant[4:]) > 2_147_483_647
        or not isinstance(reference_id, str)
        or UUID_TEXT.fullmatch(reference_id) is None
        or not isinstance(resume_version_id, str)
        or UUID_TEXT.fullmatch(resume_version_id) is None
        or type(source_version) is not int
        or not 1 <= source_version <= 9_007_199_254_740_991
        or not isinstance(content_sha256, str)
        or DIGEST.fullmatch(content_sha256) is None
        or not isinstance(payload_sha256, str)
        or DIGEST.fullmatch(payload_sha256) is None
        or content_kind not in ("pinned_text", "original_bytes")
        or (content_kind == "original_bytes" and content_sha256 != payload_sha256)
    ):
        raise IndexContractError("candidate_index_identity_refused")
    return sha256(
        canonical_json(
            [
                "candidate-index:v1",
                tenant,
                reference_id,
                resume_version_id,
                source_version,
                content_sha256,
                content_kind,
                payload_sha256,
            ]
        )
    )


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PrivacyIdentifier(StrictModel):
    identifier_type: Literal["email", "phone", "vantahire_application_id", "vantahire_resume_id"]
    value: str = Field(min_length=1, max_length=2048)


class SourceContentCommand(StrictModel):
    schema_version: Literal[1]
    reference_id: UUID
    resume_version_id: UUID
    application_id: int = Field(ge=1, le=2_147_483_647)
    job_id: int = Field(ge=1, le=2_147_483_647)
    source_version: Literal[1]
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1, le=MAX_ORIGINAL_BYTES)
    media_type: Literal[
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    source_observed_at: datetime
    captured_at: datetime
    content_kind: Literal["pinned_text", "original_bytes"]
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: str = Field(min_length=1, max_length=4 * ((MAX_ORIGINAL_BYTES + 2) // 3))
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    privacy_subject: list[PrivacyIdentifier] = Field(min_length=1, max_length=4)

    @field_validator("schema_version", "source_version", mode="before")
    @classmethod
    def integer_version(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("candidate_index_version_invalid")
        return value

    @field_validator("reference_id", "resume_version_id", mode="before")
    @classmethod
    def canonical_uuid(cls, value: Any) -> Any:
        if isinstance(value, str) and UUID_TEXT.fullmatch(value) is None:
            raise ValueError("candidate_index_uuid_invalid")
        return value

    @field_validator("source_observed_at", "captured_at")
    @classmethod
    def aware_millisecond_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.microsecond % 1000 != 0:
            raise ValueError("candidate_index_timestamp_invalid")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def content_and_privacy(self) -> Self:
        raw = self.content_bytes()
        if sha256(raw) != self.payload_sha256:
            raise ValueError("candidate_index_payload_digest_mismatch")
        if self.content_kind == "original_bytes" and (
            len(raw) != self.byte_count or self.payload_sha256 != self.content_sha256
        ):
            raise ValueError("candidate_index_original_mismatch")
        pairs = [(item.identifier_type, item.value) for item in self.privacy_subject]
        if (
            len(set(pairs)) != len(pairs)
            or pairs.count(("vantahire_application_id", str(self.application_id))) != 1
        ):
            raise ValueError("candidate_index_privacy_binding_invalid")
        return self

    def content_bytes(self) -> bytes:
        if self.content_kind == "pinned_text":
            try:
                raw = self.content.encode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise IndexContractError("candidate_index_text_invalid") from exc
            if not 1 <= len(raw) <= MAX_TEXT_BYTES or b"\x00" in raw:
                raise IndexContractError("candidate_index_text_size")
            return raw
        try:
            raw = base64.b64decode(self.content, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise IndexContractError("candidate_index_base64_invalid") from exc
        if (
            not 1 <= len(raw) <= MAX_ORIGINAL_BYTES
            or base64.b64encode(raw).decode("ascii") != self.content
        ):
            raise IndexContractError("candidate_index_base64_invalid")
        return raw

    def validate_key(self, tenant: str) -> None:
        expected = command_key(
            tenant,
            str(self.reference_id),
            str(self.resume_version_id),
            self.source_version,
            self.content_sha256,
            self.content_kind,
            self.payload_sha256,
        )
        if self.idempotency_key != expected:
            raise IndexContractError("candidate_index_key_mismatch")

    def persistent_digest(self, tenant: str) -> str:
        # Match the receiver's immutable replay boundary. Transient privacy input
        # is re-admitted each time and is not an alternate persistent profile.
        value = self.model_dump(
            mode="json", exclude={"privacy_subject", "idempotency_key", "content"}
        )
        value["tenant_id"] = tenant
        for name in ("source_observed_at", "captured_at"):
            value[name] = (
                getattr(self, name).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            )
        return sha256(canonical_json(value))


@dataclass(frozen=True)
class IndexPolicy:
    """Explicit policy identity; there is no mutable model-name-only default."""

    extraction_schema_sha256: str
    extraction_prompt_sha256: str
    primary_model_id: str
    fallback_model_id: str
    embedding_model_id: str
    embedding_artifact_revision: str
    parser_version: str = "candidate-index-parser:v1"
    extraction_adapter_version: str = "candidate-index-extraction:v1"
    text_builder_version: str = "candidate-index-text:v1"
    dimension: int = DIMENSION
    max_chunks: int = MAX_CHUNKS
    chunk_characters: int = MAX_CHUNK_CHARACTERS
    minimum_confidence: float = 0.65

    def __post_init__(self) -> None:
        for value in (self.extraction_schema_sha256, self.extraction_prompt_sha256):
            if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
                raise IndexContractError("candidate_index_policy_digest_invalid")
        for value in (
            self.primary_model_id,
            self.fallback_model_id,
            self.embedding_model_id,
            self.parser_version,
            self.extraction_adapter_version,
            self.text_builder_version,
        ):
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[a-zA-Z0-9_.:/-]{1,200}", value) is None
            ):
                raise IndexContractError("candidate_index_policy_identity_invalid")
        if (
            not isinstance(self.embedding_artifact_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.embedding_artifact_revision) is None
        ):
            raise IndexContractError("candidate_index_model_revision_required")
        if (
            type(self.dimension) is not int
            or self.dimension != DIMENSION
            or type(self.max_chunks) is not int
            or not 1 <= self.max_chunks <= MAX_CHUNKS
            or type(self.chunk_characters) is not int
            or not 1 <= self.chunk_characters <= MAX_CHUNK_CHARACTERS
            or type(self.minimum_confidence) not in (int, float)
            or not math.isfinite(self.minimum_confidence)
            or not 0.65 <= self.minimum_confidence <= 1
        ):
            raise IndexContractError("candidate_index_policy_bounds")

    @property
    def digest(self) -> str:
        return sha256(canonical_json(asdict(self)))


@dataclass(frozen=True)
class IndexTunables:
    """Platform resources, never candidate/recruiter credits."""

    poll_ms: int = 1000
    claim_limit: int = 8
    concurrent_per_stage: int = 2
    concurrent_per_scope_stage: int = 1
    extraction_lease_ms: int = 120_000
    extraction_dispatch_ms: int = 45_000
    embedding_lease_ms: int = 60_000
    embedding_dispatch_ms: int = 20_000
    extraction_attempts: int = 2
    embedding_attempts: int = 3
    pending_per_scope: int = 1000
    pending_total: int = 10000
    catchup_batch: int = 100
    maintenance_batch: int = 10
    interactive_weight: int = 4
    provider_weight: int = 2
    maintenance_weight: int = 1

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in asdict(self).values()):
            raise IndexContractError("candidate_index_tunable_invalid")
        if (
            self.claim_limit > 8
            or self.concurrent_per_stage > 2
            or self.concurrent_per_scope_stage != 1
            or self.extraction_attempts > 2
            or self.embedding_attempts > 3
            or self.catchup_batch > 100
            or self.maintenance_batch > 10
            or self.pending_per_scope > 1000
            or self.pending_total > 10000
            or self.extraction_dispatch_ms > 45000
            or self.embedding_dispatch_ms > 20000
            or self.extraction_lease_ms > 300000
            or self.embedding_lease_ms > 300000
            or self.extraction_lease_ms <= self.extraction_dispatch_ms + 10000
            or self.embedding_lease_ms <= self.embedding_dispatch_ms + 10000
            or self.pending_per_scope > self.pending_total
            or max(self.interactive_weight, self.provider_weight, self.maintenance_weight) > 16
        ):
            raise IndexContractError("candidate_index_tunable_bounds")


def chunk_manifest(text: str, policy: IndexPolicy) -> list[dict[str, Any]]:
    """Lossless deterministic boundaries, no clipping and no empty publication.

    Prefer the last sentence or whitespace boundary within the limit. Preserve
    every character so the concatenated chunks remain byte-identical to input.
    """
    if not isinstance(text, str) or not text.strip():
        raise IndexContractError("candidate_index_text_invalid")
    sha256(text)  # Reject lone surrogates with the closed contract error.
    if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
        raise IndexContractError("candidate_index_text_invalid")
    result: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        if len(result) >= policy.max_chunks:
            raise IndexContractError("candidate_index_chunk_overflow")
        stop = min(start + policy.chunk_characters, len(text))
        if stop < len(text):
            section = text[start:stop]
            boundaries = list(re.finditer(r"[.!?](?:\s+)|\n+", section))
            if not boundaries:
                boundaries = list(re.finditer(r"\s+", section))
            if boundaries:
                stop = start + boundaries[-1].end()
        value = text[start:stop]
        result.append({"ordinal": len(result), "text": value, "sha256": sha256(value)})
        start = stop
    return result


def validate_vectors(vectors: Any, expected_chunks: int) -> list[list[float]]:
    if type(expected_chunks) is not int or not 1 <= expected_chunks <= MAX_CHUNKS:
        raise IndexContractError("candidate_index_chunk_count_invalid")
    if not isinstance(vectors, list) or len(vectors) != expected_chunks:
        raise IndexContractError("candidate_index_vector_count_invalid")
    result = []
    for vector in vectors:
        if (
            not isinstance(vector, list)
            or len(vector) != DIMENSION
            or any(
                type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1
                for value in vector
            )
        ):
            raise IndexContractError("candidate_index_vector_invalid")
        norm = math.fsum(value * value for value in vector)
        if not 0.999 <= norm <= 1.001:
            raise IndexContractError("candidate_index_vector_not_normalized")
        result.append([float(value) for value in vector])
    return result
