"""Private sourcing wire contracts, with no provider, SQL or authentication IO.

Tenant authority must come from the authenticated service claims, never a body
field. These contracts validate shape only: SQL and Flow must still validate
ownership, privacy, same-job exclusions and current source/generation authority.
"""

from __future__ import annotations

import json
from typing import Literal, TypeVar
from uuid import UUID

from pydantic import Field, ValidationError, field_validator, model_validator
from typing_extensions import Self

from activekg.candidate_index.contracts import StrictModel

SEARCH_BODY_BYTES = 16 * 1024
REVALIDATE_BODY_BYTES = 64 * 1024
RETRIEVAL_LIMIT = 100
MAX_APPLICATION_ID = 2_147_483_647


class SourcingContractError(ValueError):
    """Fixed error only; validation details can contain private input."""


def _text(value: str) -> str:
    if not value.strip() or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError("candidate_sourcing_text_invalid")
    return value


class SourcingRequestIdentity(StrictModel):
    version: Literal[1]
    requestId: str = Field(min_length=1, max_length=200)

    @field_validator("version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        # Literal[1] otherwise accepts True and 1.0 in Pydantic.
        if type(value) is not int:
            raise ValueError("candidate_sourcing_version_invalid")
        return value

    @field_validator("requestId")
    @classmethod
    def request_id(cls, value: str) -> str:
        return _text(value)


class SourcingEnvelope(SourcingRequestIdentity):
    externalJobId: int = Field(ge=1, le=MAX_APPLICATION_ID)


class SourcingSearchRequest(SourcingEnvelope):
    flowRunId: UUID
    artifactHash: str = Field(pattern=r"^[0-9a-f]{64}$")
    query: str = Field(min_length=1, max_length=2000)
    limit: int = Field(ge=1, le=RETRIEVAL_LIMIT)

    @field_validator("flowRunId")
    @classmethod
    def nonzero_run(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("candidate_sourcing_identity_invalid")
        return value

    @field_validator("query")
    @classmethod
    def query_text(cls, value: str) -> str:
        if not value.strip() or any(ord(c) < 32 and c not in "\r\n\t" for c in value):
            raise ValueError("candidate_sourcing_query_invalid")
        return value


class SourcingIdentity(StrictModel):
    """Exact private provenance, not a LinkedIn substitute or display DTO."""

    candidateId: UUID
    referenceId: UUID
    applicationId: int = Field(ge=1, le=MAX_APPLICATION_ID)
    applicationJobId: int = Field(ge=1, le=MAX_APPLICATION_ID)
    resumeVersionId: UUID
    sourceId: UUID
    generationId: UUID
    sourceHash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("candidateId", "referenceId", "resumeVersionId", "sourceId", "generationId")
    @classmethod
    def nonzero_identity(cls, value: UUID) -> UUID:
        if value.int == 0:
            raise ValueError("candidate_sourcing_identity_invalid")
        return value


class SourcingRevalidateRequest(SourcingEnvelope):
    items: list[SourcingIdentity] = Field(min_length=1, max_length=RETRIEVAL_LIMIT)

    @model_validator(mode="after")
    def unique_items(self) -> Self:
        keys = [item.model_dump_json() for item in self.items]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate_sourcing_duplicate_identity")
        return self


class PublicApplicantRequest(SourcingRequestIdentity):
    publicIds: list[UUID] = Field(min_length=1, max_length=100)

    @field_validator("publicIds")
    @classmethod
    def distinct_nonzero_ids(cls, values: list[UUID]) -> list[UUID]:
        if any(value.int == 0 for value in values) or len(set(values)) != len(values):
            raise ValueError("candidate_sourcing_identity_invalid")
        return values


SourcingDecision = Literal[
    "allow", "same_job", "privacy_restricted", "source_changed", "not_found", "unavailable"
]
RequestModel = TypeVar("RequestModel", bound=SourcingRequestIdentity)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_key")
        result[key] = value
    return result


def _parse(raw: bytes, model: type[RequestModel], cap: int) -> RequestModel:  # noqa: UP047 -- retain Python 3.10/3.11 worker compatibility
    if type(raw) is not bytes or not 1 <= len(raw) <= cap:
        raise SourcingContractError("candidate_sourcing_body_invalid") from None
    try:
        # Detect duplicate keys before a JSON parser can silently overwrite them.
        json.loads(raw, object_pairs_hook=_unique_object)
        return model.model_validate_json(raw)
    except (ValueError, ValidationError, UnicodeError, RecursionError):
        raise SourcingContractError("candidate_sourcing_body_invalid") from None


def parse_sourcing_search(raw: bytes) -> SourcingSearchRequest:
    return _parse(raw, SourcingSearchRequest, SEARCH_BODY_BYTES)


def parse_sourcing_revalidate(raw: bytes) -> SourcingRevalidateRequest:
    return _parse(raw, SourcingRevalidateRequest, REVALIDATE_BODY_BYTES)


def parse_public_applicants(raw: bytes) -> PublicApplicantRequest:
    return _parse(raw, PublicApplicantRequest, SEARCH_BODY_BYTES)
