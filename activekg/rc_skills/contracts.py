"""Closed wire grammar shared with Discover. Raw provider profiles never enter Memory."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

HORIZON_DAYS = 1095
UTC = timezone.utc
MAX_BODY_BYTES = 65536
MAX_RESPONSE_BYTES = 1048576
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Uuid = Annotated[
    str,
    Field(pattern=r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
]
_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def timestamp(value: str) -> str:
    if len(value) > 40 or not _TIME.fullmatch(value):
        raise ValueError("invalid timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("timestamp requires an offset")
    return value


def parsed_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=False)

    @field_validator("*", mode="before")
    @classmethod
    def literal_version_is_not_boolean(cls, value: Any, info: Any) -> Any:
        if info.field_name == "version" and type(value) is not int:
            raise ValueError("version must be an integer")
        return value


class Observation(StrictModel):
    version: Literal[1]
    outcome: Literal["success", "no_skills", "not_found", "unusable_age"]
    identity_hash: Hash = Field(alias="identityHash")
    skills: list[str] = Field(max_length=500)
    upstream_at: str | None = Field(alias="upstreamAt")
    observed_at: str = Field(alias="observedAt")
    valid_until: str | None = Field(alias="validUntil")
    timestamp_source: Literal["check_last_update", "fetch_updated_at"] = Field(
        alias="timestampSource"
    )
    content_hash: Hash = Field(alias="contentHash")

    @field_validator("observed_at", "upstream_at", "valid_until")
    @classmethod
    def valid_time(cls, value: str | None) -> str | None:
        if value is None:
            return None
        timestamp(value)
        canonical = (
            parsed_time(value)
            .astimezone(UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        if value != canonical:
            raise ValueError("observation times must be canonical UTC microseconds")
        return value

    @field_validator("skills")
    @classmethod
    def valid_skills(cls, values: list[str]) -> list[str]:
        for value in values:
            if (
                not value.strip()
                or not 1 <= len(value.encode("utf-16-le")) // 2 <= 500
                or any(unicodedata.category(c) in {"Cc", "Cf", "Cs"} for c in value)
            ):
                raise ValueError("invalid skill label")
        # Do not normalize/trim/enrich provider evidence in a storage authority.
        return values

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        wire = self.model_dump(by_alias=True)
        if len(canonical_json(wire).encode("utf-8")) > MAX_BODY_BYTES:
            raise ValueError("observation too large")
        expected = wire.pop("contentHash")
        if content_hash(wire) != expected:
            raise ValueError("observation hash mismatch")
        observed = parsed_time(self.observed_at)
        upstream = None if self.upstream_at is None else parsed_time(self.upstream_at)
        valid = None if self.valid_until is None else parsed_time(self.valid_until)
        if self.outcome == "success":
            if (
                not self.skills
                or upstream is None
                or valid is None
                or upstream > observed
                or valid != upstream + timedelta(days=HORIZON_DAYS)
                or valid < observed
            ):
                raise ValueError("invalid positive observation")
        elif self.skills or valid is not None:
            raise ValueError("no-data observation cannot carry positive facts")
        return self


class ReadRequest(StrictModel):
    version: Literal[1]
    candidate_ids: list[Uuid] = Field(alias="candidateIds", max_length=100)

    @field_validator("candidate_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        if len({v.lower() for v in values}) != len(values):
            raise ValueError("duplicate candidate identity")
        return values


class FinishRequest(StrictModel):
    version: Literal[1]
    global_candidate_id: Uuid = Field(alias="globalCandidateId")
    origin_run: Uuid = Field(alias="originRun")
    origin_attempt: Uuid = Field(alias="originAttempt")
    observation: Observation

    @model_validator(mode="after")
    def bounded(self) -> Self:
        if len(canonical_json(self.model_dump(by_alias=True)).encode("utf-8")) > MAX_BODY_BYTES:
            raise ValueError("request too large")
        return self


class ReadItem(StrictModel):
    global_candidate_id: Uuid = Field(alias="globalCandidateId")
    state: Literal["available", "no_data", "miss", "restricted"]
    observation: Observation | None

    @model_validator(mode="after")
    def consistent_state(self) -> Self:
        if self.state in {"miss", "restricted"}:
            if self.observation is not None:
                raise ValueError("unavailable item has evidence")
        elif self.observation is None or (
            (self.state == "available") != (self.observation.outcome == "success")
        ):
            raise ValueError("cache state differs from observation")
        return self


class ReadResponse(StrictModel):
    version: Literal[1]
    candidates: list[ReadItem] = Field(max_length=100)


class FinishResponse(StrictModel):
    version: Literal[1]
    status: Literal["stored", "replayed"]
    receipt_id: Uuid = Field(alias="receiptId")
    content_hash: Hash = Field(alias="contentHash")


TABLES = ("rc_skill_observations",)
RUNTIME_FUNCTIONS = ("memory_rc_read(text,jsonb)", "memory_rc_finish(text,uuid,jsonb)")
