"""Closed authority and wire contracts. No identities, clients or model imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

TABLES = (
    "organization_candidate_history_bindings",
    "organization_candidate_history_event_state",
    "organization_candidate_history_applications",
    "organization_candidate_history_subjects",
    "organization_candidate_history_scan_state",
)
RUNTIME_FUNCTIONS = (
    "organization_candidate_history_step(integer,integer)",
    "organization_candidate_history_read(text,integer,integer,uuid,bigint,uuid,bigint)",
)
OWNER_FUNCTIONS = ("organization_candidate_history_append_only()",)
MAX_BODY_BYTES = 4096
MAX_RESPONSE_BYTES = 65536
ID = Annotated[int, Field(ge=1, le=2147483647)]
Decimal = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]{0,18})$")]
Authority = Literal[
    "eligible",
    "awaiting_binding",
    "binding_conflict",
    "privacy_restricted",
    "temporarily_unavailable",
]
WorkState = Literal[
    "waiting_reference",
    "waiting_source",
    "privacy_wait",
    "privacy_restricted",
    "binding_conflict",
    "applied",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Watermark(StrictModel):
    count: Decimal
    event_id: UUID | None
    sequence: Decimal | None

    @model_validator(mode="after")
    def shape(self) -> Self:
        if int(self.count) > 9223372036854775807:
            raise ValueError("history_watermark_invalid")
        if self.count == "0":
            if self.event_id is not None or self.sequence is not None:
                raise ValueError("history_watermark_invalid")
        elif (
            self.event_id is None
            or self.sequence is None
            or not 1 <= int(self.sequence) <= 9223372036854775807
        ):
            raise ValueError("history_watermark_invalid")
        return self


class HistoryReadRequest(StrictModel):
    schema_version: Literal[1]
    organization_id: ID
    application_id: ID
    job_id: ID
    reference_id: UUID
    expected: Watermark


@dataclass(frozen=True)
class Admission:
    state: WorkState
    authority: Authority
    reason: str


def classify_status(value: object, source_id: UUID) -> Admission:
    """Mirror the SQL status mapping, including contention before its reason.

    The embedding job's state/error is intentionally not an authority input.
    Eligible block-global decisions are already resolved by the frozen SQL.
    """
    if value is None:
        return Admission("waiting_source", "awaiting_binding", "source_missing")
    if not isinstance(value, dict) or type(value.get("eligible")) is not bool:
        return Admission("privacy_wait", "temporarily_unavailable", "status_contract_invalid")
    if value.get("contended") is True and value["eligible"] is False:
        return Admission("privacy_wait", "temporarily_unavailable", "source_contended")
    if value["eligible"] is True:
        if value.get("source_id") != str(source_id):
            return Admission("binding_conflict", "binding_conflict", "source_mismatch")
        if "reason" in value and value["reason"] is None:
            return Admission("applied", "eligible", "eligible")
    else:
        reason = value.get("reason")
        if isinstance(reason, str) and reason in {"privacy_unavailable", "privacy_review"}:
            return Admission("privacy_wait", "temporarily_unavailable", reason)
        if reason == "privacy_restricted":
            return Admission("privacy_restricted", "privacy_restricted", reason)
        if reason == "superseded":
            return Admission("waiting_source", "awaiting_binding", reason)
        if reason == "consent_changed":
            return Admission("binding_conflict", "binding_conflict", reason)
    return Admission("privacy_wait", "temporarily_unavailable", "status_contract_invalid")


class HistoryConfig(StrictModel):
    enabled: bool = False
    poll_ms: int = Field(default=2000, ge=1000, le=10000)
    new_limit: int = Field(default=50, ge=1, le=100)
    retry_limit: int = Field(default=10, ge=1, le=25)

    @classmethod
    def from_env(cls, env: dict[str, str]) -> Self:
        flag = env.get("ORG_CANDIDATE_HISTORY_ENABLED", "false")
        if flag not in {"true", "false"}:
            raise ValueError("history_configuration_invalid")
        fields = {"poll_ms": "POLL_MS", "new_limit": "NEW_LIMIT", "retry_limit": "RETRY_LIMIT"}
        values: dict[str, int | bool] = {"enabled": flag == "true"}
        for field, suffix in fields.items():
            raw = env.get("ORG_CANDIDATE_HISTORY_" + suffix)
            if raw is not None:
                if not raw.isascii() or not raw.isdecimal() or len(raw) > 5:
                    raise ValueError("history_configuration_invalid")
                values[field] = int(raw)
        return cls.model_validate(values)


class Observation(StrictModel):
    event_id: UUID
    sequence: Decimal
    occurred_at: str

    @field_validator("sequence")
    @classmethod
    def positive_sequence(cls, value: str) -> str:
        if not 1 <= int(value) <= 9223372036854775807:
            raise ValueError("history_sequence_invalid")
        return value

    @field_validator("occurred_at")
    @classmethod
    def timestamp(cls, value: str) -> str:
        from datetime import datetime

        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("history_timestamp_invalid")
        return value


class Summary(StrictModel):
    observed_stage_move_count: Decimal
    first: Observation
    latest: Observation
    latest_observed_stage_id: ID
    taxonomy_version: ID
    rubric_id: UUID | None
    rubric_version: ID | None
    rubric_approval_mode: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")] | None
    jd_digest_version: ID | None
    recommendation_action: Literal["advance", "hold", "reject"] | None
    reason_code: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_]{0,79}$")] | None

    @model_validator(mode="after")
    def shape(self) -> Self:
        if not 1 <= int(self.observed_stage_move_count) <= 9223372036854775807:
            raise ValueError("history_count_invalid")
        if int(self.first.sequence) > int(self.latest.sequence):
            raise ValueError("history_order_invalid")
        if sum(
            x is not None for x in (self.rubric_id, self.rubric_version, self.rubric_approval_mode)
        ) not in {0, 3}:
            raise ValueError("history_metadata_invalid")
        return self


class Binding(StrictModel):
    namespace: Literal["organization_private"]
    organization_id: ID
    application_id: ID
    job_id: ID
    reference_id: UUID


class Coverage(StrictModel):
    event_types: tuple[Literal["application_stage_moved"]]
    identity_basis: Literal["organization_application_reference"]
    historical_complete: Literal[False]


class Freshness(StrictModel):
    expected: Watermark
    projected: Watermark
    unresolved_count: Decimal
    status: Literal[
        "caught_up_to_observed_capture",
        "awaiting_delivery",
        "awaiting_binding",
        "projection_pending",
        "capture_gap",
        "temporarily_unavailable",
        "no_captured_stage_events",
        "history_changed_retry",
    ]


class HistoryResponse(StrictModel):
    schema_version: Literal[1]
    binding: Binding
    coverage: Coverage
    freshness: Freshness
    authority_status: Authority
    summary: Summary | None

    @model_validator(mode="after")
    def honest(self) -> Self:
        f = self.freshness
        if int(f.unresolved_count) > 9223372036854775807:
            raise ValueError("history_count_invalid")
        if self.authority_status != "eligible" and self.summary is not None:
            raise ValueError("history_authority_invalid")
        if self.summary is not None and (
            self.summary.observed_stage_move_count != f.projected.count
            or self.summary.latest.event_id != f.projected.event_id
            or self.summary.latest.sequence != f.projected.sequence
        ):
            raise ValueError("history_summary_invalid")
        if f.status == "caught_up_to_observed_capture" and (
            self.authority_status != "eligible"
            or f.expected.count == "0"
            or f.expected != f.projected
            or f.unresolved_count != "0"
            or self.summary is None
        ):
            raise ValueError("history_freshness_invalid")
        if f.status == "no_captured_stage_events" and (
            self.authority_status != "eligible"
            or f.expected.count != "0"
            or f.projected.count != "0"
            or f.unresolved_count != "0"
            or self.summary is not None
        ):
            raise ValueError("history_freshness_invalid")
        return self
