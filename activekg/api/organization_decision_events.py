"""Strict service-only receiver for tenant-private Flow decision events."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Self

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.common.logger import get_enhanced_logger

router = APIRouter(tags=["organization-decision-events"])
organization_decision_events_router = router
logger = get_enhanced_logger(__name__)

_MAX_BODY_BYTES = 64 * 1024
_FLOW_ACTOR_ENV = "ORG_DECISION_INBOX_FLOW_ACTOR_ID"


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class StageState(_StrictModel):
    stage_id: int | None = Field(default=None, ge=1, le=2_147_483_647)


class FinalStageState(_StrictModel):
    stage_id: int = Field(ge=1, le=2_147_483_647)


class OrganizationDecisionEvent(_StrictModel):
    event_id: UUID
    delivery_sequence: int = Field(ge=1)
    source_event_sequence: int = Field(ge=1)
    organization_id: int = Field(ge=1, le=2_147_483_647)
    payload_schema_version: Literal[1]
    source_system: Literal["flow"]
    subject_type: Literal["application"]
    subject_id: int = Field(ge=1, le=2_147_483_647)
    job_id: int = Field(ge=1, le=2_147_483_647)
    action_code: Literal["application_stage_moved"]
    taxonomy_version: int = Field(ge=1)
    rubric_id: UUID | None
    rubric_version: int | None = Field(ge=1)
    rubric_approval_mode: str | None = Field(
        min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$"
    )
    jd_digest_version: int | None = Field(ge=1)
    recommendation_action: Literal["advance", "hold", "reject"] | None
    reason_code: str | None = Field(
        min_length=1, max_length=80, pattern=r"^[a-z0-9][a-z0-9_]{0,79}$"
    )
    before_state: StageState
    after_state: FinalStageState
    occurred_at: datetime

    @model_validator(mode="after")
    def validate_reference_and_change(self) -> Self:
        rubric = (self.rubric_id, self.rubric_version, self.rubric_approval_mode)
        populated = sum(value is not None for value in rubric)
        if populated not in {0, 3}:
            raise ValueError("rubric reference must be complete")
        if self.before_state.stage_id == self.after_state.stage_id:
            raise ValueError("decision state must change")
        return self


def decision_inbox_enabled() -> bool:
    return os.getenv("ORG_DECISION_INBOX_ENABLED", "false") == "true"


def _trusted_flow_actor() -> str:
    actor = os.getenv(_FLOW_ACTOR_ENV, "vantahire-backend").strip()
    if not actor or len(actor) > 160:
        raise HTTPException(status_code=503, detail="decision_inbox_configuration_invalid")
    return actor


async def require_decision_history_writer(
    claims: JWTClaims | None = Depends(get_jwt_claims),
) -> JWTClaims:
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(status_code=401, detail="decision_inbox_service_auth_required")
    if (
        claims.issuer != auth.JWT_ISSUER
        or claims.actor_type != "service"
        or claims.actor_id != _trusted_flow_actor()
        or "decision-history:write" not in claims.scopes
    ):
        raise HTTPException(status_code=403, detail="decision_inbox_service_auth_denied")
    return claims


async def _parse_body(request: Request) -> OrganizationDecisionEvent:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="decision_inbox_request_too_large")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="decision_inbox_request_invalid") from exc
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="decision_inbox_request_too_large")
    try:
        return OrganizationDecisionEvent.model_validate_json(bytes(body))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="decision_inbox_request_invalid") from exc


def _canonical_payload(payload: OrganizationDecisionEvent) -> tuple[str, str]:
    canonical = json.dumps(
        payload.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _connect() -> psycopg.Connection:
    dsn = (os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        raise RuntimeError("decision inbox database unavailable")
    return psycopg.connect(dsn, autocommit=False)


def _store(payload: OrganizationDecisionEvent, tenant_id: str) -> Literal["inserted", "replayed"]:
    canonical, digest = _canonical_payload(payload)
    del canonical
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT set_config('app.current_tenant_id', %s, true)", (tenant_id,))
            # Serializes stream creation and advancement without a cross-tenant
            # table lock. The verified tenant is the sole advisory-lock input.
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (tenant_id,))
            cur.execute(
                """
                SELECT event_id::text, delivery_sequence, source_event_sequence, payload_digest
                FROM organization_decision_event_inbox
                WHERE event_id=%s OR delivery_sequence=%s OR source_event_sequence=%s
                """,
                (payload.event_id, payload.delivery_sequence, payload.source_event_sequence),
            )
            existing = cur.fetchone()
            if existing is not None:
                exact = (
                    existing[0] == str(payload.event_id)
                    and int(existing[1]) == payload.delivery_sequence
                    and int(existing[2]) == payload.source_event_sequence
                    and str(existing[3]) == digest
                )
                if not exact:
                    raise HTTPException(status_code=409, detail="decision_inbox_event_conflict")
                conn.rollback()
                return "replayed"

            cur.execute(
                """
                SELECT last_delivery_sequence,last_source_event_sequence
                FROM organization_decision_stream_state
                WHERE tenant_id=%s FOR UPDATE
                """,
                (tenant_id,),
            )
            stream = cur.fetchone()
            if stream is not None and (
                payload.delivery_sequence <= int(stream[0])
                or payload.source_event_sequence <= int(stream[1])
            ):
                raise HTTPException(status_code=409, detail="decision_inbox_sequence_conflict")

            cur.execute(
                """
                INSERT INTO organization_decision_event_inbox (
                    tenant_id,source_system,event_id,delivery_sequence,source_event_sequence,
                    payload_schema_version,organization_id,subject_type,subject_id,job_id,
                    action_code,taxonomy_version,rubric_id,rubric_version,rubric_approval_mode,
                    jd_digest_version,recommendation_action,reason_code,before_state,after_state,
                    occurred_at,payload_digest
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s
                )
                RETURNING received_at
                """,
                (
                    tenant_id,
                    payload.source_system,
                    payload.event_id,
                    payload.delivery_sequence,
                    payload.source_event_sequence,
                    payload.payload_schema_version,
                    payload.organization_id,
                    payload.subject_type,
                    payload.subject_id,
                    payload.job_id,
                    payload.action_code,
                    payload.taxonomy_version,
                    payload.rubric_id,
                    payload.rubric_version,
                    payload.rubric_approval_mode,
                    payload.jd_digest_version,
                    payload.recommendation_action,
                    payload.reason_code,
                    json.dumps(payload.before_state.model_dump(mode="json"), separators=(",", ":")),
                    json.dumps(payload.after_state.model_dump(mode="json"), separators=(",", ":")),
                    payload.occurred_at,
                    digest,
                ),
            )
            received_at = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO organization_decision_stream_state (
                    tenant_id,state,last_delivery_sequence,last_source_event_sequence,
                    last_event_id,last_received_at,updated_at
                ) VALUES (%s,'current',%s,%s,%s,%s,clock_timestamp())
                ON CONFLICT (tenant_id) DO UPDATE SET
                    state='current',last_delivery_sequence=EXCLUDED.last_delivery_sequence,
                    last_source_event_sequence=EXCLUDED.last_source_event_sequence,
                    last_event_id=EXCLUDED.last_event_id,last_received_at=EXCLUDED.last_received_at,
                    updated_at=clock_timestamp()
                """,
                (
                    tenant_id,
                    payload.delivery_sequence,
                    payload.source_event_sequence,
                    payload.event_id,
                    received_at,
                ),
            )
        conn.commit()
        return "inserted"
    except HTTPException:
        conn.rollback()
        raise
    except UniqueViolation as exc:
        conn.rollback()
        raise HTTPException(status_code=409, detail="decision_inbox_event_conflict") from exc
    except Exception as exc:
        conn.rollback()
        logger.warning(
            "Decision inbox write failed", extra_fields={"error_type": type(exc).__name__}
        )
        raise HTTPException(status_code=503, detail="decision_inbox_unavailable") from exc
    finally:
        conn.close()


@router.post("/organization-decision-events/ingest", response_model=None)
async def ingest_organization_decision_event(
    request: Request,
    claims: Annotated[JWTClaims, Depends(require_decision_history_writer)],
) -> dict[str, str | int]:
    if not decision_inbox_enabled():
        raise HTTPException(status_code=503, detail="decision_inbox_disabled")
    payload = await _parse_body(request)
    expected_tenant = f"org_{payload.organization_id}"
    if claims.tenant_id != expected_tenant:
        raise HTTPException(status_code=403, detail="decision_inbox_tenant_denied")
    status = _store(payload, claims.tenant_id)
    return {
        "event_id": str(payload.event_id),
        "delivery_sequence": payload.delivery_sequence,
        "status": status,
    }
