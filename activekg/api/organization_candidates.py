"""Organization-private application candidate intake.

The command creates one opaque tenant candidate per Flow application and stores
only PII-free reference and resume evidence. Applicant identifiers are used
transiently for privacy admission and are never persisted or logged here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from typing_extensions import Self

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.common.logger import get_enhanced_logger
from activekg.privacy.config import load_candidate_privacy_config
from activekg.privacy.identity import identity_token, normalize_privacy_identifier
from activekg.privacy.models import CandidatePrivacyDecision, decision_for
from activekg.privacy.repository import CandidatePrivacyRestricted, require_allowed

router = APIRouter(tags=["organization-candidates"])
organization_candidates_router = router
logger = get_enhanced_logger(__name__)

_MAX_BODY_BYTES = 64 * 1024
_TENANT_RE = re.compile(r"^org_[1-9][0-9]*$")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PrivacyIdentifier(_StrictModel):
    identifier_type: Literal["email", "phone", "vantahire_application_id", "vantahire_resume_id"]
    value: str = Field(min_length=1, max_length=2048)


class OrganizationCandidateIntakeRequest(_StrictModel):
    schema_version: Literal[1]
    reference_id: UUID
    application_id: int = Field(ge=1, le=2_147_483_647)
    job_id: int = Field(ge=1, le=2_147_483_647)
    resume_version_id: UUID
    resume_version: Literal[1]
    origin: Literal["candidate_applied"]
    source_kind: Literal["direct_upload", "saved_resume"]
    source_resume_id: int | None = Field(default=None, ge=1, le=2_147_483_647)
    source_observed_at: datetime
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(ge=1, le=5 * 1024 * 1024)
    media_type: Literal[
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    extracted_text_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    captured_at: datetime
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    privacy_subject: list[PrivacyIdentifier] = Field(min_length=1, max_length=4)

    @field_validator("source_observed_at", "captured_at")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp requires timezone")
        if value.astimezone(timezone.utc) > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError("timestamp is in the future")
        return value

    @model_validator(mode="after")
    def validate_source_and_privacy_subject(self) -> Self:
        if (self.source_kind == "saved_resume") != (self.source_resume_id is not None):
            raise ValueError("saved-resume source shape is invalid")
        values = [(item.identifier_type, item.value) for item in self.privacy_subject]
        if len(values) != len(set(values)):
            raise ValueError("privacy identifiers must be unique")
        expected_application = str(self.application_id)
        if values.count(("vantahire_application_id", expected_application)) != 1:
            raise ValueError("privacy subject is not bound to the application")
        resume_values = [
            value for identifier_type, value in values if identifier_type == "vantahire_resume_id"
        ]
        if self.source_resume_id is None and resume_values:
            raise ValueError("direct upload cannot assert a saved-resume privacy identifier")
        if self.source_resume_id is not None and resume_values != [str(self.source_resume_id)]:
            raise ValueError("privacy subject is not bound to the saved resume")
        return self


def organization_candidate_intake_enabled() -> bool:
    return os.getenv("ORGANIZATION_CANDIDATE_INTAKE_ENABLED", "false").strip().lower() == "true"


async def require_organization_candidate_writer(
    claims: JWTClaims | None = Depends(get_jwt_claims),
) -> JWTClaims:
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(status_code=401, detail="organization_candidate_service_auth_required")
    if (
        claims.issuer != auth.JWT_ISSUER
        or claims.actor_type != "service"
        or claims.actor_id != "vantahire-backend"
        or claims.scopes != ["organization-candidate:write"]
        or _TENANT_RE.fullmatch(claims.tenant_id) is None
    ):
        raise HTTPException(status_code=403, detail="organization_candidate_service_auth_denied")
    return claims


def compute_idempotency_key(payload: OrganizationCandidateIntakeRequest, tenant_id: str) -> str:
    components = (
        "v1",
        tenant_id,
        str(payload.reference_id),
        str(payload.application_id),
        str(payload.job_id),
        str(payload.resume_version_id),
        str(payload.resume_version),
        payload.origin,
    )
    return hashlib.sha256("\0".join(components).encode()).hexdigest()


def _canonical_persistent_input(payload: OrganizationCandidateIntakeRequest, tenant_id: str) -> str:
    material = payload.model_dump(mode="json", exclude={"idempotency_key", "privacy_subject"})
    material["tenant_id"] = tenant_id
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


async def _parse_body(request: Request) -> OrganizationCandidateIntakeRequest:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_BODY_BYTES:
                raise HTTPException(
                    status_code=413, detail="organization_candidate_request_too_large"
                )
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="organization_candidate_request_invalid"
            ) from exc
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="organization_candidate_request_too_large")
    try:
        return OrganizationCandidateIntakeRequest.model_validate_json(bytes(body))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=422, detail="organization_candidate_request_invalid"
        ) from exc


def _connect() -> psycopg.Connection:
    dsn = (os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        raise RuntimeError("organization candidate database unavailable")
    return psycopg.connect(dsn, autocommit=False)


def _privacy_tokens(identifiers: list[PrivacyIdentifier]) -> list[dict[str, Any]]:
    config = load_candidate_privacy_config(require_hmac=True)
    _active, keys = config.require_hmac()
    tokens: list[dict[str, Any]] = []
    for item in identifiers:
        normalized = normalize_privacy_identifier(item.identifier_type, item.value)
        for version, key in sorted(keys.items()):
            tokens.append(
                {
                    "identifier_type": normalized.identifier_type,
                    "key_version": version,
                    "token": identity_token(key, normalized).hex(),
                }
            )
    return tokens


def _require_private_privacy(cur: psycopg.Cursor, identifiers: list[PrivacyIdentifier]) -> None:
    try:
        tokens = _privacy_tokens(identifiers)
        cur.execute(
            """
            SELECT action,state,decision
            FROM candidate_privacy_match(%s::jsonb,NULL,NULL,NULL)
            ORDER BY CASE decision WHEN 'review' THEN 4 WHEN 'block_all' THEN 3
                     WHEN 'block_global' THEN 2 ELSE 1 END DESC,
                     effective_at DESC,version DESC
            LIMIT 1
            """,
            (json.dumps(tokens),),
        )
        row = cur.fetchone()
        decision = CandidatePrivacyDecision.ALLOW if row is None else decision_for(row[1], row[0])
        if row is not None and decision is not CandidatePrivacyDecision(row[2]):
            raise RuntimeError("candidate privacy matcher is inconsistent")
        require_allowed(decision, global_use=False)
    except CandidatePrivacyRestricted as exc:
        raise HTTPException(status_code=451, detail="candidate_privacy_restricted") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="candidate_privacy_unavailable") from exc


def _receipt_response(row: tuple[Any, ...], delivery_status: str) -> dict[str, Any]:
    return {
        "delivery_status": delivery_status,
        "resolution": "replayed" if delivery_status == "replayed" else str(row[4]),
        "idempotency_key": str(row[0]),
        "reference_id": str(row[1]),
        "resume_version_id": str(row[2]),
        "candidate_id": str(row[3]),
    }


def _select_receipt(cur: psycopg.Cursor, idempotency_key: str) -> tuple[Any, ...] | None:
    cur.execute(
        """
        SELECT idempotency_key,reference_id,resume_version_id,candidate_id,resolution,input_digest
        FROM organization_candidate_ingest_receipts WHERE idempotency_key=%s
        """,
        (idempotency_key,),
    )
    return cur.fetchone()


def _store(payload: OrganizationCandidateIntakeRequest, claims: JWTClaims) -> dict[str, Any]:
    expected_key = compute_idempotency_key(payload, claims.tenant_id)
    if payload.idempotency_key != expected_key:
        raise HTTPException(status_code=422, detail="organization_candidate_idempotency_invalid")
    input_digest = _canonical_persistent_input(payload, claims.tenant_id)
    conn = _connect()
    started = time.monotonic()

    def check_deadline() -> None:
        if time.monotonic() - started > 3:
            raise TimeoutError("organization candidate transaction deadline exceeded")

    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='500ms'")
            cur.execute("SET LOCAL statement_timeout='2000ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='3000ms'")
            cur.execute("SELECT set_config('app.current_tenant_id', %s, true)", (claims.tenant_id,))

            _require_private_privacy(cur, payload.privacy_subject)
            receipt = _select_receipt(cur, payload.idempotency_key)
            if receipt is not None:
                if str(receipt[5]) != input_digest:
                    raise HTTPException(
                        status_code=409, detail="organization_candidate_idempotency_conflict"
                    )
                conn.rollback()
                return _receipt_response(receipt, "replayed")

            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (f"organization-candidate:{claims.tenant_id}:{payload.reference_id}",),
            )
            check_deadline()
            receipt = _select_receipt(cur, payload.idempotency_key)
            if receipt is not None:
                if str(receipt[5]) != input_digest:
                    raise HTTPException(
                        status_code=409, detail="organization_candidate_idempotency_conflict"
                    )
                conn.rollback()
                return _receipt_response(receipt, "replayed")

            cur.execute(
                """
                INSERT INTO candidates (
                    tenant_id,scope,display_name,primary_email,primary_phone,props,metadata,
                    profile,skills,node_id,global_candidate_id
                ) VALUES (%s,'organization_private',NULL,NULL,NULL,'{}'::jsonb,'{}'::jsonb,
                          '{}'::jsonb,'{}'::text[],NULL,NULL)
                RETURNING candidate_id
                """,
                (claims.tenant_id,),
            )
            candidate_id = cur.fetchone()[0]
            cur.execute(
                """
                INSERT INTO organization_candidate_references (
                    reference_id,tenant_id,candidate_id,application_id,job_id,origin_code,
                    schema_version,verified_issuer,verified_actor_id
                ) VALUES (%s,%s,%s,%s,%s,'candidate_applied',1,%s,%s)
                """,
                (
                    payload.reference_id,
                    claims.tenant_id,
                    candidate_id,
                    payload.application_id,
                    payload.job_id,
                    claims.issuer,
                    claims.actor_id,
                ),
            )
            cur.execute(
                """
                INSERT INTO organization_candidate_resume_evidence (
                    resume_version_id,tenant_id,reference_id,candidate_id,version,source_kind,
                    source_resume_id,source_observed_at,content_sha256,byte_count,media_type,
                    extracted_text_sha256,captured_at
                ) VALUES (%s,%s,%s,%s,1,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    payload.resume_version_id,
                    claims.tenant_id,
                    payload.reference_id,
                    candidate_id,
                    payload.source_kind,
                    payload.source_resume_id,
                    payload.source_observed_at,
                    payload.content_sha256,
                    payload.byte_count,
                    payload.media_type,
                    payload.extracted_text_sha256,
                    payload.captured_at,
                ),
            )
            cur.execute(
                """
                INSERT INTO organization_candidate_ingest_receipts (
                    idempotency_key,input_digest,tenant_id,reference_id,resume_version_id,
                    candidate_id,resolution,verified_issuer,verified_actor_id
                ) VALUES (%s,%s,%s,%s,%s,%s,'created',%s,%s)
                RETURNING idempotency_key,reference_id,resume_version_id,candidate_id,
                          resolution,input_digest
                """,
                (
                    payload.idempotency_key,
                    input_digest,
                    claims.tenant_id,
                    payload.reference_id,
                    payload.resume_version_id,
                    candidate_id,
                    claims.issuer,
                    claims.actor_id,
                ),
            )
            receipt = cur.fetchone()
            check_deadline()
        conn.commit()
        return _receipt_response(receipt, "recorded")
    except HTTPException:
        conn.rollback()
        raise
    except UniqueViolation as exc:
        conn.rollback()
        raise HTTPException(
            status_code=409, detail="organization_candidate_reference_conflict"
        ) from exc
    except Exception as exc:
        conn.rollback()
        logger.warning(
            "Organization candidate intake failed",
            extra_fields={"error_type": type(exc).__name__},
        )
        raise HTTPException(
            status_code=503, detail="organization_candidate_intake_unavailable"
        ) from exc
    finally:
        conn.close()


@router.post("/organization-candidates/intake", response_model=None)
async def ingest_organization_candidate(
    request: Request,
    claims: Annotated[JWTClaims, Depends(require_organization_candidate_writer)],
) -> dict[str, Any]:
    if not organization_candidate_intake_enabled():
        raise HTTPException(status_code=503, detail="organization_candidate_intake_disabled")
    payload = await _parse_body(request)
    return _store(payload, claims)
