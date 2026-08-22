"""Service-only candidate privacy authority endpoints."""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal, get_args
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Self

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.privacy.config import (
    CandidatePrivacyConfig,
    CandidatePrivacyConfigurationError,
    load_candidate_privacy_config,
)
from activekg.privacy.identity import (
    PRIVACY_IDENTIFIER_TYPES,
    CandidatePrivacyIdentityError,
    NormalizedIdentifier,
    normalize_privacy_identifier,
)
from activekg.privacy.models import (
    CandidatePrivacyAction,
    CandidatePrivacyAuthorityType,
    CandidatePrivacyReason,
    CandidatePrivacyTransition,
    CanonicalSubject,
    DirectiveRecord,
)
from activekg.privacy.repository import (
    CandidatePrivacyConflict,
    CandidatePrivacyRepository,
    CandidatePrivacyUnavailable,
)

router = APIRouter(tags=["candidate-privacy"])
_repository: CandidatePrivacyRepository | None = None
_MAX_BODY_BYTES = 64 * 1024


def set_repository(repository: CandidatePrivacyRepository | None) -> None:
    global _repository
    _repository = repository


def _repo() -> CandidatePrivacyRepository:
    if _repository is None:
        raise HTTPException(status_code=503, detail="candidate_privacy_unavailable")
    return _repository


def _authority_config() -> CandidatePrivacyConfig:
    try:
        config = load_candidate_privacy_config(require_hmac=True)
    except CandidatePrivacyConfigurationError as exc:
        raise HTTPException(
            status_code=503, detail="candidate_privacy_configuration_invalid"
        ) from exc
    if config.flow_issuer != (auth.JWT_ISSUER or "") or config.signal_issuer != (
        auth.SIGNAL_JWT_ISSUER or ""
    ):
        raise HTTPException(status_code=503, detail="candidate_privacy_configuration_invalid")
    return config


def _require_service_claims(claims: JWTClaims | None, *, write: bool) -> JWTClaims:
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(status_code=401, detail="candidate_privacy_service_auth_required")
    config = _authority_config()
    permitted = {(config.flow_issuer, config.flow_actor_id)}
    if not write:
        permitted.add((config.signal_issuer, config.signal_actor_id))
    scope = "candidate-privacy:write" if write else "candidate-privacy:read"
    if (
        claims.actor_type != "service"
        or (claims.issuer, claims.actor_id) not in permitted
        or scope not in claims.scopes
    ):
        raise HTTPException(status_code=403, detail="candidate_privacy_service_auth_denied")
    return claims


async def require_candidate_privacy_write(
    claims: JWTClaims | None = Depends(get_jwt_claims),
) -> JWTClaims:
    return _require_service_claims(claims, write=True)


async def require_candidate_privacy_read(
    claims: JWTClaims | None = Depends(get_jwt_claims),
) -> JWTClaims:
    return _require_service_claims(claims, write=False)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IdentifierInput(_StrictModel):
    identifier_type: Literal[
        "email",
        "phone",
        "linkedin_url",
        "github_url",
        "signal_candidate_id",
        "vantahire_application_id",
        "vantahire_resume_id",
    ]
    value: str = Field(min_length=1, max_length=2048)


class CanonicalReference(_StrictModel):
    global_candidate_id: UUID | None = None
    candidate_tenant_id: str | None = Field(default=None, min_length=1, max_length=255)
    candidate_id: UUID | None = None

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if (self.candidate_tenant_id is None) != (self.candidate_id is None):
            raise ValueError("candidate reference must include tenant and candidate")
        return self


class DirectiveCreate(_StrictModel):
    request_id: UUID
    action: CandidatePrivacyAction
    authority_type: CandidatePrivacyAuthorityType
    evidence_ref: UUID
    reason_code: CandidatePrivacyReason
    identifiers: list[IdentifierInput] = Field(default_factory=list, max_length=8)
    canonical: CanonicalReference | None = None

    @model_validator(mode="after")
    def validate_subject_and_reason(self) -> Self:
        if not self.identifiers and self.canonical is None:
            raise ValueError("subject is required")
        allowed = {
            CandidatePrivacyAction.WITHDRAW_GLOBAL_MATCHING: {
                CandidatePrivacyReason.CANDIDATE_GLOBAL_OPT_OUT,
                CandidatePrivacyReason.VERIFIED_SUPPORT_REQUEST,
            },
            CandidatePrivacyAction.REQUEST_ERASURE: {
                CandidatePrivacyReason.CANDIDATE_ERASURE_REQUEST,
                CandidatePrivacyReason.VERIFIED_SUPPORT_REQUEST,
            },
        }
        if self.reason_code not in allowed[self.action]:
            raise ValueError("action and reason do not match")
        return self


class DirectiveTransition(_StrictModel):
    request_id: UUID
    expected_version: int = Field(ge=1)
    transition: CandidatePrivacyTransition
    evidence_ref: UUID
    reason_code: CandidatePrivacyReason

    @model_validator(mode="after")
    def validate_reason(self) -> Self:
        if self.transition is CandidatePrivacyTransition.MARK_NEEDS_REVIEW:
            allowed = {
                CandidatePrivacyReason.IDENTITY_AMBIGUITY,
                CandidatePrivacyReason.OPERATOR_CORRECTION,
            }
        else:
            allowed = {
                CandidatePrivacyReason.OPERATOR_CORRECTION,
                CandidatePrivacyReason.VERIFIED_SUPPORT_REQUEST,
            }
        if self.reason_code not in allowed:
            raise ValueError("transition and reason do not match")
        return self


class EligibilitySubject(_StrictModel):
    request_ref: UUID
    identifiers: list[IdentifierInput] = Field(default_factory=list, max_length=8)
    canonical: CanonicalReference | None = None

    @model_validator(mode="after")
    def validate_subject(self) -> Self:
        if not self.identifiers and self.canonical is None:
            raise ValueError("subject is required")
        return self


class EligibilityBatch(_StrictModel):
    subjects: list[EligibilitySubject] = Field(min_length=1, max_length=200)


async def _parse_body(request: Request, model: type[_StrictModel]) -> Any:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="candidate_privacy_request_too_large")
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="candidate_privacy_request_invalid"
            ) from exc
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="candidate_privacy_request_too_large")
    try:
        return model.model_validate_json(bytes(body))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="candidate_privacy_request_invalid") from exc


def _identifiers(values: list[IdentifierInput]) -> list[NormalizedIdentifier]:
    try:
        normalized = [
            normalize_privacy_identifier(item.identifier_type, item.value) for item in values
        ]
    except CandidatePrivacyIdentityError as exc:
        raise HTTPException(status_code=422, detail="candidate_privacy_identifier_invalid") from exc
    unique = {(item.identifier_type, item.normalized): item for item in normalized}
    return list(unique.values())


def _canonical(reference: CanonicalReference | None) -> CanonicalSubject:
    if reference is None:
        return CanonicalSubject()
    return CanonicalSubject(
        global_candidate_id=reference.global_candidate_id,
        candidate_tenant_id=reference.candidate_tenant_id,
        candidate_id=reference.candidate_id,
    )


def _response(request_id: UUID, record: DirectiveRecord) -> dict[str, Any]:
    return {
        "request_id": str(request_id),
        "directive_id": str(record.directive_id),
        "action": record.action.value,
        "scope": record.scope.value,
        "state": record.state.value,
        "version": record.version,
        "effective_at": record.effective_at.isoformat(),
        "decision": record.decision.value,
    }


@router.post("/candidate-privacy/directives", response_model=None)
async def create_directive(
    request: Request,
    claims: Annotated[JWTClaims, Depends(require_candidate_privacy_write)],
) -> dict[str, Any]:
    config = _authority_config()
    if not config.intake_enabled:
        raise HTTPException(status_code=503, detail="candidate_privacy_intake_disabled")
    payload = await _parse_body(request, DirectiveCreate)
    identifiers = _identifiers(payload.identifiers)
    repository = _repo()
    try:
        request_id, record = repository.create_directive(
            request_id=payload.request_id,
            action=payload.action,
            authority_type=payload.authority_type,
            evidence_ref=payload.evidence_ref,
            reason=payload.reason_code,
            issuer=claims.issuer or "",
            actor_id=claims.actor_id,
            identifiers=identifiers,
            canonical=_canonical(payload.canonical),
        )
    except CandidatePrivacyConflict as exc:
        raise HTTPException(status_code=409, detail="candidate_privacy_request_conflict") from exc
    except CandidatePrivacyUnavailable as exc:
        raise HTTPException(status_code=503, detail="candidate_privacy_unavailable") from exc
    return _response(request_id, record)


@router.post("/candidate-privacy/directives/{directive_id}/transitions", response_model=None)
async def transition_directive(
    directive_id: UUID,
    request: Request,
    claims: Annotated[JWTClaims, Depends(require_candidate_privacy_write)],
) -> dict[str, Any]:
    config = _authority_config()
    if not config.intake_enabled:
        raise HTTPException(status_code=503, detail="candidate_privacy_intake_disabled")
    payload = await _parse_body(request, DirectiveTransition)
    try:
        request_id, record = _repo().transition_directive(
            directive_id=directive_id,
            expected_version=payload.expected_version,
            request_id=payload.request_id,
            transition=payload.transition,
            evidence_ref=payload.evidence_ref,
            reason=payload.reason_code,
            issuer=claims.issuer or "",
            actor_id=claims.actor_id,
        )
    except CandidatePrivacyConflict as exc:
        raise HTTPException(
            status_code=409, detail="candidate_privacy_transition_conflict"
        ) from exc
    except CandidatePrivacyUnavailable as exc:
        raise HTTPException(status_code=503, detail="candidate_privacy_unavailable") from exc
    return _response(request_id, record)


@router.post("/candidate-privacy/eligibility/batch", response_model=None)
async def eligibility_batch(
    request: Request,
    _claims: Annotated[JWTClaims, Depends(require_candidate_privacy_read)],
) -> dict[str, Any]:
    payload = await _parse_body(request, EligibilityBatch)
    results: list[dict[str, str]] = []
    repository = _repo()
    for subject in payload.subjects:
        identifiers = _identifiers(subject.identifiers)
        canonical = _canonical(subject.canonical)
        try:
            decision = repository.evaluate(
                identifiers=identifiers,
                global_candidate_id=canonical.global_candidate_id,
                candidate_tenant_id=canonical.candidate_tenant_id,
                candidate_id=canonical.candidate_id,
            )
        except CandidatePrivacyUnavailable as exc:
            raise HTTPException(status_code=503, detail="candidate_privacy_unavailable") from exc
        results.append({"request_ref": str(subject.request_ref), "decision": decision.value})
    return {"results": results, "count": len(results)}


@router.get("/candidate-privacy/changes", response_model=None)
def changes(
    _claims: Annotated[JWTClaims, Depends(require_candidate_privacy_read)],
    after_cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        events = _repo().changes(after_cursor=after_cursor, limit=limit)
    except CandidatePrivacyUnavailable as exc:
        raise HTTPException(status_code=503, detail="candidate_privacy_unavailable") from exc
    return {"events": events, "count": len(events)}


@router.get("/candidate-privacy/snapshot", response_model=None)
def snapshot(
    _claims: Annotated[JWTClaims, Depends(require_candidate_privacy_read)],
    after_directive_id: UUID | None = None,
    high_water_cursor: int | None = Query(default=None, ge=0),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        high_water, directives = _repo().snapshot(
            after_directive_id=after_directive_id,
            high_water_cursor=high_water_cursor,
            limit=limit,
        )
    except CandidatePrivacyConflict as exc:
        raise HTTPException(status_code=409, detail="candidate_privacy_snapshot_conflict") from exc
    except CandidatePrivacyUnavailable as exc:
        raise HTTPException(status_code=503, detail="candidate_privacy_unavailable") from exc
    return {"high_water_cursor": high_water, "directives": directives, "count": len(directives)}


assert set(PRIVACY_IDENTIFIER_TYPES) == set(
    get_args(IdentifierInput.model_fields["identifier_type"].annotation)
)
