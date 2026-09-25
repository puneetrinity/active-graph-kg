"""Service-only, organization-private read; no ingestion or command side effects."""

from __future__ import annotations

import asyncio
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.candidate_history.contracts import MAX_BODY_BYTES, HistoryConfig, HistoryReadRequest
from activekg.candidate_history.repository import HistoryRepository, HistoryUnavailable

router = APIRouter(tags=["organization-candidate-history"])


async def require_history_reader(claims: JWTClaims | None = Depends(get_jwt_claims)) -> JWTClaims:
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(401, detail="history_service_auth_required")
    actor = os.getenv("ORG_DECISION_INBOX_FLOW_ACTOR_ID", "vantahire-backend").strip()
    if not actor or len(actor) > 160:
        raise HTTPException(503, detail="history_configuration_invalid")
    if (
        claims.issuer != auth.JWT_ISSUER
        or claims.actor_type != "service"
        or claims.actor_id != actor
        or "decision-history:read" not in claims.scopes
    ):
        raise HTTPException(403, detail="history_service_auth_denied")
    return claims


@router.post("/organization-candidate-history/read")
async def read_history(request: Request, claims: JWTClaims = Depends(require_history_reader)):
    try:
        config = HistoryConfig.from_env(dict(os.environ))
    except ValueError:
        raise HTTPException(503, detail="history_configuration_invalid") from None
    if not config.enabled:
        raise HTTPException(503, detail="history_disabled")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(413, detail="history_request_too_large")
    try:
        command = HistoryReadRequest.model_validate_json(bytes(body))
    except (ValueError, ValidationError):
        raise HTTPException(422, detail="history_request_invalid") from None
    if claims.tenant_id != f"org_{command.organization_id}":
        raise HTTPException(403, detail="history_tenant_denied")
    try:
        repo = HistoryRepository(os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL") or "")
        result = await asyncio.to_thread(repo.read, command)
        from fastapi.responses import JSONResponse

        return JSONResponse(
            result.model_dump(mode="json"), headers={"Cache-Control": "private, no-store"}
        )
    except HistoryUnavailable as exc:
        code = {"not_found": 404, "binding_conflict": 409, "privacy_restricted": 451}.get(
            exc.code, 503
        )
        raise HTTPException(code, detail=exc.code) from None
