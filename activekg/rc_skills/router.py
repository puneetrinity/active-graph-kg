"""Signal-only skill cache; feature-off requests cannot touch the database."""

from __future__ import annotations

import asyncio
import os
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.rc_skills.config import rc_skills_enabled
from activekg.rc_skills.contracts import MAX_BODY_BYTES, FinishRequest, ReadRequest

router = APIRouter(tags=["rc-skills"])


def _authorize(claims: JWTClaims | None, scope: str) -> JWTClaims:
    try:
        enabled = rc_skills_enabled()
    except ValueError:
        raise HTTPException(503, detail="rc_configuration_invalid") from None
    if not enabled:
        raise HTTPException(503, detail="rc_disabled")
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(401, detail="rc_service_auth_required")
    if (
        claims.issuer != auth.SIGNAL_JWT_ISSUER
        or claims.actor_type != "service"
        or claims.actor_id != "signal-service"
        or claims.scopes != [scope]
        or re.fullmatch(r"org_[1-9][0-9]*", claims.tenant_id or "") is None
    ):
        raise HTTPException(403, detail="rc_service_auth_denied")
    return claims


async def require_reader(claims: Annotated[JWTClaims | None, Depends(get_jwt_claims)]) -> JWTClaims:
    return _authorize(claims, "rc-skills:read")


async def require_writer(claims: Annotated[JWTClaims | None, Depends(get_jwt_claims)]) -> JWTClaims:
    return _authorize(claims, "rc-skills:write")


async def _execute(request: Request, claims: JWTClaims, *, write: bool):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY_BYTES:
            raise HTTPException(413, detail="rc_request_too_large")
        body.extend(chunk)
    try:
        command = (FinishRequest if write else ReadRequest).model_validate_json(bytes(body))
    except ValueError:
        raise HTTPException(422, detail="rc_request_invalid") from None
    # Import and configuration access stay below the feature/auth/body boundary.
    from activekg.rc_skills.repository import RcRepository, RcUnavailable

    try:
        repo = RcRepository(os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL") or "")
        method = repo.finish if write else repo.read
        result = await asyncio.to_thread(method, claims.actor_id, command)
        return JSONResponse(
            result.model_dump(by_alias=True), headers={"Cache-Control": "private, no-store"}
        )
    except RcUnavailable as exc:
        status = {"unavailable": 404, "restricted": 403, "conflict": 409, "denied": 403}.get(
            exc.code, 503
        )
        raise HTTPException(status, detail=f"rc_{exc.code}") from None


@router.post("/internal/rc-skills/read")
async def read_skills(request: Request, claims: Annotated[JWTClaims, Depends(require_reader)]):
    return await _execute(request, claims, write=False)


@router.post("/internal/rc-skills/finish")
async def finish_skills(request: Request, claims: Annotated[JWTClaims, Depends(require_writer)]):
    return await _execute(request, claims, write=True)
