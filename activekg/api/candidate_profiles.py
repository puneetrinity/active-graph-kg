from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.candidate_profiles.authority import require_authority
from activekg.candidate_profiles.config import enabled
from activekg.candidate_profiles.contracts import UUID, parse_command
from activekg.candidate_profiles.repository import execute

router = APIRouter()


def available() -> None:
    if not enabled():
        raise HTTPException(404, "not_found")


@router.post("/candidate-profiles/foundation/commands", dependencies=[Depends(available)])
async def ensure(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    subject = require_authority(claims, "candidate-profile:foundation-write")
    if (
        request.query_params
        or request.headers.get("content-type", "").split(";")[0].strip() != "application/json"
    ):
        raise HTTPException(400, "invalid_request")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 4096:
            raise HTTPException(400, "invalid_request")
    try:
        command, digest = parse_command(bytes(body))
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPException(400, "invalid_request") from None
    if subject != command["subjectId"]:
        raise HTTPException(403, "service_auth_denied")
    if request.headers.getlist("x-candidate-command-sha256") != [digest]:
        raise HTTPException(400, "invalid_request")
    result, replayed = await run_in_threadpool(
        execute, subject, command["bindingId"], command, digest
    )
    return JSONResponse(result, status_code=200 if replayed else 201)


@router.get("/candidate-profiles/foundation", dependencies=[Depends(available)])
async def status(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    subject = require_authority(claims, "candidate-profile:foundation-read")
    params = list(request.query_params.multi_items())
    if len(params) != 1 or params[0][0] != "binding" or not UUID.fullmatch(params[0][1]):
        raise HTTPException(400, "invalid_request")
    if (
        request.headers.get("transfer-encoding")
        or request.headers.get("content-length", "0") != "0"
    ):
        raise HTTPException(400, "invalid_request")
    result, _ = await run_in_threadpool(execute, subject, params[0][1])
    return result
