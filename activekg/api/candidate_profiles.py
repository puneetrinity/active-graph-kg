import hashlib
import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.candidate_profiles.authority import require_authority
from activekg.candidate_profiles.config import (
    enabled,
    interpretation_enabled,
    profile_enabled,
    source_import_enabled,
    spark_enabled,
)
from activekg.candidate_profiles.consent_v2 import router as consent_v2_router
from activekg.candidate_profiles.contracts import UUID, parse_command
from activekg.candidate_profiles.profile_authority import require_owner
from activekg.candidate_profiles.profile_contracts import canonical, parse_profile_command
from activekg.candidate_profiles.profile_repository import erase_owned_profile, execute_profile
from activekg.candidate_profiles.repository import execute
from activekg.candidate_profiles.source_contracts import parse_resume_transfer
from activekg.candidate_profiles.source_repository import (
    interpretation_storage,
    preview_interpretation_exclusions,
    read_source,
    receive_source,
)
from activekg.candidate_profiles.spark_contracts import parse_command as parse_spark_command
from activekg.candidate_profiles.spark_contracts import parse_expiry, parse_interpretation_transfer
from activekg.candidate_profiles.spark_repository import execute_spark

router = APIRouter()
router.include_router(consent_v2_router)


def spark_available() -> None:
    if not spark_enabled():
        raise HTTPException(404, "not_found")


def source_available() -> None:
    if not source_import_enabled():
        raise HTTPException(404, "not_found")


def interpretation_available() -> None:
    if not interpretation_enabled():
        raise HTTPException(404, "not_found")


@router.post(
    "/candidate-profiles/exclusions/preview", dependencies=[Depends(interpretation_available)]
)
async def preview_exclusions(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    subject = require_owner(claims, write=True)
    binding = request.headers.getlist("x-candidate-binding-id")
    if (
        request.query_params
        or request.headers.getlist("content-type") != ["application/json"]
        or len(binding) != 1
        or not UUID.fullmatch(binding[0])
    ):
        raise HTTPException(400, "invalid_request")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 32768:
            raise HTTPException(413, "payload_too_large")
        body.extend(chunk)
    if request.headers.getlist("x-command-sha256") != [hashlib.sha256(body).hexdigest()]:
        raise HTTPException(422, "invalid_request")
    result = await run_in_threadpool(
        preview_interpretation_exclusions, subject, binding[0], bytes(body)
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.post(
    "/candidate-profiles/interpretations", dependencies=[Depends(interpretation_available)]
)
async def receive_interpretation(
    request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    subject = require_owner(claims, write=True)
    if request.query_params or request.headers.getlist("content-type") != ["application/json"]:
        raise HTTPException(400, "invalid_request")
    headers = {}
    for key in ("x-candidate-binding-id", "x-candidate-command-id", "x-command-sha256"):
        values = request.headers.getlist(key)
        if len(values) != 1:
            raise HTTPException(400, "invalid_request")
        headers[key] = values[0]
    if not UUID.fullmatch(headers["x-candidate-binding-id"]) or not UUID.fullmatch(
        headers["x-candidate-command-id"]
    ):
        raise HTTPException(400, "invalid_request")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 131072:
            raise HTTPException(413, "payload_too_large")
        body.extend(chunk)
    try:
        parsed = parse_interpretation_transfer(bytes(body))
        if parsed["hash"] != headers["x-command-sha256"]:
            raise ValueError("invalid_request")
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise HTTPException(422, "invalid_request") from None
    result = await run_in_threadpool(
        interpretation_storage,
        subject,
        headers["x-candidate-binding-id"],
        raw=bytes(body),
        command_id=headers["x-candidate-command-id"],
        command_hash=headers["x-command-sha256"],
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.get(
    "/candidate-profiles/interpretations/{proposal_id}",
    dependencies=[Depends(interpretation_available)],
)
async def read_interpretation(
    proposal_id: str, request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    subject = require_owner(claims)
    params = list(request.query_params.multi_items())
    if (
        not UUID.fullmatch(proposal_id)
        or len(params) != 1
        or params[0][0] != "binding"
        or not UUID.fullmatch(params[0][1])
        or request.headers.get("transfer-encoding")
        or request.headers.get("content-length", "0") != "0"
    ):
        raise HTTPException(400, "invalid_request")
    result = await run_in_threadpool(
        interpretation_storage, subject, params[0][1], proposal_id=proposal_id
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


# Retained implementation, deliberately NOT registered as an app route.
# Permanent deletion requires a separately approved owner-credential procedure.
async def erase_profile(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    # Neither ordinary owner-write nor a browser token authorizes deletion.
    # Do not mount this handler to implement operator approval.
    subject = require_authority(claims, "candidate-profile:erase")
    if request.query_params or request.headers.getlist("content-type") != ["application/json"]:
        raise HTTPException(400, "invalid_request")
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 1024:
            raise HTTPException(413, "payload_too_large")
        body.extend(chunk)
    try:
        value = json.loads(bytes(body).decode("utf-8"))
        if (
            not isinstance(value, dict)
            or set(value) != {"schemaVersion", "subjectId", "bindingId", "requestId"}
            or type(value["schemaVersion"]) is not int
            or value["schemaVersion"] != 1
            or any(
                not isinstance(value[k], str) or not UUID.fullmatch(value[k])
                for k in ("subjectId", "bindingId", "requestId")
            )
            or canonical(value).encode() != bytes(body)
        ):
            raise ValueError("invalid_request")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise HTTPException(422, "invalid_request") from None
    if value["subjectId"] != subject:
        raise HTTPException(403, "service_auth_denied")
    receipt = await run_in_threadpool(
        erase_owned_profile, subject, value["bindingId"], value["requestId"]
    )
    return JSONResponse(receipt, headers={"Cache-Control": "no-store"})


@router.get(
    "/candidate-profiles/sources/{source_id}/proposals/{proposal_id}",
    dependencies=[Depends(source_available)],
)
async def read_source_proposal(
    source_id: str,
    proposal_id: str,
    request: Request,
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    subject = require_owner(claims)
    params = list(request.query_params.multi_items())
    if (
        not UUID.fullmatch(source_id)
        or not UUID.fullmatch(proposal_id)
        or len(params) != 1
        or params[0][0] != "binding"
        or not UUID.fullmatch(params[0][1])
        or request.headers.get("transfer-encoding")
        or request.headers.get("content-length", "0") != "0"
    ):
        raise HTTPException(400, "invalid_request")
    result = await run_in_threadpool(read_source, subject, params[0][1], source_id, proposal_id)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.post("/candidate-profiles/sources/resume", dependencies=[Depends(source_available)])
async def receive_resume_source(
    request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    return await _receive_source(request, claims, "resume")


@router.post("/candidate-profiles/sources/lookup", dependencies=[Depends(source_available)])
async def receive_lookup_source(
    request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)
):
    return await _receive_source(request, claims, "professional_lookup")


async def _receive_source(request, claims, kind):
    subject = require_owner(claims, write=True)
    if (
        request.query_params
        or len(request.headers.getlist("content-type")) != 1
        or request.headers.get("content-type", "").split(";")[0].strip() != "application/json"
    ):
        raise HTTPException(400, "invalid_request")
    headers = {}
    for key in ("x-candidate-binding-id", "x-candidate-command-id"):
        values = request.headers.getlist(key)
        if len(values) != 1 or not UUID.fullmatch(values[0]):
            raise HTTPException(400, "invalid_request")
        headers[key] = values[0]
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 2 * 1024 * 1024:
            raise HTTPException(413, "payload_too_large")
        body.extend(chunk)
    try:
        parsed = parse_resume_transfer(bytes(body), kind)
        if parsed["canonical"].encode() != bytes(body) or request.headers.getlist(
            "x-command-sha256"
        ) != [parsed["hash"]]:
            raise ValueError("invalid_request")
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise HTTPException(422, "invalid_request") from None
    value = parsed["value"]
    if (subject, headers["x-candidate-binding-id"], headers["x-candidate-command-id"]) != (
        value["subjectId"],
        value["bindingId"],
        value["commandId"],
    ):
        raise HTTPException(403, "service_auth_denied")
    receipt = await run_in_threadpool(
        receive_source,
        subject,
        value["bindingId"],
        value["commandId"],
        parsed["hash"],
        bytes(body),
        kind,
    )
    return JSONResponse(receipt, headers={"Cache-Control": "no-store"})


@router.post("/candidate-profiles/spark/expiry", dependencies=[Depends(spark_available)])
async def expire_spark(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    subject = require_owner(claims, write=True)
    if (
        request.query_params
        or request.headers.get("content-type", "").split(";")[0].strip() != "application/json"
    ):
        raise HTTPException(400, "invalid_request")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 4096:
            raise HTTPException(413, "payload_too_large")
    try:
        parsed = parse_expiry(bytes(body))
        if parsed["canonical"].encode() != bytes(body) or request.headers.getlist(
            "x-candidate-expiry-sha256"
        ) != [parsed["hash"]]:
            raise ValueError("invalid_request")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise HTTPException(422, "invalid_request") from None
    value = parsed["value"]
    result = await run_in_threadpool(
        execute_spark,
        subject,
        value["bindingId"],
        raw=bytes(body),
        command_id=value["commandId"],
        digest=parsed["hash"],
        expiry=True,
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.post("/candidate-profiles/spark/commands", dependencies=[Depends(spark_available)])
async def save_spark(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    subject = require_owner(claims, write=True)
    if (
        request.query_params
        or request.headers.get("content-type", "").split(";")[0].strip() != "application/json"
    ):
        raise HTTPException(400, "invalid_request")
    values = {}
    for key in ("x-candidate-binding-id", "x-candidate-command-id"):
        parts = request.headers.getlist(key)
        if len(parts) != 1 or not UUID.fullmatch(parts[0]):
            raise HTTPException(400, "invalid_request")
        values[key] = parts[0]
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 384 * 1024:
            raise HTTPException(413, "payload_too_large")
    try:
        parsed = parse_spark_command(bytes(body))
        if parsed["value"]["action"] in ("accept_source", "reject_source"):
            source_available()
        if parsed["value"]["action"] == "accept_interpretation":
            interpretation_available()
        if parsed["canonical"].encode() != bytes(body) or request.headers.getlist(
            "x-candidate-command-sha256"
        ) != [parsed["hash"]]:
            raise ValueError("invalid_request")
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise HTTPException(422, "invalid_request") from None
    result = await run_in_threadpool(
        execute_spark,
        subject,
        values["x-candidate-binding-id"],
        raw=bytes(body),
        command_id=values["x-candidate-command-id"],
        digest=parsed["hash"],
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.get("/candidate-profiles/spark", dependencies=[Depends(spark_available)])
async def read_spark(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    subject = require_owner(claims)
    params = list(request.query_params.multi_items())
    if len(params) != 1 or params[0][0] != "binding" or not UUID.fullmatch(params[0][1]):
        raise HTTPException(400, "invalid_request")
    if (
        request.headers.get("transfer-encoding")
        or request.headers.get("content-length", "0") != "0"
    ):
        raise HTTPException(400, "invalid_request")
    result = await run_in_threadpool(execute_spark, subject, params[0][1])
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


def profile_available() -> None:
    if not profile_enabled():
        raise HTTPException(404, "not_found")


@router.post("/candidate-profiles/profile/commands", dependencies=[Depends(profile_available)])
async def save_profile(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    subject = require_owner(claims, write=True)
    if (
        request.query_params
        or request.headers.get("content-type", "").split(";")[0].strip() != "application/json"
    ):
        raise HTTPException(400, "invalid_request")
    values = {}
    for key in ("x-candidate-binding-id", "x-candidate-command-id"):
        parts = request.headers.getlist(key)
        if len(parts) != 1 or not UUID.fullmatch(parts[0]):
            raise HTTPException(400, "invalid_request")
        values[key] = parts[0]
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 384 * 1024:
            raise HTTPException(413, "payload_too_large")
    try:
        _, wire, digest = parse_profile_command(bytes(body))
        if wire.encode() != bytes(body) or request.headers.getlist(
            "x-candidate-command-sha256"
        ) != [digest]:
            raise ValueError("invalid_request")
    except (ValueError, UnicodeError, RecursionError):
        raise HTTPException(422, "invalid_request") from None
    result = await run_in_threadpool(
        execute_profile,
        subject,
        values["x-candidate-binding-id"],
        raw=bytes(body),
        command_id=values["x-candidate-command-id"],
        digest=digest,
    )
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.get("/candidate-profiles/profile", dependencies=[Depends(profile_available)])
async def read_profile(request: Request, claims: JWTClaims | None = Depends(get_jwt_claims)):
    subject = require_owner(claims)
    params = list(request.query_params.multi_items())
    if len(params) != 1 or params[0][0] != "binding" or not UUID.fullmatch(params[0][1]):
        raise HTTPException(400, "invalid_request")
    if (
        request.headers.get("transfer-encoding")
        or request.headers.get("content-length", "0") != "0"
    ):
        raise HTTPException(400, "invalid_request")
    result = await run_in_threadpool(execute_profile, subject, params[0][1])
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


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
