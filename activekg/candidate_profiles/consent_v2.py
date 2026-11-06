"""Authenticated durable-consent projection; never publishes a candidate."""

from typing import Annotated

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.candidate_profiles.authority import require_authority
from activekg.candidate_profiles.config import consent_v2_enabled
from activekg.candidate_profiles.repository import _connect
from activekg.candidate_profiles.spark_contracts import old, parse_consent_projection


def available():
    if not consent_v2_enabled():
        raise HTTPException(404, "not_found")


router = APIRouter(dependencies=[Depends(available)])


def project(parsed):
    c = parsed["value"]
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='3s'")
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='4s'")
            cur.execute(
                "SELECT public.candidate_consent_v2_project(%s,%s,%s,%s,%s,%s,%s,%s,%s) AS result",
                (
                    c["subjectId"],
                    c["bindingId"],
                    c["eventId"],
                    c["version"],
                    c["action"],
                    c["copyVersion"],
                    c["copyHash"],
                    c["purposes"],
                    parsed["hash"],
                ),
            )
            row = cur.fetchone()
            old.need(row is not None)
            result = row["result"]
            old.obj(result, "eventId version outcome effectiveVersion")
            old.need(result["eventId"] == c["eventId"] and result["version"] == c["version"])
            old.revision(result["effectiveVersion"])
            old.need(
                (result["outcome"] == "superseded" and result["effectiveVersion"] >= c["version"])
                or (
                    result["effectiveVersion"] == c["version"]
                    and result["outcome"]
                    == ("onboarding_granted" if c["action"] == "grant" else "withdrawn")
                )
            )
        # A successful response is returned only after COMMIT. Identical retries
        # return the stored SQL receipt, including after a newer withdrawal.
        return result
    except psycopg.Error as error:
        status = {"P0002": 404, "23505": 409, "22023": 400, "42501": 451}.get(error.sqlstate, 503)
        raise HTTPException(
            status,
            "consent_projection_refused" if status != 503 else "consent_projection_unavailable",
        ) from None
    except Exception:
        raise HTTPException(503, "consent_projection_unavailable") from None


@router.post("/candidate-consent/v2/events")
async def receive(request: Request, claims: Annotated[JWTClaims | None, Depends(get_jwt_claims)]):
    subject = require_authority(claims, "candidate-consent:write")
    if (
        request.query_params
        or request.headers.get("content-type", "").split(";")[0].strip().lower()
        != "application/json"
    ):
        raise HTTPException(400, "invalid_consent")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 4096:
            raise HTTPException(413, "consent_too_large")
    try:
        parsed = parse_consent_projection(bytes(body))
    except (ValueError, TypeError, KeyError, UnicodeError, RecursionError):
        raise HTTPException(400, "invalid_consent") from None
    if request.headers.getlist("x-command-sha256") != [parsed["hash"]]:
        raise HTTPException(400, "consent_hash_mismatch")
    if subject != parsed["value"]["subjectId"]:
        raise HTTPException(403, "consent_subject_denied")
    result = await run_in_threadpool(project, parsed)
    return JSONResponse(result, headers={"Cache-Control": "no-store"})
