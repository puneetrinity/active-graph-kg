"""Strict service authority for the private resume sourcing reader.

General kg:read credentials are deliberately not
sufficient, and request bodies never supply organisation authority.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass

import jwt
from fastapi import APIRouter, HTTPException, Request

from activekg.api import auth
from activekg.candidate_index import sourcing
from activekg.candidate_index.admission import configuration, enabled
from activekg.candidate_index.sourcing_contracts import (
    REVALIDATE_BODY_BYTES,
    SEARCH_BODY_BYTES,
    SourcingContractError,
    parse_public_applicants,
    parse_sourcing_revalidate,
    parse_sourcing_search,
)

SOURCING_SCOPE = "organization-candidate-sourcing:read"
MAX_TOKEN_BYTES = 16 * 1024
router = APIRouter(tags=["organization-candidate-sourcing"])


class SourcingAuthError(ValueError):
    """Fixed safe message; never expose token or decoded claims."""


@dataclass(frozen=True)
class SourcingAuthority:
    tenant_id: str
    request_id: str


def verify_sourcing_authority(token: str, request_id: str, *, public_key: str) -> SourcingAuthority:
    """Verify the dedicated Signal credential, not a general API credential.

    The eventual route must pass the configured Signal public key, never a key from
    the caller. No clock leeway: an expired credential grants no private read.
    """
    try:
        if (
            not isinstance(token, str)
            or not 1 <= len(token.encode("utf-8")) <= MAX_TOKEN_BYTES
            or not public_key
            or not isinstance(request_id, str)
            or not 1 <= len(request_id) <= 200
            or not request_id.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in request_id)
        ):
            raise ValueError("invalid")
        claims = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            issuer="signal",
            audience="activekg",
            options={
                "strict_aud": True,
                "require": [
                    "iss",
                    "aud",
                    "sub",
                    "actor_type",
                    "tenant_id",
                    "scopes",
                    "request_id",
                    "iat",
                    "exp",
                    "jti",
                ],
            },
        )
        tenant = claims["tenant_id"]
        issued, expires = claims["iat"], claims["exp"]
        nonce = claims["jti"]
        if (
            claims["sub"] != "signal-service"
            or claims["actor_type"] != "service"
            or claims["scopes"] != [SOURCING_SCOPE]
            or claims["request_id"] != request_id
            or not isinstance(tenant, str)
            or not re.fullmatch(r"org_[1-9][0-9]{0,9}", tenant)
            or int(tenant[4:]) > 2_147_483_647
            or type(issued) is not int
            or type(expires) is not int
            or not 0 < expires - issued <= 300
            or not isinstance(nonce, str)
            or not 1 <= len(nonce) <= 200
            or not nonce.strip()
            or any(ord(c) < 32 or ord(c) == 127 for c in nonce)
        ):
            raise ValueError("invalid")
        return SourcingAuthority(tenant_id=tenant, request_id=request_id)
    except (jwt.PyJWTError, ValueError, TypeError, KeyError, OverflowError):
        raise SourcingAuthError("candidate_sourcing_unauthorized") from None


async def _handle(request: Request, *, revalidation: bool, public_membership: bool = False) -> dict:
    deadline = time.monotonic() + sourcing.TOTAL_SECONDS
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not auth.JWT_ENABLED or scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "candidate_sourcing_unauthorized")
    if len(token) > MAX_TOKEN_BYTES:
        raise HTTPException(401, "candidate_sourcing_unauthorized")
    if not auth.SIGNAL_JWT_PUBLIC_KEY:
        raise HTTPException(503, "candidate_sourcing_unavailable")
    # Validate signature before reading any body. The actual request ID is then
    # checked against the signed claim after the bounded body has been parsed.
    try:
        raw_claims = jwt.decode(token, options={"verify_signature": False})
        authority = verify_sourcing_authority(
            token, raw_claims.get("request_id"), public_key=auth.SIGNAL_JWT_PUBLIC_KEY
        )
    except (SourcingAuthError, jwt.PyJWTError, AttributeError):
        raise HTTPException(401, "candidate_sourcing_unauthorized") from None
    cap = REVALIDATE_BODY_BYTES if revalidation else SEARCH_BODY_BYTES

    async def run():
        if (
            request.headers.get("content-type", "").split(";")[0].strip().lower()
            != "application/json"
        ):
            raise HTTPException(415, "candidate_sourcing_json_required")
        length = request.headers.get("content-length")
        if length is not None and (not length.isdecimal() or len(length) > 10 or int(length) > cap):
            raise HTTPException(413, "candidate_sourcing_request_too_large")
        raw = bytearray()
        async for chunk in request.stream():
            if len(raw) + len(chunk) > cap:
                raise HTTPException(413, "candidate_sourcing_request_too_large")
            raw.extend(chunk)
        try:
            parser = (
                parse_public_applicants
                if public_membership
                else (parse_sourcing_revalidate if revalidation else parse_sourcing_search)
            )
            command = parser(bytes(raw))
        except SourcingContractError:
            raise HTTPException(422, "candidate_sourcing_body_invalid") from None
        if authority.request_id != command.requestId:
            raise HTTPException(403, "candidate_sourcing_request_mismatch")
        if (not public_membership and not enabled("api")) or not os.environ.get("ACTIVEKG_DSN"):
            raise HTTPException(503, "candidate_sourcing_unavailable")
        if public_membership:
            return await sourcing.public_applicants(
                command, authority.tenant_id, os.environ["ACTIVEKG_DSN"], deadline=deadline
            )
        if revalidation:
            return await sourcing.revalidate(
                command, authority.tenant_id, os.environ["ACTIVEKG_DSN"], deadline=deadline
            )
        config = configuration()
        return await sourcing.search(
            command,
            authority.tenant_id,
            os.environ["ACTIVEKG_DSN"],
            config.policy,
            deadline=deadline,
        )

    try:
        return await asyncio.wait_for(run(), max(0, deadline - time.monotonic()))
    except HTTPException:
        raise
    except Exception:
        # No query, database error, resume values or token can reach the caller.
        raise HTTPException(503, "candidate_sourcing_unavailable") from None


@router.post("/organization-candidates/sourcing/search", response_model=None)
async def search_candidates(request: Request):
    return await _handle(request, revalidation=False)


@router.post("/organization-candidates/sourcing/revalidate", response_model=None)
async def revalidate_candidates(request: Request):
    return await _handle(request, revalidation=True)


@router.post("/organization-candidates/sourcing/public-applicants", response_model=None)
async def public_applicant_membership(request: Request):
    return await _handle(request, revalidation=False, public_membership=True)
