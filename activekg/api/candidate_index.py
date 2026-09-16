"""Flow-service-only source intake and bounded private index reads."""

from __future__ import annotations

import asyncio
import os
from typing import Annotated, Any
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.candidate_index.admission import (
    accept_source,
    configuration,
    enabled,
    require_placement,
)
from activekg.candidate_index.contracts import (
    MAX_REQUEST_BYTES,
    ORG_TENANT,
    IndexContractError,
    SourceContentCommand,
    StrictModel,
)
from activekg.candidate_index.providers import cached_embedding_path
from activekg.candidate_index.repository import IndexRepository
from activekg.candidate_index.search import SearchQuery, SearchRefused, search

router = APIRouter(tags=["organization-candidate-index"])


def api_configuration_problems() -> list[str]:
    """Name-only posture; no model load, connection, provider call or path disclosure."""
    try:
        if require_placement("api"):
            config = configuration()
            if not os.environ.get("ACTIVEKG_DSN"):
                raise IndexContractError("candidate_index_database_missing")
            cached_embedding_path(
                config.policy.embedding_model_id, config.policy.embedding_artifact_revision
            )
        return []
    except Exception:
        return ["candidate_index_api_configuration_invalid"]


def _authority(claims: JWTClaims | None, scope: str) -> JWTClaims:
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(401, "candidate_index_auth_required")
    if (
        auth.JWT_AUDIENCE != "activekg"
        or claims.issuer != "vantahire"
        or claims.actor_type != "service"
        or claims.actor_id != "vantahire-backend"
        or claims.scopes != [scope]
        or not isinstance(claims.tenant_id, str)
        or ORG_TENANT.fullmatch(claims.tenant_id) is None
        or int(claims.tenant_id[4:]) > 2_147_483_647
    ):
        raise HTTPException(403, "candidate_index_auth_denied")
    return claims


async def writer(claims: JWTClaims | None = Depends(get_jwt_claims)) -> JWTClaims:
    return _authority(claims, "organization-candidate-source:write")


async def reader(claims: JWTClaims | None = Depends(get_jwt_claims)) -> JWTClaims:
    return _authority(claims, "organization-candidate-index:read")


def _repository() -> tuple[IndexRepository, Any]:
    try:
        if not enabled("api"):
            raise IndexContractError("candidate_index_disabled")
        config = configuration()
        dsn = os.environ.get("ACTIVEKG_DSN")
        if not dsn:
            raise IndexContractError("candidate_index_database_missing")
        return IndexRepository(dsn, tunables=config.tunables), config
    except Exception:
        raise HTTPException(503, "candidate_index_unavailable") from None


async def _body(request: Request, model: type[StrictModel], cap: int) -> Any:
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise HTTPException(415, "candidate_index_json_required")
    length = request.headers.get("content-length")
    if length is not None:
        if not length.isdecimal():
            raise HTTPException(422, "candidate_index_length_invalid")
        if len(length) > 10 or int(length) > cap:
            raise HTTPException(413, "candidate_index_request_too_large")
    raw = bytearray()
    async for chunk in request.stream():
        if len(raw) + len(chunk) > cap:
            raise HTTPException(413, "candidate_index_request_too_large")
        raw.extend(chunk)
    try:
        return model.model_validate_json(bytes(raw))
    except (ValidationError, ValueError):
        # Validation exceptions contain applicant fields; never serialise them.
        raise HTTPException(422, "candidate_index_schema_invalid") from None


@router.post("/organization-candidates/source-content", response_model=None)
async def source_content(request: Request, claims: Annotated[JWTClaims, Depends(writer)]):
    repo, config = _repository()
    command = await _body(request, SourceContentCommand, MAX_REQUEST_BYTES)
    try:
        result = await asyncio.to_thread(accept_source, command, claims.tenant_id, repo, config)
    except HTTPException:
        raise
    except IndexContractError:
        raise HTTPException(422, "candidate_index_command_invalid") from None
    except psycopg.errors.UniqueViolation:
        raise HTTPException(409, "candidate_index_command_conflict") from None
    except Exception:
        raise HTTPException(503, "candidate_index_unavailable") from None
    return JSONResponse(result, status_code=201 if result["outcome"] == "accepted" else 200)


class StatusQuery(StrictModel):
    reference_ids: list[UUID] = Field(min_length=1, max_length=50)


def _statuses(query: StatusQuery, tenant: str, repo: IndexRepository) -> dict[str, Any]:
    with repo.transaction(tenant) as cur:
        cur.execute(
            "SELECT s.reference_id,s.source_id,s.source_observed_at,"
            "public.candidate_index_status(s.source_id) FROM candidate_index_sources s "
            "WHERE s.scope_key=%s AND s.reference_id=ANY(%s::uuid[]) "
            "AND s.source_kind='organization_application' ORDER BY s.reference_id,s.source_version",
            (tenant, query.reference_ids),
        )
        rows = cur.fetchall()
    results = []
    for reference, source, observed, state in rows:
        if not state or not state.get("eligible"):
            continue
        results.append(
            {
                "reference_id": str(reference),
                "source_id": str(source),
                "source_observed_at": observed.isoformat(),
                "state": state.get("state"),
                "generation_id": state.get("generation_id"),
                "published_generation_id": state.get("published_generation_id"),
                "error_code": state.get("error_code"),
            }
        )
    return {"results": results}


@router.post("/organization-candidates/index-status", response_model=None)
async def index_status(request: Request, claims: Annotated[JWTClaims, Depends(reader)]):
    repo, _ = _repository()
    query = await _body(request, StatusQuery, 8192)
    try:
        return await asyncio.to_thread(_statuses, query, claims.tenant_id, repo)
    except Exception:
        raise HTTPException(503, "candidate_index_unavailable") from None


@router.post("/organization-candidates/search", response_model=None)
async def private_search(request: Request, claims: Annotated[JWTClaims, Depends(reader)]):
    repo, config = _repository()
    query = await _body(request, SearchQuery, 16384)
    try:
        return await search(query, claims.tenant_id, repo, config.policy)
    except SearchRefused as exc:
        code = 503 if exc.code == "candidate_index_search_unavailable" else 422
        raise HTTPException(code, exc.code) from None
    except Exception:
        raise HTTPException(503, "candidate_index_search_unavailable") from None
