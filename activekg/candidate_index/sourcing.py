"""Bounded read-only resume sourcing; no provider extraction or contact payloads."""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from datetime import datetime
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb
from pydantic import Field, field_validator

from activekg.candidate_index.contracts import IndexPolicy, StrictModel
from activekg.candidate_index.providers import embed_query_once
from activekg.candidate_index.sourcing_contracts import (
    PublicApplicantRequest,
    SourcingDecision,
    SourcingIdentity,
    SourcingRevalidateRequest,
    SourcingSearchRequest,
)

TOTAL_SECONDS = 9.0
RESPONSE_BYTES = 8 * 1024 * 1024
EVIDENCE_BYTES = 65536


class SourcingUnavailable(ValueError):
    pass


def professional_name(value: str) -> str:
    # Reject rather than redact: rewriting a fact could change who qualifies.
    if (
        not value.strip()
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or re.search(r"@|https?://|www\.|gs://|(?:\d[ .()+-]*){9,}", value, re.I)
    ):
        raise ValueError("candidate_sourcing_fact_invalid")
    return value


class SourcingLocation(StrictModel):
    city: str | None = Field(default=None, max_length=200)
    country: str | None = Field(default=None, max_length=200)

    @field_validator("city", "country")
    @classmethod
    def safe_name(cls, value: str | None) -> str | None:
        return professional_name(value) if value is not None else None


class ResumeEmployment(StrictModel):
    title: str | None = Field(default=None, max_length=200)
    employmentType: str | None = Field(default=None, max_length=100)
    start: str | None = Field(default=None, pattern=r"^(19|20|21)\d{2}(-\d{2}(-\d{2})?)?$")
    end: str | None = Field(default=None, pattern=r"^(19|20|21)\d{2}(-\d{2}(-\d{2})?)?$")
    ongoing: bool = False

    @field_validator("title", "employmentType")
    @classmethod
    def safe_name(cls, value: str | None) -> str | None:
        return professional_name(value) if value is not None else None


class ResumeFacts(StrictModel):
    version: int = Field(ge=1, le=1)
    kind: str = Field(pattern=r"^candidate_provided$")
    scope: str = Field(pattern=r"^org_[1-9][0-9]{0,9}$")
    sourceVersion: int = Field(ge=1, le=9_007_199_254_740_991)
    observedAt: datetime
    skills: list[str] = Field(max_length=100)
    employment: list[ResumeEmployment] = Field(default_factory=list, max_length=20)
    location: SourcingLocation
    currentTitle: str | None = Field(default=None, max_length=500)
    seniority: str | None = Field(default=None, max_length=100)
    primaryTitles: list[str] = Field(default_factory=list, max_length=3)
    domains: list[str] = Field(default_factory=list, max_length=10)
    functions: list[str] = Field(default_factory=list, max_length=10)
    certifications: list[str] = Field(default_factory=list, max_length=10)
    industries: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("currentTitle", "seniority")
    @classmethod
    def safe_optional_name(cls, value: str | None) -> str | None:
        return professional_name(value) if value is not None else None

    @field_validator("primaryTitles", "domains", "functions", "certifications", "industries")
    @classmethod
    def safe_names(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value) > 500:
                raise ValueError("candidate_sourcing_fact_invalid")
            professional_name(value)
        return values

    @field_validator("skills")
    @classmethod
    def safe_skills(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value) > 120:
                raise ValueError("candidate_sourcing_fact_invalid")
            professional_name(value)
        return values


class SourcingHit(StrictModel):
    identity: SourcingIdentity
    linkedPublicId: UUID | None
    distance: float
    evidence: ResumeFacts


class Revalidated(StrictModel):
    identity: SourcingIdentity
    decision: SourcingDecision


def _wire(value: object) -> bytes:
    # Spaces mirror jsonb text size conservatively for these bounded structures.
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def search_response(rows: list, request: SourcingSearchRequest, tenant: str) -> dict:
    if len(rows) > request.limit:
        raise SourcingUnavailable("candidate_sourcing_unavailable")
    results, withheld = [], 0
    identities = set()
    for row in rows:
        try:
            hit = SourcingHit.model_validate_json(_wire(row))
            if hit.evidence.scope != tenant or not math.isfinite(hit.distance):
                raise ValueError("candidate_sourcing_fact_invalid")
            if hit.identity.applicationJobId == request.externalJobId:
                raise ValueError("candidate_sourcing_fact_invalid")
            key = str(hit.linkedPublicId or hit.identity.candidateId)
            if key in identities:
                raise ValueError("candidate_sourcing_fact_invalid")
            result = hit.model_dump(mode="json")
            if len(_wire(result["evidence"])) > EVIDENCE_BYTES:
                raise ValueError("candidate_sourcing_fact_invalid")
            identities.add(key)
            results.append(result)
        except (ValueError, TypeError):
            withheld += 1
    response = {
        "version": 1,
        "requestId": request.requestId,
        "results": results,
        "retrievalLimit": 100,
        "saturated": len(rows) == request.limit,
        "counts": {"returned": len(results), "withheldInvalidEvidence": withheld},
    }
    if len(_wire(response)) > RESPONSE_BYTES:
        raise SourcingUnavailable("candidate_sourcing_unavailable")
    return response


def revalidate_response(rows: list, request: SourcingRevalidateRequest) -> dict:
    if len(rows) != len(request.items):
        raise SourcingUnavailable("candidate_sourcing_unavailable")
    results = []
    for row, expected in zip(rows, request.items, strict=True):
        parsed = Revalidated.model_validate_json(_wire(row))
        if parsed.identity != expected:
            raise SourcingUnavailable("candidate_sourcing_unavailable")
        results.append(parsed.model_dump(mode="json"))
    return {"version": 1, "requestId": request.requestId, "results": results}


async def _query(dsn: str, tenant: str, statement: str, args: tuple) -> list:
    # Async connect and SQL are cancelled by the enclosing shared deadline.
    # There is no unbounded pool wait or to_thread query that can outlive it.
    async with await psycopg.AsyncConnection.connect(
        dsn,
        connect_timeout=3,
        options="-c default_transaction_read_only=on -c statement_timeout=3000 -c lock_timeout=1500",
    ) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT set_config('app.current_tenant_id',%s,true)", (tenant,))
            await cur.execute(statement, args)
            return [row[0] for row in await cur.fetchall()]


async def search(
    request: SourcingSearchRequest, tenant: str, dsn: str, policy: IndexPolicy, *, deadline: float
) -> dict:
    async def run():
        vector = await embed_query_once(
            request.query,
            policy.embedding_model_id,
            policy.embedding_artifact_revision,
            deadline=deadline,
        )
        if len(vector) != 384 or not all(math.isfinite(x) for x in vector):
            raise SourcingUnavailable("candidate_sourcing_unavailable")
        rows = await _query(
            dsn,
            tenant,
            "SELECT * FROM public.candidate_index_read_sourcing(%s::vector,%s,%s,%s,%s)",
            (
                json.dumps(vector),
                policy.embedding_model_id,
                policy.embedding_artifact_revision,
                request.limit,
                request.externalJobId,
            ),
        )
        return search_response(rows, request, tenant)

    return await asyncio.wait_for(run(), max(0, deadline - time.monotonic()))


async def revalidate(
    request: SourcingRevalidateRequest, tenant: str, dsn: str, *, deadline: float
) -> dict:
    async def run():
        rows = await _query(
            dsn,
            tenant,
            "SELECT * FROM public.candidate_index_revalidate_sourcing(%s,%s::jsonb)",
            (
                request.externalJobId,
                Jsonb([item.model_dump(mode="json") for item in request.items]),
            ),
        )
        return revalidate_response(rows, request)

    return await asyncio.wait_for(run(), max(0, deadline - time.monotonic()))


async def public_applicants(
    request: PublicApplicantRequest, tenant: str, dsn: str, *, deadline: float
) -> dict:
    async def run():
        rows = await _query(
            dsn,
            tenant,
            "SELECT * FROM public.candidate_index_public_applicants(%s::jsonb)",
            (Jsonb([str(item) for item in request.publicIds]),),
        )
        if len(rows) != len(request.publicIds) or any(
            type(row) is not dict
            or set(row) != {"publicId", "isApplicant"}
            or row["publicId"] != str(expected)
            or type(row["isApplicant"]) is not bool
            for row, expected in zip(rows, request.publicIds, strict=True)
        ):
            raise SourcingUnavailable("candidate_sourcing_unavailable")
        return {"version": 1, "requestId": request.requestId, "results": rows}

    return await asyncio.wait_for(run(), max(0, deadline - time.monotonic()))
