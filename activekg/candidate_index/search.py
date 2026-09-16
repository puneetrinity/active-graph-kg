"""Bounded, organisation-only fusion; model work never holds a SQL transaction.

The legacy lane delegates to the frozen graph reader, with an exact-tenant,
read-only connection adapter. No legacy node, provenance or consent record is
fabricated. A complete managed generation replaces that application's legacy
payload, not its authority. Unsupported metadata is a refusal, not a dropped filter.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

import numpy as np
from pgvector.psycopg import register_vector
from pydantic import Field, field_validator

from activekg.candidate_index.contracts import ORG_TENANT, IndexPolicy, StrictModel, sha256
from activekg.candidate_index.providers import WorkRefused, embed_query_once, rerank_once
from activekg.candidate_index.repository import IndexRepository
from activekg.graph.repository import GraphRepository

RETRIEVAL_LIMIT = 100
RRF_K = 60
COUNT_LIMIT = 1000


class SearchRefused(Exception):
    def __init__(self, code: str = "candidate_index_search_unavailable"):
        if code not in {
            "candidate_index_search_unavailable",
            "candidate_index_filter_unsupported",
            "candidate_index_filter_conflict",
            "candidate_index_query_too_long",
        }:
            raise ValueError("candidate_index_search_error_invalid")
        self.code = code
        super().__init__(code)


class SearchQuery(StrictModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=20, ge=1, le=100)
    job_id: int | None = Field(default=None, ge=1, le=2_147_483_647)
    use_hybrid: bool = True
    use_reranker: bool = True
    metadata_filters: dict[str, Any] = Field(default_factory=dict, max_length=3)

    @field_validator("query")
    @classmethod
    def bounded_query(cls, value: str) -> str:
        if not value.strip() or any(ord(c) < 32 and c not in "\n\t\r" for c in value):
            raise ValueError("candidate_index_query_invalid")
        return value


def filters_for(query: SearchQuery, tenant: str) -> tuple[int | None, dict[str, Any]]:
    if not isinstance(tenant, str) or not ORG_TENANT.fullmatch(tenant):
        raise SearchRefused()
    org = int(tenant[4:])
    if org > 2_147_483_647:
        raise SearchRefused()
    supplied = query.metadata_filters
    if set(supplied) - {"source", "org_id", "job_id"}:
        raise SearchRefused("candidate_index_filter_unsupported")
    # The old graph reader compares metadata->>key with str(value). Preserve
    # canonical integer/string equality, not truthiness, bool coercion or JSON SQL.
    if "source" in supplied and supplied["source"] != "vantahire":
        raise SearchRefused("candidate_index_filter_conflict")
    if "org_id" in supplied and str(supplied["org_id"]) != str(org):
        raise SearchRefused("candidate_index_filter_conflict")
    job = query.job_id
    if "job_id" in supplied:
        raw = supplied["job_id"]
        if type(raw) not in (int, str) or not re.fullmatch(r"[1-9][0-9]{0,9}", str(raw)):
            raise SearchRefused("candidate_index_filter_conflict")
        candidate = int(raw)
        if candidate > 2_147_483_647 or (job is not None and candidate != job):
            raise SearchRefused("candidate_index_filter_conflict")
        job = candidate
    result = {"source": "vantahire", "org_id": org}
    if job is not None:
        result["job_id"] = job
    return job, result


class _QuietReaderLog:
    # The frozen reader logs metadata filter values in debug/info messages.
    # This dedicated instance never emits applicant text, keys, ids or queries.
    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


class LegacyPrivateReader(GraphRepository):
    """Only search is used; SQL algorithms remain the frozen implementation."""

    def __init__(self, repo: IndexRepository, tenant: str):
        self.index_repo = repo
        self.tenant = tenant
        self.candidate_factor = 2.0
        self.logger = _QuietReaderLog()
        self._cross_encoder = None  # Reranking happens once, on the authorised union.

    @contextmanager
    def _conn(self, tenant_id: str | None = None):
        if tenant_id != self.tenant or not ORG_TENANT.fullmatch(tenant_id or ""):
            raise SearchRefused()
        with self.index_repo.transaction(tenant_id) as cur:
            cur.execute("SET LOCAL transaction_read_only=on")
            register_vector(cur.connection)
            yield cur.connection


@dataclass(frozen=True)
class Hit:
    application_id: int
    job_id: int
    cosine: float | None
    highlights: tuple[str, ...]
    document: str
    lane_score: float
    reference_id: str | None = None
    resume_version_id: str | None = None
    generation_id: str | None = None
    generation: int | None = None
    observed: str | None = None
    state: str = "legacy"
    matched_chunks: int = 1
    ranking: float = 0
    evidence: tuple[str, ...] = ()

    @property
    def identity(self) -> tuple:
        return (
            self.application_id,
            self.job_id,
            self.reference_id,
            self.resume_version_id,
            self.generation_id,
            self.generation,
            self.evidence,
        )

    def public(self) -> dict[str, Any]:
        return {
            "application_id": self.application_id,
            "job_id": self.job_id,
            "reference_id": self.reference_id,
            "resume_version_id": self.resume_version_id,
            "generation_id": self.generation_id,
            "generation": self.generation,
            "source_observed_at": self.observed,
            "state": self.state,
            "cosine_score": self.cosine,
            "ranking_score": self.ranking,
            "matched_chunks": self.matched_chunks,
            "highlights": list(self.highlights),
        }


def professional_highlight(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    # Only short professional snippets cross this serializer. Raw legacy props,
    # contacts and storage locators never do. Redact before shortening so an email
    # or phone straddling the snippet edge cannot escape partially.
    value = re.sub(r"[^\s@]+@[^\s@]+", "[contact removed]", value[:65536])
    value = re.sub(r"(?:https?://|gs://|www\.)\S+", "[link removed]", value, flags=re.I)
    value = re.sub(r"(?<!\w)\+?(?:\d[\s().-]*){9,}(?!\w)", "[contact removed]", value)
    value = "".join(c if ord(c) >= 32 else " " for c in value)
    return " ".join(value.split())[:240]


def _identifier(value: Any) -> int:
    if type(value) not in (int, str) or not re.fullmatch(r"[1-9][0-9]{0,9}", str(value)):
        raise SearchRefused()
    result = int(value)
    if result > 2_147_483_647:
        raise SearchRefused()
    return result


def legacy_hits(rows: list[tuple[Any, float]], tenant: str, vector: list[float]) -> list[Hit]:
    found: dict[int, Hit] = {}
    for node, score in rows:
        meta = node.metadata or {}
        if (
            node.tenant_id != tenant
            or str(meta.get("org_id")) != tenant[4:]
            or meta.get("source") != "vantahire"
        ):
            raise SearchRefused()
        app, job = _identifier(meta.get("application_id")), _identifier(meta.get("job_id"))
        if not math.isfinite(score):
            raise SearchRefused()
        props = node.props or {}
        text = next(
            (
                props[k]
                for k in ("text", "resume_text", "content", "body", "description")
                if isinstance(props.get(k), str) and props[k]
            ),
            "",
        )
        title = next(
            (
                props[k]
                for k in ("title", "job_title", "current_title")
                if isinstance(props.get(k), str) and props[k]
            ),
            "",
        )
        highlight = professional_highlight(text)
        cosine = None
        embedding = getattr(node, "embedding", None)
        if embedding is not None:
            if np.shape(embedding) != (384,):
                raise SearchRefused()
            raw = float(np.dot(embedding, vector))
            if not math.isfinite(raw):
                raise SearchRefused()
            cosine = max(-1.0, min(1.0, raw))
        hit = Hit(
            app,
            job,
            cosine,
            (highlight,) if highlight else (),
            " ".join(part for part in (title, text) if part)[:512],
            float(score),
            evidence=(
                sha256(
                    str(node.id)
                    + ":"
                    + str(getattr(node, "version", ""))
                    + ":"
                    + title
                    + ":"
                    + text
                ),
            ),
        )
        prior = found.get(app)
        if prior:
            if prior.job_id != job:
                raise SearchRefused()
            best = hit if hit.lane_score > prior.lane_score else prior
            hit = replace(
                best,
                highlights=tuple(dict.fromkeys(prior.highlights + hit.highlights))[:3],
                matched_chunks=prior.matched_chunks + 1,
                evidence=tuple(sorted(prior.evidence + hit.evidence)),
            )
        found[app] = hit
    return sorted(found.values(), key=lambda hit: (-hit.lane_score, hit.application_id))


def managed_hits(rows: list[dict[str, Any]]) -> list[Hit]:
    results = []
    seen = set()
    for row in rows:
        app, job = _identifier(row["application_id"]), _identifier(row["job_id"])
        if app in seen or row["state"] not in {"ready", "updating", "refresh_failed"}:
            raise SearchRefused()
        seen.add(app)
        cosine = float(row["score"])
        if not math.isfinite(cosine) or not -1.001 <= cosine <= 1.001:
            raise SearchRefused()
        chunks = tuple(
            dict.fromkeys(
                filter(None, (professional_highlight(h) for h in row.get("highlights") or []))
            )
        )[:3]
        observed = row["source_observed_at"]
        if isinstance(observed, datetime):
            observed = observed.isoformat()
        results.append(
            Hit(
                app,
                job,
                max(-1.0, min(1.0, cosine)),
                chunks,
                " ".join(chunks)[:512],
                cosine,
                str(row["reference_id"]),
                str(row["resume_version_id"]),
                str(row["generation_id"]),
                row["generation"],
                observed,
                row["state"],
            )
        )
    return sorted(results, key=lambda hit: (-hit.lane_score, hit.application_id))


def fuse(managed: list[Hit], legacy: list[Hit]) -> list[Hit]:
    selected = {hit.application_id: hit for hit in legacy}
    selected.update({hit.application_id: hit for hit in managed})
    scores: dict[int, float] = {}
    for lane in (managed, legacy):
        for rank, hit in enumerate(lane, 1):
            scores[hit.application_id] = scores.get(hit.application_id, 0) + 1 / (RRF_K + rank)
    return sorted(
        (replace(hit, ranking=scores[app]) for app, hit in selected.items()),
        key=lambda hit: (-hit.ranking, hit.application_id),
    )


def retrieve(
    query: SearchQuery, tenant: str, repo: IndexRepository, policy: IndexPolicy, vector: list[float]
) -> tuple[list[Hit], bool, dict[str, Any]]:
    job, filters = filters_for(query, tenant)
    with repo.transaction(tenant) as cur:
        cur.execute("SET LOCAL transaction_read_only=on")
        cur.execute(
            "SELECT public.candidate_index_read_private(%s::vector,%s,%s,%s,%s)",
            (
                str(vector),
                policy.embedding_model_id,
                policy.embedding_artifact_revision,
                RETRIEVAL_LIMIT,
                job,
            ),
        )
        managed = managed_hits([row[0] for row in cur.fetchall()])
        cur.execute(
            "SELECT public.candidate_index_status(s.source_id) FROM candidate_index_sources s "
            "JOIN organization_candidate_references r ON r.tenant_id=s.tenant_id "
            "AND r.reference_id=s.reference_id AND r.candidate_id=s.candidate_id "
            "WHERE s.scope_key=%s AND s.source_kind='organization_application' "
            "AND (%s::integer IS NULL OR r.job_id=%s) ORDER BY s.source_id LIMIT %s",
            (tenant, job, job, COUNT_LIMIT + 1),
        )
        states = [row[0] for row in cur.fetchall()]
    counts = dict.fromkeys(
        ("ready", "updating", "refresh_failed", "pending", "needs_review", "failed"), 0
    )
    for state in states[:COUNT_LIMIT]:
        if not state or state.get("eligible") is not True:
            continue
        published = state.get("published_generation_id")
        bad = state.get("state") in {"failed", "needs_review", "quarantined"}
        key = (
            (
                "ready"
                if published == state.get("generation_id")
                else "refresh_failed"
                if bad
                else "updating"
            )
            if published
            else (
                "needs_review"
                if state.get("state") in {"needs_review", "quarantined"}
                else "failed"
                if bad
                else "pending"
            )
        )
        counts[key] += 1
    legacy = LegacyPrivateReader(repo, tenant)
    args = {
        "query_embedding": np.asarray(vector, dtype=np.float32),
        "top_k": RETRIEVAL_LIMIT,
        "metadata_filters": filters,
        "tenant_id": tenant,
    }
    if query.use_hybrid:
        rows = legacy.hybrid_search(query_text=query.query, use_reranker=False, **args)
        if not rows:
            rows = legacy.vector_search(**args)
    else:
        rows = legacy.vector_search(**args)
    hits = legacy_hits(rows, tenant, vector)
    return (
        fuse(managed, hits),
        len(managed) >= RETRIEVAL_LIMIT or len(rows) >= RETRIEVAL_LIMIT,
        {
            "counts": counts,
            "bounded": len(states) > COUNT_LIMIT,
            "limit": COUNT_LIMIT,
        },
    )


async def search(
    query: SearchQuery, tenant: str, repo: IndexRepository, policy: IndexPolicy
) -> dict[str, Any]:
    filters_for(query, tenant)  # Refuse unsupported input before any model or SQL work.
    deadline = time.monotonic() + 9
    try:
        vector = await embed_query_once(
            query.query,
            policy.embedding_model_id,
            policy.embedding_artifact_revision,
            deadline=deadline,
        )
        hits, saturated, processing = await asyncio.to_thread(
            retrieve, query, tenant, repo, policy, vector
        )
        reranker = "not_requested" if not query.use_reranker else "not_needed"
        score_type = "rrf_fused"
        if query.use_reranker and len(hits) > 1:
            try:
                # Honour a tighter pre-existing local rerank budget. Zero means
                # the old reader's unbounded default; this lane still caps work.
                configured_ms = float(os.getenv("MAX_RERANK_BUDGET_MS", "0"))
                if not math.isfinite(configured_ms) or configured_ms < 0:
                    raise WorkRefused("provider_unavailable")
                seconds = min(2.0, configured_ms / 1000) if configured_ms > 0 else 2.0
                scores = await rerank_once(
                    query.query,
                    [hit.document for hit in hits],
                    deadline=min(deadline, time.monotonic() + seconds),
                )
                hits = sorted(
                    (replace(hit, ranking=score) for hit, score in zip(hits, scores, strict=True)),
                    key=lambda hit: (-hit.ranking, hit.application_id),
                )
                reranker, score_type = "applied", "cross_encoder"
            except WorkRefused:
                reranker = "fallback"
            # No stale authority or generation survives time spent in model work.
            current, again_saturated, processing = await asyncio.to_thread(
                retrieve, query, tenant, repo, policy, vector
            )
            admitted = {hit.identity: hit for hit in current}
            retained = [
                replace(admitted[hit.identity], ranking=hit.ranking)
                for hit in hits
                if hit.identity in admitted
            ]
            saturated = saturated or again_saturated or len(retained) != len(hits)
            hits = retained
        if time.monotonic() >= deadline:
            raise SearchRefused()
        return {
            "results": [hit.public() for hit in hits[: query.top_k]],
            "score_type": score_type,
            "display_score_type": "cosine",
            "reranker": reranker,
            "saturated": saturated,
            "processing": processing,
            "retrieval_limit": RETRIEVAL_LIMIT,
        }
    except WorkRefused as exc:
        raise SearchRefused(
            "candidate_index_query_too_long"
            if exc.code == "chunk_overflow"
            else "candidate_index_search_unavailable"
        ) from None
