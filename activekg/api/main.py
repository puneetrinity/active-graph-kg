from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal, cast

from dotenv import load_dotenv

load_dotenv()  # Load .env file at startup

import numpy as np
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# JWT authentication and rate limiting
from activekg.api.auth import (
    JWT_ENABLED,
    JWTClaims,
    get_jwt_claims,
    require_scope,
    verification_key_problems,
)
from activekg.api.global_memory import (
    LEGACY_GLOBAL_SEARCH_ENABLED,
    PUBLIC_PROFILE_SEARCH_ENABLED,
)
from activekg.api.global_memory import (
    router as global_memory_router,
)
from activekg.api.middleware import apply_rate_limit, get_tenant_context, require_rate_limit
from activekg.api.operational import (
    MetricsBoundary,
    OperationalBusy,
    OperationalPayloadTooLarge,
    ReadinessCoordinator,
    bounded_readiness_check,
)
from activekg.api.rate_limiter import RATE_LIMIT_ENABLED, get_identifier, rate_limiter
from activekg.api.retirement import (
    connector_retirement_router,
    grounded_qa_retirement_router,
    public_observability_retirement_router,
    semantic_triggers_router,
)
from activekg.common.control_plane import (
    ControlPlaneUnauthorized,
    ControlPlaneUnavailable,
    verify_control_plane_authorization,
)
from activekg.common.env import env_str
from activekg.common.logger import clear_log_context, get_enhanced_logger, set_log_context
from activekg.common.metrics import get_redis_client, metrics
from activekg.common.validation import (
    EdgeCreate,
    KGSearchRequest,
    NodeBatchCreate,
    NodeCreate,
)
from activekg.embedding.queue import (
    enqueue_embedding_job,
    get_pending_count,
    queue_depth,
)
from activekg.engine.embedding_provider import EmbeddingProvider
from activekg.extraction.queue import (
    enqueue_extraction_job,
    extraction_queue_depth,
)
from activekg.graph.candidate_identifiers import (
    IDENTIFIER_TYPES,
    IdentifierNormalizationError,
    normalize_identifier,
)
from activekg.graph.candidate_repository import (
    CandidateRepository,
    IdentifierConflict,
)
from activekg.graph.models import Candidate, CandidateSourceRecord, Edge, Node
from activekg.graph.repository import GraphRepository

# Prometheus observability
from activekg.observability import track_embedding_health, track_search_request
from activekg.observability.metrics import record_api_error
from activekg.refresh.scheduler import RefreshScheduler

# Metrics enabled flag
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() == "true"

_embedding_redis_client = None


def _get_embedding_redis():
    global _embedding_redis_client
    if _embedding_redis_client is not None:
        return _embedding_redis_client
    try:
        _embedding_redis_client = get_redis_client()
        return _embedding_redis_client
    except Exception as e:
        logger.warning("Embedding Redis unavailable", extra_fields={"error": str(e)})
        return None


def _check_embedding_queue_capacity(redis_client, tenant_id: str | None, requested: int) -> None:
    depth = queue_depth(redis_client)
    if depth["queue"] + depth["retry"] + requested > EMBEDDING_QUEUE_MAX_DEPTH:
        raise HTTPException(
            status_code=429,
            detail="Embedding queue overloaded, please retry later",
        )
    if EMBEDDING_TENANT_MAX_PENDING > 0:
        pending = get_pending_count(redis_client, tenant_id)
        if pending + requested > EMBEDDING_TENANT_MAX_PENDING:
            raise HTTPException(
                status_code=429,
                detail="Tenant embedding queue limit exceeded, please retry later",
            )


class EmbeddingRequeueRequest(BaseModel):
    """Request model to requeue embeddings and backfill statuses."""

    tenant_id: str | None = None
    node_ids: list[str] | None = None
    status: str | None = Field(
        "failed", description="Filter nodes by status (failed, queued, ready, etc.)"
    )
    only_missing_embedding: bool = Field(
        False, description="Only requeue nodes without embeddings (embedding IS NULL)"
    )
    backfill_ready: bool = Field(
        True, description="Mark nodes with embeddings as 'ready' before requeuing"
    )
    limit: int = 2000


class ExtractionRequeueRequest(BaseModel):
    """Request model to requeue extraction jobs."""

    tenant_id: str | None = None
    node_ids: list[str] | None = None
    status: str | None = Field(
        None, description="Filter by extraction_status (null, failed, queued, etc.)"
    )
    only_null_status: bool = Field(
        True, description="Only requeue nodes with no extraction_status (never queued)"
    )
    limit: int = 2000


APP_VERSION = os.getenv("ACTIVEKG_VERSION", "1.0.0")
# Prefer ACTIVEKG_DSN; fall back to DATABASE_URL for PaaS (e.g., Railway Postgres plugin)
DSN = env_str(
    ["ACTIVEKG_DSN", "DATABASE_URL"], "postgresql://activekg:activekg@localhost:5432/activekg"
)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
WEIGHTED_SEARCH_CANDIDATE_FACTOR = float(os.getenv("WEIGHTED_SEARCH_CANDIDATE_FACTOR", "2.0"))

AUTO_EMBED_ON_CREATE = os.getenv("AUTO_EMBED_ON_CREATE", "true").lower() == "true"
EMBEDDING_ASYNC = os.getenv("EMBEDDING_ASYNC", "false").lower() == "true"
EMBEDDING_QUEUE_MAX_DEPTH = int(os.getenv("EMBEDDING_QUEUE_MAX_DEPTH", "5000"))
EMBEDDING_TENANT_MAX_PENDING = int(os.getenv("EMBEDDING_TENANT_MAX_PENDING", "2000"))
EMBEDDING_QUEUE_REQUIRE_REDIS = os.getenv("EMBEDDING_QUEUE_REQUIRE_REDIS", "true").lower() == "true"
NODE_BATCH_MAX = int(os.getenv("NODE_BATCH_MAX", "200"))

# Extraction settings
EXTRACTION_ENABLED = os.getenv("EXTRACTION_ENABLED", "false").lower() == "true"
EXTRACTION_MODE = os.getenv("EXTRACTION_MODE", "async")  # "async" or "sync"
RUN_SCHEDULER = os.getenv("RUN_SCHEDULER", "true").lower() == "true"

MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(10 * 1024 * 1024)))
_BODYLESS_RETIREMENT_PATHS = {"/ask", "/ask/stream", "/debug/search_explain"}


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # These compatibility tombstones never read a body. Preserve their stable
        # retirement contract even when callers send malformed or oversized metadata.
        if request.url.path in _BODYLESS_RETIREMENT_PATHS:
            return await call_next(request)

        # Enforce Content-Length if present
        try:
            cl = request.headers.get("content-length")
            if cl is not None and int(cl) > MAX_REQUEST_SIZE:
                return PlainTextResponse("Request too large", status_code=413)
        except Exception:
            pass

        # For chunked transfers (no Content-Length), wrap receive to enforce limit
        if request.headers.get("transfer-encoding", "").lower() == "chunked":
            original_receive = request.receive
            total_size = 0

            async def limited_receive():
                nonlocal total_size
                message = await original_receive()
                if message.get("type") == "http.request":
                    body = message.get("body", b"")
                    total_size += len(body or b"")
                    if total_size > MAX_REQUEST_SIZE:
                        # Abort with 413 once size exceeded
                        raise HTTPException(status_code=413, detail="Request too large")
                return message

            # Monkey-patch receive for this request scope
            request._receive = limited_receive

        return await call_next(request)


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Always create/bind a request ID; do not trust tenant headers here.
        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        try:
            # Bind only request_id at this stage. Tenant context is derived from JWT
            # by endpoint dependencies and may be added to logs at call sites.
            set_log_context(request_id=req_id)
            response = await call_next(request)
        finally:
            clear_log_context()
        response.headers["X-Request-ID"] = req_id
        return response


app = FastAPI(
    title="Active Graph KG",
    version=APP_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.add_middleware(BodySizeLimitMiddleware)
app.add_middleware(CorrelationIDMiddleware)


def get_route_name(request: Request) -> str:
    """Extract route name/template from Starlette request.

    Returns route template like "/nodes/{node_id}" instead of "/nodes/abc-123"
    to avoid high cardinality in metrics.
    """
    try:
        # Access Starlette's route matching
        if hasattr(request, "scope") and "route" in request.scope:
            route = request.scope["route"]
            if hasattr(route, "path"):
                return cast(str, route.path)

        # Fallback: try to match against app routes
        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match.name == "full":  # Full match
                if hasattr(route, "path"):
                    return cast(str, route.path)

        # Final fallback to raw path
        return request.url.path
    except Exception:
        return request.url.path


class ApiErrorMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            if response.status_code >= 400:
                try:
                    route_name = get_route_name(request)
                    # Categorize error type based on status code
                    if response.status_code == 400:
                        error_type = "bad_request"
                    elif response.status_code == 401:
                        error_type = "unauthorized"
                    elif response.status_code == 403:
                        error_type = "forbidden"
                    elif response.status_code == 404:
                        error_type = "not_found"
                    elif response.status_code == 413:
                        error_type = "request_too_large"
                    elif response.status_code == 422:
                        error_type = "validation_error"
                    elif response.status_code == 429:
                        error_type = "rate_limit_exceeded"
                    elif 400 <= response.status_code < 500:
                        error_type = "client_error"
                    else:
                        error_type = "server_error"

                    record_api_error(route_name, response.status_code, error_type)
                except Exception:
                    pass
            return response
        except Exception as exc:
            # Count as 500
            try:
                route_name = get_route_name(request)
                error_type = type(exc).__name__.lower() if exc else "internal_error"
                record_api_error(route_name, 500, error_type)
            except Exception:
                pass
            raise


app.add_middleware(ApiErrorMetricsMiddleware)
logger = get_enhanced_logger(__name__)

# Lazy initialization for test mode (allows import without DB connection)
TEST_MODE = os.getenv("ACTIVEKG_TEST_NO_DB", "false").lower() == "true"

if TEST_MODE:
    # Test mode: defer initialization, use None/mocks
    repo = None
    embedder = None
    scheduler: RefreshScheduler | None = None
    candidate_repo: CandidateRepository | None = None
    logger.warning("Running in TEST_MODE - DB connections deferred")
else:
    # Normal mode: eager initialization
    repo = GraphRepository(DSN, candidate_factor=WEIGHTED_SEARCH_CANDIDATE_FACTOR)
    embedder = EmbeddingProvider(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)
    scheduler = None
    candidate_repo = CandidateRepository(DSN)

app.include_router(global_memory_router)
app.include_router(semantic_triggers_router)
app.include_router(connector_retirement_router)
app.include_router(grounded_qa_retirement_router)
app.include_router(public_observability_retirement_router)

# Global-candidate vector search reuses the process-wide embedding model.
if embedder is not None:
    from activekg.api.global_memory import set_embedder as _gm_set_embedder

    _gm_set_embedder(embedder)


@app.on_event("startup")
def startup_event():
    """Initialize system on startup."""
    logger.info(
        "Active Graph KG startup",
        extra_fields={
            "version": APP_VERSION,
            "weighted_search_candidate_factor": WEIGHTED_SEARCH_CANDIDATE_FACTOR,
            "run_scheduler": RUN_SCHEDULER,
        },
    )

    # Log runtime ML dependencies to verify loaded versions
    try:
        import sentence_transformers
        import torch

        logger.info(
            "ML runtime versions loaded",
            extra_fields={
                "torch": torch.__version__,
                "sentence_transformers": sentence_transformers.__version__,
            },
        )
        # Fail-fast if sentence-transformers >= 5.0
        st_version = sentence_transformers.__version__
        if st_version.startswith("5."):
            raise RuntimeError(
                f"sentence-transformers {st_version} has meta tensor bugs. "
                "Please downgrade to 3.3.1 (see requirements.txt)"
            )
    except ImportError as e:
        logger.warning(f"ML dependencies check failed: {e}")

    # Auto-enable vector index if not present
    repo.ensure_vector_index()

    # Start refresh scheduler (only if RUN_SCHEDULER=true)
    global scheduler
    if RUN_SCHEDULER:
        try:
            scheduler = RefreshScheduler(repo, embedder, trigger_engine=None)
            scheduler.start()
            logger.info("RefreshScheduler started on startup")
        except Exception as e:
            logger.error("Failed to start RefreshScheduler", extra_fields={"error": str(e)})
    else:
        logger.info("Scheduler disabled (RUN_SCHEDULER=false)")

    # Verify JWT and rate limiting configuration
    if JWT_ENABLED:
        logger.info("JWT authentication enabled", extra_fields={"algorithm": "RS256/HS256"})
    else:
        logger.info("JWT authentication DISABLED (dev mode)")

    if RATE_LIMIT_ENABLED:
        if rate_limiter.enabled:
            logger.info("Rate limiter enabled", extra_fields={"redis_available": True})
        else:
            logger.warning(
                "Rate limiting requested but Redis unavailable. Limiter will fail open (allow all requests)."
            )
    else:
        logger.info("Rate limiting DISABLED")


@app.on_event("shutdown")
def shutdown_event():
    """Clean shutdown for background components."""
    global scheduler
    try:
        if scheduler:
            scheduler.shutdown()
            logger.info("RefreshScheduler stopped on shutdown")
    except Exception as e:
        logger.error("Scheduler shutdown error", extra_fields={"error": str(e)})


@app.get("/health", response_model=None)
def health() -> JSONResponse:
    """Constant-cost process liveness; deliberately performs no dependency I/O."""

    return JSONResponse(
        content={"status": "alive", "service": "activekg-api"},
        headers={"Cache-Control": "no-store"},
    )


_readiness_coordinator = ReadinessCoordinator()
_metrics_boundary = MetricsBoundary()


def _require_control_plane(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    """Fail closed before an operational endpoint touches a dependency."""

    try:
        verify_control_plane_authorization(authorization)
    except ControlPlaneUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "CONTROL_PLANE_AUTH_UNAVAILABLE",
                "message": "Operational authentication is unavailable.",
            },
            headers={"Cache-Control": "no-store"},
        ) from exc
    except ControlPlaneUnauthorized as exc:
        raise HTTPException(
            status_code=401,
            detail={
                "code": "CONTROL_PLANE_AUTH_REQUIRED",
                "message": "Operational authentication is required.",
            },
            headers={"Cache-Control": "no-store", "WWW-Authenticate": "Bearer"},
        ) from exc


@app.get("/readyz", response_model=None)
def readyz(
    _control_plane: None = Depends(_require_control_plane),
    cache_control: str | None = Header(default=None, alias="Cache-Control"),
) -> JSONResponse:
    """Return one cached, single-flight, bounded readiness snapshot."""

    del _control_plane
    raw_cache_control = cache_control if isinstance(cache_control, str) else ""
    force_refresh = any(
        directive.partition("=")[0].strip().lower() == "no-cache"
        for directive in raw_cache_control.split(",")
    )
    try:
        result = _readiness_coordinator.run(
            lambda: bounded_readiness_check(
                candidate_repo,
                unsafe_search_configuration=(
                    PUBLIC_PROFILE_SEARCH_ENABLED and LEGACY_GLOBAL_SEARCH_ENABLED
                ),
                jwt_enabled=JWT_ENABLED,
                jwt_problems=verification_key_problems(),
            ),
            force_refresh=force_refresh,
        )
    except OperationalBusy:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reasons": ["readiness_busy"]},
            headers={"Cache-Control": "no-store"},
        )

    return JSONResponse(
        status_code=200 if result.ready else 503,
        content={
            "status": "ready" if result.ready else "not_ready",
            **({"reasons": list(result.reasons)} if result.reasons else {}),
        },
        headers={"Cache-Control": "no-store"},
    )


@app.get("/_admin/security/limits", response_model=None)
def get_security_limits(claims: JWTClaims | None = Depends(get_jwt_claims)) -> dict[str, Any]:
    """Report the active request and closed external-content boundaries.

    Returns current configuration for:
    - External payload-reference availability
    - Request body size limits

    Security:
        - When JWT is enabled, requires authenticated token
        - No admin scope required (read-only configuration)
    """
    return {
        "external_payload_loading": {
            "enabled": False,
            "accepted_sources": ["inline_node_properties", "bounded_multipart_upload"],
        },
        "ssrf_protection": {
            "enabled": False,
            "reason": "Remote payload-reference loading is unavailable.",
        },
        "file_payload_ref_loading": {
            "enabled": False,
            "reason": "Local file payload-reference loading is unavailable.",
        },
        "request_limits": {
            "max_request_body_bytes": MAX_REQUEST_SIZE,
            "max_request_body_mb": round(MAX_REQUEST_SIZE / (1024 * 1024), 2),
            "enforced_for": ["Content-Length header", "chunked transfers"],
        },
    }


@app.get("/debug/dbinfo", response_model=None)
def debug_dbinfo(
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Debug endpoint to inspect DB and tenant context.

    Security:
        - When JWT is enabled, require an authenticated token with admin privileges (admin:refresh scope).
        - When JWT is disabled (dev mode), allow access.

    Returns:
        {
          "database": str,
          "tenant_context": Optional[str],
          "server_host": Optional[str],
          "server_port": Optional[int]
        }
    """
    assert repo is not None, "GraphRepository not initialized"
    # Enforce admin scope when JWT is enabled
    if JWT_ENABLED:
        if not claims:
            raise HTTPException(status_code=401, detail="Authentication required")
        if "admin:refresh" not in (claims.scopes or []):
            raise HTTPException(
                status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
            )

    try:
        # Use a pooled connection without setting tenant_id; report current tenant context from server
        with repo._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT current_database(), current_setting('app.current_tenant_id', true), inet_server_addr(), inet_server_port()"
                )
                row = cur.fetchone()
                database = row[0]
                tenant_ctx = row[1]
                server_host = str(row[2]) if row[2] is not None else None
                server_port = int(row[3]) if row[3] is not None else None

        return {
            "database": database,
            "tenant_context": tenant_ctx,
            "server_host": server_host,
            "server_port": server_port,
        }
    except Exception as e:
        logger.error("/debug/dbinfo failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"dbinfo error: {str(e)}")


@app.get("/debug/search_sanity", response_model=None)
def debug_search_sanity(claims: JWTClaims | None = Depends(get_jwt_claims)):
    """Debug endpoint for retrieval sanity checks.

    Returns node counts, embedding coverage, and sample nodes to help
    diagnose empty search results or low citation rates.

    Security:
        - When JWT is enabled, require admin:refresh scope.
        - When JWT is disabled (dev mode), allow access.

    Returns:
        {
          "tenant_id": str,
          "total_nodes": int,
          "nodes_with_embeddings": int,
          "nodes_with_text_search": int,
          "embedding_coverage_pct": float,
          "text_search_coverage_pct": float,
          "sample_nodes_with_embedding": List[{id, classes, has_text}],
          "sample_nodes_without_embedding": List[{id, classes, has_text}]
        }
    """
    assert repo is not None, "GraphRepository not initialized"
    # Enforce admin scope when JWT is enabled
    if JWT_ENABLED:
        if not claims:
            raise HTTPException(status_code=401, detail="Authentication required")
        if "admin:refresh" not in (claims.scopes or []):
            raise HTTPException(
                status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
            )

    # Get tenant from claims or default
    tenant_id = claims.tenant_id if claims else "default"

    try:
        with repo._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Total visible nodes
                cur.execute("SELECT COUNT(*) FROM nodes")
                total_nodes = cur.fetchone()[0]

                # Nodes with embeddings
                cur.execute("SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL")
                nodes_with_embeddings = cur.fetchone()[0]

                # Nodes with text_search_vector
                cur.execute("SELECT COUNT(*) FROM nodes WHERE text_search_vector IS NOT NULL")
                nodes_with_text_search = cur.fetchone()[0]

                # Sample nodes WITH embeddings (up to 5)
                cur.execute("""
                    SELECT id, classes, (props->>'text' IS NOT NULL AND props->>'text' != '') as has_text
                    FROM nodes
                    WHERE embedding IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                sample_with_embedding = [
                    {"id": row[0], "classes": row[1], "has_text": row[2]} for row in cur.fetchall()
                ]

                # Sample nodes WITHOUT embeddings (up to 5)
                cur.execute("""
                    SELECT id, classes, (props->>'text' IS NOT NULL AND props->>'text' != '') as has_text
                    FROM nodes
                    WHERE embedding IS NULL
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                sample_without_embedding = [
                    {"id": row[0], "classes": row[1], "has_text": row[2]} for row in cur.fetchall()
                ]

        embedding_coverage = (
            (nodes_with_embeddings / total_nodes * 100.0) if total_nodes > 0 else 0.0
        )
        text_search_coverage = (
            (nodes_with_text_search / total_nodes * 100.0) if total_nodes > 0 else 0.0
        )

        return {
            "tenant_id": tenant_id,
            "total_nodes": total_nodes,
            "nodes_with_embeddings": nodes_with_embeddings,
            "nodes_with_text_search": nodes_with_text_search,
            "embedding_coverage_pct": round(embedding_coverage, 2),
            "text_search_coverage_pct": round(text_search_coverage, 2),
            "sample_nodes_with_embedding": sample_with_embedding,
            "sample_nodes_without_embedding": sample_without_embedding,
        }
    except Exception as e:
        logger.error("/debug/search_sanity failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"search_sanity error: {str(e)}")


@app.get("/debug/embed_info", response_model=None)
def debug_embed_info(claims: JWTClaims | None = Depends(get_jwt_claims)):
    """Debug endpoint to inspect embedding configuration and stored vectors.

    Security:
        - When JWT is enabled, require admin:refresh scope.
        - When JWT is disabled (dev mode), allow access.

    Returns:
        {
          "embedding_backend": str,
          "embedding_model": str,
          "counts": {"total_nodes": int, "with_embedding": int, "without_embedding": int},
          "vector_dimension": {"db_type": str | None, "db_dim": int | None, "sampled_dims": List[int]},
          "sample": {"n": int, "norm_min": float, "norm_max": float, "norm_mean": float, "example_ids": List[str]}
        }
    """
    assert repo is not None, "GraphRepository not initialized"
    # Enforce admin scope when JWT is enabled
    if JWT_ENABLED:
        if not claims:
            raise HTTPException(status_code=401, detail="Authentication required")
        if "admin:refresh" not in (claims.scopes or []):
            raise HTTPException(
                status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
            )

    try:
        tenant_id = claims.tenant_id if claims else "default"
        total_nodes = 0
        with_embedding = 0
        without_embedding = 0
        db_type: str | None = None
        db_dim: int | None = None
        sampled_dims: list[int] = []
        norms: list[float] = []
        example_ids: list[str] = []
        # last_refreshed stats
        lr_count = 0
        lr_age_min = None
        lr_age_avg = None
        lr_age_max = None

        with repo._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Counts
                cur.execute("SELECT COUNT(*) FROM nodes")
                total_nodes = int(cur.fetchone()[0])

                cur.execute("SELECT COUNT(*) FROM nodes WHERE embedding IS NOT NULL")
                with_embedding = int(cur.fetchone()[0])
                without_embedding = max(0, total_nodes - with_embedding)

                # Column type (e.g., vector(384)) from catalog
                try:
                    cur.execute(
                        """
                        SELECT format_type(a.atttypid, a.atttypmod)
                        FROM pg_attribute a
                        WHERE a.attrelid = 'public.nodes'::regclass AND a.attname = 'embedding'
                        """
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        db_type = str(row[0])
                        import re as _re

                        m = _re.search(r"vector\((\d+)\)", db_type)
                        if m:
                            db_dim = int(m.group(1))
                except Exception:
                    pass

                # Sample up to 100 embeddings for norm and dimension checks
                cur.execute("SELECT id, embedding FROM nodes WHERE embedding IS NOT NULL LIMIT 100")
                rows = cur.fetchall()
                for rid, remb in rows:
                    try:
                        vec = np.array(remb, dtype=np.float32)
                        if vec.size > 0:
                            nrm = float(np.linalg.norm(vec))
                            norms.append(nrm)
                            sampled_dims.append(int(vec.size))
                            if len(example_ids) < 5:
                                example_ids.append(str(rid))
                    except Exception:
                        continue

                # last_refreshed stats for nodes with embeddings
                try:
                    cur.execute(
                        """
                        SELECT
                          COUNT(last_refreshed) AS n,
                          MIN(EXTRACT(EPOCH FROM (now() - last_refreshed))) AS age_min,
                          AVG(EXTRACT(EPOCH FROM (now() - last_refreshed))) AS age_avg,
                          MAX(EXTRACT(EPOCH FROM (now() - last_refreshed))) AS age_max
                        FROM nodes
                        WHERE embedding IS NOT NULL AND last_refreshed IS NOT NULL
                        """
                    )
                    row = cur.fetchone()
                    if row:
                        lr_count = int(row[0] or 0)
                        lr_age_min = float(row[1]) if row[1] is not None else None
                        lr_age_avg = float(row[2]) if row[2] is not None else None
                        lr_age_max = float(row[3]) if row[3] is not None else None
                except Exception:
                    pass

        # Aggregate stats
        norm_min = float(min(norms)) if norms else None
        norm_max = float(max(norms)) if norms else None
        norm_mean = float(sum(norms) / len(norms)) if norms else None

        # Track embedding health metrics (if enabled)
        if METRICS_ENABLED:
            # Calculate coverage ratio
            coverage_ratio = float(with_embedding / total_nodes) if total_nodes > 0 else 0.0

            # Use max staleness if available, otherwise 0
            max_staleness_seconds = float(lr_age_max) if lr_age_max is not None else 0.0

            track_embedding_health(
                coverage_ratio=coverage_ratio,
                max_staleness_seconds=max_staleness_seconds,
                tenant_id=tenant_id,
            )

        return {
            "embedding_backend": EMBEDDING_BACKEND,
            "embedding_model": EMBEDDING_MODEL,
            "counts": {
                "total_nodes": total_nodes,
                "with_embedding": with_embedding,
                "without_embedding": without_embedding,
            },
            "vector_dimension": {
                "db_type": db_type,
                "db_dim": db_dim,
                "sampled_dims": sorted(set(sampled_dims)) if sampled_dims else [],
            },
            "sample": {
                "n": len(norms),
                "norm_min": round(norm_min, 6) if norm_min is not None else None,
                "norm_max": round(norm_max, 6) if norm_max is not None else None,
                "norm_mean": round(norm_mean, 6) if norm_mean is not None else None,
                "example_ids": example_ids,
            },
            "last_refreshed": {
                "count": lr_count,
                "age_seconds": {
                    "min": round(lr_age_min, 3) if lr_age_min is not None else None,
                    "avg": round(lr_age_avg, 3) if lr_age_avg is not None else None,
                    "max": round(lr_age_max, 3) if lr_age_max is not None else None,
                },
            },
        }

    except Exception as e:
        logger.error("/debug/embed_info failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"embed_info error: {str(e)}")


@app.get("/debug/intent", response_model=None)
def debug_intent(
    q: str,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Debug endpoint to test intent detection without running full /ask.

    Example: GET /debug/intent?q=What%20ML%20frameworks%20does%20the%20position%20require

    Returns: {
        "query": str,
        "normalized": str,
        "intent_type": str | None,
        "params": dict | None
    }
    """
    del _rl
    if JWT_ENABLED:
        if not claims:
            raise HTTPException(status_code=401, detail="Authentication required")
        if "admin:refresh" not in (claims.scopes or []):
            raise HTTPException(
                status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
            )

    try:
        import re

        # Normalize query
        question_normalized = q.lower()
        question_normalized = re.sub(r"\bml\b", "machine learning", question_normalized)
        question_normalized = re.sub(r"\s+", " ", question_normalized).strip()

        # Detect intent
        intent_type, params = detect_intent(q)

        return {
            "query": q,
            "normalized": question_normalized,
            "intent_type": intent_type,
            "params": params,
        }
    except Exception as e:
        logger.error(
            "/debug/intent failed",
            extra_fields={"error_type": type(e).__name__},
        )
        raise HTTPException(status_code=500, detail="intent detection failed") from e


def _metrics_unavailable(code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": {"code": code, "message": message}},
        headers={"Cache-Control": "no-store"},
    )


@app.get("/metrics", response_model=None)
def get_metrics(_control_plane: None = Depends(_require_control_plane)) -> Response:
    """Return a private, tenant-label-free, bounded JSON metrics snapshot."""

    del _control_plane
    if not METRICS_ENABLED:
        return _metrics_unavailable("METRICS_DISABLED", "Metrics are disabled.")
    try:
        payload = _metrics_boundary.json_bytes(metrics.get_all_metrics)
    except OperationalBusy:
        return _metrics_unavailable("METRICS_BUSY", "Metrics snapshot is busy.")
    except OperationalPayloadTooLarge:
        return _metrics_unavailable(
            "METRICS_RESPONSE_TOO_LARGE", "Metrics snapshot exceeds its response budget."
        )
    return Response(
        content=payload,
        status_code=200,
        media_type="application/json",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/prometheus")
def prometheus_metrics(_control_plane: None = Depends(_require_control_plane)) -> Response:
    """Return private, tenant-label-free, bounded Prometheus exposition."""

    del _control_plane
    if not METRICS_ENABLED:
        return _metrics_unavailable("METRICS_DISABLED", "Metrics are disabled.")
    try:
        payload = _metrics_boundary.prometheus_bytes(generate_latest)
    except OperationalBusy:
        return _metrics_unavailable("METRICS_BUSY", "Metrics snapshot is busy.")
    except OperationalPayloadTooLarge:
        return _metrics_unavailable(
            "METRICS_RESPONSE_TOO_LARGE", "Metrics snapshot exceeds its response budget."
        )
    return Response(
        content=payload,
        status_code=200,
        headers={"Cache-Control": "no-store", "Content-Type": CONTENT_TYPE_LATEST},
    )


def _background_embed(node_id: str, tenant_id: str | None = None):
    """Background task to embed a node and persist embedding/drift/history."""
    assert repo is not None, "GraphRepository not initialized"
    assert embedder is not None, "EmbeddingProvider not initialized"
    try:
        n = repo.get_node(node_id, tenant_id=tenant_id)
        if not n:
            return
        text = repo.build_embedding_text(n)
        if not text:
            return
        extraction_version = os.getenv("EXTRACTION_VERSION", "1.0.0")
        node_version = (n.props or {}).get("extraction_version")
        content_hash = None
        if (not (n.props or {}).get("content_hash")) or (node_version != extraction_version):
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        old = n.embedding
        new = embedder.encode([text])[0]
        if old is None:
            drift = 0.0
        else:
            denom = (float((old**2).sum()) ** 0.5) * (float((new**2).sum()) ** 0.5)
            drift = 0.0 if denom == 0 else 1.0 - float((old @ new) / denom)
        ts = datetime.now(timezone.utc).isoformat()
        repo.update_node_embedding(
            node_id,
            new,
            drift,
            ts,
            tenant_id=n.tenant_id,
            content_hash=content_hash,
            extraction_version=extraction_version,
        )
        repo.write_embedding_history(
            node_id, drift, embedding_ref=n.payload_ref, tenant_id=n.tenant_id
        )
        drift_threshold = n.refresh_policy.get("drift_threshold", 0.1) if n.refresh_policy else 0.1
        if drift > drift_threshold:
            repo.append_event(
                node_id,
                "refreshed",
                {"drift_score": drift, "last_refreshed": ts, "auto_embed": True},
                tenant_id=n.tenant_id,
                actor_id="auto_embed",
                actor_type="system",
            )
    except Exception as e:
        logger.error("Background embed failed", extra_fields={"node_id": node_id, "error": str(e)})
        try:
            repo.mark_embedding_failed(node_id, str(e), tenant_id=tenant_id)
        except Exception:
            pass


@app.post("/nodes", response_model=None, dependencies=[Depends(require_scope("kg:write"))])
def create_node(
    node: NodeCreate,
    background_tasks: BackgroundTasks,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Create a new node with validated input.

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims (secure).
        When JWT_ENABLED=false (dev mode), tenant_id can be provided in request body.

    Extraction:
        When EXTRACTION_ENABLED=true, structured field extraction is available.
        - extract_before_embed=true: Extract first, then embed (best quality)
        - extract_before_embed=false: Embed immediately, extract async (faster)
        - extract=false: Skip extraction for this request (even if enabled)
        - Default behavior controlled by EXTRACTION_MODE env var
    """
    assert repo is not None, "GraphRepository not initialized"
    # Extract tenant_id from JWT (secure) or request body (dev mode only)
    if JWT_ENABLED and claims:
        tenant_id = claims.tenant_id
    else:
        tenant_id = node.tenant_id or "default"

    redis_client = None
    if AUTO_EMBED_ON_CREATE and EMBEDDING_ASYNC:
        redis_client = _get_embedding_redis()
        if not redis_client:
            if EMBEDDING_QUEUE_REQUIRE_REDIS:
                raise HTTPException(
                    status_code=503,
                    detail="Embedding queue unavailable (Redis not configured)",
                )
        else:
            _check_embedding_queue_capacity(redis_client, tenant_id, requested=1)

    n = Node(
        classes=node.classes,
        props=node.props,
        payload_ref=node.payload_ref,
        metadata=node.metadata,
        refresh_policy=node.refresh_policy,
        triggers=node.triggers,
        tenant_id=tenant_id,  # From JWT in production
    )
    node_id = repo.create_node(n)

    # Determine extraction behavior
    extract_sync = False
    extraction_job_id = None
    extract_enabled = EXTRACTION_ENABLED and redis_client and (node.extract is None or node.extract)
    if extract_enabled:
        # Determine if we should extract before embedding
        if node.extract_before_embed is not None:
            extract_sync = node.extract_before_embed
        else:
            extract_sync = EXTRACTION_MODE == "sync"

        if extract_sync:
            # Sync mode: queue extraction first, worker will trigger embed after
            try:
                extraction_job_id = enqueue_extraction_job(
                    redis_client, node_id, tenant_id, priority="high"
                )
                # Mark extraction as queued
                _update_extraction_status(node_id, tenant_id, "queued")
                # Don't embed yet - extraction worker will trigger re-embed
                return {
                    "id": node_id,
                    "extraction_status": "queued",
                    "extraction_job_id": extraction_job_id,
                    "embedding_status": "pending_extraction",
                }
            except Exception as e:
                logger.error(
                    "Failed to enqueue extraction job",
                    extra_fields={"node_id": node_id, "error": str(e)},
                )
                # Fall through to normal embedding

    # Optionally auto-embed on create to make node searchable immediately
    response: dict[str, Any] = {"id": node_id}
    if AUTO_EMBED_ON_CREATE:
        if EMBEDDING_ASYNC and redis_client:
            try:
                job_id = enqueue_embedding_job(redis_client, node_id, n.tenant_id)
                repo.mark_embedding_queued(node_id, tenant_id=n.tenant_id)
                response["embedding_status"] = "queued"
                response["job_id"] = job_id
            except Exception as e:
                logger.error(
                    "Failed to enqueue embedding job",
                    extra_fields={"node_id": node_id, "error": str(e)},
                )
                if EMBEDDING_QUEUE_REQUIRE_REDIS:
                    raise HTTPException(
                        status_code=503,
                        detail="Embedding queue unavailable",
                    )
        else:
            try:
                background_tasks.add_task(_background_embed, node_id, n.tenant_id)
            except Exception as e:
                logger.error(
                    "Failed to schedule background embed",
                    extra_fields={"node_id": node_id, "error": str(e)},
                )
    else:
        try:
            repo.mark_embedding_skipped(node_id, "auto_embed_disabled", tenant_id=n.tenant_id)
        except Exception:
            pass

    # Queue async extraction after embedding (if enabled and not sync mode)
    if extract_enabled and not extract_sync:
        try:
            extraction_job_id = enqueue_extraction_job(
                redis_client, node_id, tenant_id, priority="normal"
            )
            _update_extraction_status(node_id, tenant_id, "queued")
            response["extraction_status"] = "queued"
            response["extraction_job_id"] = extraction_job_id
        except Exception as e:
            logger.warning(
                "Failed to enqueue extraction job (non-blocking)",
                extra_fields={"node_id": node_id, "error": str(e)},
            )

    return response


def _update_extraction_status(node_id: str, tenant_id: str | None, status: str) -> None:
    """Update extraction status in node props."""
    assert repo is not None
    with repo._conn(tenant_id=tenant_id) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE nodes
                SET props = COALESCE(props, '{}'::jsonb) || %s::jsonb,
                    updated_at = now()
                WHERE id = %s
                """,
                (json.dumps({"extraction_status": status}), node_id),
            )


@app.post("/nodes/batch", response_model=None, dependencies=[Depends(require_scope("kg:write"))])
def create_nodes_batch(
    batch: NodeBatchCreate,
    background_tasks: BackgroundTasks,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Create multiple nodes in a single request.

    Extraction:
        When EXTRACTION_ENABLED=true, structured field extraction is available.
        - batch.extract_before_embed=true: Extract first for all nodes (best quality)
        - batch.extract_before_embed=false: Embed immediately, extract async (faster)
        - batch.extract=false: Skip extraction for all nodes in this batch
        - Default behavior controlled by EXTRACTION_MODE env var
    """
    assert repo is not None, "GraphRepository not initialized"

    if not batch.nodes:
        raise HTTPException(status_code=400, detail="nodes list is required")
    if len(batch.nodes) > NODE_BATCH_MAX:
        raise HTTPException(
            status_code=400,
            detail=f"Batch too large (max {NODE_BATCH_MAX})",
        )

    if JWT_ENABLED and claims:
        effective_tenant_id = claims.tenant_id
    else:
        effective_tenant_id = batch.tenant_id or "default"

    redis_client = None
    if AUTO_EMBED_ON_CREATE and EMBEDDING_ASYNC:
        redis_client = _get_embedding_redis()
        if not redis_client:
            if EMBEDDING_QUEUE_REQUIRE_REDIS:
                raise HTTPException(
                    status_code=503,
                    detail="Embedding queue unavailable (Redis not configured)",
                )
        else:
            _check_embedding_queue_capacity(
                redis_client, effective_tenant_id, requested=len(batch.nodes)
            )

    # Determine batch-level extraction behavior
    batch_extract_sync = False
    batch_extract_enabled = (
        EXTRACTION_ENABLED and redis_client and (batch.extract is None or batch.extract)
    )
    if batch_extract_enabled:
        if batch.extract_before_embed is not None:
            batch_extract_sync = batch.extract_before_embed
        else:
            batch_extract_sync = EXTRACTION_MODE == "sync"

    results: list[dict[str, Any]] = []
    created = 0
    failed = 0

    for item in batch.nodes:
        tenant_id = effective_tenant_id
        if not JWT_ENABLED and not batch.tenant_id:
            tenant_id = item.tenant_id or "default"

        try:
            n = Node(
                classes=item.classes,
                props=item.props,
                payload_ref=item.payload_ref,
                metadata=item.metadata,
                refresh_policy=item.refresh_policy,
                triggers=item.triggers,
                tenant_id=tenant_id,
            )
            node_id = repo.create_node(n)
            created += 1

            result_item: dict[str, Any] = {"id": node_id, "tenant_id": tenant_id}

            # Determine extraction mode for this item
            item_extract_enabled = batch_extract_enabled
            if item.extract is not None:
                item_extract_enabled = EXTRACTION_ENABLED and redis_client and item.extract
            item_extract_sync = batch_extract_sync
            if item.extract_before_embed is not None:
                item_extract_sync = item.extract_before_embed

            # Handle sync extraction mode (extract first, then embed)
            if item_extract_enabled and item_extract_sync:
                try:
                    extraction_job_id = enqueue_extraction_job(
                        redis_client, node_id, tenant_id, priority="high"
                    )
                    _update_extraction_status(node_id, tenant_id, "queued")
                    result_item["extraction_status"] = "queued"
                    result_item["extraction_job_id"] = extraction_job_id
                    result_item["embedding_status"] = "pending_extraction"
                    results.append(result_item)
                    continue  # Skip embedding - worker will trigger it
                except Exception as e:
                    logger.warning(
                        "Failed to enqueue extraction, falling back to embed",
                        extra_fields={"node_id": node_id, "error": str(e)},
                    )

            # Normal embedding flow
            embedding_status = None
            job_id = None
            if AUTO_EMBED_ON_CREATE:
                if EMBEDDING_ASYNC and redis_client:
                    job_id = enqueue_embedding_job(redis_client, node_id, tenant_id)
                    repo.mark_embedding_queued(node_id, tenant_id=tenant_id)
                    embedding_status = "queued"
                else:
                    background_tasks.add_task(_background_embed, node_id, tenant_id)
            else:
                repo.mark_embedding_skipped(node_id, "auto_embed_disabled", tenant_id=tenant_id)
                embedding_status = "skipped"

            result_item["embedding_status"] = embedding_status
            result_item["job_id"] = job_id

            # Queue async extraction (if enabled and not sync mode)
            if item_extract_enabled and not item_extract_sync:
                try:
                    extraction_job_id = enqueue_extraction_job(
                        redis_client, node_id, tenant_id, priority="normal"
                    )
                    _update_extraction_status(node_id, tenant_id, "queued")
                    result_item["extraction_status"] = "queued"
                    result_item["extraction_job_id"] = extraction_job_id
                except Exception:
                    pass  # Non-blocking

            results.append(result_item)
        except Exception as e:
            failed += 1
            results.append({"error": str(e), "tenant_id": tenant_id})
            if not batch.continue_on_error:
                break

    return {"created": created, "failed": failed, "results": results}


@app.get("/nodes", response_model=None)
def list_nodes(
    limit: int = 100,
    offset: int = 0,
    has_embedding: bool | None = None,
    tenant_id: str | None = None,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """List all nodes with optional filtering by embedding status.

    Args:
        limit: Maximum number of nodes to return (default 100)
        offset: Number of nodes to skip for pagination (default 0)
        has_embedding: Filter by embedding status (None=all, True=with embedding, False=without)
        tenant_id: Tenant ID (ignored when JWT_ENABLED)

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims (secure).
        Query param tenant_id is IGNORED in production to prevent RLS bypass.

    Returns:
        {
            "nodes": [{"id": str, "classes": List[str], "has_embedding": bool}, ...],
            "total": int,
            "limit": int,
            "offset": int
        }
    """
    assert repo is not None, "GraphRepository not initialized"

    # CRITICAL: Use JWT tenant_id in production, ignore query param
    if JWT_ENABLED and claims:
        effective_tenant_id = claims.tenant_id
    else:
        effective_tenant_id = tenant_id if tenant_id else "default"  # Dev mode only

    nodes_list = []
    total = 0

    with repo._conn(tenant_id=effective_tenant_id) as conn:
        with conn.cursor() as cur:
            # Build query based on filter
            where_parts: list[str] = []
            params: list[Any] = []
            if not JWT_ENABLED and tenant_id:
                logger.info(
                    "Dev tenant filter applied for /nodes",
                    extra_fields={"tenant_id": tenant_id},
                )
                where_parts.append("tenant_id = %s")
                params.append(tenant_id)
            if has_embedding is True:
                where_parts.append("embedding IS NOT NULL")
            elif has_embedding is False:
                where_parts.append("embedding IS NULL")
            where_clause = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

            # Get total count
            count_query = f"SELECT COUNT(*) FROM nodes {where_clause}"
            cur.execute(count_query, params)
            total = int(cur.fetchone()[0])

            # Get nodes with pagination
            query = f"""
                SELECT id, classes, embedding IS NOT NULL as has_embedding,
                       embedding_status, embedding_error, embedding_attempts
                FROM nodes
                {where_clause}
                ORDER BY id
                LIMIT %s OFFSET %s
            """
            cur.execute(query, (*params, limit, offset))
            rows = cur.fetchall()

            for row in rows:
                nodes_list.append(
                    {
                        "id": str(row[0]),
                        "classes": row[1] if row[1] else [],
                        "has_embedding": bool(row[2]),
                        "embedding_status": row[3],
                        "embedding_error": row[4],
                        "embedding_attempts": row[5],
                    }
                )

    return {"nodes": nodes_list, "total": total, "limit": limit, "offset": offset}


@app.get("/nodes/by-external-id", response_model=None)
def get_node_by_external_id(
    external_id: str,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Lookup a node by its external_id prop (tenant-scoped)."""
    assert repo is not None, "GraphRepository not initialized"
    if JWT_ENABLED and claims:
        tenant_id = claims.tenant_id
    else:
        tenant_id = None  # Dev mode - RLS will handle isolation if enabled

    node = repo.get_node_by_external_id(external_id, tenant_id=tenant_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    return {
        "id": node.id,
        "tenant_id": node.tenant_id,
        "classes": node.classes,
        "props": node.props,
        "metadata": node.metadata,
        "payload_ref": node.payload_ref,
    }


@app.get("/nodes/{node_id}", response_model=None)
def get_node(
    node_id: str,
    tenant_id: str | None = None,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Get a node by ID.

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims (secure).
        Query param tenant_id is IGNORED in production to prevent RLS bypass.
    """
    assert repo is not None, "GraphRepository not initialized"
    # CRITICAL: Use JWT tenant_id in production, ignore query param
    if JWT_ENABLED and claims:
        effective_tenant_id = claims.tenant_id
    else:
        effective_tenant_id = tenant_id if tenant_id else "default"  # Dev mode only

    n = repo.get_node(node_id, tenant_id=effective_tenant_id)
    if not n:
        raise HTTPException(status_code=404, detail="Node not found")
    if not JWT_ENABLED and tenant_id and n.tenant_id != tenant_id:
        logger.info(
            "Dev tenant filter mismatch for /nodes/{id}",
            extra_fields={"node_id": node_id, "tenant_id": tenant_id},
        )
        raise HTTPException(status_code=404, detail="Node not found")
    return {
        "id": n.id,
        "classes": n.classes,
        "props": n.props,
        "payload_ref": n.payload_ref,
        "metadata": n.metadata,
        "refresh_policy": n.refresh_policy,
        "triggers": n.triggers,
        "version": n.version,
        "embedding_status": n.embedding_status,
        "embedding_error": n.embedding_error,
        "embedding_attempts": n.embedding_attempts,
        "embedding_updated_at": n.embedding_updated_at.isoformat()
        if n.embedding_updated_at
        else None,
    }


@app.post("/nodes/{node_id}/refresh", response_model=None)
def refresh_node(
    node_id: str,
    tenant_id: str | None = None,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Manually refresh a single node's embedding and write history/events.

    Computes drift vs prior embedding; emits a refreshed event if drift exceeds threshold.

    Security:
        Requires JWT authentication when JWT_ENABLED=true.
        Tenant ID derived from JWT claims to prevent cross-tenant refresh.
    """
    assert repo is not None, "GraphRepository not initialized"
    assert embedder is not None, "EmbeddingProvider not initialized"
    # Require authentication for node refresh (modifies embeddings)
    if JWT_ENABLED and not claims:
        raise HTTPException(status_code=401, detail="Authentication required")

    # CRITICAL: Use JWT tenant_id in production, ignore query param
    if JWT_ENABLED and claims:
        effective_tenant_id = claims.tenant_id
        actor_id = claims.actor_id
    else:
        effective_tenant_id = tenant_id if tenant_id else "default"  # Dev mode only
        actor_id = "dev_user"

    try:
        n = repo.get_node(node_id, tenant_id=effective_tenant_id)
        if not n:
            raise HTTPException(status_code=404, detail="Node not found")

        if EMBEDDING_ASYNC:
            redis_client = _get_embedding_redis()
            if not redis_client and EMBEDDING_QUEUE_REQUIRE_REDIS:
                raise HTTPException(
                    status_code=503,
                    detail="Embedding queue unavailable (Redis not configured)",
                )
            if redis_client:
                _check_embedding_queue_capacity(redis_client, n.tenant_id, requested=1)
                job_id = enqueue_embedding_job(
                    redis_client, node_id, n.tenant_id, action="refresh", force=True
                )
                repo.mark_embedding_queued(node_id, tenant_id=n.tenant_id)
                return {"id": node_id, "status": "queued", "job_id": job_id}

        text = repo.load_payload_text(n)
        old = n.embedding
        new = embedder.encode([text])[0]
        denom = (
            (float((old**2).sum()) ** 0.5) * (float((new**2).sum()) ** 0.5)
            if old is not None
            else 0.0
        )
        drift = 0.0 if old is None or denom == 0 else 1.0 - float((old @ new) / denom)
        ts = datetime.now(timezone.utc).isoformat()
        repo.update_node_embedding(node_id, new, drift, ts, tenant_id=n.tenant_id)
        repo.write_embedding_history(
            node_id, drift, embedding_ref=n.payload_ref, tenant_id=n.tenant_id
        )

        # Emit event if drift exceeds threshold
        drift_threshold = n.refresh_policy.get("drift_threshold", 0.1) if n.refresh_policy else 0.1
        event_id = None
        if drift > drift_threshold:
            event_id = repo.append_event(
                node_id,
                "refreshed",
                {"drift_score": drift, "last_refreshed": ts, "manual_trigger": True},
                tenant_id=n.tenant_id,
                actor_id=actor_id,
                actor_type="user",
            )

        return {"id": node_id, "drift_score": drift, "last_refreshed": ts, "event_id": event_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Node refresh failed", extra_fields={"node_id": node_id, "error": str(e)})
        raise HTTPException(status_code=500, detail=f"Node refresh failed: {str(e)}")


UPLOAD_MAX_FILES = int(os.getenv("UPLOAD_MAX_FILES", "50"))

# MIME type mapping for common file extensions
_EXT_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".html": "text/html",
    ".htm": "text/html",
    ".txt": "text/plain",
}


@app.post("/upload", response_model=None, dependencies=[Depends(require_scope("kg:write"))])
async def upload_files(
    files: list[UploadFile] = File(...),
    tenant_id: str | None = Form(None),
    classes: str = Form("Document,Resume"),
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Upload PDF/DOCX files, extract text, chunk, and enqueue embeddings.

    Accepts multipart/form-data with one or more files. Each file is
    extracted, chunked via ``create_chunk_nodes``, and embedding jobs are
    enqueued for each chunk.

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims.
        The ``tenant_id`` form field is only used in dev mode.
    """
    from hashlib import sha256

    from activekg.connectors.chunker import create_chunk_nodes
    from activekg.connectors.extract import extract_text

    assert repo is not None, "GraphRepository not initialized"

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    if len(files) > UPLOAD_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files (max {UPLOAD_MAX_FILES})",
        )

    # Resolve tenant
    if JWT_ENABLED and claims:
        effective_tenant = claims.tenant_id
    else:
        effective_tenant = tenant_id or "default"

    # Parse classes
    class_list = [c.strip() for c in classes.split(",") if c.strip()]

    # Prepare embedding queue
    redis_client = None
    if AUTO_EMBED_ON_CREATE and EMBEDDING_ASYNC:
        redis_client = _get_embedding_redis()
        if not redis_client and EMBEDDING_QUEUE_REQUIRE_REDIS:
            raise HTTPException(
                status_code=503,
                detail="Embedding queue unavailable (Redis not configured)",
            )

    uploaded = 0
    skipped = 0
    total_chunks = 0
    total_embeddings = 0
    file_results: list[dict[str, Any]] = []

    for f in files:
        fname = f.filename or "unknown"
        try:
            data = await f.read()
            if not data:
                file_results.append({"filename": fname, "chunks": 0, "status": "skipped"})
                skipped += 1
                continue

            # Determine content type from header or extension
            ct = f.content_type or ""
            if not ct or ct == "application/octet-stream":
                ext = os.path.splitext(fname)[1].lower()
                ct = _EXT_MIME.get(ext, "application/octet-stream")

            text = extract_text(data, ct)
            if not text or not text.strip():
                file_results.append({"filename": fname, "chunks": 0, "status": "skipped"})
                skipped += 1
                continue

            content_hash = sha256(text.encode()).hexdigest()[:16]
            external_id = f"upload:{effective_tenant}:{fname}:{content_hash}"

            chunk_ids = create_chunk_nodes(
                parent_node_id=external_id,
                parent_title=fname,
                parent_classes=class_list,
                text=text,
                parent_metadata={
                    "source": "manual_upload",
                    "content_type": ct,
                    "size": len(data),
                    "content_hash": content_hash,
                },
                repo=repo,
                tenant_id=effective_tenant,
            )

            # Enqueue embeddings
            enqueued = 0
            if redis_client:
                if chunk_ids:
                    _check_embedding_queue_capacity(
                        redis_client, effective_tenant, requested=len(chunk_ids)
                    )
                for cid in chunk_ids:
                    try:
                        job_id = enqueue_embedding_job(redis_client, cid, effective_tenant)
                        if job_id:
                            repo.mark_embedding_queued(cid, tenant_id=effective_tenant)
                            enqueued += 1
                    except Exception as e:
                        logger.warning(
                            "Failed to enqueue embedding for chunk",
                            extra_fields={"chunk_id": cid, "error": str(e)},
                        )

            uploaded += 1
            total_chunks += len(chunk_ids)
            total_embeddings += enqueued
            file_results.append({"filename": fname, "chunks": len(chunk_ids), "status": "ok"})
        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                "File upload processing failed",
                extra_fields={"filename": fname, "error": str(e)},
            )
            file_results.append({"filename": fname, "chunks": 0, "status": f"error: {e}"})
            skipped += 1

    return {
        "uploaded": uploaded,
        "skipped": skipped,
        "chunks_created": total_chunks,
        "embeddings_queued": total_embeddings,
        "files": file_results,
    }


@app.post("/search", response_model=None, dependencies=[Depends(require_scope("search:read"))])
def search_nodes(
    http_request: Request,
    http_response: Response,
    search_request: KGSearchRequest,
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Semantic search across knowledge graph nodes using pgvector.

    Embeds the query text and returns top-K similar nodes with similarity scores.

    Supports two search modes:
    1. Vector-only (default): Pure semantic similarity using embeddings
    2. Hybrid (use_hybrid=True): BM25 + vector fusion with optional cross-encoder reranking

    Weighted scoring (when use_weighted_score=True):
    - Applies age decay: fresher nodes score higher
    - Applies drift penalty: lower drift scores higher
    - Formula: similarity * exp(-decay_lambda * age_days) * (1 - drift_beta * drift_score)

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims (secure).
        Query param tenant_id is IGNORED in production to prevent RLS bypass.
    """
    assert repo is not None, "GraphRepository not initialized"
    assert embedder is not None, "EmbeddingProvider not initialized"
    try:
        # Start timing for latency tracking
        start_time = time.time()

        # CRITICAL: Use JWT tenant_id in production, ignore query param
        if JWT_ENABLED and claims:
            effective_tenant_id = claims.tenant_id
        else:
            effective_tenant_id = (
                search_request.tenant_id if search_request.tenant_id else "default"
            )  # Dev mode only

        # Apply rate limiting with headers
        if RATE_LIMIT_ENABLED:
            identifier = get_identifier(http_request, effective_tenant_id)
            limit_info = rate_limiter.check_limit(identifier, endpoint="search")
            http_response.headers["X-RateLimit-Limit"] = str(limit_info.limit)
            http_response.headers["X-RateLimit-Remaining"] = str(limit_info.remaining)
            http_response.headers["X-RateLimit-Reset"] = str(limit_info.reset_at)
            if not limit_info.allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(limit_info.retry_after or 1)},
                )

        # Embed the query
        query_embedding = embedder.encode([search_request.query])[0]

        # Execute search (hybrid or vector-only)
        if search_request.use_hybrid:
            results = repo.hybrid_search(
                query_text=search_request.query,
                query_embedding=query_embedding,
                top_k=search_request.top_k,
                metadata_filters=search_request.metadata_filters,
                compound_filter=search_request.compound_filter,
                tenant_id=effective_tenant_id,
                use_reranker=search_request.use_reranker,
            )
            # Fallback: if hybrid returns 0 results (e.g., text_search_vector missing), try vector-only
            if not results:
                try:
                    results = repo.vector_search(
                        query_embedding=query_embedding,
                        top_k=search_request.top_k,
                        metadata_filters=search_request.metadata_filters,
                        compound_filter=search_request.compound_filter,
                        tenant_id=effective_tenant_id,
                        use_weighted_score=search_request.use_weighted_score,
                        decay_lambda=search_request.decay_lambda,
                        drift_beta=search_request.drift_beta,
                    )
                except Exception:
                    results = []
        else:
            results = repo.vector_search(
                query_embedding=query_embedding,
                top_k=search_request.top_k,
                metadata_filters=search_request.metadata_filters,
                compound_filter=search_request.compound_filter,
                tenant_id=effective_tenant_id,
                use_weighted_score=search_request.use_weighted_score,
                decay_lambda=search_request.decay_lambda,
                drift_beta=search_request.drift_beta,
            )

        # Format response (keep "similarity" key for backward compatibility)
        # Also include a non-null text snippet to avoid clients receiving null
        formatted_results: list[dict[str, Any]] = []
        for node, similarity in results:
            raw_text = None
            try:
                raw_text = (node.props or {}).get("text")
            except Exception:
                raw_text = None

            vector_similarity = None
            try:
                if getattr(node, "embedding", None) is not None:
                    vector_similarity = float(node.embedding @ query_embedding)
                    if vector_similarity < 0.0:
                        vector_similarity = 0.0
                    elif vector_similarity > 1.0:
                        vector_similarity = 1.0
            except Exception:
                vector_similarity = None

            text_snippet = raw_text or ""
            if isinstance(text_snippet, str) and len(text_snippet) > 300:
                text_snippet = text_snippet[:300]
            item: dict[str, Any] = {
                "id": node.id,
                "classes": node.classes,
                "props": node.props,
                "payload_ref": node.payload_ref,
                "metadata": node.metadata,
                "similarity": round(similarity, 4),
                "text": text_snippet,
            }
            if vector_similarity is not None:
                item["vector_similarity"] = round(vector_similarity, 4)
            formatted_results.append(item)

        if search_request.use_hybrid:
            mode = "hybrid"
            rrf_enabled = os.getenv("HYBRID_RRF_ENABLED", "true").lower() == "true"
            score_type = "rrf_fused" if rrf_enabled else "weighted_fusion"
            reranked = search_request.use_reranker
        else:
            mode = "vector"
            score_type = "weighted_fusion" if search_request.use_weighted_score else "cosine"
            reranked = False

        # Track Prometheus metrics (if enabled)
        if METRICS_ENABLED:
            latency_ms = (time.time() - start_time) * 1000

            track_search_request(
                mode=mode,
                score_type=score_type,
                latency_ms=latency_ms,
                result_count=len(formatted_results),
                reranked=reranked,
            )

        return {
            "query": search_request.query,
            "results": formatted_results,
            "count": len(formatted_results),
            "search_mode": mode,
            "score_type": score_type,
            "mode": mode,
        }
    except Exception as e:
        logger.error("Search failed", extra_fields={"error": str(e), "query": search_request.query})
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


def detect_intent(question: str) -> tuple[str | None, dict | None]:
    """Detect structured query intents that need specialized retrieval.

    Args:
        question: User question

    Returns:
        Tuple of (intent_type, intent_params) where:
        - intent_type: "open_positions", "performance_issues", "entity_job", "entity_resume", "entity_article", or None
        - intent_params: Dict with extracted parameters (role_terms, expected_classes, must_have_terms, etc.)
    """
    import re

    # Normalize query before processing
    question_normalized = question.lower()
    # Normalize "ML" → "machine learning" for better pattern matching
    question_normalized = re.sub(r"\bml\b", "machine learning", question_normalized)
    # Collapse multiple spaces
    question_normalized = re.sub(r"\s+", " ", question_normalized).strip()
    q_lower = question_normalized

    # DEBUG: Log normalized question
    import logging

    logger = logging.getLogger("activekg.api.main")
    logger.info(f"detect_intent: original='{question}' normalized='{q_lower}'")

    # Intent: Open positions
    # Patterns: "open ML engineer positions", "positions are open", "what positions are available", "hiring for..."
    # Match if query contains position/role/job/engineer keywords AND open/available/hiring keywords (in any order)
    has_position_keyword = bool(
        re.search(r"\b(position|role|job|engineer|developer|scientist)\b", q_lower)
    )
    has_open_keyword = bool(
        re.search(r"\b(open|available|hiring|recruiting|looking for)\b", q_lower)
    )
    logger.info(f"detect_intent: has_position={has_position_keyword}, has_open={has_open_keyword}")

    if has_position_keyword and has_open_keyword:
        # Extract role terms
        role_terms = []
        role_keywords = {
            "ml": ["ml", "machine learning"],
            "data": ["data scientist", "data engineer", "data analyst"],
            "software": ["software engineer", "developer", "backend", "frontend"],
            "devops": ["devops", "sre", "site reliability"],
        }
        for _key, terms in role_keywords.items():
            if any(term in q_lower for term in terms):
                role_terms.extend(terms)

        return ("open_positions", {"role_terms": role_terms if role_terms else None})

    # Intent: Performance issues
    # Patterns: "performance issues", "slow queries", "latency problems", "reported issues", etc.
    # Simplified pattern to match word stems (issues, reported, problems, etc.)
    perf_issue_pattern = (
        r"(performance|slow|latency|timeout|bottleneck).*(issue|problem|bug|incident|report)"
    )
    if re.search(perf_issue_pattern, q_lower):
        return ("performance_issues", {"lookback_days": 30})

    # TRACK 1: Entity-typed queries with class filtering
    # Detect queries asking about specific entity types (jobs, resumes, articles)

    # Entity: Job posting queries
    # Patterns: "What position...", "ML engineer job", "job about...", "position requires..."
    job_patterns = [
        r"\b(position|job|role)\b.*\b(require|need|about|available|open|description)\b",
        r"\b(machine learning|data scientist|sre|ux|developer)\b.*\b(position|job|role)\b",
        r"what.*\b(position|job|role)\b",
        r"\b(frameworks?|stack|libraries|requirements?|skills?)\b.*\b(position|job|role|engineer)\b",
        r"\b(job|role|position)\b.*(about|description)",
    ]
    job_match = any(re.search(pat, q_lower) for pat in job_patterns)
    logger.info(f"detect_intent: job_patterns_match={job_match}")
    if job_match:
        # Extract role-specific terms as must-haves
        must_have_terms = []
        role_indicators = {
            "machine learning engineer": ["machine learning engineer", "ml engineer"],
            "data scientist": ["data scientist"],
            "site reliability engineer": ["site reliability engineer", "sre"],
            "ux designer": ["ux designer", "ux"],
            "python developer": ["python developer", "python"],
        }
        for _role, terms in role_indicators.items():
            if any(term in q_lower for term in terms):
                must_have_terms.extend(terms)

        result = (
            "entity_job",
            {
                "expected_classes": ["Job"],
                "must_have_terms": must_have_terms if must_have_terms else None,
            },
        )
        logger.info(f"detect_intent: returning intent={result[0]}, params={result[1]}")
        return result

    # Entity: Resume/experience queries
    # Patterns: "What experience does...", "Who has...", "data scientist experience", "resume about..."
    resume_patterns = [
        r"\b(experience|resume|candidate|engineer|scientist)\b.*\b(has|have|with|know)\b",
        r"who\s+(has|have)\b",
        r"what\s+.*\b(experience|skills|knowledge)\b",
    ]
    if any(re.search(pat, q_lower) for pat in resume_patterns):
        # Extract entity-specific terms
        must_have_terms = []
        entity_indicators = {
            "data scientist": ["data scientist"],
            "machine learning": ["machine learning", "ml"],
            "python": ["python"],
            "site reliability": ["site reliability", "sre"],
        }
        for _entity, terms in entity_indicators.items():
            if any(term in q_lower for term in terms):
                must_have_terms.extend(terms)

        return (
            "entity_resume",
            {
                "expected_classes": ["Resume"],
                "must_have_terms": must_have_terms if must_have_terms else None,
            },
        )

    # Entity: Article/knowledge queries
    # Patterns: "What are the...", "kubernetes article", "autoscaling tools", "article about..."
    article_patterns = [
        r"\b(article|paper|guide|documentation|tutorial)\b",
        r"what\s+are\s+the\s+\b(tools|patterns|approaches|best practices)\b",
        r"\b(kubernetes|docker|monitoring|autoscaling)\b.*\b(tools|patterns|mentioned)\b",
    ]
    if any(re.search(pat, q_lower) for pat in article_patterns):
        # Extract topic-specific terms
        must_have_terms = []
        topic_indicators = {
            "kubernetes": ["kubernetes", "k8s"],
            "autoscaling": ["autoscaling", "autoscaler"],
            "monitoring": ["monitoring", "prometheus", "grafana"],
        }
        for _topic, terms in topic_indicators.items():
            if any(term in q_lower for term in terms):
                must_have_terms.extend(terms)

        return (
            "entity_article",
            {
                "expected_classes": ["Article"],
                "must_have_terms": must_have_terms if must_have_terms else None,
            },
        )

    return (None, None)


@app.post("/edges", response_model=None, dependencies=[Depends(require_scope("kg:write"))])
def create_edge(
    edge: EdgeCreate,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Create a relationship between two nodes with validated input.

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims (secure).
        When JWT_ENABLED=false (dev mode), tenant_id can be provided in request body.
    """
    assert repo is not None, "GraphRepository not initialized"
    assert embedder is not None, "EmbeddingProvider not initialized"
    # Extract tenant_id from JWT (secure) or request body (dev mode only)
    if JWT_ENABLED and claims:
        tenant_id = claims.tenant_id
    else:
        tenant_id = edge.tenant_id if edge.tenant_id else "default"

    try:
        e = Edge(
            src=edge.src,
            rel=edge.rel,
            dst=edge.dst,
            props=edge.props,
            tenant_id=tenant_id,  # From JWT in production
        )
        repo.create_edge(e)
        return {"status": "created", "src": e.src, "rel": e.rel, "dst": e.dst}
    except Exception as ex:
        logger.error("Edge creation failed", extra_fields={"error": str(ex)})
        raise HTTPException(status_code=500, detail=f"Edge creation failed: {str(ex)}")


@app.get("/events", response_model=None)
def list_events(
    node_id: str | None = None,
    event_type: str | None = None,
    tenant_id: str | None = None,
    limit: int = 100,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """List events with optional filtering by node_id, event_type, and tenant.

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims (secure).
        Query param tenant_id is IGNORED in production to prevent RLS bypass.
    """
    assert repo is not None, "GraphRepository not initialized"
    # CRITICAL: Use JWT tenant_id in production, ignore query param
    if JWT_ENABLED and claims:
        effective_tenant_id = claims.tenant_id
    else:
        effective_tenant_id = tenant_id if tenant_id else "default"  # Dev mode only

    try:
        # Use repo connection for RLS support
        with repo._conn(tenant_id=effective_tenant_id) as conn:
            with conn.cursor() as cur:
                query = "SELECT id, node_id, type, payload, created_at FROM events WHERE 1=1"
                params = []

                if node_id:
                    query += " AND node_id = %s"
                    params.append(node_id)

                if event_type:
                    query += " AND type = %s"
                    params.append(event_type)

                query += " ORDER BY created_at DESC LIMIT %s"
                params.append(str(min(limit, 1000)))  # Cap at 1000

                cur.execute(query, params)

                events = []
                for row in cur.fetchall():
                    events.append(
                        {
                            "id": str(row[0]),
                            "node_id": str(row[1]) if row[1] else None,
                            "type": row[2],
                            "payload": row[3],
                            "created_at": row[4].isoformat() if row[4] else None,
                        }
                    )

                return {"events": events, "count": len(events)}
    except Exception as e:
        logger.error("Event listing failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Event listing failed: {str(e)}")


@app.get("/lineage/{node_id}", response_model=None)
def get_lineage(
    node_id: str,
    max_depth: int = 5,
    tenant_id: str | None = None,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Traverse DERIVED_FROM edges to retrieve provenance lineage.

    Returns recursive ancestor chain with depth and edge metadata.

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims (secure).
        Query param tenant_id is IGNORED in production to prevent RLS bypass.
    """
    assert repo is not None, "GraphRepository not initialized"
    # CRITICAL: Use JWT tenant_id in production, ignore query param
    if JWT_ENABLED and claims:
        effective_tenant_id = claims.tenant_id
    else:
        effective_tenant_id = tenant_id if tenant_id else "default"  # Dev mode only

    try:
        lineage = repo.get_lineage(node_id, max_depth, tenant_id=effective_tenant_id)
        return {
            "node_id": node_id,
            "ancestors": lineage,
            "depth": len(lineage),
        }
    except Exception as e:
        logger.error("Lineage retrieval failed", extra_fields={"error": str(e), "node_id": node_id})
        raise HTTPException(status_code=500, detail=f"Lineage retrieval failed: {str(e)}")


@app.post("/admin/refresh")
async def admin_refresh(
    http_request: Request,
    http_response: Response,
    payload: Any | None = Body(
        default=None, description='Either ["id", ...] or {"node_ids": ["id", ...]}'
    ),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Trigger on-demand refresh cycle.

    Requires 'admin:refresh' scope when JWT authentication is enabled.

    Args:
        node_ids: Optional list of specific node IDs to refresh. If not provided, refreshes all due nodes.

    Returns:
        Summary of refresh operation
    """
    assert repo is not None, "GraphRepository not initialized"
    assert embedder is not None, "EmbeddingProvider not initialized"
    # Require admin:refresh scope when JWT is enabled
    if JWT_ENABLED and claims and "admin:refresh" not in claims.scopes:
        raise HTTPException(
            status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
        )

    # Extract tenant context
    tenant_id, actor_id, actor_type = get_tenant_context(
        cast(Request, http_request), claims, allow_override=not JWT_ENABLED
    )

    # Apply rate limiting (lighter limits for admin endpoints)
    if http_request and http_response:
        await apply_rate_limit(
            http_request,
            http_response,
            endpoint="admin_refresh",
            tenant_id=tenant_id,
            check_concurrency=False,  # No concurrency cap for admin
        )

    # Accept both raw array and wrapped object inputs
    node_ids: list[str] | None = None
    try:
        if isinstance(payload, list):
            node_ids = payload
        elif isinstance(payload, dict):
            node_ids = payload.get("node_ids")
    except Exception:
        node_ids = None

    try:
        from activekg.refresh.scheduler import RefreshScheduler

        # Create scheduler instance
        scheduler = RefreshScheduler(repo, embedder, trigger_engine=None)

        if node_ids:
            # Refresh specific nodes
            refreshed_count = 0
            for node_id in node_ids:
                node = repo.get_node(node_id, tenant_id=tenant_id)
                if not node:
                    continue

                try:
                    # Load payload and re-embed
                    text = repo.load_payload_text(node)
                    old = node.embedding
                    new = embedder.encode([text])[0]

                    # Calculate drift
                    if old is not None:
                        drift = 1.0 - float(
                            (old @ new) / ((old**2).sum() ** 0.5 * (new**2).sum() ** 0.5)
                        )
                    else:
                        drift = 0.0

                    # Update
                    timestamp = datetime.now(timezone.utc).isoformat()
                    repo.update_node_embedding(
                        node.id, new, drift, timestamp, tenant_id=node.tenant_id
                    )
                    repo.write_embedding_history(
                        node.id, drift, embedding_ref=node.payload_ref, tenant_id=node.tenant_id
                    )

                    # Emit event if drift > threshold
                    drift_threshold = node.refresh_policy.get("drift_threshold", 0.1)
                    if drift > drift_threshold:
                        repo.append_event(
                            node.id,
                            "refreshed",
                            {
                                "drift_score": drift,
                                "last_refreshed": timestamp,
                                "manual_trigger": True,
                            },
                            tenant_id=node.tenant_id,
                            actor_id=actor_id,  # From JWT, not hardcoded
                            actor_type=actor_type,  # From JWT
                        )

                    refreshed_count += 1
                except Exception as e:
                    logger.error(
                        "Node refresh failed", extra_fields={"node_id": node_id, "error": str(e)}
                    )

            return {
                "status": "completed",
                "mode": "specific_nodes",
                "requested": len(node_ids),
                "refreshed": refreshed_count,
            }
        else:
            # Run full refresh cycle
            scheduler.run_cycle()
            return {
                "status": "completed",
                "mode": "all_due_nodes",
                "message": "Check logs for refresh count",
            }

    except Exception as e:
        logger.error("Admin refresh failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Admin refresh failed: {str(e)}")


@app.get("/admin/embedding/status", response_model=None)
def embedding_status(
    tenant_id: str | None = None,
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Return embedding queue status and DB counts."""
    assert repo is not None, "GraphRepository not initialized"
    if JWT_ENABLED and claims and "admin:refresh" not in claims.scopes:
        raise HTTPException(
            status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
        )

    # Restrict tenant scope under JWT
    effective_tenant_id = claims.tenant_id if (JWT_ENABLED and claims) else tenant_id

    status_counts: dict[str, int] = {}
    with repo._conn(tenant_id=effective_tenant_id) as conn:
        with conn.cursor() as cur:
            where = ""
            params: list[Any] = []
            if effective_tenant_id:
                where = "WHERE tenant_id = %s"
                params.append(effective_tenant_id)
            cur.execute(
                f"""
                SELECT embedding_status, COUNT(*)
                FROM nodes
                {where}
                GROUP BY embedding_status
                """,
                params,
            )
            for row in cur.fetchall():
                status_counts[str(row[0])] = int(row[1])

    redis_client = _get_embedding_redis()
    queue_info: dict[str, int] | dict[str, str] | None = None
    if redis_client:
        try:
            queue_info = queue_depth(redis_client)
        except Exception as e:
            queue_info = {"error": str(e)}

    return {
        "tenant_id": effective_tenant_id,
        "status_counts": status_counts,
        "queue": queue_info,
    }


@app.get("/admin/extraction/status", response_model=None)
def extraction_status(
    tenant_id: str | None = None,
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Return extraction queue status and DB counts.

    Shows:
    - Count of nodes by extraction_status (queued, processing, ready, failed, skipped)
    - Extraction queue depth (if Redis available)
    """
    assert repo is not None, "GraphRepository not initialized"
    if JWT_ENABLED and claims and "admin:refresh" not in claims.scopes:
        raise HTTPException(
            status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
        )

    effective_tenant_id = claims.tenant_id if (JWT_ENABLED and claims) else tenant_id

    status_counts: dict[str, int] = {}
    with repo._conn(tenant_id=effective_tenant_id) as conn:
        with conn.cursor() as cur:
            where = ""
            params: list[Any] = []
            if effective_tenant_id:
                where = "WHERE tenant_id = %s"
                params.append(effective_tenant_id)
            cur.execute(
                f"""
                SELECT props->>'extraction_status' as status, COUNT(*)
                FROM nodes
                {where}
                GROUP BY props->>'extraction_status'
                """,
                params,
            )
            for row in cur.fetchall():
                status = row[0] or "none"
                status_counts[status] = int(row[1])

    redis_client = _get_embedding_redis()
    queue_info: dict[str, int] | dict[str, str] | None = None
    if redis_client:
        try:
            queue_info = extraction_queue_depth(redis_client)
        except Exception as e:
            queue_info = {"error": str(e)}

    return {
        "enabled": EXTRACTION_ENABLED,
        "mode": EXTRACTION_MODE,
        "tenant_id": effective_tenant_id,
        "status_counts": status_counts,
        "queue": queue_info,
    }


@app.post("/admin/extraction/requeue", response_model=None)
def extraction_requeue(
    request: ExtractionRequeueRequest,
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Requeue extraction jobs for nodes.

    Supports:
    - Requeuing by extraction_status (null, failed, queued, etc.)
    - Filtering nodes that never had extraction queued (only_null_status=true)
    - Requeuing specific node_ids
    """
    assert repo is not None, "GraphRepository not initialized"
    if JWT_ENABLED and claims and "admin:refresh" not in claims.scopes:
        raise HTTPException(
            status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
        )

    redis_client = _get_embedding_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Extraction queue unavailable (Redis)")

    if not EXTRACTION_ENABLED:
        raise HTTPException(status_code=503, detail="Extraction not enabled")

    effective_tenant_id = claims.tenant_id if (JWT_ENABLED and claims) else request.tenant_id
    limit = max(1, min(2000, int(request.limit)))
    if request.status is not None:
        request.only_null_status = False

    nodes_to_requeue: list[tuple[str, str | None]] = []

    if request.node_ids:
        # Requeue specific nodes
        with repo._conn(tenant_id=effective_tenant_id) as conn:
            with conn.cursor() as cur:
                if effective_tenant_id:
                    cur.execute(
                        """
                        SELECT id, tenant_id
                        FROM nodes
                        WHERE id = ANY(%s)
                          AND tenant_id = %s
                        """,
                        (request.node_ids, effective_tenant_id),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, tenant_id
                        FROM nodes
                        WHERE id = ANY(%s)
                        """,
                        (request.node_ids,),
                    )
                for row in cur.fetchall():
                    nodes_to_requeue.append((str(row[0]), row[1]))
    else:
        # Query nodes by extraction_status
        with repo._conn(tenant_id=effective_tenant_id) as conn:
            with conn.cursor() as cur:
                where = "WHERE 1=1"
                filter_params: list[Any] = []

                if effective_tenant_id:
                    where += " AND tenant_id = %s"
                    filter_params.append(effective_tenant_id)

                if request.only_null_status:
                    where += " AND (props->>'extraction_status') IS NULL"
                elif request.status:
                    if request.status.lower() == "null":
                        where += " AND (props->>'extraction_status') IS NULL"
                    else:
                        where += " AND props->>'extraction_status' = %s"
                        filter_params.append(request.status)

                cur.execute(
                    f"""
                    SELECT id, tenant_id
                    FROM nodes
                    {where}
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (*filter_params, limit),
                )
                for row in cur.fetchall():
                    nodes_to_requeue.append((str(row[0]), row[1]))

    # Enqueue extraction jobs
    enqueued = 0
    for node_id, tenant_id in nodes_to_requeue:
        try:
            job_id = enqueue_extraction_job(
                redis_client, node_id, tenant_id, force=True, priority="normal"
            )
            if job_id:
                _update_extraction_status(node_id, tenant_id, "queued")
                enqueued += 1
        except Exception as e:
            logger.warning(
                "Failed to enqueue extraction job (non-blocking)",
                extra_fields={"node_id": node_id, "tenant_id": tenant_id, "error": str(e)},
            )

    return {
        "requested": len(nodes_to_requeue),
        "enqueued": enqueued,
        "tenant_id": effective_tenant_id,
    }


@app.post("/admin/embedding/requeue", response_model=None)
def embedding_requeue(
    request: EmbeddingRequeueRequest,
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Requeue embedding jobs and backfill statuses.

    Supports:
    - Requeuing by status (failed, queued, etc.)
    - Filtering nodes without embeddings (only_missing_embedding=true)
    - Backfilling status='ready' for nodes with embeddings (backfill_ready=true)
    """
    assert repo is not None, "GraphRepository not initialized"
    if JWT_ENABLED and claims and "admin:refresh" not in claims.scopes:
        raise HTTPException(
            status_code=403, detail="Insufficient permissions. Required scope: admin:refresh"
        )

    redis_client = _get_embedding_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Embedding queue unavailable")

    # Restrict tenant scope under JWT
    effective_tenant_id = claims.tenant_id if (JWT_ENABLED and claims) else request.tenant_id
    limit = max(1, min(2000, int(request.limit)))

    # Backfill status='ready' for nodes with embeddings if requested
    backfilled = 0
    if request.backfill_ready:
        with repo._conn(tenant_id=effective_tenant_id) as conn:
            with conn.cursor() as cur:
                where = "WHERE embedding IS NOT NULL AND embedding_status != 'ready'"
                params: list[Any] = []
                if effective_tenant_id:
                    where += " AND tenant_id = %s"
                    params.append(effective_tenant_id)
                cur.execute(
                    f"""
                    UPDATE nodes
                    SET embedding_status = 'ready',
                        embedding_error = NULL,
                        embedding_updated_at = NOW(),
                        updated_at = NOW()
                    {where}
                    RETURNING id
                    """,
                    params,
                )
                backfilled = cur.rowcount

    # Determine nodes to requeue
    nodes_to_requeue: list[tuple[str, str | None]] = []
    if request.node_ids:
        with repo._conn(tenant_id=effective_tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, tenant_id, embedding IS NULL AS missing
                    FROM nodes
                    WHERE id = ANY(%s)
                    """,
                    (request.node_ids,),
                )
                for row in cur.fetchall():
                    missing = bool(row[2])
                    if request.only_missing_embedding and not missing:
                        continue
                    nodes_to_requeue.append((str(row[0]), row[1]))
    else:
        with repo._conn(tenant_id=effective_tenant_id) as conn:
            with conn.cursor() as cur:
                where = "WHERE 1=1"
                filter_params: list[Any] = []

                if request.status and request.status.lower() != "all":
                    where += " AND embedding_status = %s"
                    filter_params.append(request.status)

                if request.only_missing_embedding:
                    where += " AND embedding IS NULL"

                if effective_tenant_id:
                    where += " AND tenant_id = %s"
                    filter_params.append(effective_tenant_id)

                cur.execute(
                    f"""
                    SELECT id, tenant_id
                    FROM nodes
                    {where}
                    ORDER BY embedding_updated_at DESC NULLS LAST
                    LIMIT %s
                    """,
                    (*filter_params, limit),
                )
                for row in cur.fetchall():
                    nodes_to_requeue.append((str(row[0]), row[1]))

    enqueued = 0
    for node_id, node_tenant in nodes_to_requeue:
        try:
            repo.mark_embedding_queued(node_id, tenant_id=node_tenant)
            enqueue_embedding_job(redis_client, node_id, node_tenant, action="refresh", force=True)
            enqueued += 1
        except Exception as e:
            logger.error(
                "Failed to requeue embedding",
                extra_fields={"node_id": node_id, "error": str(e)},
            )

    return {
        "backfilled": backfilled,
        "requested": len(nodes_to_requeue),
        "enqueued": enqueued,
    }


@app.get("/admin/anomalies", response_model=None)
def get_anomalies(
    types: str | None = None,
    lookback_hours: int = 24,
    drift_spike_threshold: float = 2.0,
    trigger_storm_threshold: int = 50,
    scheduler_lag_multiplier: float = 2.0,
    tenant_id: str | None = None,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Detect operational anomalies in the knowledge graph.

    Supported anomaly types:
    - drift_spike: Nodes with drift > 2x mean for 3+ consecutive refreshes
    - trigger_storm: >50 trigger_fired events in 1 hour (runaway triggers)
    - scheduler_lag: Nodes overdue for refresh (>2x expected interval)

    Args:
        types: Comma-separated list of anomaly types to check (default: all)
        lookback_hours: Hours to look back for drift/trigger analysis (default: 24)
        drift_spike_threshold: Drift multiplier threshold (default: 2.0 = 2x mean)
        trigger_storm_threshold: Min trigger events to flag as storm (default: 50)
        scheduler_lag_multiplier: Lag multiplier for overdue nodes (default: 2.0 = 2x late)
        tenant_id: Optional tenant ID for multi-tenancy filtering

    Returns:
        Dictionary with anomaly type as key, list of detected anomalies as value
    """
    assert repo is not None, "GraphRepository not initialized"
    try:
        # Parse requested types (default: all)
        requested_types = (
            set(types.split(",")) if types else {"drift_spike", "trigger_storm", "scheduler_lag"}
        )

        results: dict[str, Any] = {}

        # Detect drift spikes
        if "drift_spike" in requested_types:
            drift_spikes = repo.detect_drift_spikes(
                lookback_hours=lookback_hours,
                spike_threshold=drift_spike_threshold,
                min_refreshes=3,
                tenant_id=tenant_id,
            )
            results["drift_spike"] = drift_spikes

        # Detect trigger storms
        if "trigger_storm" in requested_types:
            trigger_storms = repo.detect_trigger_storms(
                lookback_hours=lookback_hours,
                event_threshold=trigger_storm_threshold,
                tenant_id=tenant_id,
            )
            results["trigger_storm"] = trigger_storms

        # Detect scheduler lag
        if "scheduler_lag" in requested_types:
            scheduler_lag = repo.detect_scheduler_lag(
                lag_multiplier=scheduler_lag_multiplier, tenant_id=tenant_id
            )
            results["scheduler_lag"] = scheduler_lag

        # Summary stats
        total_anomalies = sum(len(v) for v in results.values())

        return {
            "anomalies": results,
            "summary": {
                "total": total_anomalies,
                "by_type": {k: len(v) for k, v in results.items()},
                "lookback_hours": lookback_hours,
            },
        }

    except Exception as e:
        logger.error("Anomaly detection failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")


@app.get("/nodes/{node_id}/versions", response_model=None)
def get_node_versions(
    node_id: str,
    limit: int = 10,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Get embedding version history for a node.

    Returns the embedding refresh history including drift scores and timestamps.
    Useful for debugging drift trends and understanding content evolution.

    Security:
        When JWT_ENABLED=true, tenant_id is derived from JWT claims for RLS enforcement.

    Args:
        node_id: Node ID to query
        limit: Maximum number of versions to return (default: 10, max: 100)

    Returns:
        List of version records with version_index, drift_score, created_at, embedding_ref
    """
    assert repo is not None, "GraphRepository not initialized"
    try:
        # Validate limit
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 100")

        # Derive tenant_id from JWT when enabled
        if JWT_ENABLED and claims:
            tenant_id = claims.tenant_id
        else:
            tenant_id = None

        versions = repo.get_node_versions(node_id, limit=limit, tenant_id=tenant_id)

        return {"node_id": node_id, "versions": versions, "count": len(versions)}

    except Exception as e:
        logger.error(
            "Node versions query failed", extra_fields={"node_id": node_id, "error": str(e)}
        )
        raise HTTPException(status_code=500, detail=f"Node versions query failed: {str(e)}")


# ----------------------------------------------------------------------
# Candidate identity resolution
# ----------------------------------------------------------------------

STRONG_SIGNAL_TYPES: frozenset[str] = frozenset(
    {"linkedin_url", "github_url", "medium_url", "email", "phone"}
)


class CandidateIdentifierInput(BaseModel):
    identifier_type: str = Field(..., description="Type of identifier (email, linkedin_url, ...)")
    value: str = Field(..., description="Raw identifier value as sent by the upstream source")
    confidence: float | None = Field(
        default=None,
        description="Optional per-identifier confidence from the upstream source",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional per-identifier metadata (e.g. bridge_tier, identity platform)",
    )


class CandidateProfileInput(BaseModel):
    display_name: str | None = None
    primary_email: str | None = None
    primary_phone: str | None = None
    props: dict[str, Any] = Field(default_factory=dict)
    profile: dict[str, Any] | None = None
    headline: str | None = None
    location_raw: str | None = None
    skills: list[str] | None = None
    seniority_level: str | None = None
    linkedin_url: str | None = None
    linkedin_id: str | None = None
    profile_picture_url: str | None = None


class CandidateResolveRequest(BaseModel):
    source: str = Field(..., description="Upstream source name, e.g. 'vantahire', 'signal'")
    source_record_type: str = Field(
        ..., description="Source record type, e.g. 'application', 'profile'"
    )
    source_record_id: str = Field(..., description="Upstream-stable record id")
    identifiers: list[CandidateIdentifierInput] = Field(
        default_factory=list,
        description="Identifiers for exact-match resolution",
    )
    profile: CandidateProfileInput | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_url: str | None = None
    fetched_at: datetime | None = None
    tenant_id: str | None = None
    # Structured VantaHire provenance — forwarded to candidate_source_records structured
    # columns so downstream Talent Search can scope by org/recruiter without JSONB scans.
    org_id: str | None = None
    job_id: str | None = None
    effective_recruiter_id: str | None = None
    created_by_user_id: str | None = None
    resume_source: str | None = None
    # Structured Signal tags — forwarded to candidate_source_records.job_tags so
    # tag-based candidate search uses a GIN index rather than scanning JSONB payloads.
    job_tags: list[str] = Field(default_factory=list)


class MatchedIdentifier(BaseModel):
    identifier_type: str
    value_normalized: str


class AttachedIdentifier(BaseModel):
    identifier_type: str
    value_normalized: str


class SkippedIdentifier(BaseModel):
    identifier_type: str | None = None
    value: str | None = None
    reason: str


class ResolveConflict(BaseModel):
    identifier_type: str
    value_normalized: str
    candidate_id: str | None = None
    reason: str | None = None


class CandidateResolveResponse(BaseModel):
    candidate_id: str | None
    global_candidate_id: str | None = None
    resolution_status: str  # "created" | "matched" | "review_required"
    matched_identifier: MatchedIdentifier | None = None
    attached_identifiers: list[AttachedIdentifier] = Field(default_factory=list)
    skipped_identifiers: list[SkippedIdentifier] = Field(default_factory=list)
    source_record_id: str | None = None
    conflicts: list[ResolveConflict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CandidateTagSearchResult(BaseModel):
    candidate_id: str
    display_name: str | None = None
    primary_email: str | None = None
    signal_candidate_id: str
    stored_tags: list[str]
    matched_tags: list[str]
    overlap_count: int
    overlap_ratio: float
    profile: dict[str, Any] | None = None


# Server-side ceiling for tag-search results per call. Requests above it are
# clamped (not rejected) and reported via total_matched/truncated so callers
# can log honestly instead of silently losing candidates.
# Clamped to the repository's hard ceiling (1000) so applied_limit can never
# overstate what the storage layer will actually return.
TAG_SEARCH_MAX_LIMIT = min(int(os.getenv("ACTIVEKG_TAG_SEARCH_MAX_LIMIT", "500")), 1000)
PRIVATE_SEARCH_MAX_LIMIT = min(
    int(os.getenv("ACTIVEKG_PRIVATE_SEARCH_MAX_LIMIT", "1000")),
    1000,
)


class CandidateSearchByTagsRequest(BaseModel):
    tags: list[str]
    tenant_id: str | None = None
    limit: int = Field(default=100, ge=1)


class CandidateSearchByTagsResponse(BaseModel):
    results: list[CandidateTagSearchResult]
    query_tags: list[str]
    total: int
    # Contract fields for callers: how many candidates cleared the overlap
    # threshold in total, and whether the response was cut by the limit.
    total_matched: int = 0
    truncated: bool = False
    applied_limit: int = 0


class TenantPrivateCandidateSearchRequest(BaseModel):
    query_text: str = Field(default="", max_length=10_000)
    skills_any: list[str] = Field(default_factory=list, max_length=100)
    tenant_id: str | None = None
    limit: int = Field(default=100, ge=1)


class TenantPrivateCandidateSearchResult(BaseModel):
    candidate_id: str
    global_candidate_id: str | None = None
    display_name: str | None = None
    linkedin_url: str | None = None
    linkedin_id: str | None = None
    headline: str | None = None
    location_raw: str | None = None
    skills: list[str] = Field(default_factory=list)
    seniority_level: str | None = None
    keyword_score: float = 0.0
    skill_overlap_count: int = 0
    evidence_surface: Literal["tenant_private_v1"] = "tenant_private_v1"


class TenantPrivateCandidateSearchResponse(BaseModel):
    surface: Literal["tenant_private_v1"] = "tenant_private_v1"
    results: list[TenantPrivateCandidateSearchResult]
    total: int
    total_available: int
    truncated: bool
    applied_limit: int


_PRIVATE_SEARCH_TOKEN = re.compile(r"[a-z0-9][a-z0-9+#.-]{1,63}")


def _private_search_terms(query_text: str, skills: list[str]) -> tuple[list[str], list[str]]:
    query_terms = list(
        dict.fromkeys(
            match.group(0) for match in _PRIVATE_SEARCH_TOKEN.finditer(query_text.lower())
        )
    )[:64]
    normalized_skills = list(
        dict.fromkeys(value for value in (str(skill).strip().lower() for skill in skills) if value)
    )[:100]
    return query_terms, normalized_skills


def _evaluate_strong_signal_mismatch(
    incoming_normalized: list[tuple[str, str, str, Any, Any]],
    canonical_identifiers: list,
) -> tuple[list[tuple[str, str]], list[tuple[str, str, str]]]:
    """Compare incoming strong-signal identifiers against the canonical candidate.

    Returns:
        strong_matches: (itype, incoming_norm) where values agree with canonical
        strong_mismatches: (itype, incoming_norm, canonical_norm) where values conflict

    Types absent on the canonical candidate are ignored (not mismatches).
    """
    canonical_by_type: dict[str, set[str]] = {}
    for ident in canonical_identifiers:
        if ident.identifier_type in STRONG_SIGNAL_TYPES:
            canonical_by_type.setdefault(ident.identifier_type, set()).add(ident.value_normalized)

    strong_matches: list[tuple[str, str]] = []
    strong_mismatches: list[tuple[str, str, str]] = []

    seen_incoming: set[str] = set()
    for itype, _raw, norm, *_ in incoming_normalized:
        if itype not in STRONG_SIGNAL_TYPES:
            continue
        if itype not in canonical_by_type:
            continue
        if itype in seen_incoming:
            continue
        seen_incoming.add(itype)
        if norm in canonical_by_type[itype]:
            strong_matches.append((itype, norm))
        else:
            canonical_norm = next(iter(canonical_by_type[itype]))
            strong_mismatches.append((itype, norm, canonical_norm))

    return strong_matches, strong_mismatches


@app.post(
    "/candidates/resolve",
    response_model=CandidateResolveResponse,
    dependencies=[Depends(require_scope("kg:write"))],
)
def resolve_candidate(
    payload: CandidateResolveRequest,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Resolve-or-create a canonical ActiveKG candidate from upstream evidence.

    Exact identifier-based matching only: if any normalized identifier already
    belongs to a candidate, that candidate is returned; otherwise a new
    canonical candidate is created. Upstream payloads are preserved verbatim in
    ``candidate_source_records``. If identifiers point at multiple distinct
    candidates the request is flagged ``review_required`` and no merge happens
    — merging candidates is an explicit operation, not an implicit side-effect.
    """
    if JWT_ENABLED and claims:
        tenant_id = claims.tenant_id
    else:
        tenant_id = payload.tenant_id
    return _execute_candidate_resolve(payload, tenant_id=tenant_id)


def _upsert_resolve_source_record(
    payload: CandidateResolveRequest,
    *,
    candidate_id: str,
    tenant_id: str | None,
) -> CandidateSourceRecord:
    """Persist upstream evidence without implying a canonical-profile refresh."""
    assert candidate_repo is not None
    return candidate_repo.upsert_source_record(
        CandidateSourceRecord(
            candidate_id=candidate_id,
            tenant_id=tenant_id,
            source=payload.source,
            source_record_type=payload.source_record_type,
            source_record_id=payload.source_record_id,
            source_url=payload.source_url,
            payload=payload.payload or {},
            fetched_at=payload.fetched_at,
            org_id=payload.org_id,
            job_id=payload.job_id,
            effective_recruiter_id=payload.effective_recruiter_id,
            created_by_user_id=payload.created_by_user_id,
            resume_source=payload.resume_source,
            job_tags=list(payload.job_tags) if payload.job_tags else [],
        )
    )


def _execute_candidate_resolve(
    payload: CandidateResolveRequest,
    *,
    tenant_id: str | None,
    pre_skipped: list[SkippedIdentifier] | None = None,
) -> CandidateResolveResponse:
    if candidate_repo is None:
        raise HTTPException(status_code=503, detail="CandidateRepository not initialized")

    if not payload.identifiers:
        raise HTTPException(status_code=400, detail="at least one identifier is required")

    skipped: list[SkippedIdentifier] = list(pre_skipped or [])
    warnings: list[str] = []

    # 1. Normalize all identifiers up-front; any failure is a 400.
    normalized: list[tuple[str, str, str, float | None, dict[str, Any]]] = []
    # (type, raw, normalized, confidence, metadata)
    seen_norm: set[tuple[str, str]] = set()
    for ident in payload.identifiers:
        if ident.identifier_type not in IDENTIFIER_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"unknown identifier_type: {ident.identifier_type!r}",
            )
        try:
            norm = normalize_identifier(ident.identifier_type, ident.value)
        except IdentifierNormalizationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        key = (ident.identifier_type, norm)
        if key in seen_norm:
            skipped.append(
                SkippedIdentifier(
                    identifier_type=ident.identifier_type,
                    value=ident.value,
                    reason="duplicate identifier in request",
                )
            )
            continue
        seen_norm.add(key)
        normalized.append(
            (ident.identifier_type, ident.value, norm, ident.confidence, ident.metadata or {})
        )

    # 2. For each identifier, check if it already belongs to a candidate.
    matches: list[tuple[str, str, str]] = []  # (type, normalized, candidate_id)
    seen_candidate_ids: set[str] = set()
    for itype, _raw, norm, _conf, _meta in normalized:
        existing = candidate_repo.find_candidate_by_identifier(itype, norm, tenant_id=tenant_id)
        if existing is not None:
            matches.append((itype, norm, existing.candidate_id))
            seen_candidate_ids.add(existing.candidate_id)

    # 3. Conflicting matches across different candidates → review_required. We
    # do NOT attach identifiers or write a source record in this case: merging
    # canonical candidates is an explicit operation, not a silent side-effect.
    if len(seen_candidate_ids) > 1:
        conflicts = [
            ResolveConflict(
                identifier_type=itype,
                value_normalized=norm,
                candidate_id=cid,
                reason="identifier resolves to a different canonical candidate",
            )
            for itype, norm, cid in matches
        ]
        logger.warning(
            "candidate resolve flagged review_required",
            extra_fields={
                "source": payload.source,
                "source_record_id": payload.source_record_id,
                "candidate_count": len(seen_candidate_ids),
                "tenant_id": tenant_id,
            },
        )
        warnings.append(
            f"identifiers point at {len(seen_candidate_ids)} distinct candidates; "
            "merging requires explicit review"
        )
        return CandidateResolveResponse(
            candidate_id=None,
            resolution_status="review_required",
            matched_identifier=None,
            attached_identifiers=[],
            skipped_identifiers=skipped,
            source_record_id=None,
            conflicts=conflicts,
            warnings=warnings,
        )

    profile = payload.profile or CandidateProfileInput()

    # Identifiers blocked from attachment due to a tolerated strong-signal mismatch.
    mismatch_blocked: set[tuple[str, str]] = set()

    if len(seen_candidate_ids) == 1:
        candidate_id = next(iter(seen_candidate_ids))
        candidate = candidate_repo.get_candidate(candidate_id, tenant_id=tenant_id)
        assert candidate is not None

        # 3a. Strong-signal mismatch evaluation — compare incoming strong-signal
        # identifiers against the canonical candidate's existing identifiers of the
        # same type. Absent types are ignored; present-but-different types are conflicts.
        canonical_identifiers = candidate_repo.list_identifiers(candidate_id, tenant_id=tenant_id)
        strong_matches, strong_mismatches = _evaluate_strong_signal_mismatch(
            normalized, canonical_identifiers
        )

        if strong_mismatches:
            n_matches = len(strong_matches)
            n_mismatches = len(strong_mismatches)
            # Tolerate only the case of ≥2 corroborating strong matches vs exactly 1 conflict.
            tolerated = n_matches >= 2 and n_mismatches == 1

            if not tolerated:
                mismatch_conflicts = [
                    ResolveConflict(
                        identifier_type=itype,
                        value_normalized=incoming_norm,
                        candidate_id=candidate_id,
                        reason=(
                            f"strong signal mismatch: incoming {itype}={incoming_norm!r} "
                            f"conflicts with canonical value {canonical_norm!r}"
                        ),
                    )
                    for itype, incoming_norm, canonical_norm in strong_mismatches
                ]
                logger.warning(
                    "candidate resolve flagged review_required due to strong-signal mismatch",
                    extra_fields={
                        "source": payload.source,
                        "source_record_id": payload.source_record_id,
                        "strong_matches": n_matches,
                        "strong_mismatches": n_mismatches,
                        "candidate_id": candidate_id,
                        "tenant_id": tenant_id,
                    },
                )
                warnings.append(
                    f"strong-signal mismatch: {n_mismatches} conflict(s) against "
                    f"{n_matches} match(es); manual review required"
                )
                return CandidateResolveResponse(
                    candidate_id=None,
                    resolution_status="review_required",
                    matched_identifier=None,
                    attached_identifiers=[],
                    skipped_identifiers=skipped,
                    source_record_id=None,
                    conflicts=mismatch_conflicts,
                    warnings=warnings,
                )
            else:
                # Tolerated: accept the match but block the mismatching identifier from
                # being attached so canonical strong-signal values are not diluted.
                for itype, incoming_norm, canonical_norm in strong_mismatches:
                    mismatch_blocked.add((itype, incoming_norm))
                    skipped.append(
                        SkippedIdentifier(
                            identifier_type=itype,
                            value=incoming_norm,
                            reason=(
                                f"strong signal mismatch: incoming {itype}={incoming_norm!r} "
                                f"conflicts with canonical value {canonical_norm!r}; not attached"
                            ),
                        )
                    )
                warnings.append(
                    f"strong-signal mismatch tolerated: {n_matches} match(es) outweigh "
                    f"{n_mismatches} mismatch(es); conflicting identifier(s) not attached"
                )

        status = "matched"
        matched_itype, matched_norm, _ = matches[0]
        matched_identifier = MatchedIdentifier(
            identifier_type=matched_itype,
            value_normalized=matched_norm,
        )
        # Fill in profile fields only when currently empty — never clobber
        # canonical data with upstream drift. Surface drift as a warning so
        # callers can reconcile upstream.
        updates: dict[str, Any] = {}
        if profile.display_name and not candidate.display_name:
            updates["display_name"] = profile.display_name
        elif (
            profile.display_name
            and candidate.display_name
            and profile.display_name != candidate.display_name
        ):
            warnings.append("upstream display_name differs from canonical; canonical preserved")
        if profile.primary_email and not candidate.primary_email:
            updates["primary_email"] = profile.primary_email
        elif (
            profile.primary_email
            and candidate.primary_email
            and profile.primary_email.lower() != candidate.primary_email.lower()
        ):
            warnings.append("upstream primary_email differs from canonical; canonical preserved")
        if profile.primary_phone and not candidate.primary_phone:
            updates["primary_phone"] = profile.primary_phone
        if (
            profile.primary_phone
            and candidate.primary_phone
            and profile.primary_phone != candidate.primary_phone
        ):
            warnings.append("upstream primary_phone differs from canonical; canonical preserved")
        # Enrichment fields: freshest-wins, but NEVER replace something with
        # nothing. The Signal handler defaults profile to {} and skills to []
        # (never None), so `is not None` checks let a blob-less payload WIPE
        # canonical data — a truthiness guard makes emptiness a no-op while a
        # fuller payload still always overwrites. Canonical-trust invariant:
        # "whatever is in Memory is the freshest representation of this person."
        if profile.profile:
            updates["profile"] = profile.profile
        if profile.headline:
            updates["headline"] = profile.headline
        if profile.location_raw:
            updates["location_raw"] = profile.location_raw
        if profile.skills:
            updates["skills"] = profile.skills
        if profile.seniority_level:
            updates["seniority_level"] = profile.seniority_level
        if profile.linkedin_url:
            updates["linkedin_url"] = profile.linkedin_url
        if profile.linkedin_id:
            updates["linkedin_id"] = profile.linkedin_id
        if profile.profile_picture_url:
            updates["profile_picture_url"] = profile.profile_picture_url

        if updates:
            candidate_repo.update_candidate(candidate.candidate_id, tenant_id=tenant_id, **updates)
    else:
        candidate = Candidate(
            tenant_id=tenant_id,
            display_name=profile.display_name,
            primary_email=profile.primary_email,
            primary_phone=profile.primary_phone,
            props=profile.props or {},
            metadata=payload.metadata or {},
            profile=profile.profile or {},
            headline=profile.headline,
            location_raw=profile.location_raw,
            skills=profile.skills or [],
            seniority_level=profile.seniority_level,
            linkedin_url=profile.linkedin_url,
            linkedin_id=profile.linkedin_id,
            profile_picture_url=profile.profile_picture_url,
        )
        candidate_repo.create_candidate(candidate)
        status = "created"
        matched_identifier = None

    # 4. Attach every (valid) identifier. Any conflict here means a concurrent
    # writer grabbed the identifier between our lookup and insert, or the
    # identifier is already owned by a different candidate — surface it and
    # downgrade to review_required rather than silently re-pointing rows.
    # Identifiers blocked by the strong-signal mismatch rule are also skipped.
    attached: list[AttachedIdentifier] = []
    conflicts: list[ResolveConflict] = []
    for itype, raw, norm, conf, meta in normalized:
        if (itype, norm) in mismatch_blocked:
            continue
        try:
            candidate_repo.add_identifier(
                candidate.candidate_id,
                itype,
                raw,
                tenant_id=tenant_id,
                source=payload.source,
                confidence=conf,
                metadata=meta or None,
            )
            attached.append(AttachedIdentifier(identifier_type=itype, value_normalized=norm))
        except IdentifierConflict:
            owner = candidate_repo.find_candidate_by_identifier(itype, norm, tenant_id=tenant_id)
            owner_id = owner.candidate_id if owner is not None else None
            conflicts.append(
                ResolveConflict(
                    identifier_type=itype,
                    value_normalized=norm,
                    candidate_id=owner_id,
                    reason="identifier owned by a different candidate",
                )
            )

    if conflicts:
        warnings.append(
            f"{len(conflicts)} identifier(s) could not be attached due to ownership conflicts"
        )
        return CandidateResolveResponse(
            candidate_id=candidate.candidate_id,
            resolution_status="review_required",
            matched_identifier=matched_identifier,
            attached_identifiers=attached,
            skipped_identifiers=skipped,
            source_record_id=None,
            conflicts=conflicts,
            warnings=warnings,
        )

    # 5. Upsert the source record — idempotent on
    # (tenant_id, source, source_record_type, source_record_id).
    record = _upsert_resolve_source_record(
        payload,
        candidate_id=candidate.candidate_id,
        tenant_id=tenant_id,
    )

    return CandidateResolveResponse(
        candidate_id=candidate.candidate_id,
        resolution_status=status,
        matched_identifier=matched_identifier,
        attached_identifiers=attached,
        skipped_identifiers=skipped,
        source_record_id=record.source_record_id,
        conflicts=[],
        warnings=warnings,
    )


# ----------------------------------------------------------------------
# VantaHire application evidence → canonical candidate
# ----------------------------------------------------------------------


class VantahireApplicationResolveRequest(BaseModel):
    """Raw VantaHire application payload.

    ActiveKG translates the VantaHire-specific fields below into canonical
    identifiers and profile data, then routes them through the same resolve-
    or-create flow used by :func:`resolve_candidate`.
    """

    application_id: str = Field(..., description="VantaHire application id")
    resume_id: str | None = None
    job_id: str | None = None
    org_id: str | None = None

    # Recruiter/uploader provenance — written to candidate_source_records structured
    # columns so Talent Search can filter by recruiter or org without JSONB scans.
    effective_recruiter_id: str | None = None
    created_by_user_id: str | None = None
    resume_source: str | None = None

    applicant_name: str | None = None
    email: str | None = None
    phone: str | None = None

    linkedin_url: str | None = None
    github_url: str | None = None
    medium_url: str | None = None
    other_links: list[str] = Field(default_factory=list)

    resume_gcp_url: str | None = None
    skills: list[str] = Field(default_factory=list)

    source_metadata: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str | None = None


_VANTAHIRE_LINK_TYPES: tuple[tuple[str, str], ...] = (
    ("linkedin_url", "linkedin_url"),
    ("github_url", "github_url"),
    ("medium_url", "medium_url"),
)


def _collect_vantahire_identifiers(
    payload: VantahireApplicationResolveRequest,
) -> tuple[list[CandidateIdentifierInput], list[dict[str, str]]]:
    """Translate VantaHire fields into canonical identifier inputs.

    Invalid optional links are dropped (and reported in ``skipped``) rather than
    failing the whole request — VantaHire data is user-entered and one bad
    profile URL shouldn't block ingestion of an application whose email is
    still perfectly good.
    """
    identifiers: list[CandidateIdentifierInput] = []
    skipped: list[dict[str, str]] = []

    def _add_required(itype: str, value: str) -> None:
        try:
            normalize_identifier(itype, value)
        except IdentifierNormalizationError as e:
            raise HTTPException(status_code=400, detail=f"{itype}: {e}")
        identifiers.append(CandidateIdentifierInput(identifier_type=itype, value=value))

    def _add_optional(itype: str, value: str | None) -> None:
        if not value:
            return
        try:
            normalize_identifier(itype, value)
        except IdentifierNormalizationError as e:
            skipped.append({"identifier_type": itype, "value": value, "reason": str(e)})
            return
        identifiers.append(CandidateIdentifierInput(identifier_type=itype, value=value))

    _add_required("vantahire_application_id", payload.application_id)
    _add_optional("vantahire_resume_id", payload.resume_id)

    for field_name, itype in _VANTAHIRE_LINK_TYPES:
        _add_optional(itype, getattr(payload, field_name))

    _add_optional("email", payload.email)
    _add_optional("phone", payload.phone)

    for link in payload.other_links:
        if not link:
            continue
        # Pick the most specific canonical type we can, falling back to a
        # generic website url.
        guessed: str | None = None
        lowered = link.lower()
        if "linkedin.com" in lowered:
            guessed = "linkedin_url"
        elif "github.com" in lowered:
            guessed = "github_url"
        elif "medium.com" in lowered:
            guessed = "medium_url"
        elif "twitter.com" in lowered or "x.com" in lowered:
            guessed = "twitter_url"
        elif "stackoverflow.com" in lowered:
            guessed = "stackoverflow_url"
        else:
            guessed = "website_url"
        _add_optional(guessed, link)

    return identifiers, skipped


@app.post(
    "/candidates/resolve/vantahire/application",
    response_model=CandidateResolveResponse,
    dependencies=[Depends(require_scope("kg:write"))],
)
def resolve_candidate_from_vantahire_application(
    payload: VantahireApplicationResolveRequest,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Resolve-or-create a canonical candidate from a VantaHire application.

    Maps VantaHire application fields onto canonical identifiers
    (``vantahire_application_id``, ``vantahire_resume_id``, ``linkedin_url``,
    ``github_url``, ``medium_url``, ``email``, ``phone``, plus any profile
    URLs under ``other_links``) and stores the full application payload in
    ``candidate_source_records`` with ``source='vantahire'`` /
    ``source_record_type='application'``.

    Per-field normalization failures on *optional* identifiers are dropped
    silently; the request only 400s if no usable identifier survives.
    """
    identifiers, skipped = _collect_vantahire_identifiers(payload)
    if not identifiers:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "no usable identifiers in VantaHire payload",
                "skipped": skipped,
            },
        )

    source_payload: dict[str, Any] = payload.model_dump(exclude_none=False)
    if skipped:
        source_payload["_skipped_identifiers"] = skipped

    # Ensure provenance fields are always present in the payload under canonical keys
    # so consumers that read raw JSONB can still find them.
    source_payload["source"] = "vantahire"
    source_payload["source_record_type"] = "application"

    resolve_request = CandidateResolveRequest(
        source="vantahire",
        source_record_type="application",
        source_record_id=payload.application_id,
        identifiers=identifiers,
        profile=CandidateProfileInput(
            display_name=payload.applicant_name,
            primary_email=payload.email,
            primary_phone=payload.phone,
            # org_id and job_id are NOT put on the canonical candidate — they are
            # VantaHire-specific scoping metadata and live on the source record only.
            props={
                k: v
                for k, v in {
                    "resume_gcp_url": payload.resume_gcp_url,
                    "skills": payload.skills or None,
                }.items()
                if v
            },
        ),
        payload=source_payload,
        metadata=payload.source_metadata or {},
        source_url=payload.resume_gcp_url,
        tenant_id=payload.tenant_id,
        # Structured provenance — written to candidate_source_records columns
        org_id=payload.org_id,
        job_id=payload.job_id,
        effective_recruiter_id=payload.effective_recruiter_id,
        created_by_user_id=payload.created_by_user_id,
        resume_source=payload.resume_source,
    )

    if JWT_ENABLED and claims:
        tenant_id = claims.tenant_id
    else:
        tenant_id = payload.tenant_id

    pre_skipped = [
        SkippedIdentifier(
            identifier_type=s.get("identifier_type"),
            value=s.get("value"),
            reason=s.get("reason", ""),
        )
        for s in skipped
    ]
    return _execute_candidate_resolve(resolve_request, tenant_id=tenant_id, pre_skipped=pre_skipped)


# ----------------------------------------------------------------------
# Signal sourced-candidate evidence → canonical candidate
# ----------------------------------------------------------------------


class SignalIdentityInput(BaseModel):
    """A single identity link discovered by Signal (linkedin, github, ...)."""

    platform: str | None = None
    profileUrl: str | None = None
    profile_url: str | None = None
    confidence: float | None = None
    bridgeTier: str | None = None
    bridge_tier: str | None = None
    model_config = {"extra": "allow"}


class SignalCandidateResolveRequest(BaseModel):
    """Raw Signal sourced-candidate / profile payload.

    ActiveKG translates Signal-specific fields below into canonical identifiers
    and profile data, then routes them through the same resolve-or-create flow
    used by :func:`resolve_candidate`. The full payload is preserved verbatim
    in ``candidate_source_records``.
    """

    signal_candidate_id: str = Field(..., description="Signal's stable candidate id")
    source_record_type: str = Field(
        default="sourced_candidate",
        description="Signal record type: 'sourced_candidate' or 'profile'",
    )

    linkedinUrl: str | None = None
    identities: list[SignalIdentityInput] = Field(default_factory=list)

    display_name: str | None = None
    headline: str | None = None
    identitySummary: str | None = None
    aiSummary: str | None = None

    rank: int | float | None = None
    request_id: str | None = None
    external_job_id: str | None = None
    crustdata: dict[str, Any] | None = Field(default=None, description="Raw Crustdata profile blob")
    profile_observed_at: datetime | None = Field(
        default=None,
        description="When the upstream profile was observed, not when this ingest was retried",
    )

    tags: list[str] = Field(default_factory=list)
    sourcing_context: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str | None = None

    model_config = {"extra": "allow"}


_SIGNAL_PLATFORM_TO_ITYPE: dict[str, str] = {
    "linkedin": "linkedin_url",
    "github": "github_url",
    "medium": "medium_url",
    "twitter": "twitter_url",
    "x": "twitter_url",
    "stackoverflow": "stackoverflow_url",
    "stack_overflow": "stackoverflow_url",
    "portfolio": "portfolio_url",
    "website": "website_url",
}


def _normalize_signal_tags(tags: list[str]) -> list[str]:
    """Trim, lowercase, deduplicate, and drop empty strings from a tag list."""
    seen: set[str] = set()
    result: list[str] = []
    for t in tags:
        normalized = t.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _as_utc(value: datetime) -> datetime:
    """Normalize observation timestamps before comparing provider evidence."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _signal_observation_is_not_newer(
    incoming: datetime | None,
    existing: CandidateSourceRecord | None,
) -> bool:
    """Return true when an ingest cannot improve the stored observation."""
    if existing is None or existing.fetched_at is None:
        return False
    if incoming is None:
        # Once a record has trustworthy observation time, an undated replay
        # must not be allowed to masquerade as newer evidence.
        return True
    return _as_utc(incoming) <= _as_utc(existing.fetched_at)


def _signal_mirror_fields(
    payload: SignalCandidateResolveRequest,
) -> tuple[dict[str, Any], str | None, str | None, list[str]]:
    """Derive mirror fields defensively from accepted Signal evidence."""
    crustdata = payload.crustdata or {}
    basic_profile = crustdata.get("basic_profile")
    basic_profile = basic_profile if isinstance(basic_profile, dict) else {}
    headline = payload.headline or basic_profile.get("headline")

    skills_node = crustdata.get("skills")
    skills_node = skills_node if isinstance(skills_node, dict) else {}
    skills = skills_node.get("professional_network_skills")
    skills = skills if isinstance(skills, list) else []

    experience = crustdata.get("experience")
    experience = experience if isinstance(experience, dict) else {}
    employment = experience.get("employment_details")
    employment = employment if isinstance(employment, dict) else {}
    current = employment.get("current")
    seniority = (
        current[0].get("seniority_level")
        if isinstance(current, list) and current and isinstance(current[0], dict)
        else None
    )
    return crustdata, headline, seniority, skills


def _mirror_signal_candidate_to_global(
    *,
    payload: SignalCandidateResolveRequest,
    result: CandidateResolveResponse,
    tenant_id: str | None,
    crustdata: dict[str, Any],
    headline_idx: str | None,
    seniority: str | None,
    skills_idx: list[str],
    require_public_mirror: bool | None = None,
) -> None:
    """Mirror a durable tenant resolve into global memory.

    Legacy callers keep best-effort behavior. ``public_v1`` callers require a
    global ID and receive 503 on mirror failure so the same tenant-side resolve
    can be retried idempotently.
    """
    from activekg.api.global_memory import GLOBAL_MEMORY_ENABLED as _GM_ENABLED

    strict_public_mirror = (
        (payload.source_metadata or {}).get("public_memory_surface") == "public_v1"
        if require_public_mirror is None
        else require_public_mirror
    )
    if not _GM_ENABLED:
        if strict_public_mirror:
            raise HTTPException(status_code=503, detail="public memory is disabled")
        return
    if result.resolution_status not in ("created", "matched") or not result.candidate_id:
        if strict_public_mirror:
            raise HTTPException(
                status_code=503,
                detail="public memory mirror requires a durable created or matched candidate",
            )
        return

    try:
        from activekg.api.global_memory import (
            _get_tenant_conn as _gm_tenant_conn,
        )
        from activekg.api.global_memory import (
            upsert_signal_candidate_to_global,
        )

        bp = crustdata.get("basic_profile") or {}
        loc = bp.get("location") or {}
        gm_conn = _gm_tenant_conn(tenant_id)
        try:
            with gm_conn.cursor() as gm_cur:
                gc_id = upsert_signal_candidate_to_global(
                    gm_cur,
                    tenant_id=tenant_id,
                    linkedin_url=payload.linkedinUrl,
                    name=bp.get("name") or payload.display_name,
                    headline=bp.get("headline") or headline_idx,
                    location_city=loc.get("city"),
                    location_country=loc.get("country"),
                    seniority_band=seniority,
                    skills=skills_idx or None,
                    signal_candidate_id=payload.signal_candidate_id,
                    profile_observed_at=payload.profile_observed_at,
                    public_profile=crustdata,
                    public_role_family=(payload.source_metadata or {}).get(
                        "public_candidate_role_family"
                    ),
                    public_market=(payload.source_metadata or {}).get("public_market"),
                )
                if gc_id:
                    result.global_candidate_id = gc_id
                    gm_cur.execute(
                        "UPDATE candidates SET global_candidate_id = %s"
                        " WHERE tenant_id = %s AND candidate_id = %s"
                        " AND global_candidate_id IS DISTINCT FROM %s",
                        (gc_id, tenant_id, result.candidate_id, gc_id),
                    )
            gm_conn.commit()
        finally:
            gm_conn.close()
    except Exception as gm_err:
        logger.warning(
            "Global-memory mirror failed for signal candidate",
            extra_fields={
                "signal_candidate_id": payload.signal_candidate_id,
                "error": str(gm_err),
            },
        )
        if strict_public_mirror:
            raise HTTPException(
                status_code=503,
                detail="public memory mirror unavailable; retry this idempotent ingest",
            ) from gm_err

    if strict_public_mirror and not result.global_candidate_id:
        raise HTTPException(
            status_code=503,
            detail="public memory mirror returned no global candidate ID; retry after identity repair",
        )


def _guess_itype_from_url(url: str) -> str:
    lowered = url.lower()
    if "linkedin.com" in lowered:
        return "linkedin_url"
    if "github.com" in lowered:
        return "github_url"
    if "medium.com" in lowered:
        return "medium_url"
    if "twitter.com" in lowered or "x.com" in lowered:
        return "twitter_url"
    if "stackoverflow.com" in lowered:
        return "stackoverflow_url"
    return "website_url"


def _collect_signal_identifiers(
    payload: SignalCandidateResolveRequest,
) -> tuple[list[CandidateIdentifierInput], list[dict[str, Any]]]:
    """Translate Signal fields into canonical identifier inputs.

    Invalid optional identities are dropped and reported in ``skipped`` — one
    bad profileUrl from a Signal enrichment shouldn't block ingestion of a
    candidate whose signal_candidate_id and primary linkedin are valid.
    """
    identifiers: list[CandidateIdentifierInput] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(
        itype: str,
        value: str,
        *,
        required: bool,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        try:
            norm = normalize_identifier(itype, value)
        except IdentifierNormalizationError as e:
            if required:
                raise HTTPException(status_code=400, detail=f"{itype}: {e}")
            skipped.append({"identifier_type": itype, "value": value, "reason": str(e)})
            return
        key = (itype, norm)
        if key in seen:
            return
        seen.add(key)
        identifiers.append(
            CandidateIdentifierInput(
                identifier_type=itype,
                value=value,
                confidence=confidence,
                metadata=metadata or {},
            )
        )

    _add("signal_candidate_id", payload.signal_candidate_id, required=True)

    if payload.linkedinUrl:
        _add("linkedin_url", payload.linkedinUrl, required=False)

    for identity in payload.identities:
        url = identity.profileUrl or identity.profile_url
        if not url:
            continue
        platform = (identity.platform or "").strip().lower()
        itype = _SIGNAL_PLATFORM_TO_ITYPE.get(platform) or _guess_itype_from_url(url)
        bridge_tier = identity.bridgeTier or identity.bridge_tier
        meta: dict[str, Any] = {}
        if platform:
            meta["signal_platform"] = platform
        if bridge_tier:
            meta["bridge_tier"] = bridge_tier
        _add(
            itype,
            url,
            required=False,
            confidence=identity.confidence,
            metadata=meta,
        )

    return identifiers, skipped


@app.post(
    "/candidates/resolve/signal/candidate",
    response_model=CandidateResolveResponse,
    dependencies=[Depends(require_scope("kg:write"))],
)
def resolve_candidate_from_signal(
    payload: SignalCandidateResolveRequest,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Resolve-or-create a canonical candidate from a Signal sourced-candidate payload.

    Maps Signal fields onto canonical identifiers (``signal_candidate_id``,
    ``linkedin_url`` from ``candidate.linkedinUrl``, and one identifier per
    entry of ``identities[]``) and stores the full Signal payload in
    ``candidate_source_records`` with ``source='signal'``. Identity confidence
    and bridge tier are preserved in identifier metadata so downstream match
    review can weigh Signal's enrichment quality.
    """
    if payload.source_record_type not in {"sourced_candidate", "profile"}:
        raise HTTPException(
            status_code=400,
            detail=(f"unsupported Signal source_record_type: {payload.source_record_type!r}"),
        )

    identifiers, skipped = _collect_signal_identifiers(payload)

    source_payload: dict[str, Any] = payload.model_dump(mode="json", exclude_none=False)
    if skipped:
        source_payload["_skipped_identifiers"] = skipped

    profile_props: dict[str, Any] = {
        k: v
        for k, v in {
            "headline": payload.headline,
            "identity_summary": payload.identitySummary,
            "ai_summary": payload.aiSummary,
            "rank": payload.rank,
            "request_id": payload.request_id,
            "external_job_id": payload.external_job_id,
            "sourcing_context": payload.sourcing_context or None,
        }.items()
        if v is not None
    }

    crustdata = payload.crustdata or {}

    # Extract fields from crustdata for indexing
    headline_idx = payload.headline or crustdata.get("basic_profile", {}).get("headline")

    location_raw = None
    if loc := crustdata.get("basic_profile", {}).get("location"):
        location_raw = loc.get("full_location") or loc.get("raw")

    skills_idx = []
    if s := crustdata.get("skills"):
        skills_idx = s.get("professional_network_skills") or []

    seniority = None
    exp = crustdata.get("experience", {}).get("employment_details", {})
    if current := exp.get("current"):
        if isinstance(current, list) and len(current) > 0:
            seniority = current[0].get("seniority_level")

    linkedin_id = None
    if payload.linkedinUrl:
        try:
            canonical_linkedin = normalize_identifier("linkedin_url", payload.linkedinUrl)
            linkedin_id = canonical_linkedin.rsplit("/", 1)[-1]
        except IdentifierNormalizationError:
            # The identifier collector records the malformed anchor under
            # _skipped_identifiers; never write a second, looser profile ID.
            linkedin_id = None

    profile_pic = None
    if bp := crustdata.get("basic_profile"):
        profile_pic = bp.get("profile_picture_permalink")
    if not profile_pic and (pn := crustdata.get("professional_network")):
        profile_pic = pn.get("profile_picture_permalink")

    resolve_request = CandidateResolveRequest(
        source="signal",
        source_record_type=payload.source_record_type,
        source_record_id=payload.signal_candidate_id,
        identifiers=identifiers,
        profile=CandidateProfileInput(
            display_name=payload.display_name,
            props=profile_props,
            profile=crustdata,
            headline=headline_idx,
            location_raw=location_raw,
            skills=skills_idx,
            seniority_level=seniority,
            linkedin_url=payload.linkedinUrl,
            linkedin_id=linkedin_id,
            profile_picture_url=profile_pic,
        ),
        payload=source_payload,
        metadata=payload.source_metadata or {},
        source_url=payload.linkedinUrl,
        fetched_at=payload.profile_observed_at,
        tenant_id=payload.tenant_id,
        job_tags=_normalize_signal_tags(payload.tags),
    )

    if JWT_ENABLED and claims:
        tenant_id = claims.tenant_id
    else:
        tenant_id = payload.tenant_id

    pre_skipped = [
        SkippedIdentifier(
            identifier_type=s.get("identifier_type"),
            value=s.get("value"),
            reason=s.get("reason", ""),
        )
        for s in skipped
    ]
    if candidate_repo is None:
        raise HTTPException(status_code=503, detail="CandidateRepository not initialized")

    require_public_mirror = (payload.source_metadata or {}).get(
        "public_memory_surface"
    ) == "public_v1"
    with candidate_repo.serialized_source_record(
        tenant_id=tenant_id,
        source="signal",
        source_record_id=payload.signal_candidate_id,
    ):
        typed_record = candidate_repo.get_source_record(
            tenant_id=tenant_id,
            source="signal",
            source_record_type=payload.source_record_type,
            source_record_id=payload.signal_candidate_id,
        )
        latest_record = candidate_repo.get_latest_source_record(
            tenant_id=tenant_id,
            source="signal",
            source_record_id=payload.signal_candidate_id,
        )
        mirror_payload = payload
        if _signal_observation_is_not_newer(payload.profile_observed_at, latest_record):
            assert latest_record is not None
            if not _signal_observation_is_not_newer(payload.profile_observed_at, typed_record):
                typed_record = _upsert_resolve_source_record(
                    resolve_request,
                    candidate_id=latest_record.candidate_id,
                    tenant_id=tenant_id,
                )
            result = CandidateResolveResponse(
                candidate_id=latest_record.candidate_id,
                resolution_status="matched",
                source_record_id=(
                    typed_record.source_record_id
                    if typed_record is not None
                    else latest_record.source_record_id
                ),
                skipped_identifiers=pre_skipped,
                warnings=[
                    "profile_observed_at is not newer than stored evidence; "
                    "canonical profile was preserved"
                ],
            )
            try:
                mirror_payload = SignalCandidateResolveRequest.model_validate(latest_record.payload)
            except Exception as exc:
                logger.error(
                    "Durable Signal source record could not be reconstructed for global mirror",
                    extra_fields={
                        "signal_candidate_id": payload.signal_candidate_id,
                        "source_record_type": payload.source_record_type,
                        "error": str(exc),
                    },
                )
                if require_public_mirror:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "public memory mirror cannot reconstruct durable source evidence; "
                            "retry after source-record repair"
                        ),
                    ) from exc
                return result
        else:
            result = _execute_candidate_resolve(
                resolve_request,
                tenant_id=tenant_id,
                pre_skipped=pre_skipped,
            )

        # The tenant resolve above commits before mirroring. Keep the source
        # advisory lock through the mirror so concurrent retries cannot race the
        # public projection, while strict failures remain safely retryable.
        mirror_crustdata, mirror_headline, mirror_seniority, mirror_skills = _signal_mirror_fields(
            mirror_payload
        )
        _mirror_signal_candidate_to_global(
            payload=mirror_payload,
            result=result,
            tenant_id=tenant_id,
            crustdata=mirror_crustdata,
            headline_idx=mirror_headline,
            seniority=mirror_seniority,
            skills_idx=mirror_skills,
            require_public_mirror=require_public_mirror,
        )
        return result


@app.post(
    "/candidates/search/by-tags",
    response_model=CandidateSearchByTagsResponse,
    dependencies=[Depends(require_scope("kg:read"))],
)
def search_candidates_by_tags(
    payload: CandidateSearchByTagsRequest,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Search for candidates whose Signal source record tags overlap with the query tags.

    A candidate is returned when at least 70 % of the query tags are present in
    its stored Signal ``job_tags``. Results are ranked by overlap ratio descending.
    At most 100 candidates are returned.
    """
    if candidate_repo is None:
        raise HTTPException(status_code=503, detail="CandidateRepository not initialized")

    if JWT_ENABLED and claims:
        tenant_id = claims.tenant_id
    else:
        tenant_id = payload.tenant_id

    normalized_query_tags = _normalize_signal_tags(payload.tags)
    if not normalized_query_tags:
        return CandidateSearchByTagsResponse(results=[], query_tags=[], total=0)

    applied_limit = min(payload.limit, TAG_SEARCH_MAX_LIMIT)
    rows, total_matched = candidate_repo.search_candidates_by_signal_tags(
        normalized_query_tags,
        tenant_id=tenant_id,
        limit=applied_limit,
    )

    results = [
        CandidateTagSearchResult(
            candidate_id=row.candidate_id,
            display_name=row.display_name,
            primary_email=row.primary_email,
            signal_candidate_id=row.signal_source_record_id,
            stored_tags=row.stored_tags,
            matched_tags=sorted(set(normalized_query_tags) & set(row.stored_tags)),
            overlap_count=row.overlap_count,
            overlap_ratio=row.overlap_ratio,
            profile=row.profile,
        )
        for row in rows
    ]

    return CandidateSearchByTagsResponse(
        results=results,
        query_tags=normalized_query_tags,
        total=len(results),
        total_matched=total_matched,
        truncated=total_matched > len(results),
        applied_limit=applied_limit,
    )


@app.post(
    "/candidates/search/private",
    response_model=TenantPrivateCandidateSearchResponse,
    dependencies=[Depends(require_scope("kg:read"))],
)
def search_tenant_private_candidates(
    payload: TenantPrivateCandidateSearchRequest,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Recall only the requesting tenant's applicant/upload candidates.

    The response is an explicit typed projection. Raw candidate profiles,
    source payloads, resumes, contact data, and recruiting activity never
    cross this boundary. Signal remains the final ranking authority.
    """
    if candidate_repo is None:
        raise HTTPException(status_code=503, detail="CandidateRepository not initialized")

    tenant_id = claims.tenant_id if JWT_ENABLED and claims else payload.tenant_id
    if not tenant_id or not tenant_id.strip():
        raise HTTPException(status_code=400, detail="tenant identity is required")

    query_terms, normalized_skills = _private_search_terms(
        payload.query_text,
        payload.skills_any,
    )
    applied_limit = min(payload.limit, PRIVATE_SEARCH_MAX_LIMIT)
    rows, total_available = candidate_repo.search_tenant_private_candidates(
        tenant_id=tenant_id,
        query_terms=query_terms,
        skills_any=normalized_skills,
        limit=applied_limit,
    )
    results = [
        TenantPrivateCandidateSearchResult(
            candidate_id=row.candidate_id,
            global_candidate_id=row.global_candidate_id,
            display_name=row.display_name,
            linkedin_url=row.linkedin_url,
            linkedin_id=row.linkedin_id,
            headline=row.headline,
            location_raw=row.location_raw,
            skills=row.skills,
            seniority_level=row.seniority_level,
            keyword_score=row.keyword_score,
            skill_overlap_count=row.skill_overlap_count,
        )
        for row in rows
    ]
    return TenantPrivateCandidateSearchResponse(
        results=results,
        total=len(results),
        total_available=total_available,
        truncated=total_available > len(results),
        applied_limit=applied_limit,
    )


# deploy-trigger: 2026-07-28 — CI-skip recovery (docs workflow is Pages-only; excluded from this trigger).
# deploy-trigger: 2026-07-28 CI-skip recovery.
