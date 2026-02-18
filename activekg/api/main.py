from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, cast

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
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from activekg.api.admin_connectors import router as connectors_admin_router

# JWT authentication and rate limiting
from activekg.api.auth import JWT_ENABLED, JWTClaims, get_jwt_claims
from activekg.api.middleware import apply_rate_limit, get_tenant_context, require_rate_limit
from activekg.api.rate_limiter import RATE_LIMIT_ENABLED, get_identifier, rate_limiter
from activekg.common.env import env_str
from activekg.common.logger import clear_log_context, get_enhanced_logger, set_log_context
from activekg.common.metrics import get_redis_client, metrics
from activekg.common.validation import (
    AskRequest,
    EdgeCreate,
    HealthCheckResponse,
    KGSearchRequest,
    MetricsResponse,
    NodeBatchCreate,
    NodeCreate,
)
from activekg.connectors.cache_subscriber import get_subscriber_health
from activekg.connectors.webhooks import router as connectors_webhook_router
from activekg.embedding.queue import (
    enqueue_embedding_job,
    get_pending_count,
    queue_depth,
)
from activekg.engine.embedding_provider import EmbeddingProvider
from activekg.engine.llm_provider import (
    LLMProvider,
    build_strict_citation_prompt,
    calculate_confidence,
    extract_citation_numbers,
    filter_context_by_similarity,
)
from activekg.extraction.queue import (
    enqueue_extraction_job,
    extraction_queue_depth,
)
from activekg.graph.models import Edge, Node
from activekg.graph.repository import GraphRepository

# Prometheus observability
from activekg.observability import (
    get_metrics_handler,
    track_ask_request,
    track_embedding_health,
    track_search_request,
)
from activekg.observability.metrics import record_api_error
from activekg.refresh.scheduler import RefreshScheduler
from activekg.triggers.pattern_store import PatternStore
from activekg.triggers.trigger_engine import TriggerEngine

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


# Request models
class RotateKeysRequest(BaseModel):
    """Request model for connector key rotation endpoint."""

    providers: list[str] | None = None
    tenants: list[str] | None = None
    batch_size: int = 100
    dry_run: bool = False


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

# LLM provider for /ask endpoint (optional, falls back gracefully)
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")  # "openai", "groq", or "litellm"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")  # Groq's Llama 3.1 8B (ultra-fast)
LLM_ENABLED = os.getenv("LLM_ENABLED", "true").lower() == "true"
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
RUN_GCS_POLLER = os.getenv("RUN_GCS_POLLER", "true").lower() == "true"

# Hybrid routing: fast model for simple queries, fallback for complex/low-confidence
HYBRID_ROUTING_ENABLED = os.getenv("HYBRID_ROUTING_ENABLED", "false").lower() == "true"
ASK_FAST_BACKEND = os.getenv("ASK_FAST_BACKEND", "groq")
ASK_FAST_MODEL = os.getenv("ASK_FAST_MODEL", "llama-3.1-8b-instant")
ASK_FALLBACK_BACKEND = os.getenv("ASK_FALLBACK_BACKEND", "openai")
ASK_FALLBACK_MODEL = os.getenv("ASK_FALLBACK_MODEL", "gpt-4o-mini")

# Routing thresholds
ASK_ROUTER_TOPSIM = float(os.getenv("ASK_ROUTER_TOPSIM", "0.70"))  # Use fast if top_sim >= this
ASK_ROUTER_MINCONF = float(os.getenv("ASK_ROUTER_MINCONF", "0.60"))  # Fallback if conf < this

# Ask/Q&A tuning knobs (env configurable)
ASK_SIM_THRESHOLD = float(
    os.getenv("ASK_SIM_THRESHOLD", "0.30")
)  # similarity cutoff for gating and context
ASK_MAX_TOKENS = int(os.getenv("ASK_MAX_TOKENS", "256"))  # LLM token budget
ASK_MAX_SNIPPETS = int(os.getenv("ASK_MAX_SNIPPETS", "3"))  # max context snippets
ASK_SNIPPET_LEN = int(os.getenv("ASK_SNIPPET_LEN", "300"))  # chars per snippet

# Fast path uses smaller budget for speed
ASK_FAST_MAX_TOKENS = int(os.getenv("ASK_FAST_MAX_TOKENS", "192"))
ASK_FALLBACK_MAX_TOKENS = int(os.getenv("ASK_FALLBACK_MAX_TOKENS", "320"))

# Similarity gating thresholds (scale-aware: RRF vs raw cosine)
# RRF fused scores are in 0.01-0.04 range, cosine scores are 0.0-1.0
RRF_LOW_SIM_THRESHOLD = float(os.getenv("RRF_LOW_SIM_THRESHOLD", "0.01"))
RAW_LOW_SIM_THRESHOLD = float(os.getenv("RAW_LOW_SIM_THRESHOLD", "0.15"))

# Reranker configuration (ops control toggles)
ASK_USE_RERANKER = os.getenv("ASK_USE_RERANKER", "true").lower() == "true"
RERANK_SKIP_TOPSIM = float(os.getenv("RERANK_SKIP_TOPSIM", "0.80"))

MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE_BYTES", str(10 * 1024 * 1024)))


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
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


app = FastAPI(title="Active Graph KG", version=APP_VERSION)
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
    pattern_store = None
    trigger_engine = None
    scheduler: RefreshScheduler | None = None
    logger.warning("Running in TEST_MODE - DB connections deferred")
else:
    # Normal mode: eager initialization
    repo = GraphRepository(DSN, candidate_factor=WEIGHTED_SEARCH_CANDIDATE_FACTOR)
    embedder = EmbeddingProvider(backend=EMBEDDING_BACKEND, model_name=EMBEDDING_MODEL)
    pattern_store = PatternStore(DSN)
    trigger_engine = TriggerEngine(pattern_store, repo)
    scheduler = None

# Mount admin connectors router (minimal MVP)
app.include_router(connectors_admin_router)
app.include_router(connectors_webhook_router)

# Initialize LLM provider(s)
llm = None
llm_fast = None
llm_fallback = None

if LLM_ENABLED:
    try:
        if HYBRID_ROUTING_ENABLED:
            # Initialize both fast and fallback models for hybrid routing
            llm_fast = LLMProvider(backend=ASK_FAST_BACKEND, model=ASK_FAST_MODEL)
            llm_fallback = LLMProvider(backend=ASK_FALLBACK_BACKEND, model=ASK_FALLBACK_MODEL)
            llm = llm_fallback  # Default to fallback for backward compat
            logger.info(
                "Hybrid routing enabled",
                extra_fields={
                    "fast_backend": ASK_FAST_BACKEND,
                    "fast_model": ASK_FAST_MODEL,
                    "fallback_backend": ASK_FALLBACK_BACKEND,
                    "fallback_model": ASK_FALLBACK_MODEL,
                    "router_topsim": ASK_ROUTER_TOPSIM,
                    "router_minconf": ASK_ROUTER_MINCONF,
                },
            )
        else:
            # Single LLM mode
            llm = LLMProvider(backend=LLM_BACKEND, model=LLM_MODEL)
            logger.info(
                "LLM provider enabled", extra_fields={"backend": LLM_BACKEND, "model": LLM_MODEL}
            )
    except Exception as e:
        logger.warning(
            "LLM provider initialization failed, /ask endpoint disabled",
            extra_fields={"error": str(e)},
        )
        LLM_ENABLED = False

# In-memory LRU cache for /ask responses (context-aware, thread-safe)
ASK_CACHE_MAX = int(os.getenv("ASK_CACHE_MAX", "512"))
ASK_CACHE_TTL = int(os.getenv("ASK_CACHE_TTL", "600"))  # seconds
_ASK_CACHE: OrderedDict[str, tuple[float, dict]] = OrderedDict()
_ASK_CACHE_LOCK = threading.RLock()  # Reentrant lock for thread safety


def _ask_cache_get(key: str) -> dict | None:
    """Thread-safe cache get with TTL expiration."""
    with _ASK_CACHE_LOCK:
        now = time.time()
        entry = _ASK_CACHE.get(key)
        if not entry:
            return None
        ts, value = entry
        if now - ts > ASK_CACHE_TTL:
            _ASK_CACHE.pop(key, None)
            return None
        _ASK_CACHE.move_to_end(key)
        return value


def _ask_cache_put(key: str, value: dict) -> None:
    """Thread-safe cache put with LRU eviction."""
    with _ASK_CACHE_LOCK:
        _ASK_CACHE[key] = (time.time(), value)
        _ASK_CACHE.move_to_end(key)
        while len(_ASK_CACHE) > ASK_CACHE_MAX:
            try:
                _ASK_CACHE.popitem(last=False)
            except KeyError:
                break


@app.on_event("startup")
def startup_event():
    """Initialize system on startup."""
    logger.info(
        "Active Graph KG startup",
        extra_fields={
            "version": APP_VERSION,
            "weighted_search_candidate_factor": WEIGHTED_SEARCH_CANDIDATE_FACTOR,
            "run_scheduler": RUN_SCHEDULER,
            "run_gcs_poller": RUN_GCS_POLLER,
            "rrf_low_sim_threshold": RRF_LOW_SIM_THRESHOLD,
            "raw_low_sim_threshold": RAW_LOW_SIM_THRESHOLD,
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

    # Quick Win 1: Fail-fast KEK validation
    try:
        from activekg.connectors.encryption import get_encryption

        enc = get_encryption()
        logger.info(f"KEK validation passed (active version: {enc.active_version})")
    except Exception as e:
        logger.error(f"KEK validation failed: {e}")
        raise RuntimeError(f"Failed to load connector encryption keys: {e}")

    # Quick Win 3: Cache warmup for connector configs
    try:
        import os

        from activekg.connectors.config_store import get_config_store

        if os.getenv("ACTIVEKG_DSN"):
            store = get_config_store()
            configs = store.list_all()
            logger.info(f"Connector config cache warmup: {len(configs)} configs preloaded")
    except Exception as e:
        logger.warning(f"Connector config cache warmup failed (non-critical): {e}")

    # Phase 2: Start cache subscriber for multi-worker cache invalidation
    try:
        import os

        from activekg.connectors.cache_subscriber import start_subscriber
        from activekg.connectors.config_store import get_config_store

        redis_url = os.getenv("REDIS_URL")
        activekg_dsn = os.getenv("ACTIVEKG_DSN")
        if redis_url and activekg_dsn:
            store = get_config_store()
            start_subscriber(redis_url, store)
            logger.info("Cache subscriber started for multi-worker cache invalidation")
        else:
            logger.info("Cache subscriber disabled (REDIS_URL or ACTIVEKG_DSN not set)")
    except Exception as e:
        logger.warning(f"Cache subscriber failed to start (non-critical): {e}")

    # Auto-enable vector index if not present
    repo.ensure_vector_index()

    # Start refresh scheduler (only if RUN_SCHEDULER=true)
    global scheduler
    if RUN_SCHEDULER:
        try:
            scheduler = RefreshScheduler(
                repo, embedder, trigger_engine=trigger_engine, gcs_poller_enabled=RUN_GCS_POLLER
            )
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


@app.get("/health", response_model=HealthCheckResponse)
def health() -> HealthCheckResponse:
    now = datetime.now(timezone.utc).isoformat()
    return HealthCheckResponse(
        status="ok",
        timestamp=now,
        version=APP_VERSION,
        uptime_seconds=0.0,
        components={"db": {"status": "unknown"}},
        llm_backend=LLM_BACKEND if LLM_ENABLED and llm else None,
        llm_model=LLM_MODEL if LLM_ENABLED and llm else None,
    )


@app.get("/_admin/connectors/cache/health", response_model=None)
def connector_cache_health(claims: JWTClaims | None = Depends(get_jwt_claims)) -> dict[str, Any]:
    """Health endpoint for connector cache subscriber.

    Security:
        - When JWT is enabled, require authenticated token (no specific scope required for health check)
        - When JWT is disabled (dev mode), allow access

    Returns:
        Status dict with subscriber health information
    """
    subscriber_health = get_subscriber_health()

    if subscriber_health is None:
        # Subscriber not running
        return {"status": "degraded", "subscriber": None}

    # Subscriber is running - check if connected
    connected = subscriber_health.get("connected", False)
    status = "ok" if connected else "degraded"

    return {"status": status, "subscriber": subscriber_health}


@app.post("/_admin/connectors/rotate_keys", response_model=None)
def connector_rotate_keys(
    request: RotateKeysRequest, claims: JWTClaims | None = Depends(get_jwt_claims)
) -> dict[str, Any]:
    """Rotate encryption keys for connector configs.

    Selects rows where key_version != ACTIVE_VERSION, decrypts with old key,
    re-encrypts with active key, and updates key_version.

    Security:
        - When JWT is enabled, require authenticated token
        - When JWT is disabled (dev mode), allow access

    Args:
        request: Rotation parameters (providers, tenants, batch_size, dry_run)

    Returns:
        Summary dict with rotation results
    """
    try:
        from activekg.connectors.config_store import get_config_store

        store = get_config_store()

        # Call rotate_keys method
        result = store.rotate_keys(
            providers=request.providers,
            tenants=request.tenants,
            batch_size=request.batch_size,
            dry_run=request.dry_run,
        )

        return cast("dict[str, Any]", result)

    except Exception as e:
        logger.error(f"Key rotation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Key rotation failed: {str(e)}")


@app.get("/_admin/security/limits", response_model=None)
def get_security_limits(claims: JWTClaims | None = Depends(get_jwt_claims)) -> dict[str, Any]:
    """Get configured security limits and SSRF protection settings.

    Returns current configuration for:
    - SSRF protection (URL allowlist, blocked IP ranges)
    - File access controls (allowed directories, size limits)
    - Request body size limits

    Security:
        - When JWT is enabled, requires authenticated token
        - No admin scope required (read-only configuration)
    """
    import os

    # Parse URL allowlist
    allow_raw = os.getenv("ACTIVEKG_URL_ALLOWLIST", "")
    allow_hosts = [h.strip() for h in allow_raw.split(",") if h.strip()]

    # Parse file basedirs
    basedirs_raw = os.getenv("ACTIVEKG_FILE_BASEDIRS", "")
    basedirs = [d.strip() for d in basedirs_raw.split(",") if d.strip()]
    if not basedirs:
        basedirs = ["<current working directory>"]

    return {
        "ssrf_protection": {
            "enabled": True,
            "url_allowlist": allow_hosts if allow_hosts else ["*any domain*"],
            "blocked_ip_ranges": [
                "127.0.0.0/8 (localhost)",
                "10.0.0.0/8 (private)",
                "172.16.0.0/12 (private)",
                "192.168.0.0/16 (private)",
                "169.254.0.0/16 (link-local / AWS metadata)",
                "224.0.0.0/4 (multicast)",
            ],
            "max_fetch_bytes": int(os.getenv("ACTIVEKG_MAX_FETCH_BYTES", "10485760")),
            "max_fetch_mb": round(
                int(os.getenv("ACTIVEKG_MAX_FETCH_BYTES", "10485760")) / (1024 * 1024), 2
            ),
            "fetch_timeout_seconds": float(os.getenv("ACTIVEKG_FETCH_TIMEOUT", "10")),
            "allowed_content_types": ["text/*", "application/json"],
        },
        "file_access": {
            "enabled": True,
            "allowed_base_directories": basedirs,
            "symlinks_blocked": True,
            "max_file_bytes": int(os.getenv("ACTIVEKG_MAX_FILE_BYTES", "1048576")),
            "max_file_mb": round(
                int(os.getenv("ACTIVEKG_MAX_FILE_BYTES", "1048576")) / (1024 * 1024), 2
            ),
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


@app.post("/debug/search_explain", response_model=None)
def debug_search_explain(
    query: str = Body(..., embed=True),
    use_hybrid: bool = Body(False, embed=True),
    top_k: int = Body(5, embed=True),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Debug endpoint for detailed search result triage.

    Returns top-k results with similarity scores, snippets, and metadata
    to help diagnose retrieval issues and tune thresholds.

    Security:
        - When JWT is enabled, require admin:refresh scope.
        - When JWT is disabled (dev mode), allow access.

    Args:
        query: Search query text
        use_hybrid: Whether to use hybrid BM25+vector search
        top_k: Number of results to return (max 20)

    Returns:
        {
          "query": str,
          "mode": "vector" | "hybrid",
          "result_count": int,
          "results": [
            {
              "node_id": str,
              "similarity": float,
              "classes": List[str],
              "snippet": str (first 300 chars of props.text),
              "metadata": dict,
              "has_embedding": bool,
              "has_text_search": bool
            }
          ],
          "threshold_info": {
            "recommended_min": float,
            "recommended_max": float,
            "top_similarity": float,
            "bottom_similarity": float
          }
        }
    """
    assert repo is not None, "GraphRepository not initialized"
    assert embedder is not None, "EmbeddingProvider not initialized"
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

    # Limit top_k to prevent abuse
    top_k = min(max(1, top_k), 20)

    try:
        # Generate query embedding
        query_embedding = embedder.encode([query])[0]

        # Perform search
        if use_hybrid:
            # Hybrid search
            search_results = repo.hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                top_k=top_k,
                tenant_id=tenant_id,
                use_reranker=ASK_USE_RERANKER,
            )
            mode = "hybrid"
        else:
            # Vector-only search
            search_results = repo.vector_search(
                query_embedding=query_embedding, top_k=top_k, tenant_id=tenant_id
            )
            mode = "vector"

        # Determine score type for explanatory metadata
        rrf_enabled = os.getenv("HYBRID_RRF_ENABLED", "true").lower() == "true"
        if use_hybrid:
            score_type = "rrf_fused" if rrf_enabled else "weighted_fusion"
            score_range = "0.01-0.04 (low)" if rrf_enabled else "0.0-1.0 (cosine-like)"
        else:
            score_type = "cosine"
            score_range = "0.0-1.0"

        # Format results with detailed info
        detailed_results = []
        for node, similarity in search_results:
            node_id = str(node.id)
            classes = node.classes or []
            props = node.props or {}
            metadata = node.metadata or {}

            # Extract snippet (first 300 chars of text)
            text = props.get("text", "")
            snippet = text[:300] + "..." if len(text) > 300 else text

            # Check for embedding and text search
            has_embedding = node.embedding is not None
            # text_search_vector presence not directly in result, infer from hybrid success
            has_text_search = True if use_hybrid else None

            detailed_results.append(
                {
                    "node_id": node_id,
                    "similarity": round(similarity, 4),
                    "score_type": score_type,
                    "classes": classes,
                    "snippet": snippet,
                    "metadata": metadata,
                    "has_embedding": has_embedding,
                    "has_text_search": has_text_search,
                }
            )

        # Calculate threshold recommendations
        similarities = [r["similarity"] for r in detailed_results if r["similarity"] > 0]

        if similarities:
            top_sim = max(similarities)
            bottom_sim = min(similarities)
            # Recommended min: 20% below bottom similarity (but >= 0.15)
            recommended_min = max(0.15, bottom_sim * 0.8)
            # Recommended max: 10% below top similarity
            recommended_max = top_sim * 0.9
        else:
            top_sim = 0.0
            bottom_sim = 0.0
            recommended_min = 0.15
            recommended_max = 0.30

        return {
            "query": query,
            "mode": mode,
            "score_type": score_type,
            "score_range": score_range,
            "result_count": len(detailed_results),
            "results": detailed_results,
            "threshold_info": {
                "recommended_min": round(recommended_min, 2),
                "recommended_max": round(recommended_max, 2),
                "top_similarity": round(top_sim, 4),
                "bottom_similarity": round(bottom_sim, 4),
            },
            "scoring_notes": {
                "rrf_fused": "RRF scores range 0.01-0.04 (rank-based fusion of vector+BM25)",
                "weighted_fusion": "Weighted scores range 0.0-1.0 (linear combination of vector+BM25)",
                "cosine": "Cosine similarity range 0.0-1.0 (vector-only)",
            },
        }

    except Exception as e:
        logger.error("/debug/search_explain failed", extra_fields={"error": str(e), "query": query})
        raise HTTPException(status_code=500, detail=f"search_explain error: {str(e)}")


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
def debug_intent(q: str):
    """Debug endpoint to test intent detection without running full /ask.

    Example: GET /debug/intent?q=What%20ML%20frameworks%20does%20the%20position%20require

    Returns: {
        "query": str,
        "normalized": str,
        "intent_type": str | None,
        "params": dict | None
    }
    """
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
        logger.error("/debug/intent failed", extra_fields={"error": str(e), "query": q})
        raise HTTPException(status_code=500, detail=f"intent detection error: {str(e)}")


@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    """Get metrics in JSON format."""
    assert repo is not None, "GraphRepository not initialized"
    assert embedder is not None, "EmbeddingProvider not initialized"
    return metrics.get_all_metrics()


@app.get("/prometheus")
async def prometheus_metrics():
    """Get metrics in Prometheus exposition format using prometheus_client.

    Provides detailed metrics for observability:
    - Request counters by score_type and rejection status
    - Gating score histograms
    - Citation quality metrics
    - Latency histograms
    - Embedding health gauges

    Format: https://prometheus.io/docs/instrumenting/exposition_formats/
    """
    if not METRICS_ENABLED:
        return PlainTextResponse(content="# Metrics disabled\n", status_code=503)

    # Use the dedicated Prometheus handler from activekg.observability
    handler = get_metrics_handler()
    return await handler()


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


@app.post("/nodes", response_model=None)
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


@app.post("/nodes/batch", response_model=None)
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


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    """Minimal self-serve console for demo and testing."""
    return """
<!doctype html>
<html>
  <head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>actvgraph-kg Demo Console</title>
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; }
      .row { display: flex; gap: 16px; flex-wrap: wrap; }
      .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; width: 360px; }
      h2 { margin: 8px 0; font-size: 18px; }
      input, textarea, select, button { font: inherit; padding: 6px 8px; }
      ul { padding-left: 20px; }
      code { background: #f3f4f6; padding: 2px 4px; border-radius: 4px; }
      small { color: #6b7280; }
      .res { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; max-height: 200px; overflow: auto; background: #f9fafb; padding: 8px; border-radius: 6px; }
    </style>
  </head>
  <body>
    <h1>actvgraph-kg Demo Console</h1>
    <p><small>Postgres + pgvector · Drift-aware refresh · Lineage · Semantic triggers</small></p>
    <div class="row">
      <div class="card">
        <h2>Search</h2>
        <input id="q" placeholder="query" style="width: 100%" />
        <label><input type="checkbox" id="weighted" checked /> weighted</label>
        <button onclick="doSearch()">Search</button>
        <div class="res" id="searchRes"></div>
      </div>

      <div class="card">
        <h2>Triggers</h2>
        <input id="tname" placeholder="name (e.g., senior_java)" style="width:100%" />
        <textarea id="texample" rows="3" placeholder="example text" style="width:100%"></textarea>
        <button onclick="addTrigger()">Register</button>
        <button onclick="listTriggers()">List</button>
        <div class="res" id="trigRes"></div>
      </div>

      <div class="card">
        <h2>Events</h2>
        <button onclick="listEvents()">List recent</button>
        <div class="res" id="eventsRes"></div>
      </div>

      <div class="card">
        <h2>Lineage & Refresh</h2>
        <input id="nodeId" placeholder="node id" style="width:100%" />
        <button onclick="showLineage()">Show Lineage</button>
        <button onclick="refreshNode()">Manual Refresh</button>
        <div class="res" id="lineageRes"></div>
      </div>

      <div class="card">
        <h2>Anomalies</h2>
        <button onclick="listAnomalies()">Detect</button>
        <div class="res" id="anRes"></div>
      </div>
    </div>

    <script>
      async function doSearch() {
        const q = document.getElementById('q').value;
        const weighted = document.getElementById('weighted').checked;
        const resEl = document.getElementById('searchRes');
        resEl.textContent = 'Searching...';
        try {
          const r = await fetch('/search', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ query: q, top_k: 10, use_weighted_score: weighted }) });
          const data = await r.json();
          const lines = (data.results || []).map((it, idx) => `${idx+1}. ${it.id}  sim=${it.similarity}`);
          resEl.textContent = lines.join('\n') || 'No results';
        } catch (e) {
          resEl.textContent = 'Error: ' + e;
        }
      }

      async function addTrigger() {
        const name = document.getElementById('tname').value;
        const example = document.getElementById('texample').value;
        const resEl = document.getElementById('trigRes');
        resEl.textContent = 'Registering...';
        try {
          const r = await fetch('/triggers', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ name: name, example_text: example }) });
          const data = await r.json();
          resEl.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          resEl.textContent = 'Error: ' + e;
        }
      }

      async function listTriggers() {
        const resEl = document.getElementById('trigRes');
        resEl.textContent = 'Loading...';
        try {
          const r = await fetch('/triggers');
          const data = await r.json();
          resEl.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          resEl.textContent = 'Error: ' + e;
        }
      }

      async function listEvents() {
        const resEl = document.getElementById('eventsRes');
        resEl.textContent = 'Loading...';
        try {
          const r = await fetch('/events?limit=50');
          const data = await r.json();
          resEl.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          resEl.textContent = 'Error: ' + e;
        }
      }

      async function showLineage() {
        const id = document.getElementById('nodeId').value;
        const resEl = document.getElementById('lineageRes');
        resEl.textContent = 'Loading...';
        try {
          const r = await fetch('/lineage/' + encodeURIComponent(id));
          const data = await r.json();
          resEl.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          resEl.textContent = 'Error: ' + e;
        }
      }

      async function refreshNode() {
        const id = document.getElementById('nodeId').value;
        const resEl = document.getElementById('lineageRes');
        resEl.textContent = 'Refreshing...';
        try {
          const r = await fetch('/nodes/' + encodeURIComponent(id) + '/refresh', { method: 'POST' });
          const data = await r.json();
          resEl.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          resEl.textContent = 'Error: ' + e;
        }
      }

      async function listAnomalies() {
        const resEl = document.getElementById('anRes');
        resEl.textContent = 'Detecting...';
        try {
          const r = await fetch('/admin/anomalies');
          const data = await r.json();
          resEl.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          resEl.textContent = 'Error: ' + e;
        }
      }
    </script>
  </body>
</html>
    """


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


@app.post("/upload", response_model=None)
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


@app.post("/search", response_model=None)
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

            text_snippet = raw_text or ""
            if isinstance(text_snippet, str) and len(text_snippet) > 300:
                text_snippet = text_snippet[:300]

            formatted_results.append(
                {
                    "id": node.id,
                    "classes": node.classes,
                    "props": node.props,
                    "payload_ref": node.payload_ref,
                    "metadata": node.metadata,
                    "similarity": round(similarity, 4),
                    "text": text_snippet,
                }
            )

        # Track Prometheus metrics (if enabled)
        if METRICS_ENABLED:
            latency_ms = (time.time() - start_time) * 1000

            # Determine mode and score_type
            if search_request.use_hybrid:
                mode = "hybrid"
                rrf_enabled = os.getenv("HYBRID_RRF_ENABLED", "true").lower() == "true"
                score_type = "rrf_fused" if rrf_enabled else "weighted_fusion"
                reranked = search_request.use_reranker
            else:
                mode = "vector"
                score_type = "weighted_fusion" if search_request.use_weighted_score else "cosine"
                reranked = False

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


def filter_by_must_have_terms(
    results: list[tuple[Node, float]], must_have_terms: list[str] | None
) -> list[tuple[Node, float]]:
    """Filter results to only include nodes containing required terms.

    Args:
        results: List of (Node, similarity) tuples
        must_have_terms: List of terms that must appear in node text (case-insensitive)

    Returns:
        Filtered results, or original if no matches (graceful fallback)
    """
    if not must_have_terms:
        return results

    filtered = []
    for node, score in results:
        text = (node.props or {}).get("text", "").lower()
        # Node must contain at least one of the must-have terms
        if any(term.lower() in text for term in must_have_terms):
            filtered.append((node, score))

    # Graceful fallback: if filter removes everything, return original
    return filtered if filtered else results


def should_use_fast_path(
    question: str, top_similarity: float, result_count: int
) -> tuple[bool, str]:
    """Decide whether to use fast or fallback LLM based on query characteristics.

    Args:
        question: User question
        top_similarity: Similarity score of top search result
        result_count: Number of search results found

    Returns:
        Tuple of (use_fast: bool, reason: str)
    """
    if not HYBRID_ROUTING_ENABLED:
        return (False, "hybrid_disabled")

    # Skip complex queries that need reasoning
    complex_keywords = ["explain", "why", "how", "compare", "difference", "analyze"]
    if any(kw in question.lower() for kw in complex_keywords):
        return (False, "complex_query")

    # Skip long queries (likely complex)
    if len(question.split()) > 20:
        return (False, "long_query")

    # Use fast path if high confidence (top similarity above threshold)
    if top_similarity >= ASK_ROUTER_TOPSIM:
        return (True, f"high_confidence_sim={top_similarity:.3f}")

    # Use fast path for listing queries with multiple good results
    listing_patterns = [
        r"^(who|what|which|list|show|find)\s+(has|are|have|with|need)",
        r"^list\s+",
        r"^show\s+(me\s+)?(all|the)",
    ]
    import re

    if result_count >= 3 and any(re.match(pat, question.lower()) for pat in listing_patterns):
        return (True, "listing_query")

    # Default: use fallback for safety
    return (False, f"low_confidence_sim={top_similarity:.3f}")


@app.post("/ask")
async def ask_question(
    http_request: Request,
    http_response: Response,
    request: AskRequest,
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """LLM-powered Q&A with grounded citations from knowledge graph.

    Uses structured KG for retrieval, LLM for natural language answer generation.
    All facts are cited with node IDs and lineage chains (Zhu et al., 2023 design).

    Example:
        POST /ask {"question": "What are the best ML engineering candidates?"}

    Returns:
        {
            "answer": "Based on recent resumes:\\n1. Jane Doe (5yrs PyTorch) [0]\\n2. ...",
            "citations": [
                {
                    "node_id": "resume_123",
                    "classes": ["Resume"],
                    "drift_score": 0.08,
                    "age_days": 1.2,
                    "lineage": [{"ancestor": "linkedin_profile_456", "depth": 1}]
                }
            ],
            "confidence": 0.92,
            "metadata": {"searched_nodes": 5, "cited_nodes": 3}
        }
    """
    assert repo is not None, "GraphRepository not initialized"
    assert embedder is not None, "EmbeddingProvider not initialized"
    if not LLM_ENABLED or llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM provider not enabled. Set LLM_ENABLED=true and configure LLM_BACKEND/LLM_MODEL.",
        )

    # Extract tenant context from JWT (secure) or request body (dev mode only)
    tenant_id, actor_id, actor_type = get_tenant_context(
        http_request, claims, allow_override=not JWT_ENABLED
    )

    # In dev mode, allow tenant_id override from request body
    if not JWT_ENABLED and request.tenant_id:
        tenant_id = request.tenant_id

    # Apply rate limiting with concurrency control
    request_id = await apply_rate_limit(
        http_request,
        http_response,
        endpoint="ask",
        tenant_id=tenant_id,
        check_concurrency=True,  # Max 3 concurrent /ask per tenant
    )

    try:
        # Start timing for latency tracking
        start_time = time.time()

        # 0. Intent detection for structured queries
        intent_type, intent_params = detect_intent(request.question)
        # Ensure intent_params is not None for .get() calls
        if intent_params is None:
            intent_params = {}
        structured_results: list[Any] = []

        # TRACK 1: Extract class filtering and term gating from intent
        classes_filter = None
        must_have_terms = None
        if intent_type in ("entity_job", "entity_resume", "entity_article"):
            classes_filter = intent_params.get("expected_classes")
            must_have_terms = intent_params.get("must_have_terms")

        if intent_type == "open_positions":
            # Use structured query for open positions
            role_terms = intent_params.get("role_terms")
            # Normalize role_terms defensively to avoid ambiguous truth checks
            try:
                import numpy as _np

                if isinstance(role_terms, _np.ndarray):
                    role_terms = role_terms.tolist()
            except Exception:
                pass
            if role_terms is not None and not isinstance(role_terms, list):
                role_terms = [str(role_terms)]
            structured_results = repo.list_open_positions(
                role_terms=role_terms,
                limit=10,
                tenant_id=tenant_id,  # From JWT, not request body
            )
            logger.info(
                "Intent detected: open_positions",
                extra_fields={"role_terms": role_terms, "results": len(structured_results)},
            )

        elif intent_type == "performance_issues":
            # Use structured query for performance issues
            lookback_days = intent_params.get("lookback_days", 30)
            structured_results = repo.list_performance_issues(
                lookback_days=lookback_days,
                limit=10,
                tenant_id=tenant_id,  # From JWT, not request body
            )
            logger.info(
                "Intent detected: performance_issues",
                extra_fields={"lookback_days": lookback_days, "results": len(structured_results)},
            )

        # 1. Hybrid search with BM25+vector fusion and reranking
        query_embedding = embedder.encode([request.question])[0]

        # Determine if we should use cross-encoder reranking
        # Skip reranking for: structured intents (bypass hybrid), small result sets, ultra-low latency mode
        intent_detected = bool(intent_type and intent_type != "search")
        skip_rerank = (
            intent_detected or not ASK_USE_RERANKER
        )  # Skip for structured queries or if master toggle disabled

        # Use hybrid search for better retrieval (50 candidates → rerank to top 20)
        logger.info(
            f"Calling hybrid_search with classes_filter={classes_filter}, tenant_id={tenant_id}"
        )
        try:
            hybrid_results = repo.hybrid_search(
                query_text=request.question,
                query_embedding=query_embedding,
                top_k=20,  # Increased from 5 to get more candidates
                classes_filter=classes_filter,  # TRACK 1: Filter by node class
                tenant_id=tenant_id,  # From JWT, not request body
                use_reranker=not skip_rerank,  # Skip rerank for structured queries; gate on hybrid_score
                rerank_skip_threshold=RERANK_SKIP_TOPSIM,  # Use env-configurable threshold
            )
            # Fallback: if hybrid returns 0, try vector-only to avoid empty context
            if not hybrid_results:
                hybrid_results = repo.vector_search(
                    query_embedding=query_embedding,
                    top_k=request.max_results or 5,
                    classes_filter=classes_filter,  # TRACK 1: Filter by node class
                    tenant_id=tenant_id,
                    use_weighted_score=request.use_weighted_score,
                )
        except Exception:
            # Fallback if hybrid is unavailable (e.g., missing text_search_vector)
            hybrid_results = repo.vector_search(
                query_embedding=query_embedding,
                top_k=request.max_results or 5,
                classes_filter=classes_filter,  # TRACK 1: Filter by node class
                tenant_id=tenant_id,  # From JWT, not request body
                use_weighted_score=request.use_weighted_score,
            )

        # Merge structured and hybrid results (structured results take priority)
        # Use len() to avoid ambiguous array truthiness
        if len(structured_results) > 0:
            # De-duplicate: use structured results first, then add hybrid results not already in structured
            seen_ids = {node.id for node, _ in structured_results}
            for node, score in hybrid_results:
                if node.id not in seen_ids:
                    structured_results.append((node, score))
                    if len(structured_results) >= 20:
                        break
            results = structured_results
        else:
            results = hybrid_results

        # TRACK 1: Apply must-have term gating for entity-typed queries with logging
        gating_before_cnt = None
        gating_after_cnt = None
        gating_relaxed = None
        if intent_type in ("entity_job", "entity_resume", "entity_article"):
            try:
                before_cnt = len(results)
                after_cnt = before_cnt
                relaxed = False

                if must_have_terms:
                    # Compute matches
                    matches = []
                    for node, score in results:
                        txt = (node.props or {}).get("text", "").lower()
                        if any(term.lower() in txt for term in must_have_terms):
                            matches.append((node, score))
                    after_cnt = len(matches)
                    if after_cnt == 0:
                        # One-pass relax: keep original results
                        relaxed = True
                        filtered = results
                    else:
                        filtered = matches
                else:
                    filtered = results

                # Emit gating log
                logger.info(
                    "term_gating",
                    extra_fields={
                        "intent": intent_type,
                        "before": before_cnt,
                        "after": after_cnt,
                        "relaxed": relaxed,
                        "terms": must_have_terms[:5] if must_have_terms else None,
                    },
                )

                results = filtered
                gating_before_cnt = before_cnt
                gating_after_cnt = after_cnt
                gating_relaxed = relaxed
            except Exception:
                # Fallback to existing filter function on any error
                results = filter_by_must_have_terms(results, must_have_terms)

        if not results:
            # Determine gating score type for consistent metadata
            rrf_enabled = os.getenv("HYBRID_RRF_ENABLED", "true").lower() == "true"
            gating_score_type = "rrf_fused" if rrf_enabled else "cosine"
            response_data = {
                "answer": "I couldn't find relevant information in the knowledge graph to answer your question.",
                "citations": [],
                "confidence": 0.0,
                "metadata": {
                    "searched_nodes": 0,
                    "cited_nodes": 0,
                    "filtered_nodes": 0,
                    "gating_score": 0.0,
                    "gating_score_type": gating_score_type,
                },
            }
            # Track metrics
            if METRICS_ENABLED:
                latency_ms = (time.time() - start_time) * 1000
                track_ask_request(
                    gating_score=0.0,
                    gating_score_type=gating_score_type,
                    cited_nodes=0,
                    latency_ms=latency_ms,
                    rejected=True,
                    rejection_reason="no_results",
                    reranked=False,
                )
            return response_data

        # 2. Low-similarity guardrail with fallback policy
        # Ensure top_similarity is Python float, not numpy scalar (avoid ambiguous array truthiness)
        top_similarity = float(results[0][1]) if results else 0.0
        # Preserve true cosine similarity for the top-ranked result (if embedding is available)
        top_vector_similarity = 0.0
        max_vector_similarity = 0.0
        try:
            if results:
                top_node = results[0][0]
                if top_node.embedding is not None:
                    top_vector_similarity = float(top_node.embedding @ query_embedding)
                # Track max cosine similarity across candidate results for debugging
                for node, _score in results:
                    if node.embedding is None:
                        continue
                    sim = float(node.embedding @ query_embedding)
                    if sim > max_vector_similarity:
                        max_vector_similarity = sim
        except Exception:
            top_vector_similarity = 0.0
            max_vector_similarity = 0.0

        # Determine gating score type based on fusion mode
        rrf_enabled = os.getenv("HYBRID_RRF_ENABLED", "true").lower() == "true"
        gating_score_type = "rrf_fused" if rrf_enabled else "cosine"

        # Hard reject if similarity is extremely low
        # When RRF is enabled, scores are much lower (0.01-0.04 range), so use a lower threshold
        extremely_low_threshold = RRF_LOW_SIM_THRESHOLD if rrf_enabled else RAW_LOW_SIM_THRESHOLD
        if top_similarity < extremely_low_threshold:
            response_data = {
                "answer": "I don't have enough information to answer this question confidently.",
                "citations": [],
                "confidence": 0.2,
                "metadata": {
                    "searched_nodes": len(results),
                    "cited_nodes": 0,
                    "filtered_nodes": 0,
                    "top_similarity": round(top_similarity, 3),
                    "top_vector_similarity": round(top_vector_similarity, 3),
                    "max_vector_similarity": round(max_vector_similarity, 3),
                    "gating_score": round(top_similarity, 3),
                    "gating_score_type": gating_score_type,
                    "reason": "extremely_low_similarity",
                },
            }
            # Track metrics
            if METRICS_ENABLED:
                latency_ms = (time.time() - start_time) * 1000
                track_ask_request(
                    gating_score=top_similarity,
                    gating_score_type=gating_score_type,
                    cited_nodes=0,
                    latency_ms=latency_ms,
                    rejected=True,
                    rejection_reason="extremely_low_similarity",
                    reranked=not skip_rerank,
                )
            return response_data

        # Soft fallback: if below threshold but above 0.15, proceed with top-1 only and capped confidence
        low_sim_fallback = top_similarity < ASK_SIM_THRESHOLD
        if low_sim_fallback:
            # Limit to top-1 result only for low-confidence queries
            results = results[:1]

        # 2.5. Ambiguity gating (loosened): if top result isn't significantly better than 3rd, results are ambiguous
        # Skip for structured queries (intent-based) since they're already pre-filtered for relevance
        if len(results) >= 3 and not low_sim_fallback and intent_type is None:
            kth_similarity = results[2][1]  # 3rd result (0-indexed)
            if top_similarity - kth_similarity < 0.02:
                response_data = {
                    "answer": "I don't have enough information to answer this question confidently.",
                    "citations": [],
                    "confidence": 0.2,
                    "metadata": {
                        "searched_nodes": len(results),
                        "cited_nodes": 0,
                        "filtered_nodes": 0,
                        "top_similarity": round(top_similarity, 3),
                        "top_vector_similarity": round(top_vector_similarity, 3),
                        "max_vector_similarity": round(max_vector_similarity, 3),
                        "kth_similarity": round(kth_similarity, 3),
                        "gating_score": round(top_similarity, 3),
                        "gating_score_type": gating_score_type,
                        "ambiguity_reason": "top-k gap too small",
                    },
                }
                # Track metrics
                if METRICS_ENABLED:
                    latency_ms = (time.time() - start_time) * 1000
                    track_ask_request(
                        gating_score=top_similarity,
                        gating_score_type=gating_score_type,
                        cited_nodes=0,
                        latency_ms=latency_ms,
                        rejected=True,
                        rejection_reason="ambiguity",
                        reranked=not skip_rerank,
                    )
                return response_data

        # 3. Filter context by similarity threshold and cap to top-3 for latency
        filtered_results = filter_context_by_similarity(
            results,
            similarity_threshold=ASK_SIM_THRESHOLD,
            min_results=2,
            max_results=min(request.max_results or ASK_MAX_SNIPPETS, ASK_MAX_SNIPPETS),
        )

        # 3. Build context from filtered nodes
        context_items = []
        for i, (node, similarity) in enumerate(filtered_results):
            # Calculate age in days
            age_days = None
            if node.last_refreshed:
                age_seconds = (datetime.now(timezone.utc) - node.last_refreshed).total_seconds()
                age_days = age_seconds / 86400.0

            # Format context with metadata
            text = node.props.get("text", "")[:ASK_SNIPPET_LEN]
            age_str = f"{age_days:.1f}d" if age_days else "unknown"
            context_items.append(
                f"[{i}] {text}\n"
                f"   (similarity={similarity:.3f}, drift={node.drift_score or 0:.3f}, age={age_str})"
            )

        # 4. Hybrid routing: select fast or fallback LLM
        use_fast, routing_reason = should_use_fast_path(
            request.question, top_similarity, len(filtered_results)
        )

        # Select LLM provider and token budget
        if use_fast and llm_fast is not None:
            selected_llm = llm_fast
            max_tokens = ASK_FAST_MAX_TOKENS
            llm_path = "fast"
        else:
            selected_llm = llm_fallback if llm_fallback is not None else llm
            max_tokens = ASK_FALLBACK_MAX_TOKENS if HYBRID_ROUTING_ENABLED else ASK_MAX_TOKENS
            llm_path = "fallback"

        # Log routing decision
        logger.info(
            "LLM routing decision",
            extra_fields={
                "question": request.question[:50],
                "path": llm_path,
                "reason": routing_reason,
                "top_similarity": round(top_similarity, 3),
            },
        )

        # 5. LLM call with optimized prompt and parameters (with simple context-aware cache)
        system_message, prompt = build_strict_citation_prompt(context_items, request.question)
        # Cache key based on tenant + question + context node IDs + LLM path
        ctx_ids = ",".join([n.id for n, _ in filtered_results])
        cache_key = f"ask::{request.tenant_id or ''}::{request.question}::{ctx_ids}::{llm_path}"
        cached = _ask_cache_get(cache_key)
        if cached is not None:
            return cached

        answer = selected_llm.generate(
            prompt,
            system_message=system_message,
            max_tokens=max_tokens,
            temperature=0.1,  # Keep deterministic for consistency
        )

        # 5. Extract citations and add lineage (use filtered_results for indexing)
        citation_indices = extract_citation_numbers(answer)
        cited_nodes = [
            filtered_results[i][0] for i in citation_indices if i < len(filtered_results)
        ]

        citations = []
        for node in cited_nodes:
            # Get lineage (up to 3 ancestors)
            lineage = repo.get_lineage(node.id, max_depth=3)

            # Calculate age
            age_days = None
            if node.last_refreshed:
                age_seconds = (datetime.now(timezone.utc) - node.last_refreshed).total_seconds()
                age_days = round(age_seconds / 86400.0, 2)

            citations.append(
                {
                    "node_id": node.id,
                    "classes": node.classes,
                    "drift_score": node.drift_score,
                    "age_days": age_days,
                    "lineage": [{"ancestor": anc["id"], "depth": anc["depth"]} for anc in lineage],
                }
            )

        # 6. Calculate confidence (use filtered results for confidence calculation)
        confidence = calculate_confidence(answer, filtered_results, citation_indices, intent_type)

        # Cap confidence at 0.6 for low-sim fallback queries
        if low_sim_fallback:
            confidence = min(confidence, 0.6)

        # 7. Per-question metrics for evaluation and debugging
        first_citation_idx = citation_indices[0] if citation_indices else None
        first_citation_precision = (
            1.0 if first_citation_idx == 0 else 0.0 if first_citation_idx is not None else None
        )

        # 7.5 Include term gating diagnostics when available
        gating_info = {}
        try:
            if gating_before_cnt is not None:
                gating_info = {
                    "term_gating_before": gating_before_cnt,
                    "term_gating_after": gating_after_cnt,
                    "term_gating_relaxed": gating_relaxed,
                }
        except Exception:
            gating_info = {}

        response = {
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "metadata": {
                "searched_nodes": len(results),
                "filtered_nodes": len(filtered_results),
                "cited_nodes": len(citations),
                "top_similarity": round(top_similarity, 3),
                "top_vector_similarity": round(top_vector_similarity, 3),
                "max_vector_similarity": round(max_vector_similarity, 3),
                "gating_score": round(top_similarity, 3),
                "gating_score_type": gating_score_type,
                "first_citation_idx": first_citation_idx,
                "citation_at_1_precision": first_citation_precision,
                "llm_path": llm_path,
                "routing_reason": routing_reason,
                "intent_detected": intent_type,
                "intent_type": intent_type,  # TRACK 1: Include intent type
                "classes_filter": classes_filter
                if classes_filter
                else None,  # TRACK 1: Show class filtering
                "must_have_terms": must_have_terms
                if must_have_terms
                else None,  # TRACK 1: Show term gating
                "structured_results_count": len(structured_results) if structured_results else 0,
                **gating_info,
            },
        }

        _ask_cache_put(cache_key, response)

        # Track Prometheus metrics (if enabled)
        if METRICS_ENABLED:
            latency_ms = (time.time() - start_time) * 1000
            track_ask_request(
                gating_score=top_similarity,
                gating_score_type=gating_score_type,
                cited_nodes=len(citations),
                latency_ms=latency_ms,
                rejected=False,
                rejection_reason=None,
                reranked=not skip_rerank,
            )

        return response

    except Exception as e:
        # Handle any unexpected errors during /ask processing
        import traceback

        tb = traceback.format_exc()
        logger.error(
            "Error in /ask endpoint",
            extra_fields={"error": str(e), "traceback": tb, "actor_id": actor_id},
        )
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
    finally:
        # ALWAYS mark request as complete for concurrency tracking
        # This runs even if exception occurs or response returned early
        if request_id and tenant_id:
            identifier = get_identifier(http_request, tenant_id)
            rate_limiter.mark_request_end(identifier, "ask", request_id)


@app.post("/ask/stream")
async def ask_stream(
    http_request: Request,
    http_response: Response,
    request: AskRequest,
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Server-Sent Events streaming for LLM Q&A with citations.

    Streams tokens as they are generated and emits a final JSON payload with
    citations, confidence, and metadata.
    """
    if not LLM_ENABLED or llm is None:
        raise HTTPException(
            status_code=503,
            detail="LLM provider not enabled. Set LLM_ENABLED=true and configure LLM_BACKEND/LLM_MODEL.",
        )

    # Extract tenant context from JWT (secure) or request body (dev mode only)
    tenant_id, actor_id, actor_type = get_tenant_context(
        http_request, claims, allow_override=not JWT_ENABLED
    )

    # In dev mode, allow tenant_id override from request body
    if not JWT_ENABLED and request.tenant_id:
        tenant_id = request.tenant_id

    # Apply rate limiting with concurrency control (stricter limits for streaming)
    request_id = await apply_rate_limit(
        http_request,
        http_response,
        endpoint="ask_stream",
        tenant_id=tenant_id,
        check_concurrency=True,  # Max 2 concurrent /ask/stream per tenant
    )

    def sse(data: str, event: str | None = None) -> bytes:
        assert repo is not None, "GraphRepository not initialized"
        assert embedder is not None, "EmbeddingProvider not initialized"
        if event:
            return f"event: {event}\ndata: {data}\n\n".encode()
        return f"data: {data}\n\n".encode()

    def gen():
        try:
            # 1) Retrieval (same as /ask): try hybrid + reranker with fallback
            query_embedding = embedder.encode([request.question])[0]
            try:
                results = repo.hybrid_search(
                    query_text=request.question,
                    query_embedding=query_embedding,
                    top_k=20,
                    tenant_id=tenant_id,  # From JWT, not request body
                    use_reranker=True,
                )
            except Exception:
                results = repo.vector_search(
                    query_embedding=query_embedding,
                    top_k=request.max_results or 5,
                    tenant_id=tenant_id,  # From JWT, not request body
                    use_weighted_score=request.use_weighted_score,
                )

            if not results:
                yield sse(
                    '{"type":"final","answer":"I couldn\'t find relevant information.","citations":[],"confidence":0.0}',
                    "final",
                )
                return

            top_similarity = results[0][1]
            if top_similarity < ASK_SIM_THRESHOLD:
                yield sse(
                    '{"type":"final","answer":"I don\'t have enough information to answer this question confidently.","citations":[],"confidence":0.2}',
                    "final",
                )
                return

            # Filter context to top-3
            filtered_results = filter_context_by_similarity(
                results,
                similarity_threshold=ASK_SIM_THRESHOLD,
                min_results=2,
                max_results=min(request.max_results or ASK_MAX_SNIPPETS, ASK_MAX_SNIPPETS),
            )

            # Build context and prompt
            context_items = []
            for i, (node, similarity) in enumerate(filtered_results):
                age_days = None
                if node.last_refreshed:
                    age_seconds = (datetime.now(timezone.utc) - node.last_refreshed).total_seconds()
                    age_days = age_seconds / 86400.0
                text = node.props.get("text", "")[
                    :ASK_SNIPPET_LEN
                ]  # snippet length is configurable
                age_str = f"{age_days:.1f}d" if age_days else "unknown"
                context_items.append(
                    f"[{i}] {text}\n   (similarity={similarity:.3f}, drift={node.drift_score or 0:.3f}, age={age_str})"
                )

            system_message, prompt = build_strict_citation_prompt(context_items, request.question)

            # Send initial context metadata event
            try:
                ctx_ids = [n.id for n, _ in filtered_results]
                yield sse(
                    data=json.dumps(
                        {
                            "type": "context",
                            "node_ids": ctx_ids,
                            "top_similarity": round(top_similarity, 3),
                            "count": len(filtered_results),
                        }
                    ),
                    event="context",
                )
            except Exception:
                pass

            # 2) Stream tokens
            answer_buf = []
            for piece in llm.generate_stream(
                prompt,
                system_message=system_message,
                max_tokens=ASK_MAX_TOKENS,
                temperature=0.1,
            ):
                answer_buf.append(piece)
                # Emit token event
                yield sse(json.dumps({"type": "token", "text": piece}), event="token")

            # 3) Finalize with citations + confidence
            answer = "".join(answer_buf)
            citation_indices = extract_citation_numbers(answer)
            cited_nodes = [
                filtered_results[i][0] for i in citation_indices if i < len(filtered_results)
            ]

            citations = []
            for node in cited_nodes:
                lineage = repo.get_lineage(node.id, max_depth=3)
                age_days = None
                if node.last_refreshed:
                    age_seconds = (datetime.now(timezone.utc) - node.last_refreshed).total_seconds()
                    age_days = round(age_seconds / 86400.0, 2)
                citations.append(
                    {
                        "node_id": node.id,
                        "classes": node.classes,
                        "drift_score": node.drift_score,
                        "age_days": age_days,
                        "lineage": [
                            {"ancestor": anc["id"], "depth": anc["depth"]} for anc in lineage
                        ],
                    }
                )

            confidence = calculate_confidence(answer, filtered_results, citation_indices)

            final_payload = {
                "type": "final",
                "answer": answer,
                "citations": citations,
                "confidence": confidence,
                "metadata": {
                    "searched_nodes": len(results),
                    "filtered_nodes": len(filtered_results),
                    "cited_nodes": len(citations),
                    "top_similarity": round(top_similarity, 3),
                },
            }
            yield sse(json.dumps(final_payload), event="final")

        except Exception as e:
            yield sse(json.dumps({"type": "error", "message": str(e)}), event="error")
        finally:
            # ALWAYS mark request as complete for concurrency tracking
            if request_id and tenant_id:
                identifier = get_identifier(http_request, tenant_id)
                rate_limiter.mark_request_end(identifier, "ask_stream", request_id)

    # Ensure X-RateLimit headers make it to streaming responses
    rl_headers = {}
    try:
        for h in ("X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"):
            v = http_response.headers.get(h)
            if v is not None:
                rl_headers[h] = v
    except Exception:
        pass

    return StreamingResponse(gen(), media_type="text/event-stream", headers=rl_headers)


@app.post("/edges", response_model=None)
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


@app.post("/triggers", response_model=None)
def register_trigger_pattern(
    pattern: dict[str, Any],
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Register a semantic trigger pattern.

    Expects: {"name": "pattern_name", "example_text": "...", "description": "..."}

    Security:
        Requires JWT authentication when JWT_ENABLED=true.
        Triggers are global resources (not tenant-scoped).
    """
    assert embedder is not None, "EmbeddingProvider not initialized"
    assert pattern_store is not None, "PatternStore not initialized"
    # Require authentication for trigger management
    if JWT_ENABLED and not claims:
        raise HTTPException(status_code=401, detail="Authentication required")
    if "name" not in pattern or "example_text" not in pattern:
        raise HTTPException(status_code=400, detail="Missing required fields: name, example_text")

    try:
        name = pattern["name"]
        example_text = pattern["example_text"]
        description = pattern.get("description")

        # Embed the example text to create the pattern
        embedding = embedder.encode([example_text])[0]

        # Store pattern
        pattern_store.set(name, embedding, description)

        return {
            "status": "registered",
            "name": name,
            "description": description,
        }
    except Exception as e:
        logger.error("Pattern registration failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Pattern registration failed: {str(e)}")


@app.get("/triggers", response_model=None)
def list_trigger_patterns(
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """List all registered trigger patterns.

    Security:
        Returns all patterns (no tenant filtering for system-level triggers).
        Rate limited for read protection.
    """
    assert pattern_store is not None, "PatternStore not initialized"
    try:
        patterns = pattern_store.list_patterns()
        return {"patterns": patterns, "count": len(patterns)}
    except Exception as e:
        logger.error("Pattern listing failed", extra_fields={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Pattern listing failed: {str(e)}")


@app.delete("/triggers/{name}", response_model=None)
def delete_trigger_pattern(
    name: str,
    _rl: None = Depends(require_rate_limit("default")),
    claims: JWTClaims | None = Depends(get_jwt_claims),
):
    """Delete a trigger pattern by name.

    Security:
        Requires JWT authentication when JWT_ENABLED=true.
    """
    assert repo is not None, "GraphRepository not initialized"
    assert pattern_store is not None, "PatternStore not initialized"
    # Require authentication for trigger management
    if JWT_ENABLED and not claims:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        deleted = pattern_store.delete(name)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Pattern not found: {name}")
        return {"status": "deleted", "name": name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Pattern deletion failed", extra_fields={"error": str(e), "name": name})
        raise HTTPException(status_code=500, detail=f"Pattern deletion failed: {str(e)}")


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
