from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import timezone
from typing import Any, Literal, TypedDict, cast
from urllib.parse import urlparse

# NotRequired was added in Python 3.11, use typing_extensions for 3.10
if sys.version_info >= (3, 11):
    from typing import NotRequired
else:
    from typing_extensions import NotRequired

import numpy as np
import psycopg
from pgvector.psycopg import Vector, register_vector
from psycopg_pool import ConnectionPool

from activekg.common.logger import get_enhanced_logger
from activekg.graph.models import Edge, Node

# RLS Configuration
# auto = detect DB RLS and set tenant context accordingly (default)
# on = always set tenant context
# off = skip tenant context (only safe if DB RLS is disabled)
RLS_MODE = os.getenv("RLS_MODE", "auto").lower()


class RefreshPolicyTD(TypedDict, total=False):
    interval: str
    cron: str
    drift_threshold: float


class TriggerPatternTD(TypedDict, total=False):
    name: str
    threshold: float
    description: NotRequired[str]


# Row shape for core node selection queries
# (id, tenant_id, classes, props, payload_ref, embedding, metadata, refresh_policy,
#  triggers, version, last_refreshed, drift_score)
NodeRow = tuple[
    str,
    str | None,
    list[str],
    dict[str, Any],
    str | None,
    Any,  # pgvector array / list
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    int,
    Any,  # datetime | None
    float | None,
]


class LineageRecordTD(TypedDict):
    id: str
    depth: int
    edge_props: dict[str, Any] | None
    classes: list[str]
    props: dict[str, Any]
    created_at: str | None


class NodeVersionTD(TypedDict):
    version_index: int
    drift_score: float | None
    created_at: str | None
    embedding_ref: str | None


class DriftSpikeAnomalyTD(TypedDict):
    type: Literal["drift_spike"]
    node_id: str
    spike_count: int
    avg_drift: float
    drift_scores: list[float]
    mean_drift: float
    spike_threshold: float
    classes: list[str]
    props: dict[str, Any]


class TriggerStormAnomalyTD(TypedDict):
    type: Literal["trigger_storm"]
    node_id: str
    event_count: int
    first_event: str | None
    last_event: str | None
    recent_events: list[Any]
    event_threshold: int
    classes: list[str]
    props: dict[str, Any]


class SchedulerLagAnomalyTD(TypedDict):
    type: Literal["scheduler_lag"]
    node_id: str
    expected_interval_seconds: float | int | None
    actual_lag_seconds: float | None
    lag_ratio: float
    lag_multiplier: float
    last_refreshed: str | None
    classes: list[str]
    props: dict[str, Any]


# Extended row shapes for specific SELECTs (extra computed columns)
# Node with appended vec_similarity (float) at index 12
NodeVecSimRow = tuple[
    str,
    str | None,
    list[str],
    dict[str, Any],
    str | None,
    Any,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    int,
    Any,
    float | None,
    float,
]

# Node with appended vec_similarity (float) and ts_rank (float) at indexes 12 and 13
NodeHybridRow = tuple[
    str,
    str | None,
    list[str],
    dict[str, Any],
    str | None,
    Any,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    int,
    Any,
    float | None,
    float,
    float,
]

# Node with a single relevance float appended at the end
NodeRelevanceRow = tuple[
    str,
    str | None,
    list[str],
    dict[str, Any],
    str | None,
    Any,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    int,
    Any,
    float | None,
    float,
]


# Event payloads (best-effort typed shapes)
class TriggerFiredPayloadTD(TypedDict, total=False):
    pattern: str
    score: float
    node_id: str


class RefreshedPayloadTD(TypedDict, total=False):
    drift_score: float
    previous_ref: str | None
    embedding_ref: str | None


class RecentEventTD(TypedDict, total=False):
    type: str
    created_at: str | None
    payload: dict[str, Any]


EventPayloadTD = dict[str, Any] | TriggerFiredPayloadTD | RefreshedPayloadTD


class GraphRepository:
    """Postgres + pgvector repository with connection pooling.

    Uses psycopg_pool.ConnectionPool for efficient connection reuse.
    """

    def __init__(self, dsn: str, candidate_factor: float = 2.0):
        self.dsn = dsn
        self.candidate_factor = candidate_factor  # For weighted search re-ranking
        self.logger = get_enhanced_logger(__name__)

        # Initialize connection pool
        # min_size=2: Keep 2 connections warm for low-latency requests
        # max_size=10: Allow up to 10 concurrent connections (adjust for load)
        # timeout=30: Wait up to 30s for available connection
        self.pool = ConnectionPool(
            self.dsn,
            min_size=2,
            max_size=10,
            timeout=30.0,
            open=True,
            configure=self._configure_connection,
        )
        self.logger.info(
            "Connection pool initialized", extra_fields={"min_size": 2, "max_size": 10}
        )

        # Detect and resolve RLS mode at startup
        self._rls_enabled = self._resolve_rls_mode()
        self.logger.info(
            "RLS mode resolved",
            extra_fields={
                "rls_mode_config": RLS_MODE,
                "rls_enabled_runtime": self._rls_enabled,
            },
        )

    def _detect_rls(self) -> bool:
        """Detect if RLS is enabled on core tables at the database level.

        Checks pg_class.relrowsecurity for the 'nodes' table.

        Returns:
            bool: True if RLS is enabled in the database, False otherwise
        """
        try:
            conn = self.pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT relrowsecurity
                        FROM pg_class
                        WHERE relname = 'nodes'
                        """
                    )
                    row = cur.fetchone()
                    db_rls_enabled = bool(row and row[0]) if row else False
                    self.logger.info(
                        "RLS detection from database",
                        extra_fields={"db_rls_enabled": db_rls_enabled},
                    )
                    return db_rls_enabled
            finally:
                self.pool.putconn(conn)
        except Exception as e:
            self.logger.warning(
                "Failed to detect RLS status, assuming disabled",
                extra_fields={"error": str(e)},
            )
            return False

    def _resolve_rls_mode(self) -> bool:
        """Resolve RLS mode based on environment and database state.

        Behavior Matrix:
        - RLS_MODE=auto: Use database RLS state
        - RLS_MODE=on: Always enable tenant context
        - RLS_MODE=off: Skip tenant context (but force ON if DB RLS is enabled to prevent lockout)

        Returns:
            bool: True if tenant context should be set, False otherwise
        """
        db_rls = self._detect_rls()

        if RLS_MODE == "auto":
            self.logger.info("RLS_MODE=auto, using database state", extra_fields={"db_rls": db_rls})
            return db_rls

        if RLS_MODE == "on":
            self.logger.info("RLS_MODE=on, forcing tenant context enabled")
            return True

        if RLS_MODE == "off":
            if db_rls:
                self.logger.error(
                    "RLS_MODE=off but database RLS is enabled! "
                    "Forcing tenant context ON to prevent query lockout. "
                    "To disable RLS, run: ALTER TABLE nodes DISABLE ROW LEVEL SECURITY;",
                    extra_fields={"db_rls": db_rls, "rls_mode": RLS_MODE},
                )
                # Safety: Force ON to prevent complete lockout
                return True
            self.logger.info("RLS_MODE=off and database RLS is disabled, skipping tenant context")
            return False

        # Unknown RLS_MODE value, default to auto-detect
        self.logger.warning(
            f"Unknown RLS_MODE='{RLS_MODE}', defaulting to auto-detect",
            extra_fields={"rls_mode": RLS_MODE, "db_rls": db_rls},
        )
        return db_rls

    def _distance_operator(self) -> tuple[str, str]:
        """Return (op_symbol, similarity_sql_expr_template).

        Controlled by env SEARCH_DISTANCE one of: 'cosine' (default), 'l2', 'ip'.
        - cosine: uses '<=>', similarity = 1 - (embedding <=> %s)
        - l2:     uses '<->', similarity = 1.0 - LEAST((embedding <-> %s), 1.0)  (clamped)
        - ip:     uses '<#>', similarity = (embedding <#> %s)  (note: higher is better)
        """
        metric = os.getenv("SEARCH_DISTANCE", "cosine").lower()
        if metric == "l2":
            # L2 distance: lower is better; clamp to [0,1] for UI similarity display
            return "<->", "1.0 - LEAST((embedding <-> %s), 1.0)"
        if metric == "ip":
            # Inner product: higher is better; return raw IP as similarity
            return "<#>", "(embedding <#> %s)"
        # Default: cosine distance
        return "<=>", "1 - (embedding <=> %s)"

    def _configure_connection(self, conn):
        """Configure each new connection from the pool."""
        register_vector(conn)

    def _build_node_from_row(self, row: Sequence[Any]) -> Node:
        """Safely construct a Node from a DB row with type guards and casts."""
        emb: np.ndarray | None
        raw_emb = row[5]
        if raw_emb is None:
            emb = None
        else:
            # pgvector returns a list/Vector that can be cast to ndarray
            try:
                emb = np.array(raw_emb, dtype=np.float32)
            except Exception:
                emb = None

        props = cast(dict[str, Any], row[3] or {})
        metadata = cast(dict[str, Any], row[6] or {})
        refresh_policy = cast(RefreshPolicyTD, cast(dict[str, Any], row[7] or {}))
        triggers = cast(list[TriggerPatternTD], cast(list[dict[str, Any]], row[8] or []))

        embedding_status = cast(str | None, row[12]) if len(row) > 12 else None
        embedding_error = cast(str | None, row[13]) if len(row) > 13 else None
        embedding_attempts = cast(int | None, row[14]) if len(row) > 14 else None
        embedding_updated_at = cast(Any, row[15]) if len(row) > 15 else None

        return Node(
            id=str(row[0]),
            tenant_id=cast(str | None, row[1]),
            classes=cast(list[str], row[2] or []),
            props=props,
            payload_ref=cast(str | None, row[4]),
            embedding=emb,
            embedding_status=embedding_status,
            embedding_error=embedding_error,
            embedding_attempts=embedding_attempts,
            embedding_updated_at=embedding_updated_at,
            metadata=metadata,
            refresh_policy=cast("dict[str, Any]", refresh_policy or {}),
            triggers=cast("list[dict[str, Any]]", triggers or []),
            version=cast(int, row[9]),
            last_refreshed=cast(Any, row[10]),
            drift_score=cast(float | None, row[11]),
        )

    @contextmanager
    def _conn(self, tenant_id: str | None = None):
        """Get pooled connection with optional tenant context for RLS, with commit-on-exit.

        - Uses a transaction block (`with conn:`) so writes are committed on success and rolled back on error.
        - Applies `SET LOCAL app.current_tenant_id` inside the transaction to enforce RLS without leaking state.
        - RLS tenant context is only set if self._rls_enabled is True (controlled by RLS_MODE env var).
        """
        conn = self.pool.getconn()
        try:
            # Start a transaction context to ensure SET LOCAL applies and writes are committed
            with conn:
                from psycopg import sql

                with conn.cursor() as cur:
                    # Only set tenant context if RLS is enabled
                    if self._rls_enabled:
                        if tenant_id:
                            # Prefer set_config() to avoid failures on servers that don't accept bare SET for custom GUCs
                            try:
                                cur.execute(
                                    "SELECT set_config('app.current_tenant_id', %s, true)",
                                    (tenant_id,),
                                )
                                self.logger.debug(
                                    "RLS: set_config tenant", extra_fields={"tenant_id": tenant_id}
                                )
                            except Exception as e:
                                # Best-effort fallback to SET LOCAL; keep transaction usable even if it fails
                                self.logger.warning(
                                    "RLS: set_config failed; attempting SET LOCAL",
                                    extra_fields={"error": str(e)},
                                )
                                try:
                                    cur.execute(
                                        sql.SQL("SET LOCAL app.current_tenant_id = {}").format(
                                            sql.Literal(tenant_id)
                                        )
                                    )
                                    self.logger.debug(
                                        "RLS: SET tenant", extra_fields={"tenant_id": tenant_id}
                                    )
                                except Exception as e2:
                                    # Do not poison the transaction; continue without tenant filter (RLS will fall back to NULL)
                                    self.logger.warning(
                                        "RLS: SET LOCAL failed; proceeding without explicit tenant setting",
                                        extra_fields={"error": str(e2), "tenant_id": tenant_id},
                                    )
                        else:
                            # For NULL tenant: use special sentinel that matches RLS policy for NULL rows
                            # RLS policy: (tenant_id IS NULL) OR (tenant_id = current_setting(...))
                            # Disable tenant scoping by resetting the setting so current_setting(..., true) returns NULL
                            try:
                                cur.execute("RESET app.current_tenant_id")
                                self.logger.debug("RLS: RESET tenant (allow NULL rows)")
                            except Exception as e:
                                self.logger.warning(
                                    "RLS: RESET failed, continuing", extra_fields={"error": str(e)}
                                )

                        # Optional debug: verify tenant context
                        if os.getenv("ACTIVEKG_DEBUG_RLS", "false").lower() == "true":
                            cur.execute("SELECT current_setting('app.current_tenant_id', true)")
                            current_tenant = cur.fetchone()[0]
                            self.logger.info(
                                "RLS context verified",
                                extra_fields={
                                    "requested_tenant": tenant_id,
                                    "current_setting": current_tenant,
                                },
                            )
                    else:
                        # RLS disabled - skip tenant context setting
                        self.logger.debug(
                            "RLS disabled, skipping tenant context",
                            extra_fields={"tenant_id": tenant_id, "rls_mode": RLS_MODE},
                        )

                # Hand control back to caller within the same transaction
                yield conn
        finally:
            # Always return connection to pool
            self.pool.putconn(conn)

    def ensure_vector_index(self):
        """Ensure configured pgvector indexes exist.

        Supports one or both of: IVFFLAT and HNSW via env configuration.
        - PGVECTOR_INDEX: ivfflat|hnsw (single)
        - PGVECTOR_INDEXES: comma list (e.g., "ivfflat,hnsw")
        Metric is derived from SEARCH_DISTANCE (l2|cosine). Defaults to cosine.

        Uses CREATE INDEX CONCURRENTLY and is safe/idempotent.
        """
        try:
            # Determine targets
            raw = os.getenv("PGVECTOR_INDEXES") or os.getenv("PGVECTOR_INDEX") or ""
            targets = [t.strip().lower() for t in raw.split(",") if t.strip()] or ["ivfflat"]

            metric = os.getenv("SEARCH_DISTANCE", "cosine").lower()
            if metric not in ("l2", "cosine"):
                metric = "cosine"
            opclass = "vector_l2_ops" if metric == "l2" else "vector_cosine_ops"

            # Params
            try:
                lists = int(os.getenv("IVFFLAT_LISTS", "100"))
            except Exception:
                lists = 100
            try:
                hnsw_m = int(os.getenv("HNSW_M", "16"))
            except Exception:
                hnsw_m = 16
            try:
                hnsw_efc = int(os.getenv("HNSW_EF_CONSTRUCTION", "128"))
            except Exception:
                hnsw_efc = 128

            # Open autocommit connection for index creation
            conn = psycopg.connect(self.dsn, autocommit=True)
            register_vector(conn)

            with conn.cursor() as cur:
                # Helper to check existence
                def index_exists(name: str) -> bool:
                    cur.execute(
                        "SELECT 1 FROM pg_indexes WHERE tablename='nodes' AND indexname=%s",
                        (name,),
                    )
                    return cur.fetchone() is not None

                created: list[str] = []
                for t in targets:
                    if t not in ("ivfflat", "hnsw"):
                        continue
                    suffix = "l2" if metric == "l2" else "cos"
                    name = f"idx_nodes_embedding_{t}_{suffix}"
                    if index_exists(name):
                        self.logger.info("Index exists", extra_fields={"index": name})
                        continue

                    if t == "ivfflat":
                        sql = (
                            f"CREATE INDEX CONCURRENTLY {name} ON nodes "
                            f"USING ivfflat (embedding {opclass}) WITH (lists = {lists})"
                        )
                    else:  # hnsw
                        sql = (
                            f"CREATE INDEX CONCURRENTLY {name} ON nodes "
                            f"USING hnsw (embedding {opclass}) WITH (m = {hnsw_m}, ef_construction = {hnsw_efc})"
                        )

                    try:
                        self.logger.info(
                            "Creating vector index",
                            extra_fields={"index": name, "metric": metric, "type": t},
                        )
                        import time

                        start_ts = time.time()
                        cur.execute(sql)
                        duration = time.time() - start_ts
                        try:
                            from activekg.observability.metrics import track_index_build

                            track_index_build(duration, type_=t, metric=metric, result="success")
                        except Exception:
                            pass
                        created.append(name)
                    except psycopg.errors.DuplicateTable:
                        self.logger.info(
                            "Index created by another replica", extra_fields={"index": name}
                        )
                        try:
                            from activekg.observability.metrics import track_index_build

                            track_index_build(0.0, type_=t, metric=metric, result="duplicate")
                        except Exception:
                            pass
                    except Exception as ce:
                        self.logger.error(
                            "Index creation failed", extra_fields={"index": name, "error": str(ce)}
                        )
                        try:
                            import time

                            from activekg.observability.metrics import track_index_build

                            track_index_build(0.0, type_=t, metric=metric, result="error")
                        except Exception:
                            pass

                if created:
                    # Optional: Analyze to help planner
                    try:
                        cur.execute("VACUUM ANALYZE nodes")
                    except Exception:
                        pass
                    self.logger.info("Vector indexes created", extra_fields={"created": created})

            conn.close()

        except Exception as e:
            self.logger.error("ensure_vector_index failed", extra_fields={"error": str(e)})
            # Non-fatal: system still works without an ANN index

    # --- Admin helpers for ANN indexes ---
    def list_vector_indexes(self) -> list[str]:
        """List ANN index names on nodes.embedding."""
        try:
            conn = psycopg.connect(self.dsn, autocommit=True)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname='public' AND tablename='nodes' AND indexname LIKE 'idx_nodes_embedding%'
                    ORDER BY indexname
                    """
                )
                rows = cur.fetchall() or []
            conn.close()
            return [r[0] for r in rows]
        except Exception:
            return []

    def drop_vector_indexes(self, names: list[str]) -> list[str]:
        """Drop specified ANN indexes by name (CONCURRENTLY, non-fatal on error)."""
        dropped: list[str] = []
        if not names:
            return dropped
        try:
            conn = psycopg.connect(self.dsn, autocommit=True)
            with conn.cursor() as cur:
                for name in names:
                    try:
                        cur.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {name}")
                        dropped.append(name)
                    except Exception as e:
                        self.logger.error(
                            "DROP INDEX failed", extra_fields={"index": name, "error": str(e)}
                        )
            conn.close()
        except Exception as e:
            self.logger.error("drop_vector_indexes failed", extra_fields={"error": str(e)})
        return dropped

    def planned_index_names(
        self, types: list[str] | None = None, metric: str | None = None
    ) -> list[str]:
        """Compute expected ANN index names for given types and metric."""
        metric = cast(str, metric or os.getenv("SEARCH_DISTANCE", "cosine")).lower()
        suffix = "l2" if metric == "l2" else "cos"
        types = (
            types
            or [
                t.strip().lower()
                for t in (os.getenv("PGVECTOR_INDEXES") or os.getenv("PGVECTOR_INDEX") or "").split(
                    ","
                )
                if t.strip()
            ]
            or ["ivfflat"]
        )
        out: list[str] = []
        for t in types:
            if t in ("ivfflat", "hnsw"):
                out.append(f"idx_nodes_embedding_{t}_{suffix}")
        return out

    # --- Listing ---
    def list_nodes(
        self,
        classes_filter: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
        tenant_id: str | None = None,
    ) -> list[Node]:
        """List nodes with optional class filtering and pagination.

        RLS-aware: uses tenant-scoped connection when tenant_id is provided.
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Per-query tuning for ANN search (best-effort; do not poison transaction)
                try:
                    raw = os.getenv("PGVECTOR_INDEXES") or os.getenv("PGVECTOR_INDEX") or ""
                    targets = [t.strip().lower() for t in raw.split(",") if t.strip()]
                    if not targets:
                        targets = ["ivfflat"]
                    # Create a savepoint before GUC changes so any failure won't abort the whole tx
                    cur.execute("SAVEPOINT akg_guc")
                    try:
                        # Helper to check if GUC exists to avoid aborting the transaction
                        def guc_exists(name: str) -> bool:
                            try:
                                cur.execute("SELECT 1 FROM pg_settings WHERE name = %s", (name,))
                                return cur.fetchone() is not None
                            except Exception:
                                return False

                        if "hnsw" in targets and guc_exists("hnsw.ef_search"):
                            ef_search = int(os.getenv("HNSW_EF_SEARCH", "80"))
                            # SET LOCAL is not parameterizable in all servers; inline sanitized integer
                            cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                        if "ivfflat" in targets and guc_exists("ivfflat.probes"):
                            probes = int(os.getenv("IVFFLAT_PROBES", "4"))
                            # Inline sanitized integer value
                            cur.execute(f"SET LOCAL ivfflat.probes = {probes}")
                        cur.execute("RELEASE SAVEPOINT akg_guc")
                    except Exception as e:
                        try:
                            cur.execute("ROLLBACK TO SAVEPOINT akg_guc")
                        except Exception:
                            pass
                        self.logger.warning(
                            "ANN per-query tuning skipped", extra_fields={"error": str(e)}
                        )
                except Exception:
                    # Savepoint creation or env parsing failed: skip tuning entirely
                    pass
                where_clauses: list[str] = []
                params: list[Any] = []

                if classes_filter:
                    where_clauses.append("classes && %s::text[]")
                    params.append(classes_filter)

                where_sql = (" WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

                cur.execute(
                    f"""
                    SELECT id, tenant_id, classes, props, payload_ref, embedding, metadata,
                           refresh_policy, triggers, version, last_refreshed, drift_score
                    FROM nodes
                    {where_sql}
                    ORDER BY created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (*params, max(0, int(limit)), max(0, int(offset))),
                )

                out: list[Node] = []
                for row in cur.fetchall():
                    out.append(self._build_node_from_row(cast(Sequence[Any], row)))
                return out

    # --- Nodes ---
    def create_node(self, node: Node) -> str:
        with self._conn(tenant_id=node.tenant_id) as conn:
            with conn.cursor() as cur:
                emb = node.embedding.tolist() if isinstance(node.embedding, np.ndarray) else None
                cur.execute(
                    """
                    INSERT INTO nodes (id, tenant_id, classes, props, payload_ref, embedding, metadata, refresh_policy, triggers, version, last_refreshed, drift_score, embedding_status, embedding_error, embedding_attempts, embedding_updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        node.id,
                        node.tenant_id,
                        node.classes,
                        json.dumps(node.props),
                        node.payload_ref,
                        emb,
                        json.dumps(node.metadata),
                        json.dumps(node.refresh_policy),
                        json.dumps(node.triggers),
                        node.version,
                        node.last_refreshed,
                        node.drift_score,
                        node.embedding_status or "queued",
                        node.embedding_error,
                        node.embedding_attempts or 0,
                        node.embedding_updated_at,
                    ),
                )
                new_id = cur.fetchone()[0]
                return str(new_id)

    def get_node(self, node_id: str, tenant_id: str | None = None) -> Node | None:
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, tenant_id, classes, props, payload_ref, embedding, metadata,
                           refresh_policy, triggers, version, last_refreshed, drift_score,
                           embedding_status, embedding_error, embedding_attempts, embedding_updated_at
                    FROM nodes WHERE id = %s
                    """,
                    (node_id,),
                )
                row = cast(NodeRow | None, cur.fetchone())
                if not row:
                    return None
                return self._build_node_from_row(row)

    def get_node_by_external_id(
        self, external_id: str, tenant_id: str | None = None
    ) -> Node | None:
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, tenant_id, classes, props, payload_ref, embedding, metadata,
                           refresh_policy, triggers, version, last_refreshed, drift_score,
                           embedding_status, embedding_error, embedding_attempts, embedding_updated_at
                    FROM nodes WHERE props->>'external_id' = %s
                    LIMIT 1
                    """,
                    (external_id,),
                )
                row = cast(NodeRow | None, cur.fetchone())
                if not row:
                    return None
                return self._build_node_from_row(row)

    def update_node(
        self,
        node_id: str,
        *,
        classes: list[str] | None = None,
        props: dict[str, Any] | None = None,
        payload_ref: str | None = None,
        metadata: dict[str, Any] | None = None,
        refresh_policy: RefreshPolicyTD | None = None,
        triggers: list[TriggerPatternTD] | None = None,
        tenant_id: str | None = None,
    ) -> bool:
        """Update mutable fields for a node.

        RLS-aware: uses tenant-scoped connection when tenant_id is provided.
        Returns True if a row was updated.
        """
        sets: list[str] = ["updated_at = now()"]
        params: list[Any] = []

        if classes is not None:
            sets.append("classes = %s")
            params.append(classes)
        if props is not None:
            sets.append("props = %s")
            params.append(json.dumps(props))
        if payload_ref is not None:
            sets.append("payload_ref = %s")
            params.append(payload_ref)
        if metadata is not None:
            sets.append("metadata = %s")
            params.append(json.dumps(metadata))
        if refresh_policy is not None:
            sets.append("refresh_policy = %s")
            params.append(json.dumps(refresh_policy))
        if triggers is not None:
            sets.append("triggers = %s")
            params.append(json.dumps(triggers))

        if len(sets) == 1:  # only updated_at
            return True

        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                sql = f"UPDATE nodes SET {', '.join(sets)} WHERE id = %s"
                params.append(node_id)
                cur.execute(sql, params)
                return cast(int, cur.rowcount) > 0

    def delete_node(self, node_id: str, hard: bool = False, tenant_id: str | None = None) -> bool:
        """Delete a node.

        - hard=True: permanent delete
        - hard=False: soft delete (adds 'Deleted' class and grace window)
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                if hard:
                    cur.execute("DELETE FROM nodes WHERE id = %s", (node_id,))
                    return cast(int, cur.rowcount) > 0
                # Soft-delete: add Deleted class if absent, set deletion metadata
                cur.execute(
                    """
                    UPDATE nodes
                    SET classes = CASE WHEN NOT ('Deleted' = ANY(classes))
                                       THEN array_append(classes, 'Deleted')
                                       ELSE classes END,
                        props = props || jsonb_build_object(
                            'deleted_at', now(),
                            'deletion_grace_until', (now() + interval '30 days')
                        )
                    WHERE id = %s
                    """,
                    (node_id,),
                )
                return cast(int, cur.rowcount) > 0

    def update_node_embedding(
        self,
        node_id: str,
        embedding: np.ndarray,
        drift: float,
        timestamp: str,
        tenant_id: str | None = None,
        content_hash: str | None = None,
        extraction_version: str | None = None,
    ) -> None:
        """Update node embedding, drift score, and last_refreshed timestamp.

        RLS-aware: uses tenant-scoped connection when tenant_id is provided.
        """
        if extraction_version is None:
            extraction_version = os.getenv("EXTRACTION_VERSION", "1.0.0")
        extra_props: dict[str, Any] = {}
        if content_hash:
            extra_props["content_hash"] = content_hash
        if extraction_version:
            extra_props["extraction_version"] = extraction_version

        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                sets = [
                    "embedding = %s",
                    "drift_score = %s",
                    "last_refreshed = %s",
                    "embedding_status = 'ready'",
                    "embedding_error = NULL",
                    "embedding_updated_at = now()",
                    "updated_at = now()",
                ]
                params: list[Any] = [embedding.tolist(), drift, timestamp]
                if extra_props:
                    sets.append("props = COALESCE(props, '{}'::jsonb) || %s::jsonb")
                    params.append(json.dumps(extra_props))
                params.append(node_id)
                cur.execute(
                    f"""
                    UPDATE nodes
                    SET {", ".join(sets)}
                    WHERE id = %s
                    """,
                    params,
                )

    def mark_embedding_queued(self, node_id: str, tenant_id: str | None = None) -> None:
        """Mark node embedding as queued."""
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE nodes
                    SET embedding_status = 'queued',
                        embedding_error = NULL,
                        embedding_updated_at = now(),
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (node_id,),
                )

    def mark_embedding_processing(self, node_id: str, tenant_id: str | None = None) -> int:
        """Mark node embedding as processing and increment attempt counter."""
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE nodes
                    SET embedding_status = 'processing',
                        embedding_error = NULL,
                        embedding_attempts = embedding_attempts + 1,
                        embedding_updated_at = now(),
                        updated_at = now()
                    WHERE id = %s
                    RETURNING embedding_attempts
                    """,
                    (node_id,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def mark_embedding_failed(self, node_id: str, error: str, tenant_id: str | None = None) -> None:
        """Mark node embedding as failed with error message."""
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE nodes
                    SET embedding_status = 'failed',
                        embedding_error = %s,
                        embedding_updated_at = now(),
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (error[:1000], node_id),
                )

    def mark_embedding_skipped(
        self, node_id: str, reason: str, tenant_id: str | None = None
    ) -> None:
        """Mark node embedding as skipped with reason."""
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE nodes
                    SET embedding_status = 'skipped',
                        embedding_error = %s,
                        embedding_updated_at = now(),
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (reason[:1000], node_id),
                )

    def mark_embedding_ready(self, node_id: str, tenant_id: str | None = None) -> None:
        """Mark node embedding as ready without modifying the vector."""
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE nodes
                    SET embedding_status = 'ready',
                        embedding_error = NULL,
                        embedding_updated_at = now(),
                        updated_at = now()
                    WHERE id = %s
                    """,
                    (node_id,),
                )

    def vector_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        compound_filter: dict[str, Any] | None = None,
        classes_filter: list[str] | None = None,
        tenant_id: str | None = None,
        use_weighted_score: bool = False,
        decay_lambda: float = 0.01,
        drift_beta: float = 0.1,
    ) -> list[tuple[Node, float]]:
        """Search nodes by vector similarity with optional metadata filters.

        Args:
            query_embedding: Query vector (384 dimensions)
            top_k: Number of results to return
            metadata_filters: Simple equality filters (e.g., {"category": "AI"})
            compound_filter: JSONB containment filter for nested/typed queries (e.g., {"tags": ["research"], "metrics.views": 1000})
            classes_filter: Filter by node classes (array overlap: classes && ['Job', 'Resume'])
            tenant_id: Filter by tenant for multi-tenancy
            use_weighted_score: If True, apply recency/drift weighting (default: False)
            decay_lambda: Decay rate for age penalty (default: 0.01 = ~1% per day)
            drift_beta: Drift penalty weight (default: 0.1 = 10% penalty per drift unit)

        Returns list of (Node, similarity_score) tuples ordered by similarity.

        Weighted scoring formula (when use_weighted_score=True):
            age_days = (now - last_refreshed) / 86400
            decay = exp(-decay_lambda * age_days)
            drift_penalty = 1 - (drift_beta * COALESCE(drift_score, 0))
            weighted_score = similarity * decay * drift_penalty

        Performance: Uses ANN for candidate retrieval (ORDER BY <=>), then re-ranks
        top candidates by weighted score to retain index performance.
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                if os.getenv("ACTIVEKG_DEBUG_RLS", "false").lower() == "true":
                    try:
                        cur.execute("SELECT current_setting('app.current_tenant_id', true)")
                        self.logger.info(
                            "vector_search tenant", extra_fields={"tenant": cur.fetchone()[0]}
                        )
                    except Exception:
                        pass
                query_vec = query_embedding.tolist()
                # Wrap with Vector() for proper pgvector type adaptation
                query_vec_param = Vector(query_vec)

                # Build WHERE clause for filters
                # Note: tenant isolation is handled by RLS, not application WHERE clause
                where_clauses: list[str] = []
                params: list[Any] = [query_vec_param]

                if classes_filter:
                    # Array overlap: classes && ['Job', 'Resume']::text[]
                    where_clauses.append("classes && %s::text[]")
                    params.append(classes_filter)

                if metadata_filters:
                    for key, value in metadata_filters.items():
                        # Compare stringified JSONB field to provided value
                        where_clauses.append("metadata->>%s = %s")
                        params.extend([key, str(value)])

                # Compound JSONB containment filter (uses GIN index)
                if compound_filter:
                    where_clauses.append("metadata @> %s::jsonb")
                    params.append(json.dumps(compound_filter))

                where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

                # Fetch more candidates if using weighted scoring for re-ranking
                # Use configurable candidate_factor (default 2.0, set via env WEIGHTED_SEARCH_CANDIDATE_FACTOR)
                fetch_limit = int(top_k * self.candidate_factor) if use_weighted_score else top_k

                # Execute with param order: select_vec, filters..., order_vec, limit
                self.logger.info(
                    "vector_search executing",
                    extra_fields={
                        "top_k": top_k,
                        "use_weighted": use_weighted_score,
                        "has_filters": bool(metadata_filters or compound_filter),
                    },
                )
                op, sim_expr = self._distance_operator()
                sql = f"""
                    SELECT
                        id, tenant_id, classes, props, payload_ref, embedding,
                        metadata, refresh_policy, triggers, version,
                        last_refreshed, drift_score,
                        {sim_expr} as similarity
                    FROM nodes
                    WHERE embedding IS NOT NULL{where_sql}
                    ORDER BY embedding {op} %s
                    LIMIT %s
                """
                cur.execute(sql, params + [query_vec_param, fetch_limit])

                results: list[tuple[Node, float]] = []
                for row in cur.fetchall():
                    row_t = cast(NodeVecSimRow, row)
                    node = self._build_node_from_row(cast(Sequence[Any], row_t))
                    similarity = float(row_t[12])
                    results.append((node, similarity))

                # Apply weighted scoring if enabled
                if use_weighted_score:
                    import math
                    from datetime import datetime

                    weighted_results: list[tuple[Node, float]] = []
                    now = datetime.now(timezone.utc)

                    for node, similarity in results:
                        # Calculate age in days
                        if node.last_refreshed:
                            age_seconds = (now - node.last_refreshed).total_seconds()
                            age_days = age_seconds / 86400.0
                        else:
                            # No refresh timestamp - assume very old (e.g., 365 days)
                            age_days = 365.0

                        # Calculate decay: exp(-lambda * age_days)
                        decay = math.exp(-decay_lambda * age_days)

                        # Calculate drift penalty: 1 - (beta * drift_score)
                        drift = node.drift_score if node.drift_score is not None else 0.0
                        drift_penalty = max(0.0, 1.0 - (drift_beta * drift))

                        # Weighted score
                        weighted_score = similarity * decay * drift_penalty

                        weighted_results.append((node, weighted_score))

                    # Re-rank by weighted score and return top_k
                    weighted_results.sort(key=lambda x: x[1], reverse=True)
                    out = weighted_results[:top_k]
                else:
                    out = results

                self.logger.info("vector_search results", extra_fields={"count": len(out)})
                return out

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        compound_filter: dict[str, Any] | None = None,
        classes_filter: list[str] | None = None,
        tenant_id: str | None = None,
        use_reranker: bool = True,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        rerank_skip_threshold: float = 0.80,
    ) -> list[tuple[Node, float]]:
        """Hybrid search with PostgreSQL ts_rank + vector similarity.

        Combines full-text search (ts_rank) with vector similarity for better recall.
        Optionally applies cross-encoder reranking for precision.

        Args:
            query_text: Raw query string for full-text search
            query_embedding: Query vector for semantic search
            top_k: Number of results to return
            metadata_filters: Simple equality filters
            compound_filter: JSONB containment filter
            classes_filter: Filter by node classes (array overlap: classes && ['Job', 'Resume'])
            tenant_id: Optional tenant ID for RLS
            use_reranker: Apply cross-encoder reranking (default: True)
            vector_weight: Weight for vector similarity (default: 0.7)
            text_weight: Weight for text rank score (default: 0.3)
            rerank_skip_threshold: Skip reranking if top hybrid_score >= this (default: 0.80)

        Returns:
            List of (Node, hybrid_score) tuples ordered by fused score
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Per-query tuning for ANN search (best-effort; do not poison transaction)
                try:
                    raw = os.getenv("PGVECTOR_INDEXES") or os.getenv("PGVECTOR_INDEX") or ""
                    targets = [t.strip().lower() for t in raw.split(",") if t.strip()]
                    if not targets:
                        targets = ["ivfflat"]
                    cur.execute("SAVEPOINT akg_guc")
                    try:

                        def guc_exists(name: str) -> bool:
                            try:
                                cur.execute("SELECT 1 FROM pg_settings WHERE name = %s", (name,))
                                return cur.fetchone() is not None
                            except Exception:
                                return False

                        if "hnsw" in targets and guc_exists("hnsw.ef_search"):
                            ef_search = int(os.getenv("HNSW_EF_SEARCH", "80"))
                            cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
                        if "ivfflat" in targets and guc_exists("ivfflat.probes"):
                            probes = int(os.getenv("IVFFLAT_PROBES", "4"))
                            cur.execute(f"SET LOCAL ivfflat.probes = {probes}")
                        cur.execute("RELEASE SAVEPOINT akg_guc")
                    except Exception as e:
                        try:
                            cur.execute("ROLLBACK TO SAVEPOINT akg_guc")
                        except Exception:
                            pass
                        self.logger.warning(
                            "ANN per-query tuning skipped", extra_fields={"error": str(e)}
                        )
                except Exception:
                    pass
                if os.getenv("ACTIVEKG_DEBUG_RLS", "false").lower() == "true":
                    try:
                        cur.execute("SELECT current_setting('app.current_tenant_id', true)")
                        self.logger.info(
                            "hybrid_search tenant", extra_fields={"tenant": cur.fetchone()[0]}
                        )
                    except Exception:
                        pass
                query_vec = query_embedding.tolist()
                # Wrap with Vector() for proper pgvector type adaptation
                query_vec_param = Vector(query_vec)

                # Build WHERE filters
                # Note: tenant isolation is handled by RLS, not application WHERE clause
                where_clauses: list[str] = []
                filter_params: list[Any] = []

                if classes_filter:
                    # Array overlap: classes && ['Job', 'Resume']::text[]
                    where_clauses.append("classes && %s::text[]")
                    filter_params.append(classes_filter)

                if metadata_filters:
                    for key, value in metadata_filters.items():
                        where_clauses.append("metadata->>%s = %s")
                        filter_params.extend([key, str(value)])

                if compound_filter:
                    where_clauses.append("metadata @> %s::jsonb")
                    filter_params.append(json.dumps(compound_filter))

                where_sql = " AND " + " AND ".join(where_clauses) if where_clauses else ""

                # Adaptive candidate_k based on expected similarity
                # Base: 20 candidates, increase to 40-50 when uncertain (lower top score)
                if use_reranker:
                    try:
                        base_candidates = int(os.getenv("HYBRID_RERANKER_BASE", "20"))
                        _boost_candidates = int(
                            os.getenv("HYBRID_RERANKER_BOOST", "45")
                        )  # Reserved for future adaptive boosting
                        adaptive_threshold = float(os.getenv("HYBRID_ADAPTIVE_THRESHOLD", "0.55"))
                    except Exception:
                        base_candidates, _boost_candidates, adaptive_threshold = 20, 45, 0.55

                    # Ensure we fetch at least top_k candidates for reranking
                    candidate_k = max(base_candidates, top_k)
                else:
                    candidate_k = top_k

                # Build params in SQL order: [query_vec_param, query_text, ...filters..., query_vec_param, candidate_k]
                params = [
                    query_vec_param,  # For SELECT vec_similarity
                    query_text,  # For ts_rank
                ]
                params.extend(filter_params)  # WHERE filters
                params.extend(
                    [
                        query_vec_param,  # For ORDER BY
                        candidate_k,  # For LIMIT
                    ]
                )

                self.logger.info(
                    "hybrid_search executing",
                    extra_fields={
                        "top_k": top_k,
                        "use_reranker": use_reranker,
                        "has_filters": bool(metadata_filters or compound_filter),
                        "classes_filter": classes_filter,
                        "where_sql": where_sql,
                    },
                )

                # DEBUG: Log the SQL and parameters
                op, sim_expr = self._distance_operator()
                sql_query = f"""
                    SELECT
                        id, tenant_id, classes, props, payload_ref, embedding,
                        metadata, refresh_policy, triggers, version,
                        last_refreshed, drift_score,
                        {sim_expr} as vec_similarity,
                        ts_rank(text_search_vector, websearch_to_tsquery('english', %s)) as ts_rank_score
                    FROM nodes
                    WHERE embedding IS NOT NULL
                      AND text_search_vector IS NOT NULL
                      {where_sql}
                    ORDER BY embedding {op} %s
                    LIMIT %s
                    """
                self.logger.info(
                    "hybrid_search SQL",
                    extra_fields={
                        "sql": sql_query.replace("\n", " ").strip(),
                        "filter_params": str(filter_params),
                        "candidate_k": candidate_k,
                    },
                )

                cur.execute(sql_query, params)

                # Parse results and gather candidates
                candidates: list[tuple[Node, float, float]] = []  # (node, vec_sim, ts_rank)
                max_ts_rank = 0.0

                for row in cur.fetchall():
                    row_t = cast(NodeHybridRow, row)
                    node = self._build_node_from_row(cast(Sequence[Any], row_t))
                    vec_sim = float(row_t[12])
                    ts_rank = float(row_t[13])
                    max_ts_rank = max(max_ts_rank, ts_rank)
                    candidates.append((node, vec_sim, ts_rank))

                # DEBUG: Log raw candidates with their classes
                self.logger.info(
                    "hybrid_search raw candidates",
                    extra_fields={
                        "count": len(candidates),
                        "classes": [node.classes for node, _, _ in candidates[:5]],  # First 5
                    },
                )

                # Fuse scores: RRF (default) or weighted (fallback)
                results: list[tuple[Node, float]] = []

                rrf_enabled = os.getenv("HYBRID_RRF_ENABLED", "true").lower() == "true"
                try:
                    rrf_k = int(os.getenv("HYBRID_RRF_K", "60"))
                except Exception:
                    rrf_k = 60

                if rrf_enabled and candidates:
                    # Compute ranks for vec_sim and ts_rank
                    # Higher is better, so sort descending
                    # Build maps: node_id -> rank
                    by_vec = sorted(
                        ((i, v[1]) for i, v in enumerate(candidates)),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    by_txt = sorted(
                        ((i, v[2]) for i, v in enumerate(candidates)),
                        key=lambda x: x[1],
                        reverse=True,
                    )

                    vec_rank: dict[int, int] = {}
                    txt_rank: dict[int, int] = {}
                    for r, (idx, _) in enumerate(by_vec, start=1):
                        vec_rank[idx] = r
                    for r, (idx, _) in enumerate(by_txt, start=1):
                        txt_rank[idx] = r

                    # RRF fusion: score = 1/(k + rank_vec) + 1/(k + rank_text)
                    fused: list[tuple[Node, float]] = []
                    for i, (node, _vec, _txt) in enumerate(candidates):
                        rv = vec_rank.get(i, len(candidates))
                        rt = txt_rank.get(i, len(candidates))
                        rrf_score = (1.0 / (rrf_k + rv)) + (1.0 / (rrf_k + rt))
                        fused.append((node, rrf_score))

                    # Sort by fused score desc
                    fused.sort(key=lambda x: x[1], reverse=True)
                    results = fused
                else:
                    # Weighted fusion (fallback)
                    for node, vec_sim, ts_rank in candidates:
                        norm_ts_rank = ts_rank / max_ts_rank if max_ts_rank > 0 else 0.0
                        hybrid_score = vector_weight * vec_sim + text_weight * norm_ts_rank
                        results.append((node, hybrid_score))

                # Log fusion mode
                try:
                    self.logger.info(
                        "hybrid_search fusion",
                        extra_fields={
                            "mode": "rrf" if (rrf_enabled and candidates) else "weighted",
                            "candidates": len(candidates),
                        },
                    )
                except Exception:
                    pass

                # Sort by hybrid score
                results.sort(key=lambda x: x[1], reverse=True)

                # Apply cross-encoder reranking if enabled and beneficial
                # Skip reranking if: disabled, no results, <3 candidates, or top score ≥ threshold (already confident)
                # NOTE: Skip threshold only meaningful for cosine/weighted fusion (0.6-1.0 range).
                # RRF scores are 0.01-0.04, so skip is disabled when RRF is active.
                top_score = results[0][1] if results else 0.0
                skip_high_conf = (not rrf_enabled) and (top_score >= rerank_skip_threshold)

                # Adaptive reranking: if top score is uncertain, consider more candidates
                # If we fetched base_candidates but top_score < adaptive_threshold, we may want more depth
                # However, this requires re-query which is expensive. Instead, log for monitoring.
                if use_reranker and top_score < adaptive_threshold:
                    self.logger.info(
                        "Low confidence query detected",
                        extra_fields={
                            "top_score": top_score,
                            "threshold": adaptive_threshold,
                            "recommend": "Consider HYBRID_RERANKER_BASE=40+",
                        },
                    )

                if use_reranker and len(results) >= 3 and not skip_high_conf:
                    # Rerank with optional budget guard
                    budget_ms = None
                    try:
                        budget_ms = float(os.getenv("MAX_RERANK_BUDGET_MS", "0"))
                    except Exception:
                        budget_ms = None

                    start = time.time()
                    reranked = self._cross_encoder_rerank(query_text, results, top_k)
                    duration_ms = (time.time() - start) * 1000.0

                    if budget_ms and budget_ms > 0 and duration_ms > budget_ms:
                        # Budget exceeded: fall back to pre-rerank ordering
                        try:
                            self.logger.info(
                                "Rerank budget exceeded",
                                extra_fields={
                                    "duration_ms": round(duration_ms, 2),
                                    "budget_ms": budget_ms,
                                    "candidates": len(results),
                                },
                            )
                        except Exception:
                            pass
                        out = results[:top_k]
                    else:
                        # Convert back to (node, hybrid_score) for API compatibility
                        # Note: ordering is now by rerank_score, but we return hybrid_score
                        out = [(node, hybrid_score) for node, hybrid_score, _ in reranked]
                else:
                    self.logger.info(
                        "Skipping rerank",
                        extra_fields={
                            "use_reranker": use_reranker,
                            "num_results": len(results),
                            "skip_high_conf": skip_high_conf,
                            "top_score": top_score,
                        },
                    )
                    out = results[:top_k]

                self.logger.info("hybrid_search results", extra_fields={"count": len(out)})
                return out

    def _cross_encoder_rerank(
        self, query: str, candidates: list[tuple[Node, float]], top_k: int
    ) -> list[tuple[Node, float, float]]:
        """Rerank candidates using cross-encoder model.

        Uses sentence-transformers cross-encoder for more accurate relevance scoring.
        Preserves original hybrid_score for gating, adds rerank_score for ordering.

        Args:
            query: Query text
            candidates: List of (node, hybrid_score) tuples to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of (node, hybrid_score, rerank_score) tuples
            where hybrid_score is preserved for thresholding and rerank_score is for ordering
        """
        try:
            import torch
            from sentence_transformers import CrossEncoder

            # Lazy load cross-encoder (caches after first load)
            if not hasattr(self, "_cross_encoder"):
                automodel_args = {
                    "device_map": None,
                    "low_cpu_mem_usage": False,
                    "dtype": torch.float32,
                }
                self._cross_encoder = CrossEncoder(
                    "cross-encoder/ms-marco-MiniLM-L-6-v2",
                    device="cpu",
                    automodel_args=automodel_args,
                )

            # Prepare pairs for cross-encoder
            pairs = []
            for node, _ in candidates:
                # Use title + text for reranking (fallback to common resume/job keys)
                props = node.props or {}
                title = (
                    props.get("title") or props.get("job_title") or props.get("current_title") or ""
                )
                text = (
                    props.get("text")
                    or props.get("resume_text")
                    or props.get("content")
                    or props.get("body")
                    or props.get("description")
                    or ""
                )
                doc_text = " ".join(part for part in (title, text) if part)
                pairs.append([query, doc_text[:512]])  # Limit to 512 chars

            # Get cross-encoder scores (unbounded logits)
            rerank_scores = self._cross_encoder.predict(pairs)

            # Preserve hybrid_score, add rerank_score
            reranked = [
                (candidates[i][0], candidates[i][1], float(rerank_scores[i]))
                for i in range(len(candidates))
            ]

            # Sort by rerank_score (ordering), keep hybrid_score (gating)
            reranked.sort(key=lambda x: x[2], reverse=True)
            return reranked[:top_k]

        except ImportError:
            self.logger.warning("sentence-transformers not installed, skipping reranking")
            # Return with rerank_score = hybrid_score (no reranking)
            return [(node, score, score) for node, score in candidates[:top_k]]
        except Exception as e:
            self.logger.error("Cross-encoder reranking failed", extra_fields={"error": str(e)})
            # Return with rerank_score = hybrid_score (fallback)
            return [(node, score, score) for node, score in candidates[:top_k]]

    def write_embedding_history(
        self,
        node_id: str,
        drift_score: float,
        embedding_ref: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Record embedding update in history table.

        RLS-aware: uses tenant-scoped connection when tenant_id is provided.
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO embedding_history (node_id, drift_score, embedding_ref) VALUES (%s, %s, %s)",
                    (node_id, drift_score, embedding_ref),
                )

    def get_lineage(
        self, node_id: str, max_depth: int = 5, tenant_id: str | None = None
    ) -> list[LineageRecordTD]:
        """Recursively traverse DERIVED_FROM edges to build lineage graph.

        Returns list of ancestors with metadata: [{id, classes, props, depth, edge_props}, ...]
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Recursive CTE to traverse lineage
                cur.execute(
                    """
                    WITH RECURSIVE lineage AS (
                        -- Base case: direct parents
                        SELECT e.dst as node_id, 1 as depth, e.props as edge_props
                        FROM edges e
                        WHERE e.src = %s AND e.rel = 'DERIVED_FROM'

                        UNION ALL

                        -- Recursive case: parents of parents
                        SELECT e.dst, l.depth + 1, e.props
                        FROM edges e
                        JOIN lineage l ON e.src = l.node_id
                        WHERE e.rel = 'DERIVED_FROM' AND l.depth < %s
                    )
                    SELECT DISTINCT l.node_id, l.depth, l.edge_props,
                           n.classes, n.props, n.created_at
                    FROM lineage l
                    JOIN nodes n ON l.node_id = n.id
                    ORDER BY l.depth
                    """,
                    (node_id, max_depth),
                )

                out: list[LineageRecordTD] = []
                for row in cur.fetchall():
                    out.append(
                        {
                            "id": str(row[0]),
                            "depth": int(row[1]),
                            "edge_props": cast(dict[str, Any] | None, row[2]),
                            "classes": cast(list[str], row[3] or []),
                            "props": cast(dict[str, Any], row[4] or {}),
                            "created_at": row[5].isoformat() if row[5] else None,
                        }
                    )
                return out

    def find_nodes_due_for_refresh(self, current_time: str | None = None) -> list[Node]:
        """Find nodes due for refresh based on their refresh_policy.

        Supports:
        - interval: "1m", "5m", "1h", "1d" (minutes, hours, days)
        - cron: standard cron expression (not yet implemented)
        - last_refreshed comparison via metadata
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                # Query nodes with refresh_policy set
                cur.execute(
                    """
                    SELECT id, tenant_id, classes, props, payload_ref, embedding,
                           metadata, refresh_policy, triggers, version,
                           last_refreshed, drift_score
                    FROM nodes
                    WHERE refresh_policy IS NOT NULL
                      AND refresh_policy != '{}'::jsonb
                    """
                )

                out: list[Node] = []
                for row in cur.fetchall():
                    node = self._build_node_from_row(cast(Sequence[Any], row))
                    if self._is_due_for_refresh(node):
                        out.append(node)

                return out

    def _is_due_for_refresh(self, node: Node) -> bool:
        """Check if a node is due for refresh based on its policy.

        Uses explicit last_refreshed column (not metadata JSONB) for performance.

        Supports two policy types:
        1. Cron: {"cron": "*/5 * * * *"} - Standard cron expression (UTC)
        2. Interval: {"interval": "5m"} - Simple interval (5m, 1h, 2d)

        Precedence: cron > interval (cron takes priority if both present)
        """
        from datetime import datetime

        policy = node.refresh_policy
        if not policy:
            return False

        # Get last_refreshed from node (queried from DB column)
        last_refreshed = getattr(node, "last_refreshed", None)

        now = datetime.now(timezone.utc)

        # PRECEDENCE: cron > interval (with fallback)
        # Parse cron policy: {"cron": "*/5 * * * *"}
        if "cron" in policy:
            try:
                from croniter import croniter  # type: ignore[import-untyped]

                cron_expr = policy["cron"]

                # If never refreshed, it's due
                if last_refreshed is None:
                    return True

                # Create croniter instance from last_refreshed
                # croniter uses UTC by default when datetime has timezone
                cron = croniter(cron_expr, last_refreshed)

                # Get next scheduled run after last_refreshed
                next_run = cron.get_next(datetime)

                # Debug logging
                self.logger.debug(
                    f"Cron check: now={now}, last_refreshed={last_refreshed}, next_run={next_run}, is_due={now >= next_run}",
                    extra_fields={"cron": cron_expr, "node_id": node.id},
                )

                # If current time is past next_run, it's due
                return bool(now >= next_run)

            except Exception as e:
                self.logger.warning(
                    f"Invalid cron expression, falling back to interval if present: {policy.get('cron')}",
                    extra_fields={"error": str(e), "node_id": node.id},
                )
                # Fall through to interval check instead of returning False

        # Parse interval policy: {"interval": "5m"} or {"interval": "1h"} or {"interval": "1d"}
        if "interval" in policy:
            interval_str = policy["interval"]

            # Legacy support for 'minute'
            if interval_str == "minute":
                interval_seconds = 60
            else:
                # Parse format like "5m", "1h", "2d"
                import re

                match = re.match(r"^(\d+)([mhd])$", interval_str.lower())
                if not match:
                    self.logger.warning(f"Invalid interval format: {interval_str}")
                    return False

                value, unit = int(match.group(1)), match.group(2)
                if unit == "m":
                    interval_seconds = value * 60
                elif unit == "h":
                    interval_seconds = value * 3600
                elif unit == "d":
                    interval_seconds = value * 86400
                else:
                    return False

            # Check if enough time has passed
            if last_refreshed is None:
                return True  # Never refreshed, so it's due

            elapsed = (now - last_refreshed).total_seconds()
            return bool(elapsed >= interval_seconds)

        return False

    def all_nodes(self) -> list[Node]:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, tenant_id, classes, props, payload_ref, embedding, metadata,
                           refresh_policy, triggers, version, last_refreshed, drift_score
                    FROM nodes
                    """
                )
                out: list[Node] = []
                for row in cur.fetchall():
                    out.append(self._build_node_from_row(cast(Sequence[Any], row)))
                return out

    # --- Edges ---
    def create_edge(self, edge: Edge) -> None:
        with self._conn(tenant_id=edge.tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO edges (src, rel, dst, props, tenant_id) VALUES (%s, %s, %s, %s, %s)",
                    (edge.src, edge.rel, edge.dst, json.dumps(edge.props), edge.tenant_id),
                )

    # --- Events ---
    def append_event(
        self,
        node_id: str,
        event_type: str,
        payload: EventPayloadTD,
        tenant_id: str | None = None,
        actor_id: str | None = None,
        actor_type: str | None = None,
    ) -> str:
        """Append event with audit trail support.

        Args:
            node_id: Node this event relates to
            event_type: Type of event (refreshed, trigger_fired, etc.)
            payload: Event data (typed best-effort)
            tenant_id: Tenant ID for RLS
            actor_id: Who triggered this (user ID, api key, 'scheduler', 'trigger')
            actor_type: Type of actor (user, api_key, scheduler, trigger, system)

        Returns:
            Event ID
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO events (node_id, type, payload, tenant_id, actor_id, actor_type)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (node_id, event_type, json.dumps(payload), tenant_id, actor_id, actor_type),
                )
                event_id = cur.fetchone()[0]
                return str(event_id)

    # --- Payload loading ---
    def load_payload_text(self, node: Node) -> str:
        """Load payload text from various sources.

        Supports:
        - payload_ref starting with 's3://' -> fetch from S3
        - payload_ref starting with 'file://' -> read from local file
        - payload_ref starting with 'http://' or 'https://' -> fetch from URL
        - props['text'], props['resume_text'], props['content'], etc. -> inline text
        - Empty payload_ref -> check common text keys or return empty string
        """
        # Try payload_ref first
        if node.payload_ref:
            if node.payload_ref.startswith("s3://"):
                return self._load_from_s3(node.payload_ref)
            elif node.payload_ref.startswith("file://"):
                return self._load_from_file(node.payload_ref[7:])  # Strip 'file://'
            elif node.payload_ref.startswith(("http://", "https://")):
                return self._load_from_url(node.payload_ref)
            else:
                # Assume local file path
                return self._load_from_file(node.payload_ref)

        # Fallback to inline text - check multiple common keys
        props = node.props or {}
        for key in ("text", "resume_text", "content", "body", "description"):
            value = props.get(key)
            if value and isinstance(value, str):
                return cast(str, value)
        return ""

    def build_embedding_text(self, node: Node) -> str:
        """Build embedding text with optional structured prefix for resume/job nodes.

        If EMBEDDING_PREFIX_ENABLED=true and the node is a resume/job node (has 'Resume'
        in classes or 'resume_text' in props), prepends structured fields to improve
        embedding quality for job matching.

        Prefix fields (when present):
        - job_title
        - required_skills
        - good_to_have_skills
        - experience_years
        - education_requirement
        - primary_skills (extraction)
        - recent_job_titles (extraction)
        - certifications (extraction)
        - industries (extraction)
        - years_experience_total (extraction)

        Returns:
            Embedding text (with prefix if applicable, otherwise raw payload text)
        """
        import os

        base_text = self.load_payload_text(node)
        if not base_text:
            return ""

        # Check if prefix is enabled
        if os.getenv("EMBEDDING_PREFIX_ENABLED", "false").lower() != "true":
            return base_text

        # Guard: only apply prefix for resume/job nodes
        props = node.props or {}
        classes = node.classes or []
        is_resume_or_job = (
            "Resume" in classes
            or "Job" in classes
            or "resume_text" in props
            or "job_title" in props
        )
        if not is_resume_or_job:
            return base_text

        # Build structured prefix
        prefix_lines = []
        prefix_fields = [
            ("current_title", "CURRENT_TITLE"),
            ("primary_titles", "PRIMARY_TITLES"),
            ("seniority", "SENIORITY"),
            ("job_title", "JOB_TITLE"),
            ("required_skills", "REQUIRED_SKILLS"),
            ("good_to_have_skills", "GOOD_TO_HAVE"),
            ("skills_raw", "SKILLS_RAW"),
            ("skills_normalized", "SKILLS_NORMALIZED"),
            ("years_by_skill", "YEARS_BY_SKILL"),
            ("experience_years", "EXPERIENCE"),
            ("education_requirement", "EDUCATION"),
            ("primary_skills", "EXTRACTED_SKILLS"),
            ("recent_job_titles", "RECENT_TITLES"),
            ("certifications", "CERTIFICATIONS"),
            ("industries", "INDUSTRIES"),
            ("domains", "DOMAINS"),
            ("functions", "FUNCTIONS"),
            ("location", "LOCATION"),
            ("years_experience_total", "YEARS_EXPERIENCE_TOTAL"),
            ("total_years_experience", "TOTAL_YEARS_EXPERIENCE"),
        ]

        stable_sort_fields = {
            "primary_skills",
            "recent_job_titles",
            "certifications",
            "industries",
            "skills_raw",
            "skills_normalized",
            "primary_titles",
            "domains",
            "functions",
        }

        for prop_key, label in prefix_fields:
            value = props.get(prop_key)
            if value:
                # Handle both string and list values
                if isinstance(value, list):
                    items = [str(v) for v in value if str(v).strip()]
                    if prop_key in stable_sort_fields:
                        items = sorted(items, key=lambda s: s.lower())
                    value = ", ".join(items)
                elif isinstance(value, dict):
                    items = []
                    for k in sorted(value.keys(), key=lambda s: str(s).lower()):
                        items.append(f"{k}={value[k]}")
                    value = "; ".join(items)
                elif not isinstance(value, str):
                    value = str(value)
                if value.strip():
                    prefix_lines.append(f"{label}: {value.strip()}")

        if not prefix_lines:
            return base_text

        prefix = "\n".join(prefix_lines)
        return f"{prefix}\n\n{base_text}"

    def _load_from_file(self, file_path: str) -> str:
        """Load text from local file."""
        try:
            import os

            # Security: realpath normalization + base dir allowlist
            allowlist_raw = os.getenv("ACTIVEKG_FILE_BASEDIRS", "")
            allowed_dirs = [d.strip() for d in allowlist_raw.split(",") if d.strip()]
            # Default to repo cwd if not provided
            if not allowed_dirs:
                allowed_dirs = [os.getcwd()]

            real = os.path.realpath(file_path)
            # Reject symlinks
            if os.path.islink(file_path):
                self.logger.warning(f"Symlink path rejected: {file_path}")
                return ""
            # Ensure within an allowed base dir
            try:
                allowed = any(
                    os.path.commonpath([real, base]) == os.path.abspath(base)
                    for base in allowed_dirs
                )
            except ValueError:
                # Different-drive edge cases (Windows) or invalid paths
                allowed = False
            if not allowed:
                self.logger.warning(
                    "Path outside allowed base directories rejected",
                    extra_fields={"path": file_path, "real": real, "allowed": allowed_dirs},
                )
                return ""

            if not os.path.exists(real) or not os.path.isfile(real):
                self.logger.warning(f"File not found or not a file: {real}")
                return ""

            # Enforce max bytes (default 1MB) for file read
            max_bytes = int(os.getenv("ACTIVEKG_MAX_FILE_BYTES", "1048576"))
            with open(real, "rb") as f:
                data = f.read(max_bytes + 1)
            if len(data) > max_bytes:
                self.logger.warning(
                    "File too large; truncated", extra_fields={"path": real, "max_bytes": max_bytes}
                )
                data = data[:max_bytes]
            return data.decode("utf-8", errors="ignore")
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {e}")
            return ""

    def _load_from_s3(self, s3_uri: str) -> str:
        """Load text from S3 bucket."""
        try:
            import boto3

            # Parse s3://bucket/key
            parts = s3_uri[5:].split("/", 1)
            if len(parts) != 2:
                self.logger.warning(f"Invalid S3 URI: {s3_uri}")
                return ""

            bucket, key = parts
            s3_client = boto3.client("s3")
            response = s3_client.get_object(Bucket=bucket, Key=key)
            return cast(str, response["Body"].read().decode("utf-8", errors="ignore"))

        except ImportError:
            self.logger.warning("boto3 not installed, cannot load from S3")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to load from S3 {s3_uri}: {e}")
            return ""

    def _load_from_url(self, url: str) -> str:
        """Load text from HTTP/HTTPS URL."""
        try:
            import requests

            # Security: SSRF protections
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                self.logger.warning("Blocked non-http(s) URL", extra_fields={"url": url})
                return ""

            # Domain allowlist (optional)
            allow_raw = os.getenv("ACTIVEKG_URL_ALLOWLIST", "")
            allow_hosts = {h.strip().lower() for h in allow_raw.split(",") if h.strip()}
            host = parsed.hostname or ""
            if allow_hosts and host.lower() not in allow_hosts:
                self.logger.warning("URL host not in allowlist", extra_fields={"host": host})
                return ""

            # Resolve and block private/localhost ranges
            try:
                addrs = {ai[4][0] for ai in socket.getaddrinfo(host, None)}
            except Exception:
                self.logger.warning("DNS resolution failed", extra_fields={"host": host})
                return ""
            for addr in addrs:
                try:
                    ip = ipaddress.ip_address(addr)
                    if (
                        ip.is_private
                        or ip.is_loopback
                        or ip.is_link_local
                        or ip.is_reserved
                        or ip.is_multicast
                    ):
                        self.logger.warning(
                            "Blocked private/loopback address", extra_fields={"ip": addr}
                        )
                        return ""
                except ValueError:
                    continue

            # Fetch with limits
            max_bytes = int(os.getenv("ACTIVEKG_MAX_FETCH_BYTES", "10485760"))  # 10MB default
            timeout = float(os.getenv("ACTIVEKG_FETCH_TIMEOUT", "10"))
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()

            # Content-Type allow (default: text/*, application/json)
            ctype = r.headers.get("Content-Type", "").lower()
            allowed_types = {"text/", "application/json"}
            if not any(ctype.startswith(p) for p in allowed_types):
                self.logger.warning("Blocked content-type", extra_fields={"content_type": ctype})
                return ""

            # Respect Content-Length if present
            try:
                clen = int(r.headers.get("Content-Length", "0"))
                if clen > 0 and clen > max_bytes:
                    self.logger.warning(
                        "Content-Length too large", extra_fields={"content_length": clen}
                    )
                    return ""
            except Exception:
                pass

            # Stream up to max_bytes
            buf = bytearray()
            for chunk in r.iter_content(chunk_size=4096):
                if not chunk:
                    break
                buf.extend(chunk)
                if len(buf) > max_bytes:
                    self.logger.warning(
                        "Fetched content exceeded max_bytes; truncated",
                        extra_fields={"max_bytes": max_bytes},
                    )
                    buf = buf[:max_bytes]
                    break
            return bytes(buf).decode("utf-8", errors="ignore")

        except ImportError:
            self.logger.warning("requests not installed, cannot load from URL")
            return ""
        except Exception as e:
            self.logger.error(f"Failed to load from URL {url}: {e}")
            return ""

    # --- Anomaly Detection ---
    def detect_drift_spikes(
        self,
        lookback_hours: int = 24,
        spike_threshold: float = 2.0,
        min_refreshes: int = 3,
        tenant_id: str | None = None,
    ) -> list[DriftSpikeAnomalyTD]:
        """Detect nodes with drift scores consistently above mean (drift spike anomaly).

        A drift spike is when a node's drift score exceeds spike_threshold * mean_drift
        for at least min_refreshes consecutive refreshes within the lookback window.

        Args:
            lookback_hours: Hours to look back in embedding_history
            spike_threshold: Multiplier for mean drift (e.g., 2.0 = 2x mean)
            min_refreshes: Minimum number of consecutive high-drift refreshes
            tenant_id: Optional tenant ID for multi-tenancy filtering

        Returns:
            List of anomalies with node_id, recent_drift_scores, avg_drift, mean_drift
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Calculate mean drift across all nodes in the lookback window
                cur.execute(
                    """
                    SELECT AVG(drift_score) as mean_drift
                    FROM embedding_history
                    WHERE created_at > NOW() - INTERVAL '%s hours'
                    """,
                    (lookback_hours,),
                )
                result = cur.fetchone()
                mean_drift = (
                    float(result[0]) if result and result[0] else 0.1
                )  # Default to 0.1 if no data

                spike_cutoff = mean_drift * spike_threshold

                # Find nodes with consecutive high-drift refreshes
                cur.execute(
                    """
                    WITH recent_history AS (
                        SELECT
                            node_id,
                            drift_score,
                            created_at,
                            ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY created_at DESC) as rn
                        FROM embedding_history
                        WHERE created_at > NOW() - INTERVAL '%s hours'
                          AND drift_score > %s
                    ),
                    consecutive_spikes AS (
                        SELECT
                            node_id,
                            COUNT(*) as spike_count,
                            AVG(drift_score) as avg_drift,
                            ARRAY_AGG(drift_score ORDER BY created_at DESC) as drift_scores
                        FROM recent_history
                        WHERE rn <= %s
                        GROUP BY node_id
                        HAVING COUNT(*) >= %s
                    )
                    SELECT
                        cs.node_id,
                        cs.spike_count,
                        cs.avg_drift,
                        cs.drift_scores,
                        n.classes,
                        n.props
                    FROM consecutive_spikes cs
                    JOIN nodes n ON cs.node_id = n.id
                    ORDER BY cs.avg_drift DESC
                    """,
                    (lookback_hours, spike_cutoff, min_refreshes, min_refreshes),
                )

                anomalies: list[DriftSpikeAnomalyTD] = []
                for row in cur.fetchall():
                    anomalies.append(
                        {
                            "type": "drift_spike",
                            "node_id": str(row[0]),
                            "spike_count": row[1],
                            "avg_drift": float(row[2]),
                            "drift_scores": [float(d) for d in row[3]],
                            "mean_drift": mean_drift,
                            "spike_threshold": spike_threshold,
                            "classes": row[4],
                            "props": row[5],
                        }
                    )

                return anomalies

    def detect_trigger_storms(
        self, lookback_hours: int = 1, event_threshold: int = 50, tenant_id: str | None = None
    ) -> list[TriggerStormAnomalyTD]:
        """Detect trigger storm anomalies (excessive trigger_fired events).

        A trigger storm occurs when more than event_threshold trigger_fired events
        occur within lookback_hours, indicating potential runaway triggers.

        Args:
            lookback_hours: Hours to look back in events table
            event_threshold: Minimum number of trigger_fired events to flag as storm
            tenant_id: Optional tenant ID for multi-tenancy filtering

        Returns:
            List of anomalies with node_id, event_count, recent_events
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH trigger_counts AS (
                        SELECT
                            node_id,
                            COUNT(*) as event_count,
                            MIN(created_at) as first_event,
                            MAX(created_at) as last_event,
                            ARRAY_AGG(
                                jsonb_build_object(
                                    'type', type,
                                    'created_at', created_at,
                                    'payload', payload
                                )
                                ORDER BY created_at DESC
                                LIMIT 10
                            ) as recent_events
                        FROM events
                        WHERE created_at > NOW() - INTERVAL '%s hours'
                          AND type = 'trigger_fired'
                        GROUP BY node_id
                        HAVING COUNT(*) > %s
                    )
                    SELECT
                        tc.node_id,
                        tc.event_count,
                        tc.first_event,
                        tc.last_event,
                        tc.recent_events,
                        n.classes,
                        n.props
                    FROM trigger_counts tc
                    JOIN nodes n ON tc.node_id = n.id
                    ORDER BY tc.event_count DESC
                    """,
                    (lookback_hours, event_threshold),
                )

                anomalies: list[TriggerStormAnomalyTD] = []
                for row in cur.fetchall():
                    # Normalize recent events structure
                    recent: list[RecentEventTD] = []
                    raw_recent = row[4] or []
                    try:
                        iterable = list(raw_recent)[:10]
                    except Exception:
                        iterable = []
                    for ev in iterable:
                        try:
                            ev_type = str(ev.get("type")) if isinstance(ev, dict) else None
                            created = ev.get("created_at") if isinstance(ev, dict) else None
                            created_s = (
                                created.isoformat()  # type: ignore[union-attr]
                                if hasattr(created, "isoformat")
                                else (str(created) if created is not None else None)
                            )
                            payload = ev.get("payload") if isinstance(ev, dict) else {}
                            recent.append(
                                {
                                    "type": ev_type or "unknown",
                                    "created_at": created_s,
                                    "payload": cast(dict[str, Any], payload or {}),
                                }
                            )
                        except Exception:
                            continue

                    anomalies.append(
                        {
                            "type": "trigger_storm",
                            "node_id": str(row[0]),
                            "event_count": int(row[1]),
                            "first_event": row[2].isoformat() if row[2] else None,
                            "last_event": row[3].isoformat() if row[3] else None,
                            "recent_events": recent,
                            "event_threshold": event_threshold,
                            "classes": cast(list[str], row[5] or []),
                            "props": cast(dict[str, Any], row[6] or {}),
                        }
                    )

                return anomalies

    def detect_scheduler_lag(
        self, lag_multiplier: float = 2.0, tenant_id: str | None = None
    ) -> list[SchedulerLagAnomalyTD]:
        """Detect nodes overdue for refresh (scheduler lag anomaly).

        A node is overdue if it hasn't been refreshed within lag_multiplier times
        its scheduled refresh interval.

        Args:
            lag_multiplier: How many intervals past due (e.g., 2.0 = 2x late)
            tenant_id: Optional tenant ID for multi-tenancy filtering

        Returns:
            List of anomalies with node_id, expected_interval, actual_lag, lag_ratio
        """
        from datetime import datetime

        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Query all nodes with refresh_policy
                cur.execute(
                    """
                    SELECT
                        id,
                        classes,
                        props,
                        refresh_policy,
                        last_refreshed
                    FROM nodes
                    WHERE refresh_policy IS NOT NULL
                      AND refresh_policy != '{}'::jsonb
                    """
                )

                anomalies = []
                now = datetime.now(timezone.utc)

                for row in cur.fetchall():
                    node_id = str(row[0])
                    classes = row[1]
                    props = row[2]
                    policy = cast(RefreshPolicyTD, row[3] or {})
                    last_refreshed = row[4]

                    # Calculate expected interval in seconds
                    expected_interval_seconds = None

                    if "interval" in policy:
                        interval_str = policy["interval"]
                        import re

                        match = re.match(r"^(\d+)([mhd])$", interval_str.lower())
                        if match:
                            value, unit = int(match.group(1)), match.group(2)
                            if unit == "m":
                                expected_interval_seconds = value * 60
                            elif unit == "h":
                                expected_interval_seconds = value * 3600
                            elif unit == "d":
                                expected_interval_seconds = value * 86400

                    elif "cron" in policy:
                        # For cron, estimate interval from expression (rough heuristic)
                        try:
                            from croniter import croniter

                            if last_refreshed:
                                cron = croniter(policy["cron"], last_refreshed)
                                next_run = cron.get_next(datetime)
                                expected_interval_seconds = (
                                    next_run - last_refreshed
                                ).total_seconds()
                        except Exception:
                            pass

                    # Check if overdue
                    if expected_interval_seconds and last_refreshed:
                        actual_lag_seconds = (now - last_refreshed).total_seconds()
                        lag_ratio = actual_lag_seconds / expected_interval_seconds

                        if lag_ratio >= lag_multiplier:
                            anomalies.append(
                                {
                                    "type": "scheduler_lag",
                                    "node_id": node_id,
                                    "expected_interval_seconds": expected_interval_seconds,
                                    "actual_lag_seconds": actual_lag_seconds,
                                    "lag_ratio": round(lag_ratio, 2),
                                    "lag_multiplier": lag_multiplier,
                                    "last_refreshed": last_refreshed.isoformat(),
                                    "classes": classes,
                                    "props": props,
                                }
                            )

                    elif not last_refreshed:
                        # Never refreshed but has policy - definitely overdue
                        anomalies.append(
                            {
                                "type": "scheduler_lag",
                                "node_id": node_id,
                                "expected_interval_seconds": expected_interval_seconds,
                                "actual_lag_seconds": None,
                                "lag_ratio": float("inf"),
                                "lag_multiplier": lag_multiplier,
                                "last_refreshed": None,
                                "classes": classes,
                                "props": props,
                            }
                        )

                # Sort by lag_ratio descending (worst first)
                anomalies.sort(key=lambda x: x["lag_ratio"], reverse=True)
                return cast("list[SchedulerLagAnomalyTD]", anomalies)

    def get_node_versions(
        self, node_id: str, limit: int = 10, tenant_id: str | None = None
    ) -> list[NodeVersionTD]:
        """Get embedding history versions for a node.

        RLS-aware: uses tenant-scoped connection when tenant_id is provided.

        Args:
            node_id: Node ID to query
            limit: Maximum number of versions to return (default: 10)
            tenant_id: Tenant ID for RLS enforcement (optional)

        Returns:
            List of version records with drift_score, created_at, embedding_ref
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        drift_score,
                        created_at,
                        embedding_ref
                    FROM embedding_history
                    WHERE node_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (node_id, limit),
                )

                versions: list[NodeVersionTD] = []
                for idx, row in enumerate(cur.fetchall()):
                    versions.append(
                        {
                            "version_index": idx,
                            "drift_score": float(row[0]) if row[0] is not None else None,
                            "created_at": row[1].isoformat() if row[1] else None,
                            "embedding_ref": cast(str | None, row[2]),
                        }
                    )
                return versions

    def list_open_positions(
        self,
        role_terms: list[str] | None = None,
        limit: int = 10,
        tenant_id: str | None = None,
    ) -> list[tuple[Node, float]]:
        """Structured query for open job positions with role matching.

        Uses class filtering + status check + BM25 text search for role terms.
        Addresses Q5: "What ML engineer positions are open?"

        Args:
            role_terms: Optional role keywords (e.g., ["ml", "machine learning", "engineer"])
            limit: Maximum results to return
            tenant_id: Optional tenant ID for RLS

        Returns:
            List of (Node, relevance_score) tuples for open positions
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Build query based on whether role_terms are provided
                # Be explicit to avoid ambiguous truth checks with numpy-like objects
                normalized_terms: list[str] = []
                if role_terms is not None:
                    for t in role_terms:
                        if isinstance(t, str):
                            ts = t.strip()
                            if ts:
                                normalized_terms.append(ts)

                if len(normalized_terms) > 0:
                    # Use ts_rank for relevance when we have role terms
                    role_query = " | ".join(normalized_terms)  # OR search in tsquery

                    cur.execute(
                        """
                        SELECT
                            id, tenant_id, classes, props, payload_ref, embedding,
                            metadata, refresh_policy, triggers, version,
                            last_refreshed, drift_score,
                            ts_rank(text_search_vector, websearch_to_tsquery('english', %s)) as relevance
                        FROM nodes
                        WHERE classes @> ARRAY['Job']::text[]
                          AND (
                              (props->>'status' = 'open')
                              OR (metadata->>'status' = 'open')
                              OR (props->>'status' IS NULL AND metadata->>'status' IS NULL)
                          )
                          AND text_search_vector @@ websearch_to_tsquery('english', %s)
                        ORDER BY relevance DESC, created_at DESC
                        LIMIT %s
                        """,
                        (role_query, role_query, limit),
                    )
                else:
                    # No role terms - just filter by class and status
                    cur.execute(
                        """
                        SELECT
                            id, tenant_id, classes, props, payload_ref, embedding,
                            metadata, refresh_policy, triggers, version,
                            last_refreshed, drift_score,
                            1.0 as relevance
                        FROM nodes
                        WHERE classes @> ARRAY['Job']::text[]
                          AND (
                              (props->>'status' = 'open')
                              OR (metadata->>'status' = 'open')
                              OR (props->>'status' IS NULL AND metadata->>'status' IS NULL)
                          )
                        ORDER BY created_at DESC
                        LIMIT %s
                        """,
                        (limit,),
                    )

                results: list[tuple[Node, float]] = []
                for row in cur.fetchall():
                    row_t = cast(NodeRelevanceRow, row)
                    node = self._build_node_from_row(cast(Sequence[Any], row_t))
                    score = float(row_t[12]) if row_t[12] is not None else 1.0
                    results.append((node, score))

                return results

    def list_performance_issues(
        self, lookback_days: int = 30, limit: int = 10, tenant_id: str | None = None
    ) -> list[tuple[Node, float]]:
        """Structured query for performance-related issues/tickets.

        Uses class filtering + performance keywords + recency.
        Addresses Q8: "What are the main performance issues reported?"

        Args:
            lookback_days: Only return issues from last N days (default: 30)
            limit: Maximum results to return
            tenant_id: Optional tenant ID for RLS

        Returns:
            List of (Node, relevance_score) tuples for performance issues
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                # Performance-related keywords for text search
                perf_query = "performance | slow | latency | timeout | bottleneck | degradation"

                cur.execute(
                    """
                    SELECT
                        id, tenant_id, classes, props, payload_ref, embedding,
                        metadata, refresh_policy, triggers, version,
                        last_refreshed, drift_score,
                        ts_rank(text_search_vector, to_tsquery('english', %s)) as relevance
                    FROM nodes
                    WHERE (classes @> ARRAY['Ticket']::text[]
                           OR classes @> ARRAY['Incident']::text[]
                           OR classes @> ARRAY['Issue']::text[]
                           OR classes @> ARRAY['Bug']::text[])
                      AND text_search_vector @@ to_tsquery('english', %s)
                      AND created_at >= NOW() - make_interval(days => %s)
                    ORDER BY relevance DESC, created_at DESC
                    LIMIT %s
                    """,
                    (perf_query, perf_query, lookback_days, limit),
                )

                results: list[tuple[Node, float]] = []
                for row in cur.fetchall():
                    row_t = cast(NodeRelevanceRow, row)
                    node = self._build_node_from_row(cast(Sequence[Any], row_t))
                    score = float(row_t[12]) if row_t[12] is not None else 1.0
                    results.append((node, score))

                return results

    def purge_deleted_nodes(
        self, tenant_id: str | None = None, batch_size: int = 500, dry_run: bool = False
    ) -> dict:
        """Permanently delete soft-deleted nodes past their grace period.

        Deletes nodes with 'Deleted' class where deletion_grace_until < NOW().
        Deletes chunks first (via parent_id), then parent documents.

        Args:
            tenant_id: Optional tenant ID for RLS-scoped purge
            batch_size: Maximum number of nodes to purge per call (default: 500)
            dry_run: If True, count candidates without deleting (default: False)

        Returns:
            Dict with purged_chunks, purged_parents counts

        Safety:
            - Uses RLS for tenant isolation
            - Respects batch_size to avoid long locks
            - Dry-run mode for preview
        """
        with self._conn(tenant_id=tenant_id) as conn:
            with conn.cursor() as cur:
                if dry_run:
                    # Count candidates without deleting
                    # Count chunks: nodes where parent has Deleted class and grace expired
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM nodes n
                        INNER JOIN nodes p ON n.props->>'parent_id' = p.props->>'external_id'
                        WHERE 'Deleted' = ANY(p.classes)
                          AND (p.props->>'deletion_grace_until')::timestamptz < NOW()
                        LIMIT %s
                        """,
                        (batch_size,),
                    )
                    chunk_count = cur.fetchone()[0] or 0

                    # Count parents: nodes with Deleted class and grace expired
                    cur.execute(
                        """
                        SELECT COUNT(*)
                        FROM nodes p
                        WHERE 'Deleted' = ANY(p.classes)
                          AND (p.props->>'deletion_grace_until')::timestamptz < NOW()
                        LIMIT %s
                        """,
                        (batch_size,),
                    )
                    parent_count = cur.fetchone()[0] or 0

                    self.logger.info(
                        "Purge dry-run complete",
                        extra_fields={
                            "chunk_candidates": chunk_count,
                            "parent_candidates": parent_count,
                            "tenant_id": tenant_id,
                        },
                    )

                    return {
                        "purged_chunks": chunk_count,
                        "purged_parents": parent_count,
                        "dry_run": True,
                    }

                # Actual purge: delete chunks first, then parents
                # Step 1: Delete chunks whose parents are marked Deleted and past grace
                cur.execute(
                    """
                    WITH to_delete AS (
                        SELECT n.id
                        FROM nodes n
                        INNER JOIN nodes p ON n.props->>'parent_id' = p.props->>'external_id'
                        WHERE 'Deleted' = ANY(p.classes)
                          AND (p.props->>'deletion_grace_until')::timestamptz < NOW()
                        LIMIT %s
                    )
                    DELETE FROM nodes
                    WHERE id IN (SELECT id FROM to_delete)
                    """,
                    (batch_size,),
                )
                purged_chunks = cur.rowcount

                # Step 2: Delete parent documents marked Deleted and past grace
                cur.execute(
                    """
                    DELETE FROM nodes
                    WHERE 'Deleted' = ANY(classes)
                      AND (props->>'deletion_grace_until')::timestamptz < NOW()
                    LIMIT %s
                    """,
                    (batch_size,),
                )
                purged_parents = cur.rowcount

                self.logger.info(
                    "Purge complete",
                    extra_fields={
                        "purged_chunks": purged_chunks,
                        "purged_parents": purged_parents,
                        "tenant_id": tenant_id,
                        "batch_size": batch_size,
                    },
                )

                return {
                    "purged_chunks": purged_chunks,
                    "purged_parents": purged_parents,
                    "dry_run": False,
                }
