"""Bounded readiness and metrics helpers for private operational endpoints."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from activekg.common.migration_manifest import MIGRATIONS

READINESS_SUCCESS_TTL_SECONDS = 30.0
READINESS_FAILURE_TTL_SECONDS = 5.0
READINESS_POOL_TIMEOUT_SECONDS = 0.25
READINESS_STATEMENT_TIMEOUT_MS = 250
READINESS_TOTAL_BUDGET_SECONDS = 2.0
METRICS_MAX_BYTES = 1024 * 1024

_CANDIDATE_TABLES = (
    "candidates",
    "candidate_identifiers",
    "candidate_source_records",
    "candidate_contact_evidence",
)
_SHARED_TABLES = (
    "contact_suppression_tombstones",
    "contact_person_suppressions",
    "contact_suppression_receipts",
    "public_candidate_market_memberships",
)
_REQUIRED_INDEXES = {
    "idx_gc_public_crustdata_person_id",
    "idx_gc_public_embedding_status",
    "idx_cce_one_primary",
    "idx_cce_email_hash",
    "idx_contact_suppression_provider_event",
    "idx_contact_suppression_receipts_email_hash",
    "idx_contact_suppression_receipts_candidate",
    "idx_contact_suppression_receipts_tenant_created",
    "idx_pcmm_market_last_observed",
}
_REQUIRED_FUNCTIONS = {
    "activekg_pick_public_fields",
    "activekg_pick_public_rows",
    "activekg_public_crustdata_projection",
    "activekg_assert_public_crustdata_backfill_safe",
    "contact_suppression_receipts_append_only",
}
_REQUIRED_CONSTRAINTS = {
    "global_candidates_public_embedding_status_check",
    "global_candidates_public_headline_from_profile",
    "candidate_contact_evidence_unique",
    "candidate_contact_evidence_primary_usable",
    "contact_suppression_reason_check",
    "contact_suppression_provider_event_hash",
    "contact_person_suppressions_pkey",
    "contact_person_suppressions_global_candidate_fkey",
    "contact_person_suppression_reason_check",
    "contact_person_suppression_provider_event_hash",
    "contact_suppression_receipts_pkey",
    "contact_suppression_receipt_email_hash_check",
    "contact_suppression_receipt_signal_candidate_nonblank",
    "contact_suppression_receipt_tenant_nonblank",
    "contact_suppression_receipt_provider_event_hash",
    "contact_suppression_receipt_authority_check",
    "contact_suppression_receipt_scope_reason_check",
    "contact_suppression_receipts_provider_event_unique",
    "public_candidate_market_country_code_check",
    "public_candidate_market_memberships_pkey",
}
_SENSITIVE_LABELS = {
    "tenant",
    "tenant_id",
    "org",
    "org_id",
    "organization",
    "organization_id",
}
_PROM_LABEL_RE = re.compile(r"(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
logger = logging.getLogger(__name__)


class OperationalBusy(RuntimeError):
    """An operational snapshot is already being built."""


class OperationalPayloadTooLarge(RuntimeError):
    """A metrics snapshot exceeded its fixed response budget."""


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    reasons: tuple[str, ...] = ()


class ReadinessCoordinator:
    """Single-flight readiness checks with short success/failure caches."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: ReadinessResult | None = None

    def run(self, check: Callable[[], ReadinessResult]) -> ReadinessResult:
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cached
            cached_at = self._cached_at
        if cached is not None:
            ttl = READINESS_SUCCESS_TTL_SECONDS if cached.ready else READINESS_FAILURE_TTL_SECONDS
            if now - cached_at < ttl:
                return cached

        if not self._lock.acquire(blocking=False):
            raise OperationalBusy("readiness check already in progress")
        try:
            # Recheck after winning the single-flight lock.
            now = time.monotonic()
            with self._cache_lock:
                cached = self._cached
                cached_at = self._cached_at
            if cached is not None:
                ttl = (
                    READINESS_SUCCESS_TTL_SECONDS if cached.ready else READINESS_FAILURE_TTL_SECONDS
                )
                if now - cached_at < ttl:
                    return cached

            result = check()
            with self._cache_lock:
                self._cached = result
                self._cached_at = time.monotonic()
            return result
        finally:
            self._lock.release()


class MetricsBoundary:
    """Serialize and size-bound both operational metrics representations."""

    def __init__(self, max_bytes: int = METRICS_MAX_BYTES) -> None:
        self._lock = threading.Lock()
        self._max_bytes = max_bytes

    def json_bytes(self, snapshot: Mapping[str, Any] | Callable[[], Mapping[str, Any]]) -> bytes:
        if not self._lock.acquire(blocking=False):
            raise OperationalBusy("metrics snapshot already in progress")
        try:
            materialized = snapshot() if callable(snapshot) else snapshot
            filtered = filter_json_metrics(materialized)
            payload = json.dumps(filtered, separators=(",", ":")).encode("utf-8")
            self._enforce_size(payload)
            return payload
        finally:
            self._lock.release()

    def prometheus_bytes(self, snapshot: bytes | Callable[[], bytes]) -> bytes:
        if not self._lock.acquire(blocking=False):
            raise OperationalBusy("metrics snapshot already in progress")
        try:
            materialized = snapshot() if callable(snapshot) else snapshot
            kept: list[bytes] = []
            for raw_line in materialized.splitlines(keepends=True):
                line = raw_line.decode("utf-8", errors="replace")
                if _prometheus_line_has_sensitive_label(line):
                    continue
                kept.append(raw_line)
            payload = b"".join(kept)
            self._enforce_size(payload)
            return payload
        finally:
            self._lock.release()

    def _enforce_size(self, payload: bytes) -> None:
        if len(payload) > self._max_bytes:
            raise OperationalPayloadTooLarge("metrics snapshot exceeds response budget")


def _metric_key_has_sensitive_label(key: str) -> bool:
    bracket = key.find("[")
    if bracket < 0 or not key.endswith("]"):
        return False
    for item in key[bracket + 1 : -1].split(","):
        label, separator, _value = item.partition("=")
        if separator and label.strip().lower() in _SENSITIVE_LABELS:
            return True
    return False


def _mapping_has_sensitive_labels(value: Mapping[str, Any]) -> bool:
    labels = value.get("labels")
    return isinstance(labels, Mapping) and any(
        str(label).lower() in _SENSITIVE_LABELS for label in labels
    )


def filter_json_metrics(value: Any) -> Any:
    """Remove every metric entry carrying a tenant or organization label."""

    if isinstance(value, Mapping):
        if _mapping_has_sensitive_labels(value):
            return None
        filtered: dict[str, Any] = {}
        for key, child in value.items():
            key_text = str(key)
            if _metric_key_has_sensitive_label(key_text):
                continue
            clean_child = filter_json_metrics(child)
            if clean_child is not None:
                filtered[key_text] = clean_child
        return filtered
    if isinstance(value, list):
        return [clean for child in value if (clean := filter_json_metrics(child)) is not None]
    return value


def _prometheus_line_has_sensitive_label(line: str) -> bool:
    if line.lstrip().startswith("#"):
        return False
    start = line.find("{")
    end = line.rfind("}")
    if start < 0 or end <= start:
        return False
    return any(
        match.group(1).lower() in _SENSITIVE_LABELS
        for match in _PROM_LABEL_RE.finditer(line[start + 1 : end])
    )


def _check_budget(started_at: float) -> None:
    if time.monotonic() - started_at >= READINESS_TOTAL_BUDGET_SECONDS:
        raise TimeoutError("readiness total budget exceeded")


def _migration_checksums_match(applied: Mapping[str, str | None], started_at: float) -> bool:
    migrations_dir = Path(__file__).resolve().parents[2] / "db" / "migrations"
    if set(MIGRATIONS) - set(applied):
        return False
    for filename in MIGRATIONS:
        _check_budget(started_at)
        recorded = applied.get(filename)
        if not recorded:
            return False
        try:
            on_disk = hashlib.sha256((migrations_dir / filename).read_bytes()).hexdigest()
        except OSError:
            return False
        if not hmac.compare_digest(on_disk, recorded):
            return False
    return True


def bounded_readiness_check(
    candidate_repository: Any,
    *,
    unsafe_search_configuration: bool,
    jwt_enabled: bool,
    jwt_problems: list[str],
) -> ReadinessResult:
    """Run a fixed, read-only readiness census with at most eight SQL statements."""

    reasons: list[str] = []
    if unsafe_search_configuration:
        reasons.append("unsafe_search_configuration")
    if not jwt_enabled:
        reasons.append("jwt_disabled")
    if jwt_problems:
        reasons.append("jwt_verification_unavailable")
    if candidate_repository is None:
        reasons.append("candidate_repository_unavailable")
    if reasons:
        return ReadinessResult(False, tuple(sorted(set(reasons))))

    started_at = time.monotonic()
    allow_owner = os.getenv("ACTIVEKG_READYZ_ALLOW_OWNER", "false").lower() == "true"
    try:
        with candidate_repository.pool.connection(timeout=READINESS_POOL_TIMEOUT_SECONDS) as conn:
            with conn.cursor() as cur:
                _check_budget(started_at)
                # Statement 1: bound every following catalog/ledger query.
                cur.execute(f"SET LOCAL statement_timeout = {READINESS_STATEMENT_TIMEOUT_MS}")

                _check_budget(started_at)
                # Statement 2: basic database reachability.
                cur.execute("SELECT 1")
                if cur.fetchone() != (1,):
                    reasons.append("database_unavailable")

                _check_budget(started_at)
                # Statement 3: migration-ledger presence.
                cur.execute("SELECT to_regclass('public.schema_migrations')")
                ledger_exists = cur.fetchone()[0] is not None
                if not ledger_exists:
                    reasons.append("migration_ledger_missing")
                else:
                    _check_budget(started_at)
                    # Statement 4: bounded ledger metadata only; no application rows.
                    cur.execute("SELECT filename, checksum FROM schema_migrations")
                    applied = dict(cur.fetchall())
                    if not _migration_checksums_match(applied, started_at):
                        reasons.append("migration_ledger_incomplete")

                _check_budget(started_at)
                # Statement 5: table existence, RLS posture and ownership.
                cur.execute(
                    """
                    SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity,
                           pg_get_userbyid(c.relowner), current_user
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public' AND c.relkind = 'r'
                      AND c.relname = ANY(%s)
                    """,
                    (list(_CANDIDATE_TABLES + _SHARED_TABLES),),
                )
                relation_rows = cur.fetchall()
                relation_map = {row[0]: row for row in relation_rows}
                if set(_CANDIDATE_TABLES + _SHARED_TABLES) - set(relation_map):
                    reasons.append("required_schema_missing")
                for table in _CANDIDATE_TABLES:
                    row = relation_map.get(table)
                    if row is None or not bool(row[1]):
                        reasons.append("tenant_rls_incomplete")
                        break
                    if table == "candidate_contact_evidence" and not bool(row[2]):
                        reasons.append("tenant_force_rls_incomplete")
                        break
                    if not allow_owner and row[3] == row[4]:
                        reasons.append("runtime_role_owns_tenant_table")
                        break

                _check_budget(started_at)
                # Statement 6: tenant-policy installation, without executing policy code.
                cur.execute(
                    """
                    SELECT tablename, qual, with_check
                    FROM pg_policies
                    WHERE schemaname = 'public' AND tablename = ANY(%s)
                    """,
                    (list(_CANDIDATE_TABLES),),
                )
                policies: dict[str, list[tuple[str | None, str | None]]] = {}
                for table, qual, with_check in cur.fetchall():
                    policies.setdefault(table, []).append((qual, with_check))
                for table in _CANDIDATE_TABLES:
                    expressions = " ".join(
                        (qual or "") + " " + (with_check or "")
                        for qual, with_check in policies.get(table, [])
                    )
                    if "app.current_tenant_id" not in expressions:
                        reasons.append("tenant_policy_missing")
                        break

                _check_budget(started_at)
                # Statement 7: runtime-role escalation posture.
                cur.execute(
                    """
                    SELECT r.rolsuper, r.rolbypassrls,
                           pg_has_role(current_user, 'pg_write_all_data', 'MEMBER'),
                           pg_has_role(current_user, 'pg_read_all_data', 'MEMBER'),
                           (EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'admin_role')
                            AND pg_has_role(current_user, 'admin_role', 'MEMBER'))
                    FROM pg_roles r WHERE r.rolname = current_user
                    """
                )
                role = cur.fetchone()
                if role is None or any(bool(value) for value in role):
                    reasons.append("runtime_role_overprivileged")

                _check_budget(started_at)
                # Statement 8: required catalog objects, presence only.
                cur.execute(
                    """
                    SELECT 'index', indexname FROM pg_indexes
                    WHERE schemaname = 'public' AND indexname = ANY(%s)
                    UNION ALL
                    SELECT 'function', p.proname
                    FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = 'public' AND p.proname = ANY(%s)
                    UNION ALL
                    SELECT 'constraint', c.conname
                    FROM pg_constraint c JOIN pg_namespace n ON n.oid = c.connamespace
                    WHERE n.nspname = 'public' AND c.conname = ANY(%s)
                    """,
                    (
                        list(_REQUIRED_INDEXES),
                        list(_REQUIRED_FUNCTIONS),
                        list(_REQUIRED_CONSTRAINTS),
                    ),
                )
                objects: dict[str, set[str]] = {
                    "index": set(),
                    "function": set(),
                    "constraint": set(),
                }
                for object_type, name in cur.fetchall():
                    objects[object_type].add(name)
                if _REQUIRED_INDEXES - objects["index"]:
                    reasons.append("required_index_missing")
                if _REQUIRED_FUNCTIONS - objects["function"]:
                    reasons.append("required_function_missing")
                if _REQUIRED_CONSTRAINTS - objects["constraint"]:
                    reasons.append("required_constraint_missing")
        _check_budget(started_at)
    except TimeoutError:
        reasons.append("readiness_timeout")
    except Exception as exc:
        logger.warning(
            "Bounded readiness database check failed",
            extra={"error_type": type(exc).__name__},
        )
        reasons.append("readiness_database_error")

    return ReadinessResult(not reasons, tuple(sorted(set(reasons))))
