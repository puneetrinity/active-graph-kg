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

from activekg.common.migration_manifest import CHECKSUM_TRANSITIONS, MIGRATIONS
from activekg.common.schema_control import CONTROL_SCHEMA, validate_environment, validate_target_id

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
_PRIVACY_TABLES = (
    "candidate_privacy_directive_events",
    "candidate_privacy_directives",
    "candidate_privacy_identity_tokens",
)
_PRIVACY_INDEXES = {
    "candidate_privacy_events_cursor_idx",
    "candidate_privacy_events_directive_idx",
    "candidate_privacy_directives_global_idx",
    "candidate_privacy_directives_candidate_idx",
    "candidate_privacy_identity_tokens_lookup_idx",
}
_PRIVACY_FUNCTIONS = {
    "candidate_privacy_append_only",
    "candidate_privacy_decision_for",
    "candidate_privacy_global_decision",
    "candidate_privacy_candidate_decision",
    "candidate_privacy_node_decision",
    "candidate_privacy_resolve_subject",
    "candidate_privacy_resolve_canonical",
    "candidate_privacy_match",
    "candidate_privacy_token_key_versions",
    "candidate_privacy_create_directive",
    "candidate_privacy_transition_directive",
}
_PRIVACY_FUNCTION_ARGUMENTS = {
    "candidate_privacy_decision_for": 3,
    "candidate_privacy_global_decision": 1,
    "candidate_privacy_candidate_decision": 2,
    "candidate_privacy_node_decision": 1,
    "candidate_privacy_resolve_subject": 2,
    "candidate_privacy_resolve_canonical": 3,
    "candidate_privacy_match": 4,
    "candidate_privacy_token_key_versions": 0,
    "candidate_privacy_create_directive": 16,
    "candidate_privacy_transition_directive": 9,
}
_PRIVACY_APPEND_ONLY_FUNCTION = "candidate_privacy_append_only"
_PRIVACY_APPEND_ONLY_FUNCTION_BODY = (
    "beginraiseexception'candidateprivacyauthorityisappend-only(attempted%)',tg_op;end;"
)
_PRIVACY_TRIGGERS = (
    (
        "candidate_privacy_directive_events",
        "candidate_privacy_events_no_mutation",
        _PRIVACY_APPEND_ONLY_FUNCTION,
        27,
    ),
    (
        "candidate_privacy_directive_events",
        "candidate_privacy_events_no_truncate",
        _PRIVACY_APPEND_ONLY_FUNCTION,
        34,
    ),
    (
        "candidate_privacy_identity_tokens",
        "candidate_privacy_tokens_no_mutation",
        _PRIVACY_APPEND_ONLY_FUNCTION,
        27,
    ),
    (
        "candidate_privacy_identity_tokens",
        "candidate_privacy_tokens_no_truncate",
        _PRIVACY_APPEND_ONLY_FUNCTION,
        34,
    ),
)
_PRIVACY_SEQUENCE = "candidate_privacy_directive_events_cursor_seq"
_PUBLIC_COLUMNS = {
    "public_profile",
    "public_profile_observed_at",
    "public_crustdata_person_id",
    "public_headline",
    "public_location_city",
    "public_location_country_code",
    "public_role_family",
    "public_seniority_band",
    "public_skills_normalized",
    "public_embedding",
    "public_embedding_status",
    "public_embed_version",
}
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
_REQUIRED_CONSTRAINTS_BY_TABLE = {
    "global_candidates": {
        "global_candidates_public_embedding_status_check",
        "global_candidates_public_headline_from_profile",
    },
    "candidate_contact_evidence": {
        "candidate_contact_evidence_unique",
        "candidate_contact_evidence_primary_usable",
    },
    "contact_suppression_tombstones": {
        "contact_suppression_reason_check",
        "contact_suppression_provider_event_hash",
    },
    "contact_person_suppressions": {
        "contact_person_suppressions_pkey",
        "contact_person_suppressions_global_candidate_fkey",
        "contact_person_suppression_reason_check",
        "contact_person_suppression_provider_event_hash",
    },
    "contact_suppression_receipts": {
        "contact_suppression_receipts_pkey",
        "contact_suppression_receipt_email_hash_check",
        "contact_suppression_receipt_signal_candidate_nonblank",
        "contact_suppression_receipt_tenant_nonblank",
        "contact_suppression_receipt_provider_event_hash",
        "contact_suppression_receipt_authority_check",
        "contact_suppression_receipt_scope_reason_check",
        "contact_suppression_receipts_provider_event_unique",
    },
    "public_candidate_market_memberships": {
        "public_candidate_market_country_code_check",
        "public_candidate_market_memberships_pkey",
    },
    "candidate_privacy_directive_events": {
        "candidate_privacy_directive_events_pkey",
        "candidate_privacy_directive_events_event_id_key",
        "candidate_privacy_directive_events_directive_version_check",
        "candidate_privacy_directive_events_event_type_check",
        "candidate_privacy_directive_events_action_check",
        "candidate_privacy_directive_events_scope_check",
        "candidate_privacy_directive_events_resulting_state_check",
        "candidate_privacy_directive_events_authority_type_check",
        "candidate_privacy_directive_events_reason_code_check",
        "candidate_privacy_directive_events_issuer_check",
        "candidate_privacy_directive_events_actor_id_check",
        "candidate_privacy_directive_events_actor_type_check",
        "candidate_privacy_directive_events_global_candidate_id_fkey",
        "candidate_privacy_directive_events_key_version_check",
        "candidate_privacy_directive_events_schema_version_check",
        "candidate_privacy_event_action_scope_check",
        "candidate_privacy_event_candidate_pair_check",
        "candidate_privacy_event_candidate_fkey",
        "candidate_privacy_event_directive_version_unique",
        "candidate_privacy_event_request_type_unique",
    },
    "candidate_privacy_directives": {
        "candidate_privacy_directives_pkey",
        "candidate_privacy_directives_action_check",
        "candidate_privacy_directives_scope_check",
        "candidate_privacy_directives_state_check",
        "candidate_privacy_directives_version_check",
        "candidate_privacy_directives_authority_type_check",
        "candidate_privacy_directives_reason_code_check",
        "candidate_privacy_directives_global_candidate_id_fkey",
        "candidate_privacy_directives_last_event_cursor_key",
        "candidate_privacy_directives_last_event_cursor_fkey",
        "candidate_privacy_directive_action_scope_check",
        "candidate_privacy_directive_candidate_pair_check",
        "candidate_privacy_directive_candidate_fkey",
    },
    "candidate_privacy_identity_tokens": {
        "candidate_privacy_identity_tokens_pkey",
        "candidate_privacy_identity_tokens_directive_id_fkey",
        "candidate_privacy_identity_tokens_identifier_type_check",
        "candidate_privacy_identity_tokens_key_version_check",
        "candidate_privacy_identity_tokens_token_check",
    },
}
_REQUIRED_CONSTRAINTS = {
    constraint
    for constraints in _REQUIRED_CONSTRAINTS_BY_TABLE.values()
    for constraint in constraints
}
_EXPECTED_CHECK_DEFINITIONS = {
    (
        "contact_suppression_tombstones",
        "contact_suppression_reason_check",
    ): "check((reason=any(array['hard_bounce','complaint'])))",
    (
        "contact_suppression_tombstones",
        "contact_suppression_provider_event_hash",
    ): "check(((provider_event_idisnull)or(provider_event_id~'^[0-9a-f]{64}$')))",
    (
        "contact_person_suppressions",
        "contact_person_suppression_reason_check",
    ): "check((reason='complaint'))",
    (
        "contact_person_suppressions",
        "contact_person_suppression_provider_event_hash",
    ): "check((provider_event_id~'^[0-9a-f]{64}$'))",
    (
        "contact_suppression_receipts",
        "contact_suppression_receipt_email_hash_check",
    ): "check((email_hash~'^[0-9a-f]{64}$'))",
    (
        "contact_suppression_receipts",
        "contact_suppression_receipt_signal_candidate_nonblank",
    ): "check(((signal_candidate_idisnull)or(btrim(signal_candidate_id)<>'')))",
    (
        "contact_suppression_receipts",
        "contact_suppression_receipt_tenant_nonblank",
    ): "check(((btrim(tenant_id)<>'')and(tenant_id<>'__quarantine__')))",
    (
        "contact_suppression_receipts",
        "contact_suppression_receipt_provider_event_hash",
    ): "check((provider_event_id~'^[0-9a-f]{64}$'))",
    (
        "contact_suppression_receipts",
        "contact_suppression_receipt_authority_check",
    ): "check(((btrim(issuer)<>'')and(btrim(actor_id)<>'')and(actor_type='service')))",
    (
        "contact_suppression_receipts",
        "contact_suppression_receipt_scope_reason_check",
    ): (
        "check((((reason='hard_bounce')and(scope='address'))or"
        "((reason='complaint')and(scope='person')and(global_candidate_idisnotnull)"
        "and(signal_candidate_idisnotnull))))"
    ),
}
_EXPECTED_STRUCTURAL_DEFINITIONS = {
    (
        "contact_person_suppressions",
        "contact_person_suppressions_pkey",
    ): ("p", "primarykey(global_candidate_id)"),
    (
        "contact_person_suppressions",
        "contact_person_suppressions_global_candidate_fkey",
    ): (
        "f",
        "foreignkey(global_candidate_id)referencesglobal_candidates(id)ondeleterestrict",
    ),
    (
        "contact_suppression_receipts",
        "contact_suppression_receipts_pkey",
    ): ("p", "primarykey(id)"),
    (
        "contact_suppression_receipts",
        "contact_suppression_receipts_provider_event_unique",
    ): ("u", "unique(issuer,provider_event_id)"),
}
_APPEND_ONLY_FUNCTION = "contact_suppression_receipts_append_only"
_APPEND_ONLY_FUNCTION_BODY = (
    "beginraiseexception'contact_suppression_receiptsisappend-only(attempted%)',tg_op;end;"
)
_SUPPRESSION_TRIGGERS = (
    (
        "contact_suppression_receipts",
        "contact_suppression_receipts_no_mutation",
        _APPEND_ONLY_FUNCTION,
        27,
    ),
    (
        "contact_suppression_receipts",
        "contact_suppression_receipts_no_truncate",
        _APPEND_ONLY_FUNCTION,
        34,
    ),
)
_SUPPRESSION_SEQUENCE = "contact_suppression_receipts_id_seq"
_SUPPRESSION_TABLES = {
    "contact_suppression_tombstones",
    "contact_person_suppressions",
    "contact_suppression_receipts",
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

    def run(
        self,
        check: Callable[[], ReadinessResult],
        *,
        force_refresh: bool = False,
    ) -> ReadinessResult:
        now = time.monotonic()
        if not force_refresh:
            with self._cache_lock:
                cached = self._cached
                cached_at = self._cached_at
            if cached is not None:
                ttl = (
                    READINESS_SUCCESS_TTL_SECONDS if cached.ready else READINESS_FAILURE_TTL_SECONDS
                )
                if now - cached_at < ttl:
                    return cached

        if not self._lock.acquire(blocking=False):
            raise OperationalBusy("readiness check already in progress")
        try:
            # Recheck after winning the single-flight lock.
            if not force_refresh:
                now = time.monotonic()
                with self._cache_lock:
                    cached = self._cached
                    cached_at = self._cached_at
                if cached is not None:
                    ttl = (
                        READINESS_SUCCESS_TTL_SECONDS
                        if cached.ready
                        else READINESS_FAILURE_TTL_SECONDS
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


def _normalize_sql_definition(value: str) -> str:
    return re.sub(r"\s+", "", value.lower()).replace("::text", "")


def _tenant_policy_expression_ok(expression: str) -> bool:
    normalized = _normalize_sql_definition(expression)
    tenant_clause = "(tenant_id=current_setting('app.current_tenant_id',true))"
    quarantine_clause = "(tenant_id<>'__quarantine__')"
    return normalized in {
        f"({tenant_clause}and{quarantine_clause})",
        f"({quarantine_clause}and{tenant_clause})",
    }


def _migration_checksums_match(applied: Mapping[str, str | None], started_at: float) -> bool:
    migrations_dir = Path(__file__).resolve().parents[2] / "db" / "migrations"
    if set(MIGRATIONS) != set(applied):
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
            transition = CHECKSUM_TRANSITIONS.get(filename, {}).get(recorded)
            if transition is None or not hmac.compare_digest(on_disk, transition):
                return False
    return True


def bounded_readiness_check(
    candidate_repository: Any,
    *,
    unsafe_search_configuration: bool,
    jwt_enabled: bool,
    jwt_problems: list[str],
    privacy_problems: list[str] | None = None,
    privacy_key_versions: set[int] | None = None,
) -> ReadinessResult:
    """Run a fixed, read-only readiness census with at most eight SQL statements."""

    reasons: list[str] = []
    if unsafe_search_configuration:
        reasons.append("unsafe_search_configuration")
    if not jwt_enabled:
        reasons.append("jwt_disabled")
    if jwt_problems:
        reasons.append("jwt_verification_unavailable")
    if privacy_problems:
        reasons.extend(privacy_problems)
    if candidate_repository is None:
        reasons.append("candidate_repository_unavailable")
    try:
        expected_target_id = validate_target_id(os.getenv("ACTIVEKG_SCHEMA_TARGET_ID"))
        expected_environment = validate_environment(os.getenv("ACTIVEKG_SCHEMA_ENVIRONMENT"))
    except Exception:
        expected_target_id = ""
        expected_environment = ""
        reasons.append("schema_control_configuration_invalid")
    if reasons:
        return ReadinessResult(False, tuple(sorted(set(reasons))))

    started_at = time.monotonic()
    allow_owner = (
        expected_environment != "production"
        and os.getenv("ACTIVEKG_READYZ_ALLOW_OWNER", "false").lower() == "true"
    )
    try:
        with candidate_repository.pool.connection(timeout=READINESS_POOL_TIMEOUT_SECONDS) as conn:
            with conn.cursor() as cur:
                _check_budget(started_at)
                # Statement 1: force the transaction read-only and bound every
                # following catalog/ledger query without adding an unbounded
                # preflight statement.
                cur.execute(
                    "SET TRANSACTION READ ONLY; "
                    f"SET LOCAL statement_timeout = {READINESS_STATEMENT_TIMEOUT_MS}; "
                    f"SET LOCAL lock_timeout = {READINESS_STATEMENT_TIMEOUT_MS}"
                )

                _check_budget(started_at)
                # Statement 2: reachability and both readiness authorities.
                cur.execute(
                    "SELECT to_regclass('public.schema_migrations'), "
                    "to_regclass(%s), to_regclass(%s), "
                    "to_regclass('public.candidate_privacy_directive_events'), "
                    "to_regclass('public.candidate_privacy_directives'), "
                    "to_regclass('public.candidate_privacy_identity_tokens')",
                    (
                        f"{CONTROL_SCHEMA}.target_identity",
                        f"{CONTROL_SCHEMA}.release_attempts",
                    ),
                )
                (
                    ledger_relation,
                    identity_relation,
                    attempts_relation,
                    privacy_events_relation,
                    privacy_directives_relation,
                    privacy_tokens_relation,
                ) = cur.fetchone()
                if ledger_relation is None:
                    reasons.append("migration_ledger_missing")
                if identity_relation is None or attempts_relation is None:
                    reasons.append("schema_control_missing")
                if None in (
                    privacy_events_relation,
                    privacy_directives_relation,
                    privacy_tokens_relation,
                ):
                    reasons.append("candidate_privacy_schema_missing")

                if identity_relation is not None and attempts_relation is not None:
                    _check_budget(started_at)
                    # Statement 3: exact target identity, append-only guards
                    # and release health.
                    cur.execute(
                        f"""
                        SELECT
                          (SELECT count(*) FROM {CONTROL_SCHEMA}.target_identity),
                          (SELECT min(product) FROM {CONTROL_SCHEMA}.target_identity),
                          (SELECT min(environment) FROM {CONTROL_SCHEMA}.target_identity),
                          (SELECT min(target_id::text) FROM {CONTROL_SCHEMA}.target_identity),
                          (SELECT count(*) FROM pg_trigger t
                           JOIN pg_class c ON c.oid = t.tgrelid
                           JOIN pg_namespace n ON n.oid = c.relnamespace
                           WHERE n.nspname = '{CONTROL_SCHEMA}' AND NOT t.tgisinternal
                             AND t.tgenabled IN ('O','A')
                             AND (c.relname, t.tgname) IN (
                               ('target_identity', 'target_identity_no_mutation'),
                               ('target_identity', 'target_identity_no_truncate'),
                               ('release_attempts', 'release_attempts_finish_only'),
                               ('release_attempts', 'release_attempts_no_truncate')
                             )),
                          (SELECT count(*) FROM {CONTROL_SCHEMA}.release_attempts
                           WHERE outcome = 'running' OR finished_at IS NULL),
                          (SELECT outcome FROM {CONTROL_SCHEMA}.release_attempts
                           ORDER BY id DESC LIMIT 1)
                        """
                    )
                    (
                        identity_count,
                        product,
                        environment,
                        target_id,
                        control_guard_count,
                        unfinished,
                        latest,
                    ) = cur.fetchone()
                    if (
                        identity_count != 1
                        or product != "memory"
                        or environment != expected_environment
                        or target_id != expected_target_id
                    ):
                        reasons.append("schema_control_identity_mismatch")
                    if control_guard_count != 4:
                        reasons.append("schema_control_guards_missing")
                    if unfinished != 0:
                        reasons.append("schema_release_unfinished")
                    if latest != "success":
                        reasons.append("schema_release_latest_failed")

                if ledger_relation is not None:
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
                    (list(_CANDIDATE_TABLES + _SHARED_TABLES + _PRIVACY_TABLES),),
                )
                relation_rows = cur.fetchall()
                relation_map = {row[0]: row for row in relation_rows}
                if set(_CANDIDATE_TABLES + _SHARED_TABLES) - set(relation_map):
                    reasons.append("required_schema_missing")
                if set(_PRIVACY_TABLES) - set(relation_map):
                    reasons.append("candidate_privacy_schema_missing")
                elif any(not bool(relation_map[table][1]) for table in _PRIVACY_TABLES):
                    reasons.append("candidate_privacy_rls_incomplete")
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
                if not allow_owner and any(
                    (row := relation_map.get(table)) is not None and row[3] == row[4]
                    for table in _SUPPRESSION_TABLES
                ):
                    reasons.append("runtime_role_owns_suppression_table")

                _check_budget(started_at)
                # Statement 6: exact tenant/admin policy definitions.
                cur.execute(
                    """
                    SELECT tablename, policyname, permissive, roles::text, cmd,
                           COALESCE(qual::text, ''), COALESCE(with_check::text, '')
                    FROM pg_policies
                    WHERE schemaname = 'public' AND tablename = ANY(%s)
                    """,
                    (list(_CANDIDATE_TABLES + _PRIVACY_TABLES),),
                )
                policies = {(row[0], row[1]): row for row in cur.fetchall()}
                for table in _CANDIDATE_TABLES:
                    tenant_policy = policies.get((table, f"tenant_isolation_{table}"))
                    if tenant_policy is None:
                        reasons.append("tenant_policy_missing")
                        break
                    (
                        _table,
                        _name,
                        permissive,
                        roles,
                        command,
                        using_expression,
                        check_expression,
                    ) = tenant_policy
                    if (
                        permissive != "PERMISSIVE"
                        or command != "ALL"
                        or "public" not in roles.lower()
                        or not _tenant_policy_expression_ok(using_expression)
                        or not _tenant_policy_expression_ok(check_expression)
                    ):
                        reasons.append("tenant_policy_definition_unexpected")
                        break
                    admin_policy = policies.get((table, f"admin_all_{table}"))
                    if admin_policy is None:
                        reasons.append("admin_policy_missing")
                        break
                    admin_roles = admin_policy[3]
                    admin_using = _normalize_sql_definition(admin_policy[5])
                    admin_check = _normalize_sql_definition(admin_policy[6])
                    if (
                        "admin_role" not in admin_roles.lower()
                        or admin_policy[2] != "PERMISSIVE"
                        or admin_policy[4] != "ALL"
                        or admin_using != "true"
                        or admin_check != "true"
                    ):
                        reasons.append("admin_policy_definition_unexpected")
                        break
                for table, policy_name in (
                    (
                        "candidate_privacy_directive_events",
                        "candidate_privacy_events_runtime_read",
                    ),
                    (
                        "candidate_privacy_directives",
                        "candidate_privacy_directives_runtime_read",
                    ),
                ):
                    policy = policies.get((table, policy_name))
                    if (
                        policy is None
                        or policy[2] != "PERMISSIVE"
                        or policy[3].lower() != "{public}"
                        or policy[4] != "SELECT"
                        or _normalize_sql_definition(policy[5]) != "true"
                    ):
                        reasons.append("candidate_privacy_policy_unexpected")
                        break
                if any(
                    table == "candidate_privacy_identity_tokens" for table, _policy_name in policies
                ):
                    reasons.append("candidate_privacy_token_policy_unsafe")

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
                # Statement 8: required catalog invariants. Multiple bounded
                # catalog branches remain one application SQL statement.
                cur.execute(
                    """
                    SELECT 'index'::text, i.relname::text,
                           jsonb_build_object('valid', ix.indisvalid AND ix.indisready)
                    FROM pg_class i
                    JOIN pg_namespace n ON n.oid = i.relnamespace
                    JOIN pg_index ix ON ix.indexrelid = i.oid
                    WHERE n.nspname = 'public' AND i.relname = ANY(%s)
                    UNION ALL
                    SELECT 'function'::text, p.proname::text,
                           jsonb_build_object(
                               'arguments', p.pronargs,
                               'returns_trigger', p.prorettype = 'trigger'::regtype,
                               'language', l.lanname,
                               'security_definer', p.prosecdef,
                               'config', COALESCE(to_jsonb(p.proconfig), '[]'::jsonb),
                               'source', p.prosrc,
                               'owned_by_runtime', pg_get_userbyid(p.proowner) = current_user
                           )
                    FROM pg_proc p
                    JOIN pg_namespace n ON n.oid = p.pronamespace
                    JOIN pg_language l ON l.oid = p.prolang
                    WHERE n.nspname = 'public' AND p.proname = ANY(%s)
                    UNION ALL
                    SELECT 'constraint'::text,
                           (c.relname || '.' || con.conname)::text,
                           jsonb_build_object(
                               'type', con.contype,
                               'delete_action', con.confdeltype,
                               'validated', con.convalidated,
                               'definition', pg_get_constraintdef(con.oid)
                           )
                    FROM pg_constraint con
                    JOIN pg_class c ON c.oid = con.conrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public' AND con.conname = ANY(%s)
                    UNION ALL
                    SELECT 'trigger'::text, (c.relname || '.' || t.tgname)::text,
                           jsonb_build_object(
                               'function', p.proname,
                               'type', t.tgtype,
                               'enabled', t.tgenabled
                           )
                    FROM pg_trigger t
                    JOIN pg_class c ON c.oid = t.tgrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    JOIN pg_proc p ON p.oid = t.tgfoid
                    WHERE n.nspname = 'public' AND NOT t.tgisinternal
                      AND t.tgname = ANY(%s)
                    UNION ALL
                    SELECT 'sequence'::text, c.relname::text,
                           jsonb_build_object(
                               'kind', c.relkind,
                               'usage', has_sequence_privilege(
                                   current_user,
                                   quote_ident(n.nspname) || '.' || quote_ident(c.relname),
                                   'USAGE'
                               ),
                               'owned_by_runtime', pg_get_userbyid(c.relowner) = current_user
                           )
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = 'public' AND c.relname = ANY(%s)
                    UNION ALL
                    SELECT 'public_column'::text, column_name::text, '{}'::jsonb
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'global_candidates'
                      AND column_name = ANY(%s)
                    UNION ALL
                    SELECT 'privilege'::text, 'suppression_tables'::text,
                           jsonb_build_object(
                               'receipt_select', has_table_privilege(
                                   current_user, 'public.contact_suppression_receipts', 'SELECT'
                               ),
                               'receipt_insert', has_table_privilege(
                                   current_user, 'public.contact_suppression_receipts', 'INSERT'
                               ),
                               'receipt_update', has_table_privilege(
                                   current_user, 'public.contact_suppression_receipts', 'UPDATE'
                               ),
                               'receipt_delete', has_table_privilege(
                                   current_user, 'public.contact_suppression_receipts', 'DELETE'
                               ),
                               'receipt_truncate', has_table_privilege(
                                   current_user, 'public.contact_suppression_receipts', 'TRUNCATE'
                               ),
                               'tombstone_delete', has_table_privilege(
                                   current_user, 'public.contact_suppression_tombstones', 'DELETE'
                               ),
                               'tombstone_truncate', has_table_privilege(
                                   current_user, 'public.contact_suppression_tombstones', 'TRUNCATE'
                               ),
                               'person_delete', has_table_privilege(
                                   current_user, 'public.contact_person_suppressions', 'DELETE'
                               ),
                               'person_truncate', has_table_privilege(
                                   current_user, 'public.contact_person_suppressions', 'TRUNCATE'
                               )
                           )
                    UNION ALL
                    SELECT 'privacy_privilege'::text, 'runtime'::text,
                           jsonb_build_object(
                               'events_select', has_table_privilege(
                                   current_user,
                                   'public.candidate_privacy_directive_events', 'SELECT'
                               ),
                               'events_write', (
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_directive_events', 'INSERT') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_directive_events', 'UPDATE') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_directive_events', 'DELETE') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_directive_events', 'TRUNCATE')
                               ),
                               'directives_select', has_table_privilege(
                                   current_user,
                                   'public.candidate_privacy_directives', 'SELECT'
                               ),
                               'directives_write', (
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_directives', 'INSERT') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_directives', 'UPDATE') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_directives', 'DELETE') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_directives', 'TRUNCATE')
                               ),
                               'tokens_select', has_table_privilege(
                                   current_user,
                                   'public.candidate_privacy_identity_tokens', 'SELECT'
                               ),
                               'tokens_write', (
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_identity_tokens', 'INSERT') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_identity_tokens', 'UPDATE') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_identity_tokens', 'DELETE') OR
                                   has_table_privilege(current_user,
                                       'public.candidate_privacy_identity_tokens', 'TRUNCATE')
                               )
                           )
                    UNION ALL
                    SELECT 'privacy_key_version'::text, key_version::text, '{}'::jsonb
                    FROM candidate_privacy_token_key_versions()
                    UNION ALL
                    SELECT 'privacy_function_privilege'::text, p.proname::text,
                           jsonb_build_object(
                               'execute', has_function_privilege(current_user, p.oid, 'EXECUTE')
                           )
                    FROM pg_proc p
                    JOIN pg_namespace n ON n.oid = p.pronamespace
                    WHERE n.nspname = 'public'
                      AND p.proname = ANY(%s)
                    """,
                    (
                        list(_REQUIRED_INDEXES | _PRIVACY_INDEXES),
                        list(_REQUIRED_FUNCTIONS | _PRIVACY_FUNCTIONS),
                        list(_REQUIRED_CONSTRAINTS),
                        [trigger[1] for trigger in _SUPPRESSION_TRIGGERS]
                        + [trigger[1] for trigger in _PRIVACY_TRIGGERS],
                        [_SUPPRESSION_SEQUENCE, _PRIVACY_SEQUENCE],
                        list(_PUBLIC_COLUMNS),
                        list(_PRIVACY_FUNCTION_ARGUMENTS),
                    ),
                )
                objects: dict[str, dict[str, Mapping[str, Any]]] = {}
                for object_type, name, details in cur.fetchall():
                    objects.setdefault(object_type, {})[name] = details

                indexes = objects.get("index", {})
                if _REQUIRED_INDEXES - set(indexes) or any(
                    not bool(details.get("valid")) for details in indexes.values()
                ):
                    reasons.append("required_index_missing")
                if _PRIVACY_INDEXES - set(indexes):
                    reasons.append("candidate_privacy_index_missing")
                functions = objects.get("function", {})
                if _REQUIRED_FUNCTIONS - set(functions):
                    reasons.append("required_function_missing")
                if _PRIVACY_FUNCTIONS - set(functions):
                    reasons.append("candidate_privacy_function_missing")
                for name in _PRIVACY_FUNCTIONS - {"candidate_privacy_append_only"}:
                    function = functions.get(name)
                    if function is None:
                        continue
                    configs = {str(value) for value in function.get("config", [])}
                    if (
                        not bool(function.get("security_definer"))
                        or bool(function.get("owned_by_runtime"))
                        or "search_path=pg_catalog, public" not in configs
                    ):
                        reasons.append("candidate_privacy_function_posture_unsafe")
                        break
                    if int(function.get("arguments", -1)) != _PRIVACY_FUNCTION_ARGUMENTS[name]:
                        reasons.append("candidate_privacy_function_signature_unexpected")
                        break
                append_only = functions.get(_APPEND_ONLY_FUNCTION)
                if append_only is not None and (
                    int(append_only.get("arguments", -1)) != 0
                    or not bool(append_only.get("returns_trigger"))
                    or append_only.get("language") != "plpgsql"
                    or bool(append_only.get("security_definer"))
                    or _normalize_sql_definition(str(append_only.get("source", "")))
                    != _APPEND_ONLY_FUNCTION_BODY
                    or (not allow_owner and bool(append_only.get("owned_by_runtime")))
                ):
                    reasons.append("append_only_function_definition_unexpected")
                privacy_append_only = functions.get(_PRIVACY_APPEND_ONLY_FUNCTION)
                if privacy_append_only is not None and (
                    int(privacy_append_only.get("arguments", -1)) != 0
                    or not bool(privacy_append_only.get("returns_trigger"))
                    or privacy_append_only.get("language") != "plpgsql"
                    or bool(privacy_append_only.get("security_definer"))
                    or _normalize_sql_definition(str(privacy_append_only.get("source", "")))
                    != _PRIVACY_APPEND_ONLY_FUNCTION_BODY
                    or (not allow_owner and bool(privacy_append_only.get("owned_by_runtime")))
                ):
                    reasons.append("candidate_privacy_append_only_function_unexpected")

                constraints = objects.get("constraint", {})
                expected_constraint_keys = {
                    f"{table}.{constraint}"
                    for table, names in _REQUIRED_CONSTRAINTS_BY_TABLE.items()
                    for constraint in names
                }
                if expected_constraint_keys - set(constraints) or any(
                    not bool(details.get("validated")) for details in constraints.values()
                ):
                    reasons.append("required_constraint_missing")
                for (table, name), expected_definition in _EXPECTED_CHECK_DEFINITIONS.items():
                    constraint = constraints.get(f"{table}.{name}")
                    if constraint is not None and (
                        constraint.get("type") != "c"
                        or _normalize_sql_definition(str(constraint.get("definition", "")))
                        != expected_definition
                    ):
                        reasons.append("constraint_definition_unexpected")
                        break
                for (table, name), expected in _EXPECTED_STRUCTURAL_DEFINITIONS.items():
                    constraint = constraints.get(f"{table}.{name}")
                    if constraint is not None and (
                        constraint.get("type") != expected[0]
                        or _normalize_sql_definition(str(constraint.get("definition", "")))
                        != expected[1]
                    ):
                        reasons.append("constraint_definition_unexpected")
                        break

                triggers = objects.get("trigger", {})
                for (
                    table,
                    trigger_name,
                    trigger_function_name,
                    trigger_type,
                ) in _SUPPRESSION_TRIGGERS:
                    trigger = triggers.get(f"{table}.{trigger_name}")
                    if trigger is None:
                        reasons.append("required_trigger_missing")
                        break
                    if (
                        trigger.get("function") != trigger_function_name
                        or int(trigger.get("type", -1)) != trigger_type
                        or trigger.get("enabled") not in {"O", "A"}
                    ):
                        reasons.append("trigger_definition_unexpected")
                        break
                for table, trigger_name, trigger_function_name, trigger_type in _PRIVACY_TRIGGERS:
                    trigger = triggers.get(f"{table}.{trigger_name}")
                    if trigger is None:
                        reasons.append("candidate_privacy_trigger_missing")
                        break
                    if (
                        trigger.get("function") != trigger_function_name
                        or int(trigger.get("type", -1)) != trigger_type
                        or trigger.get("enabled") not in {"O", "A"}
                    ):
                        reasons.append("candidate_privacy_trigger_unexpected")
                        break

                sequence = objects.get("sequence", {}).get(_SUPPRESSION_SEQUENCE)
                if sequence is None:
                    reasons.append("required_sequence_missing")
                elif (
                    sequence.get("kind") != "S"
                    or not bool(sequence.get("usage"))
                    or (not allow_owner and bool(sequence.get("owned_by_runtime")))
                ):
                    reasons.append("sequence_posture_unexpected")
                privacy_sequence = objects.get("sequence", {}).get(_PRIVACY_SEQUENCE)
                if privacy_sequence is None:
                    reasons.append("candidate_privacy_sequence_missing")
                elif bool(privacy_sequence.get("usage")) or (
                    not allow_owner and bool(privacy_sequence.get("owned_by_runtime"))
                ):
                    reasons.append("candidate_privacy_sequence_posture_unsafe")

                public_columns = set(objects.get("public_column", {}))
                if _PUBLIC_COLUMNS - public_columns:
                    reasons.append("required_public_column_missing")

                privileges = objects.get("privilege", {}).get("suppression_tables")
                if privileges is None or (
                    not bool(privileges.get("receipt_select"))
                    or not bool(privileges.get("receipt_insert"))
                    or any(
                        bool(privileges.get(name))
                        for name in (
                            "receipt_update",
                            "receipt_delete",
                            "receipt_truncate",
                            "tombstone_delete",
                            "tombstone_truncate",
                            "person_delete",
                            "person_truncate",
                        )
                    )
                ):
                    reasons.append("suppression_table_privileges_unsafe")
                privacy_privileges = objects.get("privacy_privilege", {}).get("runtime")
                if privacy_privileges is None or (
                    not bool(privacy_privileges.get("events_select"))
                    or bool(privacy_privileges.get("events_write"))
                    or not bool(privacy_privileges.get("directives_select"))
                    or bool(privacy_privileges.get("directives_write"))
                    or bool(privacy_privileges.get("tokens_select"))
                    or bool(privacy_privileges.get("tokens_write"))
                ):
                    reasons.append("candidate_privacy_privileges_unsafe")
                stored_privacy_versions = {
                    int(version) for version in objects.get("privacy_key_version", {})
                }
                if privacy_key_versions is not None and not stored_privacy_versions.issubset(
                    privacy_key_versions
                ):
                    reasons.append("candidate_privacy_hmac_version_missing")
                privacy_function_privileges = objects.get("privacy_function_privilege", {})
                if set(_PRIVACY_FUNCTION_ARGUMENTS) - set(privacy_function_privileges) or any(
                    not bool(details.get("execute"))
                    for details in privacy_function_privileges.values()
                ):
                    reasons.append("candidate_privacy_function_privilege_missing")
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
