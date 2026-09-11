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
_DECISION_INBOX_TABLES = (
    "organization_decision_event_inbox",
    "organization_decision_stream_state",
)
_SOURCED_CANDIDATE_TABLES = (
    "global_candidate_source_identities",
    "global_candidate_source_observations",
    "global_candidate_ingest_receipts",
)
_ORGANIZATION_CANDIDATE_TABLES = (
    "organization_candidate_references",
    "organization_candidate_resume_evidence",
    "organization_candidate_ingest_receipts",
)
_ORGANIZATION_CANDIDATE_INDEXES = {
    "organization_candidate_references_candidate_idx",
    "organization_candidate_references_job_idx",
    "organization_candidate_resume_evidence_candidate_idx",
    "organization_candidate_ingest_receipts_candidate_idx",
}
_ORGANIZATION_CANDIDATE_CONSTRAINTS_BY_TABLE = {
    "organization_candidate_references": {
        "organization_candidate_references_pkey",
        "organization_candidate_references_tenant_nonblank",
        "organization_candidate_references_application_positive",
        "organization_candidate_references_job_positive",
        "organization_candidate_references_origin_v1",
        "organization_candidate_references_authority",
        "organization_candidate_references_tenant_candidate_fkey",
        "organization_candidate_references_tenant_application_unique",
        "organization_candidate_references_tenant_reference_unique",
        # PostgreSQL truncates migration 026's longer identifier to 63 bytes.
        "organization_candidate_references_tenant_reference_candidate_un",
    },
    "organization_candidate_resume_evidence": {
        "organization_candidate_resume_evidence_pkey",
        "organization_candidate_resume_evidence_tenant_nonblank",
        "organization_candidate_resume_evidence_version_v1",
        "organization_candidate_resume_evidence_source_kind",
        "organization_candidate_resume_evidence_content_digest",
        "organization_candidate_resume_evidence_byte_count",
        "organization_candidate_resume_evidence_media_type",
        "organization_candidate_resume_evidence_extracted_digest",
        "organization_candidate_resume_evidence_reference_fkey",
        "organization_candidate_resume_evidence_tenant_candidate_fkey",
        "organization_candidate_resume_evidence_reference_unique",
        "organization_candidate_resume_evidence_tenant_version_candidate",
        "organization_candidate_resume_evidence_tenant_version_unique",
    },
    "organization_candidate_ingest_receipts": {
        "organization_candidate_ingest_receipts_pkey",
        "organization_candidate_ingest_receipts_idempotency",
        "organization_candidate_ingest_receipts_input_digest",
        "organization_candidate_ingest_receipts_tenant_nonblank",
        "organization_candidate_ingest_receipts_resolution",
        "organization_candidate_ingest_receipts_authority",
        "organization_candidate_ingest_receipts_reference_fkey",
        "organization_candidate_ingest_receipts_resume_fkey",
        "organization_candidate_ingest_receipts_tenant_candidate_fkey",
        "organization_candidate_ingest_receipts_reference_unique",
        "organization_candidate_ingest_receipts_resume_unique",
    },
}
_ORGANIZATION_CANDIDATE_APPEND_ONLY_FUNCTION = "organization_candidate_evidence_append_only"
_ORGANIZATION_CANDIDATE_APPEND_ONLY_FUNCTION_BODY = (
    "declarehas_evidenceboolean;beginiftg_op<>'truncate'thenraiseexceptionusing"
    "errcode='55000',message=tg_table_name||'isappend-only(attempted'||tg_op||')';"
    "endif;executeformat('selectexists(select1from%ilimit1)',tg_table_name)into"
    "has_evidence;ifhas_evidencethenraiseexceptionusingerrcode='55000',message="
    "tg_table_name||'containscommittedevidenceandcannotbetruncated';endif;returnnull;end;"
)
_ORGANIZATION_CANDIDATE_TRIGGERS = tuple(
    trigger
    for table in _ORGANIZATION_CANDIDATE_TABLES
    for trigger in (
        (table, f"{table}_no_mutation", _ORGANIZATION_CANDIDATE_APPEND_ONLY_FUNCTION, 27),
        (
            table,
            f"{table}_no_nonempty_truncate",
            _ORGANIZATION_CANDIDATE_APPEND_ONLY_FUNCTION,
            34,
        ),
    )
)
_SOURCED_CANDIDATE_INDEXES = {
    "global_candidate_source_identities_candidate_idx",
    "global_candidate_source_identities_linkedin_idx",
    "global_candidate_source_observations_candidate_freshness_idx",
    "global_candidate_source_observations_provider_idx",
    "global_candidate_source_observations_acquisition_idx",
    "global_candidate_ingest_receipts_candidate_idx",
    "global_candidate_ingest_receipts_acquisition_idx",
}
_SOURCED_CANDIDATE_CONSTRAINTS_BY_TABLE = {
    "global_candidate_source_identities": {
        "global_candidate_source_identities_pkey",
        "global_candidate_source_identities_provider_v1",
        "global_candidate_source_identities_provider_id",
        "global_candidate_source_identities_linkedin",
        "global_candidate_source_identities_authority",
        "global_candidate_source_identities_provider_unique",
        "global_candidate_source_identities_global_candidate_id_fkey",
    },
    "global_candidate_source_observations": {
        "global_candidate_source_observations_pkey",
        "global_candidate_source_observations_idempotency_key_key",
        "global_candidate_source_observations_provider_v1",
        "global_candidate_source_observations_provider_id",
        "global_candidate_source_observations_linkedin",
        "global_candidate_source_observations_acquisition_receipt",
        "global_candidate_source_observations_generation",
        "global_candidate_source_observations_slot",
        "global_candidate_source_observations_profile",
        "global_candidate_source_observations_profile_digest",
        "global_candidate_source_observations_outcome",
        "global_candidate_source_observations_conflict_code",
        "global_candidate_source_observations_resolution_shape",
        "global_candidate_source_observations_authority",
        "global_candidate_source_observations_source_identity_id_fkey",
        "global_candidate_source_observations_global_candidate_id_fkey",
    },
    "global_candidate_ingest_receipts": {
        "global_candidate_ingest_receipts_pkey",
        "global_candidate_ingest_receipts_source_observation_id_key",
        "global_candidate_ingest_receipts_idempotency",
        "global_candidate_ingest_receipts_input_digest",
        "global_candidate_ingest_receipts_resolution",
        "global_candidate_ingest_receipts_provider_v1",
        "global_candidate_ingest_receipts_provider_id",
        "global_candidate_ingest_receipts_acquisition_receipt",
        "global_candidate_ingest_receipts_generation",
        "global_candidate_ingest_receipts_slot",
        "global_candidate_ingest_receipts_resolution_shape",
        "global_candidate_ingest_receipts_authority",
        "global_candidate_ingest_receipts_source_observation_id_fkey",
        "global_candidate_ingest_receipts_source_identity_id_fkey",
        "global_candidate_ingest_receipts_global_candidate_id_fkey",
    },
}
_SOURCED_CANDIDATE_APPEND_ONLY_FUNCTION = "approved_provider_candidate_evidence_append_only"
_SOURCED_CANDIDATE_APPEND_ONLY_FUNCTION_BODY = (
    "declarehas_evidenceboolean;beginiftg_op<>'truncate'thenraiseexceptionusing"
    "errcode='55000',message=tg_table_name||'isappend-only(attempted'||tg_op||')';"
    "endif;executeformat('selectexists(select1from%ilimit1)',tg_table_name)into"
    "has_evidence;ifhas_evidencethenraiseexceptionusingerrcode='55000',message="
    "tg_table_name||'containscommittedevidenceandcannotbetruncated';endif;returnnull;end;"
)
_SOURCED_CANDIDATE_TRIGGERS = tuple(
    trigger
    for table in _SOURCED_CANDIDATE_TABLES
    for trigger in (
        (
            table,
            f"{table}_no_mutation",
            _SOURCED_CANDIDATE_APPEND_ONLY_FUNCTION,
            27,
        ),
        (
            table,
            f"{table}_no_nonempty_truncate",
            _SOURCED_CANDIDATE_APPEND_ONLY_FUNCTION,
            34,
        ),
    )
)
_DECISION_INBOX_INDEXES = {
    "organization_decision_event_inbox_tenant_delivery_idx",
    "organization_decision_event_inbox_tenant_source_idx",
}
_DECISION_INBOX_CONSTRAINTS_BY_TABLE = {
    "organization_decision_event_inbox": {
        "organization_decision_event_inbox_pkey",
        "organization_decision_event_inbox_delivery_sequence_key",
        "organization_decision_event_inbox_source_event_sequence_key",
        "organization_decision_event_inbox_tenant_check",
        "organization_decision_event_inbox_source_check",
        "organization_decision_event_inbox_delivery_positive",
        "organization_decision_event_inbox_source_sequence_positive",
        "organization_decision_event_inbox_schema_v1",
        "organization_decision_event_inbox_organization_positive",
        "organization_decision_event_inbox_subject_v1",
        "organization_decision_event_inbox_action_v1",
        "organization_decision_event_inbox_taxonomy_positive",
        "organization_decision_event_inbox_rubric_shape",
        "organization_decision_event_inbox_jd_digest_positive",
        "organization_decision_event_inbox_recommendation_v1",
        "organization_decision_event_inbox_reason_bounded",
        "organization_decision_event_inbox_before_state_v1",
        "organization_decision_event_inbox_after_state_v1",
        "organization_decision_event_inbox_state_changed",
        "organization_decision_event_inbox_digest_check",
    },
    "organization_decision_stream_state": {
        "organization_decision_stream_state_pkey",
        "organization_decision_stream_state_tenant_id_check",
        "organization_decision_stream_state_state_check",
        "organization_decision_stream_state_last_delivery_sequence_check",
        "organization_decision_stream_source_sequence_check",
        "organization_decision_stream_state_last_event_id_key",
        "organization_decision_stream_state_last_event_id_fkey",
    },
}
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
    "candidates": {"candidates_scope_check"},
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
        "candidates",
        "candidates_scope_check",
    ): "check((scope=any(array['shared','organization_private'])))",
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


def _decision_tenant_policy_expression_ok(expression: str) -> bool:
    return _normalize_sql_definition(expression) == (
        "(tenant_id=current_setting('app.current_tenant_id',true))"
    )


def _organization_candidate_tenant_policy_expression_ok(expression: str) -> bool:
    # Migration 026 excludes quarantine tenants through table CHECK constraints.
    return _normalize_sql_definition(expression) == (
        "(tenant_id=current_setting('app.current_tenant_id',true))"
    )


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


CANDIDATE_CONSENT_TABLES = (
    "candidate_consent_state",
    "candidate_consent_sources",
    "candidate_consent_receipts",
)
CANDIDATE_CONSENT_UPDATE_COLUMNS = (
    "global_candidate_id",
    "highest_version",
    "last_action",
    "effective_version",
    "effective_action",
    "active_source_id",
    "effective_at",
    "updated_at",
)

# Used by the real release provisioner and the API's existing single catalog census.
# Only this source-owned role placeholder is substituted; operator names remain query parameters.
CANDIDATE_CONSENT_CATALOG_SQL = """
WITH consent_role AS (SELECT __CONSENT_ROLE__::text AS name)
SELECT (
  SELECT count(*)=3 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
  WHERE n.nspname='public' AND c.relname IN ('candidate_consent_state','candidate_consent_sources','candidate_consent_receipts')
    AND c.relrowsecurity AND c.relforcerowsecurity
    AND has_table_privilege((SELECT name FROM consent_role),c.oid,'SELECT')
    AND has_table_privilege((SELECT name FROM consent_role),c.oid,'INSERT')
    AND NOT has_table_privilege((SELECT name FROM consent_role),c.oid,'UPDATE')
    AND NOT has_table_privilege((SELECT name FROM consent_role),c.oid,'DELETE')
    AND NOT has_table_privilege((SELECT name FROM consent_role),c.oid,'TRUNCATE')
    AND NOT has_table_privilege((SELECT name FROM consent_role),c.oid,'REFERENCES')
    AND NOT has_table_privilege((SELECT name FROM consent_role),c.oid,'TRIGGER')
    AND NOT EXISTS(SELECT 1 FROM aclexplode(c.relacl) acl WHERE acl.grantee=0)
    AND NOT EXISTS(SELECT 1 FROM pg_attribute a WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
      AND has_column_privilege((SELECT name FROM consent_role),c.oid,a.attnum,'UPDATE') IS DISTINCT FROM
        (c.relname='candidate_consent_state' AND a.attname IN ('global_candidate_id','highest_version','last_action',
          'effective_version','effective_action','active_source_id','effective_at','updated_at')))
    AND (SELECT count(*)=1 AND bool_and(p.polcmd='*' AND p.polpermissive AND p.polroles=ARRAY[0::oid]
      AND regexp_replace(pg_get_expr(p.polqual,p.polrelid),'[[:space:]]','','g')=
        '(tenant_id=current_setting(''app.current_tenant_id''::text,true))'
      AND regexp_replace(pg_get_expr(p.polwithcheck,p.polrelid),'[[:space:]]','','g')=
        '(tenant_id=current_setting(''app.current_tenant_id''::text,true))')
      FROM pg_policy p WHERE p.polrelid=c.oid)
) AND (
  SELECT count(*)=5 FROM (VALUES
    ('candidate_consent_state','consent_state_binding','candidate_consent_binding_immutable',19),
    ('candidate_consent_sources','consent_sources_no_mutation','candidate_consent_append_only',27),
    ('candidate_consent_sources','consent_sources_no_truncate','candidate_consent_append_only',34),
    ('candidate_consent_receipts','consent_receipts_no_mutation','candidate_consent_append_only',27),
    ('candidate_consent_receipts','consent_receipts_no_truncate','candidate_consent_append_only',34)
  ) expected(relation,trigger_name,function_name,trigger_type)
  JOIN pg_trigger t ON t.tgrelid=to_regclass('public.'||expected.relation) AND t.tgname=expected.trigger_name
  JOIN pg_proc p ON p.oid=t.tgfoid AND p.proname=expected.function_name
  JOIN pg_namespace n ON n.oid=p.pronamespace AND n.nspname='public'
  WHERE NOT t.tgisinternal AND t.tgenabled='O' AND t.tgtype=expected.trigger_type
    AND p.pronargs=0 AND p.prorettype='trigger'::regtype AND NOT p.prosecdef
    AND p.proconfig @> ARRAY['search_path=pg_catalog, public']::text[]
    AND (p.proname<>'candidate_consent_append_only' OR p.proconfig @> ARRAY['row_security=off']::text[])
    AND NOT has_function_privilege((SELECT name FROM consent_role),p.oid,'EXECUTE')
) AND (
  SELECT count(*)=44 FROM (VALUES
    ('candidate_consent_receipts','candidate_consent_receipts_action_check','c','f8baec0ffc2899951c35ca936fe97813'),
    ('candidate_consent_receipts','candidate_consent_receipts_check','c','77de686c544c483941670e4b4fa0d712'),
    ('candidate_consent_receipts','candidate_consent_receipts_check1','c','41414f887f2a18003b0e51a848f29201'),
    ('candidate_consent_receipts','candidate_consent_receipts_command_digest_check','c','33104cd83ac3c89eb46b2ada38dbc5bb'),
    ('candidate_consent_receipts','candidate_consent_receipts_effective_action_check','c','c4097c1792801089189293ca8213f836'),
    ('candidate_consent_receipts','candidate_consent_receipts_effective_version_check','c','db178b81e60dfafe261180e651906293'),
    ('candidate_consent_receipts','candidate_consent_receipts_global_candidate_id_fkey','f','0fef2c0b3373b84134f3716e334547e0'),
    ('candidate_consent_receipts','candidate_consent_receipts_idempotency_key_check','c','4e6f49a4abb058a5eb136e4de2214b3b'),
    ('candidate_consent_receipts','candidate_consent_receipts_idempotency_key_key','u','1edb563e32a4bf1046f01b8e99e8e3bd'),
    ('candidate_consent_receipts','candidate_consent_receipts_outcome_check','c','b2c25fa2f05af59824ea668079aa8f6f'),
    ('candidate_consent_receipts','candidate_consent_receipts_pkey','p','1e909a41847e2371a95110af439aa15e'),
    ('candidate_consent_receipts','candidate_consent_receipts_tenant_id_subject_id_fkey','f','fb45fd42d273695a05dbcf1b70173c09'),
    ('candidate_consent_receipts','candidate_consent_receipts_tenant_id_subject_id_source_id_fkey','f','97f4402e8ebd5a0853f3cd0c514c3d25'),
    ('candidate_consent_receipts','candidate_consent_receipts_tenant_id_subject_id_version_key','u','a2e05bda1d2f0a439bb1ea16d1707fdb'),
    ('candidate_consent_receipts','candidate_consent_receipts_verified_actor_id_check','c','505fcb32315f40235005615b5ba44b90'),
    ('candidate_consent_receipts','candidate_consent_receipts_verified_issuer_check','c','5442308800c6314cf6e9beb22d5514f0'),
    ('candidate_consent_receipts','candidate_consent_receipts_version_check','c','72175895b5c5708506e5936cb1053585'),
    ('candidate_consent_sources','candidate_consent_sources_check','c','77de686c544c483941670e4b4fa0d712'),
    ('candidate_consent_sources','candidate_consent_sources_copy_sha256_check','c','7868e2e17dd2299588997beeb4a9b46b'),
    ('candidate_consent_sources','candidate_consent_sources_copy_version_check','c','e1fb59b2d3de11f2b37240e7639c3abf'),
    ('candidate_consent_sources','candidate_consent_sources_global_candidate_id_fkey','f','0fef2c0b3373b84134f3716e334547e0'),
    ('candidate_consent_sources','candidate_consent_sources_pkey','p','fa7c89ca3643d31a04934ae436c5f197'),
    ('candidate_consent_sources','candidate_consent_sources_profile_sha256_check','c','b64ffeb0bec2311b0243eb3c3aaa38eb'),
    ('candidate_consent_sources','candidate_consent_sources_purpose_check','c','4adae87552136aa5b4ddd45e8453d0fb'),
    ('candidate_consent_sources','candidate_consent_sources_purpose_version_check','c','10ae93a6dc0cdd5c761e3db41a0e4d25'),
    ('candidate_consent_sources','candidate_consent_sources_resume_sha256_check','c','f0c8293196e6150ca589ceec72a10333'),
    ('candidate_consent_sources','candidate_consent_sources_source_version_check','c','788b8c6e78d44e34af3f92e22f2ca164'),
    ('candidate_consent_sources','candidate_consent_sources_tenant_id_subject_id_fkey','f','fb45fd42d273695a05dbcf1b70173c09'),
    ('candidate_consent_sources','candidate_consent_sources_tenant_id_subject_id_source_id_key','u','f25d79b17c2779702d67f868a7430073'),
    ('candidate_consent_sources','candidate_consent_sources_tenant_id_subject_id_source_versi_key','u','a5a8123e0b1afd70a4d48d4af1ee078f'),
    ('candidate_consent_sources','consent_source_profile_shape','c','fd9b8acc0a140c608516508821cb2d45'),
    ('candidate_consent_sources','consent_source_resume_shape','c','414a5d751f852275893add948f5f690e'),
    ('candidate_consent_state','candidate_consent_state_check','c','77de686c544c483941670e4b4fa0d712'),
    ('candidate_consent_state','candidate_consent_state_check1','c','873ed466b93e4f76533803ce6a32f0f4'),
    ('candidate_consent_state','candidate_consent_state_check2','c','5b45731dceb819340c8331c7b9533ddf'),
    ('candidate_consent_state','candidate_consent_state_check3','c','4c311dad7447ea221a3a6ce3f7b81425'),
    ('candidate_consent_state','candidate_consent_state_effective_action_check','c','c4097c1792801089189293ca8213f836'),
    ('candidate_consent_state','candidate_consent_state_global_candidate_id_fkey','f','0fef2c0b3373b84134f3716e334547e0'),
    ('candidate_consent_state','candidate_consent_state_highest_version_check','c','8eea671cce1902bb147a2edd595fa7d8'),
    ('candidate_consent_state','candidate_consent_state_last_action_check','c','72abafd120bad43e7ced792c240db677'),
    ('candidate_consent_state','candidate_consent_state_pkey','p','ed6a3b6ca977ba2af4edc138d1371ce3'),
    ('candidate_consent_state','candidate_consent_state_tenant_id_key','u','5014d6ea211895d80fc06566189444a8'),
    ('candidate_consent_state','candidate_consent_state_tenant_id_subject_id_key','u','e3056d9812dd366333f2524d1ba35e0c'),
    ('candidate_consent_state','consent_active_source_fk','f','0658c772a722599db92541ebdd19ed56')
  ) expected(relation,name,type,definition_hash)
  JOIN pg_constraint c ON c.conrelid=to_regclass('public.'||expected.relation) AND c.conname=expected.name
  WHERE c.contype::text=expected.type AND c.convalidated AND NOT c.condeferrable
    AND md5(pg_get_constraintdef(c.oid))=expected.definition_hash
) AS ready
"""


def bounded_readiness_check(
    candidate_repository: Any,
    *,
    unsafe_search_configuration: bool,
    jwt_enabled: bool,
    jwt_problems: list[str],
    privacy_problems: list[str] | None = None,
    privacy_key_versions: set[int] | None = None,
    decision_inbox_enabled: bool | None = None,
    organization_candidate_intake_enabled: bool | None = None,
    candidate_consent_intake_enabled: bool | None = None,
    sourced_candidate_ingest_mode: str | None = None,
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
    check_decision_inbox = decision_inbox_enabled is not None
    if decision_inbox_enabled is False:
        reasons.append("decision_inbox_disabled")
    check_organization_candidates = organization_candidate_intake_enabled is not None
    if organization_candidate_intake_enabled is False:
        reasons.append("organization_candidate_intake_disabled")
    if candidate_consent_intake_enabled is False:
        reasons.append("candidate_consent_intake_disabled")
    if sourced_candidate_ingest_mode is not None and sourced_candidate_ingest_mode not in {
        "dual",
        "canonical_only",
    }:
        reasons.append("sourced_candidate_ingest_disabled")
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
                    (
                        list(
                            _CANDIDATE_TABLES
                            + _SHARED_TABLES
                            + _PRIVACY_TABLES
                            + _DECISION_INBOX_TABLES
                            + _SOURCED_CANDIDATE_TABLES
                            + _ORGANIZATION_CANDIDATE_TABLES
                        ),
                    ),
                )
                relation_rows = cur.fetchall()
                relation_map = {row[0]: row for row in relation_rows}
                if set(_CANDIDATE_TABLES + _SHARED_TABLES) - set(relation_map):
                    reasons.append("required_schema_missing")
                if set(_PRIVACY_TABLES) - set(relation_map):
                    reasons.append("candidate_privacy_schema_missing")
                elif any(not bool(relation_map[table][1]) for table in _PRIVACY_TABLES):
                    reasons.append("candidate_privacy_rls_incomplete")
                if check_decision_inbox and set(_DECISION_INBOX_TABLES) - set(relation_map):
                    reasons.append("decision_inbox_schema_missing")
                elif check_decision_inbox and any(
                    not bool(relation_map[table][1]) or not bool(relation_map[table][2])
                    for table in _DECISION_INBOX_TABLES
                ):
                    reasons.append("decision_inbox_rls_incomplete")
                elif (
                    check_decision_inbox
                    and not allow_owner
                    and any(
                        relation_map[table][3] == relation_map[table][4]
                        for table in _DECISION_INBOX_TABLES
                    )
                ):
                    reasons.append("runtime_role_owns_decision_inbox")
                if set(_SOURCED_CANDIDATE_TABLES) - set(relation_map):
                    reasons.append("sourced_candidate_schema_missing")
                elif not allow_owner and any(
                    relation_map[table][3] == relation_map[table][4]
                    for table in _SOURCED_CANDIDATE_TABLES
                ):
                    reasons.append("runtime_role_owns_sourced_candidate_authority")
                if check_organization_candidates and set(_ORGANIZATION_CANDIDATE_TABLES) - set(
                    relation_map
                ):
                    reasons.append("organization_candidate_schema_missing")
                elif check_organization_candidates and any(
                    not bool(relation_map[table][1]) or not bool(relation_map[table][2])
                    for table in _ORGANIZATION_CANDIDATE_TABLES
                ):
                    reasons.append("organization_candidate_rls_incomplete")
                elif (
                    check_organization_candidates
                    and not allow_owner
                    and any(
                        relation_map[table][3] == relation_map[table][4]
                        for table in _ORGANIZATION_CANDIDATE_TABLES
                    )
                ):
                    reasons.append("runtime_role_owns_organization_candidate_authority")
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
                    (
                        list(
                            _CANDIDATE_TABLES
                            + _PRIVACY_TABLES
                            + _DECISION_INBOX_TABLES
                            + _ORGANIZATION_CANDIDATE_TABLES
                        ),
                    ),
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
                for table, policy_name in (
                    (
                        "organization_decision_event_inbox",
                        "organization_decision_event_inbox_tenant",
                    ),
                    (
                        "organization_decision_stream_state",
                        "organization_decision_stream_state_tenant",
                    ),
                ):
                    policy = policies.get((table, policy_name))
                    if check_decision_inbox and (
                        policy is None
                        or policy[2] != "PERMISSIVE"
                        or policy[3].lower() != "{public}"
                        or policy[4] != "ALL"
                        or not _decision_tenant_policy_expression_ok(policy[5])
                        or not _decision_tenant_policy_expression_ok(policy[6])
                    ):
                        reasons.append("decision_inbox_policy_unexpected")
                        break
                if check_decision_inbox and any(
                    table in _DECISION_INBOX_TABLES
                    and policy_name
                    not in {
                        "organization_decision_event_inbox_tenant",
                        "organization_decision_stream_state_tenant",
                    }
                    for table, policy_name in policies
                ):
                    reasons.append("decision_inbox_policy_unexpected")
                for table in _ORGANIZATION_CANDIDATE_TABLES:
                    tenant_policy = policies.get((table, f"tenant_isolation_{table}"))
                    admin_policy = policies.get((table, f"admin_all_{table}"))
                    if check_organization_candidates and (
                        tenant_policy is None
                        or tenant_policy[2] != "PERMISSIVE"
                        or tenant_policy[3].lower() != "{public}"
                        or tenant_policy[4] != "ALL"
                        or not _organization_candidate_tenant_policy_expression_ok(tenant_policy[5])
                        or not _organization_candidate_tenant_policy_expression_ok(tenant_policy[6])
                        or admin_policy is None
                        or admin_policy[2] != "PERMISSIVE"
                        or "admin_role" not in admin_policy[3].lower()
                        or admin_policy[4] != "ALL"
                        or _normalize_sql_definition(admin_policy[5]) != "true"
                        or _normalize_sql_definition(admin_policy[6]) != "true"
                    ):
                        reasons.append("organization_candidate_policy_unexpected")
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
                    SELECT 'decision_inbox_privilege'::text, 'runtime'::text,
                           jsonb_build_object(
                               'inbox_select', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_event_inbox', 'SELECT'
                               ),
                               'inbox_insert', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_event_inbox', 'INSERT'
                               ),
                               'inbox_update', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_event_inbox', 'UPDATE'
                               ),
                               'inbox_delete', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_event_inbox', 'DELETE'
                               ),
                               'inbox_truncate', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_event_inbox', 'TRUNCATE'
                               ),
                               'stream_select', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_stream_state', 'SELECT'
                               ),
                               'stream_insert', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_stream_state', 'INSERT'
                               ),
                               'stream_update', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_stream_state', 'UPDATE'
                               ),
                               'stream_delete', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_stream_state', 'DELETE'
                               ),
                               'stream_truncate', has_table_privilege(
                                   current_user,
                                   'public.organization_decision_stream_state', 'TRUNCATE'
                               )
                           )
                    UNION ALL
                    SELECT 'sourced_candidate_privilege'::text, c.relname::text,
                           jsonb_build_object(
                               'select', has_table_privilege(
                                   current_user, c.oid, 'SELECT'
                               ),
                               'insert', has_table_privilege(
                                   current_user, c.oid, 'INSERT'
                               ),
                               'update', has_table_privilege(
                                   current_user, c.oid, 'UPDATE'
                               ),
                               'delete', has_table_privilege(
                                   current_user, c.oid, 'DELETE'
                               ),
                               'truncate', has_table_privilege(
                                   current_user, c.oid, 'TRUNCATE'
                               ),
                               'references', has_table_privilege(
                                   current_user, c.oid, 'REFERENCES'
                               ),
                               'trigger', has_table_privilege(
                                   current_user, c.oid, 'TRIGGER'
                               )
                           )
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname='public' AND c.relname=ANY(%s)
                    UNION ALL
                    SELECT 'organization_candidate_privilege'::text, c.relname::text,
                           jsonb_build_object(
                               'select', has_table_privilege(current_user, c.oid, 'SELECT'),
                               'insert', has_table_privilege(current_user, c.oid, 'INSERT'),
                               'update', has_table_privilege(current_user, c.oid, 'UPDATE'),
                               'delete', has_table_privilege(current_user, c.oid, 'DELETE'),
                               'truncate', has_table_privilege(current_user, c.oid, 'TRUNCATE'),
                               'references', has_table_privilege(current_user, c.oid, 'REFERENCES'),
                               'trigger', has_table_privilege(current_user, c.oid, 'TRIGGER')
                           )
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid=c.relnamespace
                    WHERE n.nspname='public' AND c.relname=ANY(%s)
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
                    """
                    + " UNION ALL SELECT 'candidate_consent_catalog','authority',jsonb_build_object('ready',consent.ready) FROM ("
                    + CANDIDATE_CONSENT_CATALOG_SQL.replace("__CONSENT_ROLE__", "current_user")
                    + ") consent",
                    (
                        list(
                            _REQUIRED_INDEXES
                            | _PRIVACY_INDEXES
                            | _DECISION_INBOX_INDEXES
                            | _SOURCED_CANDIDATE_INDEXES
                            | _ORGANIZATION_CANDIDATE_INDEXES
                        ),
                        list(
                            _REQUIRED_FUNCTIONS
                            | _PRIVACY_FUNCTIONS
                            | {_SOURCED_CANDIDATE_APPEND_ONLY_FUNCTION}
                            | {_ORGANIZATION_CANDIDATE_APPEND_ONLY_FUNCTION}
                        ),
                        list(
                            _REQUIRED_CONSTRAINTS
                            | {
                                constraint
                                for names in _DECISION_INBOX_CONSTRAINTS_BY_TABLE.values()
                                for constraint in names
                            }
                            | {
                                constraint
                                for names in _SOURCED_CANDIDATE_CONSTRAINTS_BY_TABLE.values()
                                for constraint in names
                            }
                            | {
                                constraint
                                for names in _ORGANIZATION_CANDIDATE_CONSTRAINTS_BY_TABLE.values()
                                for constraint in names
                            }
                        ),
                        [trigger[1] for trigger in _SUPPRESSION_TRIGGERS]
                        + [trigger[1] for trigger in _PRIVACY_TRIGGERS]
                        + [trigger[1] for trigger in _SOURCED_CANDIDATE_TRIGGERS]
                        + [trigger[1] for trigger in _ORGANIZATION_CANDIDATE_TRIGGERS],
                        # No sequence is introduced by migration 026.
                        [_SUPPRESSION_SEQUENCE, _PRIVACY_SEQUENCE],
                        list(_PUBLIC_COLUMNS),
                        list(_SOURCED_CANDIDATE_TABLES),
                        list(_ORGANIZATION_CANDIDATE_TABLES),
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
                if check_decision_inbox and _DECISION_INBOX_INDEXES - set(indexes):
                    reasons.append("decision_inbox_index_missing")
                if _SOURCED_CANDIDATE_INDEXES - set(indexes):
                    reasons.append("sourced_candidate_index_missing")
                if check_organization_candidates and _ORGANIZATION_CANDIDATE_INDEXES - set(indexes):
                    reasons.append("organization_candidate_index_missing")
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
                sourced_append_only = functions.get(_SOURCED_CANDIDATE_APPEND_ONLY_FUNCTION)
                if sourced_append_only is None or (
                    int(sourced_append_only.get("arguments", -1)) != 0
                    or not bool(sourced_append_only.get("returns_trigger"))
                    or sourced_append_only.get("language") != "plpgsql"
                    or bool(sourced_append_only.get("security_definer"))
                    or _normalize_sql_definition(str(sourced_append_only.get("source", "")))
                    != _SOURCED_CANDIDATE_APPEND_ONLY_FUNCTION_BODY
                    or (not allow_owner and bool(sourced_append_only.get("owned_by_runtime")))
                ):
                    reasons.append("sourced_candidate_append_only_function_unexpected")
                organization_candidate_append_only = functions.get(
                    _ORGANIZATION_CANDIDATE_APPEND_ONLY_FUNCTION
                )
                if check_organization_candidates and (
                    organization_candidate_append_only is None
                    or int(organization_candidate_append_only.get("arguments", -1)) != 0
                    or not bool(organization_candidate_append_only.get("returns_trigger"))
                    or organization_candidate_append_only.get("language") != "plpgsql"
                    or bool(organization_candidate_append_only.get("security_definer"))
                    or _normalize_sql_definition(
                        str(organization_candidate_append_only.get("source", ""))
                    )
                    != _ORGANIZATION_CANDIDATE_APPEND_ONLY_FUNCTION_BODY
                    or (
                        not allow_owner
                        and bool(organization_candidate_append_only.get("owned_by_runtime"))
                    )
                ):
                    reasons.append("organization_candidate_append_only_function_unexpected")

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
                decision_constraint_keys = {
                    f"{table}.{constraint}"
                    for table, names in _DECISION_INBOX_CONSTRAINTS_BY_TABLE.items()
                    for constraint in names
                }
                if check_decision_inbox and decision_constraint_keys - set(constraints):
                    reasons.append("decision_inbox_constraint_missing")
                sourced_constraint_keys = {
                    f"{table}.{constraint}"
                    for table, names in _SOURCED_CANDIDATE_CONSTRAINTS_BY_TABLE.items()
                    for constraint in names
                }
                if sourced_constraint_keys - set(constraints):
                    reasons.append("sourced_candidate_constraint_missing")
                organization_candidate_constraint_keys = {
                    f"{table}.{constraint}"
                    for table, names in _ORGANIZATION_CANDIDATE_CONSTRAINTS_BY_TABLE.items()
                    for constraint in names
                }
                if check_organization_candidates and organization_candidate_constraint_keys - set(
                    constraints
                ):
                    reasons.append("organization_candidate_constraint_missing")
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
                for (
                    table,
                    trigger_name,
                    trigger_function_name,
                    trigger_type,
                ) in _SOURCED_CANDIDATE_TRIGGERS:
                    trigger = triggers.get(f"{table}.{trigger_name}")
                    if trigger is None:
                        reasons.append("sourced_candidate_trigger_missing")
                        break
                    if (
                        trigger.get("function") != trigger_function_name
                        or int(trigger.get("type", -1)) != trigger_type
                        or trigger.get("enabled") not in {"O", "A"}
                    ):
                        reasons.append("sourced_candidate_trigger_unexpected")
                        break
                for (
                    table,
                    trigger_name,
                    trigger_function_name,
                    trigger_type,
                ) in _ORGANIZATION_CANDIDATE_TRIGGERS:
                    trigger = triggers.get(f"{table}.{trigger_name}")
                    if check_organization_candidates and (
                        trigger is None
                        or trigger.get("function") != trigger_function_name
                        or int(trigger.get("type", -1)) != trigger_type
                        or trigger.get("enabled") not in {"O", "A"}
                    ):
                        reasons.append("organization_candidate_trigger_unexpected")
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
                decision_privileges = objects.get("decision_inbox_privilege", {}).get("runtime")
                if check_decision_inbox and (
                    decision_privileges is None
                    or (
                        not bool(decision_privileges.get("inbox_select"))
                        or not bool(decision_privileges.get("inbox_insert"))
                        or bool(decision_privileges.get("inbox_update"))
                        or bool(decision_privileges.get("inbox_delete"))
                        or bool(decision_privileges.get("inbox_truncate"))
                        or not bool(decision_privileges.get("stream_select"))
                        or not bool(decision_privileges.get("stream_insert"))
                        or not bool(decision_privileges.get("stream_update"))
                        or bool(decision_privileges.get("stream_delete"))
                        or bool(decision_privileges.get("stream_truncate"))
                    )
                ):
                    reasons.append("decision_inbox_privileges_unsafe")
                sourced_privileges = objects.get("sourced_candidate_privilege", {})
                if set(_SOURCED_CANDIDATE_TABLES) - set(sourced_privileges) or any(
                    not bool(details.get("select"))
                    or not bool(details.get("insert"))
                    or any(
                        bool(details.get(privilege))
                        for privilege in (
                            "update",
                            "delete",
                            "truncate",
                            "references",
                            "trigger",
                        )
                    )
                    for details in sourced_privileges.values()
                ):
                    reasons.append("sourced_candidate_privileges_unsafe")
                organization_candidate_privileges = objects.get(
                    "organization_candidate_privilege", {}
                )
                if check_organization_candidates and (
                    set(_ORGANIZATION_CANDIDATE_TABLES) - set(organization_candidate_privileges)
                    or any(
                        not bool(details.get("select"))
                        or not bool(details.get("insert"))
                        or any(
                            bool(details.get(privilege))
                            for privilege in (
                                "update",
                                "delete",
                                "truncate",
                                "references",
                                "trigger",
                            )
                        )
                        for details in organization_candidate_privileges.values()
                    )
                ):
                    reasons.append("organization_candidate_privileges_unsafe")
                if candidate_consent_intake_enabled is not None and not objects.get(
                    "candidate_consent_catalog", {}
                ).get("authority", {}).get("ready", False):
                    reasons.append("candidate_consent_authority_unsafe")
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
