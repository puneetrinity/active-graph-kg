#!/usr/bin/env python3
"""Run one explicitly authorized Memory schema release.

This is a manual release-service entrypoint, never an API/worker startup hook.
Responsibilities:

* base schema (db/init.sql) + extensions on a fresh database
* RLS policies (enable_rls_policies.sql — contains no login roles)
* ordered migrations from activekg.common.migration_manifest, tracked in a
  ``schema_migrations`` ledger under a Postgres advisory lock
* optional provisioning of the restricted runtime role from env vars

Production requires ACTIVEKG_MIGRATE_DSN, ACTIVEKG_MIGRATION_APPLY=1, an
existing exact target identity, environment and source commit. There is no DSN
fallback and no drift override. Fresh initialization is available only for a
positively proved local disposable target.

Runtime role provisioning (optional, all-or-nothing):
    ACTIVEKG_RUNTIME_ROLE      role name to create/harden (NOSUPERUSER,
                               NOBYPASSRLS, no ownership — RLS applies to it)
    ACTIVEKG_RUNTIME_PASSWORD  credential, injected via the environment;
                               never stored in this repository

Failure policy: a missing migration file, or any error that is not a
duplicate-object condition (e.g. unique-constraint violations from duplicate
data), aborts startup with a non-zero exit. Pre-ledger databases baseline
cleanly: objects that already exist are recorded as baselined, data errors
are not forgiven.
"""

import hashlib
import importlib.util
import os
import re
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import psycopg  # noqa: E402
from psycopg import sql  # noqa: E402

from activekg.common.migration_manifest import CHECKSUM_TRANSITIONS  # noqa: E402
from activekg.common.schema_control import (  # noqa: E402
    ADVISORY_LOCK_KEY,
    CONTROL_SCHEMA,
    RUNTIME_ROLE_DEFAULT,
    SchemaControlError,
    assert_identity,
    assert_ledger,
    create_control_schema,
    finish_attempt,
    grant_control_read,
    load_migration_records,
    manifest_digest,
    read_ledger,
    resolve_control_environment,
    safe_error_class,
    safe_target_fingerprint,
    start_attempt,
)

MAX_RETRIES = 10
RETRY_DELAY = 3  # seconds

# SQLSTATEs that mean "object already exists" — safe to baseline on a database
# migrated before the ledger existed. Data errors (23xxx) are deliberately
# absent: a unique violation during a migration is a real failure.
DUPLICATE_OBJECT_SQLSTATES = {
    "42P07",  # duplicate_table
    "42701",  # duplicate_column
    "42710",  # duplicate_object
    "42723",  # duplicate_function
    "42P06",  # duplicate_schema
}

# Baseline verifiers: before a migration may be recorded as baselined off a
# duplicate-object error, EVERY object it creates must already exist — one
# stray duplicate must not vouch for a partially migrated legacy database.
# Previously-baselined ledger rows are re-verified on every boot.
# Forms: ("table", name) | ("column", table, column) | ("index", name)
#        | ("constraint", name)
#        | ("constraint_definition", table, name, definition)
#        | ("constraint_definition", table, name, type, definition)
#        | ("policy", table, name) | ("function", name)
#        | ("trigger_function", name) | ("trigger_function_body", name, body)
#        | ("trigger", table, name[, function, tgtype])
#        | ("notnull", table, column) | ("forcerls", table)
#        | ("sequence", name) | ("serial", table, column, sequence)
#        | ("fk_delete", table, constraint, action)
#        | ("unique", table, constraint, comma-separated-columns)
#        | ("column_type", table, column, format_type)
#        | ("default", table, column, normalized-expression)
#        | ("rls", table)
#        | ("security_definer_function", regprocedure-signature)
BASELINE_VERIFIERS: dict[str, list[tuple[str, ...]]] = {
    "001_add_embedding_history_index.sql": [("index", "idx_embedding_history_created_at")],
    "004_add_external_id_index.sql": [
        ("index", "idx_nodes_external_id"),
        ("index", "idx_nodes_external_id_parent"),
    ],
    "add_text_search.sql": [
        ("column", "nodes", "text_search_vector"),
        ("function", "update_text_search_vector"),
        ("trigger", "nodes", "nodes_text_search_update"),
        ("index", "idx_nodes_text_search"),
    ],
    "005_connector_configs_table.sql": [
        ("table", "connector_configs"),
        ("index", "idx_connector_configs_enabled"),
        ("index", "idx_connector_configs_tenant"),
        ("function", "update_connector_configs_updated_at"),
        ("trigger", "connector_configs", "connector_configs_updated_at"),
    ],
    "006_add_key_version.sql": [
        ("column", "connector_configs", "key_version"),
        ("index", "idx_connector_configs_key_version"),
        ("index", "idx_connector_configs_provider_key_version"),
    ],
    "007_add_provider_check.sql": [("constraint", "chk_provider_valid")],
    "008_connector_cursors_table.sql": [
        ("table", "connector_cursors"),
        ("index", "idx_connector_cursors_provider"),
        ("index", "idx_connector_cursors_tenant"),
        ("index", "idx_connector_cursors_updated_at"),
        ("function", "update_connector_cursors_updated_at"),
        ("trigger", "connector_cursors", "connector_cursors_updated_at"),
    ],
    "009_embedding_queue_status.sql": [
        ("column", "nodes", "embedding_status"),
        ("column", "nodes", "embedding_error"),
        ("column", "nodes", "embedding_attempts"),
        ("column", "nodes", "embedding_updated_at"),
        ("index", "idx_nodes_embedding_status"),
        ("index", "idx_nodes_embedding_updated_at"),
    ],
    "010_update_text_search_vector.sql": [("function", "update_text_search_vector")],
    "011_unique_tenant_external_id.sql": [("index", "idx_nodes_tenant_external_id_unique")],
    "012_global_memory.sql": [
        ("table", "global_candidates"),
        ("table", "candidate_provenance"),
        ("table", "tenant_candidate_access"),
        ("table", "feedback_events"),
        ("index", "idx_gc_linkedin_id"),
        ("index", "idx_gc_linkedin_url"),
        ("index", "idx_gc_github_id"),
        ("index", "idx_gc_email_hash"),
        ("index", "idx_gc_role_family"),
        ("index", "idx_gc_location"),
        ("index", "idx_gc_skills"),
        ("index", "idx_gc_embedding_status"),
        ("index", "idx_gc_last_evidence"),
        ("index", "idx_cp_global_candidate"),
        ("index", "idx_cp_global_source_null_tenant"),
        ("index", "idx_cp_source_type"),
        ("index", "idx_cp_tenant"),
        ("index", "idx_tca_tenant"),
        ("index", "idx_tca_global_candidate"),
        ("index", "idx_tca_visibility"),
        ("index", "idx_fe_tenant_job"),
        ("index", "idx_fe_global_candidate"),
        ("index", "idx_fe_action"),
        ("index", "idx_fe_created"),
        ("index", "idx_fe_role_location"),
    ],
    "012_candidate_identity.sql": [
        ("table", "candidates"),
        ("table", "candidate_identifiers"),
        ("table", "candidate_source_records"),
        ("index", "idx_candidates_tenant"),
        ("index", "idx_candidates_node_id"),
        ("index", "idx_candidates_primary_email"),
        ("index", "idx_candidates_props"),
        ("index", "idx_candidate_identifiers_unique"),
        ("index", "idx_candidate_identifiers_candidate"),
        ("index", "idx_candidate_identifiers_lookup"),
        ("index", "idx_candidate_source_records_unique"),
        ("index", "idx_candidate_source_records_candidate"),
        ("index", "idx_candidate_source_records_payload"),
    ],
    "013_vantahire_provenance.sql": [
        ("column", "candidate_source_records", "org_id"),
        ("column", "candidate_source_records", "job_id"),
        ("column", "candidate_source_records", "effective_recruiter_id"),
        ("column", "candidate_source_records", "created_by_user_id"),
        ("column", "candidate_source_records", "resume_source"),
        ("index", "idx_csr_vantahire_org_id"),
        ("index", "idx_csr_vantahire_recruiter"),
        ("index", "idx_csr_vantahire_uploader"),
        ("index", "idx_csr_vantahire_job_id"),
    ],
    "014_signal_job_tags.sql": [
        ("column", "candidate_source_records", "job_tags"),
        ("index", "idx_csr_signal_job_tags"),
    ],
    "015_candidate_profile.sql": [
        ("column", "candidates", "profile"),
        ("column", "candidates", "headline"),
        ("column", "candidates", "location_raw"),
        ("column", "candidates", "skills"),
        ("column", "candidates", "seniority_level"),
        ("column", "candidates", "linkedin_url"),
        ("column", "candidates", "linkedin_id"),
        ("column", "candidates", "profile_picture_url"),
        ("index", "idx_candidates_skills_gin"),
        ("index", "idx_candidates_location"),
        ("index", "idx_candidates_seniority"),
        ("index", "idx_candidates_linkedin_id"),
    ],
    "016_candidate_rls.sql": [
        ("policy", "candidates", "tenant_isolation_candidates"),
        ("policy", "candidates", "admin_all_candidates"),
        ("policy", "candidate_identifiers", "tenant_isolation_candidate_identifiers"),
        ("policy", "candidate_identifiers", "admin_all_candidate_identifiers"),
        ("policy", "candidate_source_records", "tenant_isolation_candidate_source_records"),
        ("policy", "candidate_source_records", "admin_all_candidate_source_records"),
        ("constraint", "candidates_tenant_candidate_uniq"),
        ("constraint", "candidate_identifiers_tenant_candidate_fkey"),
        ("constraint", "candidate_source_records_tenant_candidate_fkey"),
        ("notnull", "candidates", "tenant_id"),
        ("notnull", "candidate_identifiers", "tenant_id"),
        ("notnull", "candidate_source_records", "tenant_id"),
    ],
    "017_reserve_quarantine_tenant.sql": [
        ("policy", "candidates", "tenant_isolation_candidates"),
        ("policy", "candidate_identifiers", "tenant_isolation_candidate_identifiers"),
        ("policy", "candidate_source_records", "tenant_isolation_candidate_source_records"),
    ],
    "018_tenant_nonblank.sql": [
        ("constraint", "candidates_tenant_nonblank"),
        ("constraint", "candidate_identifiers_tenant_nonblank"),
        ("constraint", "candidate_source_records_tenant_nonblank"),
    ],
    "019_global_reconciliation.sql": [
        ("column", "candidates", "global_candidate_id"),
        ("index", "idx_candidates_global_id"),
        ("table", "candidate_merge_queue"),
        ("index", "idx_cmq_open_pair"),
        ("index", "idx_cmq_status"),
    ],
    "020_embed_version.sql": [
        ("column", "global_candidates", "embed_version"),
        ("index", "idx_global_candidates_embed_version"),
    ],
    "021_public_memory_contact_evidence.sql": [
        ("column", "global_candidates", "public_profile"),
        ("column", "global_candidates", "public_profile_observed_at"),
        ("column", "global_candidates", "public_crustdata_person_id"),
        ("column", "global_candidates", "public_headline"),
        ("column", "global_candidates", "public_location_city"),
        ("column", "global_candidates", "public_location_country_code"),
        ("column", "global_candidates", "public_role_family"),
        ("column", "global_candidates", "public_seniority_band"),
        ("column", "global_candidates", "public_skills_normalized"),
        ("column", "global_candidates", "public_embedding"),
        ("column", "global_candidates", "public_embedding_status"),
        ("column", "global_candidates", "public_embed_version"),
        ("notnull", "global_candidates", "public_profile"),
        ("notnull", "global_candidates", "public_embedding_status"),
        ("notnull", "global_candidates", "public_embed_version"),
        ("constraint", "global_candidates_public_embedding_status_check"),
        ("constraint", "global_candidates_public_headline_from_profile"),
        ("function", "activekg_redact_public_contact_text"),
        ("function", "activekg_pick_public_fields"),
        ("function", "activekg_pick_public_rows"),
        ("function", "activekg_public_crustdata_projection"),
        ("function", "activekg_assert_public_crustdata_backfill_safe"),
        ("table", "candidate_contact_evidence"),
        ("column", "candidate_contact_evidence", "id"),
        ("column", "candidate_contact_evidence", "global_candidate_id"),
        ("column", "candidate_contact_evidence", "tenant_id"),
        ("column", "candidate_contact_evidence", "email"),
        ("column", "candidate_contact_evidence", "email_hash"),
        ("column", "candidate_contact_evidence", "provider"),
        ("column", "candidate_contact_evidence", "provider_record_id"),
        ("column", "candidate_contact_evidence", "confidence"),
        ("column", "candidate_contact_evidence", "observed_at"),
        ("column", "candidate_contact_evidence", "validated_at"),
        ("column", "candidate_contact_evidence", "status"),
        ("column", "candidate_contact_evidence", "suppressed_at"),
        ("column", "candidate_contact_evidence", "bounce_reason"),
        ("column", "candidate_contact_evidence", "is_primary"),
        ("column", "candidate_contact_evidence", "created_at"),
        ("column", "candidate_contact_evidence", "updated_at"),
        ("constraint", "candidate_contact_evidence_pkey"),
        ("constraint", "candidate_contact_evidence_global_candidate_fkey"),
        ("constraint", "candidate_contact_evidence_tenant_nonblank"),
        ("constraint", "candidate_contact_evidence_email_nonblank"),
        ("constraint", "candidate_contact_evidence_email_hash_nonblank"),
        ("constraint", "candidate_contact_evidence_provider_check"),
        ("constraint", "candidate_contact_evidence_confidence_check"),
        ("constraint", "candidate_contact_evidence_status_check"),
        ("constraint", "candidate_contact_evidence_provider_record_required"),
        ("constraint", "candidate_contact_evidence_primary_usable"),
        ("constraint", "candidate_contact_evidence_unique"),
        ("notnull", "candidate_contact_evidence", "id"),
        ("notnull", "candidate_contact_evidence", "global_candidate_id"),
        ("notnull", "candidate_contact_evidence", "tenant_id"),
        ("notnull", "candidate_contact_evidence", "email"),
        ("notnull", "candidate_contact_evidence", "email_hash"),
        ("notnull", "candidate_contact_evidence", "provider"),
        ("notnull", "candidate_contact_evidence", "confidence"),
        ("notnull", "candidate_contact_evidence", "observed_at"),
        ("notnull", "candidate_contact_evidence", "status"),
        ("notnull", "candidate_contact_evidence", "is_primary"),
        ("notnull", "candidate_contact_evidence", "created_at"),
        ("notnull", "candidate_contact_evidence", "updated_at"),
        ("forcerls", "candidate_contact_evidence"),
        ("index", "idx_gc_public_embedding_status"),
        ("index", "idx_gc_public_embed_version"),
        ("index", "idx_gc_public_location"),
        ("index", "idx_gc_public_role_family"),
        ("index", "idx_gc_public_crustdata_person_id"),
        ("index", "idx_cce_tenant_global_primary"),
        ("index", "idx_cce_email_hash"),
        ("index", "idx_cce_one_primary"),
        ("policy", "candidate_contact_evidence", "tenant_isolation_candidate_contact_evidence"),
        ("policy", "candidate_contact_evidence", "admin_all_candidate_contact_evidence"),
        ("table", "contact_suppression_tombstones"),
        ("column", "contact_suppression_tombstones", "email_hash"),
        ("column", "contact_suppression_tombstones", "global_candidate_id"),
        ("column", "contact_suppression_tombstones", "reason"),
        ("column", "contact_suppression_tombstones", "first_observed_at"),
        ("column", "contact_suppression_tombstones", "last_observed_at"),
        ("column", "contact_suppression_tombstones", "source_evidence_id"),
        ("column", "contact_suppression_tombstones", "provider_event_id"),
        ("constraint", "contact_suppression_tombstones_pkey"),
        ("constraint", "contact_suppression_source_evidence_fkey"),
        ("constraint", "contact_suppression_global_candidate_fkey"),
        ("constraint", "contact_suppression_email_hash_nonblank"),
        ("constraint", "contact_suppression_reason_check"),
        ("notnull", "contact_suppression_tombstones", "email_hash"),
        ("notnull", "contact_suppression_tombstones", "reason"),
        ("notnull", "contact_suppression_tombstones", "first_observed_at"),
        ("notnull", "contact_suppression_tombstones", "last_observed_at"),
        ("index", "idx_contact_suppression_provider_event"),
        ("index", "idx_contact_suppression_global_candidate"),
        ("table", "public_candidate_market_memberships"),
        ("column", "public_candidate_market_memberships", "global_candidate_id"),
        ("column", "public_candidate_market_memberships", "coarse_market_key"),
        ("column", "public_candidate_market_memberships", "role_family"),
        ("column", "public_candidate_market_memberships", "location_city"),
        ("column", "public_candidate_market_memberships", "location_country_code"),
        ("column", "public_candidate_market_memberships", "seniority_band"),
        ("column", "public_candidate_market_memberships", "first_observed_at"),
        ("column", "public_candidate_market_memberships", "last_observed_at"),
        ("constraint", "public_candidate_market_global_candidate_fkey"),
        ("constraint", "public_candidate_market_key_nonblank"),
        ("constraint", "public_candidate_market_role_nonblank"),
        ("constraint", "public_candidate_market_city_nonblank"),
        ("constraint", "public_candidate_market_country_code_check"),
        ("constraint", "public_candidate_market_seniority_nonblank"),
        ("constraint", "public_candidate_market_memberships_pkey"),
        ("notnull", "public_candidate_market_memberships", "global_candidate_id"),
        ("notnull", "public_candidate_market_memberships", "coarse_market_key"),
        ("notnull", "public_candidate_market_memberships", "role_family"),
        ("notnull", "public_candidate_market_memberships", "location_city"),
        ("notnull", "public_candidate_market_memberships", "location_country_code"),
        ("notnull", "public_candidate_market_memberships", "seniority_band"),
        ("notnull", "public_candidate_market_memberships", "first_observed_at"),
        ("notnull", "public_candidate_market_memberships", "last_observed_at"),
        ("index", "idx_pcmm_market_last_observed"),
    ],
    "022_contact_suppression_person_and_audit.sql": [
        ("table", "contact_person_suppressions"),
        ("column", "contact_person_suppressions", "global_candidate_id"),
        ("column", "contact_person_suppressions", "reason"),
        ("column", "contact_person_suppressions", "first_observed_at"),
        ("column", "contact_person_suppressions", "last_observed_at"),
        ("column", "contact_person_suppressions", "provider_event_id"),
        ("column_type", "contact_person_suppressions", "global_candidate_id", "uuid"),
        ("column_type", "contact_person_suppressions", "reason", "text"),
        (
            "column_type",
            "contact_person_suppressions",
            "first_observed_at",
            "timestamp with time zone",
        ),
        (
            "column_type",
            "contact_person_suppressions",
            "last_observed_at",
            "timestamp with time zone",
        ),
        ("column_type", "contact_person_suppressions", "provider_event_id", "text"),
        ("default", "contact_person_suppressions", "first_observed_at", "now()"),
        ("default", "contact_person_suppressions", "last_observed_at", "now()"),
        ("notnull", "contact_person_suppressions", "global_candidate_id"),
        ("notnull", "contact_person_suppressions", "reason"),
        ("notnull", "contact_person_suppressions", "first_observed_at"),
        ("notnull", "contact_person_suppressions", "last_observed_at"),
        ("notnull", "contact_person_suppressions", "provider_event_id"),
        ("constraint", "contact_person_suppressions_pkey"),
        ("constraint", "contact_person_suppressions_global_candidate_fkey"),
        ("constraint", "contact_person_suppression_reason_check"),
        ("constraint", "contact_person_suppression_provider_event_hash"),
        (
            "constraint_definition",
            "contact_person_suppressions",
            "contact_person_suppressions_pkey",
            "p",
            "primarykey(global_candidate_id)",
        ),
        (
            "constraint_definition",
            "contact_person_suppressions",
            "contact_person_suppressions_global_candidate_fkey",
            "f",
            "foreignkey(global_candidate_id)referencesglobal_candidates(id)ondeleterestrict",
        ),
        (
            "constraint_definition",
            "contact_person_suppressions",
            "contact_person_suppression_reason_check",
            "check((reason='complaint'))",
        ),
        (
            "constraint_definition",
            "contact_person_suppressions",
            "contact_person_suppression_provider_event_hash",
            "check((provider_event_id~'^[0-9a-f]{64}$'))",
        ),
        (
            "fk_delete",
            "contact_person_suppressions",
            "contact_person_suppressions_global_candidate_fkey",
            "r",
        ),
        ("table", "contact_suppression_receipts"),
        ("column", "contact_suppression_receipts", "id"),
        ("column", "contact_suppression_receipts", "email_hash"),
        ("column", "contact_suppression_receipts", "global_candidate_id"),
        ("column", "contact_suppression_receipts", "signal_candidate_id"),
        ("column", "contact_suppression_receipts", "reason"),
        ("column", "contact_suppression_receipts", "scope"),
        ("column", "contact_suppression_receipts", "evidence_present"),
        ("column", "contact_suppression_receipts", "tenant_id"),
        ("column", "contact_suppression_receipts", "issuer"),
        ("column", "contact_suppression_receipts", "actor_id"),
        ("column", "contact_suppression_receipts", "actor_type"),
        ("column", "contact_suppression_receipts", "provider_event_id"),
        ("column", "contact_suppression_receipts", "created_at"),
        ("column_type", "contact_suppression_receipts", "id", "bigint"),
        ("column_type", "contact_suppression_receipts", "email_hash", "text"),
        ("column_type", "contact_suppression_receipts", "global_candidate_id", "uuid"),
        ("column_type", "contact_suppression_receipts", "signal_candidate_id", "text"),
        ("column_type", "contact_suppression_receipts", "reason", "text"),
        ("column_type", "contact_suppression_receipts", "scope", "text"),
        ("column_type", "contact_suppression_receipts", "evidence_present", "boolean"),
        ("column_type", "contact_suppression_receipts", "tenant_id", "text"),
        ("column_type", "contact_suppression_receipts", "issuer", "text"),
        ("column_type", "contact_suppression_receipts", "actor_id", "text"),
        ("column_type", "contact_suppression_receipts", "actor_type", "text"),
        ("column_type", "contact_suppression_receipts", "provider_event_id", "text"),
        (
            "column_type",
            "contact_suppression_receipts",
            "created_at",
            "timestamp with time zone",
        ),
        ("default", "contact_suppression_receipts", "created_at", "now()"),
        ("notnull", "contact_suppression_receipts", "id"),
        ("notnull", "contact_suppression_receipts", "email_hash"),
        ("notnull", "contact_suppression_receipts", "reason"),
        ("notnull", "contact_suppression_receipts", "scope"),
        ("notnull", "contact_suppression_receipts", "evidence_present"),
        ("notnull", "contact_suppression_receipts", "tenant_id"),
        ("notnull", "contact_suppression_receipts", "issuer"),
        ("notnull", "contact_suppression_receipts", "actor_id"),
        ("notnull", "contact_suppression_receipts", "actor_type"),
        ("notnull", "contact_suppression_receipts", "provider_event_id"),
        ("notnull", "contact_suppression_receipts", "created_at"),
        ("constraint", "contact_suppression_receipts_pkey"),
        ("constraint", "contact_suppression_receipt_email_hash_check"),
        ("constraint", "contact_suppression_receipt_signal_candidate_nonblank"),
        ("constraint", "contact_suppression_receipt_tenant_nonblank"),
        ("constraint", "contact_suppression_receipt_provider_event_hash"),
        ("constraint", "contact_suppression_receipt_authority_check"),
        ("constraint", "contact_suppression_receipt_scope_reason_check"),
        ("constraint", "contact_suppression_receipts_provider_event_unique"),
        (
            "constraint_definition",
            "contact_suppression_receipts",
            "contact_suppression_receipts_pkey",
            "p",
            "primarykey(id)",
        ),
        (
            "constraint_definition",
            "contact_suppression_receipts",
            "contact_suppression_receipt_email_hash_check",
            "check((email_hash~'^[0-9a-f]{64}$'))",
        ),
        (
            "constraint_definition",
            "contact_suppression_receipts",
            "contact_suppression_receipt_signal_candidate_nonblank",
            "check(((signal_candidate_idisnull)or(btrim(signal_candidate_id)<>'')))",
        ),
        (
            "constraint_definition",
            "contact_suppression_receipts",
            "contact_suppression_receipt_tenant_nonblank",
            "check(((btrim(tenant_id)<>'')and(tenant_id<>'__quarantine__')))",
        ),
        (
            "constraint_definition",
            "contact_suppression_receipts",
            "contact_suppression_receipt_provider_event_hash",
            "check((provider_event_id~'^[0-9a-f]{64}$'))",
        ),
        (
            "constraint_definition",
            "contact_suppression_receipts",
            "contact_suppression_receipt_authority_check",
            "check(((btrim(issuer)<>'')and(btrim(actor_id)<>'')and(actor_type='service')))",
        ),
        (
            "constraint_definition",
            "contact_suppression_receipts",
            "contact_suppression_receipt_scope_reason_check",
            "check((((reason='hard_bounce')and(scope='address'))or"
            "((reason='complaint')and(scope='person')and(global_candidate_idisnotnull)"
            "and(signal_candidate_idisnotnull))))",
        ),
        (
            "unique",
            "contact_suppression_receipts",
            "contact_suppression_receipts_provider_event_unique",
            "issuer,provider_event_id",
        ),
        ("index", "idx_contact_suppression_receipts_email_hash"),
        ("index", "idx_contact_suppression_receipts_candidate"),
        ("index", "idx_contact_suppression_receipts_tenant_created"),
        ("sequence", "contact_suppression_receipts_id_seq"),
        (
            "serial",
            "contact_suppression_receipts",
            "id",
            "contact_suppression_receipts_id_seq",
        ),
        ("trigger_function", "contact_suppression_receipts_append_only"),
        (
            "trigger_function_body",
            "contact_suppression_receipts_append_only",
            "beginraiseexception'contact_suppression_receiptsisappend-only(attempted%)',tg_op;end;",
        ),
        (
            "trigger",
            "contact_suppression_receipts",
            "contact_suppression_receipts_no_mutation",
            "contact_suppression_receipts_append_only",
            "27",
        ),
        (
            "trigger",
            "contact_suppression_receipts",
            "contact_suppression_receipts_no_truncate",
            "contact_suppression_receipts_append_only",
            "34",
        ),
        (
            "constraint_definition",
            "contact_suppression_tombstones",
            "contact_suppression_reason_check",
            "check((reason=any(array['hard_bounce','complaint'])))",
        ),
        ("constraint", "contact_suppression_provider_event_hash"),
        (
            "constraint_definition",
            "contact_suppression_tombstones",
            "contact_suppression_provider_event_hash",
            "check(((provider_event_idisnull)or(provider_event_id~'^[0-9a-f]{64}$')))",
        ),
        ("forcerls", "candidate_contact_evidence"),
    ],
    "023_candidate_privacy_directives.sql": [
        ("table", "candidate_privacy_directive_events"),
        ("column", "candidate_privacy_directive_events", "cursor"),
        ("column", "candidate_privacy_directive_events", "event_id"),
        ("column", "candidate_privacy_directive_events", "directive_id"),
        ("column", "candidate_privacy_directive_events", "directive_version"),
        ("column", "candidate_privacy_directive_events", "request_id"),
        ("column", "candidate_privacy_directive_events", "event_type"),
        ("column", "candidate_privacy_directive_events", "action"),
        ("column", "candidate_privacy_directive_events", "scope"),
        ("column", "candidate_privacy_directive_events", "resulting_state"),
        ("column", "candidate_privacy_directive_events", "authority_type"),
        ("column", "candidate_privacy_directive_events", "evidence_ref"),
        ("column", "candidate_privacy_directive_events", "reason_code"),
        ("column", "candidate_privacy_directive_events", "issuer"),
        ("column", "candidate_privacy_directive_events", "actor_id"),
        ("column", "candidate_privacy_directive_events", "actor_type"),
        ("column", "candidate_privacy_directive_events", "global_candidate_id"),
        ("column", "candidate_privacy_directive_events", "candidate_tenant_id"),
        ("column", "candidate_privacy_directive_events", "candidate_id"),
        ("column", "candidate_privacy_directive_events", "key_version"),
        ("column", "candidate_privacy_directive_events", "schema_version"),
        ("column", "candidate_privacy_directive_events", "effective_at"),
        ("column", "candidate_privacy_directive_events", "created_at"),
        ("table", "candidate_privacy_directives"),
        ("column", "candidate_privacy_directives", "directive_id"),
        ("column", "candidate_privacy_directives", "action"),
        ("column", "candidate_privacy_directives", "scope"),
        ("column", "candidate_privacy_directives", "state"),
        ("column", "candidate_privacy_directives", "version"),
        ("column", "candidate_privacy_directives", "authority_type"),
        ("column", "candidate_privacy_directives", "reason_code"),
        ("column", "candidate_privacy_directives", "global_candidate_id"),
        ("column", "candidate_privacy_directives", "candidate_tenant_id"),
        ("column", "candidate_privacy_directives", "candidate_id"),
        ("column", "candidate_privacy_directives", "last_event_cursor"),
        ("column", "candidate_privacy_directives", "effective_at"),
        ("column", "candidate_privacy_directives", "created_at"),
        ("column", "candidate_privacy_directives", "updated_at"),
        ("table", "candidate_privacy_identity_tokens"),
        ("column", "candidate_privacy_identity_tokens", "directive_id"),
        ("column", "candidate_privacy_identity_tokens", "identifier_type"),
        ("column", "candidate_privacy_identity_tokens", "key_version"),
        ("column", "candidate_privacy_identity_tokens", "token"),
        ("column", "candidate_privacy_identity_tokens", "created_at"),
        ("sequence", "candidate_privacy_directive_events_cursor_seq"),
        (
            "serial",
            "candidate_privacy_directive_events",
            "cursor",
            "candidate_privacy_directive_events_cursor_seq",
        ),
        ("index", "candidate_privacy_events_cursor_idx"),
        ("index", "candidate_privacy_events_directive_idx"),
        ("index", "candidate_privacy_directives_global_idx"),
        ("index", "candidate_privacy_directives_candidate_idx"),
        ("index", "candidate_privacy_identity_tokens_lookup_idx"),
        ("constraint", "candidate_privacy_directive_events_pkey"),
        ("constraint", "candidate_privacy_directive_events_event_id_key"),
        ("constraint", "candidate_privacy_directive_events_directive_version_check"),
        ("constraint", "candidate_privacy_directive_events_event_type_check"),
        ("constraint", "candidate_privacy_directive_events_action_check"),
        ("constraint", "candidate_privacy_directive_events_scope_check"),
        ("constraint", "candidate_privacy_directive_events_resulting_state_check"),
        ("constraint", "candidate_privacy_directive_events_authority_type_check"),
        ("constraint", "candidate_privacy_directive_events_reason_code_check"),
        ("constraint", "candidate_privacy_directive_events_issuer_check"),
        ("constraint", "candidate_privacy_directive_events_actor_id_check"),
        ("constraint", "candidate_privacy_directive_events_actor_type_check"),
        ("constraint", "candidate_privacy_directive_events_key_version_check"),
        ("constraint", "candidate_privacy_directive_events_schema_version_check"),
        ("constraint", "candidate_privacy_directive_events_global_candidate_id_fkey"),
        ("constraint", "candidate_privacy_event_action_scope_check"),
        ("constraint", "candidate_privacy_event_candidate_pair_check"),
        ("constraint", "candidate_privacy_event_candidate_fkey"),
        ("constraint", "candidate_privacy_event_directive_version_unique"),
        ("constraint", "candidate_privacy_event_request_type_unique"),
        ("constraint", "candidate_privacy_directives_pkey"),
        ("constraint", "candidate_privacy_directives_action_check"),
        ("constraint", "candidate_privacy_directives_scope_check"),
        ("constraint", "candidate_privacy_directives_state_check"),
        ("constraint", "candidate_privacy_directives_version_check"),
        ("constraint", "candidate_privacy_directives_authority_type_check"),
        ("constraint", "candidate_privacy_directives_reason_code_check"),
        ("constraint", "candidate_privacy_directives_global_candidate_id_fkey"),
        ("constraint", "candidate_privacy_directives_last_event_cursor_key"),
        ("constraint", "candidate_privacy_directives_last_event_cursor_fkey"),
        ("constraint", "candidate_privacy_directive_action_scope_check"),
        ("constraint", "candidate_privacy_directive_candidate_pair_check"),
        ("constraint", "candidate_privacy_directive_candidate_fkey"),
        ("constraint", "candidate_privacy_identity_tokens_pkey"),
        ("constraint", "candidate_privacy_identity_tokens_directive_id_fkey"),
        ("constraint", "candidate_privacy_identity_tokens_identifier_type_check"),
        ("constraint", "candidate_privacy_identity_tokens_key_version_check"),
        ("constraint", "candidate_privacy_identity_tokens_token_check"),
        ("rls", "candidate_privacy_directive_events"),
        ("rls", "candidate_privacy_directives"),
        ("rls", "candidate_privacy_identity_tokens"),
        ("policy", "candidate_privacy_directive_events", "candidate_privacy_events_runtime_read"),
        ("policy", "candidate_privacy_directives", "candidate_privacy_directives_runtime_read"),
        ("trigger_function", "candidate_privacy_append_only"),
        (
            "trigger",
            "candidate_privacy_directive_events",
            "candidate_privacy_events_no_mutation",
            "candidate_privacy_append_only",
            "27",
        ),
        (
            "trigger",
            "candidate_privacy_directive_events",
            "candidate_privacy_events_no_truncate",
            "candidate_privacy_append_only",
            "34",
        ),
        (
            "trigger",
            "candidate_privacy_identity_tokens",
            "candidate_privacy_tokens_no_mutation",
            "candidate_privacy_append_only",
            "27",
        ),
        (
            "trigger",
            "candidate_privacy_identity_tokens",
            "candidate_privacy_tokens_no_truncate",
            "candidate_privacy_append_only",
            "34",
        ),
        (
            "security_definer_function",
            "candidate_privacy_decision_for(uuid,text,uuid)",
        ),
        ("security_definer_function", "candidate_privacy_global_decision(uuid)"),
        ("security_definer_function", "candidate_privacy_candidate_decision(text,uuid)"),
        ("security_definer_function", "candidate_privacy_node_decision(uuid)"),
        ("security_definer_function", "candidate_privacy_resolve_subject(text,bytea)"),
        (
            "security_definer_function",
            "candidate_privacy_resolve_canonical(uuid,text,uuid)",
        ),
        (
            "security_definer_function",
            "candidate_privacy_match(jsonb,uuid,text,uuid)",
        ),
        ("security_definer_function", "candidate_privacy_token_key_versions()"),
        (
            "security_definer_function",
            "candidate_privacy_create_directive(uuid,uuid,text,text,text,uuid,text,text,text,uuid,text,uuid,integer,jsonb,boolean,timestamp with time zone)",
        ),
        (
            "security_definer_function",
            "candidate_privacy_transition_directive(uuid,bigint,uuid,text,uuid,text,text,text,timestamp with time zone)",
        ),
    ],
    "024_organization_decision_event_inbox.sql": [
        ("table", "organization_decision_event_inbox"),
        ("table", "organization_decision_stream_state"),
        ("index", "organization_decision_event_inbox_tenant_delivery_idx"),
        ("index", "organization_decision_event_inbox_tenant_source_idx"),
        ("constraint", "organization_decision_event_inbox_pkey"),
        ("constraint", "organization_decision_event_inbox_delivery_sequence_key"),
        ("constraint", "organization_decision_event_inbox_source_event_sequence_key"),
        ("constraint", "organization_decision_event_inbox_state_changed"),
        ("constraint", "organization_decision_event_inbox_digest_check"),
        ("constraint", "organization_decision_stream_state_pkey"),
        ("constraint", "organization_decision_stream_state_last_event_id_key"),
        ("constraint", "organization_decision_stream_state_last_event_id_fkey"),
        ("forcerls", "organization_decision_event_inbox"),
        ("forcerls", "organization_decision_stream_state"),
        ("policy", "organization_decision_event_inbox", "organization_decision_event_inbox_tenant"),
        (
            "policy",
            "organization_decision_stream_state",
            "organization_decision_stream_state_tenant",
        ),
    ],
}


def _normalize_sql_definition(value: str) -> str:
    return re.sub(r"\s+", "", value.lower()).replace("::text", "")


def _object_exists(cur: psycopg.Cursor, check: tuple[str, ...]) -> bool:
    kind = check[0]
    if kind == "table":
        cur.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_name = %s",
            (check[1],),
        )
    elif kind == "column":
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s AND column_name = %s",
            (check[1], check[2]),
        )
    elif kind == "index":
        cur.execute(
            "SELECT 1 FROM pg_class i "
            "JOIN pg_namespace n ON n.oid = i.relnamespace "
            "JOIN pg_index ix ON ix.indexrelid = i.oid "
            "WHERE n.nspname = 'public' AND i.relname = %s "
            "AND ix.indisvalid AND ix.indisready",
            (check[1],),
        )
    elif kind == "constraint":
        cur.execute(
            "SELECT 1 FROM pg_constraint con "
            "JOIN pg_namespace n ON n.oid = con.connamespace "
            "WHERE n.nspname = 'public' AND con.conname = %s AND con.convalidated",
            (check[1],),
        )
    elif kind == "constraint_definition":
        cur.execute(
            "SELECT con.contype, con.convalidated, pg_get_constraintdef(con.oid) "
            "FROM pg_constraint con "
            "JOIN pg_class c ON c.oid = con.conrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s AND con.conname = %s",
            (check[1], check[2]),
        )
        row = cur.fetchone()
        expected_type = check[3] if len(check) == 5 else "c"
        expected_definition = check[4] if len(check) == 5 else check[3]
        return bool(
            row
            and row[0] == expected_type
            and row[1]
            and _normalize_sql_definition(row[2]) == expected_definition
        )
    elif kind == "policy":
        cur.execute(
            "SELECT 1 FROM pg_policies WHERE tablename = %s AND policyname = %s",
            (check[1], check[2]),
        )
    elif kind == "function":
        cur.execute(
            "SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE n.nspname = 'public' AND p.proname = %s",
            (check[1],),
        )
    elif kind == "trigger_function":
        cur.execute(
            "SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE n.nspname = 'public' AND p.proname = %s "
            "AND p.pronargs = 0 AND p.prorettype = 'trigger'::regtype",
            (check[1],),
        )
    elif kind == "trigger_function_body":
        cur.execute(
            "SELECT p.pronargs, p.prorettype = 'trigger'::regtype, l.lanname, "
            "p.prosecdef, p.prosrc "
            "FROM pg_proc p "
            "JOIN pg_namespace n ON n.oid = p.pronamespace "
            "JOIN pg_language l ON l.oid = p.prolang "
            "WHERE n.nspname = 'public' AND p.proname = %s",
            (check[1],),
        )
        row = cur.fetchone()
        return bool(
            row
            and row[0] == 0
            and row[1]
            and row[2] == "plpgsql"
            and not row[3]
            and _normalize_sql_definition(row[4]) == check[2]
        )
    elif kind == "trigger":
        sql_text = (
            "SELECT 1 FROM pg_trigger t "
            "JOIN pg_class c ON c.oid = t.tgrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_proc p ON p.oid = t.tgfoid "
            "WHERE n.nspname = 'public' AND c.relname = %s AND t.tgname = %s "
            "AND NOT t.tgisinternal AND t.tgenabled IN ('O', 'A')"
        )
        params: tuple[str, ...] = (check[1], check[2])
        if len(check) == 5:
            sql_text += " AND p.proname = %s AND t.tgtype = %s"
            params += (check[3], check[4])
        cur.execute(sql_text, params)
    elif kind == "notnull":
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = %s "
            "AND column_name = %s AND is_nullable = 'NO'",
            (check[1], check[2]),
        )
    elif kind == "forcerls":
        cur.execute(
            "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s "
            "AND c.relrowsecurity AND c.relforcerowsecurity",
            (check[1],),
        )
    elif kind == "rls":
        cur.execute(
            "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s AND c.relrowsecurity",
            (check[1],),
        )
    elif kind == "security_definer_function":
        cur.execute(
            "SELECT p.prosecdef, p.proconfig FROM pg_proc p WHERE p.oid = to_regprocedure(%s)",
            (check[1],),
        )
        row = cur.fetchone()
        return bool(row and row[0] and row[1] == ["search_path=pg_catalog, public"])
    elif kind == "sequence":
        cur.execute(
            "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s AND c.relkind = 'S'",
            (check[1],),
        )
    elif kind == "serial":
        cur.execute(
            "SELECT 1 FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum "
            "WHERE n.nspname = 'public' AND c.relname = %s AND a.attname = %s "
            "AND pg_get_serial_sequence(%s, %s) = %s "
            "AND lower(pg_get_expr(d.adbin, d.adrelid)) LIKE 'nextval(%%'",
            (
                check[1],
                check[2],
                f"public.{check[1]}",
                check[2],
                f"public.{check[3]}",
            ),
        )
    elif kind == "fk_delete":
        cur.execute(
            "SELECT 1 FROM pg_constraint con "
            "JOIN pg_class c ON c.oid = con.conrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s "
            "AND con.conname = %s AND con.contype = 'f' "
            "AND con.confdeltype = %s AND con.convalidated",
            (check[1], check[2], check[3]),
        )
    elif kind == "unique":
        cur.execute(
            "SELECT 1 FROM pg_constraint con "
            "JOIN pg_class c ON c.oid = con.conrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s "
            "AND con.conname = %s AND con.contype = 'u' AND con.convalidated "
            "AND replace(lower(pg_get_constraintdef(con.oid)), ' ', '') = %s",
            (check[1], check[2], f"unique({check[3].lower()})"),
        )
    elif kind == "column_type":
        cur.execute(
            "SELECT 1 FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relname = %s AND a.attname = %s "
            "AND NOT a.attisdropped AND format_type(a.atttypid, a.atttypmod) = %s",
            (check[1], check[2], check[3]),
        )
    elif kind == "default":
        cur.execute(
            "SELECT 1 FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum "
            "WHERE n.nspname = 'public' AND c.relname = %s AND a.attname = %s "
            "AND replace(lower(pg_get_expr(d.adbin, d.adrelid)), ' ', '') = %s",
            (check[1], check[2], check[3].lower().replace(" ", "")),
        )
    else:
        return False
    return cur.fetchone() is not None


def _verify_baseline(cur: psycopg.Cursor, migration_file: str) -> tuple[bool, str]:
    """Return (ok, detail) — whether every object this migration creates exists."""
    checks = BASELINE_VERIFIERS.get(migration_file)
    if checks is None:
        return False, "no baseline verifier defined for this migration"
    missing = [" ".join(c) for c in checks if not _object_exists(cur, c)]
    if missing:
        return False, f"objects missing: {', '.join(missing)}"
    return True, ""


def _load_manifest() -> tuple[str, ...]:
    """Load the migration manifest without importing the activekg package."""
    manifest_path = os.path.join(
        os.path.dirname(__file__), "..", "activekg", "common", "migration_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("migration_manifest", manifest_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MIGRATIONS


def _connect_with_retry(dsn: str) -> psycopg.Connection:
    """Try connecting to the database with retries (Railway services start concurrently)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return psycopg.connect(dsn, autocommit=True)
        except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout):
            if attempt == MAX_RETRIES:
                raise
            print(f"  Connection attempt {attempt}/{MAX_RETRIES} failed")
            print(f"  Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)
    raise RuntimeError("unreachable")


def _is_duplicate_object_error(e: Exception) -> bool:
    # When a SQLSTATE is available it is authoritative: a unique-data
    # violation (23505) carries "already exists" in its DETAIL text and must
    # NOT be mistaken for a duplicate object. The string fallback only covers
    # errors that surface without a SQLSTATE (e.g. wrapped DO-block DDL).
    sqlstate = getattr(e, "sqlstate", None)
    if sqlstate is not None:
        return sqlstate in DUPLICATE_OBJECT_SQLSTATES
    return "already exists" in str(e).lower()


def _ensure_extensions_and_schema(cur: psycopg.Cursor) -> None:
    print("Checking pgvector extension availability...")
    cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector';")
    if not cur.fetchone():
        print("ERROR: pgvector extension is not available in this PostgreSQL instance")
        print("Deploy a PostgreSQL image that ships pgvector (e.g. pgvector/pgvector:pg16).")
        sys.exit(1)
    print("✓ pgvector extension is available")

    print("Creating extensions...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    print("✓ Extensions created")

    cur.execute("SELECT 1 FROM information_schema.tables WHERE table_name = 'nodes';")
    if cur.fetchone() is not None:
        print("✓ Database schema already initialized (skipping init.sql)")
    else:
        print("Initializing database schema...")
        init_sql_path = os.path.join(os.path.dirname(__file__), "..", "db", "init.sql")
        with open(init_sql_path) as f:
            cur.execute(f.read())
        print("✓ Database schema initialized")

    rls_sql_path = os.path.join(os.path.dirname(__file__), "..", "enable_rls_policies.sql")
    if os.path.exists(rls_sql_path):
        print("Applying RLS policies...")
        with open(rls_sql_path) as f:
            rls_sql = f.read()
        try:
            cur.execute(rls_sql)
            print("✓ RLS policies applied")
        except Exception as e:
            if _is_duplicate_object_error(e):
                print("⊙ RLS policies already applied (skipped)")
            else:
                raise


def _execute_migration_sql(cur: psycopg.Cursor, sql_text: str) -> None:
    if "create index concurrently" not in sql_text.lower():
        cur.execute(sql_text)
        return
    # Execute statements one-by-one (needed for CREATE INDEX CONCURRENTLY)
    for raw_stmt in sql_text.split(";"):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        has_sql = any(
            line.strip() and not line.strip().startswith("--") for line in stmt.splitlines()
        )
        if has_sql:
            cur.execute(stmt)


def _apply_migrations(cur: psycopg.Cursor, migrations: tuple[str, ...]) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            baselined BOOLEAN NOT NULL DEFAULT false
        )
        """
    )
    cur.execute("ALTER TABLE schema_migrations ADD COLUMN IF NOT EXISTS checksum TEXT")
    cur.execute("SELECT filename, checksum, baselined FROM schema_migrations")
    rows = cur.fetchall()
    ledger = {row[0]: row[1] for row in rows}

    # Baselined rows were accepted on trust that their objects exist — re-prove
    # that on every boot so a bad historical baseline cannot persist silently.
    for filename, _checksum, was_baselined in rows:
        if not was_baselined or filename not in migrations:
            continue
        ok, detail = _verify_baseline(cur, filename)
        if not ok:
            print(
                f"ERROR: previously baselined migration {filename} fails "
                f"re-verification: {detail}. The database is partially "
                "migrated; reconcile it manually (or delete the ledger row to "
                "re-apply the migration)."
            )
            sys.exit(1)

    applied = 0
    baselined = 0
    ledger_skipped = 0

    for migration_file in migrations:
        migration_path = os.path.join(
            os.path.dirname(__file__), "..", "db", "migrations", migration_file
        )
        if not os.path.exists(migration_path):
            print(f"ERROR: migration listed in manifest but missing on disk: {migration_file}")
            sys.exit(1)

        with open(migration_path) as f:
            migration_sql = f.read()
        checksum = hashlib.sha256(migration_sql.encode("utf-8")).hexdigest()

        if migration_file in ledger:
            recorded = ledger[migration_file]
            if recorded is None:
                # Rows recorded before the checksum column existed: adopt the
                # current file as the trusted baseline.
                cur.execute(
                    "UPDATE schema_migrations SET checksum = %s WHERE filename = %s",
                    (checksum, migration_file),
                )
                print(f"⊙ Backfilled checksum for {migration_file}")
            elif recorded != checksum:
                transition = CHECKSUM_TRANSITIONS.get(migration_file, {}).get(recorded)
                if transition == checksum:
                    cur.execute(
                        "UPDATE schema_migrations SET checksum = %s WHERE filename = %s",
                        (checksum, migration_file),
                    )
                    print(
                        f"⊙ {migration_file}: known checksum transition applied "
                        f"({recorded[:12]} → {checksum[:12]})"
                    )
                    ledger_skipped += 1
                    continue
                if (
                    os.getenv("ACTIVEKG_SCHEMA_ENVIRONMENT") != "production"
                    and os.getenv("ACTIVEKG_ALLOW_MIGRATION_DRIFT", "false").lower() == "true"
                ):
                    print(
                        f"WARNING: {migration_file} changed since it was applied "
                        f"(recorded {recorded[:12]}, on disk {checksum[:12]}); "
                        "continuing because ACTIVEKG_ALLOW_MIGRATION_DRIFT=true."
                    )
                else:
                    print(
                        f"ERROR: {migration_file} changed since it was applied "
                        f"(recorded {recorded[:12]}, on disk {checksum[:12]}). "
                        "Applied migrations are immutable; add a new migration instead, "
                        "Production drift cannot be overridden."
                    )
                    sys.exit(1)
            ledger_skipped += 1
            continue

        print(f"Applying migration: {migration_file}...")
        try:
            _execute_migration_sql(cur, migration_sql)
        except Exception as e:
            if _is_duplicate_object_error(e):
                # One duplicate error is not proof the whole migration is
                # present — verify every object it creates before baselining.
                ok, detail = _verify_baseline(cur, migration_file)
                if not ok:
                    print(
                        f"ERROR: migration {migration_file} hit a duplicate-object "
                        f"error but cannot be baselined: {detail}. The database "
                        "appears partially migrated; reconcile it manually."
                    )
                    sys.exit(1)
                print(f"⊙ Migration {migration_file} baselined (all objects verified present)")
                cur.execute(
                    "INSERT INTO schema_migrations (filename, baselined, checksum) "
                    "VALUES (%s, true, %s)",
                    (migration_file, checksum),
                )
                baselined += 1
                continue
            print(f"ERROR: migration {migration_file} failed ({safe_error_class(e)})")
            sys.exit(1)

        # IF NOT EXISTS can make a partially pre-existing table look like a
        # successful migration. Re-prove migrations with an object manifest
        # before writing their ledger row, not only after duplicate errors.
        if migration_file in BASELINE_VERIFIERS:
            ok, detail = _verify_baseline(cur, migration_file)
            if not ok:
                print(
                    f"ERROR: migration {migration_file} completed but failed "
                    f"post-apply verification: {detail}. The database is partially migrated."
                )
                sys.exit(1)

        cur.execute(
            "INSERT INTO schema_migrations (filename, checksum) VALUES (%s, %s)",
            (migration_file, checksum),
        )
        print(f"✓ Migration {migration_file} applied")
        applied += 1

    print(
        f"Migrations complete (applied={applied}, baselined={baselined}, "
        f"ledger_skipped={ledger_skipped})"
    )


def _provision_runtime_role(cur: psycopg.Cursor) -> None:
    role = os.environ.get("ACTIVEKG_RUNTIME_ROLE")
    if not role:
        print("⊙ ACTIVEKG_RUNTIME_ROLE not set — skipping runtime role provisioning")
        return
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role):
        print(f"ERROR: invalid ACTIVEKG_RUNTIME_ROLE name: {role!r}")
        sys.exit(1)

    # Refuse to provision reserved or self-defeating role names: hardening the
    # migration user would demote the owner mid-flight, and app_user is set
    # NOLOGIN by the legacy remediation right after provisioning.
    cur.execute("SELECT current_user")
    migration_user = cur.fetchone()[0]
    if role in {migration_user, "postgres", "app_user", "admin_role"}:
        print(
            f"ERROR: ACTIVEKG_RUNTIME_ROLE must be a dedicated role, not {role!r} "
            f"(migration user: {migration_user!r}; reserved: postgres, app_user, admin_role)"
        )
        sys.exit(1)

    password = os.environ.get("ACTIVEKG_RUNTIME_PASSWORD")
    role_ident = sql.Identifier(role)

    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
    exists = cur.fetchone() is not None

    if not exists:
        if not password:
            print("ERROR: ACTIVEKG_RUNTIME_PASSWORD required to create the runtime role")
            sys.exit(1)
        cur.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN PASSWORD {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS"
            ).format(role_ident, sql.Literal(password))
        )
        print(f"✓ Runtime role {role} created (NOSUPERUSER NOBYPASSRLS)")
    else:
        # Enforce the security posture even if the role pre-exists.
        cur.execute(
            sql.SQL("ALTER ROLE {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS").format(
                role_ident
            )
        )
        if password:
            cur.execute(
                sql.SQL("ALTER ROLE {} PASSWORD {}").format(role_ident, sql.Literal(password))
            )
        print(f"✓ Runtime role {role} hardened (NOSUPERUSER NOBYPASSRLS)")

    # Data access only — never ownership, so RLS policies apply.
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA public TO {}").format(role_ident))
    cur.execute(
        sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {}").format(
            role_ident
        )
    )
    cur.execute(
        sql.SQL("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {}").format(role_ident)
    )
    cur.execute(
        sql.SQL(
            "ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {}"
        ).format(role_ident)
    )

    # Compliance state is intentionally non-destructive. The API may upsert
    # tombstones/person suppressions, and may select+insert audit receipts, but
    # it must never erase suppression history or rewrite append-only receipts.
    cur.execute(
        sql.SQL(
            "REVOKE DELETE, TRUNCATE ON "
            "contact_suppression_tombstones, contact_person_suppressions, "
            "contact_suppression_receipts FROM {}"
        ).format(role_ident)
    )
    cur.execute(sql.SQL("REVOKE UPDATE ON contact_suppression_receipts FROM {}").format(role_ident))

    # The runtime role must never hold the admin_role RLS bypass — including
    # through inherited (indirect) membership.
    cur.execute("SELECT pg_has_role(%s, 'admin_role', 'MEMBER')", (role,))
    if cur.fetchone()[0]:
        cur.execute(sql.SQL("REVOKE admin_role FROM {}").format(role_ident))
        cur.execute("SELECT pg_has_role(%s, 'admin_role', 'MEMBER')", (role,))
        if cur.fetchone()[0]:
            print(
                f"ERROR: {role} still inherits admin_role through an intermediate "
                "role; revoke that membership chain manually."
            )
            sys.exit(1)
        print(f"✓ Revoked admin_role membership from {role}")

    # The ledger is readiness-trusted state: the app may read it, never write it.
    cur.execute(
        sql.SQL("REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON schema_migrations FROM {}").format(
            role_ident
        )
    )
    print(f"✓ Runtime role {role} granted table access (no ownership; ledger/receipts hardened)")


CANDIDATE_PRIVACY_RUNTIME_FUNCTIONS = (
    "candidate_privacy_decision_for(uuid,text,uuid)",
    "candidate_privacy_global_decision(uuid)",
    "candidate_privacy_candidate_decision(text,uuid)",
    "candidate_privacy_node_decision(uuid)",
    "candidate_privacy_resolve_subject(text,bytea)",
    "candidate_privacy_resolve_canonical(uuid,text,uuid)",
    "candidate_privacy_match(jsonb,uuid,text,uuid)",
    "candidate_privacy_token_key_versions()",
    (
        "candidate_privacy_create_directive(uuid,uuid,text,text,text,uuid,text,text,text,"
        "uuid,text,uuid,integer,jsonb,boolean,timestamp with time zone)"
    ),
    (
        "candidate_privacy_transition_directive(uuid,bigint,uuid,text,uuid,text,text,text,"
        "timestamp with time zone)"
    ),
)


def _harden_candidate_privacy_runtime_privileges(cur: psycopg.Cursor, role: str) -> None:
    """Restore the exact 023 runtime privilege boundary after blanket app grants."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role):
        raise SchemaControlError("ACTIVEKG_RUNTIME_ROLE is invalid")
    cur.execute("SELECT current_user")
    migration_user = cur.fetchone()[0]
    if role in {migration_user, "postgres", "app_user", "admin_role"}:
        raise SchemaControlError("ACTIVEKG_RUNTIME_ROLE is reserved")
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,))
    if cur.fetchone() is None:
        raise SchemaControlError("ACTIVEKG_RUNTIME_ROLE does not exist")
    role_ident = sql.Identifier(role)
    cur.execute("SELECT to_regclass('public.candidate_privacy_directives')")
    if cur.fetchone()[0] is None:
        raise SchemaControlError("candidate privacy authority is missing")

    cur.execute(
        sql.SQL(
            "REVOKE ALL ON candidate_privacy_directive_events, "
            "candidate_privacy_directives, candidate_privacy_identity_tokens FROM {}"
        ).format(role_ident)
    )
    cur.execute(
        sql.SQL(
            "REVOKE ALL ON SEQUENCE candidate_privacy_directive_events_cursor_seq FROM {}"
        ).format(role_ident)
    )
    cur.execute(
        sql.SQL(
            "GRANT SELECT ON candidate_privacy_directive_events, candidate_privacy_directives TO {}"
        ).format(role_ident)
    )
    for function_signature in CANDIDATE_PRIVACY_RUNTIME_FUNCTIONS:
        function_sql = sql.SQL(function_signature)
        cur.execute(sql.SQL("REVOKE ALL ON FUNCTION {} FROM {}").format(function_sql, role_ident))
        cur.execute(sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(function_sql, role_ident))
    cur.execute(
        sql.SQL("REVOKE ALL ON FUNCTION candidate_privacy_append_only() FROM {}").format(role_ident)
    )


def _assert_candidate_privacy_runtime_privileges(cur: psycopg.Cursor, role: str) -> None:
    checks: list[tuple[str, str, bool]] = []
    for relation in (
        "candidate_privacy_directive_events",
        "candidate_privacy_directives",
    ):
        checks.append(("table", f"public.{relation}", True))
    checks.append(("token_table", "public.candidate_privacy_identity_tokens", False))

    for kind, relation, select_expected in checks:
        cur.execute(
            "SELECT has_table_privilege(%s,%s,'SELECT'), "
            "has_table_privilege(%s,%s,'INSERT'), "
            "has_table_privilege(%s,%s,'UPDATE'), "
            "has_table_privilege(%s,%s,'DELETE'), "
            "has_table_privilege(%s,%s,'TRUNCATE'), "
            "has_table_privilege(%s,%s,'REFERENCES'), "
            "has_table_privilege(%s,%s,'TRIGGER')",
            tuple(value for _ in range(7) for value in (role, relation)),
        )
        privileges = cur.fetchone()
        if privileges != (select_expected, False, False, False, False, False, False):
            raise SchemaControlError(f"candidate privacy {kind} privileges are invalid")

    cur.execute(
        "SELECT has_sequence_privilege(%s,"
        "'public.candidate_privacy_directive_events_cursor_seq','USAGE'), "
        "has_sequence_privilege(%s,"
        "'public.candidate_privacy_directive_events_cursor_seq','SELECT'), "
        "has_sequence_privilege(%s,"
        "'public.candidate_privacy_directive_events_cursor_seq','UPDATE')",
        (role, role, role),
    )
    if cur.fetchone() != (False, False, False):
        raise SchemaControlError("candidate privacy sequence privileges are invalid")

    for function_signature in CANDIDATE_PRIVACY_RUNTIME_FUNCTIONS:
        cur.execute(
            "SELECT has_function_privilege(%s,%s,'EXECUTE')",
            (role, f"public.{function_signature}"),
        )
        if cur.fetchone() != (True,):
            raise SchemaControlError("candidate privacy function privileges are invalid")
    cur.execute(
        "SELECT has_function_privilege(%s,'public.candidate_privacy_append_only()','EXECUTE')",
        (role,),
    )
    if cur.fetchone() != (False,):
        raise SchemaControlError("candidate privacy trigger function privilege is invalid")


def _harden_decision_inbox_runtime_privileges(cur: psycopg.Cursor, role: str) -> None:
    """Restore the exact 024 tenant-inbox ACL after blanket application grants."""

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role):
        raise SchemaControlError("ACTIVEKG_RUNTIME_ROLE is invalid")
    cur.execute("SELECT current_user")
    migration_user = cur.fetchone()[0]
    if role in {migration_user, "postgres", "app_user", "admin_role"}:
        raise SchemaControlError("ACTIVEKG_RUNTIME_ROLE is reserved")
    cur.execute("SELECT 1 FROM pg_roles WHERE rolname=%s", (role,))
    if cur.fetchone() is None:
        raise SchemaControlError("ACTIVEKG_RUNTIME_ROLE does not exist")
    cur.execute(
        "SELECT to_regclass('public.organization_decision_event_inbox'), "
        "to_regclass('public.organization_decision_stream_state')"
    )
    if cur.fetchone() != (
        "organization_decision_event_inbox",
        "organization_decision_stream_state",
    ):
        raise SchemaControlError("organization decision inbox authority is missing")
    role_ident = sql.Identifier(role)
    cur.execute(
        sql.SQL(
            "REVOKE ALL ON organization_decision_event_inbox, "
            "organization_decision_stream_state FROM {}"
        ).format(role_ident)
    )
    cur.execute(
        sql.SQL("GRANT SELECT, INSERT ON organization_decision_event_inbox TO {}").format(
            role_ident
        )
    )
    cur.execute(
        sql.SQL("GRANT SELECT, INSERT, UPDATE ON organization_decision_stream_state TO {}").format(
            role_ident
        )
    )


def _assert_decision_inbox_runtime_privileges(cur: psycopg.Cursor, role: str) -> None:
    expected = {
        "public.organization_decision_event_inbox": (
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ),
        "public.organization_decision_stream_state": (
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ),
    }
    for relation, privileges_expected in expected.items():
        cur.execute(
            "SELECT has_table_privilege(%s,%s,'SELECT'), "
            "has_table_privilege(%s,%s,'INSERT'), "
            "has_table_privilege(%s,%s,'UPDATE'), "
            "has_table_privilege(%s,%s,'DELETE'), "
            "has_table_privilege(%s,%s,'TRUNCATE'), "
            "has_table_privilege(%s,%s,'REFERENCES'), "
            "has_table_privilege(%s,%s,'TRIGGER')",
            tuple(value for _ in range(7) for value in (role, relation)),
        )
        if cur.fetchone() != privileges_expected:
            raise SchemaControlError("organization decision inbox privileges are invalid")

    cur.execute(
        "SELECT r.rolsuper,r.rolbypassrls,"
        "pg_has_role(%s,'admin_role','MEMBER') "
        "FROM pg_roles r WHERE r.rolname=%s",
        (role, role),
    )
    posture = cur.fetchone()
    if posture is None or any(bool(value) for value in posture):
        raise SchemaControlError("organization decision inbox runtime role is elevated")


def _remediate_legacy_app_user(cur: psycopg.Cursor) -> None:
    """Disable the app_user role older installs created with a known password."""
    cur.execute("SELECT rolcanlogin FROM pg_roles WHERE rolname = 'app_user'")
    row = cur.fetchone()
    if row is None:
        return
    if row[0]:
        cur.execute("ALTER ROLE app_user NOLOGIN")
        print(
            "✓ Legacy app_user role disabled (NOLOGIN) — it was provisioned with a "
            "known default password. Use ACTIVEKG_RUNTIME_ROLE provisioning instead."
        )


def _assert_disposable_fresh_target(cur: psycopg.Cursor, environment: str) -> None:
    if environment == "production":
        raise SchemaControlError("fresh initialization is forbidden in production")
    cur.execute(
        """
        SELECT current_database(), current_user, host(inet_server_addr()),
               (SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname NOT IN ('pg_catalog','information_schema')
                  AND n.nspname NOT LIKE 'pg_toast%%' AND c.relkind IN ('r','p','v','m','S','f')),
               to_regnamespace(%s)
        """,
        (CONTROL_SCHEMA,),
    )
    database_name, role_name, server_host, relation_count, control_schema = cur.fetchone()
    if (
        not str(database_name).endswith("_test")
        or not str(role_name).endswith("_test")
        or server_host not in {None, "127.0.0.1", "::1"}
        or int(relation_count) != 0
        or control_schema is not None
    ):
        raise SchemaControlError(
            "fresh initialization requires an empty local *_test database and *_test role"
        )


def _assert_full_baseline(cur: psycopg.Cursor, migrations: tuple[str, ...]) -> None:
    for filename in migrations:
        ok, detail = _verify_baseline(cur, filename)
        if not ok:
            raise SchemaControlError(f"migration postcondition failed for {filename}: {detail}")


def _assert_runtime_role_catalog(cur: psycopg.Cursor, role: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role):
        raise SchemaControlError("ACTIVEKG_RUNTIME_ROLE is invalid")
    cur.execute(
        """
        SELECT r.rolcanlogin, r.rolsuper, r.rolcreatedb, r.rolcreaterole, r.rolbypassrls,
               has_schema_privilege(%s, 'public', 'USAGE'),
               has_schema_privilege(%s, 'public', 'CREATE'),
               has_table_privilege(%s, 'public.schema_migrations', 'SELECT'),
               (has_table_privilege(%s, 'public.schema_migrations', 'INSERT') OR
                has_table_privilege(%s, 'public.schema_migrations', 'UPDATE') OR
                has_table_privilege(%s, 'public.schema_migrations', 'DELETE') OR
                has_table_privilege(%s, 'public.schema_migrations', 'TRUNCATE')),
               EXISTS (
                 SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                 WHERE n.nspname = 'public' AND pg_get_userbyid(c.relowner) = %s
                   AND c.relkind IN ('r','p','v','m','S','f')
               )
        FROM pg_roles r WHERE r.rolname = %s
        """,
        (role, role, role, role, role, role, role, role, role),
    )
    row = cur.fetchone()
    if row is None or (
        not row[0]
        or any(bool(value) for value in row[1:5])
        or not row[5]
        or row[6]
        or not row[7]
        or row[8]
        or row[9]
    ):
        raise SchemaControlError("restricted Memory runtime role posture is invalid")


def _prepare_target(cur: psycopg.Cursor, target_id: str, environment: str, fresh: bool) -> None:
    if fresh:
        _assert_disposable_fresh_target(cur, environment)
        with cur.connection.transaction():
            create_control_schema(cur, target_id, environment)
        return
    assert_identity(cur, target_id, environment)


def main():
    attempt_id: int | None = None
    conn: psycopg.Connection | None = None
    try:
        control = resolve_control_environment()
        if os.getenv("ACTIVEKG_MIGRATION_APPLY") != "1":
            raise SchemaControlError("ACTIVEKG_MIGRATION_APPLY=1 is required")
        if control.environment == "production" and os.getenv("ACTIVEKG_ALLOW_MIGRATION_DRIFT"):
            raise SchemaControlError("ACTIVEKG_ALLOW_MIGRATION_DRIFT is forbidden in production")
        fresh = os.getenv("ACTIVEKG_SCHEMA_FRESH_INIT") == "1"

        print("Connecting to the fingerprinted Memory target...")
        conn = _connect_with_retry(control.dsn)
        with conn.cursor() as cur:
            # Target proof happens before manifest or migration-file reads.
            _prepare_target(cur, control.target_id, control.environment, fresh)

            migrations = _load_manifest()
            records = load_migration_records()
            if tuple(record.filename for record in records) != migrations:
                raise SchemaControlError("migration manifest/record order mismatch")
            digest = manifest_digest(records)

            cur.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
            try:
                assert_identity(cur, control.target_id, control.environment)
                if not fresh:
                    assert_ledger(read_ledger(cur), records, allow_prefix=True)
                attempt_id = start_attempt(cur, "migration", control.source_commit, digest)

                # Existing/adopted targets advance only through the append-only
                # migration manifest. Baseline and RLS assets are fresh-install
                # inputs and must never be replayed in production.
                if fresh:
                    _ensure_extensions_and_schema(cur)
                _apply_migrations(cur, migrations)
                if fresh:
                    _provision_runtime_role(cur)
                    _remediate_legacy_app_user(cur)
                    grant_control_read(
                        cur, os.getenv("ACTIVEKG_RUNTIME_ROLE", RUNTIME_ROLE_DEFAULT)
                    )
                runtime_role = os.getenv("ACTIVEKG_RUNTIME_ROLE", RUNTIME_ROLE_DEFAULT)
                _harden_candidate_privacy_runtime_privileges(cur, runtime_role)
                _harden_decision_inbox_runtime_privileges(cur, runtime_role)
                assert_ledger(read_ledger(cur), records, allow_prefix=False)
                _assert_full_baseline(cur, migrations)
                _assert_runtime_role_catalog(cur, runtime_role)
                _assert_candidate_privacy_runtime_privileges(cur, runtime_role)
                _assert_decision_inbox_runtime_privileges(cur, runtime_role)
                finish_attempt(cur, attempt_id, "success")
            except BaseException as exc:
                if attempt_id is not None:
                    try:
                        finish_attempt(cur, attempt_id, "failure", safe_error_class(exc))
                    except Exception:
                        pass
                raise
            finally:
                cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))

        print(
            "[Schema release] OK "
            f"(target={safe_target_fingerprint(control.target_id)}; migrations={len(records)})"
        )
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        print(f"[Schema release] REFUSED ({safe_error_class(exc)})", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
