#!/usr/bin/env python3
"""Initialize/migrate the database on the deploy path.

Runs before the API starts (scripts/start_railway.sh). Responsibilities:

* base schema (db/init.sql) + extensions on a fresh database
* RLS policies (enable_rls_policies.sql — contains no login roles)
* ordered migrations from activekg.common.migration_manifest, tracked in a
  ``schema_migrations`` ledger under a Postgres advisory lock
* optional provisioning of the restricted runtime role from env vars

DSNs:
    ACTIVEKG_MIGRATE_DSN   privileged/owner role used ONLY here (preferred)
    ACTIVEKG_DSN           runtime DSN; used as a fallback with a warning so
                           single-DSN dev environments keep working
    DATABASE_URL           last-resort fallback (Railway convention)

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

import psycopg
from psycopg import sql

MAX_RETRIES = 10
RETRY_DELAY = 3  # seconds
ADVISORY_LOCK_KEY = 0x41435447  # 'ACTG'

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

# Migrations that were edited before the immutability rule took effect.
# Maps filename -> {old recorded checksum: new expected checksum}; a mismatch
# matching one of these pairs is upgraded in place instead of failing boot.
CHECKSUM_TRANSITIONS: dict[str, dict[str, str]] = {
    "016_candidate_rls.sql": {
        # PR #11 preflight -> PR #12 effective-tenant preflight rewrite
        "34f02ce7137003697e1a3e0a675883b5203d55150ea1a0c258892308ae344b21": (
            "2294ef74ce9436782dc5f3c1484939bb53edec69e963233f5ee705a3849d6a63"
        ),
    },
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
        except (psycopg.OperationalError, psycopg.errors.ConnectionTimeout) as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"  Connection attempt {attempt}/{MAX_RETRIES} failed: {e}")
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
                if os.getenv("ACTIVEKG_ALLOW_MIGRATION_DRIFT", "false").lower() == "true":
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
                        "or set ACTIVEKG_ALLOW_MIGRATION_DRIFT=true to override."
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
            print(f"ERROR: migration {migration_file} failed: {e}")
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


def main():
    migrate_dsn = os.environ.get("ACTIVEKG_MIGRATE_DSN")
    if not migrate_dsn:
        migrate_dsn = os.environ.get("ACTIVEKG_DSN") or os.environ.get("DATABASE_URL")
        if migrate_dsn:
            print(
                "WARNING: ACTIVEKG_MIGRATE_DSN not set — falling back to the runtime DSN. "
                "Migrations should run as a separate privileged role; the runtime role "
                "must stay non-owner so RLS applies to it."
            )
    if not migrate_dsn:
        print("ERROR: ACTIVEKG_MIGRATE_DSN (or ACTIVEKG_DSN/DATABASE_URL) not set")
        sys.exit(1)

    migrations = _load_manifest()
    print("Connecting to database...")

    try:
        with _connect_with_retry(migrate_dsn) as conn:
            with conn.cursor() as cur:
                # One migrator at a time; concurrent replicas wait here.
                cur.execute("SELECT pg_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
                try:
                    _ensure_extensions_and_schema(cur)
                    _apply_migrations(cur, migrations)
                    _provision_runtime_role(cur)
                    _remediate_legacy_app_user(cur)
                finally:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
        print("\n✅ Database initialization complete!")
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: Database initialization failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
