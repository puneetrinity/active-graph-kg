"""Single source of truth for the ordered migration manifest.

Consumed by scripts/init_railway_db.py (applies migrations and records them in
the schema_migrations ledger) and by the /readyz endpoint (verifies the ledger
is complete before reporting ready). Keep this list append-only and in apply
order. Must stay import-light: no activekg imports, no third-party deps.
"""

MIGRATIONS: tuple[str, ...] = (
    "001_add_embedding_history_index.sql",
    "004_add_external_id_index.sql",
    "add_text_search.sql",
    "005_connector_configs_table.sql",
    "006_add_key_version.sql",
    "007_add_provider_check.sql",
    "008_connector_cursors_table.sql",
    "009_embedding_queue_status.sql",
    "010_update_text_search_vector.sql",
    "011_unique_tenant_external_id.sql",
    "012_global_memory.sql",
    "012_candidate_identity.sql",
    "013_vantahire_provenance.sql",
    "014_signal_job_tags.sql",
    "015_candidate_profile.sql",
    "016_candidate_rls.sql",
    "017_reserve_quarantine_tenant.sql",
    "018_tenant_nonblank.sql",
    "019_global_reconciliation.sql",
    "020_embed_version.sql",
)
