-- Migration 018: tenant_id must be non-blank on candidate tables
--
-- The repository installs an empty-string GUC for calls without tenant
-- context, relying on '' never being a real tenant. Make that a database
-- guarantee: a blank (or whitespace-only) tenant_id can never be stored, so
-- no row can become visible to a context-less session. Idempotent.

ALTER TABLE candidates DROP CONSTRAINT IF EXISTS candidates_tenant_nonblank;
ALTER TABLE candidates
    ADD CONSTRAINT candidates_tenant_nonblank CHECK (btrim(tenant_id) <> '');

ALTER TABLE candidate_identifiers DROP CONSTRAINT IF EXISTS candidate_identifiers_tenant_nonblank;
ALTER TABLE candidate_identifiers
    ADD CONSTRAINT candidate_identifiers_tenant_nonblank CHECK (btrim(tenant_id) <> '');

ALTER TABLE candidate_source_records DROP CONSTRAINT IF EXISTS candidate_source_records_tenant_nonblank;
ALTER TABLE candidate_source_records
    ADD CONSTRAINT candidate_source_records_tenant_nonblank CHECK (btrim(tenant_id) <> '');
