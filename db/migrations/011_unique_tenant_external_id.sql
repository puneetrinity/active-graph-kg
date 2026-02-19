-- Migration 011: Add unique index for external_id per tenant
-- Ensures idempotent node creation by external_id within a tenant
--
-- Uses plain CREATE INDEX (not CONCURRENTLY) so it can run inside a transaction.
-- For large production tables under load, run CONCURRENTLY via psql instead:
--   psql $DATABASE_URL -c "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_nodes_tenant_external_id_unique ON nodes (tenant_id, (props->>'external_id')) WHERE props ? 'external_id';"

CREATE UNIQUE INDEX IF NOT EXISTS idx_nodes_tenant_external_id_unique
ON nodes (tenant_id, (props->>'external_id'))
WHERE props ? 'external_id';
