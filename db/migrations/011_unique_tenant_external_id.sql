-- Migration 011: Add unique index for external_id per tenant
-- Ensures idempotent node creation by external_id within a tenant
--
-- IMPORTANT: CREATE UNIQUE INDEX CONCURRENTLY cannot run inside a transaction.
-- If your migration runner wraps migrations in a transaction, run step 2 manually
-- outside of the runner (e.g. via psql or a deploy script).
--
-- Step 1: Check for duplicate external_id values that would violate the unique constraint.
--         Run this BEFORE creating the index. If rows are returned, deduplicate first.
--
-- Step 2: Create the index CONCURRENTLY to avoid locking writes.
--         Must be run outside a transaction block.

-- Step 1: Duplicate check (safe to run inside a transaction)
-- Review output before proceeding. If any rows are returned, resolve duplicates first.
SELECT tenant_id, props->>'external_id' AS external_id, COUNT(*) AS cnt
FROM nodes
WHERE props ? 'external_id'
GROUP BY tenant_id, props->>'external_id'
HAVING COUNT(*) > 1
ORDER BY cnt DESC;

-- Step 2: Create unique index CONCURRENTLY (run outside transaction)
-- If your migration runner auto-wraps in a transaction, execute this manually:
--   psql $DATABASE_URL -c "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS ..."
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_nodes_tenant_external_id_unique
ON nodes (tenant_id, (props->>'external_id'))
WHERE props ? 'external_id';
