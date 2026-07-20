-- Migration 016: Tenant hardening + Row-Level Security for candidate tables
--
-- The candidate identity layer (migrations 012-015) launched with nullable
-- tenant_id columns, plain candidate_id foreign keys, and no RLS. This
-- migration:
--   1. Backfills child tenant_id from the parent candidate, quarantines any
--      remaining NULL-tenant rows, and makes tenant_id NOT NULL everywhere.
--   2. Replaces the candidate_id FKs with (tenant_id, candidate_id) composite
--      FKs so a child row can never reference a parent in another tenant.
--   3. Enables RLS with explicit-equality USING and WITH CHECK policies
--      (no NULL-tenant escape hatch) plus an admin_role bypass, matching the
--      session-GUC pattern of enable_rls_policies.sql.
-- Idempotent: safe to re-run.

-- admin_role normally exists via enable_rls_policies.sql; guard for
-- databases where this migration runs standalone.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'admin_role') THEN
        CREATE ROLE admin_role;
    END IF;
END$$;

-- ============================================================================
-- 0. PREFLIGHT — refuse to run against data the backfill could corrupt
-- ============================================================================
-- Collapsing NULL-tenant rows into one sentinel tenant can violate the
-- unique merge keys or silently merge distinct identities. Fail loudly with
-- an actionable message instead; an operator must resolve these rows first.

-- The backfill assigns every row an EFFECTIVE tenant:
--   candidates:  COALESCE(tenant_id, '__quarantine__')
--   children:    COALESCE(own tenant_id, parent tenant_id, '__quarantine__')
-- All uniqueness and FK validation below is computed against that effective
-- tenant, so rows that end up in *different* tenants never trip the check,
-- while every collision the backfill would actually create does.
DO $$
DECLARE n int;
BEGIN
    -- Identifier merge keys must stay unique under the effective tenant.
    WITH eff AS (
        SELECT COALESCE(ci.tenant_id, c.tenant_id, '__quarantine__') AS eff_tenant,
               ci.identifier_type,
               ci.value_normalized
        FROM candidate_identifiers ci
        JOIN candidates c USING (candidate_id)
    )
    SELECT count(*) INTO n FROM (
        SELECT eff_tenant, identifier_type, value_normalized
        FROM eff
        GROUP BY eff_tenant, identifier_type, value_normalized
        HAVING count(*) > 1
    ) d;
    IF n > 0 THEN
        RAISE EXCEPTION 'migration 016 preflight: % identifier merge-key group(s) would collide after the tenant backfill (including collisions with existing rows in the target tenant); resolve manually first', n;
    END IF;

    -- Source-record keys must stay unique under the effective tenant.
    WITH eff AS (
        SELECT COALESCE(csr.tenant_id, c.tenant_id, '__quarantine__') AS eff_tenant,
               csr.source,
               csr.source_record_type,
               csr.source_record_id
        FROM candidate_source_records csr
        JOIN candidates c USING (candidate_id)
    )
    SELECT count(*) INTO n FROM (
        SELECT eff_tenant, source, source_record_type, source_record_id
        FROM eff
        GROUP BY eff_tenant, source, source_record_type, source_record_id
        HAVING count(*) > 1
    ) d;
    IF n > 0 THEN
        RAISE EXCEPTION 'migration 016 preflight: % source-record key group(s) would collide after the tenant backfill; resolve manually first', n;
    END IF;

    -- Any child whose effective tenant differs from its parent's effective
    -- tenant would violate the composite FK (covers non-NULL child with NULL
    -- parent, and plain cross-tenant references); surface them clearly.
    SELECT count(*) INTO n
    FROM candidate_identifiers ci
    JOIN candidates c USING (candidate_id)
    WHERE COALESCE(ci.tenant_id, c.tenant_id, '__quarantine__')
          <> COALESCE(c.tenant_id, '__quarantine__');
    IF n > 0 THEN
        RAISE EXCEPTION 'migration 016 preflight: % candidate_identifiers row(s) would end up in a different tenant than their parent candidate; resolve manually first', n;
    END IF;

    SELECT count(*) INTO n
    FROM candidate_source_records csr
    JOIN candidates c USING (candidate_id)
    WHERE COALESCE(csr.tenant_id, c.tenant_id, '__quarantine__')
          <> COALESCE(c.tenant_id, '__quarantine__');
    IF n > 0 THEN
        RAISE EXCEPTION 'migration 016 preflight: % candidate_source_records row(s) would end up in a different tenant than their parent candidate; resolve manually first', n;
    END IF;
END$$;

-- ============================================================================
-- 1. TENANT BACKFILL + NOT NULL
-- ============================================================================

-- Children inherit the parent's tenant where the parent has one.
UPDATE candidate_identifiers ci
SET tenant_id = c.tenant_id
FROM candidates c
WHERE ci.candidate_id = c.candidate_id
  AND ci.tenant_id IS NULL
  AND c.tenant_id IS NOT NULL;

UPDATE candidate_source_records csr
SET tenant_id = c.tenant_id
FROM candidates c
WHERE csr.candidate_id = c.candidate_id
  AND csr.tenant_id IS NULL
  AND c.tenant_id IS NOT NULL;

-- Anything still NULL is quarantined under a sentinel tenant no runtime
-- session ever uses; operators can inspect and reassign later.
UPDATE candidates SET tenant_id = '__quarantine__' WHERE tenant_id IS NULL;
UPDATE candidate_identifiers SET tenant_id = '__quarantine__' WHERE tenant_id IS NULL;
UPDATE candidate_source_records SET tenant_id = '__quarantine__' WHERE tenant_id IS NULL;

ALTER TABLE candidates ALTER COLUMN tenant_id SET NOT NULL;
ALTER TABLE candidate_identifiers ALTER COLUMN tenant_id SET NOT NULL;
ALTER TABLE candidate_source_records ALTER COLUMN tenant_id SET NOT NULL;

-- ============================================================================
-- 2. COMPOSITE (tenant_id, candidate_id) FOREIGN KEYS
-- ============================================================================

-- Parent key for the composite FKs.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT FROM pg_constraint WHERE conname = 'candidates_tenant_candidate_uniq'
    ) THEN
        ALTER TABLE candidates
            ADD CONSTRAINT candidates_tenant_candidate_uniq UNIQUE (tenant_id, candidate_id);
    END IF;
END$$;

DO $$
BEGIN
    IF EXISTS (
        SELECT FROM pg_constraint WHERE conname = 'candidate_identifiers_candidate_id_fkey'
    ) THEN
        ALTER TABLE candidate_identifiers
            DROP CONSTRAINT candidate_identifiers_candidate_id_fkey;
    END IF;
    IF NOT EXISTS (
        SELECT FROM pg_constraint WHERE conname = 'candidate_identifiers_tenant_candidate_fkey'
    ) THEN
        ALTER TABLE candidate_identifiers
            ADD CONSTRAINT candidate_identifiers_tenant_candidate_fkey
            FOREIGN KEY (tenant_id, candidate_id)
            REFERENCES candidates (tenant_id, candidate_id)
            ON DELETE CASCADE;
    END IF;
END$$;

DO $$
BEGIN
    IF EXISTS (
        SELECT FROM pg_constraint WHERE conname = 'candidate_source_records_candidate_id_fkey'
    ) THEN
        ALTER TABLE candidate_source_records
            DROP CONSTRAINT candidate_source_records_candidate_id_fkey;
    END IF;
    IF NOT EXISTS (
        SELECT FROM pg_constraint WHERE conname = 'candidate_source_records_tenant_candidate_fkey'
    ) THEN
        ALTER TABLE candidate_source_records
            ADD CONSTRAINT candidate_source_records_tenant_candidate_fkey
            FOREIGN KEY (tenant_id, candidate_id)
            REFERENCES candidates (tenant_id, candidate_id)
            ON DELETE CASCADE;
    END IF;
END$$;

-- ============================================================================
-- 3. ROW-LEVEL SECURITY — explicit equality, USING + WITH CHECK
-- ============================================================================
-- current_setting(..., true) yields NULL when the GUC is unset, and
-- NULL = tenant_id is never true: a session without tenant context sees and
-- writes nothing. There is deliberately no NULL-tenant escape hatch.

ALTER TABLE candidates ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_candidates ON candidates;
CREATE POLICY tenant_isolation_candidates ON candidates
    FOR ALL
    TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true));

DROP POLICY IF EXISTS admin_all_candidates ON candidates;
CREATE POLICY admin_all_candidates ON candidates
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

ALTER TABLE candidate_identifiers ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_candidate_identifiers ON candidate_identifiers;
CREATE POLICY tenant_isolation_candidate_identifiers ON candidate_identifiers
    FOR ALL
    TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true));

DROP POLICY IF EXISTS admin_all_candidate_identifiers ON candidate_identifiers;
CREATE POLICY admin_all_candidate_identifiers ON candidate_identifiers
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

ALTER TABLE candidate_source_records ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_candidate_source_records ON candidate_source_records;
CREATE POLICY tenant_isolation_candidate_source_records ON candidate_source_records
    FOR ALL
    TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true));

DROP POLICY IF EXISTS admin_all_candidate_source_records ON candidate_source_records;
CREATE POLICY admin_all_candidate_source_records ON candidate_source_records
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);
