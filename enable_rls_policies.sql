-- Row-Level Security (RLS) for Multi-Tenant Isolation
-- Run this after init.sql to enable tenant isolation

-- ============================================================================
-- CREATE ROLES FIRST (moved from bottom)
-- ============================================================================

-- Create admin role (if not exists)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'admin_role') THEN
        CREATE ROLE admin_role;
    END IF;
END$$;

-- Create app_user role for regular API access
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'app_user') THEN
        CREATE ROLE app_user LOGIN PASSWORD 'change_me_in_production';
    END IF;
END$$;

-- Grant usage to app_user
GRANT USAGE ON SCHEMA public TO app_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO app_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- ============================================================================
-- NODES TABLE
-- ============================================================================

ALTER TABLE nodes ENABLE ROW LEVEL SECURITY;

-- Policy: Tenants can only see their own nodes
DROP POLICY IF EXISTS tenant_isolation_nodes ON nodes;
CREATE POLICY tenant_isolation_nodes ON nodes
    FOR ALL
    TO PUBLIC
    USING (
        tenant_id IS NULL
        OR tenant_id = current_setting('app.current_tenant_id', true)
    );

-- Policy: Super admins can see all nodes
DROP POLICY IF EXISTS admin_all_nodes ON nodes;
CREATE POLICY admin_all_nodes ON nodes
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- EDGES TABLE
-- ============================================================================

ALTER TABLE edges ENABLE ROW LEVEL SECURITY;

-- Policy: Tenants can only see edges within their tenant
-- Note: For cross-tenant edges, both src and dst nodes must be accessible
DROP POLICY IF EXISTS tenant_isolation_edges ON edges;
CREATE POLICY tenant_isolation_edges ON edges
    FOR ALL
    TO PUBLIC
    USING (
        tenant_id IS NULL
        OR tenant_id = current_setting('app.current_tenant_id', true)
    );

-- Policy: Super admins can see all edges
DROP POLICY IF EXISTS admin_all_edges ON edges;
CREATE POLICY admin_all_edges ON edges
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- EVENTS TABLE
-- ============================================================================

ALTER TABLE events ENABLE ROW LEVEL SECURITY;

-- Policy: Tenants can only see their own events
DROP POLICY IF EXISTS tenant_isolation_events ON events;
CREATE POLICY tenant_isolation_events ON events
    FOR ALL
    TO PUBLIC
    USING (
        tenant_id IS NULL
        OR tenant_id = current_setting('app.current_tenant_id', true)
    );

-- Policy: Super admins can see all events
DROP POLICY IF EXISTS admin_all_events ON events;
CREATE POLICY admin_all_events ON events
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- NODE_VERSIONS TABLE (optional - for audit)
-- ============================================================================

ALTER TABLE node_versions ENABLE ROW LEVEL SECURITY;

-- Policy: Tenants can only see versions of their own nodes
-- This requires joining with nodes table in the policy
DROP POLICY IF EXISTS tenant_isolation_versions ON node_versions;
CREATE POLICY tenant_isolation_versions ON node_versions
    FOR ALL
    TO PUBLIC
    USING (
        EXISTS (
            SELECT 1 FROM nodes
            WHERE nodes.id = node_versions.node_id
            AND (nodes.tenant_id IS NULL OR nodes.tenant_id = current_setting('app.current_tenant_id', true))
        )
    );

-- Policy: Super admins can see all versions
DROP POLICY IF EXISTS admin_all_versions ON node_versions;
CREATE POLICY admin_all_versions ON node_versions
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- PATTERNS TABLE (shared across tenants by default)
-- ============================================================================

-- Patterns are typically shared, but you can add tenant_id if needed
-- ALTER TABLE patterns ADD COLUMN tenant_id TEXT;
-- Then add RLS policies similar to above

-- ============================================================================
-- EMBEDDING_HISTORY TABLE (optional - for audit)
-- ============================================================================

ALTER TABLE embedding_history ENABLE ROW LEVEL SECURITY;

-- Policy: Tenants can only see history of their own nodes
DROP POLICY IF EXISTS tenant_isolation_history ON embedding_history;
CREATE POLICY tenant_isolation_history ON embedding_history
    FOR ALL
    TO PUBLIC
    USING (
        EXISTS (
            SELECT 1 FROM nodes
            WHERE nodes.id = embedding_history.node_id
            AND (nodes.tenant_id IS NULL OR nodes.tenant_id = current_setting('app.current_tenant_id', true))
        )
    );

-- Policy: Super admins can see all history
DROP POLICY IF EXISTS admin_all_history ON embedding_history;
CREATE POLICY admin_all_history ON embedding_history
    FOR ALL
    TO admin_role
    USING (true)
    WITH CHECK (true);

-- ============================================================================
-- ROLE GRANTS (roles created at top of file)
-- ============================================================================

-- Grant admin role to specific user (replace 'admin_user' with actual username)
-- GRANT admin_role TO admin_user;

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to set tenant context (call this at session start)
CREATE OR REPLACE FUNCTION set_tenant_context(p_tenant_id TEXT)
RETURNS void AS $$
BEGIN
    PERFORM set_config('app.current_tenant_id', p_tenant_id, false);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get current tenant
CREATE OR REPLACE FUNCTION get_current_tenant()
RETURNS TEXT AS $$
BEGIN
    RETURN current_setting('app.current_tenant_id', true);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- TESTING
-- ============================================================================

-- Test tenant isolation
-- 1. Set tenant context
-- SELECT set_tenant_context('tenant_a');

-- 2. Insert test node
-- INSERT INTO nodes (tenant_id, classes, props) VALUES ('tenant_a', ARRAY['Test'], '{}');

-- 3. Verify you can see it
-- SELECT * FROM nodes WHERE tenant_id = 'tenant_a';

-- 4. Switch tenant
-- SELECT set_tenant_context('tenant_b');

-- 5. Verify you CANNOT see tenant_a's node
-- SELECT * FROM nodes WHERE tenant_id = 'tenant_a';  -- Should return empty

-- 6. Reset (for admin access)
-- RESET app.current_tenant_id;

-- ============================================================================
-- NOTES
-- ============================================================================

-- 1. Performance: RLS policies may slow down queries. Ensure indexes on tenant_id exist.
-- 2. NULL tenant_id: Nodes with NULL tenant_id are visible to all (use for shared data).
-- 3. Admin bypass: Create admin_role and grant to users who need full access.
-- 4. Application layer: Always call set_tenant_context() at session start based on JWT/auth.
-- 5. Connection pooling: Use SET LOCAL in transactions for pooled connections:
--    BEGIN;
--    SET LOCAL app.current_tenant_id = 'tenant_x';
--    -- ... queries ...
--    COMMIT;
