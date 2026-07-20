-- Migration 017: Reserve the __quarantine__ sentinel tenant
--
-- Migration 016 parks legacy NULL-tenant rows under '__quarantine__'. Without
-- this migration, a JWT (or dev-mode request) carrying tenant_id
-- '__quarantine__' could read all quarantined PII through the ordinary tenant
-- policies. Recreate the tenant-isolation policies so the sentinel tenant is
-- never readable or writable via the tenant path — quarantined rows are
-- reachable only by admin_role (operator remediation). Idempotent.

DROP POLICY IF EXISTS tenant_isolation_candidates ON candidates;
CREATE POLICY tenant_isolation_candidates ON candidates
    FOR ALL
    TO PUBLIC
    USING (
        tenant_id <> '__quarantine__'
        AND tenant_id = current_setting('app.current_tenant_id', true)
    )
    WITH CHECK (
        tenant_id <> '__quarantine__'
        AND tenant_id = current_setting('app.current_tenant_id', true)
    );

DROP POLICY IF EXISTS tenant_isolation_candidate_identifiers ON candidate_identifiers;
CREATE POLICY tenant_isolation_candidate_identifiers ON candidate_identifiers
    FOR ALL
    TO PUBLIC
    USING (
        tenant_id <> '__quarantine__'
        AND tenant_id = current_setting('app.current_tenant_id', true)
    )
    WITH CHECK (
        tenant_id <> '__quarantine__'
        AND tenant_id = current_setting('app.current_tenant_id', true)
    );

DROP POLICY IF EXISTS tenant_isolation_candidate_source_records ON candidate_source_records;
CREATE POLICY tenant_isolation_candidate_source_records ON candidate_source_records
    FOR ALL
    TO PUBLIC
    USING (
        tenant_id <> '__quarantine__'
        AND tenant_id = current_setting('app.current_tenant_id', true)
    )
    WITH CHECK (
        tenant_id <> '__quarantine__'
        AND tenant_id = current_setting('app.current_tenant_id', true)
    );
