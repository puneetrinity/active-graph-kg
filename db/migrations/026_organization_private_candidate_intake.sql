-- Migration 026: organization-private application candidate reference.
--
-- One opaque tenant candidate is created for a newly accepted Flow public
-- application. Applicant identifiers are privacy-admission inputs only and
-- never enter these relations. No existing row is relabelled or backfilled.

ALTER TABLE candidates DROP CONSTRAINT candidates_scope_check;
ALTER TABLE candidates ADD CONSTRAINT candidates_scope_check
    CHECK (scope IN ('shared', 'organization_private'));

CREATE TABLE organization_candidate_references (
    reference_id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    candidate_id UUID NOT NULL,
    application_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    origin_code TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    verified_issuer TEXT NOT NULL,
    verified_actor_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT organization_candidate_references_tenant_nonblank CHECK (
        btrim(tenant_id) <> '' AND tenant_id <> '__quarantine__'
    ),
    CONSTRAINT organization_candidate_references_application_positive CHECK (application_id > 0),
    CONSTRAINT organization_candidate_references_job_positive CHECK (job_id > 0),
    CONSTRAINT organization_candidate_references_origin_v1 CHECK (
        origin_code = 'candidate_applied' AND schema_version = 1
    ),
    CONSTRAINT organization_candidate_references_authority CHECK (
        verified_issuer = 'vantahire' AND verified_actor_id = 'vantahire-backend'
    ),
    CONSTRAINT organization_candidate_references_tenant_candidate_fkey
        FOREIGN KEY (tenant_id, candidate_id)
        REFERENCES candidates (tenant_id, candidate_id) ON DELETE RESTRICT,
    CONSTRAINT organization_candidate_references_tenant_application_unique
        UNIQUE (tenant_id, application_id),
    CONSTRAINT organization_candidate_references_tenant_reference_unique
        UNIQUE (tenant_id, reference_id),
    CONSTRAINT organization_candidate_references_tenant_reference_candidate_unique
        UNIQUE (tenant_id, reference_id, candidate_id)
);

CREATE INDEX organization_candidate_references_candidate_idx
    ON organization_candidate_references (tenant_id, candidate_id);
CREATE INDEX organization_candidate_references_job_idx
    ON organization_candidate_references (tenant_id, job_id, created_at DESC);

CREATE TABLE organization_candidate_resume_evidence (
    resume_version_id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    reference_id UUID NOT NULL,
    candidate_id UUID NOT NULL,
    version INTEGER NOT NULL,
    source_kind TEXT NOT NULL,
    source_resume_id INTEGER,
    source_observed_at TIMESTAMPTZ NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    byte_count INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    extracted_text_sha256 CHAR(64),
    captured_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT organization_candidate_resume_evidence_tenant_nonblank CHECK (
        btrim(tenant_id) <> '' AND tenant_id <> '__quarantine__'
    ),
    CONSTRAINT organization_candidate_resume_evidence_version_v1 CHECK (version = 1),
    CONSTRAINT organization_candidate_resume_evidence_source_kind CHECK (
        (source_kind = 'direct_upload' AND source_resume_id IS NULL)
        OR (source_kind = 'saved_resume' AND source_resume_id > 0)
    ),
    CONSTRAINT organization_candidate_resume_evidence_content_digest CHECK (
        content_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT organization_candidate_resume_evidence_byte_count CHECK (
        byte_count BETWEEN 1 AND 5242880
    ),
    CONSTRAINT organization_candidate_resume_evidence_media_type CHECK (
        media_type IN (
            'application/pdf',
            'application/msword',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    ),
    CONSTRAINT organization_candidate_resume_evidence_extracted_digest CHECK (
        extracted_text_sha256 IS NULL OR extracted_text_sha256 ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT organization_candidate_resume_evidence_reference_fkey
        FOREIGN KEY (tenant_id, reference_id, candidate_id)
        REFERENCES organization_candidate_references (tenant_id, reference_id, candidate_id)
        ON DELETE RESTRICT,
    CONSTRAINT organization_candidate_resume_evidence_tenant_candidate_fkey
        FOREIGN KEY (tenant_id, candidate_id)
        REFERENCES candidates (tenant_id, candidate_id) ON DELETE RESTRICT,
    CONSTRAINT organization_candidate_resume_evidence_reference_unique UNIQUE (reference_id),
    CONSTRAINT organization_candidate_resume_evidence_tenant_version_candidate_unique
        UNIQUE (tenant_id, resume_version_id, candidate_id),
    CONSTRAINT organization_candidate_resume_evidence_tenant_version_unique
        UNIQUE (tenant_id, reference_id, version)
);

CREATE INDEX organization_candidate_resume_evidence_candidate_idx
    ON organization_candidate_resume_evidence (tenant_id, candidate_id);

CREATE TABLE organization_candidate_ingest_receipts (
    idempotency_key CHAR(64) PRIMARY KEY,
    input_digest CHAR(64) NOT NULL,
    tenant_id TEXT NOT NULL,
    reference_id UUID NOT NULL,
    resume_version_id UUID NOT NULL,
    candidate_id UUID NOT NULL,
    resolution TEXT NOT NULL,
    verified_issuer TEXT NOT NULL,
    verified_actor_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT organization_candidate_ingest_receipts_idempotency CHECK (
        idempotency_key ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT organization_candidate_ingest_receipts_input_digest CHECK (
        input_digest ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT organization_candidate_ingest_receipts_tenant_nonblank CHECK (
        btrim(tenant_id) <> '' AND tenant_id <> '__quarantine__'
    ),
    CONSTRAINT organization_candidate_ingest_receipts_resolution CHECK (resolution = 'created'),
    CONSTRAINT organization_candidate_ingest_receipts_authority CHECK (
        verified_issuer = 'vantahire' AND verified_actor_id = 'vantahire-backend'
    ),
    CONSTRAINT organization_candidate_ingest_receipts_reference_fkey
        FOREIGN KEY (tenant_id, reference_id, candidate_id)
        REFERENCES organization_candidate_references (tenant_id, reference_id, candidate_id)
        ON DELETE RESTRICT,
    CONSTRAINT organization_candidate_ingest_receipts_resume_fkey
        FOREIGN KEY (tenant_id, resume_version_id, candidate_id)
        REFERENCES organization_candidate_resume_evidence (tenant_id, resume_version_id, candidate_id)
        ON DELETE RESTRICT,
    CONSTRAINT organization_candidate_ingest_receipts_tenant_candidate_fkey
        FOREIGN KEY (tenant_id, candidate_id)
        REFERENCES candidates (tenant_id, candidate_id) ON DELETE RESTRICT,
    CONSTRAINT organization_candidate_ingest_receipts_reference_unique UNIQUE (reference_id),
    CONSTRAINT organization_candidate_ingest_receipts_resume_unique UNIQUE (resume_version_id)
);

CREATE INDEX organization_candidate_ingest_receipts_candidate_idx
    ON organization_candidate_ingest_receipts (tenant_id, candidate_id);

CREATE FUNCTION organization_candidate_evidence_append_only()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    has_evidence BOOLEAN;
BEGIN
    IF TG_OP <> 'TRUNCATE' THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = TG_TABLE_NAME || ' is append-only (attempted ' || TG_OP || ')';
    END IF;

    EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I LIMIT 1)', TG_TABLE_NAME)
        INTO has_evidence;
    IF has_evidence THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = TG_TABLE_NAME || ' contains committed evidence and cannot be truncated';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER organization_candidate_references_no_mutation
    BEFORE UPDATE OR DELETE ON organization_candidate_references
    FOR EACH ROW EXECUTE FUNCTION organization_candidate_evidence_append_only();
CREATE TRIGGER organization_candidate_references_no_nonempty_truncate
    BEFORE TRUNCATE ON organization_candidate_references
    FOR EACH STATEMENT EXECUTE FUNCTION organization_candidate_evidence_append_only();

CREATE TRIGGER organization_candidate_resume_evidence_no_mutation
    BEFORE UPDATE OR DELETE ON organization_candidate_resume_evidence
    FOR EACH ROW EXECUTE FUNCTION organization_candidate_evidence_append_only();
CREATE TRIGGER organization_candidate_resume_evidence_no_nonempty_truncate
    BEFORE TRUNCATE ON organization_candidate_resume_evidence
    FOR EACH STATEMENT EXECUTE FUNCTION organization_candidate_evidence_append_only();

CREATE TRIGGER organization_candidate_ingest_receipts_no_mutation
    BEFORE UPDATE OR DELETE ON organization_candidate_ingest_receipts
    FOR EACH ROW EXECUTE FUNCTION organization_candidate_evidence_append_only();
CREATE TRIGGER organization_candidate_ingest_receipts_no_nonempty_truncate
    BEFORE TRUNCATE ON organization_candidate_ingest_receipts
    FOR EACH STATEMENT EXECUTE FUNCTION organization_candidate_evidence_append_only();

ALTER TABLE organization_candidate_references ENABLE ROW LEVEL SECURITY;
ALTER TABLE organization_candidate_references FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_organization_candidate_references
    ON organization_candidate_references FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true));
CREATE POLICY admin_all_organization_candidate_references
    ON organization_candidate_references FOR ALL TO admin_role USING (true) WITH CHECK (true);

ALTER TABLE organization_candidate_resume_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE organization_candidate_resume_evidence FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_organization_candidate_resume_evidence
    ON organization_candidate_resume_evidence FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true));
CREATE POLICY admin_all_organization_candidate_resume_evidence
    ON organization_candidate_resume_evidence FOR ALL TO admin_role USING (true) WITH CHECK (true);

ALTER TABLE organization_candidate_ingest_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE organization_candidate_ingest_receipts FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_organization_candidate_ingest_receipts
    ON organization_candidate_ingest_receipts FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true));
CREATE POLICY admin_all_organization_candidate_ingest_receipts
    ON organization_candidate_ingest_receipts FOR ALL TO admin_role USING (true) WITH CHECK (true);

REVOKE ALL ON organization_candidate_references FROM PUBLIC;
REVOKE ALL ON organization_candidate_resume_evidence FROM PUBLIC;
REVOKE ALL ON organization_candidate_ingest_receipts FROM PUBLIC;
REVOKE ALL ON FUNCTION organization_candidate_evidence_append_only() FROM PUBLIC;

COMMENT ON TABLE organization_candidate_references IS
    'Append-only PII-free organization-private application reference; one opaque candidate per application.';
COMMENT ON TABLE organization_candidate_resume_evidence IS
    'Append-only resume digests and metadata; raw locator, filename, bytes, and extracted text stay in Flow.';
COMMENT ON TABLE organization_candidate_ingest_receipts IS
    'Append-only idempotent result for organization-private candidate intake.';
