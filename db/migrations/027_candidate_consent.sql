-- Candidate-approved source authority only. No application/provider backfill or indexing.
CREATE TABLE candidate_consent_state (
    subject_id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL UNIQUE CHECK (tenant_id = 'candidate_' || subject_id::text),
    global_candidate_id UUID REFERENCES global_candidates(id) ON DELETE RESTRICT,
    highest_version BIGINT NOT NULL DEFAULT 0 CHECK (highest_version BETWEEN 0 AND 9007199254740991),
    last_action TEXT CHECK (last_action IN ('grant','withdraw')),
    effective_version BIGINT NOT NULL DEFAULT 0 CHECK (effective_version BETWEEN 0 AND highest_version),
    effective_action TEXT CHECK (effective_action IN ('grant','withdraw')),
    active_source_id UUID,
    effective_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (tenant_id,subject_id),
    CHECK ((highest_version=0 AND last_action IS NULL) OR (highest_version>0 AND last_action IS NOT NULL)),
    CHECK (((effective_version=0 AND effective_action IS NULL AND effective_at IS NULL AND active_source_id IS NULL)
      OR (effective_version>0 AND effective_action='withdraw' AND effective_at IS NOT NULL AND active_source_id IS NULL)
      OR (effective_version>0 AND effective_action='grant' AND effective_at IS NOT NULL
        AND active_source_id IS NOT NULL AND global_candidate_id IS NOT NULL)) IS TRUE)
);
CREATE TABLE candidate_consent_sources (
    source_id UUID PRIMARY KEY,
    subject_id UUID NOT NULL,
    tenant_id TEXT NOT NULL CHECK (tenant_id='candidate_' || subject_id::text),
    source_version BIGINT NOT NULL CHECK (source_version BETWEEN 1 AND 9007199254740991),
    global_candidate_id UUID NOT NULL REFERENCES global_candidates(id) ON DELETE RESTRICT,
    profile JSONB NOT NULL,
    profile_sha256 CHAR(64) NOT NULL CHECK (profile_sha256 ~ '^[0-9a-f]{64}$'),
    resume JSONB,
    resume_sha256 CHAR(64) CHECK (resume_sha256 ~ '^[0-9a-f]{64}$'),
    purpose TEXT NOT NULL CHECK (purpose='platform_professional_matching'),
    purpose_version INTEGER NOT NULL CHECK (purpose_version=1),
    copy_version INTEGER NOT NULL CHECK (copy_version=1),
    copy_sha256 CHAR(64) NOT NULL CHECK (copy_sha256 ~ '^[0-9a-f]{64}$'),
    approved_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (tenant_id,subject_id,source_id),
    UNIQUE (tenant_id,subject_id,source_version),
    FOREIGN KEY(tenant_id,subject_id) REFERENCES candidate_consent_state(tenant_id,subject_id) ON DELETE RESTRICT,
    CONSTRAINT consent_source_profile_shape CHECK ((jsonb_typeof(profile)='object' AND octet_length(profile::text)<=32768
      AND profile ?& ARRAY['display_name','headline','location','skills','linkedin']
      AND profile - ARRAY['display_name','headline','location','skills','linkedin']='{}'::jsonb
      AND jsonb_typeof(profile->'display_name')='string' AND length(profile->>'display_name') BETWEEN 1 AND 200
      AND jsonb_typeof(profile->'headline')='string' AND length(profile->>'headline')<=300
      AND jsonb_typeof(profile->'location')='string' AND length(profile->>'location')<=200
      AND jsonb_typeof(profile->'skills')='array' AND jsonb_array_length(profile->'skills')<=100
      AND NOT jsonb_path_exists(profile, 'strict $.skills[*] ? (@.type() != "string" || !(@ like_regex "^.{1,100}$") || @ like_regex "^[[:space:]]|[[:space:]]$|[[:cntrl:]]")')
    AND profile->>'display_name'=btrim(profile->>'display_name') AND profile->>'display_name' !~ '[[:cntrl:]]'
    AND profile->>'headline'=btrim(profile->>'headline') AND profile->>'headline' !~ '[[:cntrl:]]'
    AND profile->>'location'=btrim(profile->>'location') AND profile->>'location' !~ '[[:cntrl:]]'
      AND (profile->'linkedin'='null'::jsonb OR
        (jsonb_typeof(profile->'linkedin')='string' AND length(profile->>'linkedin')<=2048
      AND profile->>'linkedin' ~ '^https://www[.]linkedin[.]com/in/[a-zA-Z0-9_%.-]+$'))) IS TRUE),
    CONSTRAINT consent_source_resume_shape CHECK (((resume IS NULL AND resume_sha256 IS NULL)
      OR (resume IS NOT NULL AND resume_sha256 IS NOT NULL AND jsonb_typeof(resume)='object'
        AND octet_length(resume::text)<=2048
        AND resume ?& ARRAY['reference_id','resume_version_id','organization_id','application_id','job_id',
          'content_sha256','byte_count','media_type','source_observed_at']
        AND resume - ARRAY['reference_id','resume_version_id','organization_id','application_id','job_id',
          'content_sha256','byte_count','media_type','source_observed_at']='{}'::jsonb
      AND jsonb_typeof(resume->'reference_id')='string'
      AND resume->>'reference_id' ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
      AND jsonb_typeof(resume->'resume_version_id')='string'
      AND resume->>'resume_version_id' ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
      AND CASE WHEN jsonb_typeof(resume->'organization_id')='number' THEN
        resume->>'organization_id' ~ '^[1-9][0-9]*$' AND (resume->>'organization_id')::numeric BETWEEN 1 AND 2147483647 ELSE false END
      AND CASE WHEN jsonb_typeof(resume->'application_id')='number' THEN
        resume->>'application_id' ~ '^[1-9][0-9]*$' AND (resume->>'application_id')::numeric BETWEEN 1 AND 2147483647 ELSE false END
      AND CASE WHEN jsonb_typeof(resume->'job_id')='number' THEN
        resume->>'job_id' ~ '^[1-9][0-9]*$' AND (resume->>'job_id')::numeric BETWEEN 1 AND 2147483647 ELSE false END
      AND CASE WHEN jsonb_typeof(resume->'byte_count')='number' THEN
        resume->>'byte_count' ~ '^[1-9][0-9]*$' AND (resume->>'byte_count')::numeric BETWEEN 1 AND 5242880 ELSE false END
      AND jsonb_typeof(resume->'content_sha256')='string'
      AND resume->>'content_sha256' ~ '^[0-9a-f]{64}$'
      AND jsonb_typeof(resume->'media_type')='string'
      AND resume->>'media_type' IN ('application/pdf','application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document')
      AND jsonb_typeof(resume->'source_observed_at')='string'
      AND resume->>'source_observed_at' ~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}[.][0-9]{3}Z$'
    )) IS TRUE)
);
ALTER TABLE candidate_consent_state ADD CONSTRAINT consent_active_source_fk
    FOREIGN KEY(tenant_id,subject_id,active_source_id)
    REFERENCES candidate_consent_sources(tenant_id,subject_id,source_id) ON DELETE RESTRICT;
CREATE TABLE candidate_consent_receipts (
    event_id UUID PRIMARY KEY,
    idempotency_key CHAR(64) NOT NULL UNIQUE CHECK (idempotency_key ~ '^[0-9a-f]{64}$'),
    command_digest CHAR(64) NOT NULL CHECK (command_digest ~ '^[0-9a-f]{64}$'),
    subject_id UUID NOT NULL,
    tenant_id TEXT NOT NULL CHECK (tenant_id='candidate_' || subject_id::text),
    version BIGINT NOT NULL CHECK (version BETWEEN 1 AND 9007199254740991),
    action TEXT NOT NULL CHECK (action IN ('grant','withdraw')),
    outcome TEXT NOT NULL CHECK (outcome IN ('granted','withdrawn','superseded','identity_review_required')),
    source_id UUID,
    global_candidate_id UUID REFERENCES global_candidates(id) ON DELETE RESTRICT,
    effective_version BIGINT NOT NULL CHECK (effective_version BETWEEN 0 AND 9007199254740991),
    effective_action TEXT CHECK (effective_action IN ('grant','withdraw')),
    effective_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    verified_issuer TEXT NOT NULL CHECK (verified_issuer='vantahire'),
    verified_actor_id TEXT NOT NULL CHECK (verified_actor_id='vantahire-backend'),
    UNIQUE (tenant_id,subject_id,version),
    FOREIGN KEY(tenant_id,subject_id) REFERENCES candidate_consent_state(tenant_id,subject_id) ON DELETE RESTRICT,
    FOREIGN KEY(tenant_id,subject_id,source_id)
      REFERENCES candidate_consent_sources(tenant_id,subject_id,source_id) ON DELETE RESTRICT,
    CHECK ((outcome='granted' AND source_id IS NOT NULL AND global_candidate_id IS NOT NULL)
      OR (outcome<>'granted' AND source_id IS NULL))
);
CREATE INDEX consent_sources_canonical_idx ON candidate_consent_sources(global_candidate_id,source_version);

CREATE FUNCTION candidate_consent_append_only() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog,public SET row_security=off AS $$
DECLARE has_rows BOOLEAN;
BEGIN
    IF TG_OP='TRUNCATE' THEN
        EXECUTE format('SELECT EXISTS(SELECT 1 FROM %I.%I LIMIT 1)',TG_TABLE_SCHEMA,TG_TABLE_NAME) INTO has_rows;
        IF NOT has_rows THEN RETURN NULL; END IF;
    END IF;
    RAISE EXCEPTION USING ERRCODE='55000',MESSAGE='candidate_consent_append_only';
EXCEPTION WHEN insufficient_privilege THEN
    -- A tenant-filtered count must never mistake another subject's evidence for an empty table.
    RAISE EXCEPTION USING ERRCODE='55000',MESSAGE='candidate_consent_append_only';
END;
$$;
CREATE TRIGGER consent_sources_no_mutation BEFORE UPDATE OR DELETE ON candidate_consent_sources
    FOR EACH ROW EXECUTE FUNCTION candidate_consent_append_only();
CREATE TRIGGER consent_sources_no_truncate BEFORE TRUNCATE ON candidate_consent_sources
    FOR EACH STATEMENT EXECUTE FUNCTION candidate_consent_append_only();
CREATE TRIGGER consent_receipts_no_mutation BEFORE UPDATE OR DELETE ON candidate_consent_receipts
    FOR EACH ROW EXECUTE FUNCTION candidate_consent_append_only();
CREATE TRIGGER consent_receipts_no_truncate BEFORE TRUNCATE ON candidate_consent_receipts
    FOR EACH STATEMENT EXECUTE FUNCTION candidate_consent_append_only();
CREATE FUNCTION candidate_consent_binding_immutable() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog,public AS $$
BEGIN
    IF NEW.subject_id IS DISTINCT FROM OLD.subject_id OR NEW.tenant_id IS DISTINCT FROM OLD.tenant_id
      OR NEW.created_at IS DISTINCT FROM OLD.created_at
      OR (OLD.global_candidate_id IS NOT NULL AND NEW.global_candidate_id IS DISTINCT FROM OLD.global_candidate_id) THEN
        RAISE EXCEPTION USING ERRCODE='55000',MESSAGE='candidate_consent_binding_immutable';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER consent_state_binding BEFORE UPDATE ON candidate_consent_state
    FOR EACH ROW EXECUTE FUNCTION candidate_consent_binding_immutable();

ALTER TABLE candidate_consent_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE candidate_consent_state FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_consent_state ON candidate_consent_state FOR ALL TO PUBLIC
    USING (tenant_id=current_setting('app.current_tenant_id',true))
    WITH CHECK (tenant_id=current_setting('app.current_tenant_id',true));
ALTER TABLE candidate_consent_sources ENABLE ROW LEVEL SECURITY;
ALTER TABLE candidate_consent_sources FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_consent_sources ON candidate_consent_sources FOR ALL TO PUBLIC
    USING (tenant_id=current_setting('app.current_tenant_id',true))
    WITH CHECK (tenant_id=current_setting('app.current_tenant_id',true));
ALTER TABLE candidate_consent_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE candidate_consent_receipts FORCE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation_consent_receipts ON candidate_consent_receipts FOR ALL TO PUBLIC
    USING (tenant_id=current_setting('app.current_tenant_id',true))
    WITH CHECK (tenant_id=current_setting('app.current_tenant_id',true));
REVOKE ALL ON candidate_consent_state,candidate_consent_sources,candidate_consent_receipts FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_consent_append_only() FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_consent_binding_immutable() FROM PUBLIC;
