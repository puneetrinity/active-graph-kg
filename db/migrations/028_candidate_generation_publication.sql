-- Wave 4D. Separate immutable artifacts from mutable worker coordination.
-- No backfill, provider adoption, legacy-vector change, or model invocation.
CREATE TABLE public.candidate_index_sources (
    source_id uuid PRIMARY KEY,
    scope_key text NOT NULL,
    source_kind text NOT NULL CHECK (source_kind IN ('organization_application','candidate_consent','approved_provider')),
    authority_id uuid NOT NULL,
    source_version bigint NOT NULL CHECK (source_version BETWEEN 1 AND 9007199254740991),
    source_hash char(64) NOT NULL CHECK (source_hash ~ '^[0-9a-f]{64}$'),
    command_digest char(64) NOT NULL CHECK (command_digest ~ '^[0-9a-f]{64}$'),
    tenant_id text,
    candidate_id uuid,
    global_candidate_id uuid REFERENCES public.global_candidates(id) ON DELETE RESTRICT,
    reference_id uuid,
    resume_version_id uuid REFERENCES public.organization_candidate_resume_evidence(resume_version_id) ON DELETE RESTRICT,
    consent_subject_id uuid,
    consent_source_id uuid,
    provider_identity_id uuid REFERENCES public.global_candidate_source_identities(id) ON DELETE RESTRICT,
    provider_observation_id uuid REFERENCES public.global_candidate_source_observations(id) ON DELETE RESTRICT,
    content_kind text NOT NULL CHECK (content_kind IN ('pinned_text','original_bytes','approved_profile','provider_profile')),
    professional_text text,
    original_bytes bytea,
    approved_profile jsonb,
    match_tokens jsonb NOT NULL,
    key_versions integer[] NOT NULL,
    source_observed_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (scope_key,source_id),
    UNIQUE (scope_key,authority_id,source_id),
    UNIQUE (scope_key,source_kind,authority_id,source_version),
    FOREIGN KEY (tenant_id,reference_id,candidate_id)
        REFERENCES public.organization_candidate_references(tenant_id,reference_id,candidate_id) ON DELETE RESTRICT,
    CONSTRAINT index_source_resume_tuple_fk FOREIGN KEY (tenant_id,resume_version_id,candidate_id)
        REFERENCES public.organization_candidate_resume_evidence(tenant_id,resume_version_id,candidate_id) ON DELETE RESTRICT,
    FOREIGN KEY (tenant_id,consent_subject_id,consent_source_id)
        REFERENCES public.candidate_consent_sources(tenant_id,subject_id,source_id) ON DELETE RESTRICT,
    CONSTRAINT index_source_authority_shape CHECK ((
      (source_kind='organization_application' AND scope_key=tenant_id AND scope_key ~ '^org_[1-9][0-9]*$'
        AND candidate_id IS NOT NULL AND reference_id IS NOT NULL AND authority_id=reference_id
        AND resume_version_id=source_id AND consent_subject_id IS NULL AND consent_source_id IS NULL
        AND provider_identity_id IS NULL AND provider_observation_id IS NULL)
      OR (source_kind='candidate_consent' AND scope_key=tenant_id AND scope_key='candidate_'||consent_subject_id::text
        AND authority_id=consent_subject_id AND source_id=consent_source_id AND global_candidate_id IS NOT NULL
        AND candidate_id IS NULL AND reference_id IS NULL AND provider_identity_id IS NULL AND provider_observation_id IS NULL)
      OR (source_kind='approved_provider' AND scope_key='public_provider' AND tenant_id IS NULL
        AND global_candidate_id IS NOT NULL AND provider_identity_id=authority_id AND provider_observation_id=source_id
        AND candidate_id IS NULL AND reference_id IS NULL AND resume_version_id IS NULL
        AND consent_subject_id IS NULL AND consent_source_id IS NULL)
    ) IS TRUE),
    CONSTRAINT index_source_content_shape CHECK ((
      (source_kind='organization_application' AND content_kind='pinned_text' AND professional_text IS NOT NULL
        AND octet_length(professional_text) BETWEEN 1 AND 2097152 AND original_bytes IS NULL AND approved_profile IS NULL
        AND encode(sha256(convert_to(professional_text,'UTF8')),'hex')=source_hash)
      OR (source_kind='organization_application' AND content_kind='original_bytes' AND original_bytes IS NOT NULL
        AND octet_length(original_bytes) BETWEEN 1 AND 5242880 AND professional_text IS NULL AND approved_profile IS NULL
        AND encode(sha256(original_bytes),'hex')=source_hash)
      OR (((source_kind='candidate_consent' AND content_kind='approved_profile')
        OR (source_kind='approved_provider' AND content_kind='provider_profile'))
        AND professional_text IS NULL AND original_bytes IS NULL AND jsonb_typeof(approved_profile)='object'
        AND octet_length(approved_profile::text) BETWEEN 2 AND 65536)
    ) IS TRUE),
    CONSTRAINT index_source_tokens_shape CHECK ((jsonb_typeof(match_tokens)='array'
      AND jsonb_array_length(match_tokens) BETWEEN 1 AND 64 AND octet_length(match_tokens::text)<=32768
      AND cardinality(key_versions) BETWEEN 1 AND 16 AND array_position(key_versions,NULL) IS NULL
      AND 0<ALL(key_versions)) IS TRUE)
);
CREATE INDEX index_sources_authority_idx ON public.candidate_index_sources(scope_key,authority_id,source_version);
CREATE INDEX index_sources_resume_idx ON public.candidate_index_sources(resume_version_id);

CREATE TABLE public.candidate_index_generations (
    generation_id uuid PRIMARY KEY,
    scope_key text NOT NULL,
    authority_id uuid NOT NULL,
    source_id uuid NOT NULL,
    generation bigint NOT NULL CHECK (generation BETWEEN 1 AND 9007199254740991),
    source_manifest jsonb NOT NULL CHECK (jsonb_typeof(source_manifest)='object' AND octet_length(source_manifest::text)<=8192),
    input_sha256 char(64) NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    policy jsonb NOT NULL,
    policy_sha256 char(64) NOT NULL CHECK (policy_sha256 ~ '^[0-9a-f]{64}$'),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (scope_key,generation_id),
    UNIQUE (scope_key,authority_id,generation_id),
    UNIQUE (scope_key,authority_id,generation),
    UNIQUE (source_id,policy_sha256),
    FOREIGN KEY (scope_key,authority_id,source_id) REFERENCES public.candidate_index_sources(scope_key,authority_id,source_id) ON DELETE RESTRICT,
    CONSTRAINT index_generation_policy_shape CHECK ((jsonb_typeof(policy)='object' AND octet_length(policy::text)<=8192
      AND policy ?& ARRAY['extraction_schema_sha256','extraction_prompt_sha256','primary_model_id','fallback_model_id',
        'embedding_model_id','embedding_artifact_revision','parser_version','extraction_adapter_version',
        'text_builder_version','dimension','max_chunks','chunk_characters','minimum_confidence']
      AND policy-ARRAY['extraction_schema_sha256','extraction_prompt_sha256','primary_model_id','fallback_model_id',
        'embedding_model_id','embedding_artifact_revision','parser_version','extraction_adapter_version',
        'text_builder_version','dimension','max_chunks','chunk_characters','minimum_confidence']='{}'::jsonb
      AND policy->>'extraction_schema_sha256' ~ '^[0-9a-f]{64}$' AND policy->>'extraction_prompt_sha256' ~ '^[0-9a-f]{64}$'
      AND policy->>'embedding_artifact_revision' ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'
      AND policy->'dimension'='384'::jsonb
      AND (policy->>'max_chunks')::integer BETWEEN 1 AND 32
      AND (policy->>'chunk_characters')::integer BETWEEN 1 AND 1200
      AND (policy->>'minimum_confidence')::numeric BETWEEN 0.65 AND 1) IS TRUE)
);

CREATE TABLE public.candidate_index_extractions (
    extraction_id uuid PRIMARY KEY,
    scope_key text NOT NULL,
    generation_id uuid NOT NULL,
    input_sha256 char(64) NOT NULL CHECK (input_sha256 ~ '^[0-9a-f]{64}$'),
    extraction_sha256 char(64) NOT NULL CHECK (extraction_sha256 ~ '^[0-9a-f]{64}$'),
    namespace jsonb NOT NULL CHECK (jsonb_typeof(namespace)='object' AND octet_length(namespace::text)<=65536),
    evidence jsonb NOT NULL CHECK (jsonb_typeof(evidence)='array' AND jsonb_array_length(evidence) BETWEEN 1 AND 32
      AND octet_length(evidence::text)<=16384),
    confidence double precision NOT NULL CHECK (confidence BETWEEN 0.65 AND 1),
    model_id text NOT NULL CHECK (octet_length(model_id) BETWEEN 1 AND 200),
    attempt integer NOT NULL CHECK (attempt BETWEEN 0 AND 2),
    chunks jsonb NOT NULL CHECK (jsonb_typeof(chunks)='array' AND jsonb_array_length(chunks) BETWEEN 1 AND 32
      AND octet_length(chunks::text)<=262144),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (generation_id), UNIQUE (scope_key,generation_id,extraction_id),
    FOREIGN KEY (scope_key,generation_id) REFERENCES public.candidate_index_generations(scope_key,generation_id) ON DELETE RESTRICT
);
CREATE TABLE public.candidate_index_vectors (
    scope_key text NOT NULL,
    generation_id uuid NOT NULL,
    extraction_id uuid NOT NULL,
    ordinal integer NOT NULL CHECK (ordinal BETWEEN 0 AND 31),
    chunk_sha256 char(64) NOT NULL CHECK (chunk_sha256 ~ '^[0-9a-f]{64}$'),
    model_id text NOT NULL CHECK (octet_length(model_id) BETWEEN 1 AND 200),
    artifact_revision text NOT NULL CHECK (artifact_revision ~ '^([0-9a-f]{40}|[0-9a-f]{64})$'),
    embedding vector(384) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (generation_id,ordinal),
    FOREIGN KEY (scope_key,generation_id,extraction_id)
        REFERENCES public.candidate_index_extractions(scope_key,generation_id,extraction_id) ON DELETE RESTRICT,
    CONSTRAINT index_vector_unit_norm CHECK ((embedding <#> embedding) BETWEEN -1.001 AND -0.999)
);
CREATE INDEX index_vectors_scope_idx ON public.candidate_index_vectors(scope_key,generation_id);

CREATE TABLE public.candidate_index_publication_events (
    event_id uuid PRIMARY KEY,
    scope_key text NOT NULL,
    authority_id uuid NOT NULL,
    outcome text NOT NULL CHECK (outcome IN ('ready','switch','invalidate')),
    previous_generation_id uuid,
    generation_id uuid,
    authority_source_id uuid NOT NULL,
    reason_code text NOT NULL CHECK (reason_code IN ('complete','privacy_restricted','consent_changed','superseded','source_removed')),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    FOREIGN KEY (scope_key,authority_id,previous_generation_id) REFERENCES public.candidate_index_generations(scope_key,authority_id,generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (scope_key,authority_id,generation_id) REFERENCES public.candidate_index_generations(scope_key,authority_id,generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (scope_key,authority_id,authority_source_id) REFERENCES public.candidate_index_sources(scope_key,authority_id,source_id) ON DELETE RESTRICT,
    CHECK ((outcome='invalidate' AND generation_id IS NULL) OR (outcome<>'invalidate' AND generation_id IS NOT NULL))
);
CREATE UNIQUE INDEX index_publication_once_idx ON public.candidate_index_publication_events(generation_id)
    WHERE outcome IN ('ready','switch');

CREATE TABLE public.candidate_index_heads (
    scope_key text NOT NULL,
    authority_id uuid NOT NULL,
    source_kind text NOT NULL CHECK (source_kind IN ('organization_application','candidate_consent','approved_provider')),
    desired_generation_id uuid NOT NULL,
    published_generation_id uuid,
    last_complete_generation_id uuid,
    logical_generation bigint NOT NULL CHECK (logical_generation BETWEEN 1 AND 9007199254740991),
    authority_source_id uuid NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (scope_key,authority_id),
    FOREIGN KEY (scope_key,authority_id,desired_generation_id) REFERENCES public.candidate_index_generations(scope_key,authority_id,generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (scope_key,authority_id,published_generation_id) REFERENCES public.candidate_index_generations(scope_key,authority_id,generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (scope_key,authority_id,last_complete_generation_id) REFERENCES public.candidate_index_generations(scope_key,authority_id,generation_id) ON DELETE RESTRICT,
    FOREIGN KEY (scope_key,authority_id,authority_source_id) REFERENCES public.candidate_index_sources(scope_key,authority_id,source_id) ON DELETE RESTRICT
);
CREATE TABLE public.candidate_index_jobs (
    job_id uuid PRIMARY KEY,
    scope_key text NOT NULL,
    generation_id uuid NOT NULL,
    stage text NOT NULL CHECK (stage IN ('extract','embed')),
    priority_class text NOT NULL CHECK (priority_class IN ('interactive','provider','maintenance')),
    state text NOT NULL CHECK (state IN ('waiting_source','waiting_admission','pending_extraction','extracting','needs_review',
      'pending_embedding','embedding','ready','failed','superseded','quarantined','cancelled')),
    attempts integer NOT NULL DEFAULT 0,
    lease_generation bigint NOT NULL DEFAULT 0 CHECK (lease_generation BETWEEN 0 AND 9007199254740991),
    lease_token uuid,
    lease_expires_at timestamptz,
    dispatch_deadline timestamptz,
    dispatch_reserved_generation bigint NOT NULL DEFAULT 0 CHECK (dispatch_reserved_generation BETWEEN 0 AND lease_generation),
    next_attempt_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    error_code text CHECK (error_code IN ('privacy_restricted','privacy_review','privacy_unavailable','source_missing',
      'consent_changed','policy_mismatch','low_confidence','unsupported_format','incomplete','chunk_overflow',
      'provider_timeout','provider_unavailable','rate_limited','attempts_exhausted','dispatch_exhausted','superseded')),
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (generation_id,stage),
    FOREIGN KEY (scope_key,generation_id) REFERENCES public.candidate_index_generations(scope_key,generation_id) ON DELETE RESTRICT,
    CONSTRAINT index_job_attempts CHECK (attempts BETWEEN 0 AND CASE WHEN stage='extract' THEN 2 ELSE 3 END
      AND attempts<=lease_generation),
    CONSTRAINT index_job_lease_shape CHECK (((state IN ('extracting','embedding') AND lease_token IS NOT NULL
      AND lease_expires_at IS NOT NULL) OR (state NOT IN ('extracting','embedding') AND lease_token IS NULL
      AND lease_expires_at IS NULL)) IS TRUE)
);
CREATE INDEX index_jobs_due_idx ON public.candidate_index_jobs(stage,priority_class,next_attempt_at,scope_key);
CREATE INDEX index_jobs_generation_idx ON public.candidate_index_jobs(scope_key,generation_id);
CREATE TABLE public.candidate_index_scheduler (
    scope_key text NOT NULL,
    stage text NOT NULL CHECK (stage IN ('extract','embed')),
    priority_class text NOT NULL CHECK (priority_class IN ('interactive','provider','maintenance','cursor')),
    turns bigint NOT NULL DEFAULT 0 CHECK (turns BETWEEN 0 AND 9007199254740991),
    last_served_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    PRIMARY KEY (scope_key,stage,priority_class)
);

CREATE FUNCTION public.candidate_index_append_only() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    IF TG_OP='TRUNCATE' THEN
        -- Static branches: no caller-selected relation and no row-security-off workaround.
        IF TG_TABLE_NAME='candidate_index_sources' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_sources) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_generations' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_generations) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_extractions' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_extractions) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_vectors' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_vectors) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_publication_events' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_publication_events) THEN RETURN NULL; END IF;
    END IF;
    RAISE EXCEPTION USING ERRCODE='55000',MESSAGE='candidate_index_append_only';
END;
$$;

-- DDL-only policy/trigger expansion with a literal table inventory. Runtime
-- routines below contain no dynamic scope/relation SQL. The coordinator is the
-- non-superuser migration owner, never PUBLIC, runtime or a GUC-selected role.
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['candidate_index_sources','candidate_index_generations','candidate_index_extractions',
      'candidate_index_vectors','candidate_index_publication_events','candidate_index_jobs','candidate_index_heads','candidate_index_scheduler'] LOOP
        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',t);
        EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY',t);
        EXECUTE format('CREATE POLICY index_scope_read ON public.%I FOR SELECT TO PUBLIC USING (scope_key=current_setting(''app.current_tenant_id'',true))',t);
        EXECUTE format('CREATE POLICY index_owner_coordination ON public.%I FOR ALL TO %I USING (true) WITH CHECK (true)',t,current_user);
        EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC',t);
        IF t NOT IN ('candidate_index_jobs','candidate_index_heads','candidate_index_scheduler') THEN
            EXECUTE format('CREATE TRIGGER index_evidence_no_mutation BEFORE UPDATE OR DELETE ON public.%I FOR EACH ROW EXECUTE FUNCTION public.candidate_index_append_only()',t);
            EXECUTE format('CREATE TRIGGER index_evidence_no_truncate BEFORE TRUNCATE ON public.%I FOR EACH STATEMENT EXECUTE FUNCTION public.candidate_index_append_only()',t);
        END IF;
    END LOOP;
END;
$$;
REVOKE ALL ON FUNCTION public.candidate_index_append_only() FROM PUBLIC;

-- Claim has two explicit phases. Selection leases a job without spending a
-- model attempt. A second call with the exact lease reserves ONE dispatch only
-- after another authority check. A lost dispatch response is conservatively
-- spent. Neither a worker restart nor a lease reclaim resets the deadline.
CREATE FUNCTION public.candidate_index_claim(
    p_stage text,p_limit integer,p_options jsonb,
    p_job_id uuid DEFAULT NULL,p_token uuid DEFAULT NULL,p_generation bigint DEFAULT NULL,
    p_policy_sha256 text DEFAULT NULL
) RETURNS SETOF jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE original_scope text:=current_setting('app.current_tenant_id',true); item record;
    j public.candidate_index_jobs%ROWTYPE; g public.candidate_index_generations%ROWTYPE;
    s public.candidate_index_sources%ROWTYPE; resume_source public.candidate_index_sources%ROWTYPE;
    allowed jsonb; lease_ms integer; deadline_ms integer; concurrency integer; scope_concurrency integer;
    iw integer; mw integer; turn bigint; chosen_class text; n integer:=0; deterministic boolean;
BEGIN
    IF (p_stage IN ('extract','embed') AND p_limit BETWEEN 1 AND 8
      AND jsonb_typeof(p_options)='object' AND p_options ?& ARRAY['lease_ms','deadline_ms','concurrency','scope_concurrency','interactive_weight','maintenance_weight']
      AND p_options-ARRAY['lease_ms','deadline_ms','concurrency','scope_concurrency','interactive_weight','maintenance_weight']='{}'::jsonb) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_claim_bounds'; END IF;
    lease_ms:=(p_options->>'lease_ms')::integer; deadline_ms:=(p_options->>'deadline_ms')::integer;
    concurrency:=(p_options->>'concurrency')::integer; scope_concurrency:=(p_options->>'scope_concurrency')::integer;
    iw:=(p_options->>'interactive_weight')::integer; mw:=(p_options->>'maintenance_weight')::integer;
    IF (deadline_ms BETWEEN 1000 AND CASE p_stage WHEN 'extract' THEN 45000 ELSE 20000 END
      AND lease_ms BETWEEN deadline_ms+10001 AND 300000
      AND concurrency BETWEEN 1 AND 2 AND scope_concurrency=1 AND iw BETWEEN 1 AND 16 AND mw BETWEEN 1 AND 16) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_claim_bounds'; END IF;
    IF p_job_id IS NOT NULL THEN
        IF p_limit<>1 OR p_token IS NULL OR p_generation IS NULL OR p_policy_sha256 !~ '^[0-9a-f]{64}$' THEN
            RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_dispatch_bounds'; END IF;
        SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id AND stage=p_stage;
        IF NOT FOUND THEN RETURN; END IF;
        SELECT * INTO g FROM public.candidate_index_generations WHERE generation_id=j.generation_id;
        SELECT * INTO s FROM public.candidate_index_sources WHERE source_id=g.source_id;
        IF s.source_kind='approved_provider' THEN RETURN; END IF;
        PERFORM set_config('app.current_tenant_id',j.scope_key,true);
        allowed:=public.candidate_index_status(s.source_id);
        IF (allowed->>'eligible')::boolean IS DISTINCT FROM true THEN
            PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN; END IF;
        SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id FOR UPDATE;
        IF j.lease_token IS DISTINCT FROM p_token OR j.lease_generation IS DISTINCT FROM p_generation
          OR j.lease_expires_at<=clock_timestamp() OR j.state<>(CASE p_stage WHEN 'extract' THEN 'extracting' ELSE 'embedding' END)
          OR (allowed->>'generation_id')::uuid IS DISTINCT FROM j.generation_id OR g.policy_sha256<>p_policy_sha256 THEN
            PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN; END IF;
        deterministic:=p_stage='extract' AND s.source_kind='candidate_consent' AND s.resume_version_id IS NULL;
        -- dispatch_deadline is a generation-level budget, reserved once. The
        -- dispatch marker distinguishes an unspent lease from an ambiguous one.
        IF j.dispatch_reserved_generation=j.lease_generation OR
          (j.dispatch_deadline IS NOT NULL AND j.dispatch_deadline<=clock_timestamp()) OR
          (NOT deterministic AND j.attempts>=(CASE p_stage WHEN 'extract' THEN 2 ELSE 3 END)) THEN
            PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN; END IF;
        UPDATE public.candidate_index_jobs SET attempts=attempts+CASE WHEN deterministic THEN 0 ELSE 1 END,
          dispatch_reserved_generation=lease_generation,
          dispatch_deadline=coalesce(dispatch_deadline,clock_timestamp()+deadline_ms*interval '1 millisecond'),
          updated_at=clock_timestamp() WHERE job_id=p_job_id RETURNING * INTO j;
        IF s.source_kind='candidate_consent' AND s.resume_version_id IS NOT NULL THEN
            SELECT * INTO resume_source FROM public.candidate_index_sources WHERE source_id=s.resume_version_id
              AND source_kind='organization_application'
              AND scope_key='org_'||(g.source_manifest->'resume'->>'organization_id');
            IF NOT FOUND THEN RAISE EXCEPTION USING ERRCODE='23503',MESSAGE='candidate_index_resume_dependency_missing'; END IF;
        END IF;
        RETURN NEXT jsonb_build_object('job_id',j.job_id,'scope_key',j.scope_key,'generation_id',j.generation_id,
          'lease_token',j.lease_token,'lease_generation',j.lease_generation,'lease_expires_at',j.lease_expires_at,
          'dispatch_deadline',j.dispatch_deadline,'attempt',j.attempts,'stage',j.stage,
          'source_id',s.source_id,'source_kind',s.source_kind,'policy',g.policy,'policy_sha256',g.policy_sha256,
          'input_sha256',g.input_sha256,'source_manifest',g.source_manifest,
          'content_kind',coalesce(resume_source.content_kind,s.content_kind),
          'professional_text',coalesce(resume_source.professional_text,s.professional_text),
          'original_bytes',CASE WHEN coalesce(resume_source.original_bytes,s.original_bytes) IS NOT NULL
            THEN replace(encode(coalesce(resume_source.original_bytes,s.original_bytes),'base64'),E'\n','') END,
          'approved_profile',s.approved_profile,
          'extraction',(SELECT jsonb_build_object('extraction_id',e.extraction_id,'chunks',e.chunks,
            'extraction_sha256',e.extraction_sha256) FROM public.candidate_index_extractions e WHERE e.generation_id=j.generation_id));
        PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN;
    END IF;
    IF p_token IS NOT NULL OR p_generation IS NOT NULL OR p_policy_sha256 IS NOT NULL THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_claim_identity'; END IF;
    -- A stage cursor serializes reservations across replicas, not model work.
    -- Try-lock avoids waiting while another replica owns the scheduling turn.
    IF NOT pg_try_advisory_xact_lock(hashtextextended('candidate-index-scheduler:'||p_stage,0)) THEN RETURN; END IF;
    INSERT INTO public.candidate_index_scheduler(scope_key,stage,priority_class)
      VALUES('index_scheduler',p_stage,'cursor') ON CONFLICT DO NOTHING;
    SELECT turns INTO turn FROM public.candidate_index_scheduler
      WHERE scope_key='index_scheduler' AND stage=p_stage AND priority_class='cursor';
    chosen_class:=CASE WHEN mod(turn,iw+mw)<iw THEN 'interactive' ELSE 'maintenance' END;
    FOR item IN SELECT q.job_id,gen_row.source_id FROM public.candidate_index_jobs q
      JOIN public.candidate_index_generations gen_row ON gen_row.generation_id=q.generation_id
      JOIN public.candidate_index_sources src_row ON src_row.source_id=gen_row.source_id
      LEFT JOIN public.candidate_index_scheduler c ON c.scope_key=q.scope_key AND c.stage=q.stage AND c.priority_class=q.priority_class
      WHERE q.stage=p_stage AND q.priority_class IN ('interactive','maintenance') AND src_row.source_kind<>'approved_provider'
        AND ((q.state=CASE p_stage WHEN 'extract' THEN 'pending_extraction' ELSE 'pending_embedding' END AND q.next_attempt_at<=clock_timestamp())
          OR (q.state IN ('extracting','embedding') AND q.lease_expires_at<=clock_timestamp()))
      ORDER BY CASE WHEN q.priority_class=chosen_class THEN 0 ELSE 1 END,
        c.last_served_at NULLS FIRST,q.created_at,q.job_id LIMIT 256
    LOOP
      BEGIN
        SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=item.job_id;
        PERFORM set_config('app.current_tenant_id',j.scope_key,true);
        allowed:=public.candidate_index_status(item.source_id,true);
        IF allowed->>'contended'='true' THEN CONTINUE; END IF;
        SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=item.job_id FOR UPDATE SKIP LOCKED;
        IF NOT FOUND THEN CONTINUE; END IF;
        IF (allowed->>'eligible')::boolean IS DISTINCT FROM true OR (allowed->>'generation_id')::uuid IS DISTINCT FROM j.generation_id THEN
            UPDATE public.candidate_index_jobs SET state=CASE WHEN allowed->>'reason'='superseded'
                OR ((allowed->>'eligible')::boolean AND (allowed->>'generation_id')::uuid<>j.generation_id) THEN 'superseded' ELSE 'quarantined' END,
              error_code=CASE WHEN (allowed->>'eligible')::boolean AND (allowed->>'generation_id')::uuid<>j.generation_id
                THEN 'superseded' ELSE coalesce(allowed->>'reason','privacy_unavailable') END,
              lease_token=NULL,lease_expires_at=NULL,updated_at=clock_timestamp()
              WHERE job_id=j.job_id; CONTINUE; END IF;
        IF j.dispatch_deadline<=clock_timestamp() OR j.attempts>=(CASE p_stage WHEN 'extract' THEN 2 ELSE 3 END) THEN
            UPDATE public.candidate_index_jobs SET state='failed',error_code='attempts_exhausted',lease_token=NULL,
              lease_expires_at=NULL,updated_at=clock_timestamp() WHERE job_id=j.job_id; CONTINUE; END IF;
        IF (SELECT count(*) FROM public.candidate_index_jobs WHERE stage=p_stage
          AND state IN ('extracting','embedding') AND lease_expires_at>clock_timestamp())>=concurrency THEN EXIT; END IF;
        IF EXISTS(SELECT 1 FROM public.candidate_index_jobs WHERE stage=p_stage AND scope_key=j.scope_key
          AND state IN ('extracting','embedding') AND lease_expires_at>clock_timestamp()) THEN CONTINUE; END IF;
        UPDATE public.candidate_index_jobs SET state=CASE p_stage WHEN 'extract' THEN 'extracting' ELSE 'embedding' END,
          lease_generation=lease_generation+1,lease_token=gen_random_uuid(),lease_expires_at=clock_timestamp()+lease_ms*interval '1 millisecond',
          updated_at=clock_timestamp() WHERE job_id=j.job_id RETURNING * INTO j;
        INSERT INTO public.candidate_index_scheduler(scope_key,stage,priority_class,turns,last_served_at)
          VALUES(j.scope_key,p_stage,j.priority_class,1,clock_timestamp()) ON CONFLICT(scope_key,stage,priority_class)
          DO UPDATE SET turns=candidate_index_scheduler.turns+1,last_served_at=EXCLUDED.last_served_at,updated_at=clock_timestamp();
        UPDATE public.candidate_index_scheduler SET turns=turns+1,updated_at=clock_timestamp()
          WHERE scope_key='index_scheduler' AND stage=p_stage AND priority_class='cursor';
        RETURN NEXT jsonb_build_object('job_id',j.job_id,'scope_key',j.scope_key,'generation_id',j.generation_id,
          'lease_token',j.lease_token,'lease_generation',j.lease_generation,'lease_expires_at',j.lease_expires_at,'stage',j.stage);
        n:=n+1;
        -- One selected class per call preserves the weighted order even for a
        -- batch. The caller may issue its remaining bounded turns separately.
        EXIT WHEN n>=p_limit OR n>=1;
      EXCEPTION WHEN lock_not_available THEN
        -- Privacy contention reserves no attempt and cannot starve other scopes.
        CONTINUE;
      END;
    END LOOP;
    PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true);
END;
$$;

-- This routine is also the common authority check for coordination. The caller
-- supplies an opaque source id, never a tenant. No content or tokens are returned.
CREATE FUNCTION public.candidate_index_status(p_source_id uuid,p_nonblocking boolean DEFAULT false) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE s public.candidate_index_sources%ROWTYPE; token jsonb; canonical record; checked record; decision text; h record; j record;
BEGIN
    SELECT * INTO s FROM public.candidate_index_sources WHERE source_id=p_source_id
      AND scope_key=current_setting('app.current_tenant_id',true);
    IF NOT FOUND OR s.source_kind='approved_provider' THEN RETURN NULL; END IF;
    -- The shipped consent receiver takes this lock before its privacy locks.
    -- Follow that order; an index completion and a withdrawal cannot interleave.
    IF s.source_kind='candidate_consent' THEN
        IF p_nonblocking THEN
            IF NOT pg_try_advisory_xact_lock(hashtextextended('candidate-consent:'||s.consent_subject_id::text,0)) THEN
                RETURN jsonb_build_object('eligible',false,'reason','privacy_unavailable','contended',true); END IF;
        ELSE PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||s.consent_subject_id::text,0)); END IF;
    END IF;
    FOR token IN SELECT value FROM jsonb_array_elements(s.match_tokens)
      ORDER BY value->>'identifier_type',value->>'key_version',value->>'token' LOOP
        IF (jsonb_typeof(token)='object' AND token ?& ARRAY['identifier_type','key_version','token']
          AND token-ARRAY['identifier_type','key_version','token']='{}'::jsonb
          AND token->>'identifier_type' IN ('email','phone','linkedin_url','github_url','signal_candidate_id',
            'vantahire_application_id','vantahire_resume_id') AND token->>'token' ~ '^[0-9a-f]{64}$'
          AND token->>'key_version' ~ '^[1-9][0-9]{0,8}$'
          AND (token->>'key_version')::integer=ANY(s.key_versions)) IS NOT TRUE THEN
            RETURN jsonb_build_object('eligible',false,'reason','privacy_unavailable');
        END IF;
        IF p_nonblocking THEN
            IF NOT pg_try_advisory_xact_lock(hashtextextended('candidate-privacy-token:'||
              (token->>'identifier_type')||':'||(token->>'key_version')||':'||(token->>'token'),0)) THEN
                RETURN jsonb_build_object('eligible',false,'reason','privacy_unavailable','contended',true); END IF;
        ELSE PERFORM pg_advisory_xact_lock(hashtextextended('candidate-privacy-token:'||
          (token->>'identifier_type')||':'||(token->>'key_version')||':'||(token->>'token'),0)); END IF;
    END LOOP;
    SELECT * INTO canonical FROM public.candidate_privacy_resolve_canonical(s.global_candidate_id,
      CASE WHEN s.candidate_id IS NOT NULL THEN s.tenant_id ELSE NULL END,s.candidate_id);
    IF canonical.needs_review IS DISTINCT FROM false THEN
        RETURN jsonb_build_object('eligible',false,'reason','privacy_review');
    END IF;
    IF canonical.global_candidate_id IS NOT NULL THEN
        IF p_nonblocking THEN
            IF NOT pg_try_advisory_xact_lock(hashtextextended('candidate-privacy-global:'||canonical.global_candidate_id::text,0)) THEN
                RETURN jsonb_build_object('eligible',false,'reason','privacy_unavailable','contended',true); END IF;
        ELSE PERFORM pg_advisory_xact_lock(hashtextextended('candidate-privacy-global:'||canonical.global_candidate_id::text,0)); END IF;
    END IF;
    IF s.candidate_id IS NOT NULL THEN
        IF p_nonblocking THEN
            IF NOT pg_try_advisory_xact_lock(hashtextextended('candidate-privacy-candidate:'||s.tenant_id||':'||s.candidate_id::text,0)) THEN
                RETURN jsonb_build_object('eligible',false,'reason','privacy_unavailable','contended',true); END IF;
        ELSE PERFORM pg_advisory_xact_lock(hashtextextended('candidate-privacy-candidate:'||s.tenant_id||':'||s.candidate_id::text,0)); END IF;
    END IF;
    SELECT * INTO checked FROM public.candidate_privacy_resolve_canonical(s.global_candidate_id,
      CASE WHEN s.candidate_id IS NOT NULL THEN s.tenant_id ELSE NULL END,s.candidate_id);
    IF checked.needs_review IS DISTINCT FROM false OR checked.global_candidate_id IS DISTINCT FROM canonical.global_candidate_id THEN
        RETURN jsonb_build_object('eligible',false,'reason','privacy_review');
    END IF;
    SELECT m.decision INTO decision FROM public.candidate_privacy_match(s.match_tokens,canonical.global_candidate_id,
      CASE WHEN s.candidate_id IS NOT NULL THEN s.tenant_id ELSE NULL END,s.candidate_id) m LIMIT 1;
    IF coalesce(decision,'allow') NOT IN ('allow','block_global') OR (s.source_kind='candidate_consent' AND decision='block_global') THEN
        RETURN jsonb_build_object('eligible',false,'reason',CASE WHEN decision='review' THEN 'privacy_review' ELSE 'privacy_restricted' END);
    END IF;
    IF s.source_kind='candidate_consent' AND NOT EXISTS (
      SELECT 1 FROM public.candidate_consent_state c WHERE c.subject_id=s.consent_subject_id
        AND c.tenant_id=s.scope_key AND c.active_source_id=s.consent_source_id AND c.effective_action='grant'
        AND c.effective_version=s.source_version AND c.global_candidate_id=s.global_candidate_id
    ) THEN RETURN jsonb_build_object('eligible',false,'reason','consent_changed'); END IF;
    SELECT * INTO h FROM public.candidate_index_heads WHERE scope_key=s.scope_key AND authority_id=s.authority_id;
    IF FOUND AND h.authority_source_id<>s.source_id THEN
        RETURN jsonb_build_object('eligible',false,'reason','superseded');
    END IF;
    SELECT state,error_code INTO j FROM public.candidate_index_jobs WHERE generation_id=h.desired_generation_id
      ORDER BY CASE stage WHEN 'embed' THEN 0 ELSE 1 END LIMIT 1;
    RETURN jsonb_build_object('eligible',true,'reason',NULL,'source_id',s.source_id,
      'generation_id',h.desired_generation_id,'published_generation_id',h.published_generation_id,
      'state',j.state,'error_code',j.error_code);
END;
$$;

CREATE FUNCTION public.candidate_index_capture_source(
    p_kind text,p_upstream_id uuid,p_command_digest text,p_content_kind text,p_content text,
    p_tokens jsonb,p_known_versions integer[],p_policy_json text,p_priority text DEFAULT 'interactive',
    p_scope_limit integer DEFAULT 1000,p_total_limit integer DEFAULT 10000
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE scope text:=current_setting('app.current_tenant_id',true); source_row public.candidate_index_sources%ROWTYPE;
    existing public.candidate_index_sources%ROWTYPE; evidence record; consent record; h public.candidate_index_heads%ROWTYPE;
    v_policy jsonb; policy_hash text; gen uuid; manifest jsonb; resume_manifest jsonb; state_name text; authorized jsonb;
BEGIN
    IF (p_kind IN ('organization_application','candidate_consent') AND p_priority IN ('interactive','maintenance')
      AND p_upstream_id IS NOT NULL AND p_command_digest ~ '^[0-9a-f]{64}$'
      AND octet_length(p_policy_json) BETWEEN 2 AND 8192 AND jsonb_typeof(p_tokens)='array'
      AND jsonb_array_length(p_tokens) BETWEEN 1 AND 64 AND cardinality(p_known_versions) BETWEEN 1 AND 16
      AND p_scope_limit BETWEEN 1 AND 1000 AND p_total_limit BETWEEN p_scope_limit AND 10000) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_capture_bounds';
    END IF;
    v_policy:=p_policy_json::jsonb; policy_hash:=encode(sha256(convert_to(p_policy_json,'UTF8')),'hex');
    IF p_kind='candidate_consent' THEN
        SELECT subject_id INTO source_row.consent_subject_id FROM public.candidate_consent_sources
          WHERE tenant_id=scope AND source_id=p_upstream_id;
        IF NOT FOUND THEN RAISE EXCEPTION USING ERRCODE='23503',MESSAGE='candidate_index_consent_authority_missing'; END IF;
        PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||source_row.consent_subject_id::text,0));
    END IF;
    PERFORM pg_advisory_xact_lock(hashtextextended('candidate-index-source:'||p_upstream_id::text,0));
    source_row.source_id:=p_upstream_id; source_row.scope_key:=scope; source_row.tenant_id:=scope;
    source_row.source_kind:=p_kind; source_row.command_digest:=p_command_digest;
    source_row.match_tokens:=p_tokens; source_row.key_versions:=p_known_versions;
    IF p_kind='organization_application' THEN
        SELECT e.*,r.application_id,r.job_id INTO evidence FROM public.organization_candidate_resume_evidence e
        JOIN public.organization_candidate_references r ON r.tenant_id=e.tenant_id AND r.reference_id=e.reference_id AND r.candidate_id=e.candidate_id
        JOIN public.organization_candidate_ingest_receipts i ON i.tenant_id=e.tenant_id AND i.reference_id=e.reference_id
          AND i.resume_version_id=e.resume_version_id AND i.candidate_id=e.candidate_id
        WHERE e.tenant_id=scope AND e.resume_version_id=p_upstream_id;
        IF NOT FOUND THEN RAISE EXCEPTION USING ERRCODE='23503',MESSAGE='candidate_index_private_authority_missing'; END IF;
        source_row.authority_id:=evidence.reference_id; source_row.reference_id:=evidence.reference_id;
        source_row.resume_version_id:=p_upstream_id; source_row.source_version:=evidence.version;
        source_row.candidate_id:=evidence.candidate_id; source_row.content_kind:=p_content_kind;
        source_row.source_observed_at:=evidence.source_observed_at;
        IF p_content_kind='pinned_text' AND evidence.extracted_text_sha256 IS NOT NULL THEN
            source_row.professional_text:=p_content; source_row.source_hash:=evidence.extracted_text_sha256;
        ELSIF p_content_kind='original_bytes' AND evidence.extracted_text_sha256 IS NULL THEN
            IF octet_length(p_content)>6990508 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_original_size'; END IF;
            source_row.original_bytes:=decode(p_content,'base64'); source_row.source_hash:=evidence.content_sha256;
            IF octet_length(source_row.original_bytes)<>evidence.byte_count
              OR replace(encode(source_row.original_bytes,'base64'),E'\n','')<>p_content THEN
                RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_original_size'; END IF;
        ELSE RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_content_kind_mismatch'; END IF;
    ELSE
        SELECT * INTO consent FROM public.candidate_consent_sources WHERE tenant_id=scope AND source_id=p_upstream_id;
        IF NOT FOUND THEN RAISE EXCEPTION USING ERRCODE='23503',MESSAGE='candidate_index_consent_authority_missing'; END IF;
        source_row.authority_id:=consent.subject_id; source_row.consent_subject_id:=consent.subject_id;
        source_row.consent_source_id:=consent.source_id; source_row.source_version:=consent.source_version;
        source_row.global_candidate_id:=consent.global_candidate_id; source_row.source_hash:=consent.profile_sha256;
        source_row.approved_profile:=consent.profile; source_row.content_kind:='approved_profile';
        source_row.resume_version_id:=(consent.resume->>'resume_version_id')::uuid;
        resume_manifest:=consent.resume;
        source_row.source_observed_at:=consent.approved_at;
        IF p_content IS NOT NULL OR p_content_kind<>'approved_profile' THEN
            RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_consent_content_override'; END IF;
    END IF;
    SELECT * INTO existing FROM public.candidate_index_sources WHERE source_id=p_upstream_id;
    IF FOUND THEN
        IF existing.scope_key<>scope OR existing.command_digest<>p_command_digest OR existing.source_hash<>source_row.source_hash
          OR existing.source_kind<>p_kind OR existing.professional_text IS DISTINCT FROM source_row.professional_text
          OR existing.original_bytes IS DISTINCT FROM source_row.original_bytes THEN
            RAISE EXCEPTION USING ERRCODE='23505',MESSAGE='candidate_index_source_conflict'; END IF;
    ELSE
        source_row.created_at:=clock_timestamp();
        INSERT INTO public.candidate_index_sources SELECT (source_row).*;
    END IF;
    -- Source insertion is rolled back if current authority refuses. It cannot
    -- become a partial ingestion or a replacement grant.
    authorized:=public.candidate_index_status(p_upstream_id);
    IF authorized IS NULL OR (authorized->>'eligible')::boolean IS DISTINCT FROM true THEN
        -- An authorised replacement has not yet updated the head. Its consent
        -- source/state tuple was independently checked by status before this code.
        IF NOT (p_kind='candidate_consent' AND authorized->>'reason'='superseded'
          AND EXISTS(SELECT 1 FROM public.candidate_consent_state c WHERE c.tenant_id=scope
            AND c.subject_id=source_row.authority_id AND c.active_source_id=p_upstream_id AND c.effective_action='grant')) THEN
            RAISE EXCEPTION USING ERRCODE='42501',MESSAGE='candidate_index_privacy_or_authority_refused'; END IF;
    END IF;
    PERFORM pg_advisory_xact_lock(hashtextextended('candidate-index-head:'||scope||':'||source_row.authority_id::text,0));
    SELECT generation_id INTO gen FROM public.candidate_index_generations WHERE source_id=p_upstream_id AND policy_sha256=policy_hash;
    IF FOUND THEN RETURN jsonb_build_object('source_id',p_upstream_id,'generation_id',gen,'outcome','replayed'); END IF;
    SELECT * INTO h FROM public.candidate_index_heads WHERE scope_key=scope AND authority_id=source_row.authority_id FOR UPDATE;
    gen:=gen_random_uuid();
    manifest:=jsonb_build_object('source_id',p_upstream_id,'source_version',source_row.source_version,
      'source_hash',source_row.source_hash,'resume_version_id',source_row.resume_version_id,
      'resume',resume_manifest);
    INSERT INTO public.candidate_index_generations(generation_id,scope_key,authority_id,source_id,generation,
      source_manifest,input_sha256,policy,policy_sha256)
      VALUES(gen,scope,source_row.authority_id,p_upstream_id,coalesce(h.logical_generation,0)+1,manifest,
        encode(sha256(convert_to(manifest::text,'UTF8')),'hex'),v_policy,policy_hash);
    IF h.authority_source_id IS NOT NULL AND h.authority_source_id<>p_upstream_id AND h.published_generation_id IS NOT NULL THEN
        INSERT INTO public.candidate_index_publication_events(event_id,scope_key,authority_id,outcome,
          previous_generation_id,authority_source_id,reason_code)
          VALUES(gen_random_uuid(),scope,source_row.authority_id,'invalidate',h.published_generation_id,h.authority_source_id,'consent_changed');
    END IF;
    INSERT INTO public.candidate_index_heads(scope_key,authority_id,source_kind,desired_generation_id,logical_generation,authority_source_id)
      VALUES(scope,source_row.authority_id,p_kind,gen,coalesce(h.logical_generation,0)+1,p_upstream_id)
    ON CONFLICT(scope_key,authority_id) DO UPDATE SET desired_generation_id=EXCLUDED.desired_generation_id,
      logical_generation=EXCLUDED.logical_generation,authority_source_id=EXCLUDED.authority_source_id,
      published_generation_id=CASE WHEN candidate_index_heads.authority_source_id=EXCLUDED.authority_source_id
        AND (SELECT g.policy->'embedding_model_id'=v_policy->'embedding_model_id'
          AND g.policy->'embedding_artifact_revision'=v_policy->'embedding_artifact_revision'
          AND g.policy->'dimension'=v_policy->'dimension'
          FROM public.candidate_index_generations g WHERE g.generation_id=candidate_index_heads.published_generation_id)
        THEN candidate_index_heads.published_generation_id ELSE NULL END,
      last_complete_generation_id=CASE WHEN candidate_index_heads.authority_source_id=EXCLUDED.authority_source_id THEN candidate_index_heads.last_complete_generation_id ELSE NULL END,
      updated_at=clock_timestamp();
    state_name:='pending_extraction';
    IF p_kind='candidate_consent' AND source_row.resume_version_id IS NOT NULL AND NOT EXISTS(
      SELECT 1 FROM public.candidate_index_sources r WHERE r.source_id=source_row.resume_version_id
        AND r.source_kind='organization_application' AND r.scope_key='org_'||(resume_manifest->>'organization_id')
    ) THEN state_name:='waiting_source'; END IF;
    PERFORM pg_advisory_xact_lock(hashtextextended('candidate-index-admission',0));
    IF state_name='pending_extraction' AND ((SELECT count(DISTINCT generation_id) FROM public.candidate_index_jobs WHERE scope_key=scope
      AND state IN ('pending_extraction','extracting','pending_embedding','embedding'))>=p_scope_limit OR
      (SELECT count(DISTINCT generation_id) FROM public.candidate_index_jobs WHERE state IN ('pending_extraction','extracting','pending_embedding','embedding'))>=p_total_limit)
      THEN state_name:='waiting_admission'; END IF;
    INSERT INTO public.candidate_index_jobs(job_id,scope_key,generation_id,stage,priority_class,state)
      VALUES(gen_random_uuid(),scope,gen,'extract',p_priority,state_name);
    RETURN jsonb_build_object('source_id',p_upstream_id,'generation_id',gen,'outcome','accepted');
END;
$$;

CREATE FUNCTION public.candidate_index_complete_extract(
    p_job_id uuid,p_token uuid,p_generation bigint,p_input_sha256 text,p_policy_sha256 text,
    p_namespace jsonb,p_evidence jsonb,p_confidence double precision,p_model_id text,
    p_chunks jsonb,p_professional_text text,p_parsed_text text DEFAULT NULL
) RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE original_scope text:=current_setting('app.current_tenant_id',true); j public.candidate_index_jobs%ROWTYPE;
    g public.candidate_index_generations%ROWTYPE; s public.candidate_index_sources%ROWTYPE;
    allowed jsonb; c jsonb; evidence jsonb; ordinal integer:=0; assembled text:=''; input_text text;
    eid uuid:=gen_random_uuid(); deterministic boolean; extracted_hash text;
BEGIN
    SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id AND stage='extract';
    IF NOT FOUND THEN RETURN false; END IF;
    SELECT * INTO g FROM public.candidate_index_generations WHERE generation_id=j.generation_id;
    SELECT * INTO s FROM public.candidate_index_sources WHERE source_id=g.source_id;
    IF s.source_kind='approved_provider' THEN RETURN false; END IF;
    PERFORM set_config('app.current_tenant_id',j.scope_key,true);
    allowed:=public.candidate_index_status(s.source_id);
    SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id FOR UPDATE;
    IF (allowed->>'eligible')::boolean IS DISTINCT FROM true OR (allowed->>'generation_id')::uuid IS DISTINCT FROM j.generation_id
      OR j.state<>'extracting' OR j.lease_token IS DISTINCT FROM p_token OR j.lease_generation IS DISTINCT FROM p_generation
      OR j.lease_expires_at<=clock_timestamp() OR j.dispatch_reserved_generation<>j.lease_generation
      OR p_input_sha256 IS DISTINCT FROM g.input_sha256 OR p_policy_sha256 IS DISTINCT FROM g.policy_sha256 THEN
        PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN false; END IF;
    deterministic:=s.source_kind='candidate_consent' AND s.resume_version_id IS NULL;
    IF (jsonb_typeof(p_namespace)='object' AND octet_length(p_namespace::text) BETWEEN 2 AND 65536
      AND jsonb_typeof(p_evidence)='array' AND jsonb_array_length(p_evidence) BETWEEN 1 AND 32
      AND p_confidence BETWEEN (g.policy->>'minimum_confidence')::double precision AND 1
      AND jsonb_typeof(p_chunks)='array' AND jsonb_array_length(p_chunks) BETWEEN 1 AND (g.policy->>'max_chunks')::integer
      AND octet_length(p_professional_text) BETWEEN 1 AND 2097152
      AND p_model_id=CASE WHEN deterministic THEN 'deterministic-profile:v1' WHEN j.attempts=1 THEN g.policy->>'primary_model_id'
        WHEN j.attempts=2 THEN g.policy->>'fallback_model_id' END) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_extraction_incomplete'; END IF;
    input_text:=CASE WHEN deterministic THEN s.approved_profile::text ELSE s.professional_text END;
    IF s.source_kind='candidate_consent' AND s.resume_version_id IS NOT NULL THEN
        SELECT r.professional_text INTO input_text FROM public.candidate_index_sources r
          WHERE r.source_id=s.resume_version_id AND r.source_kind='organization_application'
            AND r.scope_key='org_'||(g.source_manifest->'resume'->>'organization_id');
    END IF;
    IF input_text IS NULL THEN
        IF (octet_length(p_parsed_text) BETWEEN 1 AND 2097152) IS NOT TRUE THEN
            RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_parsed_text_missing'; END IF;
        input_text:=p_parsed_text;
    ELSIF p_parsed_text IS NOT NULL THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_parsed_text_override';
    END IF;
    FOR evidence IN SELECT value FROM jsonb_array_elements(p_evidence) LOOP
        IF (jsonb_typeof(evidence)='object' AND evidence ?& ARRAY['field','text']
          AND evidence-ARRAY['field','text']='{}'::jsonb
          AND evidence->>'field' IN ('current_title','primary_titles','skills_raw','skills_normalized','domains','functions',
            'certifications','industries','primary_skills','recent_job_titles','headline','skills')
          AND jsonb_typeof(evidence->'text')='string' AND length(evidence->>'text') BETWEEN 2 AND 512
          AND strpos(lower(input_text),lower(evidence->>'text'))>0) IS NOT TRUE THEN
            RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_professional_evidence_missing'; END IF;
    END LOOP;
    FOR c IN SELECT value FROM jsonb_array_elements(p_chunks) LOOP
        IF (jsonb_typeof(c)='object' AND c ?& ARRAY['ordinal','text','sha256'] AND c-ARRAY['ordinal','text','sha256']='{}'::jsonb
          AND c->'ordinal'=to_jsonb(ordinal) AND jsonb_typeof(c->'text')='string'
          AND length(c->>'text') BETWEEN 1 AND (g.policy->>'chunk_characters')::integer
          AND c->>'sha256'=encode(sha256(convert_to(c->>'text','UTF8')),'hex')) IS NOT TRUE THEN
            RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_chunk_manifest_invalid'; END IF;
        assembled:=assembled||(c->>'text'); ordinal:=ordinal+1;
    END LOOP;
    IF assembled IS DISTINCT FROM p_professional_text THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_chunk_loss'; END IF;
    extracted_hash:=encode(sha256(convert_to(jsonb_build_object('namespace',p_namespace,'evidence',p_evidence,
      'confidence',p_confidence,'model_id',p_model_id,'chunks',p_chunks)::text,'UTF8')),'hex');
    INSERT INTO public.candidate_index_extractions(extraction_id,scope_key,generation_id,input_sha256,extraction_sha256,
      namespace,evidence,confidence,model_id,attempt,chunks)
      VALUES(eid,j.scope_key,j.generation_id,g.input_sha256,extracted_hash,p_namespace,p_evidence,p_confidence,p_model_id,j.attempts,p_chunks);
    UPDATE public.candidate_index_jobs SET state='pending_embedding',lease_token=NULL,lease_expires_at=NULL,
      error_code=NULL,updated_at=clock_timestamp() WHERE job_id=p_job_id;
    INSERT INTO public.candidate_index_jobs(job_id,scope_key,generation_id,stage,priority_class,state)
      VALUES(gen_random_uuid(),j.scope_key,j.generation_id,'embed',j.priority_class,'pending_embedding');
    PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN true;
END;
$$;

CREATE FUNCTION public.candidate_index_complete_embed(
    p_job_id uuid,p_token uuid,p_generation bigint,p_extraction_id uuid,p_policy_sha256 text,
    p_model_id text,p_artifact_revision text,p_vectors jsonb
) RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE original_scope text:=current_setting('app.current_tenant_id',true); j public.candidate_index_jobs%ROWTYPE;
    g public.candidate_index_generations%ROWTYPE; e public.candidate_index_extractions%ROWTYPE;
    h public.candidate_index_heads%ROWTYPE; allowed jsonb; v jsonb; ordinal integer:=0;
BEGIN
    SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id AND stage='embed';
    IF NOT FOUND THEN RETURN false; END IF;
    SELECT * INTO g FROM public.candidate_index_generations WHERE generation_id=j.generation_id;
    PERFORM set_config('app.current_tenant_id',j.scope_key,true);
    allowed:=public.candidate_index_status(g.source_id);
    PERFORM pg_advisory_xact_lock(hashtextextended('candidate-index-head:'||j.scope_key||':'||g.authority_id::text,0));
    SELECT * INTO h FROM public.candidate_index_heads WHERE scope_key=j.scope_key AND authority_id=g.authority_id FOR UPDATE;
    SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id FOR UPDATE;
    IF (allowed->>'eligible')::boolean IS DISTINCT FROM true OR h.desired_generation_id IS DISTINCT FROM j.generation_id
      OR h.source_kind='approved_provider' OR j.state<>'embedding' OR j.lease_token IS DISTINCT FROM p_token
      OR j.lease_generation IS DISTINCT FROM p_generation OR j.lease_expires_at<=clock_timestamp()
      OR j.dispatch_reserved_generation<>j.lease_generation OR j.attempts<1
      OR g.policy_sha256 IS DISTINCT FROM p_policy_sha256 THEN
        PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN false; END IF;
    SELECT * INTO e FROM public.candidate_index_extractions WHERE extraction_id=p_extraction_id AND generation_id=j.generation_id;
    IF NOT FOUND OR p_model_id IS DISTINCT FROM g.policy->>'embedding_model_id'
      OR p_artifact_revision IS DISTINCT FROM g.policy->>'embedding_artifact_revision'
      OR (jsonb_typeof(p_vectors)='array' AND jsonb_array_length(p_vectors)=jsonb_array_length(e.chunks)
        AND octet_length(p_vectors::text)<=524288) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_vector_manifest_invalid'; END IF;
    FOR v IN SELECT value FROM jsonb_array_elements(p_vectors) LOOP
        IF (jsonb_typeof(v)='array' AND jsonb_array_length(v)=384
          AND NOT EXISTS(SELECT 1 FROM jsonb_array_elements(v) n WHERE jsonb_typeof(n)<>'number'
            OR abs(n::text::numeric)>1)) IS NOT TRUE THEN
            RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_vector_invalid'; END IF;
        INSERT INTO public.candidate_index_vectors(scope_key,generation_id,extraction_id,ordinal,chunk_sha256,model_id,artifact_revision,embedding)
          VALUES(j.scope_key,j.generation_id,e.extraction_id,ordinal,e.chunks->ordinal->>'sha256',p_model_id,p_artifact_revision,v::text::public.vector(384));
        ordinal:=ordinal+1;
    END LOOP;
    INSERT INTO public.candidate_index_publication_events(event_id,scope_key,authority_id,outcome,previous_generation_id,
      generation_id,authority_source_id,reason_code) VALUES(gen_random_uuid(),j.scope_key,g.authority_id,
      CASE WHEN h.published_generation_id IS NULL THEN 'ready' ELSE 'switch' END,h.published_generation_id,j.generation_id,g.source_id,'complete');
    UPDATE public.candidate_index_heads SET published_generation_id=j.generation_id,last_complete_generation_id=j.generation_id,
      updated_at=clock_timestamp() WHERE scope_key=j.scope_key AND authority_id=g.authority_id;
    UPDATE public.candidate_index_jobs SET state='ready',lease_token=NULL,lease_expires_at=NULL,error_code=NULL,
      updated_at=clock_timestamp() WHERE generation_id=j.generation_id;
    PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN true;
END;
$$;

CREATE FUNCTION public.candidate_index_fail(p_job_id uuid,p_token uuid,p_generation bigint,p_reason text,p_retry_ms integer)
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE original_scope text:=current_setting('app.current_tenant_id',true); j public.candidate_index_jobs%ROWTYPE;
    g public.candidate_index_generations%ROWTYPE; allowed jsonb; next_state text;
BEGIN
    IF (p_retry_ms BETWEEN 0 AND 120000 AND p_reason IN ('privacy_restricted','privacy_review','privacy_unavailable',
      'source_missing','consent_changed','policy_mismatch','low_confidence','unsupported_format','incomplete','chunk_overflow',
      'provider_timeout','provider_unavailable','rate_limited','dispatch_exhausted')) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_failure_bounds'; END IF;
    SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id;
    IF NOT FOUND THEN RETURN false; END IF;
    SELECT * INTO g FROM public.candidate_index_generations WHERE generation_id=j.generation_id;
    IF EXISTS(SELECT 1 FROM public.candidate_index_sources WHERE source_id=g.source_id AND source_kind='approved_provider') THEN RETURN false; END IF;
    PERFORM set_config('app.current_tenant_id',j.scope_key,true);
    allowed:=public.candidate_index_status(g.source_id);
    SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id FOR UPDATE;
    IF j.state NOT IN ('extracting','embedding') OR j.lease_token IS DISTINCT FROM p_token
      OR j.lease_generation IS DISTINCT FROM p_generation OR j.lease_expires_at<=clock_timestamp() THEN
        PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN false; END IF;
    IF (allowed->>'eligible')::boolean IS DISTINCT FROM true THEN
        next_state:='quarantined'; p_reason:=coalesce(allowed->>'reason','privacy_unavailable');
    ELSIF (allowed->>'generation_id')::uuid IS DISTINCT FROM j.generation_id THEN next_state:='superseded'; p_reason:='superseded';
    ELSIF p_reason IN ('privacy_restricted','consent_changed','policy_mismatch') THEN next_state:='quarantined';
    ELSIF p_reason IN ('low_confidence','incomplete') AND j.stage='extract' AND j.attempts=1
      AND j.dispatch_deadline>clock_timestamp()+p_retry_ms*interval '1 millisecond'
      THEN next_state:='pending_extraction';
    ELSIF p_reason IN ('low_confidence','unsupported_format','incomplete','chunk_overflow') THEN next_state:='needs_review';
    ELSIF p_reason='source_missing' THEN next_state:='waiting_source';
    ELSIF p_reason='dispatch_exhausted' OR j.attempts>=(CASE j.stage WHEN 'extract' THEN 2 ELSE 3 END)
      OR j.dispatch_deadline<=clock_timestamp()+p_retry_ms*interval '1 millisecond' THEN next_state:='failed';
    ELSE next_state:=CASE j.stage WHEN 'extract' THEN 'pending_extraction' ELSE 'pending_embedding' END;
    END IF;
    UPDATE public.candidate_index_jobs SET state=next_state,error_code=p_reason,lease_token=NULL,lease_expires_at=NULL,
      next_attempt_at=clock_timestamp()+p_retry_ms*interval '1 millisecond',updated_at=clock_timestamp() WHERE job_id=p_job_id;
    PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN true;
END;
$$;

CREATE FUNCTION public.candidate_index_invalidate(p_source_id uuid) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE s public.candidate_index_sources%ROWTYPE; h public.candidate_index_heads%ROWTYPE; allowed jsonb; reason text;
BEGIN
    SELECT * INTO s FROM public.candidate_index_sources WHERE source_id=p_source_id
      AND scope_key=current_setting('app.current_tenant_id',true) AND source_kind<>'approved_provider';
    IF NOT FOUND THEN RETURN false; END IF;
    allowed:=public.candidate_index_status(p_source_id);
    IF (allowed->>'eligible')::boolean IS NOT DISTINCT FROM true THEN RETURN false; END IF;
    reason:=CASE WHEN allowed->>'reason' IN ('consent_changed','superseded') THEN allowed->>'reason' ELSE 'privacy_restricted' END;
    PERFORM pg_advisory_xact_lock(hashtextextended('candidate-index-head:'||s.scope_key||':'||s.authority_id::text,0));
    SELECT * INTO h FROM public.candidate_index_heads WHERE scope_key=s.scope_key AND authority_id=s.authority_id FOR UPDATE;
    IF h.authority_source_id=s.source_id AND h.published_generation_id IS NOT NULL THEN
        INSERT INTO public.candidate_index_publication_events(event_id,scope_key,authority_id,outcome,
          previous_generation_id,authority_source_id,reason_code)
          VALUES(gen_random_uuid(),s.scope_key,s.authority_id,'invalidate',h.published_generation_id,s.source_id,reason);
        UPDATE public.candidate_index_heads SET published_generation_id=NULL,last_complete_generation_id=NULL,updated_at=clock_timestamp()
          WHERE scope_key=s.scope_key AND authority_id=s.authority_id;
    END IF;
    UPDATE public.candidate_index_jobs j SET state=CASE WHEN reason='superseded' THEN 'superseded' ELSE 'cancelled' END,
      error_code=CASE WHEN reason='superseded' THEN reason ELSE 'privacy_restricted' END,
      lease_token=NULL,lease_expires_at=NULL,updated_at=clock_timestamp()
      FROM public.candidate_index_generations g WHERE g.generation_id=j.generation_id AND g.source_id=s.source_id
        AND j.state NOT IN ('cancelled','superseded');
    RETURN true;
END;
$$;

CREATE FUNCTION public.candidate_index_read_private(p_query public.vector,p_model_id text,p_revision text,p_limit integer,p_job_id integer DEFAULT NULL)
RETURNS SETOF jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE scope text:=current_setting('app.current_tenant_id',true); item record; allowed jsonb; emitted integer:=0;
BEGIN
    IF (scope ~ '^org_[1-9][0-9]*$' AND p_limit BETWEEN 1 AND 100 AND vector_dims(p_query)=384
      AND (p_query <#> p_query) BETWEEN -1.001 AND -0.999 AND octet_length(p_model_id) BETWEEN 1 AND 200
      AND p_revision ~ '^([0-9a-f]{40}|[0-9a-f]{64})$' AND (p_job_id IS NULL OR p_job_id>0)) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_read_bounds'; END IF;
    FOR item IN SELECT s.source_id,s.source_observed_at,s.reference_id,s.resume_version_id,r.application_id,r.job_id,
      g.generation_id,g.generation,h.desired_generation_id,min(v.embedding <=> p_query) distance
      FROM public.candidate_index_heads h JOIN public.candidate_index_generations g ON g.generation_id=h.published_generation_id
      JOIN public.candidate_index_sources s ON s.source_id=g.source_id AND s.source_id=h.authority_source_id
      JOIN public.organization_candidate_references r ON r.tenant_id=s.tenant_id AND r.reference_id=s.reference_id AND r.candidate_id=s.candidate_id
      JOIN public.candidate_index_vectors v ON v.generation_id=g.generation_id
      WHERE h.scope_key=scope AND s.source_kind='organization_application'
        AND g.policy->>'embedding_model_id'=p_model_id AND g.policy->>'embedding_artifact_revision'=p_revision
        AND (p_job_id IS NULL OR r.job_id=p_job_id)
      GROUP BY s.source_id,r.application_id,r.job_id,g.generation_id,h.desired_generation_id
      ORDER BY distance,r.application_id,g.generation_id
    LOOP
        allowed:=public.candidate_index_status(item.source_id);
        IF (allowed->>'eligible')::boolean IS DISTINCT FROM true THEN CONTINUE; END IF;
        RETURN NEXT jsonb_build_object('reference_id',item.reference_id,'resume_version_id',item.resume_version_id,
          'application_id',item.application_id,'job_id',item.job_id,'generation_id',item.generation_id,'generation',item.generation,
          'source_observed_at',item.source_observed_at,'score',1-item.distance,
          'state',CASE WHEN item.generation_id=item.desired_generation_id THEN 'ready'
            WHEN allowed->>'state' IN ('failed','needs_review','quarantined') THEN 'refresh_failed' ELSE 'updating' END,
          'highlights',(SELECT jsonb_agg(left(c->>'text',240)) FROM public.candidate_index_extractions e,
            LATERAL jsonb_array_elements(e.chunks) WITH ORDINALITY AS a(c,n)
            WHERE e.generation_id=item.generation_id AND n<=3));
        emitted:=emitted+1; EXIT WHEN emitted>=p_limit;
    END LOOP;
END;
$$;

-- No runtime grant and no HTTP caller in 4D. This internal proof/read exposes
-- only approved consent professional data; never tenant resume content/tokens.
CREATE FUNCTION public.candidate_index_read_public(p_limit integer) RETURNS SETOF jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE original_scope text:=current_setting('app.current_tenant_id',true); item record; allowed jsonb; n integer:=0;
BEGIN
    IF p_limit IS NULL OR p_limit NOT BETWEEN 1 AND 100 THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_read_bounds'; END IF;
    FOR item IN SELECT s.*,g.generation_id FROM public.candidate_index_heads h
      JOIN public.candidate_index_generations g ON g.generation_id=h.published_generation_id
      JOIN public.candidate_index_sources s ON s.source_id=g.source_id AND s.source_id=h.authority_source_id
      WHERE s.source_kind='candidate_consent' ORDER BY s.global_candidate_id,g.generation_id
    LOOP
        PERFORM set_config('app.current_tenant_id',item.scope_key,true);
        allowed:=public.candidate_index_status(item.source_id);
        IF (allowed->>'eligible')::boolean IS DISTINCT FROM true THEN CONTINUE; END IF;
        RETURN NEXT jsonb_build_object('global_candidate_id',item.global_candidate_id,'generation_id',item.generation_id,
          'approved_profile',item.approved_profile,'source_observed_at',item.source_observed_at);
        n:=n+1; EXIT WHEN n>=p_limit;
    END LOOP;
    PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true);
END;
$$;

CREATE FUNCTION public.candidate_index_catchup(p_limit integer,p_maintenance boolean,p_scope_limit integer,p_total_limit integer)
RETURNS integer LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE original_scope text:=current_setting('app.current_tenant_id',true); item record; allowed jsonb; n integer:=0;
BEGIN
    IF (p_limit BETWEEN 1 AND CASE WHEN p_maintenance THEN 10 ELSE 100 END
      AND p_scope_limit BETWEEN 1 AND 1000 AND p_total_limit BETWEEN p_scope_limit AND 10000) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_catchup_bounds'; END IF;
    -- Only existing authority/generation jobs. Never fabricate a grant, source
    -- observation or historical event; never reset a spent model attempt.
    FOR item IN SELECT j.job_id,j.scope_key,g.source_id,s.resume_version_id,g.source_manifest FROM public.candidate_index_jobs j
      JOIN public.candidate_index_generations g ON g.generation_id=j.generation_id
      JOIN public.candidate_index_sources s ON s.source_id=g.source_id
      WHERE j.state IN ('waiting_source','waiting_admission') AND j.stage='extract'
        AND s.source_kind IN ('organization_application','candidate_consent')
        AND j.priority_class=CASE WHEN p_maintenance THEN 'maintenance' ELSE 'interactive' END
      ORDER BY j.next_attempt_at,j.job_id LIMIT p_limit
    LOOP
        PERFORM set_config('app.current_tenant_id',item.scope_key,true);
        allowed:=public.candidate_index_status(item.source_id);
        IF (allowed->>'eligible')::boolean IS DISTINCT FROM true THEN CONTINUE; END IF;
        IF item.scope_key LIKE 'candidate_%' AND item.resume_version_id IS NOT NULL AND NOT EXISTS(
          SELECT 1 FROM public.candidate_index_sources r WHERE r.source_id=item.resume_version_id
            AND r.source_kind='organization_application' AND r.scope_key='org_'||(item.source_manifest->'resume'->>'organization_id')) THEN CONTINUE; END IF;
        PERFORM pg_advisory_xact_lock(hashtextextended('candidate-index-admission',0));
        IF (SELECT count(DISTINCT generation_id) FROM public.candidate_index_jobs WHERE state IN ('pending_extraction','extracting','pending_embedding','embedding')
          AND scope_key=item.scope_key)>=p_scope_limit OR
          (SELECT count(DISTINCT generation_id) FROM public.candidate_index_jobs WHERE state IN ('pending_extraction','extracting','pending_embedding','embedding'))>=p_total_limit THEN CONTINUE; END IF;
        UPDATE public.candidate_index_jobs SET state='pending_extraction',error_code=NULL,updated_at=clock_timestamp()
          WHERE job_id=item.job_id AND state IN ('waiting_source','waiting_admission');
        IF FOUND THEN n:=n+1; END IF;
    END LOOP;
    PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN n;
END;
$$;

CREATE FUNCTION public.candidate_index_key_versions() RETURNS SETOF integer
LANGUAGE sql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
    SELECT DISTINCT k FROM public.candidate_index_sources s CROSS JOIN LATERAL unnest(s.key_versions) k ORDER BY k
$$;

REVOKE ALL ON FUNCTION public.candidate_index_status(uuid,boolean) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_capture_source(text,uuid,text,text,text,jsonb,integer[],text,text,integer,integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_claim(text,integer,jsonb,uuid,uuid,bigint,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_complete_extract(uuid,uuid,bigint,text,text,jsonb,jsonb,double precision,text,jsonb,text,text) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_complete_embed(uuid,uuid,bigint,uuid,text,text,text,jsonb) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_fail(uuid,uuid,bigint,text,integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_invalidate(uuid) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_read_private(public.vector,text,text,integer,integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_read_public(integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_catchup(integer,boolean,integer,integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_key_versions() FROM PUBLIC;
