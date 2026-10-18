-- Forward-only, durable medium -> low rescue after confirmed truncation.
-- Existing two-attempt reservation and generation deadline are unchanged.
-- Legacy policies keep their existing primary/fallback path and reason codes.
ALTER TABLE public.candidate_index_jobs DROP CONSTRAINT candidate_index_jobs_error_code_check;
ALTER TABLE public.candidate_index_jobs ADD CONSTRAINT candidate_index_jobs_error_code_check
 CHECK (error_code IN ('privacy_restricted','privacy_review','privacy_unavailable','source_missing',
 'consent_changed','policy_mismatch','low_confidence','unsupported_format','incomplete','chunk_overflow',
 'provider_timeout','provider_unavailable','rate_limited','attempts_exhausted','dispatch_exhausted',
 'superseded','output_truncated'));

CREATE OR REPLACE FUNCTION public.candidate_index_claim(
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
          'retry_mode',CASE WHEN g.policy->>'extraction_adapter_version' ~ '^candidate-index-extraction:v2:[0-9a-f]{64}$' AND j.attempts=2 AND j.error_code='output_truncated'
            THEN 'low' ELSE 'medium' END,
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

CREATE OR REPLACE FUNCTION public.candidate_index_complete_extract(
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
        WHEN j.attempts=2 AND g.policy->>'extraction_adapter_version' ~ '^candidate-index-extraction:v2:[0-9a-f]{64}$' AND j.error_code='output_truncated'
          THEN g.policy->>'primary_model_id'
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

CREATE OR REPLACE FUNCTION public.candidate_index_fail(p_job_id uuid,p_token uuid,p_generation bigint,p_reason text,p_retry_ms integer)
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE original_scope text:=current_setting('app.current_tenant_id',true); j public.candidate_index_jobs%ROWTYPE;
    g public.candidate_index_generations%ROWTYPE; allowed jsonb; next_state text;
BEGIN
    IF (p_retry_ms BETWEEN 0 AND 120000 AND p_reason IN ('privacy_restricted','privacy_review','privacy_unavailable',
      'source_missing','consent_changed','policy_mismatch','low_confidence','unsupported_format','incomplete','output_truncated','chunk_overflow',
      'provider_timeout','provider_unavailable','rate_limited','dispatch_exhausted')) IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_failure_bounds'; END IF;
    SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job_id;
    IF NOT FOUND THEN RETURN false; END IF;
    SELECT * INTO g FROM public.candidate_index_generations WHERE generation_id=j.generation_id;
    IF EXISTS(SELECT 1 FROM public.candidate_index_sources WHERE source_id=g.source_id AND source_kind='approved_provider') THEN RETURN false; END IF;
    IF p_reason='output_truncated' AND (j.stage='extract' AND g.policy->>'extraction_adapter_version' ~ '^candidate-index-extraction:v2:[0-9a-f]{64}$') IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_failure_bounds'; END IF;
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
    ELSIF p_reason IN ('low_confidence','incomplete','output_truncated') AND j.stage='extract' AND j.attempts=1
      AND j.dispatch_deadline>clock_timestamp()+p_retry_ms*interval '1 millisecond'
      THEN next_state:='pending_extraction';
    ELSIF p_reason IN ('low_confidence','unsupported_format','incomplete','output_truncated','chunk_overflow') THEN next_state:='needs_review';
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

