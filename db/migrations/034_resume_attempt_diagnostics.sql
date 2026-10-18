-- No recovery/backfill, deletion, model call or historical row rewrite.
CREATE TABLE public.candidate_index_attempt_diagnostics (
 scope_key text NOT NULL,
 job_id uuid NOT NULL REFERENCES public.candidate_index_jobs(job_id) ON DELETE RESTRICT,
 generation_id uuid NOT NULL,
 lease_generation bigint NOT NULL CHECK(lease_generation>0),
 attempt integer NOT NULL CHECK(attempt BETWEEN 1 AND 2),
 event text NOT NULL CHECK(event IN ('reserved','input_checked','dispatch_attempted','response_received','validation_finished','settled')),
 payload jsonb NOT NULL CHECK(jsonb_typeof(payload)='object' AND octet_length(payload::text)<=2048),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(job_id,lease_generation,event),
 FOREIGN KEY(scope_key,generation_id) REFERENCES public.candidate_index_generations(scope_key,generation_id) ON DELETE RESTRICT
);
CREATE TABLE public.candidate_index_recovery_receipts (
 plan_hash text PRIMARY KEY CHECK(plan_hash ~ '^[0-9a-f]{64}$'),
 scope_key text NOT NULL CHECK(scope_key ~ '^org_[1-9][0-9]{0,9}$'),
 target_policy_sha256 text NOT NULL CHECK(target_policy_sha256 ~ '^[0-9a-f]{64}$'),
 members jsonb NOT NULL CHECK(jsonb_typeof(members)='array' AND jsonb_array_length(members) BETWEEN 1 AND 100 AND octet_length(members::text)<=65536),
 max_calls integer NOT NULL CHECK(max_calls BETWEEN 0 AND 200),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
DO $ddl$
DECLARE t text;
BEGIN
 FOREACH t IN ARRAY ARRAY['candidate_index_attempt_diagnostics','candidate_index_recovery_receipts'] LOOP
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',t);
  EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY',t);
  -- No runtime table reads: owner-only guarded read functions check live privacy.
  EXECUTE format('CREATE POLICY index_owner_coordination ON public.%I FOR ALL TO %I USING (true) WITH CHECK (true)',t,current_user);
  EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC',t);
  EXECUTE format('CREATE TRIGGER index_evidence_no_mutation BEFORE UPDATE OR DELETE ON public.%I FOR EACH ROW EXECUTE FUNCTION public.candidate_index_append_only()',t);
  EXECUTE format('CREATE TRIGGER index_evidence_no_truncate BEFORE TRUNCATE ON public.%I FOR EACH STATEMENT EXECUTE FUNCTION public.candidate_index_append_only()',t);
 END LOOP;
END $ddl$;

-- The shared append-only guard admits TRUNCATE only of an already-empty table, by
-- static per-table branches. Extend those branches to the two tables above; every
-- populated-table UPDATE/DELETE/TRUNCATE still raises. Owner/ACL are preserved.
CREATE OR REPLACE FUNCTION public.candidate_index_append_only() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
    IF TG_OP='TRUNCATE' THEN
        -- Static branches: no caller-selected relation and no row-security-off workaround.
        IF TG_TABLE_NAME='candidate_index_sources' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_sources) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_generations' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_generations) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_extractions' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_extractions) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_vectors' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_vectors) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_publication_events' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_publication_events) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_attempt_diagnostics' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_attempt_diagnostics) THEN RETURN NULL; END IF;
        IF TG_TABLE_NAME='candidate_index_recovery_receipts' AND NOT EXISTS(SELECT 1 FROM public.candidate_index_recovery_receipts) THEN RETURN NULL; END IF;
    END IF;
    RAISE EXCEPTION USING ERRCODE='55000',MESSAGE='candidate_index_append_only';
END;
$$;

CREATE FUNCTION public.candidate_index_record_diagnostic(p_job uuid,p_token uuid,p_lease bigint,p_event text,p_payload jsonb)
RETURNS boolean LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $fn$
DECLARE j public.candidate_index_jobs%ROWTYPE; g public.candidate_index_generations%ROWTYPE;
 allowed jsonb; old_scope text:=current_setting('app.current_tenant_id',true); pair record; existing jsonb;
BEGIN
 IF (p_job IS NOT NULL AND p_token IS NOT NULL AND p_lease>0
  AND p_event IN ('reserved','input_checked','dispatch_attempted','response_received','validation_finished','settled')
  AND jsonb_typeof(p_payload)='object' AND p_payload->'version'='1'::jsonb AND octet_length(p_payload::text)<=2048
  AND p_payload-ARRAY['version','reason','validation_reason','finish_reason','model_slot','reasoning_effort','elapsed_ms','input_characters','completion_tokens','skills_kept','skills_dropped','duplicates_removed']='{}'::jsonb) IS NOT TRUE
 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_diagnostic_bounds'; END IF;
 FOR pair IN SELECT key,value FROM jsonb_each(p_payload) LOOP
  IF pair.key='version' THEN CONTINUE; END IF;
  IF pair.key IN ('elapsed_ms','input_characters','completion_tokens','skills_kept','skills_dropped','duplicates_removed') THEN
   IF (jsonb_typeof(pair.value)='number' AND pair.value::text ~ '^(0|[1-9][0-9]{0,6})$'
    AND (pair.value::text)::integer<=CASE pair.key WHEN 'elapsed_ms' THEN 45000 WHEN 'input_characters' THEN 2097152
     WHEN 'completion_tokens' THEN 4096 WHEN 'skills_kept' THEN 100 ELSE 1000 END) IS NOT TRUE
   THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_diagnostic_metric'; END IF;
  ELSIF (jsonb_typeof(pair.value)='string' AND CASE pair.key
   WHEN 'reason' THEN pair.value#>>'{}' IN ('completed','incomplete','output_truncated','low_confidence','unsupported_format','chunk_overflow','provider_timeout','provider_unavailable','rate_limited','dispatch_exhausted','policy_mismatch','privacy_restricted','privacy_review','privacy_unavailable','source_missing','consent_changed','superseded','not_validated')
   WHEN 'finish_reason' THEN pair.value#>>'{}' IN ('stop','length','content_filter','tool_calls','unlisted','unavailable')
   WHEN 'model_slot' THEN pair.value#>>'{}' IN ('primary','backup')
   WHEN 'reasoning_effort' THEN pair.value#>>'{}' IN ('medium','low')
   WHEN 'validation_reason' THEN pair.value#>>'{}' IN ('not_validated','accepted','location_format','list_format','list_limit','years_format_or_limit','skill_length','expanded_list_limit','schema_invalid','namespace_bytes') ELSE false END) IS NOT TRUE
  THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_diagnostic_enum'; END IF;
 END LOOP;
 SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job;
 IF NOT FOUND THEN RETURN false; END IF;
 SELECT * INTO g FROM public.candidate_index_generations WHERE generation_id=j.generation_id;
 PERFORM set_config('app.current_tenant_id',j.scope_key,true);
 allowed:=public.candidate_index_status(g.source_id);
 SELECT * INTO j FROM public.candidate_index_jobs WHERE job_id=p_job FOR UPDATE;
 IF j.stage<>'extract' OR j.state<>'extracting' OR j.lease_token IS DISTINCT FROM p_token
  OR j.lease_generation<>p_lease OR j.dispatch_reserved_generation<>p_lease OR j.attempts NOT BETWEEN 1 AND 2
  OR j.lease_expires_at<=clock_timestamp() OR (allowed->>'eligible')::boolean IS DISTINCT FROM true
  OR (allowed->>'generation_id')::uuid IS DISTINCT FROM j.generation_id THEN
  PERFORM set_config('app.current_tenant_id',coalesce(old_scope,''),true); RETURN false;
 END IF;
 INSERT INTO public.candidate_index_attempt_diagnostics(scope_key,job_id,generation_id,lease_generation,attempt,event,payload)
 VALUES(j.scope_key,j.job_id,j.generation_id,p_lease,j.attempts,p_event,p_payload) ON CONFLICT DO NOTHING;
 SELECT payload INTO existing FROM public.candidate_index_attempt_diagnostics WHERE job_id=p_job AND lease_generation=p_lease AND event=p_event;
 IF existing IS DISTINCT FROM p_payload THEN RAISE EXCEPTION USING ERRCODE='23505',MESSAGE='candidate_index_diagnostic_conflict'; END IF;
 PERFORM set_config('app.current_tenant_id',coalesce(old_scope,''),true); RETURN true;
END $fn$;
REVOKE ALL ON FUNCTION public.candidate_index_record_diagnostic(uuid,uuid,bigint,text,jsonb) FROM PUBLIC;

CREATE FUNCTION public.candidate_index_read_diagnostics(p_scope text,p_generation uuid)
RETURNS SETOF jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $fn$
DECLARE source uuid; allowed jsonb; old_scope text:=current_setting('app.current_tenant_id',true);
BEGIN
 SELECT source_id INTO source FROM public.candidate_index_generations WHERE generation_id=p_generation AND scope_key=p_scope;
 IF NOT FOUND THEN RETURN; END IF;
 PERFORM set_config('app.current_tenant_id',p_scope,true);
 allowed:=public.candidate_index_status(source);
 IF (allowed->>'eligible')::boolean IS TRUE THEN
  RETURN QUERY SELECT jsonb_build_object('lease_generation',d.lease_generation,'attempt',d.attempt,'event',d.event,'payload',d.payload,'created_at',d.created_at)
   FROM public.candidate_index_attempt_diagnostics d WHERE d.scope_key=p_scope AND d.generation_id=p_generation ORDER BY d.lease_generation,d.created_at,d.event LIMIT 12;
 END IF;
 PERFORM set_config('app.current_tenant_id',coalesce(old_scope,''),true);
END $fn$;
REVOKE ALL ON FUNCTION public.candidate_index_read_diagnostics(text,uuid) FROM PUBLIC;

CREATE FUNCTION public.candidate_index_plan_recovery(p_scope text,p_pairs jsonb,p_policy text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $fn$
DECLARE pair jsonb; s public.candidate_index_sources%ROWTYPE; g public.candidate_index_generations%ROWTYPE;
 j public.candidate_index_jobs%ROWTYPE; allowed jsonb; items jsonb:='[]'; target text; existing uuid;
 old_scope text:=current_setting('app.current_tenant_id',true); at_time timestamptz:=clock_timestamp(); policy jsonb; new_count integer:=0;
BEGIN
 IF (p_scope ~ '^org_[1-9][0-9]{0,9}$' AND jsonb_typeof(p_pairs)='array' AND jsonb_array_length(p_pairs) BETWEEN 1 AND 100
  AND octet_length(p_pairs::text)<=32768 AND octet_length(p_policy) BETWEEN 2 AND 8192) IS NOT TRUE
 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_bounds'; END IF;
 policy:=p_policy::jsonb;
 IF (jsonb_typeof(policy)='object' AND policy->>'extraction_adapter_version' ~ '^candidate-index-extraction:v2:[0-9a-f]{64}$') IS NOT TRUE
 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_policy'; END IF;
 target:=encode(sha256(convert_to(p_policy,'UTF8')),'hex');
 IF (SELECT count(DISTINCT value->>'source_id') FROM jsonb_array_elements(p_pairs))<>jsonb_array_length(p_pairs)
 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_duplicates'; END IF;
 PERFORM set_config('app.current_tenant_id',p_scope,true);
 FOR pair IN SELECT value FROM jsonb_array_elements(p_pairs) ORDER BY value->>'source_id' COLLATE "C" LOOP
  IF (jsonb_typeof(pair)='object' AND pair ?& ARRAY['source_id','generation_id'] AND pair-ARRAY['source_id','generation_id']='{}'::jsonb) IS NOT TRUE
  THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_pair'; END IF;
  SELECT * INTO s FROM public.candidate_index_sources WHERE source_id=(pair->>'source_id')::uuid AND scope_key=p_scope AND source_kind='organization_application';
  IF NOT FOUND THEN RAISE EXCEPTION USING ERRCODE='42501',MESSAGE='candidate_index_recovery_refused'; END IF;
  SELECT * INTO g FROM public.candidate_index_generations WHERE generation_id=(pair->>'generation_id')::uuid AND source_id=s.source_id AND scope_key=p_scope;
  IF NOT FOUND OR g.policy_sha256=target THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_policy'; END IF;
  allowed:=public.candidate_index_status(s.source_id);
  IF (allowed->>'eligible')::boolean IS DISTINCT FROM true OR (allowed->>'generation_id')::uuid IS DISTINCT FROM g.generation_id
  THEN RAISE EXCEPTION USING ERRCODE='42501',MESSAGE='candidate_index_recovery_refused'; END IF;
  SELECT * INTO j FROM public.candidate_index_jobs WHERE generation_id=g.generation_id AND stage='extract';
  IF NOT FOUND OR j.state NOT IN ('failed','needs_review') OR j.lease_token IS NOT NULL
   OR EXISTS(SELECT 1 FROM public.candidate_index_jobs WHERE generation_id=g.generation_id AND lease_expires_at>at_time)
  THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_state'; END IF;
  SELECT generation_id INTO existing FROM public.candidate_index_generations WHERE source_id=s.source_id AND policy_sha256=target;
  IF existing IS NULL THEN new_count:=new_count+1; END IF;
  items:=items||jsonb_build_array(jsonb_build_object('source_id',s.source_id,'generation_id',g.generation_id,
    'source_hash',s.source_hash,'source_version',s.source_version,'state',j.state,'attempts',j.attempts,'updated_at',j.updated_at,'existing_target',existing));
 END LOOP;
 PERFORM set_config('app.current_tenant_id',coalesce(old_scope,''),true);
 RETURN jsonb_build_object('version',1,'scope',p_scope,'items',items,'policy_json',p_policy,'policy_sha256',target,
  'created_at',at_time,'expires_at',at_time+interval '15 minutes','new_generations',new_count,'max_calls',2*new_count);
END $fn$;
REVOKE ALL ON FUNCTION public.candidate_index_plan_recovery(text,jsonb,text) FROM PUBLIC;

CREATE FUNCTION public.candidate_index_apply_recovery(p_scope text,p_plan jsonb,p_hash text,p_cap integer)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $fn$
DECLARE old_scope text:=current_setting('app.current_tenant_id',true); receipt public.candidate_index_recovery_receipts%ROWTYPE;
 pair jsonb; pairs jsonb; fresh jsonb; member jsonb; s public.candidate_index_sources%ROWTYPE; allowed jsonb;
 members jsonb:='[]'; result jsonb;
BEGIN
 IF (p_scope ~ '^org_[1-9][0-9]{0,9}$' AND p_hash ~ '^[0-9a-f]{64}$' AND p_cap BETWEEN 0 AND 200
  AND jsonb_typeof(p_plan)='object' AND octet_length(p_plan::text)<=65536
  AND p_plan->>'scope'=p_scope AND p_plan->'version'='1'::jsonb
  AND p_plan ?& ARRAY['version','scope','items','policy_json','policy_sha256','created_at','expires_at','new_generations','max_calls']
  AND p_plan-ARRAY['version','scope','items','policy_json','policy_sha256','created_at','expires_at','new_generations','max_calls']='{}'::jsonb
  AND encode(sha256(convert_to(p_plan::text,'UTF8')),'hex')=p_hash
  AND jsonb_typeof(p_plan->'items')='array' AND jsonb_array_length(p_plan->'items') BETWEEN 1 AND 100) IS NOT TRUE
 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_plan'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-index-recovery:'||p_scope,0));
 PERFORM set_config('app.current_tenant_id',p_scope,true);
 SELECT * INTO receipt FROM public.candidate_index_recovery_receipts WHERE plan_hash=p_hash AND scope_key=p_scope;
 IF FOUND THEN
  IF p_cap<receipt.max_calls THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_cap'; END IF;
  FOR member IN SELECT value FROM jsonb_array_elements(receipt.members) LOOP
   allowed:=public.candidate_index_status((member->>'source_id')::uuid);
   IF (allowed->>'eligible')::boolean IS DISTINCT FROM true THEN RAISE EXCEPTION USING ERRCODE='42501',MESSAGE='candidate_index_recovery_refused'; END IF;
  END LOOP;
  PERFORM set_config('app.current_tenant_id',coalesce(old_scope,''),true);
  RETURN jsonb_build_object('plan_hash',p_hash,'members',receipt.members,'max_calls',receipt.max_calls,'replayed',true);
 END IF;
 IF (p_plan->>'expires_at')::timestamptz<=clock_timestamp() OR (p_plan->>'created_at')::timestamptz>clock_timestamp()
  OR (p_plan->>'expires_at')::timestamptz>(p_plan->>'created_at')::timestamptz+interval '15 minutes'
 THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_expired'; END IF;
 -- Same source/head locking order as ordinary admission, before privacy checks.
 FOR pair IN SELECT value FROM jsonb_array_elements(p_plan->'items') ORDER BY value->>'source_id' COLLATE "C" LOOP
  PERFORM pg_advisory_xact_lock(hashtextextended('candidate-index-source:'||(pair->>'source_id'),0));
 END LOOP;
 SELECT jsonb_agg(jsonb_build_object('source_id',value->>'source_id','generation_id',value->>'generation_id')) INTO pairs FROM jsonb_array_elements(p_plan->'items');
 fresh:=public.candidate_index_plan_recovery(p_scope,pairs,p_plan->>'policy_json');
 IF fresh->'items' IS DISTINCT FROM p_plan->'items' OR fresh->'policy_sha256' IS DISTINCT FROM p_plan->'policy_sha256'
  OR fresh->'new_generations' IS DISTINCT FROM p_plan->'new_generations' OR fresh->'max_calls' IS DISTINCT FROM p_plan->'max_calls'
  OR (fresh->>'max_calls')::integer>p_cap THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_index_recovery_stale_or_cap'; END IF;
 FOR pair IN SELECT value FROM jsonb_array_elements(fresh->'items') LOOP
  SELECT * INTO s FROM public.candidate_index_sources WHERE source_id=(pair->>'source_id')::uuid AND scope_key=p_scope;
  result:=public.candidate_index_capture_source(s.source_kind,s.source_id,s.command_digest,s.content_kind,
   CASE WHEN s.content_kind='pinned_text' THEN s.professional_text ELSE replace(encode(s.original_bytes,'base64'),E'\n','') END,
   s.match_tokens,s.key_versions,p_plan->>'policy_json','maintenance',
   coalesce(nullif(current_setting('app.recovery_scope_limit',true),''),'1000')::integer,
   coalesce(nullif(current_setting('app.recovery_total_limit',true),''),'10000')::integer);
  members:=members||jsonb_build_array(jsonb_build_object('source_id',s.source_id,'generation_id',result->>'generation_id'));
 END LOOP;
 INSERT INTO public.candidate_index_recovery_receipts(plan_hash,scope_key,target_policy_sha256,members,max_calls)
 VALUES(p_hash,p_scope,fresh->>'policy_sha256',members,(fresh->>'max_calls')::integer);
 PERFORM set_config('app.current_tenant_id',coalesce(old_scope,''),true);
 RETURN jsonb_build_object('plan_hash',p_hash,'members',members,'max_calls',(fresh->>'max_calls')::integer,'replayed',false);
END $fn$;
REVOKE ALL ON FUNCTION public.candidate_index_apply_recovery(text,jsonb,text,integer) FROM PUBLIC;
