-- 4E: private derivative history, no input rewrite or backfill.
CREATE TABLE public.organization_candidate_history_bindings (
    event_id uuid CONSTRAINT och_bind_pk PRIMARY KEY,
    tenant_id text NOT NULL,
    application_id integer NOT NULL,
    job_id integer NOT NULL,
    reference_id uuid NOT NULL,
    candidate_id uuid NOT NULL,
    source_id uuid NOT NULL,
    payload_digest char(64) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT och_bind_keys_ck CHECK (tenant_id ~ '^org_[1-9][0-9]*$' AND application_id>0 AND job_id>0),
    CONSTRAINT och_bind_digest_ck CHECK (payload_digest ~ '^[0-9a-f]{64}$'),
    CONSTRAINT och_bind_event_fk FOREIGN KEY(event_id) REFERENCES public.organization_decision_event_inbox(event_id) ON DELETE RESTRICT,
    CONSTRAINT och_bind_reference_fk FOREIGN KEY(tenant_id,reference_id,candidate_id)
      REFERENCES public.organization_candidate_references(tenant_id,reference_id,candidate_id) ON DELETE RESTRICT,
    CONSTRAINT och_bind_source_fk FOREIGN KEY(tenant_id,source_id)
      REFERENCES public.candidate_index_sources(scope_key,source_id) ON DELETE RESTRICT
);
CREATE INDEX och_bind_application_idx ON public.organization_candidate_history_bindings(tenant_id,application_id,event_id);

CREATE TABLE public.organization_candidate_history_event_state (
    event_id uuid CONSTRAINT och_work_pk PRIMARY KEY,
    tenant_id text NOT NULL,
    application_id integer NOT NULL,
    state text NOT NULL,
    reason text NOT NULL,
    attempts integer NOT NULL,
    next_attempt_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT och_work_event_fk FOREIGN KEY(event_id) REFERENCES public.organization_decision_event_inbox(event_id) ON DELETE RESTRICT,
    CONSTRAINT och_work_keys_ck CHECK (tenant_id ~ '^org_[1-9][0-9]*$' AND application_id>0),
    CONSTRAINT och_work_attempts_ck CHECK (attempts BETWEEN 1 AND 1000000),
    CONSTRAINT och_work_state_ck CHECK (state IN ('waiting_reference','waiting_source','privacy_wait','privacy_restricted','binding_conflict','applied')),
    CONSTRAINT och_work_reason_ck CHECK (reason IN ('reference_missing','source_missing','binding_conflict','source_contended',
      'privacy_unavailable','privacy_review','privacy_restricted','superseded','consent_changed','source_mismatch','status_contract_invalid','eligible'))
);
CREATE INDEX och_work_retry_idx ON public.organization_candidate_history_event_state(tenant_id,next_attempt_at,event_id)
    WHERE state NOT IN ('applied','binding_conflict');

CREATE TABLE public.organization_candidate_history_applications (
    tenant_id text NOT NULL,
    application_id integer NOT NULL,
    job_id integer NOT NULL,
    reference_id uuid NOT NULL,
    candidate_id uuid NOT NULL,
    observed_count bigint NOT NULL,
    first_sequence bigint NOT NULL,
    first_event_id uuid NOT NULL,
    latest_sequence bigint NOT NULL,
    latest_event_id uuid NOT NULL,
    latest_stage_id integer NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT och_app_pk PRIMARY KEY(tenant_id,application_id),
    CONSTRAINT och_app_keys_ck CHECK (tenant_id ~ '^org_[1-9][0-9]*$' AND application_id>0 AND job_id>0 AND latest_stage_id>0),
    CONSTRAINT och_app_counts_ck CHECK (observed_count>0 AND first_sequence>0 AND latest_sequence>=first_sequence),
    CONSTRAINT och_app_reference_fk FOREIGN KEY(tenant_id,reference_id,candidate_id)
      REFERENCES public.organization_candidate_references(tenant_id,reference_id,candidate_id) ON DELETE RESTRICT,
    CONSTRAINT och_app_first_fk FOREIGN KEY(first_event_id) REFERENCES public.organization_candidate_history_bindings(event_id) ON DELETE RESTRICT,
    CONSTRAINT och_app_latest_fk FOREIGN KEY(latest_event_id) REFERENCES public.organization_candidate_history_bindings(event_id) ON DELETE RESTRICT
);
CREATE INDEX och_app_candidate_idx ON public.organization_candidate_history_applications(tenant_id,candidate_id,application_id);

CREATE TABLE public.organization_candidate_history_subjects (
    tenant_id text NOT NULL,
    candidate_id uuid NOT NULL,
    observed_count bigint NOT NULL,
    first_sequence bigint NOT NULL,
    first_event_id uuid NOT NULL,
    latest_sequence bigint NOT NULL,
    latest_event_id uuid NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT och_subject_pk PRIMARY KEY(tenant_id,candidate_id),
    CONSTRAINT och_subject_tenant_ck CHECK (tenant_id ~ '^org_[1-9][0-9]*$'),
    CONSTRAINT och_subject_count_ck CHECK (observed_count>0 AND first_sequence>0 AND latest_sequence>=first_sequence),
    CONSTRAINT och_subject_candidate_fk FOREIGN KEY(tenant_id,candidate_id) REFERENCES public.candidates(tenant_id,candidate_id) ON DELETE RESTRICT,
    CONSTRAINT och_subject_first_fk FOREIGN KEY(first_event_id) REFERENCES public.organization_candidate_history_bindings(event_id) ON DELETE RESTRICT,
    CONSTRAINT och_subject_latest_fk FOREIGN KEY(latest_event_id) REFERENCES public.organization_candidate_history_bindings(event_id) ON DELETE RESTRICT
);

CREATE TABLE public.organization_candidate_history_scan_state (
    tenant_id text CONSTRAINT och_scan_pk PRIMARY KEY,
    source_cursor bigint NOT NULL DEFAULT 0,
    turns bigint NOT NULL DEFAULT 0,
    last_attempt_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CONSTRAINT och_scan_tenant_ck CHECK (tenant_id ~ '^org_[1-9][0-9]*$'),
    CONSTRAINT och_scan_counts_ck CHECK (source_cursor>=0 AND turns>=0)
);
CREATE INDEX och_inbox_app_seq_idx ON public.organization_decision_event_inbox(tenant_id,subject_id,source_event_sequence);

-- The production superuser bypasses these policies. Explicit keys inside both
-- routines remain mandatory. A safe non-superuser owner also needs these ACLs.
DO $$ DECLARE relation text; BEGIN
  FOREACH relation IN ARRAY ARRAY['organization_candidate_history_bindings','organization_candidate_history_event_state',
    'organization_candidate_history_applications','organization_candidate_history_subjects','organization_candidate_history_scan_state'] LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',relation);
    EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY',relation);
    EXECUTE format('CREATE POLICY och_owner ON public.%I FOR ALL TO %I USING (true) WITH CHECK (true)',relation,current_user);
    EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC',relation);
  END LOOP;
  EXECUTE format('CREATE POLICY och_inbox_owner_read ON public.organization_decision_event_inbox FOR SELECT TO %I USING (true)',current_user);
  EXECUTE format('CREATE POLICY och_stream_owner_read ON public.organization_decision_stream_state FOR SELECT TO %I USING (true)',current_user);
END $$;

CREATE FUNCTION public.organization_candidate_history_append_only() RETURNS trigger
LANGUAGE plpgsql SET search_path=pg_catalog,public AS $$
BEGIN
  IF TG_OP='TRUNCATE' AND NOT EXISTS(SELECT 1 FROM public.organization_candidate_history_bindings) THEN RETURN NULL; END IF;
  RAISE EXCEPTION USING ERRCODE='55000',MESSAGE='HISTORY_BINDING_APPEND_ONLY';
END $$;
REVOKE ALL ON FUNCTION public.organization_candidate_history_append_only() FROM PUBLIC;
CREATE TRIGGER och_bind_no_mutation BEFORE UPDATE OR DELETE ON public.organization_candidate_history_bindings
  FOR EACH ROW EXECUTE FUNCTION public.organization_candidate_history_append_only();
CREATE TRIGGER och_bind_no_truncate BEFORE TRUNCATE ON public.organization_candidate_history_bindings
  FOR EACH STATEMENT EXECUTE FUNCTION public.organization_candidate_history_append_only();

CREATE FUNCTION public.organization_candidate_history_step(p_new_limit integer,p_retry_limit integer)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET statement_timeout='3s' SET lock_timeout='500ms' SET idle_in_transaction_session_timeout='5s'
AS $$
DECLARE
  prior_scope text:=current_setting('app.current_tenant_id',true);
  tenant_key text; cursor_value bigint; turn_value bigint;
  chosen uuid; from_retry boolean:=false;
  e public.organization_decision_event_inbox%ROWTYPE;
  r public.organization_candidate_references%ROWTYPE;
  v public.organization_candidate_resume_evidence%ROWTYPE;
  i public.organization_candidate_ingest_receipts%ROWTYPE;
  s public.candidate_index_sources%ROWTYPE;
  b public.organization_candidate_history_bindings%ROWTYPE;
  status jsonb; work text:='waiting_reference'; why text:='reference_missing';
  inserted integer:=0; old_attempts integer;
BEGIN
  IF (p_new_limit BETWEEN 1 AND 100 AND p_retry_limit BETWEEN 1 AND 25) IS NOT TRUE THEN
    RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='HISTORY_LIMITS_INVALID'; END IF;
  SELECT stream.tenant_id,coalesce(scan.source_cursor,0),coalesce(scan.turns,0)
    INTO tenant_key,cursor_value,turn_value
    FROM public.organization_decision_stream_state stream
    LEFT JOIN public.organization_candidate_history_scan_state scan ON scan.tenant_id=stream.tenant_id
    WHERE stream.last_source_event_sequence>coalesce(scan.source_cursor,0)
      OR EXISTS(SELECT 1 FROM public.organization_candidate_history_event_state w
        WHERE w.tenant_id=stream.tenant_id AND w.state NOT IN ('applied','binding_conflict')
          AND w.next_attempt_at<=clock_timestamp())
    ORDER BY scan.last_attempt_at ASC NULLS FIRST,stream.tenant_id LIMIT 1;
  IF NOT FOUND THEN RETURN jsonb_build_object('outcome','idle'); END IF;
  -- No row lock is taken until the single selected source has been status-checked.
  SELECT candidate.event_id,candidate.retry INTO chosen,from_retry FROM (
    (SELECT inbox.event_id,false AS retry,inbox.source_event_sequence
      FROM public.organization_decision_event_inbox inbox
      WHERE inbox.tenant_id=tenant_key AND inbox.source_event_sequence>cursor_value
      ORDER BY inbox.source_event_sequence LIMIT p_new_limit)
    UNION ALL
    (SELECT inbox.event_id,true AS retry,inbox.source_event_sequence
      FROM public.organization_candidate_history_event_state w
      JOIN public.organization_decision_event_inbox inbox ON inbox.event_id=w.event_id AND inbox.tenant_id=w.tenant_id
      WHERE w.tenant_id=tenant_key AND w.state NOT IN ('applied','binding_conflict')
        AND w.next_attempt_at<=clock_timestamp()
      ORDER BY w.next_attempt_at,inbox.source_event_sequence LIMIT p_retry_limit)
  ) candidate ORDER BY (candidate.retry=(turn_value%2=1)) DESC,candidate.source_event_sequence LIMIT 1;
  IF chosen IS NULL THEN RETURN jsonb_build_object('outcome','idle'); END IF;
  SELECT * INTO STRICT e FROM public.organization_decision_event_inbox WHERE tenant_id=tenant_key AND event_id=chosen;
  PERFORM set_config('app.current_tenant_id',tenant_key,true);
  <<admission>> BEGIN
    IF e.source_system<>'flow' OR e.subject_type<>'application' OR e.action_code<>'application_stage_moved'
      OR e.payload_schema_version<>1 OR tenant_key<>'org_'||e.organization_id::text THEN
      work:='binding_conflict'; why:='binding_conflict'; EXIT admission; END IF;
    SELECT * INTO r FROM public.organization_candidate_references WHERE tenant_id=tenant_key AND application_id=e.subject_id;
    IF NOT FOUND THEN EXIT admission; END IF;
    IF r.job_id<>e.job_id OR r.verified_issuer<>'vantahire' OR r.verified_actor_id<>'vantahire-backend'
      OR r.origin_code<>'candidate_applied' OR r.schema_version<>1
      OR NOT EXISTS(SELECT 1 FROM public.candidates c WHERE c.tenant_id=tenant_key AND c.candidate_id=r.candidate_id AND c.scope='organization_private') THEN
      work:='binding_conflict'; why:='binding_conflict'; EXIT admission; END IF;
    SELECT * INTO v FROM public.organization_candidate_resume_evidence WHERE tenant_id=tenant_key AND reference_id=r.reference_id;
    IF NOT FOUND THEN EXIT admission; END IF;
    SELECT * INTO i FROM public.organization_candidate_ingest_receipts WHERE tenant_id=tenant_key AND reference_id=r.reference_id;
    IF NOT FOUND THEN EXIT admission; END IF;
    IF v.candidate_id<>r.candidate_id OR i.candidate_id<>r.candidate_id OR i.resume_version_id<>v.resume_version_id
      OR i.verified_issuer<>'vantahire' OR i.verified_actor_id<>'vantahire-backend' OR i.resolution<>'created' THEN
      work:='binding_conflict'; why:='binding_conflict'; EXIT admission; END IF;
    work:='waiting_source'; why:='source_missing';
    SELECT * INTO s FROM public.candidate_index_sources WHERE scope_key=tenant_key AND source_id=v.resume_version_id;
    IF NOT FOUND THEN EXIT admission; END IF;
    IF s.tenant_id IS DISTINCT FROM tenant_key OR s.source_kind<>'organization_application'
      OR s.reference_id IS DISTINCT FROM r.reference_id OR s.authority_id<>r.reference_id
      OR s.candidate_id IS DISTINCT FROM r.candidate_id OR s.resume_version_id IS DISTINCT FROM v.resume_version_id
      OR s.source_version<>v.version THEN
      work:='binding_conflict'; why:='source_mismatch'; EXIT admission; END IF;
    status:=public.candidate_index_status(s.source_id,true);
    IF status IS NULL THEN EXIT admission; END IF;
    work:='privacy_wait'; why:='status_contract_invalid';
    IF status->'eligible'='false'::jsonb THEN
      IF status->'contended'='true'::jsonb THEN why:='source_contended';
      ELSIF status->>'reason' IN ('privacy_unavailable','privacy_review') THEN why:=status->>'reason';
      ELSIF status->>'reason'='privacy_restricted' THEN work:='privacy_restricted'; why:=work;
      ELSIF status->>'reason'='superseded' THEN work:='waiting_source'; why:='superseded';
      ELSIF status->>'reason'='consent_changed' THEN work:='binding_conflict'; why:='consent_changed'; END IF;
    ELSIF status->'eligible'='true'::jsonb THEN
      IF status->>'source_id' IS DISTINCT FROM s.source_id::text THEN work:='binding_conflict'; why:='source_mismatch';
      ELSIF status->'reason'='null'::jsonb THEN work:='applied'; why:='eligible'; END IF;
    END IF;
  END admission;
  -- Status privacy locks are now held until the caller's commit; never check a
  -- second event in this transaction. Event advisory fencing prevents duplicate
  -- workers from updating either compact count twice without locking inputs.
  IF NOT pg_try_advisory_xact_lock(hashtextextended('candidate-history:'||chosen::text,0)) THEN
    PERFORM set_config('app.current_tenant_id',coalesce(prior_scope,''),true);
    RETURN jsonb_build_object('outcome','contended'); END IF;
  SELECT * INTO b FROM public.organization_candidate_history_bindings WHERE event_id=chosen AND tenant_id=tenant_key;
  IF FOUND THEN
    IF b.payload_digest<>e.payload_digest OR b.application_id<>e.subject_id OR b.job_id<>e.job_id
      OR b.reference_id IS DISTINCT FROM r.reference_id OR b.candidate_id IS DISTINCT FROM r.candidate_id
      OR b.source_id IS DISTINCT FROM s.source_id THEN
      RAISE EXCEPTION USING ERRCODE='23514',MESSAGE='HISTORY_BINDING_CONFLICT'; END IF;
    work:='applied'; why:='eligible';
  ELSIF work='applied' THEN
    INSERT INTO public.organization_candidate_history_bindings(event_id,tenant_id,application_id,job_id,
      reference_id,candidate_id,source_id,payload_digest)
      VALUES(chosen,tenant_key,e.subject_id,e.job_id,r.reference_id,r.candidate_id,s.source_id,e.payload_digest);
    inserted:=1;
    INSERT INTO public.organization_candidate_history_applications AS a
      (tenant_id,application_id,job_id,reference_id,candidate_id,observed_count,first_sequence,first_event_id,latest_sequence,latest_event_id,latest_stage_id)
      VALUES(tenant_key,e.subject_id,e.job_id,r.reference_id,r.candidate_id,1,e.source_event_sequence,chosen,e.source_event_sequence,chosen,(e.after_state->>'stage_id')::integer)
      ON CONFLICT(tenant_id,application_id) DO UPDATE SET
        observed_count=a.observed_count+1,
        first_event_id=CASE WHEN excluded.first_sequence<a.first_sequence THEN excluded.first_event_id ELSE a.first_event_id END,
        first_sequence=least(a.first_sequence,excluded.first_sequence),
        latest_event_id=CASE WHEN excluded.latest_sequence>a.latest_sequence THEN excluded.latest_event_id ELSE a.latest_event_id END,
        latest_stage_id=CASE WHEN excluded.latest_sequence>a.latest_sequence THEN excluded.latest_stage_id ELSE a.latest_stage_id END,
        latest_sequence=greatest(a.latest_sequence,excluded.latest_sequence),updated_at=clock_timestamp()
      WHERE a.reference_id=excluded.reference_id AND a.candidate_id=excluded.candidate_id AND a.job_id=excluded.job_id;
    IF NOT FOUND THEN RAISE EXCEPTION USING ERRCODE='23514',MESSAGE='HISTORY_BINDING_CONFLICT'; END IF;
    INSERT INTO public.organization_candidate_history_subjects AS a
      (tenant_id,candidate_id,observed_count,first_sequence,first_event_id,latest_sequence,latest_event_id)
      VALUES(tenant_key,r.candidate_id,1,e.source_event_sequence,chosen,e.source_event_sequence,chosen)
      ON CONFLICT(tenant_id,candidate_id) DO UPDATE SET
        observed_count=a.observed_count+1,
        first_event_id=CASE WHEN excluded.first_sequence<a.first_sequence THEN excluded.first_event_id ELSE a.first_event_id END,
        first_sequence=least(a.first_sequence,excluded.first_sequence),
        latest_event_id=CASE WHEN excluded.latest_sequence>a.latest_sequence THEN excluded.latest_event_id ELSE a.latest_event_id END,
        latest_sequence=greatest(a.latest_sequence,excluded.latest_sequence),updated_at=clock_timestamp();
  END IF;
  SELECT attempts INTO old_attempts FROM public.organization_candidate_history_event_state WHERE tenant_id=tenant_key AND event_id=chosen;
  INSERT INTO public.organization_candidate_history_event_state AS w
    (event_id,tenant_id,application_id,state,reason,attempts,next_attempt_at)
    VALUES(chosen,tenant_key,e.subject_id,work,why,least(coalesce(old_attempts,0)+1,1000000),
      clock_timestamp()+make_interval(secs=>least(300,5*power(2,least(coalesce(old_attempts,0),6)))::double precision))
    ON CONFLICT(event_id) DO UPDATE SET state=excluded.state,reason=excluded.reason,attempts=excluded.attempts,
      next_attempt_at=excluded.next_attempt_at,updated_at=clock_timestamp()
    WHERE w.tenant_id=tenant_key AND w.application_id=e.subject_id AND w.state NOT IN ('applied','binding_conflict');
  INSERT INTO public.organization_candidate_history_scan_state AS scan(tenant_id,source_cursor,turns,last_attempt_at)
    VALUES(tenant_key,CASE WHEN from_retry THEN cursor_value ELSE e.source_event_sequence END,1,clock_timestamp())
    ON CONFLICT(tenant_id) DO UPDATE SET source_cursor=greatest(scan.source_cursor,excluded.source_cursor),
      turns=scan.turns+1,last_attempt_at=clock_timestamp();
  PERFORM set_config('app.current_tenant_id',coalesce(prior_scope,''),true);
  RETURN jsonb_build_object('outcome',work,'reason',why,'applied',inserted);
END $$;
REVOKE ALL ON FUNCTION public.organization_candidate_history_step(integer,integer) FROM PUBLIC;

CREATE FUNCTION public.organization_candidate_history_read(
  p_tenant text,p_application integer,p_job integer,p_reference uuid,
  p_expected_sequence bigint,p_expected_event uuid,p_expected_count bigint
) RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET statement_timeout='3s' SET lock_timeout='500ms' SET idle_in_transaction_session_timeout='5s'
AS $$
DECLARE
  prior_scope text:=current_setting('app.current_tenant_id',true);
  r public.organization_candidate_references%ROWTYPE;
  v public.organization_candidate_resume_evidence%ROWTYPE;
  i public.organization_candidate_ingest_receipts%ROWTYPE;
  s public.candidate_index_sources%ROWTYPE;
  a public.organization_candidate_history_applications%ROWTYPE;
  first_event public.organization_decision_event_inbox%ROWTYPE;
  latest_event public.organization_decision_event_inbox%ROWTYPE;
  status jsonb; result jsonb; summary jsonb:=NULL;
  authority text:='awaiting_binding'; freshness text:='awaiting_binding';
  received bigint:=0; unresolved bigint:=0; projected bigint:=0;
  projected_event uuid; projected_sequence bigint;
BEGIN
  IF (p_tenant ~ '^org_[1-9][0-9]{0,9}$' AND substring(p_tenant FROM 5)::bigint<=2147483647
    AND p_application>0 AND p_job>0 AND p_reference IS NOT NULL AND p_expected_count>=0
    AND ((p_expected_count=0 AND p_expected_event IS NULL AND p_expected_sequence IS NULL)
      OR (p_expected_count>0 AND p_expected_event IS NOT NULL AND p_expected_sequence>0))) IS NOT TRUE THEN
    RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='HISTORY_KEYS_INVALID'; END IF;
  PERFORM set_config('app.current_tenant_id',p_tenant,true);
  <<admission>> BEGIN
    SELECT * INTO r FROM public.organization_candidate_references WHERE tenant_id=p_tenant AND application_id=p_application;
    IF NOT FOUND OR r.reference_id<>p_reference OR r.job_id<>p_job THEN
      result:=jsonb_build_object('error','not_found'); EXIT admission; END IF;
    IF r.verified_issuer<>'vantahire' OR r.verified_actor_id<>'vantahire-backend'
      OR r.origin_code<>'candidate_applied' OR r.schema_version<>1
      OR NOT EXISTS(SELECT 1 FROM public.candidates c WHERE c.tenant_id=p_tenant AND c.candidate_id=r.candidate_id AND c.scope='organization_private') THEN
      result:=jsonb_build_object('error','binding_conflict'); EXIT admission; END IF;
    SELECT * INTO v FROM public.organization_candidate_resume_evidence WHERE tenant_id=p_tenant AND reference_id=p_reference;
    IF NOT FOUND THEN EXIT admission; END IF;
    SELECT * INTO i FROM public.organization_candidate_ingest_receipts WHERE tenant_id=p_tenant AND reference_id=p_reference;
    IF NOT FOUND THEN EXIT admission; END IF;
    IF v.candidate_id<>r.candidate_id OR i.candidate_id<>r.candidate_id OR i.resume_version_id<>v.resume_version_id
      OR i.verified_issuer<>'vantahire' OR i.verified_actor_id<>'vantahire-backend' OR i.resolution<>'created' THEN
      result:=jsonb_build_object('error','binding_conflict'); EXIT admission; END IF;
    SELECT * INTO s FROM public.candidate_index_sources WHERE scope_key=p_tenant AND source_id=v.resume_version_id;
    IF NOT FOUND THEN EXIT admission; END IF;
    IF s.tenant_id IS DISTINCT FROM p_tenant OR s.source_kind<>'organization_application'
      OR s.reference_id IS DISTINCT FROM p_reference OR s.authority_id<>p_reference
      OR s.candidate_id IS DISTINCT FROM r.candidate_id OR s.resume_version_id IS DISTINCT FROM v.resume_version_id
      OR s.source_version<>v.version THEN
      result:=jsonb_build_object('error','binding_conflict'); EXIT admission; END IF;
    status:=public.candidate_index_status(s.source_id,true);
    IF status IS NULL THEN EXIT admission; END IF;
    authority:='temporarily_unavailable'; freshness:='temporarily_unavailable';
    IF status->'eligible'='false'::jsonb THEN
      IF status->'contended'='true'::jsonb THEN result:=jsonb_build_object('error','temporarily_unavailable');
      ELSIF status->>'reason' IN ('privacy_unavailable','privacy_review') THEN result:=jsonb_build_object('error','temporarily_unavailable');
      ELSIF status->>'reason'='privacy_restricted' THEN result:=jsonb_build_object('error','privacy_restricted');
      ELSIF status->>'reason'='superseded' THEN authority:='awaiting_binding'; freshness:='awaiting_binding';
      ELSIF status->>'reason'='consent_changed' THEN result:=jsonb_build_object('error','binding_conflict');
      ELSE result:=jsonb_build_object('error','temporarily_unavailable'); END IF;
      EXIT admission;
    END IF;
    IF status->'eligible' IS DISTINCT FROM 'true'::jsonb OR status->'reason' IS DISTINCT FROM 'null'::jsonb THEN
      result:=jsonb_build_object('error','temporarily_unavailable'); EXIT admission; END IF;
    IF status->>'source_id' IS DISTINCT FROM s.source_id::text THEN
      result:=jsonb_build_object('error','binding_conflict'); EXIT admission; END IF;
    authority:='eligible';
    -- Both owner variants use these tuple predicates; RLS is not the fence.
    IF EXISTS(SELECT 1 FROM public.organization_candidate_history_bindings b
      JOIN public.organization_decision_event_inbox e ON e.event_id=b.event_id AND e.tenant_id=b.tenant_id
      WHERE b.tenant_id=p_tenant AND b.application_id=p_application AND (
        b.job_id<>p_job OR b.reference_id<>p_reference OR b.candidate_id<>r.candidate_id
        OR b.source_id<>s.source_id OR b.payload_digest<>e.payload_digest OR e.subject_id<>p_application OR e.job_id<>p_job))
      OR EXISTS(SELECT 1 FROM public.organization_candidate_history_event_state w
        WHERE w.tenant_id=p_tenant AND w.application_id=p_application AND w.state='binding_conflict') THEN
      result:=jsonb_build_object('error','binding_conflict'); EXIT admission; END IF;
    SELECT count(*),count(*) FILTER(WHERE b.event_id IS NULL) INTO received,unresolved
      FROM public.organization_decision_event_inbox e
      LEFT JOIN public.organization_candidate_history_bindings b ON b.event_id=e.event_id AND b.tenant_id=e.tenant_id
        AND b.application_id=p_application AND b.job_id=p_job AND b.reference_id=p_reference AND b.candidate_id=r.candidate_id
      WHERE e.tenant_id=p_tenant AND e.subject_type='application' AND e.subject_id=p_application AND e.job_id=p_job
        AND e.action_code='application_stage_moved' AND e.source_system='flow' AND e.payload_schema_version=1;
    SELECT * INTO a FROM public.organization_candidate_history_applications
      WHERE tenant_id=p_tenant AND application_id=p_application;
    IF FOUND THEN
      IF a.job_id<>p_job OR a.reference_id<>p_reference OR a.candidate_id<>r.candidate_id THEN
        result:=jsonb_build_object('error','binding_conflict'); EXIT admission; END IF;
      projected:=a.observed_count; projected_event:=a.latest_event_id; projected_sequence:=a.latest_sequence;
      SELECT * INTO STRICT first_event FROM public.organization_decision_event_inbox
        WHERE tenant_id=p_tenant AND event_id=a.first_event_id AND subject_id=p_application AND job_id=p_job;
      SELECT * INTO STRICT latest_event FROM public.organization_decision_event_inbox
        WHERE tenant_id=p_tenant AND event_id=a.latest_event_id AND subject_id=p_application AND job_id=p_job;
      summary:=jsonb_build_object('observed_stage_move_count',projected::text,
        'first',jsonb_build_object('event_id',first_event.event_id,'sequence',first_event.source_event_sequence::text,'occurred_at',first_event.occurred_at),
        'latest',jsonb_build_object('event_id',latest_event.event_id,'sequence',latest_event.source_event_sequence::text,'occurred_at',latest_event.occurred_at),
        'latest_observed_stage_id',a.latest_stage_id,'taxonomy_version',latest_event.taxonomy_version,
        'rubric_id',latest_event.rubric_id,'rubric_version',latest_event.rubric_version,
        'rubric_approval_mode',latest_event.rubric_approval_mode,'jd_digest_version',latest_event.jd_digest_version,
        'recommendation_action',latest_event.recommendation_action,'reason_code',latest_event.reason_code);
    END IF;
    IF projected>p_expected_count OR received>p_expected_count
      OR projected_sequence>p_expected_sequence
      OR EXISTS(SELECT 1 FROM public.organization_decision_event_inbox e WHERE e.tenant_id=p_tenant
        AND e.subject_id=p_application AND e.job_id=p_job AND e.source_event_sequence>p_expected_sequence) THEN
      freshness:='history_changed_retry'; summary:=NULL;
    ELSIF p_expected_count=0 AND received=0 AND projected=0 THEN freshness:='no_captured_stage_events';
    ELSIF projected=p_expected_count AND projected_event=p_expected_event AND projected_sequence=p_expected_sequence
      AND unresolved=0 AND received=p_expected_count THEN freshness:='caught_up_to_observed_capture';
    ELSIF received<p_expected_count THEN freshness:='awaiting_delivery';
    ELSE freshness:='projection_pending'; END IF;
  END admission;
  PERFORM set_config('app.current_tenant_id',coalesce(prior_scope,''),true);
  IF result IS NOT NULL THEN RETURN result; END IF;
  RETURN jsonb_build_object('schema_version',1,
    'binding',jsonb_build_object('namespace','organization_private','organization_id',substring(p_tenant FROM 5)::integer,
      'application_id',p_application,'job_id',p_job,'reference_id',p_reference),
    'coverage',jsonb_build_object('event_types',jsonb_build_array('application_stage_moved'),
      'identity_basis','organization_application_reference','historical_complete',false),
    'freshness',jsonb_build_object('expected',jsonb_build_object('count',p_expected_count::text,'event_id',p_expected_event,'sequence',p_expected_sequence::text),
      'projected',jsonb_build_object('count',projected::text,'event_id',projected_event,'sequence',projected_sequence::text),
      'unresolved_count',unresolved::text,'status',freshness),
    'authority_status',authority,'summary',summary);
END $$;
REVOKE ALL ON FUNCTION public.organization_candidate_history_read(text,integer,integer,uuid,bigint,uuid,bigint) FROM PUBLIC;
