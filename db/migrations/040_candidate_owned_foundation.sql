-- Empty candidate-owned subjects only. No effective consent, facts or search publication.
CREATE TABLE public.candidate_owned_subjects (
 subject_id uuid PRIMARY KEY, flow_binding_id uuid NOT NULL UNIQUE,
 state text NOT NULL DEFAULT 'empty' CHECK (state='empty'),
 profile_version bigint NOT NULL DEFAULT 0 CHECK (profile_version=0),
 preference_version bigint NOT NULL DEFAULT 0 CHECK (preference_version=0),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 UNIQUE(subject_id,flow_binding_id)
);
CREATE TABLE public.candidate_owned_command_receipts (
 command_id uuid PRIMARY KEY, subject_id uuid NOT NULL, flow_binding_id uuid NOT NULL,
 sequence bigint NOT NULL CHECK(sequence BETWEEN 1 AND 9007199254740991),
 wire_sha256 text NOT NULL CHECK(wire_sha256 ~ '^[0-9a-f]{64}$'),
 action text NOT NULL CHECK(action='ensure_subject'),
 outcome text NOT NULL CHECK(outcome='subject_ready'),
 applied_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 FOREIGN KEY(subject_id,flow_binding_id) REFERENCES public.candidate_owned_subjects(subject_id,flow_binding_id),
 UNIQUE(subject_id,sequence)
);
ALTER TABLE public.candidate_owned_subjects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_owned_subjects FORCE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_owned_command_receipts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_owned_command_receipts FORCE ROW LEVEL SECURITY;
DO $policy$
BEGIN
 EXECUTE format('CREATE POLICY owner_only ON public.candidate_owned_subjects TO %I USING(true) WITH CHECK(true)',current_user);
 EXECUTE format('CREATE POLICY owner_only ON public.candidate_owned_command_receipts TO %I USING(true) WITH CHECK(true)',current_user);
END $policy$;

CREATE FUNCTION public.candidate_owned_ensure(p_subject uuid,p_binding uuid,p_command uuid,p_sequence bigint,p_hash text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public,pg_temp SET lock_timeout='5s' SET statement_timeout='10s'
AS $$
DECLARE prior public.candidate_owned_command_receipts%ROWTYPE; bound uuid; maximum bigint; replayed boolean;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_sequence IS NULL
 OR p_sequence NOT BETWEEN 1 AND 9007199254740991 OR p_hash IS NULL OR p_hash !~ '^[0-9a-f]{64}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-foundation:'||p_subject::text,0));
 SELECT flow_binding_id INTO bound FROM public.candidate_owned_subjects WHERE subject_id=p_subject FOR UPDATE;
 IF (FOUND AND bound<>p_binding) OR EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE flow_binding_id=p_binding AND subject_id<>p_subject)
 THEN RAISE EXCEPTION 'identity_conflict' USING ERRCODE='23505'; END IF;
 SELECT * INTO prior FROM public.candidate_owned_command_receipts WHERE command_id=p_command;
 replayed:=FOUND;
 IF replayed THEN
   IF (prior.subject_id,prior.flow_binding_id,prior.sequence,prior.wire_sha256) IS DISTINCT FROM (p_subject,p_binding,p_sequence,p_hash)
   THEN RAISE EXCEPTION 'receipt_conflict' USING ERRCODE='23505'; END IF;
 ELSE
   SELECT coalesce(max(sequence),0) INTO maximum FROM public.candidate_owned_command_receipts WHERE subject_id=p_subject;
   IF p_sequence<=maximum THEN RAISE EXCEPTION 'stale_sequence' USING ERRCODE='23505'; END IF;
   INSERT INTO public.candidate_owned_subjects(subject_id,flow_binding_id) VALUES(p_subject,p_binding) ON CONFLICT(subject_id) DO NOTHING;
   INSERT INTO public.candidate_owned_command_receipts(command_id,subject_id,flow_binding_id,sequence,wire_sha256,action,outcome)
    VALUES(p_command,p_subject,p_binding,p_sequence,p_hash,'ensure_subject','subject_ready');
 END IF;
 -- _replayed is internal transport metadata: stripped by the API, never added to the wire receipt.
 RETURN jsonb_build_object('schemaVersion',1,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,
 'sequence',p_sequence,'wireSha256',p_hash,'outcome','subject_ready','profileVersion',0,'preferenceVersion',0,'publicationActive',false,'_replayed',replayed);
END $$;
CREATE FUNCTION public.candidate_owned_status(p_subject uuid,p_binding uuid) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public,pg_temp
SET lock_timeout='5s' SET statement_timeout='10s' AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 RETURN jsonb_build_object('schemaVersion',1,'state','empty','profileVersion',0,'preferenceVersion',0,'publicationActive',false);
END $$;
REVOKE ALL ON public.candidate_owned_subjects,public.candidate_owned_command_receipts FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_owned_ensure(uuid,uuid,uuid,bigint,text),public.candidate_owned_status(uuid,uuid) FROM PUBLIC;
DO $deny$ DECLARE item record; BEGIN
 FOR item IN SELECT DISTINCT c.oid::regclass AS object_name,r.rolname FROM pg_class c
 JOIN pg_namespace n ON n.oid=c.relnamespace
 CROSS JOIN LATERAL aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a
 JOIN pg_roles r ON r.oid=a.grantee
 WHERE n.nspname='public' AND c.relname IN ('candidate_owned_subjects','candidate_owned_command_receipts')
 AND a.grantee<>c.relowner LOOP
  EXECUTE format('REVOKE ALL ON TABLE %s FROM %I',item.object_name,item.rolname);
 END LOOP;
 FOR item IN SELECT DISTINCT p.oid::regprocedure AS object_name,r.rolname FROM pg_proc p
 JOIN pg_namespace n ON n.oid=p.pronamespace
 CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 JOIN pg_roles r ON r.oid=a.grantee
 WHERE n.nspname='public' AND starts_with(p.proname,'candidate_owned_') AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',item.object_name,item.rolname);
 END LOOP;
END $deny$;
