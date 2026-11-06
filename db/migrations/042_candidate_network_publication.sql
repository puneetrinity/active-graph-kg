-- Native first-party publications. No provider identity, contact data or legacy
-- public/private evidence predicate is rewritten by this migration.
CREATE TABLE public.candidate_network_subjects (
 subject_id uuid PRIMARY KEY REFERENCES public.candidate_owned_subjects(subject_id),
 global_candidate_id uuid NOT NULL UNIQUE REFERENCES public.global_candidates(id),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE public.candidate_network_subjects ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_network_subjects FORCE ROW LEVEL SECURITY;
DO $$ BEGIN EXECUTE format('CREATE POLICY owner_only ON public.candidate_network_subjects TO %I USING(true) WITH CHECK(true)',current_user); END $$;
REVOKE ALL ON public.candidate_network_subjects FROM PUBLIC;

-- A native canonical anchor is NOT a merge with an email/LinkedIn match. It
-- lets the existing privacy directive system address a person without either.
CREATE FUNCTION public.candidate_network_anchor(s uuid) RETURNS uuid
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE g uuid;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-network-anchor:'||s::text,0));
 SELECT global_candidate_id INTO g FROM public.candidate_network_subjects WHERE subject_id=s;
 IF FOUND THEN RETURN g;END IF;
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=s)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 INSERT INTO public.global_candidates(embedding_status) VALUES('skipped_empty') RETURNING id INTO g;
 INSERT INTO public.candidate_network_subjects(subject_id,global_candidate_id) VALUES(s,g);
 RETURN g;
END $$;
CREATE FUNCTION public.candidate_network_anchor_created() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN PERFORM public.candidate_network_anchor(NEW.subject_id);RETURN NEW;END $$;
CREATE TRIGGER candidate_network_anchor_created AFTER INSERT ON public.candidate_owned_subjects
FOR EACH ROW EXECUTE FUNCTION public.candidate_network_anchor_created();
SELECT public.candidate_network_anchor(subject_id) FROM public.candidate_owned_subjects ORDER BY subject_id;

-- Reuse the existing resolver and directive state machine. Native identifiers
-- are additive; they never merge a person by email/name or a claimed URL.
CREATE OR REPLACE FUNCTION public.candidate_privacy_resolve_subject(p_identifier_type text,p_lookup_digest bytea)
RETURNS TABLE(global_candidate_id uuid,candidate_tenant_id text,candidate_id uuid)
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
 SELECT DISTINCT r.global_candidate_id,r.candidate_tenant_id,r.candidate_id FROM (
  SELECT c.global_candidate_id,c.tenant_id AS candidate_tenant_id,c.candidate_id
  FROM public.candidate_identifiers ci JOIN public.candidates c
   ON c.tenant_id IS NOT DISTINCT FROM ci.tenant_id AND c.candidate_id=ci.candidate_id
  WHERE ci.identifier_type=p_identifier_type
   AND digest(convert_to(ci.value_normalized,'UTF8'),'sha256')=p_lookup_digest
  UNION ALL
  SELECT c.global_candidate_id,c.tenant_id,c.candidate_id FROM public.candidates c
  WHERE p_identifier_type='email' AND c.primary_email IS NOT NULL
   AND digest(convert_to(lower(btrim(c.primary_email)),'UTF8'),'sha256')=p_lookup_digest
  UNION ALL
  SELECT gc.id,NULL::text,NULL::uuid FROM public.global_candidates gc
  WHERE (p_identifier_type='email' AND gc.email_hash=encode(p_lookup_digest,'hex'))
   OR (p_identifier_type='linkedin_url' AND gc.linkedin_id IS NOT NULL
    AND digest(convert_to('https://linkedin.com/in/'||lower(btrim(gc.linkedin_id)),'UTF8'),'sha256')=p_lookup_digest)
   OR (p_identifier_type='github_url' AND gc.github_id IS NOT NULL
    AND digest(convert_to('https://github.com/'||lower(btrim(gc.github_id)),'UTF8'),'sha256')=p_lookup_digest)
  UNION ALL
  SELECT n.global_candidate_id,NULL::text,NULL::uuid FROM public.candidate_network_subjects n
  WHERE p_identifier_type='signal_candidate_id'
   AND digest(convert_to('network:'||n.subject_id::text,'UTF8'),'sha256')=p_lookup_digest
 ) r
$$;

CREATE TABLE public.candidate_network_publications (
 publication_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
 subject_id uuid NOT NULL REFERENCES public.candidate_owned_subjects(subject_id),
 profile_version bigint NOT NULL, preference_version bigint NOT NULL, consent_revision bigint NOT NULL,
 facts jsonb NOT NULL CHECK(jsonb_typeof(facts)='object' AND octet_length(facts::text)<=393216),
 content_hash text NOT NULL CHECK(content_hash~'^[a-f0-9]{64}$'),
 state text NOT NULL DEFAULT 'staged' CHECK(state IN ('staged','embedding','active','revoked','held')),
 embedding vector(384), encoder_version integer,
 lease_token uuid, lease_until timestamptz, attempts integer NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 3),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(), activated_at timestamptz, revoked_at timestamptz,
 UNIQUE(subject_id,profile_version,preference_version,consent_revision),
 FOREIGN KEY(subject_id,profile_version) REFERENCES public.candidate_profile_versions(subject_id,version),
 CHECK((state='embedding')=(lease_token IS NOT NULL AND lease_until IS NOT NULL)),
 CHECK(state<>'active' OR (embedding IS NOT NULL AND encoder_version=1 AND activated_at IS NOT NULL))
);
CREATE UNIQUE INDEX candidate_network_one_active ON public.candidate_network_publications(subject_id) WHERE state='active';
ALTER TABLE public.candidate_network_publications ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_network_publications FORCE ROW LEVEL SECURITY;
DO $$ BEGIN EXECUTE format('CREATE POLICY owner_only ON public.candidate_network_publications TO %I USING(true) WITH CHECK(true)',current_user); END $$;
REVOKE ALL ON public.candidate_network_publications FROM PUBLIC;

CREATE FUNCTION public.candidate_network_eligible(s uuid,p bigint,f bigint,c bigint) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE previous_tenant text:=coalesce(current_setting('app.current_tenant_id',true),'');ok boolean;
BEGIN
 PERFORM set_config('app.current_tenant_id','candidate_'||s::text,true);
 SELECT EXISTS(SELECT 1 FROM public.candidate_profile_heads h
 JOIN public.candidate_profile_versions v ON v.subject_id=h.subject_id AND v.version=h.profile_version
 JOIN public.candidate_consent_state cs ON cs.subject_id=h.subject_id
 JOIN public.candidate_network_subjects ns ON ns.subject_id=h.subject_id
 JOIN public.candidate_consent_copy_catalog cc ON cc.copy_version=cs.effective_copy_version AND cc.copy_sha256=cs.effective_copy_sha256 AND cc.approved
 WHERE h.subject_id=s AND NOT h.restricted AND h.profile_version=p AND h.preference_version=f AND v.facts IS NOT NULL
 AND cs.effective_action='grant' AND cs.effective_schema_version=2 AND cs.highest_version=cs.effective_version AND cs.effective_version=c
 AND EXISTS(SELECT 1 FROM public.candidate_consent_receipts cr WHERE cr.subject_id=s AND cr.version=c AND cr.authority_schema_version=2
  AND cr.outcome='onboarding_granted' AND cr.copy_version=cs.effective_copy_version AND cr.copy_sha256=cs.effective_copy_sha256
  AND cr.purposes=ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact'])
 AND EXISTS(SELECT 1 FROM public.candidate_profile_confirmations cf WHERE cf.subject_id=s AND cf.profile_version=p AND cf.preference_version=f AND cf.consent_revision=c)
 AND public.candidate_privacy_global_decision(ns.global_candidate_id)='allow'
 AND (cs.global_candidate_id IS NULL OR public.candidate_privacy_global_decision(cs.global_candidate_id)='allow')) INTO ok;
 PERFORM set_config('app.current_tenant_id',previous_tenant,true);RETURN ok;
END $$;

-- Public facts are projected explicitly; descriptions/URLs/contacts and all
-- private preferences are omitted, not copied wholesale and then blacklisted.
CREATE FUNCTION public.candidate_network_text(v text) RETURNS text
LANGUAGE sql IMMUTABLE SET search_path=pg_catalog,public AS $$
 SELECT btrim(regexp_replace(regexp_replace(regexp_replace(coalesce(v,''),
  '[[:alnum:]_.+%-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}','[contact removed]','gi'),
  '(https?://|www\.)[^[:space:]]+','[link removed]','gi'),
  '\+?[0-9][0-9 ().-]{7,}[0-9]','[contact removed]','g'))
$$;
CREATE FUNCTION public.candidate_network_projection(v jsonb) RETURNS jsonb
LANGUAGE sql IMMUTABLE SET search_path=pg_catalog,public AS $$
 SELECT jsonb_build_object('displayName',public.candidate_network_text(v->>'displayName'),
 'headline',public.candidate_network_text(v->>'headline'),
 'currentLocation',CASE WHEN jsonb_typeof(v->'currentLocation')='object' THEN jsonb_build_object(
  'city',public.candidate_network_text(v#>>'{currentLocation,city}'),
  'countryCode',v#>>'{currentLocation,countryCode}',
  'displayCountry',public.candidate_network_text(v#>>'{currentLocation,displayCountry}')) ELSE NULL END,
 'skills',coalesce((SELECT jsonb_agg(public.candidate_network_text(s) ORDER BY n)
  FROM jsonb_array_elements_text(coalesce(v->'skills','[]'::jsonb)) WITH ORDINALITY x(s,n)),'[]'::jsonb),
 'employment',coalesce((SELECT jsonb_agg(jsonb_build_object('employer',public.candidate_network_text(r->>'employer'),
  'title',public.candidate_network_text(r->>'title'),
  'start',r->'start','end',r->'end','current',r->'current') ORDER BY n)
  FROM jsonb_array_elements(coalesce(v->'employment','[]'::jsonb)) WITH ORDINALITY x(r,n)),'[]'::jsonb),
 'education',coalesce((SELECT jsonb_agg(jsonb_build_object('institution',public.candidate_network_text(r->>'institution'),
  'qualification',public.candidate_network_text(r->>'qualification'),
  'start',r->'start','end',r->'end','current',r->'current') ORDER BY n)
  FROM jsonb_array_elements(coalesce(v->'education','[]'::jsonb)) WITH ORDINALITY x(r,n)),'[]'::jsonb))
$$;

CREATE FUNCTION public.candidate_network_confirmation() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE projected jsonb;
BEGIN
 IF NOT public.candidate_network_eligible(NEW.subject_id,NEW.profile_version,NEW.preference_version,NEW.consent_revision) THEN RETURN NEW;END IF;
 SELECT public.candidate_network_projection(facts) INTO projected FROM public.candidate_profile_versions
 WHERE subject_id=NEW.subject_id AND version=NEW.profile_version;
 INSERT INTO public.candidate_network_publications(subject_id,profile_version,preference_version,consent_revision,facts,content_hash)
 VALUES(NEW.subject_id,NEW.profile_version,NEW.preference_version,NEW.consent_revision,projected,
 encode(sha256(convert_to(public.candidate_profile_canonical(projected),'UTF8')),'hex')) ON CONFLICT DO NOTHING;
 RETURN NEW;
END $$;
CREATE TRIGGER candidate_network_confirmed AFTER INSERT ON public.candidate_profile_confirmations
FOR EACH ROW EXECUTE FUNCTION public.candidate_network_confirmation();

CREATE FUNCTION public.candidate_network_revoke_changed() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
 UPDATE public.candidate_network_publications p SET state='revoked',lease_token=NULL,lease_until=NULL,
 revoked_at=coalesce(revoked_at,clock_timestamp()) WHERE p.subject_id=NEW.subject_id AND state IN ('staged','embedding','active')
 AND NOT public.candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision);
 -- Restriction is quarantine, not approval to delete. Keep the stored snapshot
 -- inaccessible; permanent deletion belongs to the separate operator procedure.
 RETURN NEW;
END $$;
CREATE TRIGGER candidate_network_consent_changed AFTER UPDATE ON public.candidate_consent_state
FOR EACH ROW EXECUTE FUNCTION public.candidate_network_revoke_changed();
CREATE TRIGGER candidate_network_profile_changed AFTER UPDATE ON public.candidate_profile_heads
FOR EACH ROW EXECUTE FUNCTION public.candidate_network_revoke_changed();

CREATE FUNCTION public.candidate_network_operator_erased() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
 -- Only the existing owner-only erasure routine writes this audit. Quarantine
 -- and withdrawal do not. Historical Discover search snapshots are untouched.
 UPDATE public.candidate_network_recovery_audit SET content_hash=NULL
 WHERE publication_id IN (SELECT publication_id FROM public.candidate_network_publications WHERE subject_id=NEW.subject_id);
 UPDATE public.candidate_network_publications SET state='revoked',facts='{}',embedding=NULL,
  content_hash=repeat('0',64),lease_token=NULL,lease_until=NULL,revoked_at=coalesce(revoked_at,clock_timestamp())
 WHERE subject_id=NEW.subject_id;
 RETURN NEW;
END $$;
CREATE TRIGGER candidate_network_operator_erased AFTER INSERT ON public.candidate_profile_erasure_audit
FOR EACH ROW EXECUTE FUNCTION public.candidate_network_operator_erased();

CREATE FUNCTION public.candidate_network_claim() RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE p public.candidate_network_publications%ROWTYPE;
BEGIN
 SELECT * INTO p FROM public.candidate_network_publications WHERE state='staged' OR (state='embedding' AND lease_until<clock_timestamp())
 ORDER BY created_at,publication_id LIMIT 1;
 IF NOT FOUND THEN RETURN NULL;END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p.subject_id::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p.subject_id::text,0));
 SELECT * INTO p FROM public.candidate_network_publications WHERE publication_id=p.publication_id FOR UPDATE;
 IF p.state NOT IN ('staged','embedding') OR (p.state='embedding' AND p.lease_until>=clock_timestamp()) THEN RETURN NULL;END IF;
 IF NOT public.candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision) OR p.attempts>=3 THEN
  UPDATE public.candidate_network_publications SET state=CASE WHEN attempts>=3 THEN 'held' ELSE 'revoked' END,
   lease_token=NULL,lease_until=NULL WHERE publication_id=p.publication_id;RETURN NULL;
 END IF;
 UPDATE public.candidate_network_publications SET state='embedding',attempts=attempts+1,lease_token=gen_random_uuid(),lease_until=clock_timestamp()+interval '90 seconds'
 WHERE publication_id=p.publication_id RETURNING * INTO p;
 RETURN jsonb_build_object('publicationId',p.publication_id,'subjectId',p.subject_id,'profileVersion',p.profile_version,
 'preferenceVersion',p.preference_version,'consentRevision',p.consent_revision,'contentHash',p.content_hash,'facts',p.facts,'leaseToken',p.lease_token);
END $$;

CREATE FUNCTION public.candidate_network_finish(id uuid,token uuid,fingerprint text,vec vector) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE p public.candidate_network_publications%ROWTYPE;
BEGIN
 IF vec IS NULL OR vector_dims(vec)<>384 OR vector_norm(vec)=0 THEN RAISE EXCEPTION 'invalid_vector' USING ERRCODE='22023';END IF;
 SELECT * INTO p FROM public.candidate_network_publications WHERE publication_id=id;
 IF NOT FOUND THEN RETURN false;END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p.subject_id::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p.subject_id::text,0));
 SELECT * INTO p FROM public.candidate_network_publications WHERE publication_id=id FOR UPDATE;
 IF p.state<>'embedding' OR p.lease_token IS DISTINCT FROM token OR p.content_hash IS DISTINCT FROM fingerprint OR p.lease_until<clock_timestamp()
 THEN RETURN false;END IF;
 IF NOT public.candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision) THEN
  UPDATE public.candidate_network_publications SET state='revoked',lease_token=NULL,lease_until=NULL,revoked_at=clock_timestamp() WHERE publication_id=id;RETURN false;
 END IF;
 UPDATE public.candidate_network_publications SET state='revoked',lease_token=NULL,lease_until=NULL,revoked_at=clock_timestamp()
 WHERE subject_id=p.subject_id AND state='active' AND publication_id<>id;
 UPDATE public.candidate_network_publications SET state='active',embedding=vec,encoder_version=1,activated_at=clock_timestamp(),lease_token=NULL,lease_until=NULL WHERE publication_id=id;
 RETURN true;
END $$;

-- Explicit owner recovery only: workers and app roles may never reset their own cap.
CREATE TABLE public.candidate_network_recovery_audit (
 request_id uuid PRIMARY KEY,
 publication_id uuid NOT NULL REFERENCES public.candidate_network_publications(publication_id),
 content_hash text CHECK(content_hash ~ '^[0-9a-f]{64}$'),
 previous_attempts integer NOT NULL CHECK(previous_attempts BETWEEN 0 AND 3),
 operator_role name NOT NULL,
 recovered_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE public.candidate_network_recovery_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_network_recovery_audit FORCE ROW LEVEL SECURITY;
DO $$ BEGIN EXECUTE format('CREATE POLICY owner_only ON public.candidate_network_recovery_audit TO %I USING(true) WITH CHECK(true)',current_user); END $$;
REVOKE ALL ON public.candidate_network_recovery_audit FROM PUBLIC;

CREATE FUNCTION public.candidate_network_recover(id uuid,request uuid,fingerprint text) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE p public.candidate_network_publications%ROWTYPE;a public.candidate_network_recovery_audit%ROWTYPE;
BEGIN
 IF id IS NULL OR request IS NULL OR fingerprint IS NULL OR fingerprint !~ '^[0-9a-f]{64}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 SELECT * INTO p FROM public.candidate_network_publications WHERE publication_id=id;
 IF NOT FOUND THEN RETURN false;END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p.subject_id::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p.subject_id::text,0));
 SELECT * INTO p FROM public.candidate_network_publications WHERE publication_id=id FOR UPDATE;
 SELECT * INTO a FROM public.candidate_network_recovery_audit WHERE request_id=request;
 IF FOUND THEN
  IF a.publication_id IS DISTINCT FROM id OR a.content_hash IS DISTINCT FROM fingerprint
  THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505';END IF;
  RETURN true; -- Receipt replay never resets attempts or revives a revoked publication.
 END IF;
 IF p.state<>'held' OR p.content_hash IS DISTINCT FROM fingerprint
 OR NOT public.candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision)
 THEN RETURN false;END IF;
 INSERT INTO public.candidate_network_recovery_audit(request_id,publication_id,content_hash,previous_attempts,operator_role)
 VALUES(request,id,fingerprint,p.attempts,session_user);
 UPDATE public.candidate_network_publications SET state='staged',attempts=0,lease_token=NULL,lease_until=NULL
 WHERE publication_id=id;
 RETURN true;
END $$;

CREATE FUNCTION public.candidate_network_search(vec vector,n integer,filters jsonb) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET statement_timeout='3s' AS $$
BEGIN
 IF n IS NULL OR n NOT BETWEEN 1 AND 500 OR vec IS NULL OR vector_dims(vec)<>384 OR vector_norm(vec)=0
 OR jsonb_typeof(filters) IS DISTINCT FROM 'object' OR octet_length(filters::text)>16384
 OR NOT filters ?& ARRAY['cities','skills'] OR (SELECT count(*) FROM jsonb_object_keys(filters))<>2
 OR jsonb_typeof(filters->'cities') IS DISTINCT FROM 'array' OR jsonb_typeof(filters->'skills') IS DISTINCT FROM 'array'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 RETURN (SELECT coalesce(jsonb_agg(jsonb_build_object('kind','candidate_network','identity','network:'||p.subject_id::text,
 'globalCandidateId',p.global_candidate_id,
 'subjectId',p.subject_id,'publicationId',p.publication_id,'profileVersion',p.profile_version,'preferenceVersion',p.preference_version,
 'consentRevision',p.consent_revision,'contentHash',p.content_hash,'encoderVersion',1,'observedAt',p.activated_at,'facts',p.facts,'similarity',1-(p.embedding<=>vec))
 ORDER BY p.embedding<=>vec,p.publication_id),'[]') FROM (
 SELECT p.*,s.global_candidate_id FROM public.candidate_network_publications p
 JOIN public.candidate_network_subjects s USING(subject_id) WHERE p.state='active' AND p.encoder_version=1
 AND public.candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision)
 AND (jsonb_array_length(filters->'cities')=0 OR coalesce(p.facts#>>'{currentLocation,city}','')=''
  OR EXISTS(SELECT 1 FROM jsonb_array_elements_text(filters->'cities') c(city)
   WHERE strpos(lower(p.facts#>>'{currentLocation,city}'),lower(c.city))>0))
 AND (jsonb_array_length(filters->'skills')=0 OR EXISTS(
  SELECT 1 FROM jsonb_array_elements_text(filters->'skills') q(skill),jsonb_array_elements_text(p.facts->'skills') s(skill)
  WHERE lower(btrim(q.skill))=lower(btrim(s.skill))))
 ORDER BY p.embedding<=>vec,p.publication_id LIMIT n) p);
END $$;

CREATE FUNCTION public.candidate_network_owner_status(s uuid,b uuid) RETURNS boolean
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET statement_timeout='3s' AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=s AND flow_binding_id=b)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 RETURN EXISTS(SELECT 1 FROM public.candidate_network_publications p WHERE p.subject_id=s AND p.state='active'
  AND public.candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision));
END $$;

-- Include already-confirmed local profiles without requiring candidates to
-- confirm again. This stages only; the separate, default-off worker activates.
INSERT INTO public.candidate_network_publications(subject_id,profile_version,preference_version,consent_revision,facts,content_hash)
SELECT DISTINCT c.subject_id,c.profile_version,c.preference_version,c.consent_revision,
 public.candidate_network_projection(v.facts),
 encode(sha256(convert_to(public.candidate_profile_canonical(public.candidate_network_projection(v.facts)),'UTF8')),'hex')
FROM public.candidate_profile_confirmations c JOIN public.candidate_profile_versions v
 ON v.subject_id=c.subject_id AND v.version=c.profile_version
WHERE public.candidate_network_eligible(c.subject_id,c.profile_version,c.preference_version,c.consent_revision)
ON CONFLICT DO NOTHING;

CREATE FUNCTION public.candidate_network_validate(ids uuid[]) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET statement_timeout='3s' AS $$
BEGIN
 IF ids IS NULL OR cardinality(ids)>500 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 RETURN (SELECT coalesce(jsonb_agg(jsonb_build_object('publicationId',p.publication_id,'subjectId',p.subject_id,'contentHash',p.content_hash,
 'profileVersion',p.profile_version,'preferenceVersion',p.preference_version,'consentRevision',p.consent_revision) ORDER BY p.publication_id),'[]')
 FROM public.candidate_network_publications p WHERE p.publication_id=ANY(ids) AND p.state='active'
 AND public.candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision));
END $$;

DO $$ DECLARE f record;BEGIN
 FOR f IN SELECT c.oid::regclass target,r.rolname FROM pg_class c
 CROSS JOIN LATERAL aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a JOIN pg_roles r ON r.oid=a.grantee
 WHERE c.oid IN ('public.candidate_network_subjects'::regclass,'public.candidate_network_publications'::regclass,'public.candidate_network_recovery_audit'::regclass)
 AND a.grantee<>c.relowner LOOP
 EXECUTE format('REVOKE ALL ON TABLE %s FROM %I',f.target,f.rolname);END LOOP;
 FOR f IN SELECT p.oid::regprocedure target,r.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a JOIN pg_roles r ON r.oid=a.grantee
 WHERE n.nspname='public' AND starts_with(p.proname,'candidate_network_') AND a.grantee<>p.proowner LOOP
 EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',f.target,f.rolname);END LOOP;
 FOR f IN SELECT p.oid::regprocedure sig FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 WHERE n.nspname='public' AND starts_with(p.proname,'candidate_network_') LOOP
 EXECUTE format('REVOKE ALL ON FUNCTION %s FROM PUBLIC',f.sig);END LOOP;
END $$;
