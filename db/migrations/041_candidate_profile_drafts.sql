-- Candidate S2: private profile/preferences, reviewed proposals and consent v2.
-- Feature-disabled until application integration, exact readiness and role provisioning pass.
-- No publication or matching activation. Historical source comments below describe proof provenance.

-- Integrated source: memory-profile.sql
-- S2A executable storage candidate. Synthetic rehearsal only until retention,
-- migration numbering and the exact permission catalog are reviewed.
-- All object keys in the validated profile protocol are ASCII, so C ordering
-- agrees with the TypeScript/Python canonical encoders. User strings are values.
CREATE FUNCTION public.candidate_profile_canonical(j jsonb) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT SET search_path=pg_catalog,public AS $$
DECLARE t text;BEGIN
 CASE jsonb_typeof(j)
 WHEN 'object' THEN SELECT '{'||coalesce(string_agg(to_jsonb(key)::text||':'||public.candidate_profile_canonical(value),',' ORDER BY key COLLATE "C"),'')||'}' INTO t FROM jsonb_each(j);
 WHEN 'array' THEN SELECT '['||coalesce(string_agg(public.candidate_profile_canonical(value),',' ORDER BY ord),'')||']' INTO t FROM jsonb_array_elements(j) WITH ORDINALITY x(value,ord);
 ELSE t:=j::text;
 END CASE;RETURN t;
END $$;

-- Source helpers are created here but remain owner-only until the receiver below.
CREATE FUNCTION public.candidate_profile_source_date_valid(j jsonb) RETURNS boolean
LANGUAGE plpgsql IMMUTABLE SET search_path=pg_catalog,public AS $$
DECLARE v text; precision text; dt date;
BEGIN
 IF j='null'::jsonb THEN RETURN true; END IF;
 IF jsonb_typeof(j) IS DISTINCT FROM 'object' THEN RETURN false; END IF;
 IF (SELECT count(*) FROM jsonb_object_keys(j))<>2 OR NOT j ?& ARRAY['value','precision']
 OR jsonb_typeof(j->'value') IS DISTINCT FROM 'string' THEN RETURN false; END IF;
 v:=j->>'value';precision:=j->>'precision';
 IF precision='year' THEN
  RETURN v~'^[0-9]{4}$' AND v<>'0000';
 ELSIF precision='month' THEN
  IF v!~'^[0-9]{4}-[0-9]{2}$' THEN RETURN false; END IF;
  dt:=(v||'-01')::date;RETURN to_char(dt,'YYYY-MM')=v;
 ELSIF precision='day' THEN
  IF v!~'^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN RETURN false; END IF;
  dt:=v::date;RETURN to_char(dt,'YYYY-MM-DD')=v;
 END IF;
 RETURN false;
EXCEPTION WHEN OTHERS THEN RETURN false;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_source_date_valid(jsonb) FROM PUBLIC;

CREATE FUNCTION public.candidate_profile_source_fact_valid(j jsonb) RETURNS boolean
LANGUAGE plpgsql IMMUTABLE SET search_path=pg_catalog,public AS $$
DECLARE k text; v jsonb; fld text; maxlen integer; required text[]; n integer;
BEGIN
 IF jsonb_typeof(j) IS DISTINCT FROM 'object' THEN RETURN false; END IF;
 IF (SELECT count(*) FROM jsonb_object_keys(j))<>5 OR NOT j ?& ARRAY['factId','kind','targetRowId','value','evidence']
 OR jsonb_typeof(j->'factId') IS DISTINCT FROM 'string' OR j->>'factId'!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 OR jsonb_typeof(j->'evidence') IS DISTINCT FROM 'array' THEN RETURN false; END IF;
 IF jsonb_array_length(j->'evidence') NOT BETWEEN 1 AND 8 THEN RETURN false; END IF;
 k:=j->>'kind';v:=j->'value';
 IF k IS NULL OR k NOT IN ('displayName','headline','linkedinUrl','photo','currentLocation','skill','employment','education','project') THEN RETURN false; END IF;
 IF j->'targetRowId'<>'null'::jsonb AND (k NOT IN ('employment','education','project')
 OR jsonb_typeof(j->'targetRowId') IS DISTINCT FROM 'string' OR j->>'targetRowId'!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$') THEN RETURN false; END IF;
 IF k IN ('displayName','headline','skill') THEN
  IF jsonb_typeof(v) IS DISTINCT FROM 'string' THEN RETURN false; END IF;
  n:=length(v#>>'{}');
  RETURN n BETWEEN CASE WHEN k='skill' THEN 1 ELSE 0 END AND CASE WHEN k='displayName' THEN 200 WHEN k='headline' THEN 300 ELSE 100 END;
 ELSIF k='linkedinUrl' THEN
  RETURN v='null'::jsonb OR (jsonb_typeof(v)='string' AND v#>>'{}'~'^https://linkedin\.com/in/[A-Za-z0-9_%~-]+$');
 ELSIF k='photo' THEN
  IF jsonb_typeof(v) IS DISTINCT FROM 'object' THEN RETURN false; END IF;
  RETURN coalesce((SELECT count(*) FROM jsonb_object_keys(v))=1 AND jsonb_typeof(v->'url')='string'
   AND length(v->>'url')<=2048 AND v->>'url'~'^https://media[.]licdn[.]com/[A-Za-z0-9_~.%/?=&+,:;@!$()*-]+$',false);
 ELSIF k='currentLocation' THEN
  IF v='null'::jsonb THEN RETURN true; END IF;
  IF jsonb_typeof(v) IS DISTINCT FROM 'object' THEN RETURN false; END IF;
  IF (SELECT count(*) FROM jsonb_object_keys(v))<>3 OR NOT v ?& ARRAY['city','countryCode','displayCountry']
   OR jsonb_typeof(v->'countryCode') IS DISTINCT FROM 'string'
   OR NOT v->>'countryCode'=ANY(string_to_array('AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW',' ')) THEN RETURN false; END IF;
  FOREACH fld IN ARRAY ARRAY['city','displayCountry'] LOOP
   IF jsonb_typeof(v->fld) IS DISTINCT FROM 'string' OR length(v->>fld) NOT BETWEEN 1 AND 200 THEN RETURN false; END IF;
  END LOOP;RETURN true;
 END IF;
 IF jsonb_typeof(v) IS DISTINCT FROM 'object' THEN RETURN false; END IF;
 required:=CASE k WHEN 'employment' THEN ARRAY['id','start','end','current','employer','title','description']
 WHEN 'education' THEN ARRAY['id','start','end','current','institution','qualification','description']
 ELSE ARRAY['id','start','end','current','name','description'] END;
 IF NOT v ?& required OR (SELECT count(*) FROM jsonb_object_keys(v))<>cardinality(required)
 OR jsonb_typeof(v->'id') IS DISTINCT FROM 'string' OR v->>'id'!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 OR jsonb_typeof(v->'current') IS DISTINCT FROM 'boolean'
 OR NOT public.candidate_profile_source_date_valid(v->'start') OR NOT public.candidate_profile_source_date_valid(v->'end')
 OR (v->'current'='true'::jsonb AND v->'end'<>'null'::jsonb) THEN RETURN false; END IF;
 IF v->'start'<>'null'::jsonb AND v->'end'<>'null'::jsonb
 AND left((v#>>'{end,value}')||'-99-99',10)<left((v#>>'{start,value}')||'-01-01',10) THEN RETURN false; END IF;
 FOREACH fld IN ARRAY required LOOP
  IF fld IN ('id','start','end','current') THEN CONTINUE; END IF;
  maxlen:=CASE WHEN fld='description' THEN CASE WHEN k='education' THEN 5000 ELSE 10000 END WHEN fld='qualification' THEN 300 ELSE 200 END;
  IF jsonb_typeof(v->fld) IS DISTINCT FROM 'string' OR length(v->>fld)>maxlen
   OR (fld IN ('employer','title','institution','name') AND length(v->>fld)<1) THEN RETURN false; END IF;
 END LOOP;
 RETURN true;
EXCEPTION WHEN OTHERS THEN RETURN false;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_source_fact_valid(jsonb) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_profile_canonical(jsonb) FROM PUBLIC;
CREATE TABLE public.candidate_profile_heads (
 subject_id uuid PRIMARY KEY,binding_id uuid NOT NULL,
 draft_version bigint NOT NULL DEFAULT 0 CHECK(draft_version BETWEEN 0 AND 9007199254740991),
 profile_version bigint NOT NULL DEFAULT 0 CHECK(profile_version BETWEEN 0 AND 9007199254740991),
 preference_version bigint NOT NULL DEFAULT 0 CHECK(preference_version BETWEEN 0 AND 9007199254740991),
 restricted boolean NOT NULL DEFAULT false,
 FOREIGN KEY(subject_id,binding_id) REFERENCES public.candidate_owned_subjects(subject_id,flow_binding_id)
);
CREATE TABLE public.candidate_profile_drafts (
 subject_id uuid NOT NULL REFERENCES public.candidate_profile_heads(subject_id),version bigint NOT NULL CHECK(version>0),
 facts jsonb,document_sha256 text CHECK(document_sha256 IS NULL OR document_sha256~'^[0-9a-f]{64}$'),
 command_id uuid NOT NULL UNIQUE,created_at timestamptz NOT NULL DEFAULT clock_timestamp(),PRIMARY KEY(subject_id,version),
 CHECK(facts IS NULL OR jsonb_typeof(facts)='object')
);
CREATE TABLE public.candidate_profile_versions (
 subject_id uuid NOT NULL REFERENCES public.candidate_profile_heads(subject_id),version bigint NOT NULL CHECK(version>0),
 facts jsonb,document_sha256 text CHECK(document_sha256 IS NULL OR document_sha256~'^[0-9a-f]{64}$'),
 consent_version bigint NOT NULL DEFAULT 0 CHECK(consent_version=0),
 command_id uuid NOT NULL UNIQUE,confirmed_at timestamptz NOT NULL DEFAULT clock_timestamp(),PRIMARY KEY(subject_id,version),
 CHECK(facts IS NULL OR jsonb_typeof(facts)='object')
);
CREATE TABLE public.candidate_preference_versions (
 subject_id uuid NOT NULL REFERENCES public.candidate_profile_heads(subject_id),version bigint NOT NULL CHECK(version>0),
 preferences jsonb,command_id uuid NOT NULL UNIQUE,created_at timestamptz NOT NULL DEFAULT clock_timestamp(),PRIMARY KEY(subject_id,version),
 CHECK(preferences IS NULL OR jsonb_typeof(preferences)='object')
);
CREATE TABLE public.candidate_profile_receipts (
 command_id uuid PRIMARY KEY,subject_id uuid NOT NULL REFERENCES public.candidate_profile_heads(subject_id),binding_id uuid NOT NULL,
 idempotency_key uuid NOT NULL,wire_hash text NOT NULL CHECK(wire_hash~'^[0-9a-f]{64}$'),
 result jsonb NOT NULL,created_at timestamptz NOT NULL DEFAULT clock_timestamp(),UNIQUE(subject_id,idempotency_key)
);

-- Wire carries the canonical text, not PostgreSQL's differently spaced jsonb
-- rendering. Hashes therefore agree with both TypeScript and Python.
CREATE FUNCTION public.candidate_profile_apply(p_subject uuid,p_binding uuid,p_command uuid,p_hash text,p_wire jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE h public.candidate_profile_heads%ROWTYPE;r public.candidate_profile_receipts%ROWTYPE;
 j jsonb;a text;reason text;result jsonb;draft jsonb;prefs jsonb;dh text;ph text;v bigint;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_hash IS NULL OR p_hash!~'^[0-9a-f]{64}$'
  OR jsonb_typeof(p_wire) IS DISTINCT FROM 'string' OR octet_length(p_wire#>>'{}')>393216
  OR encode(sha256(convert_to(p_wire#>>'{}','UTF8')),'hex')<>p_hash
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 j:=(p_wire#>>'{}')::jsonb;
 IF jsonb_typeof(j) IS DISTINCT FROM 'object' OR (SELECT count(*) FROM jsonb_object_keys(j))<>7
  OR NOT j ?& ARRAY['schemaVersion','idempotencyKey','expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','action','payload']
  OR j->'schemaVersion' IS DISTINCT FROM '2'::jsonb OR jsonb_typeof(j->'payload') IS DISTINCT FROM 'object'
  OR j->>'action' IS NULL OR j->>'action' NOT IN ('save_draft','save_preferences','confirm')
  OR j->>'idempotencyKey' IS NULL OR j->>'idempotencyKey' !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  OR public.candidate_profile_canonical(j) IS DISTINCT FROM p_wire#>>'{}'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 FOREACH a IN ARRAY ARRAY['expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion'] LOOP
  IF jsonb_typeof(j->a) IS DISTINCT FROM 'number' OR (j->>a)!~'^(0|[1-9][0-9]{0,15})$' OR (j->>a)::numeric>9007199254740991
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) ON CONFLICT DO NOTHING;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding) THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF r.wire_hash<>p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  RETURN r.result;
 END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_receipts WHERE subject_id=p_subject AND idempotency_key=(j->>'idempotencyKey')::uuid)
 THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
 SELECT facts,document_sha256 INTO draft,dh FROM public.candidate_profile_drafts WHERE subject_id=p_subject AND version=h.draft_version;
 SELECT preferences INTO prefs FROM public.candidate_preference_versions WHERE subject_id=p_subject AND version=h.preference_version;
 a:=j->>'action';
 IF a='save_draft' THEN
  IF (SELECT count(*) FROM jsonb_object_keys(j->'payload'))<>1 OR jsonb_typeof(j#>'{payload,profile}') IS DISTINCT FROM 'object'
   OR octet_length(public.candidate_profile_canonical(j#>'{payload,profile}'))>262144
   OR NOT (j#>'{payload,profile}') ?& ARRAY['displayName','headline','linkedinUrl','currentLocation','skills','employment','education','projects']
   OR (SELECT count(*) FROM jsonb_object_keys(j#>'{payload,profile}'))<>(8+(CASE WHEN (j#>'{payload,profile}') ? 'photoUrl' THEN 1 ELSE 0 END))
   OR ((j#>'{payload,profile}') ? 'photoUrl' AND j#>'{payload,profile,photoUrl}'<>'null'::jsonb AND
    (jsonb_typeof(j#>'{payload,profile,photoUrl}') IS DISTINCT FROM 'string'
     OR length(j#>>'{payload,profile,photoUrl}')>2048
     OR j#>>'{payload,profile,photoUrl}'!~'^https://media[.]licdn[.]com/[A-Za-z0-9_~.%/?=&+,:;@!$()*-]+$'))
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
  IF h.draft_version<>(j->>'expectedDraftVersion')::bigint THEN reason:='stale_edit';
  ELSE
   v:=h.draft_version+1;
   INSERT INTO public.candidate_profile_drafts(subject_id,version,facts,document_sha256,command_id)
    VALUES(p_subject,v,j#>'{payload,profile}',encode(sha256(convert_to(public.candidate_profile_canonical(j#>'{payload,profile}'),'UTF8')),'hex'),p_command);
   UPDATE public.candidate_profile_drafts SET facts=NULL,document_sha256=NULL WHERE subject_id=p_subject AND version<>v;
   UPDATE public.candidate_profile_heads SET draft_version=v WHERE subject_id=p_subject RETURNING * INTO h;
  END IF;
 ELSIF a='save_preferences' THEN
  IF (SELECT count(*) FROM jsonb_object_keys(j->'payload'))<>1 OR jsonb_typeof(j#>'{payload,preferences}') IS DISTINCT FROM 'object'
   OR octet_length(public.candidate_profile_canonical(j#>'{payload,preferences}'))>32768
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
  IF h.preference_version<>(j->>'expectedPreferenceVersion')::bigint THEN reason:='stale_edit';
  ELSE
   v:=h.preference_version+1;
   INSERT INTO public.candidate_preference_versions(subject_id,version,preferences,command_id) VALUES(p_subject,v,j#>'{payload,preferences}',p_command);
   UPDATE public.candidate_preference_versions SET preferences=NULL WHERE subject_id=p_subject AND version<>v;
   UPDATE public.candidate_profile_heads SET preference_version=v WHERE subject_id=p_subject RETURNING * INTO h;
  END IF;
 ELSE
  IF (SELECT count(*) FROM jsonb_object_keys(j->'payload'))<>2
   OR NOT (j->'payload') ?& ARRAY['draftHash','consentVersion']
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
  IF (h.draft_version,h.profile_version,h.preference_version) IS DISTINCT FROM
    ((j->>'expectedDraftVersion')::bigint,(j->>'expectedProfileVersion')::bigint,(j->>'expectedPreferenceVersion')::bigint)
  THEN reason:='stale_edit';
  ELSIF j#>'{payload,consentVersion}' IS DISTINCT FROM '0'::jsonb OR j#>>'{payload,draftHash}' IS DISTINCT FROM dh THEN reason:='confirmation_conflict';
  ELSIF draft IS NULL OR prefs IS NULL OR coalesce(draft->>'displayName','')='' OR draft->'currentLocation'='null'::jsonb
   OR coalesce(jsonb_array_length(draft->'skills'),0)=0 THEN reason:='incomplete_profile';
  ELSE
   SELECT document_sha256 INTO ph FROM public.candidate_profile_versions WHERE subject_id=p_subject AND version=h.profile_version;
   IF ph IS DISTINCT FROM dh THEN
    v:=h.profile_version+1;
    INSERT INTO public.candidate_profile_versions(subject_id,version,facts,document_sha256,command_id) VALUES(p_subject,v,draft,dh,p_command);
    UPDATE public.candidate_profile_versions SET facts=NULL,document_sha256=NULL WHERE subject_id=p_subject AND version<>v;
    UPDATE public.candidate_profile_heads SET profile_version=v WHERE subject_id=p_subject RETURNING * INTO h;
   END IF;
  END IF;
 END IF;
 result:=jsonb_build_object('schemaVersion',2,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,'commandHash',p_hash,
  'status',CASE WHEN reason IS NULL THEN 'applied' ELSE 'rejected' END,'draftVersion',h.draft_version,'profileVersion',h.profile_version,
  'preferenceVersion',h.preference_version,'reason',reason,'publicationActive',false);
 INSERT INTO public.candidate_profile_receipts(command_id,subject_id,binding_id,idempotency_key,wire_hash,result)
  VALUES(p_command,p_subject,p_binding,(j->>'idempotencyKey')::uuid,p_hash,result);
 RETURN result;
END $$;

CREATE FUNCTION public.candidate_profile_read(p_subject uuid,p_binding uuid) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE h public.candidate_profile_heads%ROWTYPE;d jsonb;p jsonb;f jsonb;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT facts INTO d FROM public.candidate_profile_drafts WHERE subject_id=p_subject AND version=h.draft_version;
 SELECT facts INTO p FROM public.candidate_profile_versions WHERE subject_id=p_subject AND version=h.profile_version;
 SELECT preferences INTO f FROM public.candidate_preference_versions WHERE subject_id=p_subject AND version=h.preference_version;
 RETURN jsonb_build_object('schemaVersion',2,'draftVersion',coalesce(h.draft_version,0),'profileVersion',coalesce(h.profile_version,0),
  'preferenceVersion',coalesce(h.preference_version,0),'draft',d,'confirmedProfile',p,'preferences',f,'confirmedConsentVersion',0,'publicationActive',false);
END $$;

DO $$ DECLARE tab text;role_name text;sig text;BEGIN
 FOREACH tab IN ARRAY ARRAY['candidate_profile_heads','candidate_profile_drafts','candidate_profile_versions','candidate_preference_versions','candidate_profile_receipts'] LOOP
  EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC',tab);
  -- Remove inherited owner default grants as well; never assume clean defaults.
  FOR role_name IN SELECT DISTINCT r.rolname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
   CROSS JOIN LATERAL aclexplode(c.relacl) a JOIN pg_roles r ON r.oid=a.grantee
   WHERE n.nspname='public' AND c.relname=tab AND a.grantee<>c.relowner LOOP
   EXECUTE format('REVOKE ALL ON public.%I FROM %I',tab,role_name);
  END LOOP;
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',tab);
  EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY',tab);
  EXECUTE format('CREATE POLICY owner_only ON public.%I TO %I USING(true) WITH CHECK(true)',tab,current_user);
 END LOOP;
 FOREACH sig IN ARRAY ARRAY['candidate_profile_apply(uuid,uuid,uuid,text,jsonb)','candidate_profile_read(uuid,uuid)'] LOOP
  EXECUTE 'REVOKE ALL ON FUNCTION public.'||sig||' FROM PUBLIC';
 END LOOP;
 FOR sig,role_name IN SELECT p.oid::regprocedure::text,r.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
  CROSS JOIN LATERAL aclexplode(p.proacl) a JOIN pg_roles r ON r.oid=a.grantee
  WHERE n.nspname='public' AND p.proname IN ('candidate_profile_apply','candidate_profile_read','candidate_profile_canonical') AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',sig,role_name);
 END LOOP;
END $$;

-- Integrated source: memory-retention.sql
-- Unnumbered S2 erasure prototype. Owner-executed local rehearsal only.
-- The eventual authenticated privacy worker must validate the erasure authority
-- before calling this routine. No API/runtime role is granted execution here.
ALTER TABLE public.candidate_profile_receipts ALTER COLUMN wire_hash DROP NOT NULL;
CREATE TABLE public.candidate_profile_erasure_audit (
 request_id uuid PRIMARY KEY,
 subject_id uuid NOT NULL REFERENCES public.candidate_owned_subjects(subject_id),
 action text NOT NULL DEFAULT 'erasure' CHECK(action='erasure'),
 requester_kind text NOT NULL CHECK(requester_kind IN ('candidate','privacy_operator')),
 completed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE public.candidate_profile_erasure_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_profile_erasure_audit FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 EXECUTE format('CREATE POLICY owner_only ON public.candidate_profile_erasure_audit TO %I USING(true) WITH CHECK(true)',current_user);
END $$;

CREATE FUNCTION public.candidate_profile_erase(p_subject uuid,p_binding uuid,p_request uuid,p_requester text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE existing public.candidate_profile_erasure_audit%ROWTYPE;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_request IS NULL
 OR p_requester IS NULL OR p_requester NOT IN ('candidate','privacy_operator')
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 -- Same lock as apply/read: either the save commits first and is erased, or
 -- erasure commits first and apply refuses it before reading its old receipt.
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 SELECT * INTO existing FROM public.candidate_profile_erasure_audit WHERE request_id=p_request;
 IF FOUND THEN
  IF existing.subject_id<>p_subject THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  -- Replay must scrub again: an operator may be reapplying the retained
  -- erasure ledger after a restore. Never trust the audit row as proof that
  -- older payload tables have not been restored independently.
 END IF;
 -- A subject with no profile yet still needs the fence against delayed setup.
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id,restricted)
 VALUES(p_subject,p_binding,true) ON CONFLICT(subject_id) DO UPDATE SET restricted=true;
 UPDATE public.candidate_profile_drafts SET facts=NULL,document_sha256=NULL WHERE subject_id=p_subject;
 UPDATE public.candidate_profile_versions SET facts=NULL,document_sha256=NULL WHERE subject_id=p_subject;
 UPDATE public.candidate_preference_versions SET preferences=NULL WHERE subject_id=p_subject;
 UPDATE public.candidate_profile_receipts SET wire_hash=NULL,result=result-'commandHash' WHERE subject_id=p_subject;
 -- Counters and opaque receipt identities remain; never reset versions to zero.
 -- No application/private-organisation tables are touched by this routine.
 INSERT INTO public.candidate_profile_erasure_audit(request_id,subject_id,requester_kind)
 VALUES(p_request,p_subject,p_requester) ON CONFLICT(request_id) DO NOTHING;
 RETURN jsonb_build_object('requestId',p_request,'status','erased');
END $$;

REVOKE ALL ON TABLE public.candidate_profile_erasure_audit FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_profile_erase(uuid,uuid,uuid,text) FROM PUBLIC;
DO $$ DECLARE r record;BEGIN
 FOR r IN SELECT DISTINCT roles.rolname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
 CROSS JOIN LATERAL aclexplode(c.relacl) a JOIN pg_roles roles ON roles.oid=a.grantee
 WHERE n.nspname='public' AND c.relname='candidate_profile_erasure_audit' AND a.grantee<>c.relowner LOOP
  EXECUTE format('REVOKE ALL ON TABLE public.candidate_profile_erasure_audit FROM %I',r.rolname);
 END LOOP;
 FOR r IN SELECT DISTINCT roles.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 CROSS JOIN LATERAL aclexplode(p.proacl) a JOIN pg_roles roles ON roles.oid=a.grantee
 WHERE n.nspname='public' AND p.proname='candidate_profile_erase' AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION public.candidate_profile_erase(uuid,uuid,uuid,text) FROM %I',r.rolname);
 END LOOP;
END $$;

-- Integrated source: memory-command-expiry.sql
-- Unnumbered local S2 prototype. No runtime/API execution grant yet.
-- Flow drops an undelivered body at 30 days but must not claim it never arrived:
-- a response may have been lost after Memory committed. This routine either
-- returns that committed receipt or records an expiry under the apply lock.
CREATE FUNCTION public.candidate_profile_expire_command(p_subject uuid,p_binding uuid,p_command uuid,p_key uuid,p_hash text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE h public.candidate_profile_heads%ROWTYPE;r public.candidate_profile_receipts%ROWTYPE;answer jsonb;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_key IS NULL
 OR p_hash IS NULL OR p_hash!~'^[0-9a-f]{64}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) ON CONFLICT DO NOTHING;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 -- Never put erased fingerprints back, even for a delayed expiry request.
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding)
  THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash OR r.idempotency_key IS DISTINCT FROM p_key
  THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  RETURN r.result;
 END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_receipts WHERE subject_id=p_subject AND idempotency_key=p_key)
 THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
 answer:=jsonb_build_object('schemaVersion',2,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,
  'commandHash',p_hash,'status','rejected','draftVersion',h.draft_version,'profileVersion',h.profile_version,
  'preferenceVersion',h.preference_version,'reason','command_expired','publicationActive',false);
 INSERT INTO public.candidate_profile_receipts(command_id,subject_id,binding_id,idempotency_key,wire_hash,result)
 VALUES(p_command,p_subject,p_binding,p_key,p_hash,answer);
 RETURN answer;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_expire_command(uuid,uuid,uuid,uuid,text) FROM PUBLIC;
DO $$ DECLARE r record;BEGIN
 FOR r IN SELECT DISTINCT roles.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 CROSS JOIN LATERAL aclexplode(p.proacl) a JOIN pg_roles roles ON roles.oid=a.grantee
 WHERE n.nspname='public' AND p.proname='candidate_profile_expire_command' AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION public.candidate_profile_expire_command(uuid,uuid,uuid,uuid,text) FROM %I',r.rolname);
 END LOOP;
END $$;

-- Integrated source: memory-preference-draft.sql
-- Unnumbered S2 v3 local storage prototype. Requires memory-profile.sql.
-- No runtime grants. Full section semantic validation is the strict v3 parser;
-- this routine independently enforces identity, shape, revisions and receipts.
-- Not release-ready until consent projection, erasure and permission catalogues
-- are integrated and rehearsed against the combined full schema.
CREATE TABLE public.candidate_preference_drafts (
 subject_id uuid PRIMARY KEY REFERENCES public.candidate_profile_heads(subject_id),
 revision bigint NOT NULL DEFAULT 0 CHECK(revision BETWEEN 0 AND 9007199254740991),
 sections jsonb DEFAULT '{}' CHECK(sections IS NULL OR jsonb_typeof(sections)='object'),
 exclusions jsonb DEFAULT '[]' CHECK(exclusions IS NULL OR (jsonb_typeof(exclusions)='array' AND jsonb_array_length(exclusions)<=20)),
 content_hash text CHECK(content_hash IS NULL OR content_hash~'^[0-9a-f]{64}$'),
 promoted_preference_version bigint CHECK(promoted_preference_version BETWEEN 1 AND 9007199254740991),
 promoted_draft_revision bigint CHECK(promoted_draft_revision BETWEEN 0 AND 9007199254740991),
 updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 CHECK(octet_length(sections::text)<=65536)
);
ALTER TABLE public.candidate_preference_versions ADD COLUMN schema_version integer NOT NULL DEFAULT 2 CHECK(schema_version IN (2,3));
ALTER TABLE public.candidate_preference_versions ADD COLUMN source_draft_revision bigint CHECK(source_draft_revision BETWEEN 0 AND 9007199254740991);
ALTER TABLE public.candidate_preference_drafts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_preference_drafts FORCE ROW LEVEL SECURITY;
DO $$ BEGIN
 EXECUTE format('CREATE POLICY owner_only ON public.candidate_preference_drafts TO %I USING(true) WITH CHECK(true)',current_user);
END $$;
REVOKE ALL ON public.candidate_preference_drafts FROM PUBLIC;

-- Existing erasure sets the shared restricted fence under this same subject lock.
-- Scrub this new content copy in that transaction, retaining the monotone revision.
CREATE FUNCTION public.candidate_preference_draft_scrub() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$
BEGIN
 IF NEW.restricted THEN
  UPDATE public.candidate_preference_drafts SET sections=NULL,exclusions=NULL,content_hash=NULL WHERE subject_id=NEW.subject_id;
  UPDATE public.candidate_profile_receipts SET result=result-'preferenceDraftHash' WHERE subject_id=NEW.subject_id;
 END IF;
 RETURN NEW;
END $$;
REVOKE ALL ON FUNCTION public.candidate_preference_draft_scrub() FROM PUBLIC;
CREATE TRIGGER candidate_preference_draft_erasure AFTER UPDATE OF restricted ON public.candidate_profile_heads
FOR EACH ROW EXECUTE FUNCTION public.candidate_preference_draft_scrub();

CREATE FUNCTION public.candidate_preference_draft_apply(p_subject uuid,p_binding uuid,p_command uuid,p_hash text,p_wire text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE j jsonb; payload jsonb; h public.candidate_profile_heads%ROWTYPE;
 d public.candidate_preference_drafts%ROWTYPE; r public.candidate_profile_receipts%ROWTYPE;
 section text; field text; next_sections jsonb; result jsonb; reason text; changed boolean:=false; action text;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_hash IS NULL OR p_wire IS NULL
 OR octet_length(p_wire)>393216 OR p_hash!~'^[0-9a-f]{64}$'
 OR encode(sha256(convert_to(p_wire,'UTF8')),'hex')<>p_hash THEN
 RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 j:=p_wire::jsonb;
 IF public.candidate_profile_canonical(j) IS DISTINCT FROM p_wire
 OR jsonb_typeof(j) IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF (SELECT count(*) FROM jsonb_object_keys(j))<>8
 OR NOT j ?& ARRAY['schemaVersion','idempotencyKey','expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision','action','payload']
 OR j->'schemaVersion' IS DISTINCT FROM '3'::jsonb OR j->>'action' IS NULL OR j->>'action' NOT IN ('save_preference_section','promote_preferences')
 OR j->>'idempotencyKey' IS NULL OR j->>'idempotencyKey' !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 FOREACH field IN ARRAY ARRAY['expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision'] LOOP
  IF jsonb_typeof(j->field) IS DISTINCT FROM 'number' OR (j->>field)!~'^(0|[1-9][0-9]{0,15})$' OR (j->>field)::numeric>9007199254740991
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 payload:=j->'payload';
 action:=j->>'action';
 IF jsonb_typeof(payload) IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF action='save_preference_section' THEN
 IF (SELECT count(*) FROM jsonb_object_keys(payload))<>3 OR NOT payload ?& ARRAY['section','answered','value']
 OR jsonb_typeof(payload->'answered') IS DISTINCT FROM 'boolean'
 OR octet_length(public.candidate_profile_canonical(payload))>32768
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 section:=payload->>'section';
 IF section IS NULL OR section NOT IN ('roles','location','arrangement','relocation','employment_types','availability','compensation','authorisation','private_notes')
 OR (payload->'answered'='false'::jsonb AND payload->'value' IS DISTINCT FROM 'null'::jsonb)
 OR (payload->'answered'='true'::jsonb AND jsonb_typeof(payload->'value') IS DISTINCT FROM
 CASE WHEN section IN ('employment_types','authorisation','private_notes') THEN 'array' ELSE 'object' END)
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 ELSE
  IF (SELECT count(*) FROM jsonb_object_keys(payload))<>1 OR NOT payload ? 'preferenceDraftHash'
  OR jsonb_typeof(payload->'preferenceDraftHash') IS DISTINCT FROM 'string' OR payload->>'preferenceDraftHash' !~ '^[0-9a-f]{64}$'
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) ON CONFLICT DO NOTHING;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding) THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  RETURN r.result;
 END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE subject_id=p_subject AND idempotency_key=(j->>'idempotencyKey')::uuid;
 IF FOUND THEN
  IF r.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  -- Admission must reuse the original command ID; never return a mismatched
  -- command ID as if this second delivery had been acknowledged.
  RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505';
 END IF;
 INSERT INTO public.candidate_preference_drafts(subject_id,content_hash)
 VALUES(p_subject,encode(sha256(convert_to('{"exclusions":[],"sections":{}}','UTF8')),'hex')) ON CONFLICT DO NOTHING;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject FOR UPDATE;
 IF d.revision<>(j->>'expectedPreferenceDraftRevision')::bigint THEN reason:='stale_edit';
 ELSIF action='save_preference_section' THEN
  next_sections:=jsonb_set(d.sections,ARRAY[section],payload-'section',true);
  changed:=next_sections IS DISTINCT FROM d.sections;
  IF changed THEN
   UPDATE public.candidate_preference_drafts SET revision=revision+1,sections=next_sections,
    content_hash=encode(sha256(convert_to(public.candidate_profile_canonical(jsonb_build_object('sections',next_sections,'exclusions',d.exclusions)),'UTF8')),'hex'),updated_at=clock_timestamp()
   WHERE subject_id=p_subject RETURNING * INTO d;
  END IF;
 ELSE
  IF h.preference_version<>(j->>'expectedPreferenceVersion')::bigint THEN reason:='stale_edit';
  ELSIF payload->>'preferenceDraftHash' IS DISTINCT FROM d.content_hash THEN reason:='confirmation_conflict';
  ELSIF EXISTS(SELECT 1 FROM public.candidate_preference_versions WHERE subject_id=p_subject AND version=h.preference_version AND schema_version=2)
   THEN reason:='legacy_preferences_review_required';
  ELSIF EXISTS(SELECT 1 FROM unnest(ARRAY['roles','location','arrangement','relocation','employment_types','availability','compensation']) required(name)
   WHERE d.sections#>ARRAY[required.name,'answered'] IS DISTINCT FROM 'true'::jsonb)
   THEN reason:='incomplete_preferences';
  ELSIF d.promoted_draft_revision IS DISTINCT FROM d.revision OR d.promoted_preference_version IS DISTINCT FROM h.preference_version THEN
   INSERT INTO public.candidate_preference_versions(subject_id,version,preferences,command_id,schema_version,source_draft_revision)
   VALUES(p_subject,h.preference_version+1,jsonb_build_object('schemaVersion',3,'sections',d.sections,'exclusions',d.exclusions),p_command,3,d.revision);
   UPDATE public.candidate_preference_versions SET preferences=NULL WHERE subject_id=p_subject AND version<>h.preference_version+1;
   UPDATE public.candidate_profile_heads SET preference_version=preference_version+1 WHERE subject_id=p_subject RETURNING * INTO h;
   UPDATE public.candidate_preference_drafts SET promoted_preference_version=h.preference_version,promoted_draft_revision=revision
   WHERE subject_id=p_subject RETURNING * INTO d;
   changed:=true;
  END IF;
 END IF;
 result:=jsonb_build_object('schemaVersion',3,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,'commandHash',p_hash,
 'status',CASE WHEN reason IS NULL THEN 'applied' ELSE 'rejected' END,'reason',reason,'changed',changed,
 'draftVersion',h.draft_version,'profileVersion',h.profile_version,'preferenceVersion',h.preference_version,
 'preferenceDraftRevision',d.revision,'preferenceDraftHash',d.content_hash,'publicationActive',false);
 INSERT INTO public.candidate_profile_receipts(command_id,subject_id,binding_id,idempotency_key,wire_hash,result)
 VALUES(p_command,p_subject,p_binding,(j->>'idempotencyKey')::uuid,p_hash,result);
 RETURN result;
END $$;
REVOKE ALL ON FUNCTION public.candidate_preference_draft_apply(uuid,uuid,uuid,text,text) FROM PUBLIC;

CREATE FUNCTION public.candidate_profile_v3_storage_read(p_subject uuid,p_binding uuid) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE h public.candidate_profile_heads%ROWTYPE; d public.candidate_preference_drafts%ROWTYPE;
 draft jsonb; confirmed jsonb; preferences jsonb; preference_draft jsonb;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT facts INTO draft FROM public.candidate_profile_drafts WHERE subject_id=p_subject AND version=h.draft_version;
 SELECT facts INTO confirmed FROM public.candidate_profile_versions WHERE subject_id=p_subject AND version=h.profile_version;
 SELECT v.preferences INTO preferences FROM public.candidate_preference_versions v WHERE subject_id=p_subject AND version=h.preference_version;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject;
 IF FOUND THEN preference_draft:=jsonb_build_object('preferenceDraftRevision',d.revision,'preferenceDraftHash',d.content_hash,'sections',d.sections,'exclusions',d.exclusions,
  'promotedPreferenceVersion',d.promoted_preference_version,'promotedDraftRevision',d.promoted_draft_revision); END IF;
 RETURN jsonb_build_object('schemaVersion',3,'draftVersion',coalesce(h.draft_version,0),'profileVersion',coalesce(h.profile_version,0),
  'preferenceVersion',coalesce(h.preference_version,0),'draft',draft,'confirmedProfile',confirmed,'preferences',preferences,
  'preferenceDraft',preference_draft,'publicationActive',false,
  'sources',coalesce((SELECT jsonb_agg(jsonb_build_object('sourceId',s.source_id,'sourceVersion',s.source_version) ORDER BY s.created_at DESC,s.source_id)
   FROM (SELECT * FROM public.candidate_profile_sources WHERE subject_id=p_subject AND content_hash IS NOT NULL ORDER BY created_at DESC,source_id LIMIT 50) s),'[]'::jsonb),
  'hasMoreSources',(SELECT count(*)>50 FROM public.candidate_profile_sources WHERE subject_id=p_subject AND content_hash IS NOT NULL),
  'proposals',coalesce((SELECT jsonb_agg(value ORDER BY created_at DESC,id) FROM (
   SELECT jsonb_build_object('proposalId',proposal_id,'proposalHash',proposal_hash,'kind','source','sourceId',source_id,
    'sourceVersion',source_version,'expiresAt',to_char(expires_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')) value,created_at,proposal_id id
   FROM public.candidate_profile_proposals WHERE subject_id=p_subject AND binding_id=p_binding AND state='proposed' AND payload IS NOT NULL AND expires_at>clock_timestamp()
   UNION ALL
   SELECT jsonb_build_object('proposalId',proposal_id,'proposalHash',proposal_hash,'kind','interpretation','sourceId',NULL,
    'sourceVersion',source_version,'expiresAt',to_char(expires_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')),created_at,proposal_id
   FROM public.candidate_profile_interpretations WHERE subject_id=p_subject AND binding_id=p_binding AND state='proposed' AND payload IS NOT NULL AND expires_at>clock_timestamp()
   ORDER BY created_at DESC,id LIMIT 50) p),'[]'::jsonb),
  'hasMoreProposals',((SELECT count(*) FROM public.candidate_profile_proposals WHERE subject_id=p_subject AND binding_id=p_binding AND state='proposed' AND payload IS NOT NULL AND expires_at>clock_timestamp())+
   (SELECT count(*) FROM public.candidate_profile_interpretations WHERE subject_id=p_subject AND binding_id=p_binding AND state='proposed' AND payload IS NOT NULL AND expires_at>clock_timestamp()))>50);
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_v3_storage_read(uuid,uuid) FROM PUBLIC;

-- Clean inherited default grants too; an owner's defaults are not an authority.
DO $$ DECLARE row record; BEGIN
 FOR row IN SELECT c.oid::regclass AS target,r.rolname FROM pg_class c
 CROSS JOIN LATERAL aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a JOIN pg_roles r ON r.oid=a.grantee
 WHERE c.oid='public.candidate_preference_drafts'::regclass AND a.grantee<>c.relowner LOOP
  EXECUTE format('REVOKE ALL ON TABLE %s FROM %I',row.target,row.rolname);
 END LOOP;
 FOR row IN SELECT p.oid::regprocedure AS target,r.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a JOIN pg_roles r ON r.oid=a.grantee
 WHERE n.nspname='public' AND p.proname IN ('candidate_preference_draft_apply','candidate_preference_draft_scrub','candidate_profile_v3_storage_read') AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',row.target,row.rolname);
 END LOOP;
END $$;

-- Integrated source: memory-consent-v2.sql
-- UNNUMBERED LOCAL PROTOTYPE. Extends the existing consent projection/receipts;
-- it is not a second consent authority. Flow's existing ledger remains authority.
-- Requires real migration 027 plus S1/S2 profile storage. No runtime grants.
-- Production requires exact old-definition pins and full-schema rehearsal.
ALTER TABLE public.candidate_consent_state ADD COLUMN effective_schema_version integer NOT NULL DEFAULT 1 CHECK(effective_schema_version IN (1,2));
ALTER TABLE public.candidate_consent_state ADD COLUMN effective_copy_version integer;
ALTER TABLE public.candidate_consent_state ADD COLUMN effective_copy_sha256 text;
ALTER TABLE public.candidate_consent_state ADD COLUMN effective_purposes text[];
ALTER TABLE public.candidate_consent_receipts ADD COLUMN authority_schema_version integer NOT NULL DEFAULT 1 CHECK(authority_schema_version IN (1,2));
ALTER TABLE public.candidate_consent_receipts ADD COLUMN copy_version integer;
ALTER TABLE public.candidate_consent_receipts ADD COLUMN copy_sha256 text;
ALTER TABLE public.candidate_consent_receipts ADD COLUMN purposes text[];

-- Locate narrowly by column dependency, refusing unexpected catalogue shape.
-- Names for the old unnamed checks are PostgreSQL-generated.
DO $$ DECLARE names text[]; n text; BEGIN
 SELECT array_agg(conname) INTO names FROM pg_constraint WHERE conrelid='public.candidate_consent_state'::regclass AND contype='c'
 AND pg_get_constraintdef(oid) LIKE '%effective_version%' AND pg_get_constraintdef(oid) LIKE '%active_source_id%';
 IF cardinality(names) IS DISTINCT FROM 1 THEN RAISE EXCEPTION 'consent_state_catalog_mismatch'; END IF;
 EXECUTE format('ALTER TABLE public.candidate_consent_state DROP CONSTRAINT %I',names[1]);
 SELECT array_agg(conname) INTO names FROM pg_constraint WHERE conrelid='public.candidate_consent_receipts'::regclass AND contype='c'
 AND pg_get_constraintdef(oid) LIKE '%outcome%';
 IF cardinality(names) IS DISTINCT FROM 2 THEN RAISE EXCEPTION 'consent_receipt_catalog_mismatch'; END IF;
 FOREACH n IN ARRAY names LOOP EXECUTE format('ALTER TABLE public.candidate_consent_receipts DROP CONSTRAINT %I',n); END LOOP;
END $$;
ALTER TABLE public.candidate_consent_state ADD CONSTRAINT consent_effective_authority_shape CHECK ((
 (effective_version=0 AND effective_action IS NULL AND effective_at IS NULL AND active_source_id IS NULL)
 OR (effective_version>0 AND effective_action='withdraw' AND effective_at IS NOT NULL AND active_source_id IS NULL)
 OR (effective_version>0 AND effective_action='grant' AND effective_at IS NOT NULL AND
  ((effective_schema_version=1 AND active_source_id IS NOT NULL AND global_candidate_id IS NOT NULL)
   OR (effective_schema_version=2 AND active_source_id IS NULL AND effective_copy_version>0
       AND effective_copy_sha256~'^[0-9a-f]{64}$' AND effective_purposes=ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact'])))
) IS TRUE);
ALTER TABLE public.candidate_consent_receipts ADD CONSTRAINT consent_receipt_v2_outcome CHECK(outcome IN ('granted','withdrawn','superseded','identity_review_required','onboarding_granted'));
ALTER TABLE public.candidate_consent_receipts ADD CONSTRAINT consent_receipt_v2_shape CHECK ((
 (authority_schema_version=1 AND outcome<>'onboarding_granted' AND
   ((outcome='granted' AND source_id IS NOT NULL AND global_candidate_id IS NOT NULL) OR (outcome<>'granted' AND source_id IS NULL)))
 OR (authority_schema_version=2 AND source_id IS NULL AND outcome IN ('onboarding_granted','withdrawn','superseded') AND
  ((action='withdraw' AND purposes='{}'::text[] AND copy_version IS NULL AND copy_sha256 IS NULL)
   OR (action='grant' AND copy_version>0 AND copy_sha256~'^[0-9a-f]{64}$'
       AND purposes=ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact'])))
) IS TRUE);

CREATE TABLE public.candidate_consent_copy_catalog (
 copy_version integer PRIMARY KEY CHECK(copy_version>0),copy_sha256 text NOT NULL CHECK(copy_sha256~'^[0-9a-f]{64}$'),
 approved boolean NOT NULL DEFAULT false,UNIQUE(copy_version,copy_sha256)
);
REVOKE ALL ON public.candidate_consent_copy_catalog FROM PUBLIC;
ALTER TABLE public.candidate_consent_copy_catalog ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_consent_copy_catalog FORCE ROW LEVEL SECURITY;
DO $$ BEGIN EXECUTE format('CREATE POLICY owner_only ON public.candidate_consent_copy_catalog TO %I USING(true) WITH CHECK(true)',current_user); END $$;

CREATE FUNCTION public.candidate_consent_v2_project(p_subject uuid,p_binding uuid,p_event uuid,p_version bigint,p_action text,
 p_copy_version integer,p_copy_hash text,p_purposes text[],p_digest text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE s public.candidate_consent_state%ROWTYPE; r public.candidate_consent_receipts%ROWTYPE; result text; tenant text;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_event IS NULL OR p_version IS NULL OR p_version NOT BETWEEN 1 AND 9007199254740991
 OR p_action IS NULL OR p_action NOT IN ('grant','withdraw') OR p_digest IS NULL OR p_digest!~'^[0-9a-f]{64}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF encode(sha256(convert_to(public.candidate_profile_canonical(jsonb_build_object('schemaVersion',2,'subjectId',p_subject,'bindingId',p_binding,
  'eventId',p_event,'version',p_version,'action',p_action,'copyVersion',p_copy_version,'copyHash',p_copy_hash,'purposes',p_purposes)),'UTF8')),'hex')<>p_digest
 THEN RAISE EXCEPTION 'invalid_digest' USING ERRCODE='22023'; END IF;
 IF p_action='grant' THEN
  IF p_purposes IS DISTINCT FROM ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact']
  OR p_copy_version IS NULL OR p_copy_hash IS NULL OR p_copy_hash!~'^[0-9a-f]{64}$'
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 ELSE
  IF p_purposes IS DISTINCT FROM '{}'::text[] OR p_copy_version IS NOT NULL OR p_copy_hash IS NOT NULL
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END IF;
 tenant:='candidate_'||p_subject::text;
 PERFORM set_config('app.current_tenant_id',tenant,true);
 -- All v2 authority changes and confirmation take these locks in this order.
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p_subject::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 IF p_action='grant' AND EXISTS(SELECT 1 FROM public.candidate_profile_heads WHERE subject_id=p_subject AND restricted)
 THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT * INTO r FROM public.candidate_consent_receipts WHERE event_id=p_event;
 IF FOUND THEN
  IF r.subject_id<>p_subject THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF (r.version,r.action,r.command_digest::text,r.authority_schema_version,r.copy_version,r.copy_sha256,r.purposes)
   IS DISTINCT FROM (p_version,p_action,p_digest,2,p_copy_version,p_copy_hash,p_purposes)
  THEN RAISE EXCEPTION 'consent_conflict' USING ERRCODE='23505'; END IF;
  RETURN jsonb_build_object('eventId',p_event,'version',r.version,'outcome',r.outcome,'effectiveVersion',r.effective_version);
 END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_consent_receipts WHERE subject_id=p_subject AND version=p_version)
 THEN RAISE EXCEPTION 'consent_conflict' USING ERRCODE='23505'; END IF;
 INSERT INTO public.candidate_consent_state(subject_id,tenant_id) VALUES(p_subject,tenant) ON CONFLICT DO NOTHING;
 SELECT * INTO s FROM public.candidate_consent_state WHERE subject_id=p_subject FOR UPDATE;
 IF p_version<=s.highest_version THEN result:='superseded';
 ELSE
  IF p_action='grant' AND NOT EXISTS(SELECT 1 FROM public.candidate_consent_copy_catalog WHERE copy_version=p_copy_version AND copy_sha256=p_copy_hash AND approved)
  THEN RAISE EXCEPTION 'consent_copy_unapproved' USING ERRCODE='42501'; END IF;
  result:=CASE WHEN p_action='grant' THEN 'onboarding_granted' ELSE 'withdrawn' END;
  UPDATE public.candidate_consent_state SET highest_version=p_version,last_action=p_action,effective_version=p_version,effective_action=p_action,
   active_source_id=NULL,effective_at=clock_timestamp(),effective_schema_version=2,effective_copy_version=p_copy_version,
   effective_copy_sha256=p_copy_hash,effective_purposes=p_purposes,updated_at=clock_timestamp()
   WHERE subject_id=p_subject RETURNING * INTO s;
 END IF;
 INSERT INTO public.candidate_consent_receipts(event_id,idempotency_key,command_digest,subject_id,tenant_id,version,action,outcome,
  global_candidate_id,effective_version,effective_action,verified_issuer,verified_actor_id,authority_schema_version,copy_version,copy_sha256,purposes)
 VALUES(p_event,encode(sha256(convert_to('spark-consent:'||p_event::text,'UTF8')),'hex'),p_digest,p_subject,tenant,p_version,p_action,result,
  s.global_candidate_id,s.effective_version,s.effective_action,'vantahire','vantahire-backend',2,p_copy_version,p_copy_hash,p_purposes);
 RETURN jsonb_build_object('eventId',p_event,'version',p_version,'outcome',result,'effectiveVersion',s.effective_version);
END $$;
REVOKE ALL ON FUNCTION public.candidate_consent_v2_project(uuid,uuid,uuid,bigint,text,integer,text,text[],text) FROM PUBLIC;

-- No runtime role may directly execute the projection until the signed Flow
-- ledger adapter verifies email/current identity, CSRF, copy and wire digest.
-- Legacy v1 writers must set effective_schema_version=1 on their state updates;
-- even without that change, the confirmation receipt join below rejects v1.

-- Integrated source: memory-confirm-v3.sql
-- Local, unnumbered prototype. No API/runtime grants or search publication.
-- Apply after memory-consent-v2 and memory-preference-draft prototypes.
ALTER TABLE public.candidate_profile_versions DROP CONSTRAINT candidate_profile_versions_consent_version_check;
ALTER TABLE public.candidate_profile_versions ADD CONSTRAINT candidate_profile_versions_consent_version_check CHECK(consent_version BETWEEN 0 AND 9007199254740991);
CREATE TABLE public.candidate_profile_confirmations (
 command_id uuid PRIMARY KEY REFERENCES public.candidate_profile_receipts(command_id) DEFERRABLE INITIALLY DEFERRED,
 subject_id uuid NOT NULL REFERENCES public.candidate_profile_heads(subject_id),
 draft_version bigint NOT NULL,profile_version bigint NOT NULL,preference_version bigint NOT NULL,
 preference_draft_revision bigint NOT NULL,consent_revision bigint NOT NULL,
 confirmed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE public.candidate_profile_confirmations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_profile_confirmations FORCE ROW LEVEL SECURITY;
DO $$ BEGIN EXECUTE format('CREATE POLICY owner_only ON public.candidate_profile_confirmations TO %I USING(true) WITH CHECK(true)',current_user); END $$;
REVOKE ALL ON public.candidate_profile_confirmations FROM PUBLIC;

CREATE FUNCTION public.candidate_profile_confirm_v3(p_subject uuid,p_binding uuid,p_command uuid,p_hash text,p_wire text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE j jsonb; k text; h public.candidate_profile_heads%ROWTYPE; d public.candidate_preference_drafts%ROWTYPE;
 r public.candidate_profile_receipts%ROWTYPE; consent public.candidate_consent_state%ROWTYPE;
 draft jsonb; draft_hash text; confirmed_hash text; reason text; result jsonb; changed boolean:=false;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_hash IS NULL OR p_wire IS NULL
 OR octet_length(p_wire)>393216 OR p_hash!~'^[0-9a-f]{64}$'
 OR encode(sha256(convert_to(p_wire,'UTF8')),'hex')<>p_hash THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 j:=p_wire::jsonb;
 IF jsonb_typeof(j) IS DISTINCT FROM 'object' OR public.candidate_profile_canonical(j) IS DISTINCT FROM p_wire
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF (SELECT count(*) FROM jsonb_object_keys(j))<>8 OR NOT j ?& ARRAY['schemaVersion','idempotencyKey','expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision','action','payload']
 OR j->'schemaVersion' IS DISTINCT FROM '3'::jsonb OR j->>'action' IS DISTINCT FROM 'confirm'
 OR j->>'idempotencyKey' IS NULL OR j->>'idempotencyKey'!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 OR jsonb_typeof(j->'payload') IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 FOREACH k IN ARRAY ARRAY['expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision'] LOOP
  IF jsonb_typeof(j->k) IS DISTINCT FROM 'number' OR (j->>k)!~'^(0|[1-9][0-9]{0,15})$' OR (j->>k)::numeric>9007199254740991
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF (SELECT count(*) FROM jsonb_object_keys(j->'payload'))<>3 OR NOT (j->'payload') ?& ARRAY['draftHash','preferenceDraftHash','consentRevision']
 OR jsonb_typeof(j#>'{payload,draftHash}') IS DISTINCT FROM 'string' OR j#>>'{payload,draftHash}'!~'^[0-9a-f]{64}$'
 OR jsonb_typeof(j#>'{payload,preferenceDraftHash}') IS DISTINCT FROM 'string' OR j#>>'{payload,preferenceDraftHash}'!~'^[0-9a-f]{64}$'
 OR jsonb_typeof(j#>'{payload,consentRevision}') IS DISTINCT FROM 'number' OR j#>>'{payload,consentRevision}'!~'^(0|[1-9][0-9]{0,15})$'
 OR (j#>>'{payload,consentRevision}')::numeric>9007199254740991 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 PERFORM set_config('app.current_tenant_id','candidate_'||p_subject::text,true);
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p_subject::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.subject_id IS NOT NULL AND h.binding_id IS DISTINCT FROM p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding) THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  -- Historical receipt is not a current permission/publication assertion.
  RETURN r.result;
 END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_receipts WHERE subject_id=p_subject AND idempotency_key=(j->>'idempotencyKey')::uuid)
 THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
 IF h.subject_id IS NULL THEN
  INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) RETURNING * INTO h;
 END IF;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject;
 SELECT facts,document_sha256 INTO draft,draft_hash FROM public.candidate_profile_drafts WHERE subject_id=p_subject AND version=h.draft_version;
 SELECT * INTO consent FROM public.candidate_consent_state WHERE subject_id=p_subject;
 IF consent.effective_action IS DISTINCT FROM 'grant' OR consent.effective_schema_version IS DISTINCT FROM 2
 OR consent.highest_version IS DISTINCT FROM consent.effective_version
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_receipts cr WHERE cr.subject_id=p_subject AND cr.version=consent.effective_version
   AND cr.authority_schema_version=2 AND cr.outcome='onboarding_granted' AND cr.purposes=ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact']
   AND cr.copy_version=consent.effective_copy_version AND cr.copy_sha256=consent.effective_copy_sha256)
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_copy_catalog cc WHERE cc.copy_version=consent.effective_copy_version AND cc.copy_sha256=consent.effective_copy_sha256 AND cc.approved)
 THEN reason:='consent_required';
 ELSIF consent.effective_version<>(j#>>'{payload,consentRevision}')::bigint THEN reason:='consent_changed';
 ELSIF (h.draft_version,h.profile_version,h.preference_version,coalesce(d.revision,0)) IS DISTINCT FROM
 ((j->>'expectedDraftVersion')::bigint,(j->>'expectedProfileVersion')::bigint,(j->>'expectedPreferenceVersion')::bigint,(j->>'expectedPreferenceDraftRevision')::bigint)
 THEN reason:='stale_edit';
 ELSIF d.subject_id IS NULL OR d.promoted_draft_revision IS DISTINCT FROM d.revision OR d.promoted_preference_version IS DISTINCT FROM h.preference_version
 OR j#>>'{payload,draftHash}' IS DISTINCT FROM draft_hash OR j#>>'{payload,preferenceDraftHash}' IS DISTINCT FROM d.content_hash
 THEN reason:='confirmation_conflict';
 ELSIF draft IS NULL OR coalesce(draft->>'displayName','')='' OR draft->'currentLocation' IS NULL OR draft->'currentLocation'='null'::jsonb
 OR coalesce(jsonb_array_length(draft->'skills'),0)=0 THEN reason:='incomplete_profile';
 ELSE
  SELECT document_sha256 INTO confirmed_hash FROM public.candidate_profile_versions WHERE subject_id=p_subject AND version=h.profile_version;
  IF confirmed_hash IS DISTINCT FROM draft_hash THEN
   INSERT INTO public.candidate_profile_versions(subject_id,version,facts,document_sha256,consent_version,command_id)
   VALUES(p_subject,h.profile_version+1,draft,draft_hash,consent.effective_version,p_command);
   UPDATE public.candidate_profile_versions SET facts=NULL,document_sha256=NULL WHERE subject_id=p_subject AND version<>h.profile_version+1;
   UPDATE public.candidate_profile_heads SET profile_version=profile_version+1 WHERE subject_id=p_subject RETURNING * INTO h;
  END IF;
  INSERT INTO public.candidate_profile_confirmations(command_id,subject_id,draft_version,profile_version,preference_version,preference_draft_revision,consent_revision)
  VALUES(p_command,p_subject,h.draft_version,h.profile_version,h.preference_version,d.revision,consent.effective_version);
  changed:=true;
 END IF;
 result:=jsonb_build_object('schemaVersion',3,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,'commandHash',p_hash,
 'status',CASE WHEN reason IS NULL THEN 'applied' ELSE 'rejected' END,'reason',reason,'changed',changed,
 'draftVersion',h.draft_version,'profileVersion',h.profile_version,'preferenceVersion',h.preference_version,
 'preferenceDraftRevision',coalesce(d.revision,0),'preferenceDraftHash',d.content_hash,'publicationActive',false);
 INSERT INTO public.candidate_profile_receipts(command_id,subject_id,binding_id,idempotency_key,wire_hash,result)
 VALUES(p_command,p_subject,p_binding,(j->>'idempotencyKey')::uuid,p_hash,result);
 RETURN result;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_confirm_v3(uuid,uuid,uuid,text,text) FROM PUBLIC;

DO $$ DECLARE r record; BEGIN
 FOR r IN SELECT c.oid::regclass AS target,roles.rolname FROM pg_class c
 CROSS JOIN LATERAL aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a JOIN pg_roles roles ON roles.oid=a.grantee
 WHERE c.oid IN ('public.candidate_profile_confirmations'::regclass,'public.candidate_consent_copy_catalog'::regclass) AND a.grantee<>c.relowner LOOP
  EXECUTE format('REVOKE ALL ON TABLE %s FROM %I',r.target,r.rolname);
 END LOOP;
 FOR r IN SELECT p.oid::regprocedure AS target,roles.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a JOIN pg_roles roles ON roles.oid=a.grantee
 WHERE n.nspname='public' AND p.proname IN ('candidate_profile_confirm_v3','candidate_consent_v2_project') AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',r.target,r.rolname);
 END LOOP;
END $$;

-- Integrated source: memory-exclusion-proposals.sql
-- Unnumbered local prototype; no runtime grants, provider call or job filtering.
-- Registry rows must be public, independently sourced company identities, never
-- extracted from an organisation's private applications. Only owner maintenance
-- may write this registry; the candidate APIs may search/select, never create it.
CREATE TABLE public.candidate_public_employers (
 company_id uuid PRIMARY KEY,display_name text NOT NULL CHECK(length(display_name) BETWEEN 1 AND 200 AND display_name=btrim(display_name) AND display_name=normalize(display_name,NFC)),
 domain text NOT NULL CHECK(length(domain) BETWEEN 1 AND 253 AND domain~'^(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?$'),
 identity_revision bigint NOT NULL DEFAULT 1 CHECK(identity_revision BETWEEN 1 AND 9007199254740991),
 active boolean NOT NULL DEFAULT true
);
CREATE TABLE public.candidate_exclusion_proposals (
 proposal_id uuid PRIMARY KEY,subject_id uuid NOT NULL,binding_id uuid NOT NULL,
 request_key uuid NOT NULL,request_hash text CHECK(request_hash IS NULL OR request_hash~'^[0-9a-f]{64}$'),
 expected_preference_revision bigint NOT NULL CHECK(expected_preference_revision BETWEEN 0 AND 9007199254740991),
 payload jsonb,payload_hash text CHECK(payload_hash IS NULL OR payload_hash~'^[0-9a-f]{64}$'),
 state text NOT NULL DEFAULT 'pending' CHECK(state IN ('pending','accepted','erased')),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),expires_at timestamptz NOT NULL,
 FOREIGN KEY(subject_id,binding_id) REFERENCES public.candidate_owned_subjects(subject_id,flow_binding_id),
 UNIQUE(subject_id,request_key),CHECK(payload IS NULL OR octet_length(payload::text)<=32768)
);
DO $$ DECLARE tab text; BEGIN
 FOREACH tab IN ARRAY ARRAY['candidate_public_employers','candidate_exclusion_proposals'] LOOP
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',tab);
  EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY',tab);
  EXECUTE format('CREATE POLICY owner_only ON public.%I TO %I USING(true) WITH CHECK(true)',tab,current_user);
  EXECUTE format('REVOKE ALL ON TABLE public.%I FROM PUBLIC',tab);
 END LOOP;
END $$;

CREATE FUNCTION public.candidate_company_choices(p_subject uuid,p_binding uuid,p_query text) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET statement_timeout='3s' SET lock_timeout='1500ms' AS $$
DECLARE q text; result jsonb;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_heads WHERE subject_id=p_subject AND restricted)
 THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 q:=lower(btrim(p_query));
 IF q IS NULL OR length(q) NOT BETWEEN 2 AND 120 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 SELECT coalesce(jsonb_agg(x.item ORDER BY x.rank,x.company_id),'[]'::jsonb) INTO result FROM (
  SELECT company_id,CASE WHEN lower(display_name)=q OR domain=q THEN 0 WHEN starts_with(lower(display_name),q) THEN 1 ELSE 2 END rank,
   jsonb_build_object('companyId',company_id,'displayName',display_name,'domain',domain) item
  FROM public.candidate_public_employers WHERE active AND (strpos(lower(display_name),q)>0 OR domain=q)
  ORDER BY rank,company_id LIMIT 10
 ) x;
 RETURN result;
END $$;
REVOKE ALL ON FUNCTION public.candidate_company_choices(uuid,uuid,text) FROM PUBLIC;

CREATE FUNCTION public.candidate_exclusion_preview(p_subject uuid,p_binding uuid,p_key uuid,p_expected bigint,p_rules jsonb) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET statement_timeout='3s' SET lock_timeout='1500ms' AS $$
DECLARE d public.candidate_preference_drafts%ROWTYPE; p public.candidate_exclusion_proposals%ROWTYPE;
 c public.candidate_public_employers%ROWTYPE; rule jsonb; rules jsonb:='[]'; effects jsonb:='[]'; companies jsonb:='[]';
 pid uuid:=gen_random_uuid(); rid uuid; digest text; result jsonb; exp timestamptz; kind text; effect text;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_key IS NULL OR p_expected IS NULL OR p_expected NOT BETWEEN 0 AND 9007199254740991
 OR jsonb_typeof(p_rules) IS DISTINCT FROM 'array' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF jsonb_array_length(p_rules) NOT BETWEEN 1 AND 20 OR octet_length(p_rules::text)>8192
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 FOR rule IN SELECT value FROM jsonb_array_elements(p_rules) LOOP
  IF jsonb_typeof(rule) IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
  kind:=rule->>'kind';
  IF kind IS NULL OR kind NOT IN ('remote_only','no_travel','exclude_company') THEN RAISE EXCEPTION 'unsupported_restriction' USING ERRCODE='22023'; END IF;
  IF (kind='exclude_company' AND ((SELECT count(*) FROM jsonb_object_keys(rule))<>2 OR NOT rule ?& ARRAY['kind','companyId']
   OR jsonb_typeof(rule->'companyId') IS DISTINCT FROM 'string' OR rule->>'companyId'!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'))
   OR (kind<>'exclude_company' AND (SELECT count(*) FROM jsonb_object_keys(rule))<>1)
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF (SELECT count(DISTINCT value) FROM jsonb_array_elements(p_rules))<>jsonb_array_length(p_rules)
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) ON CONFLICT DO NOTHING;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_heads WHERE subject_id=p_subject AND restricted)
 THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 digest:=encode(sha256(convert_to(public.candidate_profile_canonical(jsonb_build_object('expectedPreferenceDraftRevision',p_expected,'restrictions',p_rules)),'UTF8')),'hex');
 SELECT * INTO p FROM public.candidate_exclusion_proposals WHERE subject_id=p_subject AND request_key=p_key;
 IF FOUND THEN
  IF p.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF p.request_hash IS DISTINCT FROM digest THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  IF p.expires_at<=clock_timestamp() THEN RAISE EXCEPTION 'proposal_expired' USING ERRCODE='22023'; END IF;
  RETURN p.payload||jsonb_build_object('proposalHash',p.payload_hash);
 END IF;
 INSERT INTO public.candidate_preference_drafts(subject_id,content_hash)
 VALUES(p_subject,encode(sha256(convert_to('{"exclusions":[],"sections":{}}','UTF8')),'hex')) ON CONFLICT DO NOTHING;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject FOR UPDATE;
 IF d.revision<>p_expected THEN RAISE EXCEPTION 'stale_edit' USING ERRCODE='40001'; END IF;
 FOR rule IN SELECT value FROM jsonb_array_elements(p_rules) LOOP
  rid:=gen_random_uuid();kind:=rule->>'kind';
  rules:=rules||jsonb_build_array(rule||jsonb_build_object('id',rid));
  IF kind='exclude_company' THEN
   SELECT * INTO c FROM public.candidate_public_employers WHERE company_id=(rule->>'companyId')::uuid AND active;
   IF NOT FOUND THEN RAISE EXCEPTION 'company_not_resolved' USING ERRCODE='22023'; END IF;
   companies:=companies||jsonb_build_array(jsonb_build_object('companyId',c.company_id,'displayName',c.display_name,'domain',c.domain,'identityRevision',c.identity_revision));
   effect:='Exclude jobs from the resolved company. This does not hide your profile from its recruiters.';
  ELSIF kind='remote_only' THEN effect:='Exclude jobs explicitly requiring on-site or hybrid work.';
  ELSE effect:='Exclude jobs explicitly requiring travel.'; END IF;
  effects:=effects||jsonb_build_array(jsonb_build_object('restrictionId',rid,'text',effect));
 END LOOP;
 -- Validity, not a retention policy: re-preview after 15 minutes, with no spend.
 exp:=date_trunc('milliseconds',clock_timestamp()+interval '15 minutes');
 result:=jsonb_build_object('proposalId',pid,'expectedPreferenceDraftRevision',d.revision,'restrictions',rules,'effects',effects,'companies',companies,
  'unknownPolicy','show_with_warning','expiresAt',to_char(exp AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.MS"Z"'));
 INSERT INTO public.candidate_exclusion_proposals(proposal_id,subject_id,binding_id,request_key,request_hash,expected_preference_revision,payload,payload_hash,expires_at)
 VALUES(pid,p_subject,p_binding,p_key,digest,d.revision,result,encode(sha256(convert_to(public.candidate_profile_canonical(result),'UTF8')),'hex'),exp)
 RETURNING * INTO p;
 RETURN result||jsonb_build_object('proposalHash',p.payload_hash);
END $$;
REVOKE ALL ON FUNCTION public.candidate_exclusion_preview(uuid,uuid,uuid,bigint,jsonb) FROM PUBLIC;

CREATE FUNCTION public.candidate_exclusion_proposals_scrub() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $$ BEGIN
 IF NEW.restricted THEN UPDATE public.candidate_exclusion_proposals SET payload=NULL,payload_hash=NULL,request_hash=NULL,state='erased' WHERE subject_id=NEW.subject_id; END IF;
 RETURN NEW;
END $$;
REVOKE ALL ON FUNCTION public.candidate_exclusion_proposals_scrub() FROM PUBLIC;
CREATE TRIGGER candidate_exclusion_proposals_erasure AFTER UPDATE OF restricted ON public.candidate_profile_heads
FOR EACH ROW EXECUTE FUNCTION public.candidate_exclusion_proposals_scrub();

CREATE FUNCTION public.candidate_exclusions_apply(p_subject uuid,p_binding uuid,p_command uuid,p_hash text,p_wire text) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE j jsonb; payload jsonb; field text; action text; reason text; changed boolean:=false;
 h public.candidate_profile_heads%ROWTYPE; d public.candidate_preference_drafts%ROWTYPE;
 r public.candidate_profile_receipts%ROWTYPE; p public.candidate_exclusion_proposals%ROWTYPE;
 c public.candidate_public_employers%ROWTYPE; snapshot jsonb; rule jsonb; next_rules jsonb; result jsonb;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_hash IS NULL OR p_wire IS NULL
 OR octet_length(p_wire)>8192 OR p_hash!~'^[0-9a-f]{64}$'
 OR encode(sha256(convert_to(p_wire,'UTF8')),'hex')<>p_hash THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 j:=p_wire::jsonb;
 IF jsonb_typeof(j) IS DISTINCT FROM 'object' OR public.candidate_profile_canonical(j) IS DISTINCT FROM p_wire
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF (SELECT count(*) FROM jsonb_object_keys(j))<>8
 OR NOT j ?& ARRAY['schemaVersion','idempotencyKey','expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision','action','payload']
 OR j->'schemaVersion' IS DISTINCT FROM '3'::jsonb OR coalesce(j->>'action','') NOT IN ('accept_exclusions','remove_exclusion')
 OR coalesce(j->>'idempotencyKey','')!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 FOREACH field IN ARRAY ARRAY['expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision'] LOOP
  IF jsonb_typeof(j->field) IS DISTINCT FROM 'number' OR (j->>field)!~'^(0|[1-9][0-9]{0,15})$' OR (j->>field)::numeric>9007199254740991
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 action:=j->>'action'; payload:=j->'payload';
 IF jsonb_typeof(payload) IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF action='accept_exclusions' THEN
  IF (SELECT count(*) FROM jsonb_object_keys(payload))<>2 OR NOT payload ?& ARRAY['proposalId','proposalHash']
  OR coalesce(payload->>'proposalHash','')!~'^[0-9a-f]{64}$'
  OR coalesce(payload->>'proposalId','')!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 ELSE
  IF (SELECT count(*) FROM jsonb_object_keys(payload))<>1 OR NOT payload ? 'restrictionId'
  OR coalesce(payload->>'restrictionId','')!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) ON CONFLICT DO NOTHING;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding) THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  RETURN r.result;
 END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_receipts WHERE subject_id=p_subject AND idempotency_key=(j->>'idempotencyKey')::uuid)
 THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
 -- Ownership precedes expiry, state and hash checks: no foreign proposal oracle.
 IF action='accept_exclusions' THEN
  SELECT * INTO p FROM public.candidate_exclusion_proposals
  WHERE proposal_id=(j#>>'{payload,proposalId}')::uuid AND subject_id=p_subject AND binding_id=p_binding FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 END IF;
 INSERT INTO public.candidate_preference_drafts(subject_id,content_hash)
 VALUES(p_subject,encode(sha256(convert_to('{"exclusions":[],"sections":{}}','UTF8')),'hex')) ON CONFLICT DO NOTHING;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject FOR UPDATE;
 next_rules:=d.exclusions;
 IF d.revision<>(j->>'expectedPreferenceDraftRevision')::bigint THEN reason:='stale_edit';
 ELSIF action='remove_exclusion' THEN
  SELECT coalesce(jsonb_agg(value ORDER BY ord),'[]'::jsonb) INTO next_rules
  FROM jsonb_array_elements(d.exclusions) WITH ORDINALITY a(value,ord) WHERE value->>'id'<>payload->>'restrictionId';
 ELSE
  IF p.expected_preference_revision<>d.revision THEN reason:='stale_edit';
  ELSIF p.state<>'pending' OR p.payload_hash IS DISTINCT FROM payload->>'proposalHash'
   OR p.payload_hash IS DISTINCT FROM encode(sha256(convert_to(public.candidate_profile_canonical(p.payload),'UTF8')),'hex')
   THEN reason:='source_conflict';
  ELSIF p.expires_at<=clock_timestamp() THEN reason:='proposal_expired';
  ELSE
   -- Lock registry rows in stable order until commit; a changed identity needs a
   -- fresh preview and explicit approval, never silent company substitution.
   FOR snapshot IN SELECT value FROM jsonb_array_elements(p.payload->'companies') ORDER BY value->>'companyId' LOOP
    SELECT * INTO c FROM public.candidate_public_employers WHERE company_id=(snapshot->>'companyId')::uuid FOR SHARE;
    IF NOT FOUND OR NOT c.active OR c.identity_revision<>(snapshot->>'identityRevision')::bigint
    OR c.display_name IS DISTINCT FROM snapshot->>'displayName' OR c.domain IS DISTINCT FROM snapshot->>'domain'
    THEN reason:='source_conflict'; EXIT; END IF;
   END LOOP;
   IF reason IS NULL THEN
    FOR rule IN SELECT value FROM jsonb_array_elements(p.payload->'restrictions') LOOP
     IF NOT EXISTS(SELECT 1 FROM jsonb_array_elements(next_rules) a(value) WHERE value-'id'=rule-'id')
     THEN next_rules:=next_rules||jsonb_build_array(rule); END IF;
    END LOOP;
    IF jsonb_array_length(next_rules)>20 THEN reason:='exclusion_limit'; END IF;
   END IF;
  END IF;
 END IF;
 IF reason IS NULL THEN
  changed:=next_rules IS DISTINCT FROM d.exclusions;
  IF changed THEN
   UPDATE public.candidate_preference_drafts SET exclusions=next_rules,revision=revision+1,
   content_hash=encode(sha256(convert_to(public.candidate_profile_canonical(jsonb_build_object('sections',d.sections,'exclusions',next_rules)),'UTF8')),'hex'),updated_at=clock_timestamp()
   WHERE subject_id=p_subject RETURNING * INTO d;
  END IF;
  IF action='accept_exclusions' THEN UPDATE public.candidate_exclusion_proposals SET state='accepted' WHERE proposal_id=p.proposal_id; END IF;
 END IF;
 result:=jsonb_build_object('schemaVersion',3,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,'commandHash',p_hash,
 'status',CASE WHEN reason IS NULL THEN 'applied' ELSE 'rejected' END,'reason',reason,'changed',changed,
 'draftVersion',h.draft_version,'profileVersion',h.profile_version,'preferenceVersion',h.preference_version,
 'preferenceDraftRevision',d.revision,'preferenceDraftHash',d.content_hash,'publicationActive',false);
 INSERT INTO public.candidate_profile_receipts(command_id,subject_id,binding_id,idempotency_key,wire_hash,result)
 VALUES(p_command,p_subject,p_binding,(j->>'idempotencyKey')::uuid,p_hash,result);
 RETURN result;
END $$;
REVOKE ALL ON FUNCTION public.candidate_exclusions_apply(uuid,uuid,uuid,text,text) FROM PUBLIC;

-- Owner default privileges must not accidentally expose private proposals or
-- bypass routine-only access. Runtime grants belong to the reviewed provisioner.
DO $$ DECLARE entry record; BEGIN
 FOR entry IN SELECT c.oid::regclass target,r.rolname FROM pg_class c
 CROSS JOIN LATERAL aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a JOIN pg_roles r ON r.oid=a.grantee
 WHERE c.oid IN ('public.candidate_public_employers'::regclass,'public.candidate_exclusion_proposals'::regclass) AND a.grantee<>c.relowner LOOP
  EXECUTE format('REVOKE ALL ON TABLE %s FROM %I',entry.target,entry.rolname);
 END LOOP;
 FOR entry IN SELECT p.oid::regprocedure target,r.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a JOIN pg_roles r ON r.oid=a.grantee
 WHERE n.nspname='public' AND p.proname IN ('candidate_company_choices','candidate_exclusion_preview','candidate_exclusion_proposals_scrub','candidate_exclusions_apply') AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',entry.target,entry.rolname);
 END LOOP;
END $$;

-- Atomic v3 expiry: recover the original receipt or fence late delivery.
CREATE FUNCTION public.candidate_profile_expire_v3(p_subject uuid,p_binding uuid,p_command uuid,p_key uuid,p_hash text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE h public.candidate_profile_heads%ROWTYPE;r public.candidate_profile_receipts%ROWTYPE;answer jsonb;d public.candidate_preference_drafts%ROWTYPE;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_key IS NULL
 OR p_hash IS NULL OR p_hash!~'^[0-9a-f]{64}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) ON CONFLICT DO NOTHING;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 -- Never put erased fingerprints back, even for a delayed expiry request.
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding)
  THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash OR r.idempotency_key IS DISTINCT FROM p_key
  THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  IF r.result->'schemaVersion' IS DISTINCT FROM '3'::jsonb THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  RETURN r.result;
 END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_receipts WHERE subject_id=p_subject AND idempotency_key=p_key)
 THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject;
 answer:=jsonb_build_object('schemaVersion',3,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,
  'commandHash',p_hash,'status','rejected','draftVersion',h.draft_version,'profileVersion',h.profile_version,
  'preferenceVersion',h.preference_version,'changed',false,'preferenceDraftRevision',coalesce(d.revision,0),
  'preferenceDraftHash',d.content_hash,'reason','command_expired','publicationActive',false);
 INSERT INTO public.candidate_profile_receipts(command_id,subject_id,binding_id,idempotency_key,wire_hash,result)
 VALUES(p_command,p_subject,p_binding,p_key,p_hash,answer);
 RETURN answer;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_expire_v3(uuid,uuid,uuid,uuid,text) FROM PUBLIC;
DO $$ DECLARE entry record; BEGIN
 FOR entry IN SELECT r.rolname FROM pg_proc p CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 JOIN pg_roles r ON r.oid=a.grantee WHERE p.oid='public.candidate_profile_expire_v3(uuid,uuid,uuid,uuid,text)'::regprocedure AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION public.candidate_profile_expire_v3(uuid,uuid,uuid,uuid,text) FROM %I',entry.rolname);
 END LOOP;
END $$;

-- Private source proposals: verification text is never persisted in Memory.
CREATE TABLE public.candidate_profile_sources (
 subject_id uuid NOT NULL REFERENCES public.candidate_profile_heads(subject_id),source_id uuid NOT NULL,
 source_version bigint NOT NULL CHECK(source_version BETWEEN 0 AND 9007199254740991),
 content_hash text CHECK(content_hash~'^[0-9a-f]{64}$'),text_hash text CHECK(text_hash~'^[0-9a-f]{64}$'),
 policy_version text,created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(subject_id,source_id,source_version)
);
CREATE TABLE public.candidate_profile_proposals (
 proposal_id uuid PRIMARY KEY,subject_id uuid NOT NULL,binding_id uuid NOT NULL,
 source_id uuid NOT NULL,source_version bigint NOT NULL,proposal_hash text CHECK(proposal_hash~'^[0-9a-f]{64}$'),
 payload jsonb,state text NOT NULL DEFAULT 'proposed' CHECK(state IN ('proposed','accepted','rejected','erased')),
 expires_at timestamptz NOT NULL,created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 consent_revision bigint NOT NULL CHECK(consent_revision>0),acceptance jsonb,
 FOREIGN KEY(subject_id,source_id,source_version) REFERENCES public.candidate_profile_sources(subject_id,source_id,source_version),
 FOREIGN KEY(subject_id,binding_id) REFERENCES public.candidate_owned_subjects(subject_id,flow_binding_id)
);
CREATE TABLE public.candidate_profile_source_receipts (
 command_id uuid PRIMARY KEY,subject_id uuid NOT NULL REFERENCES public.candidate_profile_heads(subject_id),binding_id uuid NOT NULL,
 wire_hash text CHECK(wire_hash~'^[0-9a-f]{64}$'),result jsonb NOT NULL,created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
DO $$ DECLARE tab text; BEGIN
 FOREACH tab IN ARRAY ARRAY['candidate_profile_sources','candidate_profile_proposals','candidate_profile_source_receipts'] LOOP
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',tab);
  EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY',tab);
  EXECUTE format('CREATE POLICY owner_only ON public.%I TO %I USING(true) WITH CHECK(true)',tab,current_user);
  EXECUTE format('REVOKE ALL ON TABLE public.%I FROM PUBLIC',tab);
 END LOOP;
END $$;

CREATE FUNCTION public.candidate_profile_source_receive(p_subject uuid,p_binding uuid,p_command uuid,p_hash text,p_wire text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='5s' AS $$
DECLARE j jsonb; p jsonb; fact jsonb; evidence jsonb; k text; txt text; offsets integer[];
 start_unit bigint; units bigint; first_char integer; last_char integer; parts text[]; exp timestamptz;
 h public.candidate_profile_heads%ROWTYPE;c public.candidate_consent_state%ROWTYPE;
 r public.candidate_profile_source_receipts%ROWTYPE;s public.candidate_profile_sources%ROWTYPE;
 pref_revision bigint;reason text;answer jsonb;quoted_bytes bigint:=0;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_hash IS NULL OR p_wire IS NULL
 OR octet_length(p_wire) NOT BETWEEN 1 AND 2097152 OR p_hash!~'^[0-9a-f]{64}$'
 OR encode(sha256(convert_to(p_wire,'UTF8')),'hex')<>p_hash THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 BEGIN j:=p_wire::jsonb;
 EXCEPTION WHEN untranslatable_character THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END;
 IF jsonb_typeof(j) IS DISTINCT FROM 'object' OR public.candidate_profile_canonical(j) IS DISTINCT FROM p_wire
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF (SELECT count(*) FROM jsonb_object_keys(j))<>12 OR NOT j ?& ARRAY['schemaVersion','commandId','subjectId','bindingId','sourceId','sourceVersion','contentHash','textHash','policyVersion','consentRevision','text','proposal']
 OR j->'schemaVersion' IS DISTINCT FROM '1'::jsonb OR j->>'subjectId' IS DISTINCT FROM p_subject::text
 OR j->>'bindingId' IS DISTINCT FROM p_binding::text OR j->>'commandId' IS DISTINCT FROM p_command::text
 OR jsonb_typeof(j->'proposal') IS DISTINCT FROM 'object' OR jsonb_typeof(j->'text') IS DISTINCT FROM 'string'
 OR jsonb_typeof(j->'policyVersion') IS DISTINCT FROM 'string' OR length(j->>'policyVersion') NOT BETWEEN 1 AND 80
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 p:=j->'proposal';txt:=j->>'text';
 IF octet_length(txt) NOT BETWEEN 1 AND 524288 OR (SELECT count(*) FROM jsonb_object_keys(p))<>13
 OR NOT p ?& ARRAY['schemaVersion','proposalId','sourceId','sourceVersion','proposalHash','originalDraftVersion','originalProfileVersion','originalPreferenceVersion','originalPreferenceDraftRevision','sourceKind','facts','reviewNotes','expiresAt']
 OR p->'schemaVersion' IS DISTINCT FROM '1'::jsonb OR coalesce(p->>'sourceKind','') NOT IN ('resume','professional_lookup')
 OR p->'sourceId' IS DISTINCT FROM j->'sourceId' OR p->'sourceVersion' IS DISTINCT FROM j->'sourceVersion'
 OR jsonb_typeof(p->'facts') IS DISTINCT FROM 'array' OR jsonb_typeof(p->'reviewNotes') IS DISTINCT FROM 'array'
 OR jsonb_typeof(p->'proposalId') IS DISTINCT FROM 'string' OR p->>'proposalId'!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF p->>'sourceKind'='professional_lookup' AND (
  j->>'policyVersion' IS DISTINCT FROM 'rc-profile-v1' OR j->>'contentHash' IS DISTINCT FROM j->>'textHash'
  OR txt IS DISTINCT FROM public.candidate_profile_canonical((SELECT coalesce(jsonb_agg(jsonb_build_object('kind',f->'kind','value',f->'value') ORDER BY n),'[]'::jsonb)
   FROM jsonb_array_elements(p->'facts') WITH ORDINALITY a(f,n))))
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 FOREACH k IN ARRAY ARRAY['sourceId'] LOOP
  IF jsonb_typeof(j->k) IS DISTINCT FROM 'string' OR j->>k!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 FOREACH k IN ARRAY ARRAY['contentHash','textHash'] LOOP
  IF jsonb_typeof(j->k) IS DISTINCT FROM 'string' OR j->>k!~'^[0-9a-f]{64}$' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 FOREACH k IN ARRAY ARRAY['sourceVersion','consentRevision'] LOOP
  IF jsonb_typeof(j->k) IS DISTINCT FROM 'number' OR j->>k!~'^(0|[1-9][0-9]{0,15})$' OR (j->>k)::numeric>9007199254740991
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF (j->>'consentRevision')::bigint<1 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 FOREACH k IN ARRAY ARRAY['originalDraftVersion','originalProfileVersion','originalPreferenceVersion','originalPreferenceDraftRevision'] LOOP
  IF jsonb_typeof(p->k) IS DISTINCT FROM 'number' OR p->>k!~'^(0|[1-9][0-9]{0,15})$' OR (p->>k)::numeric>9007199254740991
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF j->>'textHash' IS DISTINCT FROM encode(sha256(convert_to(txt,'UTF8')),'hex')
 OR p->>'proposalHash' IS DISTINCT FROM encode(sha256(convert_to(public.candidate_profile_canonical(p-'proposalHash'),'UTF8')),'hex')
 OR jsonb_array_length(p->'facts')>400 OR jsonb_array_length(p->'reviewNotes')>20
 OR (SELECT count(DISTINCT value->>'factId') FROM jsonb_array_elements(p->'facts'))<>jsonb_array_length(p->'facts')
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 FOR fact IN SELECT value FROM jsonb_array_elements(p->'reviewNotes') LOOP
  IF jsonb_typeof(fact) IS DISTINCT FROM 'string' OR length(fact#>>'{}') NOT BETWEEN 1 AND 2000 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 END LOOP;
 IF jsonb_typeof(p->'expiresAt') IS DISTINCT FROM 'string' OR p->>'expiresAt'!~'^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(:[0-9]{2}(\.[0-9]+)?)?Z$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 IF substring(p->>'expiresAt',12,2)::integer>23 OR substring(p->>'expiresAt',15,2)::integer>59
 OR (substring(p->>'expiresAt',17,1)=':' AND substring(p->>'expiresAt',18,2)::integer>59)
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
 BEGIN
  -- ISO contracts permit astronomical year zero; PostgreSQL does not. Validate
  -- its leap-calendar date using year2000, then classify it as already expired.
  IF left(p->>'expiresAt',4)='0000' THEN
   PERFORM ('2000'||substring(p->>'expiresAt',5))::timestamptz;exp:='-infinity'::timestamptz;
  ELSE exp:=(p->>'expiresAt')::timestamptz;END IF;
 EXCEPTION WHEN OTHERS THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END;
 -- One UTF16 index for the whole transfer, not a rescan of text per quote.
 SELECT ARRAY[0]||coalesce(array_agg(u ORDER BY ord),'{}'::integer[]) INTO offsets FROM
 (SELECT ord,(sum(CASE WHEN ascii(ch)>65535 THEN 2 ELSE 1 END) OVER(ORDER BY ord))::integer AS u
  FROM regexp_split_to_table(txt,'') WITH ORDINALITY x(ch,ord)) q;
 FOR fact IN SELECT value FROM jsonb_array_elements(p->'facts') LOOP
  IF NOT public.candidate_profile_source_fact_valid(fact) THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
  IF fact->>'kind'='photo' AND p->>'sourceKind'<>'professional_lookup' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
  IF p->>'sourceKind'='professional_lookup' AND (fact->'targetRowId' IS DISTINCT FROM 'null'::jsonb
   OR jsonb_array_length(fact->'evidence')<>1
   OR fact->'evidence'->0->>'quote' IS DISTINCT FROM public.candidate_profile_canonical(jsonb_build_object('kind',fact->'kind','value',fact->'value')))
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
  FOR evidence IN SELECT value FROM jsonb_array_elements(fact->'evidence') LOOP
   IF jsonb_typeof(evidence) IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
   IF (SELECT count(*) FROM jsonb_object_keys(evidence))<>5 OR NOT evidence ?& ARRAY['sourceId','sourceVersion','contentHash','locator','quote']
   OR evidence->'sourceId' IS DISTINCT FROM j->'sourceId' OR evidence->'sourceVersion' IS DISTINCT FROM j->'sourceVersion'
   OR evidence->'contentHash' IS DISTINCT FROM j->'contentHash' OR jsonb_typeof(evidence->'locator') IS DISTINCT FROM 'string'
   OR jsonb_typeof(evidence->'quote') IS DISTINCT FROM 'string' OR length(evidence->>'quote') NOT BETWEEN 1 AND 2048
   OR length(evidence->>'locator')>200 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
   parts:=regexp_match(evidence->>'locator','^text:(0|[1-9][0-9]*):(0|[1-9][0-9]*)$');
   IF parts IS NULL OR length(parts[1])>16 OR length(parts[2])>16 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
   start_unit:=parts[1]::bigint;units:=parts[2]::bigint;
   quoted_bytes:=quoted_bytes+octet_length(evidence->>'quote');
   IF units>2048 OR quoted_bytes>65536 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
   IF start_unit>1048576 OR units NOT BETWEEN 1 AND 1048576 OR start_unit+units>offsets[cardinality(offsets)]
   THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
   first_char:=array_position(offsets,start_unit::integer);last_char:=array_position(offsets,(start_unit+units)::integer);
   IF first_char IS NULL OR last_char IS NULL OR substring(txt FROM first_char FOR last_char-first_char) IS DISTINCT FROM evidence->>'quote'
   THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023'; END IF;
  END LOOP;
 END LOOP;
 PERFORM set_config('app.current_tenant_id','candidate_'||p_subject::text,true);
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p_subject::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) ON CONFLICT DO NOTHING;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501'; END IF;
 SELECT * INTO r FROM public.candidate_profile_source_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding) THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002'; END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505'; END IF;
  RETURN r.result;
 END IF;
 SELECT * INTO c FROM public.candidate_consent_state WHERE subject_id=p_subject;
 SELECT revision INTO pref_revision FROM public.candidate_preference_drafts WHERE subject_id=p_subject;
 IF c.effective_action IS DISTINCT FROM 'grant' OR c.effective_schema_version IS DISTINCT FROM 2
 OR c.highest_version IS DISTINCT FROM c.effective_version OR c.effective_version IS DISTINCT FROM (j->>'consentRevision')::bigint
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_copy_catalog WHERE copy_version=c.effective_copy_version AND copy_sha256=c.effective_copy_sha256 AND approved)
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_receipts cr WHERE cr.subject_id=p_subject AND cr.version=c.effective_version
  AND cr.authority_schema_version=2 AND cr.outcome='onboarding_granted' AND cr.copy_version=c.effective_copy_version AND cr.copy_sha256=c.effective_copy_sha256
  AND cr.purposes=ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact']) THEN reason:='consent_changed';
 ELSIF (h.draft_version,h.profile_version,h.preference_version,coalesce(pref_revision,0)) IS DISTINCT FROM
 ((p->>'originalDraftVersion')::bigint,(p->>'originalProfileVersion')::bigint,(p->>'originalPreferenceVersion')::bigint,(p->>'originalPreferenceDraftRevision')::bigint) THEN reason:='stale_edit';
 ELSIF exp<=clock_timestamp() THEN reason:='proposal_expired'; END IF;
 IF reason IS NULL THEN
  SELECT * INTO s FROM public.candidate_profile_sources WHERE subject_id=p_subject AND source_id=(j->>'sourceId')::uuid AND source_version=(j->>'sourceVersion')::bigint;
  IF FOUND AND (s.content_hash,s.text_hash,s.policy_version) IS DISTINCT FROM (j->>'contentHash',j->>'textHash',j->>'policyVersion') THEN reason:='source_conflict'; END IF;
  IF EXISTS(SELECT 1 FROM public.candidate_profile_proposals WHERE proposal_id=(p->>'proposalId')::uuid) THEN reason:='source_conflict'; END IF;
 END IF;
 IF reason IS NULL THEN
  INSERT INTO public.candidate_profile_sources(subject_id,source_id,source_version,content_hash,text_hash,policy_version)
  VALUES(p_subject,(j->>'sourceId')::uuid,(j->>'sourceVersion')::bigint,j->>'contentHash',j->>'textHash',j->>'policyVersion') ON CONFLICT DO NOTHING;
  INSERT INTO public.candidate_profile_proposals(proposal_id,subject_id,binding_id,source_id,source_version,proposal_hash,payload,expires_at,consent_revision)
  VALUES((p->>'proposalId')::uuid,p_subject,p_binding,(j->>'sourceId')::uuid,(j->>'sourceVersion')::bigint,p->>'proposalHash',p,exp,(j->>'consentRevision')::bigint);
 END IF;
 answer:=jsonb_build_object('schemaVersion',1,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,'commandHash',p_hash,
 'sourceId',j->'sourceId','sourceVersion',j->'sourceVersion','proposalId',p->'proposalId','proposalHash',p->'proposalHash',
 'outcome',CASE WHEN reason IS NULL THEN 'stored' ELSE 'rejected' END,'reason',reason,'canonicalProfileChanged',false,'publicationActive',false);
 INSERT INTO public.candidate_profile_source_receipts(command_id,subject_id,binding_id,wire_hash,result) VALUES(p_command,p_subject,p_binding,p_hash,answer);
 RETURN answer;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_source_receive(uuid,uuid,uuid,text,text) FROM PUBLIC;

-- Owner-only proposal reads return the reviewed proposal, never raw source text.
CREATE FUNCTION public.candidate_profile_source_read(p_subject uuid,p_binding uuid,p_source uuid,p_proposal uuid)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE h public.candidate_profile_heads%ROWTYPE;p public.candidate_profile_proposals%ROWTYPE;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject AND binding_id=p_binding;
 IF NOT FOUND THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501';END IF;
 SELECT * INTO p FROM public.candidate_profile_proposals WHERE proposal_id=p_proposal AND source_id=p_source
  AND subject_id=p_subject AND binding_id=p_binding AND state='proposed' AND payload IS NOT NULL;
 IF NOT FOUND THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 RETURN p.payload;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_source_read(uuid,uuid,uuid,uuid) FROM PUBLIC;

CREATE FUNCTION public.candidate_profile_source_apply(p_subject uuid,p_binding uuid,p_command uuid,p_hash text,p_wire text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public
SET lock_timeout='1500ms' SET statement_timeout='5s' AS $$
DECLARE j jsonb;b jsonb;k text;id text;fact jsonb;corrected jsonb;v jsonb;row_value jsonb;items jsonb;
 h public.candidate_profile_heads%ROWTYPE;p public.candidate_profile_proposals%ROWTYPE;
 r public.candidate_profile_receipts%ROWTYPE;d public.candidate_preference_drafts%ROWTYPE;c public.candidate_consent_state%ROWTYPE;
 draft jsonb;original jsonb;provenance jsonb:='[]';seen text[]:='{}';field text;target text;digest text;row_id text;
 reason text;answer jsonb;changed boolean:=false;is_replace boolean;found_row boolean;
BEGIN
 IF p_subject IS NULL OR p_binding IS NULL OR p_command IS NULL OR p_wire IS NULL OR p_hash IS NULL
 OR p_hash!~'^[a-f0-9]{64}$' OR octet_length(p_wire)>393216
 OR encode(sha256(convert_to(p_wire,'UTF8')),'hex')<>p_hash THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 BEGIN j:=p_wire::jsonb;EXCEPTION WHEN invalid_text_representation OR untranslatable_character THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END;
 IF jsonb_typeof(j) IS DISTINCT FROM 'object' OR public.candidate_profile_canonical(j) IS DISTINCT FROM p_wire
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 IF (SELECT count(*) FROM jsonb_object_keys(j))<>8 OR NOT j ?& ARRAY['schemaVersion','idempotencyKey','expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision','action','payload']
 OR j->'schemaVersion' IS DISTINCT FROM '3'::jsonb OR coalesce(j->>'action','') NOT IN ('accept_source','reject_source')
 OR coalesce(j->>'idempotencyKey','')!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 OR jsonb_typeof(j->'payload') IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 FOREACH k IN ARRAY ARRAY['expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision'] LOOP
  IF jsonb_typeof(j->k) IS DISTINCT FROM 'number' OR j->>k!~'^(0|[1-9][0-9]{0,15})$' OR (j->>k)::numeric>9007199254740991
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 END LOOP;
 b:=j->'payload';
 IF coalesce(b->>'proposalId','')!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 OR coalesce(b->>'proposalHash','')!~'^[0-9a-f]{64}$' OR jsonb_typeof(b->'sourceVersion') IS DISTINCT FROM 'number'
 OR b->>'sourceVersion'!~'^(0|[1-9][0-9]{0,15})$' OR (b->>'sourceVersion')::numeric>9007199254740991
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 IF j->>'action'='reject_source' THEN
  IF (SELECT count(*) FROM jsonb_object_keys(b))<>4 OR NOT b ?& ARRAY['proposalId','proposalHash','sourceVersion','reason']
  OR coalesce(b->>'reason','') NOT IN ('not_me','incorrect','not_needed') THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 ELSE
  IF (SELECT count(*) FROM jsonb_object_keys(b))<>6 OR NOT b ?& ARRAY['proposalId','proposalHash','sourceVersion','selectedFactIds','replaceFactIds','corrections']
  OR jsonb_typeof(b->'selectedFactIds') IS DISTINCT FROM 'array' OR jsonb_typeof(b->'replaceFactIds') IS DISTINCT FROM 'array'
  OR jsonb_typeof(b->'corrections') IS DISTINCT FROM 'array' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
  FOREACH k IN ARRAY ARRAY['selectedFactIds','replaceFactIds'] LOOP
   IF jsonb_array_length(b->k)>400 OR (k='selectedFactIds' AND jsonb_array_length(b->k)=0)
   OR (SELECT count(DISTINCT value) FROM jsonb_array_elements(b->k))<>jsonb_array_length(b->k)
   OR EXISTS(SELECT 1 FROM jsonb_array_elements(b->k) x WHERE jsonb_typeof(x.value)<>'string' OR x.value#>>'{}'!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')
   THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
  END LOOP;
  IF NOT (b->'selectedFactIds') @> (b->'replaceFactIds') OR jsonb_array_length(b->'corrections')>400
  OR (SELECT count(DISTINCT value->>'factId') FROM jsonb_array_elements(b->'corrections'))<>jsonb_array_length(b->'corrections')
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
  FOR corrected IN SELECT value FROM jsonb_array_elements(b->'corrections') LOOP
   IF jsonb_typeof(corrected) IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
   IF (SELECT count(*) FROM jsonb_object_keys(corrected))<>3 OR NOT corrected ?& ARRAY['factId','kind','value']
   OR NOT (b->'selectedFactIds') @> jsonb_build_array(corrected->'factId') THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
  END LOOP;
 END IF;
 PERFORM set_config('app.current_tenant_id','candidate_'||p_subject::text,true);
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p_subject::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject AND binding_id=p_binding FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501';END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding) THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
  IF r.wire_hash<>p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505';END IF;RETURN r.result;
 END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_receipts WHERE subject_id=p_subject AND idempotency_key=(j->>'idempotencyKey')::uuid)
 THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505';END IF;
 SELECT * INTO p FROM public.candidate_profile_proposals WHERE proposal_id=(b->>'proposalId')::uuid AND subject_id=p_subject AND binding_id=p_binding FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject;
 SELECT * INTO c FROM public.candidate_consent_state WHERE subject_id=p_subject;
 IF p.state<>'proposed' OR p.proposal_hash IS DISTINCT FROM b->>'proposalHash' OR p.source_version<>(b->>'sourceVersion')::bigint THEN reason:='source_conflict';
 ELSIF p.expires_at<=clock_timestamp() THEN reason:='proposal_expired';
 ELSIF (h.draft_version,h.profile_version,h.preference_version,coalesce(d.revision,0)) IS DISTINCT FROM
 ((j->>'expectedDraftVersion')::bigint,(j->>'expectedProfileVersion')::bigint,(j->>'expectedPreferenceVersion')::bigint,(j->>'expectedPreferenceDraftRevision')::bigint)
 OR (h.draft_version,h.profile_version,h.preference_version,coalesce(d.revision,0)) IS DISTINCT FROM
 ((p.payload->>'originalDraftVersion')::bigint,(p.payload->>'originalProfileVersion')::bigint,(p.payload->>'originalPreferenceVersion')::bigint,(p.payload->>'originalPreferenceDraftRevision')::bigint)
 THEN reason:='stale_edit';
 ELSIF j->>'action'='accept_source' AND (c.effective_action IS DISTINCT FROM 'grant' OR c.effective_schema_version IS DISTINCT FROM 2
 OR c.highest_version IS DISTINCT FROM c.effective_version OR c.effective_version IS DISTINCT FROM p.consent_revision
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_receipts cr WHERE cr.subject_id=p_subject AND cr.version=c.effective_version
  AND cr.authority_schema_version=2 AND cr.outcome='onboarding_granted' AND cr.copy_version=c.effective_copy_version AND cr.copy_sha256=c.effective_copy_sha256
  AND cr.purposes=ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact'])
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_copy_catalog WHERE copy_version=c.effective_copy_version AND copy_sha256=c.effective_copy_sha256 AND approved))
 THEN reason:='consent_changed';END IF;
 IF reason IS NULL AND j->>'action'='accept_source' THEN
  SELECT facts INTO original FROM public.candidate_profile_drafts WHERE subject_id=p_subject AND version=h.draft_version;
  draft:=coalesce(original,'{"displayName":"","headline":"","linkedinUrl":null,"currentLocation":null,"skills":[],"employment":[],"education":[],"projects":[]}'::jsonb);
  FOR id IN SELECT value FROM jsonb_array_elements_text(b->'selectedFactIds') LOOP
   SELECT value INTO fact FROM jsonb_array_elements(p.payload->'facts') WHERE value->>'factId'=id;
   IF NOT FOUND THEN reason:='source_conflict';EXIT;END IF;
   SELECT value INTO corrected FROM jsonb_array_elements(b->'corrections') WHERE value->>'factId'=id;
   v:=fact->'value';
   IF FOUND THEN
    IF corrected->'kind' IS DISTINCT FROM fact->'kind' OR NOT public.candidate_profile_source_fact_valid(fact||jsonb_build_object('value',corrected->'value'))
    THEN reason:='source_conflict';EXIT;END IF;v:=corrected->'value';
   END IF;
   field:=CASE fact->>'kind' WHEN 'skill' THEN 'skills' WHEN 'project' THEN 'projects' WHEN 'photo' THEN 'photoUrl' ELSE fact->>'kind' END;
   IF fact->>'kind'='photo' THEN v:=v->'url'; END IF;
   is_replace:=(b->'replaceFactIds') ? id;target:=fact->>'targetRowId';
   IF field IN ('displayName','headline','linkedinUrl','photoUrl','currentLocation') THEN
    IF field=ANY(seen) THEN reason:='source_conflict';EXIT;END IF;seen:=array_append(seen,field);
    IF draft->field IS NOT DISTINCT FROM v THEN CONTINUE;END IF;
    IF draft->field<>'null'::jsonb AND btrim(draft->>field)<>'' AND NOT is_replace THEN reason:='source_conflict';EXIT;END IF;
    draft:=jsonb_set(draft,ARRAY[field],v);
   ELSIF field='skills' THEN
    IF is_replace THEN reason:='source_conflict';EXIT;END IF;
    IF EXISTS(SELECT 1 FROM jsonb_array_elements_text(draft->field) x WHERE lower(regexp_replace(btrim(normalize(x.value,NFC)),'\s+',' ','g'))=lower(regexp_replace(btrim(normalize(v#>>'{}',NFC)),'\s+',' ','g'))) THEN CONTINUE;END IF;
    draft:=jsonb_set(draft,ARRAY[field],(draft->field)||jsonb_build_array(v));
   ELSE
    IF target IS NULL THEN
     IF is_replace THEN reason:='source_conflict';EXIT;END IF;
     IF EXISTS(SELECT 1 FROM jsonb_array_elements(draft->field) x WHERE x.value-'id'=v-'id') THEN CONTINUE;END IF;
     digest:=encode(sha256(convert_to(public.candidate_profile_canonical(jsonb_build_array(p.source_id,p.source_version,id)),'UTF8')),'hex');
     row_id:=substr(digest,1,8)||'-'||substr(digest,9,4)||'-5'||substr(digest,14,3)||'-a'||substr(digest,18,3)||'-'||substr(digest,21,12);
     IF EXISTS(SELECT 1 FROM jsonb_array_elements(draft->field) x WHERE x.value->>'id'=row_id) THEN reason:='source_conflict';EXIT;END IF;
     v:=v||jsonb_build_object('id',row_id);draft:=jsonb_set(draft,ARRAY[field],(draft->field)||jsonb_build_array(v));
    ELSE
     IF NOT is_replace OR v->>'id' IS DISTINCT FROM target OR (field||':'||target)=ANY(seen) THEN reason:='source_conflict';EXIT;END IF;
     seen:=array_append(seen,field||':'||target);items:='[]';found_row:=false;
     FOR row_value IN SELECT value FROM jsonb_array_elements(draft->field) LOOP
      IF row_value->>'id'=target THEN found_row:=true;items:=items||jsonb_build_array(v);ELSE items:=items||jsonb_build_array(row_value);END IF;
     END LOOP;
     IF NOT found_row THEN reason:='source_conflict';EXIT;END IF;draft:=jsonb_set(draft,ARRAY[field],items);
    END IF;
   END IF;
   provenance:=provenance||jsonb_build_array(jsonb_build_object('factId',id,'kind',fact->'kind','rowId',CASE WHEN field IN ('employment','education','projects') THEN v->'id' ELSE 'null'::jsonb END,'corrected',corrected IS NOT NULL));
  END LOOP;
  IF reason IS NULL AND (octet_length(public.candidate_profile_canonical(draft))>262144
   OR jsonb_array_length(draft->'skills')>200 OR jsonb_array_length(draft->'employment')>200 OR jsonb_array_length(draft->'education')>50 OR jsonb_array_length(draft->'projects')>50)
  THEN reason:='source_conflict';END IF;
  IF reason IS NULL AND draft IS DISTINCT FROM original THEN
   INSERT INTO public.candidate_profile_drafts(subject_id,version,facts,document_sha256,command_id)
   VALUES(p_subject,h.draft_version+1,draft,encode(sha256(convert_to(public.candidate_profile_canonical(draft),'UTF8')),'hex'),p_command);
   UPDATE public.candidate_profile_drafts SET facts=NULL,document_sha256=NULL WHERE subject_id=p_subject AND version<>h.draft_version+1;
   UPDATE public.candidate_profile_heads SET draft_version=draft_version+1 WHERE subject_id=p_subject RETURNING * INTO h;changed:=true;
  END IF;
 END IF;
 IF reason IS NULL THEN
  UPDATE public.candidate_profile_proposals SET state=CASE WHEN j->>'action'='accept_source' THEN 'accepted' ELSE 'rejected' END,
   acceptance=jsonb_build_object('commandId',p_command,'draftVersion',h.draft_version,'selection',b,'appliedEvidence',provenance)
   WHERE proposal_id=p.proposal_id;
 END IF;
 answer:=jsonb_build_object('schemaVersion',3,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,'commandHash',p_hash,
 'status',CASE WHEN reason IS NULL THEN 'applied' ELSE 'rejected' END,'reason',reason,'changed',changed,
 'draftVersion',h.draft_version,'profileVersion',h.profile_version,'preferenceVersion',h.preference_version,
 'preferenceDraftRevision',coalesce(d.revision,0),'preferenceDraftHash',d.content_hash,'publicationActive',false);
 INSERT INTO public.candidate_profile_receipts(command_id,subject_id,binding_id,idempotency_key,wire_hash,result)
 VALUES(p_command,p_subject,p_binding,(j->>'idempotencyKey')::uuid,p_hash,answer);RETURN answer;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_source_apply(uuid,uuid,uuid,text,text) FROM PUBLIC;

CREATE FUNCTION public.candidate_profile_sources_scrub() RETURNS trigger LANGUAGE plpgsql
SECURITY DEFINER SET search_path=pg_catalog,public AS $$ BEGIN
 IF NEW.restricted THEN
  UPDATE public.candidate_profile_sources SET content_hash=NULL,text_hash=NULL,policy_version=NULL WHERE subject_id=NEW.subject_id;
  UPDATE public.candidate_profile_proposals SET payload=NULL,proposal_hash=NULL,acceptance=NULL,state='erased' WHERE subject_id=NEW.subject_id;
  UPDATE public.candidate_profile_source_receipts SET wire_hash=NULL,result=result-'commandHash'-'proposalHash' WHERE subject_id=NEW.subject_id;
 END IF;RETURN NEW;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_sources_scrub() FROM PUBLIC;
CREATE TRIGGER candidate_profile_sources_erasure AFTER UPDATE OF restricted ON public.candidate_profile_heads
FOR EACH ROW EXECUTE FUNCTION public.candidate_profile_sources_scrub();

-- P6 immutable interpretation proposals. No raw prompt or provider transcript.
CREATE TABLE public.candidate_profile_interpretations (
 proposal_id uuid PRIMARY KEY,subject_id uuid NOT NULL,binding_id uuid NOT NULL,
 proposal_hash text,payload jsonb,source_proposal_id uuid,source_version bigint,target_row_id uuid,
 purpose text NOT NULL CHECK(purpose IN ('source_correction','private_exclusions','change_preferences')),
 consent_revision bigint NOT NULL CHECK(consent_revision>0),
 state text NOT NULL DEFAULT 'proposed' CHECK(state IN ('proposed','accepted','rejected','erased')),
 expires_at timestamptz NOT NULL,created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 FOREIGN KEY(subject_id,binding_id) REFERENCES public.candidate_owned_subjects(subject_id,flow_binding_id),
 CHECK(proposal_hash IS NULL OR proposal_hash~'^[0-9a-f]{64}$'),
 CHECK(payload IS NULL OR octet_length(payload::text)<=131072)
);
CREATE TABLE public.candidate_profile_interpretation_receipts (
 command_id uuid PRIMARY KEY,subject_id uuid NOT NULL,binding_id uuid NOT NULL,wire_hash text,result jsonb NOT NULL,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 FOREIGN KEY(subject_id,binding_id) REFERENCES public.candidate_owned_subjects(subject_id,flow_binding_id)
);
DO $$ DECLARE tab text;BEGIN
 FOREACH tab IN ARRAY ARRAY['candidate_profile_interpretations','candidate_profile_interpretation_receipts'] LOOP
  EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY',tab);
  EXECUTE format('ALTER TABLE public.%I FORCE ROW LEVEL SECURITY',tab);
  EXECUTE format('CREATE POLICY owner_only ON public.%I TO CURRENT_USER USING(true) WITH CHECK(true)',tab);
  EXECUTE format('REVOKE ALL ON public.%I FROM PUBLIC',tab);
 END LOOP;
END $$;

CREATE FUNCTION public.candidate_profile_interpretation_receive(p_subject uuid,p_binding uuid,p_command uuid,p_hash text,p_wire text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='5s' AS $$
DECLARE j jsonb;p jsonb;h public.candidate_profile_heads%ROWTYPE;c public.candidate_consent_state%ROWTYPE;
 r public.candidate_profile_interpretation_receipts%ROWTYPE;d public.candidate_preference_drafts%ROWTYPE;
 source public.candidate_profile_proposals%ROWTYPE;change jsonb;fact jsonb;prior jsonb;reason text;answer jsonb;expires timestamptz;
BEGIN
 IF p_wire IS NULL OR octet_length(p_wire)>131072 OR p_hash IS NULL OR p_hash!~'^[0-9a-f]{64}$'
 OR encode(sha256(convert_to(p_wire,'UTF8')),'hex')<>p_hash THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 j:=p_wire::jsonb;p:=j->'proposal';
 IF public.candidate_profile_canonical(j)<>p_wire OR j->'schemaVersion'<>'1'::jsonb
 OR j->>'subjectId' IS DISTINCT FROM p_subject::text OR j->>'bindingId' IS DISTINCT FROM p_binding::text
 OR j->>'commandId' IS DISTINCT FROM p_command::text OR coalesce(j->>'use','') NOT IN ('source_correction','private_exclusions','change_preferences')
 OR jsonb_typeof(p) IS DISTINCT FROM 'object' OR p->>'status' IS DISTINCT FROM 'proposal'
 OR encode(sha256(convert_to(public.candidate_profile_canonical(p-'proposalHash'),'UTF8')),'hex') IS DISTINCT FROM p->>'proposalHash'
 OR jsonb_typeof(p->'changes') IS DISTINCT FROM 'array' OR jsonb_array_length(p->'changes') NOT BETWEEN 1 AND 20
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 expires:=(p->>'expiresAt')::timestamptz;
 IF expires IS NULL OR expires>clock_timestamp()+interval '30 days' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 PERFORM set_config('app.current_tenant_id','candidate_'||p_subject::text,true);
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p_subject::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 INSERT INTO public.candidate_profile_heads(subject_id,binding_id) VALUES(p_subject,p_binding) ON CONFLICT DO NOTHING;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.binding_id<>p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501';END IF;
 SELECT * INTO r FROM public.candidate_profile_interpretation_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding) THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505';END IF;
  RETURN r.result;
 END IF;
 SELECT * INTO c FROM public.candidate_consent_state WHERE subject_id=p_subject;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject;
 IF c.effective_action IS DISTINCT FROM 'grant' OR c.effective_schema_version IS DISTINCT FROM 2
 OR c.highest_version IS DISTINCT FROM c.effective_version OR c.effective_version IS DISTINCT FROM (j->>'consentRevision')::bigint
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_copy_catalog WHERE copy_version=c.effective_copy_version AND copy_sha256=c.effective_copy_sha256 AND approved)
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_receipts cr WHERE cr.subject_id=p_subject AND cr.version=c.effective_version
  AND cr.authority_schema_version=2 AND cr.outcome='onboarding_granted'
  AND cr.copy_version=c.effective_copy_version AND cr.copy_sha256=c.effective_copy_sha256
  AND cr.purposes=ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact'])
 THEN reason:='consent_changed';
 ELSIF expires<=clock_timestamp() THEN reason:='proposal_expired';
 ELSIF (h.draft_version,h.profile_version,h.preference_version,coalesce(d.revision,0)) IS DISTINCT FROM
 ((p->>'expectedDraftVersion')::bigint,(p->>'expectedProfileVersion')::bigint,(p->>'expectedPreferenceVersion')::bigint,(p->>'expectedPreferenceDraftRevision')::bigint)
 THEN reason:='stale_edit';END IF;
 IF reason IS NULL AND j->>'use'='source_correction' THEN
  SELECT * INTO source FROM public.candidate_profile_proposals WHERE proposal_id=(j->>'sourceProposalId')::uuid AND subject_id=p_subject AND binding_id=p_binding;
  IF NOT FOUND OR source.state<>'proposed' OR source.source_version<>(j->>'sourceVersion')::bigint
   OR source.expires_at<=clock_timestamp() OR source.consent_revision<>c.effective_version THEN reason:='source_conflict';END IF;
 END IF;
 IF reason IS NULL THEN
  FOR change IN SELECT value FROM jsonb_array_elements(p->'changes') LOOP
   IF change->>'kind'='source_correction' AND j->>'use'='source_correction' THEN
    fact:=change->'fact';SELECT value INTO prior FROM jsonb_array_elements(source.payload->'facts') WHERE value->'factId'=fact->'factId';
    IF NOT FOUND OR NOT public.candidate_profile_source_fact_valid(fact) OR fact->>'kind' IN ('displayName','linkedinUrl','photo')
    OR (fact->'kind',fact->'targetRowId',fact->'evidence') IS DISTINCT FROM (prior->'kind',prior->'targetRowId',prior->'evidence')
    OR fact->'targetRowId' IS DISTINCT FROM j->'targetRowId'
    OR (prior->'value' ? 'description' AND fact->'value'->'description' IS DISTINCT FROM prior->'value'->'description')
    OR (prior->'value' ? 'id' AND fact->'value'->'id' IS DISTINCT FROM prior->'value'->'id') THEN reason:='source_conflict';EXIT;END IF;
   ELSIF change->>'kind'='preference_section' AND (j->>'use'='change_preferences'
     OR (j->>'use'='private_exclusions' AND change->'answer'->>'section'='private_notes')) THEN
    IF coalesce(change->'answer'->>'section','') NOT IN ('roles','location','arrangement','relocation','employment_types','availability','compensation','private_notes')
    THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
   ELSIF change->>'kind'='exclusions' AND j->>'use'='private_exclusions' THEN
    IF jsonb_typeof(change->'restrictions') IS DISTINCT FROM 'array' THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
   ELSE RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
  END LOOP;
 END IF;
 IF reason IS NULL THEN
  INSERT INTO public.candidate_profile_interpretations(proposal_id,subject_id,binding_id,proposal_hash,payload,source_proposal_id,source_version,target_row_id,purpose,consent_revision,expires_at)
  VALUES((p->>'proposalId')::uuid,p_subject,p_binding,p->>'proposalHash',p,(j->>'sourceProposalId')::uuid,(j->>'sourceVersion')::bigint,(j->>'targetRowId')::uuid,j->>'use',(j->>'consentRevision')::bigint,expires);
 END IF;
 answer:=jsonb_build_object('schemaVersion',1,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,'commandHash',p_hash,
 'proposalId',p->'proposalId','proposalHash',p->'proposalHash','outcome',CASE WHEN reason IS NULL THEN 'stored' ELSE 'rejected' END,'reason',reason);
 INSERT INTO public.candidate_profile_interpretation_receipts(command_id,subject_id,binding_id,wire_hash,result) VALUES(p_command,p_subject,p_binding,p_hash,answer);
 RETURN answer;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_interpretation_receive(uuid,uuid,uuid,text,text) FROM PUBLIC;

CREATE FUNCTION public.candidate_profile_interpretation_read(p_subject uuid,p_binding uuid,p_proposal uuid)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET statement_timeout='5s' AS $$
DECLARE h public.candidate_profile_heads%ROWTYPE;p public.candidate_profile_interpretations%ROWTYPE;
BEGIN
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject AND binding_id=p_binding;
 IF NOT FOUND THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501';END IF;
 SELECT * INTO p FROM public.candidate_profile_interpretations WHERE proposal_id=p_proposal AND subject_id=p_subject AND binding_id=p_binding;
 IF NOT FOUND OR p.state<>'proposed' OR p.payload IS NULL OR p.expires_at<=clock_timestamp() THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 RETURN p.payload;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_interpretation_read(uuid,uuid,uuid) FROM PUBLIC;
CREATE FUNCTION public.candidate_profile_interpretation_apply(p_subject uuid,p_binding uuid,p_command uuid,p_hash text,p_wire text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='5s' AS $$
DECLARE j jsonb;b jsonb;k text;h public.candidate_profile_heads%ROWTYPE;d public.candidate_preference_drafts%ROWTYPE;
 p public.candidate_profile_interpretations%ROWTYPE;r public.candidate_profile_receipts%ROWTYPE;c public.candidate_consent_state%ROWTYPE;
 change jsonb;sections jsonb;answer jsonb;reason text;changed boolean:=false;
 child jsonb;child_wire text;child_hash text;child_id uuid;selection jsonb:='[]';replacements jsonb:='[]';corrections jsonb:='[]';
BEGIN
 IF p_wire IS NULL OR octet_length(p_wire)>4096 OR p_hash IS NULL
 OR encode(sha256(convert_to(p_wire,'UTF8')),'hex') IS DISTINCT FROM p_hash THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 j:=p_wire::jsonb;b:=j->'payload';
 IF public.candidate_profile_canonical(j) IS DISTINCT FROM p_wire OR jsonb_typeof(j) IS DISTINCT FROM 'object'
 OR (SELECT count(*) FROM jsonb_object_keys(j))<>8
 OR NOT j ?& ARRAY['schemaVersion','action','idempotencyKey','payload','expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision']
 OR j->'schemaVersion' IS DISTINCT FROM '3'::jsonb OR j->>'action' IS DISTINCT FROM 'accept_interpretation'
 OR coalesce(j->>'idempotencyKey','')!~'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
 OR jsonb_typeof(b) IS DISTINCT FROM 'object' OR (SELECT count(*) FROM jsonb_object_keys(b))<>2
 OR NOT b ?& ARRAY['proposalId','proposalHash'] OR coalesce(b->>'proposalHash','')!~'^[0-9a-f]{64}$'
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 FOREACH k IN ARRAY ARRAY['expectedDraftVersion','expectedProfileVersion','expectedPreferenceVersion','expectedPreferenceDraftRevision'] LOOP
  IF jsonb_typeof(j->k) IS DISTINCT FROM 'number' OR (j->>k)!~'^(0|[1-9][0-9]{0,15})$' OR (j->>k)::numeric>9007199254740991
  THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 END LOOP;
 PERFORM set_config('app.current_tenant_id','candidate_'||p_subject::text,true);
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p_subject::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject AND flow_binding_id=p_binding)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject FOR UPDATE;
 IF h.binding_id IS DISTINCT FROM p_binding THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 IF h.restricted THEN RAISE EXCEPTION 'candidate_privacy_restricted' USING ERRCODE='42501';END IF;
 SELECT * INTO r FROM public.candidate_profile_receipts WHERE command_id=p_command;
 IF FOUND THEN
  IF (r.subject_id,r.binding_id) IS DISTINCT FROM (p_subject,p_binding) THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
  IF r.wire_hash IS DISTINCT FROM p_hash THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505';END IF;
  RETURN r.result;
 END IF;
 IF EXISTS(SELECT 1 FROM public.candidate_profile_receipts WHERE subject_id=p_subject AND idempotency_key=(j->>'idempotencyKey')::uuid)
 THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505';END IF;
 SELECT * INTO p FROM public.candidate_profile_interpretations WHERE proposal_id=(b->>'proposalId')::uuid AND subject_id=p_subject AND binding_id=p_binding FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject FOR UPDATE;
 SELECT * INTO c FROM public.candidate_consent_state WHERE subject_id=p_subject;
 IF p.state<>'proposed' OR p.proposal_hash IS DISTINCT FROM b->>'proposalHash' THEN reason:='source_conflict';
 ELSIF p.expires_at<=clock_timestamp() THEN reason:='proposal_expired';
 ELSIF (h.draft_version,h.profile_version,h.preference_version,coalesce(d.revision,0)) IS DISTINCT FROM
 ((j->>'expectedDraftVersion')::bigint,(j->>'expectedProfileVersion')::bigint,(j->>'expectedPreferenceVersion')::bigint,(j->>'expectedPreferenceDraftRevision')::bigint)
 OR (j->'expectedDraftVersion',j->'expectedProfileVersion',j->'expectedPreferenceVersion',j->'expectedPreferenceDraftRevision') IS DISTINCT FROM
 (p.payload->'expectedDraftVersion',p.payload->'expectedProfileVersion',p.payload->'expectedPreferenceVersion',p.payload->'expectedPreferenceDraftRevision') THEN reason:='stale_edit';
 ELSIF c.effective_action IS DISTINCT FROM 'grant' OR c.effective_schema_version IS DISTINCT FROM 2
 OR c.highest_version IS DISTINCT FROM c.effective_version OR c.effective_version IS DISTINCT FROM p.consent_revision
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_copy_catalog WHERE copy_version=c.effective_copy_version AND copy_sha256=c.effective_copy_sha256 AND approved)
 OR NOT EXISTS(SELECT 1 FROM public.candidate_consent_receipts cr WHERE cr.subject_id=p_subject AND cr.version=c.effective_version
 AND cr.authority_schema_version=2 AND cr.outcome='onboarding_granted' AND cr.copy_version=c.effective_copy_version AND cr.copy_sha256=c.effective_copy_sha256
 AND cr.purposes=ARRAY['job_matching','profile_enrichment','ai_processing','talent_network_discovery','recruiter_contact']) THEN reason:='consent_changed';END IF;
 -- Structured exclusions require their own explicit effects preview and accept.
 -- Never turn a model summary into an implicit hard-filter acceptance.
 IF reason IS NULL AND EXISTS(SELECT 1 FROM jsonb_array_elements(p.payload->'changes') x WHERE x.value->>'kind'='exclusions')
 THEN RAISE EXCEPTION 'exclusion_preview_required' USING ERRCODE='22023';END IF;
 IF reason IS NULL AND p.purpose='source_correction' THEN
  -- "Fix it, this is me" accepts the whole imported profile, not just its edits.
  -- The source apply below still checks ownership, source state, consent and versions.
  SELECT coalesce(jsonb_agg(f.value->'factId' ORDER BY f.ordinality),'[]'::jsonb) INTO selection
  FROM public.candidate_profile_proposals source,
   LATERAL jsonb_array_elements(source.payload->'facts') WITH ORDINALITY f(value,ordinality)
  WHERE source.proposal_id=p.source_proposal_id AND source.subject_id=p_subject AND source.binding_id=p_binding;
  FOR change IN SELECT value FROM jsonb_array_elements(p.payload->'changes') LOOP
   IF change#>'{fact,targetRowId}'<>'null'::jsonb OR change#>>'{fact,kind}' IN ('headline','currentLocation')
   THEN replacements:=replacements||jsonb_build_array(change#>>'{fact,factId}');END IF;
   corrections:=corrections||jsonb_build_array(jsonb_build_object('factId',change#>'{fact,factId}','kind',change#>'{fact,kind}','value',change#>'{fact,value}'));
  END LOOP;
  child_hash:=encode(sha256(convert_to(p_command::text||':interpretation-source','UTF8')),'hex');
  child_id:=(substr(child_hash,1,8)||'-'||substr(child_hash,9,4)||'-5'||substr(child_hash,14,3)||'-a'||substr(child_hash,18,3)||'-'||substr(child_hash,21,12))::uuid;
  child:=j||jsonb_build_object('action','accept_source','idempotencyKey',child_id,'payload',jsonb_build_object(
   'proposalId',p.source_proposal_id,'sourceVersion',p.source_version,
   'proposalHash',(SELECT proposal_hash FROM public.candidate_profile_proposals WHERE proposal_id=p.source_proposal_id),
   'selectedFactIds',selection,'replaceFactIds',replacements,'corrections',corrections));
  child_wire:=public.candidate_profile_canonical(child);
  answer:=public.candidate_profile_source_apply(p_subject,p_binding,child_id,encode(sha256(convert_to(child_wire,'UTF8')),'hex'),child_wire);
  reason:=answer->>'reason';changed:=(answer->>'changed')::boolean;
  SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=p_subject;
 ELSIF reason IS NULL THEN
  sections:=coalesce(d.sections,'{}'::jsonb);
  FOR change IN SELECT value FROM jsonb_array_elements(p.payload->'changes') LOOP
   IF change->>'kind' IS DISTINCT FROM 'preference_section' OR change#>>'{answer,section}'='authorisation'
   THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
   sections:=jsonb_set(sections,ARRAY[change#>>'{answer,section}'],(change->'answer')-'section',true);
  END LOOP;
  changed:=sections IS DISTINCT FROM coalesce(d.sections,'{}'::jsonb);
  IF changed THEN
   INSERT INTO public.candidate_preference_drafts(subject_id,revision,sections,content_hash)
   VALUES(p_subject,1,sections,encode(sha256(convert_to(public.candidate_profile_canonical(jsonb_build_object('sections',sections,'exclusions','[]'::jsonb)),'UTF8')),'hex'))
   ON CONFLICT(subject_id) DO UPDATE SET revision=candidate_preference_drafts.revision+1,sections=EXCLUDED.sections,
    content_hash=encode(sha256(convert_to(public.candidate_profile_canonical(jsonb_build_object('sections',EXCLUDED.sections,'exclusions',candidate_preference_drafts.exclusions)),'UTF8')),'hex'),updated_at=clock_timestamp();
   SELECT * INTO d FROM public.candidate_preference_drafts WHERE subject_id=p_subject;
  END IF;
 END IF;
 IF reason IS NULL THEN UPDATE public.candidate_profile_interpretations SET state='accepted' WHERE proposal_id=p.proposal_id;END IF;
 answer:=jsonb_build_object('schemaVersion',3,'commandId',p_command,'subjectId',p_subject,'bindingId',p_binding,'commandHash',p_hash,
 'status',CASE WHEN reason IS NULL THEN 'applied' ELSE 'rejected' END,'reason',reason,'changed',changed,
 'draftVersion',h.draft_version,'profileVersion',h.profile_version,'preferenceVersion',h.preference_version,
 'preferenceDraftRevision',coalesce(d.revision,0),'preferenceDraftHash',d.content_hash,'publicationActive',false);
 INSERT INTO public.candidate_profile_receipts(command_id,subject_id,binding_id,idempotency_key,wire_hash,result)
 VALUES(p_command,p_subject,p_binding,(j->>'idempotencyKey')::uuid,p_hash,answer);RETURN answer;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_interpretation_apply(uuid,uuid,uuid,text,text) FROM PUBLIC;
CREATE FUNCTION public.candidate_profile_interpretations_scrub() RETURNS trigger LANGUAGE plpgsql
SECURITY DEFINER SET search_path=pg_catalog,public AS $$ BEGIN
 IF NEW.restricted THEN
  UPDATE public.candidate_profile_interpretations SET payload=NULL,proposal_hash=NULL,state='erased' WHERE subject_id=NEW.subject_id;
  UPDATE public.candidate_profile_interpretation_receipts SET wire_hash=NULL,result=result-'commandHash'-'proposalHash' WHERE subject_id=NEW.subject_id;
 END IF;RETURN NEW;
END $$;
REVOKE ALL ON FUNCTION public.candidate_profile_interpretations_scrub() FROM PUBLIC;
CREATE TRIGGER candidate_profile_interpretations_erasure AFTER UPDATE OF restricted ON public.candidate_profile_heads
FOR EACH ROW EXECUTE FUNCTION public.candidate_profile_interpretations_scrub();
DO $$ DECLARE entry record; BEGIN
 FOR entry IN SELECT c.oid::regclass target,r.rolname FROM pg_class c CROSS JOIN LATERAL aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a
 JOIN pg_roles r ON r.oid=a.grantee WHERE c.relnamespace='public'::regnamespace AND c.relname IN ('candidate_profile_sources','candidate_profile_proposals','candidate_profile_source_receipts','candidate_profile_interpretations','candidate_profile_interpretation_receipts') AND a.grantee<>c.relowner LOOP
  EXECUTE format('REVOKE ALL ON TABLE %s FROM %I',entry.target,entry.rolname);
 END LOOP;
 FOR entry IN SELECT p.oid::regprocedure target,r.rolname FROM pg_proc p CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
 JOIN pg_roles r ON r.oid=a.grantee WHERE p.pronamespace='public'::regnamespace AND p.proname IN ('candidate_profile_source_receive','candidate_profile_source_read','candidate_profile_source_apply','candidate_profile_sources_scrub','candidate_profile_source_date_valid','candidate_profile_source_fact_valid','candidate_profile_interpretation_receive','candidate_profile_interpretation_read','candidate_profile_interpretation_apply','candidate_profile_interpretations_scrub') AND a.grantee<>p.proowner LOOP
  EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',entry.target,entry.rolname);
 END LOOP;
END $$;
