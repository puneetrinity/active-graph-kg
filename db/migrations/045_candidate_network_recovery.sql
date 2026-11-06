-- B5 restore recovery gate. Native Spark discovery is open only when an owner-run
-- reconciliation has attested the CURRENT recovery epoch. The epoch lives in
-- deployment configuration, outside this database: a restored database only
-- carries attestations for older epochs and therefore cannot reopen itself.
-- Old recruiter searches, applications and stored publications are untouched.
CREATE TABLE public.candidate_network_recovery_attestations (
 epoch text PRIMARY KEY CHECK(epoch ~ '^[A-Za-z0-9_-]{16,128}$'),
 run_id uuid NOT NULL UNIQUE,
 restored text NOT NULL CHECK(restored IN ('none','memory','flow')),
 summary jsonb NOT NULL CHECK(jsonb_typeof(summary)='object' AND octet_length(summary::text)<=16384),
 operator_role name NOT NULL,
 attested_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
ALTER TABLE public.candidate_network_recovery_attestations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_network_recovery_attestations FORCE ROW LEVEL SECURITY;
DO $$ BEGIN EXECUTE format('CREATE POLICY owner_only ON public.candidate_network_recovery_attestations TO %I USING(true) WITH CHECK(true)',current_user); END $$;
REVOKE ALL ON public.candidate_network_recovery_attestations FROM PUBLIC;

-- A subject whose state could not be proven current after a restore. Its existing
-- publications are held (never deleted); only a later candidate confirmation
-- creates a new, independently eligible publication.
CREATE TABLE public.candidate_network_recovery_holds (
 subject_id uuid NOT NULL REFERENCES public.candidate_owned_subjects(subject_id),
 epoch text NOT NULL CHECK(epoch ~ '^[A-Za-z0-9_-]{16,128}$'),
 reason text NOT NULL CHECK(reason IN ('profile_rewound','profile_flow_behind','profile_conflict','consent_conflict','consent_flow_behind',
  'consent_replay_failed','privacy_directive_missing','binding_missing')),
 held_publications integer NOT NULL CHECK(held_publications>=0),
 operator_role name NOT NULL,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 PRIMARY KEY(subject_id,epoch,reason)
);
ALTER TABLE public.candidate_network_recovery_holds ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.candidate_network_recovery_holds FORCE ROW LEVEL SECURITY;
DO $$ BEGIN EXECUTE format('CREATE POLICY owner_only ON public.candidate_network_recovery_holds TO %I USING(true) WITH CHECK(true)',current_user); END $$;
REVOKE ALL ON public.candidate_network_recovery_holds FROM PUBLIC;

-- Read-only gate. Runtime (search, validate, owner status) and the publication
-- worker call this; it never writes and cannot be satisfied by an old epoch.
CREATE FUNCTION public.candidate_network_recovery_open(p_epoch text) RETURNS boolean
LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
 SELECT p_epoch IS NOT NULL AND p_epoch ~ '^[A-Za-z0-9_-]{16,128}$'
  AND EXISTS(SELECT 1 FROM public.candidate_network_recovery_attestations a WHERE a.epoch=p_epoch)
$$;

-- Owner-only. Holds every not-yet-revoked publication of the subject; idempotent.
CREATE FUNCTION public.candidate_network_recovery_hold(p_subject uuid,p_epoch text,p_reason text) RETURNS integer
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE n integer;
BEGIN
 IF p_subject IS NULL OR p_epoch IS NULL OR p_epoch !~ '^[A-Za-z0-9_-]{16,128}$' OR p_reason IS NULL
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=p_subject)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-consent:'||p_subject::text,0));
 PERFORM pg_advisory_xact_lock(hashtextextended('candidate-profile:'||p_subject::text,0));
 UPDATE public.candidate_network_publications SET state='held',lease_token=NULL,lease_until=NULL
 WHERE subject_id=p_subject AND state IN ('staged','embedding','active');
 GET DIAGNOSTICS n=ROW_COUNT;
 INSERT INTO public.candidate_network_recovery_holds(subject_id,epoch,reason,held_publications,operator_role)
 VALUES(p_subject,p_epoch,p_reason,n,session_user)
 ON CONFLICT(subject_id,epoch,reason) DO UPDATE SET held_publications=public.candidate_network_recovery_holds.held_publications+EXCLUDED.held_publications;
 RETURN n;
END $$;

-- Fingerprint of the privacy, consent and profile state the reconciliation checked. Runs as the caller,
-- which must see every row (superuser or BYPASSRLS), so row security cannot hide a change.
CREATE FUNCTION public.candidate_network_recovery_fingerprint() RETURNS text
LANGUAGE plpgsql STABLE SET search_path=pg_catalog,public AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=current_user AND (rolsuper OR rolbypassrls))
 THEN RAISE EXCEPTION 'recovery_reader_not_privileged' USING ERRCODE='42501';END IF;
 RETURN md5(concat_ws('|',
  -- Every field the reconciliation reads, not just row counts.
  (SELECT coalesce(md5(string_agg(cursor||':'||directive_id::text||':'||directive_version||':'||resulting_state,',' ORDER BY cursor)),'') FROM public.candidate_privacy_directive_events),
  (SELECT coalesce(md5(string_agg(directive_id::text||':'||version||':'||state,',' ORDER BY directive_id)),'') FROM public.candidate_privacy_directives),
  (SELECT coalesce(md5(string_agg(subject_id::text||':'||highest_version||':'||effective_version||':'||coalesce(effective_action,''),',' ORDER BY subject_id)),'') FROM public.candidate_consent_state),
  (SELECT coalesce(md5(string_agg(event_id::text||':'||subject_id::text||':'||version||':'||command_digest,',' ORDER BY event_id)),'') FROM public.candidate_consent_receipts),
  (SELECT coalesce(md5(string_agg(subject_id::text||':'||draft_version||':'||profile_version||':'||preference_version,',' ORDER BY subject_id)),'') FROM public.candidate_profile_heads),
  (SELECT coalesce(md5(string_agg(command_id::text||':'||subject_id::text,',' ORDER BY command_id)),'') FROM public.candidate_profile_receipts),
  (SELECT coalesce(md5(string_agg(command_id::text||':'||subject_id::text,',' ORDER BY command_id)),'') FROM public.candidate_profile_source_receipts),
  (SELECT coalesce(md5(string_agg(command_id::text||':'||subject_id::text,',' ORDER BY command_id)),'') FROM public.candidate_profile_interpretation_receipts),
  (SELECT coalesce(md5(string_agg(subject_id::text||':'||flow_binding_id::text,',' ORDER BY subject_id)),'') FROM public.candidate_owned_subjects)));
END $$;

-- Owner-run only. One attestation per epoch; a different run cannot overwrite it. The reopening is fenced:
-- writes to the checked tables are blocked while the fingerprint is recomputed, and any change since the
-- reconciliation's final check refuses the attestation (the operator reruns).
CREATE FUNCTION public.candidate_network_recovery_attest(p_epoch text,p_run uuid,p_restored text,p_summary jsonb,p_fingerprint text) RETURNS boolean
LANGUAGE plpgsql SET search_path=pg_catalog,public SET lock_timeout='2s' SET statement_timeout='10s' AS $$
DECLARE a public.candidate_network_recovery_attestations%ROWTYPE;
BEGIN
 IF p_epoch IS NULL OR p_epoch !~ '^[A-Za-z0-9_-]{16,128}$' OR p_run IS NULL OR p_fingerprint IS NULL
 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 SELECT * INTO a FROM public.candidate_network_recovery_attestations WHERE epoch=p_epoch;
 IF FOUND THEN
  IF a.run_id IS DISTINCT FROM p_run THEN RAISE EXCEPTION 'idempotency_conflict' USING ERRCODE='23505';END IF;
  RETURN true;
 END IF;
 LOCK TABLE public.candidate_privacy_directive_events,public.candidate_privacy_directives,public.candidate_consent_state,
  public.candidate_consent_receipts,public.candidate_profile_heads,public.candidate_profile_receipts,
  public.candidate_profile_source_receipts,public.candidate_profile_interpretation_receipts,public.candidate_owned_subjects IN SHARE MODE;
 IF public.candidate_network_recovery_fingerprint() IS DISTINCT FROM p_fingerprint
 THEN RAISE EXCEPTION 'recovery_state_changed' USING ERRCODE='40001';END IF;
 INSERT INTO public.candidate_network_recovery_attestations(epoch,run_id,restored,summary,operator_role)
 VALUES(p_epoch,p_run,p_restored,p_summary,session_user);
 RETURN true;
END $$;

DO $$ DECLARE f record;BEGIN
 FOR f IN SELECT c.oid::regclass target,r.rolname FROM pg_class c
 CROSS JOIN LATERAL aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a JOIN pg_roles r ON r.oid=a.grantee
 WHERE c.oid IN ('public.candidate_network_recovery_attestations'::regclass,'public.candidate_network_recovery_holds'::regclass)
 AND a.grantee<>c.relowner LOOP
 EXECUTE format('REVOKE ALL ON TABLE %s FROM %I',f.target,f.rolname);END LOOP;
 FOR f IN SELECT p.oid::regprocedure target,r.rolname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
 CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a JOIN pg_roles r ON r.oid=a.grantee
 WHERE n.nspname='public' AND p.proname IN ('candidate_network_recovery_open','candidate_network_recovery_hold','candidate_network_recovery_attest','candidate_network_recovery_fingerprint')
 AND a.grantee<>p.proowner LOOP
 EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %I',f.target,f.rolname);END LOOP;
 REVOKE ALL ON FUNCTION public.candidate_network_recovery_open(text) FROM PUBLIC;
 REVOKE ALL ON FUNCTION public.candidate_network_recovery_hold(uuid,text,text) FROM PUBLIC;
 REVOKE ALL ON FUNCTION public.candidate_network_recovery_attest(text,uuid,text,jsonb,text) FROM PUBLIC;
 REVOKE ALL ON FUNCTION public.candidate_network_recovery_fingerprint() FROM PUBLIC;
END $$;
