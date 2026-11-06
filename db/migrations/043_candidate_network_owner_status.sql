-- Owner-only display state; never grants publication or recruiter authority.
-- Keep the boolean reader intact for older clients.
CREATE FUNCTION public.candidate_network_owner_status_detail(s uuid,b uuid) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET statement_timeout='3s' AS $$
DECLARE previous_tenant text:=coalesce(current_setting('app.current_tenant_id',true),'');
 h public.candidate_profile_heads; cs public.candidate_consent_state; reason text; active boolean:=false;
BEGIN
 IF NOT EXISTS(SELECT 1 FROM public.candidate_owned_subjects WHERE subject_id=s AND flow_binding_id=b)
 THEN RAISE EXCEPTION 'not_found' USING ERRCODE='P0002';END IF;
 PERFORM set_config('app.current_tenant_id','candidate_'||s::text,true);
 SELECT * INTO h FROM public.candidate_profile_heads WHERE subject_id=s;
 SELECT * INTO cs FROM public.candidate_consent_state WHERE subject_id=s;
 IF coalesce(h.restricted,false) THEN reason:='restricted';
 ELSIF cs.effective_action IS DISTINCT FROM 'grant' OR cs.effective_schema_version IS DISTINCT FROM 2
  OR cs.highest_version IS DISTINCT FROM cs.effective_version THEN reason:='consent_required';
 ELSIF NOT EXISTS(SELECT 1 FROM public.candidate_profile_confirmations c WHERE c.subject_id=s
  AND c.profile_version=h.profile_version AND c.preference_version=h.preference_version
  AND c.consent_revision=cs.effective_version) THEN reason:='confirmation_required';
 ELSIF NOT public.candidate_network_eligible(s,h.profile_version,h.preference_version,cs.effective_version)
 THEN reason:='publication_unavailable';
 ELSE
  SELECT CASE p.state WHEN 'active' THEN NULL WHEN 'held' THEN 'publication_held'
   WHEN 'staged' THEN 'publication_pending' WHEN 'embedding' THEN 'publication_pending'
   ELSE 'publication_unavailable' END,p.state='active' INTO reason,active
  FROM public.candidate_network_publications p WHERE p.subject_id=s AND p.profile_version=h.profile_version
   AND p.preference_version=h.preference_version AND p.consent_revision=cs.effective_version;
  IF NOT FOUND THEN reason:='publication_pending';active:=false;END IF;
 END IF;
 PERFORM set_config('app.current_tenant_id',previous_tenant,true);
 RETURN jsonb_build_object('schemaVersion',1,'publicationActive',active,'blockedReason',reason);
END $$;
REVOKE ALL ON FUNCTION public.candidate_network_owner_status_detail(uuid,uuid) FROM PUBLIC;
