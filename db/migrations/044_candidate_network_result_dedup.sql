-- Service-only result comparison. This is not an ownership link, a contact
-- entitlement, or a change to the immutable public facts/ranking evidence.
CREATE FUNCTION public.candidate_network_result_keys(ids uuid[]) RETURNS jsonb
LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public SET statement_timeout='3s' AS $$
DECLARE result jsonb:='[]'; item record; previous_tenant text:=coalesce(current_setting('app.current_tenant_id',true),''); url text;
BEGIN
 IF ids IS NULL OR cardinality(ids)>500 THEN RAISE EXCEPTION 'invalid_request' USING ERRCODE='22023';END IF;
 FOR item IN SELECT p.* FROM public.candidate_network_publications p
 WHERE p.publication_id=ANY(ids) AND p.state='active'
 AND public.candidate_network_eligible(p.subject_id,p.profile_version,p.preference_version,p.consent_revision)
 ORDER BY p.publication_id LOOP
  PERFORM set_config('app.current_tenant_id','candidate_'||item.subject_id::text,true);
  SELECT facts->>'linkedinUrl' INTO url FROM public.candidate_profile_versions
   WHERE subject_id=item.subject_id AND version=item.profile_version;
  -- Confirmed profiles already use canonical URLs. Refuse unexpected forms;
  -- Discover performs its existing normalization on the comparison URL.
  IF url ~ '^https://linkedin[.]com/in/[A-Za-z0-9_%~-]+$' THEN
   result:=result||jsonb_build_array(jsonb_build_object('publicationId',item.publication_id,
    'subjectId',item.subject_id,'profileVersion',item.profile_version,'preferenceVersion',item.preference_version,
    'consentRevision',item.consent_revision,'contentHash',item.content_hash,'linkedinUrl',url));
  END IF;
 END LOOP;
 PERFORM set_config('app.current_tenant_id',previous_tenant,true);
 RETURN result;
END $$;
REVOKE ALL ON FUNCTION public.candidate_network_result_keys(uuid[]) FROM PUBLIC;
