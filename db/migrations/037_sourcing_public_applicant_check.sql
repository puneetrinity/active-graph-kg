-- Exact organisation membership only. No names, emails, skills or inferred links.
CREATE FUNCTION public.candidate_index_public_applicants(p_ids jsonb)
RETURNS SETOF jsonb LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE v_scope text:=current_setting('app.current_tenant_id',true); item jsonb; seen uuid[]:='{}'; uid uuid;
BEGIN
 IF (v_scope ~ '^org_[1-9][0-9]{0,9}$' AND substring(v_scope FROM 5)::bigint<=2147483647
  AND jsonb_typeof(p_ids)='array' AND jsonb_array_length(p_ids) BETWEEN 1 AND 100
  AND octet_length(p_ids::text)<=8192) IS NOT TRUE THEN
  RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_sourcing_bounds'; END IF;
 FOR item IN SELECT value FROM jsonb_array_elements(p_ids) LOOP
  IF (jsonb_typeof(item)='string' AND item#>>'{}' ~ '^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$'
   AND item#>>'{}'<>'00000000-0000-0000-0000-000000000000') IS NOT TRUE THEN
   RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_sourcing_identity'; END IF;
  uid:=(item#>>'{}')::uuid;
  IF uid=ANY(seen) THEN RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_sourcing_identity'; END IF;
  seen:=array_append(seen,uid);
  RETURN NEXT jsonb_build_object('publicId',uid,'isApplicant',EXISTS(
   SELECT 1 FROM public.candidates c JOIN public.organization_candidate_references r
    ON r.tenant_id=c.tenant_id AND r.candidate_id=c.candidate_id
   WHERE c.tenant_id=v_scope AND c.global_candidate_id=uid));
 END LOOP;
END $$;
REVOKE ALL ON FUNCTION public.candidate_index_public_applicants(jsonb) FROM PUBLIC;
