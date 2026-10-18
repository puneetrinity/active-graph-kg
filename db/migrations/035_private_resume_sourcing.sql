-- Read-only, tenant-scoped professional evidence. No extraction or recovery.
-- Existing readers and existing source/generation rows are unchanged.
CREATE FUNCTION public.candidate_index_revalidate_sourcing(p_job_id integer,p_items jsonb)
RETURNS SETOF jsonb LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE v_scope text:=current_setting('app.current_tenant_id',true); item jsonb; s record;
  allowed jsonb; decision text; k text;
BEGIN
  IF (v_scope ~ '^org_[1-9][0-9]{0,9}$' AND substring(v_scope FROM 5)::bigint<=2147483647
    AND p_job_id>0 AND jsonb_typeof(p_items)='array'
    AND jsonb_array_length(p_items) BETWEEN 1 AND 100
    AND octet_length(p_items::text)<=65536) IS NOT TRUE THEN
    RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_sourcing_bounds'; END IF;
  FOR item IN SELECT value FROM jsonb_array_elements(p_items) LOOP
    IF (jsonb_typeof(item)='object' AND item ?& ARRAY['candidateId','referenceId','applicationId',
      'applicationJobId','resumeVersionId','sourceId','generationId','sourceHash']
      AND item-ARRAY['candidateId','referenceId','applicationId','applicationJobId',
        'resumeVersionId','sourceId','generationId','sourceHash']='{}'::jsonb
      AND jsonb_typeof(item->'applicationId')='number'
      AND jsonb_typeof(item->'applicationJobId')='number'
      AND item->>'applicationId' ~ '^[1-9][0-9]{0,9}$'
      AND item->>'applicationJobId' ~ '^[1-9][0-9]{0,9}$'
      AND (item->>'applicationId')::bigint<=2147483647
      AND (item->>'applicationJobId')::bigint<=2147483647
      AND jsonb_typeof(item->'sourceHash')='string'
      AND item->>'sourceHash' ~ '^[0-9a-f]{64}$') IS NOT TRUE THEN
      RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_sourcing_identity'; END IF;
    FOREACH k IN ARRAY ARRAY['candidateId','referenceId','resumeVersionId','sourceId','generationId'] LOOP
      IF (jsonb_typeof(item->k)='string'
        AND item->>k ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        AND item->>k<>'00000000-0000-0000-0000-000000000000') IS NOT TRUE THEN
        RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_sourcing_identity'; END IF;
    END LOOP;
    SELECT src.*,r.application_id,r.job_id,c.global_candidate_id linked_global_id,
      h.published_generation_id,h.authority_source_id INTO s
      FROM public.candidate_index_sources src
      JOIN public.organization_candidate_references r ON r.tenant_id=src.tenant_id
        AND r.reference_id=src.reference_id AND r.candidate_id=src.candidate_id
      JOIN public.candidates c ON c.tenant_id=src.tenant_id AND c.candidate_id=src.candidate_id
      JOIN public.candidate_index_heads h ON h.scope_key=src.scope_key AND h.authority_id=src.authority_id
      WHERE src.scope_key=v_scope AND src.tenant_id=v_scope AND src.source_kind='organization_application'
        AND src.source_id=(item->>'sourceId')::uuid AND src.candidate_id=(item->>'candidateId')::uuid;
    decision:='not_found';
    IF FOUND THEN
      IF s.reference_id<>(item->>'referenceId')::uuid OR s.resume_version_id<>(item->>'resumeVersionId')::uuid
        OR s.application_id<>(item->>'applicationId')::integer OR s.job_id<>(item->>'applicationJobId')::integer
        OR s.source_hash<>item->>'sourceHash' OR s.authority_source_id<>s.source_id
        OR s.published_generation_id IS DISTINCT FROM (item->>'generationId')::uuid THEN
        decision:='source_changed';
      ELSE
        allowed:=public.candidate_index_status(s.source_id);
        IF (allowed->>'eligible')::boolean IS DISTINCT FROM true THEN
          decision:=CASE WHEN allowed->>'reason' IN ('privacy_review','privacy_restricted')
            THEN 'privacy_restricted' ELSE 'unavailable' END;
        ELSIF EXISTS (SELECT 1 FROM public.organization_candidate_references r
          JOIN public.candidates c ON c.tenant_id=r.tenant_id AND c.candidate_id=r.candidate_id
          WHERE r.tenant_id=v_scope AND r.job_id=p_job_id AND (r.candidate_id=s.candidate_id
            OR (s.linked_global_id IS NOT NULL AND c.global_candidate_id=s.linked_global_id))) THEN
          decision:='same_job';
        ELSE decision:='allow'; END IF;
      END IF;
    END IF;
    RETURN NEXT jsonb_build_object('identity',item,'decision',decision);
  END LOOP;
END;
$$;

CREATE FUNCTION public.candidate_index_read_sourcing(p_query public.vector,p_model_id text,
  p_revision text,p_limit integer,p_job_id integer)
RETURNS SETOF jsonb LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public SET lock_timeout='1500ms' SET statement_timeout='3s' AS $$
DECLARE v_scope text:=current_setting('app.current_tenant_id',true); item record;
  identity jsonb; checked jsonb; results jsonb:='[]'; seen uuid[]:='{}'; person uuid;
BEGIN
  IF (v_scope ~ '^org_[1-9][0-9]{0,9}$' AND substring(v_scope FROM 5)::bigint<=2147483647
    AND p_job_id>0 AND p_limit BETWEEN 1 AND 100 AND vector_dims(p_query)=384
    AND (p_query <#> p_query) BETWEEN -1.001 AND -0.999
    AND octet_length(p_model_id) BETWEEN 1 AND 200
    AND p_revision ~ '^([0-9a-f]{40}|[0-9a-f]{64})$') IS NOT TRUE THEN
    RAISE EXCEPTION USING ERRCODE='22023',MESSAGE='candidate_sourcing_bounds'; END IF;
  -- Latest eligible observation per authoritative person, then vector distance.
  -- No LIMIT before privacy/same-job checks. SQL timeout bounds the scan.
  FOR item IN SELECT s.*,r.application_id,r.job_id,g.generation_id,
      c.global_candidate_id linked_global_id,e.namespace,
      (SELECT min(v.embedding <=> p_query) FROM public.candidate_index_vectors v
        WHERE v.generation_id=g.generation_id) distance
    FROM public.candidate_index_heads h
    JOIN public.candidate_index_generations g ON g.generation_id=h.published_generation_id
    JOIN public.candidate_index_sources s ON s.source_id=g.source_id AND s.source_id=h.authority_source_id
    JOIN public.organization_candidate_references r ON r.tenant_id=s.tenant_id
      AND r.reference_id=s.reference_id AND r.candidate_id=s.candidate_id
    JOIN public.candidates c ON c.tenant_id=s.tenant_id AND c.candidate_id=s.candidate_id
    JOIN public.candidate_index_extractions e ON e.generation_id=g.generation_id
    WHERE h.scope_key=v_scope AND s.scope_key=v_scope AND s.tenant_id=v_scope
      AND s.source_kind='organization_application'
      AND g.policy->>'embedding_model_id'=p_model_id AND g.policy->>'embedding_artifact_revision'=p_revision
    ORDER BY s.source_observed_at DESC,s.reference_id,s.source_id
  LOOP
    person:=coalesce(item.linked_global_id,item.candidate_id);
    IF person=ANY(seen) OR item.distance IS NULL THEN CONTINUE; END IF;
    identity:=jsonb_build_object('candidateId',item.candidate_id,'referenceId',item.reference_id,
      'applicationId',item.application_id,'applicationJobId',item.job_id,
      'resumeVersionId',item.resume_version_id,'sourceId',item.source_id,
      'generationId',item.generation_id,'sourceHash',item.source_hash);
    SELECT v INTO checked FROM public.candidate_index_revalidate_sourcing(p_job_id,jsonb_build_array(identity)) v;
    IF checked->>'decision'='unavailable' THEN
      RAISE EXCEPTION USING ERRCODE='55000',MESSAGE='candidate_sourcing_unavailable'; END IF;
    IF checked->>'decision'<>'allow' THEN CONTINUE; END IF;
    seen:=array_append(seen,person);
    results:=results || jsonb_build_array(jsonb_build_object('identity',identity,
      'linkedPublicId',item.linked_global_id,'distance',item.distance,
      'evidence',jsonb_build_object('version',1,'kind','candidate_provided','scope',v_scope,
        'sourceVersion',item.source_version,'observedAt',item.source_observed_at,
        'skills',coalesce(nullif(item.namespace->'skills_raw','null'::jsonb),
          nullif(item.namespace->'skills_normalized','null'::jsonb),'[]'::jsonb),
        'currentTitle',item.namespace->'current_title','seniority',item.namespace->'seniority',
        'primaryTitles',coalesce(nullif(item.namespace->'primary_titles','null'::jsonb),'[]'::jsonb),
        'domains',coalesce(nullif(item.namespace->'domains','null'::jsonb),'[]'::jsonb),
        'functions',coalesce(nullif(item.namespace->'functions','null'::jsonb),'[]'::jsonb),
        'certifications',coalesce(nullif(item.namespace->'certifications','null'::jsonb),'[]'::jsonb),
        'industries',coalesce(nullif(item.namespace->'industries','null'::jsonb),'[]'::jsonb),
        'location',jsonb_build_object('city',item.namespace->'location'->'city',
          'country',item.namespace->'location'->'country'))));
    -- Keep memory bounded to limit+1 while scanning; stable tie order.
    SELECT coalesce(jsonb_agg(value ORDER BY (value->>'distance')::double precision,
      value->'identity'->>'candidateId'),'[]') INTO results FROM
      (SELECT value FROM jsonb_array_elements(results) ORDER BY (value->>'distance')::double precision,
        value->'identity'->>'candidateId' LIMIT p_limit) ranked;
  END LOOP;
  RETURN QUERY SELECT value FROM jsonb_array_elements(results);
END;
$$;
REVOKE ALL ON FUNCTION public.candidate_index_read_sourcing(public.vector,text,text,integer,integer) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.candidate_index_revalidate_sourcing(integer,jsonb) FROM PUBLIC;
