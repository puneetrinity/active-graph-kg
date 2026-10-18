-- Opt-in v3 dated employment. No old rows, policy activation or recovery.
-- Existing v1/v2 extraction and two-call/45-second rules are unchanged.
CREATE FUNCTION public.candidate_index_history_valid(p_history jsonb,p_text text) RETURNS boolean
LANGUAGE plpgsql IMMUTABLE SET search_path=pg_catalog,public AS $fn$
DECLARE role jsonb; field text; value text; excerpt text; parts text[];
 normalized text; written text; captures text[]; month_name text; month_number integer; pattern text;
BEGIN
 IF jsonb_typeof(p_history) IS DISTINCT FROM 'array' OR jsonb_array_length(p_history)>20
  OR p_text IS NULL THEN RETURN false; END IF;
 FOR role IN SELECT v FROM jsonb_array_elements(p_history) v LOOP
  IF jsonb_typeof(role) IS DISTINCT FROM 'object'
   OR NOT role ?& ARRAY['title','employment_type','start','end','ongoing','start_quote','end_quote']
   OR role-ARRAY['title','employment_type','start','end','ongoing','start_quote','end_quote']<>'{}'::jsonb
   OR jsonb_typeof(role->'ongoing') IS DISTINCT FROM 'boolean' THEN RETURN false; END IF;
  FOREACH field IN ARRAY ARRAY['title','employment_type'] LOOP
   IF role->field<>'null'::jsonb AND (jsonb_typeof(role->field)<>'string'
     OR length(role->>field) NOT BETWEEN 1 AND CASE WHEN field='title' THEN 200 ELSE 100 END OR btrim(role->>field)=''
     OR strpos(p_text,role->>field)=0 OR role->>field ~ '[[:cntrl:]@]|https?://') THEN RETURN false; END IF;
  END LOOP;
  FOREACH field IN ARRAY ARRAY['start','end'] LOOP
   value:=role->>field; excerpt:=role->>(field||'_quote');
   IF field='end' AND role->'ongoing'='true'::jsonb THEN
    IF value IS NOT NULL OR excerpt IS NULL OR btrim(excerpt)='' OR length(excerpt)>120
      OR strpos(p_text,excerpt)=0
      OR lower(btrim(excerpt)) !~ '^(present|current|currently|ongoing|till date|to date|now)$' THEN RETURN false; END IF;
   ELSIF value IS NULL THEN
    IF excerpt IS NOT NULL THEN RETURN false; END IF;
   ELSE
    IF jsonb_typeof(role->field)<>'string' OR value !~ '^(19|20|21)[0-9]{2}(-[0-9]{2}(-[0-9]{2})?)?$'
      OR excerpt IS NULL OR btrim(excerpt)='' OR length(excerpt)>120 OR strpos(p_text,excerpt)=0
      OR excerpt !~ ('(^|[^0-9])'||left(value,4)||'([^0-9]|$)') THEN RETURN false; END IF;
    parts:=string_to_array(value,'-');
    IF NOT pg_input_is_valid(parts[1]||'-'||coalesce(parts[2],'01')||'-'||coalesce(parts[3],'01'),'date')
      OR (cardinality(parts)>1 AND excerpt ~ '^[[:space:]]*[0-9]{4}[[:space:]]*$') THEN RETURN false; END IF;
    written:=rtrim(lower(btrim(excerpt)),'.'); normalized:=NULL;
    IF written ~ '^(19|20|21)[0-9]{2}(-[0-9]{2}(-[0-9]{2})?)?$' THEN normalized:=written;
    ELSIF written ~ '^[0-9]{1,2}/[0-9]{4}$' THEN
      captures:=string_to_array(written,'/'); normalized:=captures[2]||'-'||lpad(captures[1],2,'0');
    ELSE
      month_number:=0;
      FOREACH month_name IN ARRAY string_to_array('january february march april may june july august september october november december',' ') LOOP
        month_number:=month_number+1;
        pattern:='(?:'||month_name||'|'||left(month_name,3)||'\.?)';
        captures:=regexp_match(written,'^'||pattern||'[[:space:]]+([0-9]{4})$');
        IF captures IS NOT NULL THEN normalized:=captures[1]||'-'||lpad(month_number::text,2,'0'); EXIT; END IF;
        captures:=regexp_match(written,'^'||pattern||'[[:space:]]+([0-9]{1,2}),?[[:space:]]+([0-9]{4})$');
        IF captures IS NULL THEN captures:=regexp_match(written,'^([0-9]{1,2})[[:space:]]+'||pattern||'[[:space:]]+([0-9]{4})$'); END IF;
        IF captures IS NOT NULL THEN normalized:=captures[2]||'-'||lpad(month_number::text,2,'0')||'-'||lpad(captures[1],2,'0'); EXIT; END IF;
      END LOOP;
    END IF;
    IF normalized IS DISTINCT FROM value THEN RETURN false; END IF;
   END IF;
  END LOOP;
  IF role->>'start' IS NOT NULL AND role->>'end' IS NOT NULL
    AND left(role->>'start',least(length(role->>'start'),length(role->>'end')))
       >left(role->>'end',least(length(role->>'start'),length(role->>'end'))) THEN RETURN false; END IF;
 END LOOP;
 RETURN true;
END $fn$;
REVOKE ALL ON FUNCTION public.candidate_index_history_valid(jsonb,text) FROM PUBLIC;

DO $forward$ DECLARE definition text; source text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.candidate_index_claim(text,integer,jsonb,uuid,uuid,bigint,text)'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>'01ec596765930196ffd61ba4da6d64e78d3eb3e84516e6fdccec5ad15e41749f' THEN
  RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 IF (length(definition)-length(replace(definition,'^candidate-index-extraction:v2:[0-9a-f]{64}$','')))/length('^candidate-index-extraction:v2:[0-9a-f]{64}$')<>1 THEN RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 definition:=replace(definition,'^candidate-index-extraction:v2:[0-9a-f]{64}$','^candidate-index-extraction:v[23]:[0-9a-f]{64}$');
 EXECUTE definition;
END $forward$;

DO $forward$ DECLARE definition text; source text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.candidate_index_complete_extract(uuid,uuid,bigint,text,text,jsonb,jsonb,double precision,text,jsonb,text,text)'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>'75e9b0811d568170be1961c9f848dfbd94109e41ec0c75723277770074deee84' THEN
  RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 IF (length(definition)-length(replace(definition,'^candidate-index-extraction:v2:[0-9a-f]{64}$','')))/length('^candidate-index-extraction:v2:[0-9a-f]{64}$')<>1 THEN RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 definition:=replace(definition,'^candidate-index-extraction:v2:[0-9a-f]{64}$','^candidate-index-extraction:v[23]:[0-9a-f]{64}$');
 IF (length(definition)-length(replace(definition,'    FOR evidence IN SELECT value FROM jsonb_array_elements(p_evidence) LOOP','')))/length('    FOR evidence IN SELECT value FROM jsonb_array_elements(p_evidence) LOOP')<>1 THEN RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 definition:=replace(definition,'    FOR evidence IN SELECT value FROM jsonb_array_elements(p_evidence) LOOP','    IF g.policy->>''extraction_adapter_version'' ~ ''^candidate-index-extraction:v3:[0-9a-f]{64}$''
      AND public.candidate_index_history_valid(p_namespace->''employment_history'',input_text) IS DISTINCT FROM true THEN
      RAISE EXCEPTION USING ERRCODE=''22023'',MESSAGE=''candidate_index_extraction_incomplete''; END IF;
    FOR evidence IN SELECT value FROM jsonb_array_elements(p_evidence) LOOP');
 EXECUTE definition;
END $forward$;

DO $forward$ DECLARE definition text; source text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.candidate_index_fail(uuid,uuid,bigint,text,integer)'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>'7124e3c92c6b0ffdd6456342abce1668d77eacca737c75165d09f9b0a9f14020' THEN
  RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 IF (length(definition)-length(replace(definition,'^candidate-index-extraction:v2:[0-9a-f]{64}$','')))/length('^candidate-index-extraction:v2:[0-9a-f]{64}$')<>1 THEN RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 definition:=replace(definition,'^candidate-index-extraction:v2:[0-9a-f]{64}$','^candidate-index-extraction:v[23]:[0-9a-f]{64}$');
 EXECUTE definition;
END $forward$;

DO $forward$ DECLARE definition text; source text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.candidate_index_plan_recovery(text,jsonb,text)'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>'b9fe193806a79af1d5df922174f672ebde1260efbed20803ab9948ae1e3a561b' THEN
  RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 IF (length(definition)-length(replace(definition,'^candidate-index-extraction:v2:[0-9a-f]{64}$','')))/length('^candidate-index-extraction:v2:[0-9a-f]{64}$')<>1 THEN RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 definition:=replace(definition,'^candidate-index-extraction:v2:[0-9a-f]{64}$','^candidate-index-extraction:v[23]:[0-9a-f]{64}$');
 EXECUTE definition;
END $forward$;

DO $forward$ DECLARE definition text; source text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.candidate_index_read_sourcing(vector,text,text,integer,integer)'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>'9b0e2d2807d2b1c9fa848bdc81b912f935a807c952a1b473319ec5c7c304207b' THEN
  RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 IF (length(definition)-length(replace(definition,'c.global_candidate_id linked_global_id,e.namespace,','')))/length('c.global_candidate_id linked_global_id,e.namespace,')<>1 THEN RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 definition:=replace(definition,'c.global_candidate_id linked_global_id,e.namespace,','c.global_candidate_id linked_global_id,e.namespace,g.policy,');
 IF (length(definition)-length(replace(definition,'''currentTitle'',item.namespace->''current_title''','')))/length('''currentTitle'',item.namespace->''current_title''')<>1 THEN RAISE EXCEPTION 'RESUME_HISTORY_ROUTINE_DRIFT'; END IF;
 definition:=replace(definition,'''currentTitle'',item.namespace->''current_title''','''employment'',CASE WHEN item.policy->>''extraction_adapter_version'' ~ ''^candidate-index-extraction:v3:[0-9a-f]{64}$''
          THEN (SELECT coalesce(jsonb_agg(jsonb_strip_nulls(jsonb_build_object(''title'',r->''title'',
            ''employmentType'',r->''employment_type'',''start'',r->''start'',''end'',r->''end'',''ongoing'',r->''ongoing''))),''[]''::jsonb)
            FROM jsonb_array_elements(item.namespace->''employment_history'') r) ELSE ''[]''::jsonb END,
        ''currentTitle'',item.namespace->''current_title''');
 EXECUTE definition;
END $forward$;

