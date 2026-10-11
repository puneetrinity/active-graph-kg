-- V-RC5: tolerate at most five seconds of application/database clock skew.
-- Only application timestamps; provider age/expiry, dispatch and paid-retry rules unchanged.
-- Forward replacement preserves signatures, owners and runtime EXECUTE grants.
CREATE OR REPLACE FUNCTION public.memory_rc_finish(p_service text,p_attempt uuid,p_command jsonb)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER
SET search_path=pg_catalog,public SET timezone='UTC'
AS $rc_finish$
DECLARE
    candidate uuid; run_id uuid; receipt uuid; obs jsonb; old_row public.rc_skill_observations%ROWTYPE;
    urls text[]; canonical text; expected_hash text; upstream timestamptz; observed timestamptz;
    valid timestamptz; value jsonb; label text; cp integer; i integer; utf16_length integer;
BEGIN
    IF p_service IS DISTINCT FROM 'signal-service' THEN RETURN jsonb_build_object('error','denied'); END IF;
    IF jsonb_typeof(p_command) IS DISTINCT FROM 'object'
       OR p_command->'version' IS DISTINCT FROM '1'::jsonb
       OR octet_length(p_command::text)>65536
       OR (SELECT array_agg(k ORDER BY k COLLATE "C") FROM jsonb_object_keys(p_command) k)
          IS DISTINCT FROM ARRAY['globalCandidateId','observation','originAttempt','originRun','version']::text[]
       OR p_command->>'globalCandidateId' !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
       OR p_command->>'originRun' !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
       OR p_command->>'originAttempt' !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
       OR jsonb_typeof(p_command->'globalCandidateId') IS DISTINCT FROM 'string'
       OR jsonb_typeof(p_command->'originRun') IS DISTINCT FROM 'string'
       OR jsonb_typeof(p_command->'originAttempt') IS DISTINCT FROM 'string'
    THEN RAISE EXCEPTION 'RC_INVALID_COMMAND'; END IF;
    candidate := (p_command->>'globalCandidateId')::uuid;
    run_id := (p_command->>'originRun')::uuid;
    IF p_attempt IS NULL OR p_attempt IS DISTINCT FROM (p_command->>'originAttempt')::uuid THEN
        RAISE EXCEPTION 'RC_INVALID_COMMAND';
    END IF;
    obs := p_command->'observation';
    IF jsonb_typeof(obs) IS DISTINCT FROM 'object' OR obs->'version' IS DISTINCT FROM '1'::jsonb
       OR (SELECT array_agg(k ORDER BY k COLLATE "C") FROM jsonb_object_keys(obs) k)
          IS DISTINCT FROM ARRAY['contentHash','identityHash','observedAt','outcome','skills','timestampSource','upstreamAt','validUntil','version']::text[]
       OR jsonb_typeof(obs->'skills') IS DISTINCT FROM 'array'
       OR coalesce(obs->>'identityHash','') !~ '^[0-9a-f]{64}$'
       OR coalesce(obs->>'contentHash','') !~ '^[0-9a-f]{64}$'
       OR obs->'timestampSource' IS DISTINCT FROM '"check_last_update"'::jsonb
       OR coalesce(obs->>'outcome','') NOT IN ('success','no_skills','not_found','unusable_age')
    THEN RAISE EXCEPTION 'RC_INVALID_OBSERVATION'; END IF;
    IF jsonb_array_length(obs->'skills')>500 THEN RAISE EXCEPTION 'RC_INVALID_OBSERVATION'; END IF;
    FOR value IN SELECT v FROM jsonb_array_elements(obs->'skills') v LOOP
        IF jsonb_typeof(value)<>'string' THEN RAISE EXCEPTION 'RC_INVALID_OBSERVATION'; END IF;
        label := value#>>'{}'; utf16_length := 0;
        IF label !~ '[^[:space:]]' THEN RAISE EXCEPTION 'RC_INVALID_OBSERVATION'; END IF;
        FOR i IN 1..length(label) LOOP
            cp := ascii(substr(label,i,1));
            utf16_length := utf16_length + CASE WHEN cp>65535 THEN 2 ELSE 1 END;
            -- Unicode control/format/surrogate ranges; no skill-alias normalization.
            IF cp<32 OR cp BETWEEN 127 AND 159 OR cp IN (173,1564,1757,1807,2274,6158,65279,69821,69837,917505)
               OR cp BETWEEN 1536 AND 1541 OR cp BETWEEN 2192 AND 2193
               OR cp BETWEEN 8203 AND 8207 OR cp BETWEEN 8234 AND 8238
               OR cp BETWEEN 8288 AND 8292 OR cp BETWEEN 8294 AND 8303
               OR cp BETWEEN 55296 AND 57343 OR cp BETWEEN 65529 AND 65531
               OR cp BETWEEN 78896 AND 78933 OR cp BETWEEN 113824 AND 113827
               OR cp BETWEEN 119155 AND 119162 OR cp BETWEEN 917536 AND 917631
            THEN RAISE EXCEPTION 'RC_INVALID_OBSERVATION'; END IF;
        END LOOP;
        IF utf16_length NOT BETWEEN 1 AND 500 THEN RAISE EXCEPTION 'RC_INVALID_OBSERVATION'; END IF;
    END LOOP;
    FOR label IN SELECT unnest(ARRAY['upstreamAt','observedAt','validUntil']) LOOP
        IF obs->label='null'::jsonb AND label<>'observedAt' THEN CONTINUE; END IF;
        IF jsonb_typeof(obs->label) IS DISTINCT FROM 'string'
          OR (obs->>label) !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$'
          OR to_char((obs->>label)::timestamptz,'YYYY-MM-DD"T"HH24:MI:SS.US"Z"') IS DISTINCT FROM obs->>label
        THEN RAISE EXCEPTION 'RC_INVALID_OBSERVATION'; END IF;
    END LOOP;
    upstream := (obs->>'upstreamAt')::timestamptz;
    observed := (obs->>'observedAt')::timestamptz;
    valid := (obs->>'validUntil')::timestamptz;
    IF observed>clock_timestamp()+interval '5 seconds' OR (obs->>'outcome'='success' AND
       (upstream IS NULL OR valid IS NULL OR upstream>observed OR valid<observed
        OR valid<>upstream+interval '26280 hours' OR jsonb_array_length(obs->'skills')=0))
       OR (obs->>'outcome'<>'success' AND (obs->'skills'<>'[]'::jsonb OR valid IS NOT NULL))
    THEN RAISE EXCEPTION 'RC_INVALID_OBSERVATION'; END IF;
    -- Canonical flat wire JSON (sorted keys, no spaces), identical to Discover/Python.
    SELECT '{'||string_agg(to_jsonb(k)::text||':'||CASE WHEN k='skills' THEN
        '['||(SELECT coalesce(string_agg(v::text,',' ORDER BY ord),'')
          FROM jsonb_array_elements(obs->'skills') WITH ORDINALITY a(v,ord))||']'
        ELSE v::text END,',' ORDER BY k COLLATE "C")||'}' INTO canonical
      FROM jsonb_each(obs-'contentHash') e(k,v);
    expected_hash := encode(public.digest(canonical,'sha256'),'hex');
    IF expected_hash IS DISTINCT FROM obs->>'contentHash' THEN RAISE EXCEPTION 'RC_INVALID_HASH'; END IF;
    PERFORM pg_advisory_xact_lock(hashtextextended('candidate-privacy-global:'||candidate::text,0));
    IF public.candidate_privacy_global_decision(candidate) IS DISTINCT FROM 'allow' THEN
        RETURN jsonb_build_object('error','restricted');
    END IF;
    SELECT array_agg(DISTINCT canonical_linkedin_url) INTO urls FROM public.global_candidate_source_identities
      WHERE global_candidate_id=candidate AND provider_namespace='crustdata' AND record_type='person'
      AND verified_issuer='signal' AND verified_actor_id='signal-service';
    IF cardinality(urls) IS DISTINCT FROM 1 THEN RETURN jsonb_build_object('error','unavailable'); END IF;
    IF encode(public.digest(urls[1],'sha256'),'hex') IS DISTINCT FROM obs->>'identityHash' THEN
        RETURN jsonb_build_object('error','conflict');
    END IF;
    -- Bind run+attempt in the immutable receipt ID without an extra stored job column.
    receipt := substr(encode(public.digest(p_service||':'||run_id::text||':'||p_attempt::text,'sha256'),'hex'),1,32)::uuid;
    PERFORM pg_advisory_xact_lock(hashtextextended('memory-rc-attempt:'||p_service||':'||p_attempt::text,0));
    SELECT * INTO old_row FROM public.rc_skill_observations WHERE origin_service=p_service AND origin_attempt=p_attempt;
    IF FOUND THEN
        IF old_row.id<>receipt OR old_row.global_candidate_id<>candidate OR old_row.content_hash<>expected_hash THEN
            RETURN jsonb_build_object('error','conflict');
        END IF;
        RETURN jsonb_build_object('version',1,'status','replayed','receiptId',receipt,'contentHash',expected_hash);
    END IF;
    INSERT INTO public.rc_skill_observations(id,global_candidate_id,identity_hash,outcome,skills,
      upstream_at,observed_at,valid_until,schema_version,content_hash,origin_service,origin_attempt)
      VALUES(receipt,candidate,obs->>'identityHash',obs->>'outcome',obs->'skills',upstream,observed,
        valid,'rc-skills-v1',expected_hash,p_service,p_attempt);
    RETURN jsonb_build_object('version',1,'status','stored','receiptId',receipt,'contentHash',expected_hash);
END;
$rc_finish$;
REVOKE ALL ON FUNCTION public.memory_rc_read(text,jsonb) FROM PUBLIC;
