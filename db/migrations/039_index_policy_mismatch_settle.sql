-- A worker whose policy differs from a job's recorded policy settles that job instead of
-- releasing it (23 Oct 2026 rollout: an old-policy embed job was re-claimed every lease
-- period, attempts 0, holding its organisation's single per-scope slot).
-- Only after the existing lease fence holds (same holder, unexpired, in-progress state,
-- eligible and current generation). The job becomes needs_review/policy_mismatch, the
-- lease is released, attempts and dispatch reservation are unchanged, nothing is marked
-- indexed. The existing recovery tool can then re-run it under the current policy.
-- No data change, no new table, no privilege change.

DO $forward$ DECLARE definition text; source text; old_check text; new_check text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.candidate_index_claim(text,integer,jsonb,uuid,uuid,bigint,text)'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>'135261e0f843808da43a1249740097c89981b113c29e255cf7ba902d75804cd7' THEN
  RAISE EXCEPTION 'INDEX_CLAIM_ROUTINE_DRIFT'; END IF;
 old_check:=$old$          OR (allowed->>'generation_id')::uuid IS DISTINCT FROM j.generation_id OR g.policy_sha256<>p_policy_sha256 THEN
            PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN; END IF;$old$;
 new_check:=$new$          OR (allowed->>'generation_id')::uuid IS DISTINCT FROM j.generation_id THEN
            PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN; END IF;
        IF g.policy_sha256<>p_policy_sha256 THEN
            UPDATE public.candidate_index_jobs SET state='needs_review',error_code='policy_mismatch',
              lease_token=NULL,lease_expires_at=NULL,updated_at=clock_timestamp() WHERE job_id=p_job_id;
            PERFORM set_config('app.current_tenant_id',coalesce(original_scope,''),true); RETURN; END IF;$new$;
 IF (length(definition)-length(replace(definition,old_check,'')))/length(old_check)<>1 THEN
  RAISE EXCEPTION 'INDEX_CLAIM_ROUTINE_DRIFT'; END IF;
 EXECUTE replace(definition,old_check,new_check);
END $forward$;
