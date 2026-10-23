-- Recovery also covers resumes whose extraction succeeded but whose embedding was refused
-- (v3 chunk_overflow, 23 Oct 2026). No data change, no new table, no policy activation.
-- Recovery still re-extracts under the target policy (budget unchanged: <=2 calls per new generation).
-- The superseded generation's extract job is retired so it no longer counts as in-flight work.

DO $forward$ DECLARE definition text; source text; old_check text; new_check text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.candidate_index_plan_recovery(text,jsonb,text)'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>'b6f4ba9e1a87954c9b30f1148d7674d421297fbe7003cb5c33e12e22b1d82109' THEN
  RAISE EXCEPTION 'RESUME_RECOVERY_ROUTINE_DRIFT'; END IF;
 old_check:=$old$IF NOT FOUND OR j.state NOT IN ('failed','needs_review') OR j.lease_token IS NOT NULL$old$;
 new_check:=$new$IF NOT FOUND OR (j.state NOT IN ('failed','needs_review') AND NOT (j.state='pending_embedding'
   AND EXISTS(SELECT 1 FROM public.candidate_index_jobs e WHERE e.generation_id=g.generation_id AND e.stage='embed'
     AND e.state IN ('failed','needs_review') AND e.lease_token IS NULL))) OR j.lease_token IS NOT NULL$new$;
 IF (length(definition)-length(replace(definition,old_check,'')))/length(old_check)<>1 THEN
  RAISE EXCEPTION 'RESUME_RECOVERY_ROUTINE_DRIFT'; END IF;
 EXECUTE replace(definition,old_check,new_check);
END $forward$;

DO $forward$ DECLARE definition text; source text; anchor text; BEGIN
 SELECT pg_get_functiondef(p.oid),p.prosrc INTO STRICT definition,source FROM pg_proc p
  WHERE p.oid='public.candidate_index_apply_recovery(text,jsonb,text,integer)'::regprocedure;
 IF encode(sha256(convert_to(source,'UTF8')),'hex')<>'fcf1a11a61ea107db214a022dc2f77847991c82fa8323e2c3b4eed0e9e5e3203' THEN
  RAISE EXCEPTION 'RESUME_RECOVERY_ROUTINE_DRIFT'; END IF;
 anchor:=$a$  members:=members||jsonb_build_array(jsonb_build_object('source_id',s.source_id,'generation_id',result->>'generation_id'));$a$;
 IF (length(definition)-length(replace(definition,anchor,'')))/length(anchor)<>1 THEN
  RAISE EXCEPTION 'RESUME_RECOVERY_ROUTINE_DRIFT'; END IF;
 EXECUTE replace(definition,anchor,$r$  UPDATE public.candidate_index_jobs SET state='superseded',error_code='superseded',updated_at=clock_timestamp()
   WHERE generation_id=(pair->>'generation_id')::uuid AND stage='extract' AND state='pending_embedding' AND lease_token IS NULL;
$r$||anchor);
END $forward$;
