-- Migration 013: VantaHire provenance structured columns
--
-- Adds queryable provenance columns to candidate_source_records for VantaHire-
-- originated records. The JSONB payload already carries this data verbatim; these
-- structured columns exist so recruiter/org-scoped Talent Search queries use
-- B-tree indexes rather than JSONB path operators.
--
-- Canonical candidates remain globally shared. Org/job/recruiter ownership is NOT
-- modeled on candidates — it lives exclusively on the source records that produced
-- them.

ALTER TABLE candidate_source_records
    ADD COLUMN IF NOT EXISTS org_id               TEXT,
    ADD COLUMN IF NOT EXISTS job_id               TEXT,
    ADD COLUMN IF NOT EXISTS effective_recruiter_id TEXT,
    ADD COLUMN IF NOT EXISTS created_by_user_id   TEXT,
    ADD COLUMN IF NOT EXISTS resume_source        TEXT;

-- Partial B-tree indexes — only vantahire rows carry these values, so the partial
-- predicate keeps the indexes small and the planner can use them on equality scans.
CREATE INDEX IF NOT EXISTS idx_csr_vantahire_org_id
    ON candidate_source_records (tenant_id, org_id)
    WHERE source = 'vantahire' AND org_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_csr_vantahire_recruiter
    ON candidate_source_records (tenant_id, effective_recruiter_id)
    WHERE source = 'vantahire' AND effective_recruiter_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_csr_vantahire_uploader
    ON candidate_source_records (tenant_id, created_by_user_id)
    WHERE source = 'vantahire' AND created_by_user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_csr_vantahire_job_id
    ON candidate_source_records (tenant_id, job_id)
    WHERE source = 'vantahire' AND job_id IS NOT NULL;
