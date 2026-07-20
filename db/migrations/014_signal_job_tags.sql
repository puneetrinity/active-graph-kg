-- Migration 014: Signal job_tags structured column
--
-- Adds a TEXT[] column to candidate_source_records for Signal-sourced records.
-- Tags describe the job context in which a candidate was surfaced by Signal.
-- The full tags array is also preserved verbatim in the JSONB payload; this
-- structured column exists so tag-based candidate search uses a GIN index
-- rather than scanning JSONB payloads.
--
-- Canonical candidates remain globally shared. Tags are scoped to the Signal
-- source record, not to the candidate.

ALTER TABLE candidate_source_records
    ADD COLUMN IF NOT EXISTS job_tags TEXT[] NOT NULL DEFAULT '{}';

-- Partial GIN index — only signal rows carry job_tags, so the partial predicate
-- keeps the index small and supports efficient array overlap (&&) and containment (@>) queries.
CREATE INDEX IF NOT EXISTS idx_csr_signal_job_tags
    ON candidate_source_records USING GIN (job_tags)
    WHERE source = 'signal';
