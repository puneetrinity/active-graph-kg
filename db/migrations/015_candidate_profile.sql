-- Migration 015: Structured profile for ranking parity with Crustdata
--
-- Adds a profile JSONB column to candidates that stores the same structured
-- data Signal's ranker expects from Crustdata. ActiveGraph is just another
-- source — the ranker is source-agnostic.

ALTER TABLE candidates
    ADD COLUMN IF NOT EXISTS profile JSONB NOT NULL DEFAULT '{}';

-- Queryable profile fields for pre-filtering during search.
-- These are extracted from the profile JSONB on write for index performance.
ALTER TABLE candidates
    ADD COLUMN IF NOT EXISTS headline TEXT,
    ADD COLUMN IF NOT EXISTS location_raw TEXT,
    ADD COLUMN IF NOT EXISTS skills TEXT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS seniority_level TEXT,
    ADD COLUMN IF NOT EXISTS linkedin_url TEXT,
    ADD COLUMN IF NOT EXISTS linkedin_id TEXT,
    ADD COLUMN IF NOT EXISTS profile_picture_url TEXT;

-- Pre-filter indexes for home-pool search
CREATE INDEX IF NOT EXISTS idx_candidates_skills_gin
    ON candidates USING GIN (skills);

CREATE INDEX IF NOT EXISTS idx_candidates_location
    ON candidates (tenant_id, location_raw)
    WHERE location_raw IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_candidates_seniority
    ON candidates (tenant_id, seniority_level)
    WHERE seniority_level IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_candidates_linkedin_id
    ON candidates (tenant_id, linkedin_id)
    WHERE linkedin_id IS NOT NULL;
