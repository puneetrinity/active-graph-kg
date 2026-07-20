-- ============================================================
-- 012: Global Candidate Memory Tables
-- ============================================================
-- Adds global_candidates, candidate_provenance, tenant_candidate_access,
-- and feedback_events tables for the cross-tenant candidate memory system.
-- Idempotent: safe to run multiple times.

-- Canonical candidate record. One per real person.
CREATE TABLE IF NOT EXISTS global_candidates (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Identity anchors (nullable, unique when present)
  linkedin_id TEXT,
  linkedin_url TEXT,
  github_id TEXT,
  email_hash TEXT,

  -- Best-known profile fields
  name TEXT,
  headline TEXT,
  location_city TEXT,
  location_country_code TEXT,
  location_confidence DOUBLE PRECISION,
  location_source TEXT,

  -- Normalized fields
  role_family TEXT,
  seniority_band TEXT,
  skills_normalized TEXT[],

  -- Identity resolution
  identity_confidence DOUBLE PRECISION,
  merge_status TEXT NOT NULL DEFAULT 'single',

  -- Embedding for retrieval
  embedding VECTOR(384),
  embedding_status TEXT NOT NULL DEFAULT 'queued',

  -- Timestamps
  first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_evidence_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Identity anchor indexes (nullable unique)
CREATE UNIQUE INDEX IF NOT EXISTS idx_gc_linkedin_id
  ON global_candidates(linkedin_id) WHERE linkedin_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_gc_linkedin_url
  ON global_candidates(linkedin_url) WHERE linkedin_url IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_gc_github_id
  ON global_candidates(github_id) WHERE github_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_gc_email_hash
  ON global_candidates(email_hash) WHERE email_hash IS NOT NULL;

-- Retrieval indexes
CREATE INDEX IF NOT EXISTS idx_gc_role_family ON global_candidates(role_family);
CREATE INDEX IF NOT EXISTS idx_gc_location ON global_candidates(location_country_code, location_city);
CREATE INDEX IF NOT EXISTS idx_gc_skills ON global_candidates USING GIN (skills_normalized);
CREATE INDEX IF NOT EXISTS idx_gc_embedding_status ON global_candidates(embedding_status);
CREATE INDEX IF NOT EXISTS idx_gc_last_evidence ON global_candidates(last_evidence_at);


-- Provenance: where each candidate came from.
CREATE TABLE IF NOT EXISTS candidate_provenance (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  global_candidate_id UUID NOT NULL REFERENCES global_candidates(id),

  source_type TEXT NOT NULL,
  tenant_id TEXT,
  source_detail JSONB NOT NULL DEFAULT '{}',

  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

  UNIQUE(global_candidate_id, source_type, tenant_id)
);

CREATE INDEX IF NOT EXISTS idx_cp_global_candidate ON candidate_provenance(global_candidate_id);
-- Partial unique index for public provenance (tenant_id IS NULL):
-- The table-level UNIQUE(global_candidate_id, source_type, tenant_id) does not collapse NULLs.
CREATE UNIQUE INDEX IF NOT EXISTS idx_cp_global_source_null_tenant
  ON candidate_provenance(global_candidate_id, source_type) WHERE tenant_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_cp_source_type ON candidate_provenance(source_type);
CREATE INDEX IF NOT EXISTS idx_cp_tenant ON candidate_provenance(tenant_id);


-- Tenant-specific access rows (private + consented-shared only).
-- Public web discoveries do NOT get rows here.
CREATE TABLE IF NOT EXISTS tenant_candidate_access (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id TEXT NOT NULL,
  global_candidate_id UUID NOT NULL REFERENCES global_candidates(id),

  visibility TEXT NOT NULL,
  consent_state TEXT,
  access_reason TEXT NOT NULL,

  granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  revoked_at TIMESTAMPTZ,

  UNIQUE(tenant_id, global_candidate_id)
);

CREATE INDEX IF NOT EXISTS idx_tca_tenant ON tenant_candidate_access(tenant_id);
CREATE INDEX IF NOT EXISTS idx_tca_global_candidate ON tenant_candidate_access(global_candidate_id);
CREATE INDEX IF NOT EXISTS idx_tca_visibility ON tenant_candidate_access(visibility);


-- Feedback events: READ-OPTIMIZED REPLICA of Vanta's recruiter_feedback_events.
-- Populated by forward-sync from Vanta, never written to directly by recruiters.
CREATE TABLE IF NOT EXISTS feedback_events (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id TEXT NOT NULL,
  job_id TEXT NOT NULL,
  recruiter_id TEXT,
  global_candidate_id UUID REFERENCES global_candidates(id),
  signal_candidate_id TEXT,

  action TEXT NOT NULL,

  rank_at_time INT,
  fit_score_at_time DOUBLE PRECISION,
  source_type_at_time TEXT,
  match_tier_at_time TEXT,
  location_match_at_time TEXT,

  role_family TEXT,
  location_country_code TEXT,
  seniority_band TEXT,

  event_id TEXT NOT NULL UNIQUE,

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_fe_tenant_job ON feedback_events(tenant_id, job_id);
CREATE INDEX IF NOT EXISTS idx_fe_global_candidate ON feedback_events(global_candidate_id);
CREATE INDEX IF NOT EXISTS idx_fe_action ON feedback_events(action);
CREATE INDEX IF NOT EXISTS idx_fe_created ON feedback_events(created_at);
CREATE INDEX IF NOT EXISTS idx_fe_role_location ON feedback_events(role_family, location_country_code);


-- ============================================================
-- RLS Policies
-- ============================================================

-- global_candidates: NO RLS (shared table, access enforced at API layer)

-- candidate_provenance: tenant can see own + public
DO $$ BEGIN
  ALTER TABLE candidate_provenance ENABLE ROW LEVEL SECURITY;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY cp_tenant_read ON candidate_provenance FOR SELECT
    USING (
      tenant_id = current_setting('app.current_tenant_id', true)::text
      OR tenant_id IS NULL
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY cp_tenant_write ON candidate_provenance FOR INSERT
    WITH CHECK (
      tenant_id = current_setting('app.current_tenant_id', true)::text
      OR tenant_id IS NULL
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY cp_tenant_update ON candidate_provenance FOR UPDATE
    USING (
      tenant_id = current_setting('app.current_tenant_id', true)::text
      OR tenant_id IS NULL
    );
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- tenant_candidate_access: strict tenant isolation
DO $$ BEGIN
  ALTER TABLE tenant_candidate_access ENABLE ROW LEVEL SECURITY;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY tca_tenant ON tenant_candidate_access
    USING (tenant_id = current_setting('app.current_tenant_id', true)::text);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- feedback_events: strict tenant isolation
DO $$ BEGIN
  ALTER TABLE feedback_events ENABLE ROW LEVEL SECURITY;
EXCEPTION WHEN OTHERS THEN NULL;
END $$;

DO $$ BEGIN
  CREATE POLICY fe_tenant ON feedback_events
    USING (tenant_id = current_setting('app.current_tenant_id', true)::text);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;
