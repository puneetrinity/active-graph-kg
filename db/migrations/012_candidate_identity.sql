-- Migration 012: Source-independent candidate identity
--
-- Introduces a canonical candidate record owned by ActiveKG, decoupled from any
-- upstream source (VantaHire, Signal, etc.). A candidate may carry many
-- identifiers (emails, phones, LinkedIn/GitHub/Medium URLs, upstream IDs) and
-- many source records that preserve the original payloads and pointers.
--
-- Scope: canonical candidates are explicitly shared/global across sources. They
-- still respect tenant_id for multi-tenant isolation, but within a tenant one
-- canonical candidate is the merge target for all sources.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Canonical candidate. ActiveKG owns candidate_id.
CREATE TABLE IF NOT EXISTS candidates (
  candidate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id TEXT,
  -- 'shared' marks the canonical/global scope choice. Kept as a column so
  -- future per-source scoping (if ever needed) is an additive change.
  scope TEXT NOT NULL DEFAULT 'shared',
  display_name TEXT,
  primary_email TEXT,
  primary_phone TEXT,
  props JSONB NOT NULL DEFAULT '{}',
  metadata JSONB NOT NULL DEFAULT '{}',
  node_id UUID,  -- optional link to a nodes row when projected into the graph
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT candidates_scope_check CHECK (scope IN ('shared'))
);

CREATE INDEX IF NOT EXISTS idx_candidates_tenant ON candidates(tenant_id);
CREATE INDEX IF NOT EXISTS idx_candidates_node_id ON candidates(node_id) WHERE node_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_candidates_primary_email ON candidates(tenant_id, primary_email) WHERE primary_email IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_candidates_props ON candidates USING GIN (props);


-- Identifiers attached to a canonical candidate. Values are stored normalized
-- so that uniqueness lookups (email, phone, linkedin url, etc.) are exact.
CREATE TABLE IF NOT EXISTS candidate_identifiers (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  candidate_id UUID NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
  tenant_id TEXT,
  identifier_type TEXT NOT NULL,
  value_normalized TEXT NOT NULL,
  value_raw TEXT,
  source TEXT,       -- which upstream first asserted this identifier
  confidence REAL,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CONSTRAINT candidate_identifiers_type_check CHECK (
    identifier_type IN (
      'signal_candidate_id',
      'vantahire_application_id',
      'vantahire_resume_id',
      'linkedin_url',
      'github_url',
      'medium_url',
      'email',
      'phone',
      'website_url',
      'twitter_url',
      'stackoverflow_url',
      'portfolio_url',
      'other'
    )
  )
);

-- One (tenant, identifier_type, value_normalized) must resolve to exactly one
-- candidate. This is the core merge key for cross-source identity.
CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_identifiers_unique
  ON candidate_identifiers (tenant_id, identifier_type, value_normalized);

CREATE INDEX IF NOT EXISTS idx_candidate_identifiers_candidate
  ON candidate_identifiers (candidate_id);

CREATE INDEX IF NOT EXISTS idx_candidate_identifiers_lookup
  ON candidate_identifiers (identifier_type, value_normalized);


-- Source records preserve full provenance from upstream systems. Each time a
-- candidate appears in VantaHire, Signal, or another source we insert or
-- update a row here so the canonical candidate keeps history of every payload.
CREATE TABLE IF NOT EXISTS candidate_source_records (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  candidate_id UUID NOT NULL REFERENCES candidates(candidate_id) ON DELETE CASCADE,
  tenant_id TEXT,
  source TEXT NOT NULL,              -- 'vantahire' | 'signal' | other
  source_record_type TEXT NOT NULL,  -- e.g. 'application', 'resume', 'profile'
  source_record_id TEXT NOT NULL,    -- upstream primary key for this record
  source_url TEXT,
  payload JSONB NOT NULL DEFAULT '{}',
  payload_ref TEXT,                  -- object storage pointer for large payloads
  fetched_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_candidate_source_records_unique
  ON candidate_source_records (tenant_id, source, source_record_type, source_record_id);

CREATE INDEX IF NOT EXISTS idx_candidate_source_records_candidate
  ON candidate_source_records (candidate_id, source, source_record_type);

CREATE INDEX IF NOT EXISTS idx_candidate_source_records_payload
  ON candidate_source_records USING GIN (payload);
