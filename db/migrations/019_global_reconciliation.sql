-- Migration 019: canonical-model reconciliation — link layer + merge queue
--
-- The tenant-scoped canonical (candidates) and the platform-global canonical
-- (global_candidates) previously shared no id, FK, or link table, so the same
-- human sourced by Signal and stored globally could never be connected.
-- Additionally, cross-anchor identity conflicts (a person matched by
-- linkedin_id on one row and email_hash on another) either crashed the upsert
-- on the partial-unique anchor indexes or silently produced duplicates, and
-- review_required resolutions were returned to the caller but persisted
-- nowhere. Idempotent.

-- 1) Tenant -> global link (the missing layer between the two stores).
ALTER TABLE candidates ADD COLUMN IF NOT EXISTS global_candidate_id UUID
    REFERENCES global_candidates(id) ON DELETE SET NULL;
CREATE INDEX IF NOT EXISTS idx_candidates_global_id
    ON candidates (global_candidate_id) WHERE global_candidate_id IS NOT NULL;

-- 2) Merge queue: identity conflicts become durable work items instead of
--    crashes (anchor_conflict), duplicates (needs_merge), or dropped API
--    responses (review_required). Platform-scope table: no RLS, access is
--    enforced at the API layer, same as global_candidates.
CREATE TABLE IF NOT EXISTS candidate_merge_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    global_candidate_id_a UUID NOT NULL REFERENCES global_candidates(id) ON DELETE CASCADE,
    global_candidate_id_b UUID REFERENCES global_candidates(id) ON DELETE CASCADE,
    tenant_id TEXT,
    reason TEXT NOT NULL CHECK (reason IN ('needs_merge', 'review_required', 'anchor_conflict')),
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved', 'dismissed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at TIMESTAMPTZ
);

-- One open item per (pair, reason); NULL b collapses to a sentinel so the
-- partial-unique holds for single-candidate reviews too.
CREATE UNIQUE INDEX IF NOT EXISTS idx_cmq_open_pair
    ON candidate_merge_queue (
        global_candidate_id_a,
        COALESCE(global_candidate_id_b, '00000000-0000-0000-0000-000000000000'::uuid),
        reason
    )
    WHERE status = 'open';
CREATE INDEX IF NOT EXISTS idx_cmq_status ON candidate_merge_queue (status, created_at);
