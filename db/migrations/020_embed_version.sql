-- Migration 020: embed_version on global_candidates
--
-- No version/model tracking existed for pool embeddings: a text-builder change
-- silently left every existing vector stale with no way to find them. The
-- Stage-3 offline gate made the cost concrete — thin embed text (headline-only
-- for sourced rows) ordered an 815-row segment so poorly that half the
-- fit-top-100 fell below vector rank 500. embed_version lets the producer
-- sweep re-embed rows built with an older text version (blanket re-embed =
-- bump the constant, the sweep drains it). Idempotent.

ALTER TABLE global_candidates
    ADD COLUMN IF NOT EXISTS embed_version INT NOT NULL DEFAULT 0;

-- The sweep looks for ready/skipped rows built with an older text version.
CREATE INDEX IF NOT EXISTS idx_global_candidates_embed_version
    ON global_candidates (embed_version)
    WHERE embedding_status IN ('ready', 'skipped_empty');
