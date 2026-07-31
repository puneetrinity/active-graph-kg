-- Contact suppression: person-level scope and an append-only audit trail.
--
-- Two gaps this closes.
--
-- 1. A complaint is PERSON-terminal and platform-wide (every org mails under one
--    sender identity), but suppression was recorded only against an email hash.
--    Another validated address for the same person therefore stayed selectable,
--    including by a different org — which contradicts the locked policy. A hard
--    bounce stays address-only: it tombstones a bad mailbox, not a person.
--
-- 2. Suppression can now be requested for an address the caller owns no evidence
--    for, so every suppression needs a durable, append-only receipt recording who
--    asked, under which issuer, and on which provider event.

CREATE TABLE IF NOT EXISTS contact_person_suppressions (
    global_candidate_id UUID PRIMARY KEY
        REFERENCES global_candidates(id) ON DELETE CASCADE,
    reason TEXT NOT NULL,
    first_observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider_event_id TEXT,
    -- Only person-terminal signals belong here. Hard bounce is address-only and
    -- must never reach this table.
    CONSTRAINT contact_person_suppression_reason_check
        CHECK (reason IN ('complaint'))
);

-- Append-only receipts. Never updated, never deleted: the trigger below refuses
-- both, so a suppression's provenance cannot be quietly rewritten.
CREATE TABLE IF NOT EXISTS contact_suppression_receipts (
    id BIGSERIAL PRIMARY KEY,
    email_hash TEXT NOT NULL,
    global_candidate_id UUID,
    reason TEXT NOT NULL,
    scope TEXT NOT NULL,
    evidence_present BOOLEAN NOT NULL,
    tenant_id TEXT,
    issuer TEXT,
    actor_id TEXT,
    actor_type TEXT,
    provider_event_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT contact_suppression_receipt_scope_check
        CHECK (scope IN ('address', 'person')),
    CONSTRAINT contact_suppression_receipt_reason_check
        CHECK (reason IN ('hard_bounce', 'complaint'))
);

CREATE INDEX IF NOT EXISTS idx_contact_suppression_receipts_email_hash
    ON contact_suppression_receipts (email_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_contact_suppression_receipts_candidate
    ON contact_suppression_receipts (global_candidate_id)
    WHERE global_candidate_id IS NOT NULL;

CREATE OR REPLACE FUNCTION contact_suppression_receipts_append_only()
RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION
        'contact_suppression_receipts is append-only (attempted %)', TG_OP;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS contact_suppression_receipts_no_mutation
    ON contact_suppression_receipts;
CREATE TRIGGER contact_suppression_receipts_no_mutation
    BEFORE UPDATE OR DELETE ON contact_suppression_receipts
    FOR EACH ROW EXECUTE FUNCTION contact_suppression_receipts_append_only();
