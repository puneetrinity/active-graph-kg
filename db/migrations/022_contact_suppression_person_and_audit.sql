-- Contact suppression: person-level complaint scope and append-only receipts.
--
-- A complaint is person-terminal and platform-wide because every organization
-- mails under one sender identity. The correlated Signal candidate id is
-- resolved inside the reporting tenant before this table is written; an email
-- address is never used to infer a person. Hard bounce remains address-only.

CREATE TABLE IF NOT EXISTS contact_person_suppressions (
    global_candidate_id UUID NOT NULL,
    reason TEXT NOT NULL,
    first_observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    provider_event_id TEXT NOT NULL,
    CONSTRAINT contact_person_suppressions_pkey
        PRIMARY KEY (global_candidate_id),
    CONSTRAINT contact_person_suppressions_global_candidate_fkey
        FOREIGN KEY (global_candidate_id) REFERENCES global_candidates(id)
        ON DELETE RESTRICT,
    CONSTRAINT contact_person_suppression_reason_check
        CHECK (reason = 'complaint'),
    CONSTRAINT contact_person_suppression_provider_event_hash
        CHECK (provider_event_id ~ '^[0-9a-f]{64}$')
);

-- Migration 021 allowed opaque provider identifiers. Preserve their replay
-- distinction while moving the shared compliance table to the hash-only
-- boundary used by Flow's Brevo event derivation.
UPDATE contact_suppression_tombstones
SET provider_event_id = encode(digest(provider_event_id, 'sha256'), 'hex')
WHERE provider_event_id IS NOT NULL
  AND provider_event_id !~ '^[0-9a-f]{64}$';

ALTER TABLE contact_suppression_tombstones
    DROP CONSTRAINT IF EXISTS contact_suppression_provider_event_hash;
ALTER TABLE contact_suppression_tombstones
    ADD CONSTRAINT contact_suppression_provider_event_hash
    CHECK (provider_event_id IS NULL OR provider_event_id ~ '^[0-9a-f]{64}$');

-- Migration 021 accepted complaint writes through the generic evidence API.
-- Recover those rows as person-terminal suppressions before the endpoint is
-- narrowed. Prefer the source evidence link over the tombstone candidate: the
-- old upsert kept the first candidate on a shared email while replacing the
-- source_evidence_id with the most recent complaint event.
-- The table is FORCE RLS. Migrations run as its owner, so temporarily remove
-- FORCE inside this transaction to inspect all tenants; runtime RLS remains
-- enabled, and any failure rolls this change back with the migration.
ALTER TABLE candidate_contact_evidence NO FORCE ROW LEVEL SECURITY;

UPDATE contact_suppression_tombstones AS tombstone
SET global_candidate_id = evidence.global_candidate_id
FROM candidate_contact_evidence AS evidence
WHERE tombstone.reason = 'complaint'
  AND tombstone.source_evidence_id = evidence.id
  AND tombstone.global_candidate_id IS DISTINCT FROM evidence.global_candidate_id;

DO $$
DECLARE
    unresolved_count BIGINT;
BEGIN
    SELECT count(*) INTO unresolved_count
    FROM contact_suppression_tombstones
    WHERE reason = 'complaint'
      AND global_candidate_id IS NULL;

    IF unresolved_count > 0 THEN
        RAISE EXCEPTION
            'cannot migrate % legacy complaint tombstone(s) without a candidate identity',
            unresolved_count;
    END IF;
END;
$$;

-- A later 021 hard-bounce write could downgrade the one address tombstone
-- from complaint to hard_bounce. Include surviving complaint evidence as a
-- second recovery source. Null legacy provider ids receive a stable synthetic
-- SHA-256 audit token; no receipt is fabricated because the original issuer,
-- actor, and request identity cannot be reconstructed truthfully.
WITH legacy_complaints AS (
    SELECT
        tombstone.global_candidate_id,
        tombstone.first_observed_at,
        tombstone.last_observed_at,
        COALESCE(
            tombstone.provider_event_id,
            encode(
                digest(
                    'legacy-021-complaint|' || tombstone.email_hash || '|'
                    || tombstone.global_candidate_id::text,
                    'sha256'
                ),
                'hex'
            )
        ) AS provider_event_id
    FROM contact_suppression_tombstones AS tombstone
    WHERE tombstone.reason = 'complaint'

    UNION ALL

    SELECT
        evidence.global_candidate_id,
        evidence.created_at AS first_observed_at,
        COALESCE(
            evidence.suppressed_at,
            evidence.updated_at,
            evidence.observed_at
        ) AS last_observed_at,
        encode(
            digest('legacy-021-evidence-complaint|' || evidence.id::text, 'sha256'),
            'hex'
        ) AS provider_event_id
    FROM candidate_contact_evidence AS evidence
    WHERE evidence.status = 'complaint'
),
aggregated AS (
    SELECT
        global_candidate_id,
        min(first_observed_at) AS first_observed_at,
        max(last_observed_at) AS last_observed_at
    FROM legacy_complaints
    GROUP BY global_candidate_id
),
latest AS (
    SELECT DISTINCT ON (global_candidate_id)
        global_candidate_id,
        provider_event_id
    FROM legacy_complaints
    ORDER BY global_candidate_id, last_observed_at DESC, provider_event_id DESC
)
INSERT INTO contact_person_suppressions
    (global_candidate_id, reason, first_observed_at, last_observed_at, provider_event_id)
SELECT
    aggregated.global_candidate_id,
    'complaint',
    aggregated.first_observed_at,
    aggregated.last_observed_at,
    latest.provider_event_id
FROM aggregated
JOIN latest USING (global_candidate_id)
ON CONFLICT (global_candidate_id) DO UPDATE SET
    first_observed_at = LEAST(
        contact_person_suppressions.first_observed_at,
        EXCLUDED.first_observed_at
    ),
    provider_event_id = CASE
        WHEN EXCLUDED.last_observed_at >= contact_person_suppressions.last_observed_at
        THEN EXCLUDED.provider_event_id
        ELSE contact_person_suppressions.provider_event_id
    END,
    last_observed_at = GREATEST(
        contact_person_suppressions.last_observed_at,
        EXCLUDED.last_observed_at
    );

ALTER TABLE candidate_contact_evidence FORCE ROW LEVEL SECURITY;

-- Receipts are the idempotency claim and the durable authority audit. The Flow
-- service reserves a receipt before changing either suppression table. Exact
-- provider retries are no-ops; a changed replay is rejected by the API.
CREATE TABLE IF NOT EXISTS contact_suppression_receipts (
    id BIGSERIAL NOT NULL,
    email_hash TEXT NOT NULL,
    global_candidate_id UUID,
    signal_candidate_id TEXT,
    reason TEXT NOT NULL,
    scope TEXT NOT NULL,
    evidence_present BOOLEAN NOT NULL,
    tenant_id TEXT NOT NULL,
    issuer TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    actor_type TEXT NOT NULL,
    provider_event_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT contact_suppression_receipts_pkey PRIMARY KEY (id),
    CONSTRAINT contact_suppression_receipt_email_hash_check
        CHECK (email_hash ~ '^[0-9a-f]{64}$'),
    CONSTRAINT contact_suppression_receipt_signal_candidate_nonblank
        CHECK (signal_candidate_id IS NULL OR btrim(signal_candidate_id) <> ''),
    CONSTRAINT contact_suppression_receipt_tenant_nonblank
        CHECK (btrim(tenant_id) <> '' AND tenant_id <> '__quarantine__'),
    CONSTRAINT contact_suppression_receipt_provider_event_hash
        CHECK (provider_event_id ~ '^[0-9a-f]{64}$'),
    -- Shape, not identity. Pinning the literal issuer/actor here would make
    -- rotating or renaming a service credential a schema migration — worst case
    -- during a compromise, when rotation has to happen in minutes. WHICH
    -- identities are trusted is a config-driven allowlist enforced in the API
    -- (UNOWNED_SUPPRESSION_ISSUERS); the schema only guarantees a receipt can
    -- never be written without recording one.
    CONSTRAINT contact_suppression_receipt_authority_check
        CHECK (
            btrim(issuer) <> ''
            AND btrim(actor_id) <> ''
            AND actor_type = 'service'
        ),
    CONSTRAINT contact_suppression_receipt_scope_reason_check
        CHECK (
            (reason = 'hard_bounce' AND scope = 'address')
            OR
            (reason = 'complaint' AND scope = 'person'
                AND global_candidate_id IS NOT NULL
                AND signal_candidate_id IS NOT NULL)
        ),
    CONSTRAINT contact_suppression_receipts_provider_event_unique
        UNIQUE (issuer, provider_event_id)
);

CREATE INDEX IF NOT EXISTS idx_contact_suppression_receipts_email_hash
    ON contact_suppression_receipts (email_hash, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_contact_suppression_receipts_candidate
    ON contact_suppression_receipts (global_candidate_id)
    WHERE global_candidate_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_contact_suppression_receipts_tenant_created
    ON contact_suppression_receipts (tenant_id, created_at DESC);

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

DROP TRIGGER IF EXISTS contact_suppression_receipts_no_truncate
    ON contact_suppression_receipts;
CREATE TRIGGER contact_suppression_receipts_no_truncate
    BEFORE TRUNCATE ON contact_suppression_receipts
    FOR EACH STATEMENT EXECUTE FUNCTION contact_suppression_receipts_append_only();
