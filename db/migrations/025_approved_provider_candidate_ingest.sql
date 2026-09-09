-- Migration 025: approved-provider sourced-candidate identity and evidence.
--
-- This authority is platform-global and deliberately provider-neutral. It
-- retains global_candidates.id as the canonical person id, records immutable
-- exact provider bindings, and preserves one normalized observation plus one
-- idempotent receipt for every accepted acquisition candidate. It contains no
-- tenant, organization, job, rank, contact, application, outreach, or resume
-- data and performs no historical backfill.

CREATE TABLE global_candidate_source_identities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider_namespace TEXT NOT NULL,
    record_type TEXT NOT NULL,
    adapter_family TEXT NOT NULL,
    adapter_version INTEGER NOT NULL,
    provider_record_id TEXT NOT NULL,
    canonical_linkedin_url TEXT NOT NULL,
    global_candidate_id UUID NOT NULL
        REFERENCES global_candidates(id) ON DELETE RESTRICT,
    verified_issuer TEXT NOT NULL,
    verified_actor_id TEXT NOT NULL,
    first_observed_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT global_candidate_source_identities_provider_v1 CHECK (
        provider_namespace = 'crustdata'
        AND record_type = 'person'
        AND adapter_family = 'crustdata_person'
        AND adapter_version = 1
    ),
    CONSTRAINT global_candidate_source_identities_provider_id CHECK (
        provider_record_id ~ '^[1-9][0-9]{0,18}$'
    ),
    CONSTRAINT global_candidate_source_identities_linkedin CHECK (
        canonical_linkedin_url ~ '^https://linkedin\.com/in/[a-z0-9][a-z0-9._-]{0,199}$'
    ),
    CONSTRAINT global_candidate_source_identities_authority CHECK (
        verified_issuer = 'signal' AND verified_actor_id = 'signal-service'
    ),
    CONSTRAINT global_candidate_source_identities_provider_unique
        UNIQUE (provider_namespace, record_type, provider_record_id)
);

CREATE INDEX global_candidate_source_identities_candidate_idx
    ON global_candidate_source_identities (global_candidate_id);
CREATE INDEX global_candidate_source_identities_linkedin_idx
    ON global_candidate_source_identities (canonical_linkedin_url);

CREATE TABLE global_candidate_source_observations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    idempotency_key CHAR(64) NOT NULL UNIQUE,
    source_identity_id UUID
        REFERENCES global_candidate_source_identities(id) ON DELETE RESTRICT,
    global_candidate_id UUID
        REFERENCES global_candidates(id) ON DELETE RESTRICT,
    provider_namespace TEXT NOT NULL,
    record_type TEXT NOT NULL,
    adapter_family TEXT NOT NULL,
    adapter_version INTEGER NOT NULL,
    provider_record_id TEXT NOT NULL,
    canonical_linkedin_url TEXT NOT NULL,
    expected_global_candidate_id UUID,
    acquisition_receipt_id TEXT NOT NULL,
    acquisition_generation INTEGER NOT NULL,
    acquisition_slot TEXT NOT NULL,
    acquired_at TIMESTAMPTZ NOT NULL,
    provider_observed_at TIMESTAMPTZ NOT NULL,
    schema_version INTEGER NOT NULL,
    normalized_profile JSONB NOT NULL,
    profile_digest CHAR(64) NOT NULL,
    outcome TEXT NOT NULL,
    conflict_code TEXT,
    verified_issuer TEXT NOT NULL,
    verified_actor_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT global_candidate_source_observations_idempotency CHECK (
        idempotency_key ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT global_candidate_source_observations_provider_v1 CHECK (
        provider_namespace = 'crustdata'
        AND record_type = 'person'
        AND adapter_family = 'crustdata_person'
        AND adapter_version = 1
        AND schema_version = 1
    ),
    CONSTRAINT global_candidate_source_observations_provider_id CHECK (
        provider_record_id ~ '^[1-9][0-9]{0,18}$'
    ),
    CONSTRAINT global_candidate_source_observations_linkedin CHECK (
        canonical_linkedin_url ~ '^https://linkedin\.com/in/[a-z0-9][a-z0-9._-]{0,199}$'
    ),
    CONSTRAINT global_candidate_source_observations_acquisition_receipt CHECK (
        length(acquisition_receipt_id) BETWEEN 1 AND 200
        AND acquisition_receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
    ),
    CONSTRAINT global_candidate_source_observations_generation CHECK (
        acquisition_generation > 0
    ),
    CONSTRAINT global_candidate_source_observations_slot CHECK (
        acquisition_slot IN ('exact', 'spill')
    ),
    CONSTRAINT global_candidate_source_observations_profile CHECK (
        jsonb_typeof(normalized_profile) = 'object'
        AND octet_length(normalized_profile::text) <= 196608
    ),
    CONSTRAINT global_candidate_source_observations_profile_digest CHECK (
        profile_digest ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT global_candidate_source_observations_outcome CHECK (
        outcome IN ('accepted', 'stale', 'conflict_review_required')
    ),
    CONSTRAINT global_candidate_source_observations_conflict_code CHECK (
        conflict_code IS NULL
        OR conflict_code IN (
            'provider_linkedin_mismatch',
            'provider_expected_mismatch',
            'linkedin_expected_mismatch',
            'linkedin_ambiguous',
            'invalid_identity_state'
        )
    ),
    CONSTRAINT global_candidate_source_observations_resolution_shape CHECK (
        (
            outcome IN ('accepted', 'stale')
            AND source_identity_id IS NOT NULL
            AND global_candidate_id IS NOT NULL
            AND conflict_code IS NULL
        )
        OR (
            outcome = 'conflict_review_required'
            AND source_identity_id IS NULL
            AND global_candidate_id IS NULL
            AND conflict_code IS NOT NULL
        )
    ),
    CONSTRAINT global_candidate_source_observations_authority CHECK (
        verified_issuer = 'signal' AND verified_actor_id = 'signal-service'
    )
);

CREATE INDEX global_candidate_source_observations_candidate_freshness_idx
    ON global_candidate_source_observations (
        global_candidate_id, provider_observed_at DESC, created_at DESC
    ) WHERE global_candidate_id IS NOT NULL;
CREATE INDEX global_candidate_source_observations_provider_idx
    ON global_candidate_source_observations (
        provider_namespace, record_type, provider_record_id, created_at DESC
    );
CREATE INDEX global_candidate_source_observations_acquisition_idx
    ON global_candidate_source_observations (
        acquisition_receipt_id, acquisition_generation, acquisition_slot
    );

CREATE TABLE global_candidate_ingest_receipts (
    idempotency_key CHAR(64) PRIMARY KEY,
    input_digest CHAR(64) NOT NULL,
    source_observation_id UUID NOT NULL UNIQUE
        REFERENCES global_candidate_source_observations(id) ON DELETE RESTRICT,
    source_identity_id UUID
        REFERENCES global_candidate_source_identities(id) ON DELETE RESTRICT,
    global_candidate_id UUID
        REFERENCES global_candidates(id) ON DELETE RESTRICT,
    resolution TEXT NOT NULL,
    provider_namespace TEXT NOT NULL,
    record_type TEXT NOT NULL,
    provider_record_id TEXT NOT NULL,
    acquisition_receipt_id TEXT NOT NULL,
    acquisition_generation INTEGER NOT NULL,
    acquisition_slot TEXT NOT NULL,
    verified_issuer TEXT NOT NULL,
    verified_actor_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT global_candidate_ingest_receipts_idempotency CHECK (
        idempotency_key ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT global_candidate_ingest_receipts_input_digest CHECK (
        input_digest ~ '^[0-9a-f]{64}$'
    ),
    CONSTRAINT global_candidate_ingest_receipts_resolution CHECK (
        resolution IN (
            'created', 'matched', 'refreshed', 'stale',
            'conflict_review_required'
        )
    ),
    CONSTRAINT global_candidate_ingest_receipts_provider_v1 CHECK (
        provider_namespace = 'crustdata' AND record_type = 'person'
    ),
    CONSTRAINT global_candidate_ingest_receipts_provider_id CHECK (
        provider_record_id ~ '^[1-9][0-9]{0,18}$'
    ),
    CONSTRAINT global_candidate_ingest_receipts_acquisition_receipt CHECK (
        length(acquisition_receipt_id) BETWEEN 1 AND 200
        AND acquisition_receipt_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$'
    ),
    CONSTRAINT global_candidate_ingest_receipts_generation CHECK (
        acquisition_generation > 0
    ),
    CONSTRAINT global_candidate_ingest_receipts_slot CHECK (
        acquisition_slot IN ('exact', 'spill')
    ),
    CONSTRAINT global_candidate_ingest_receipts_resolution_shape CHECK (
        (
            resolution = 'conflict_review_required'
            AND source_identity_id IS NULL
            AND global_candidate_id IS NULL
        )
        OR (
            resolution <> 'conflict_review_required'
            AND source_identity_id IS NOT NULL
            AND global_candidate_id IS NOT NULL
        )
    ),
    CONSTRAINT global_candidate_ingest_receipts_authority CHECK (
        verified_issuer = 'signal' AND verified_actor_id = 'signal-service'
    )
);

CREATE INDEX global_candidate_ingest_receipts_candidate_idx
    ON global_candidate_ingest_receipts (global_candidate_id)
    WHERE global_candidate_id IS NOT NULL;
CREATE INDEX global_candidate_ingest_receipts_acquisition_idx
    ON global_candidate_ingest_receipts (
        acquisition_receipt_id, acquisition_generation, acquisition_slot
    );

CREATE FUNCTION approved_provider_candidate_evidence_append_only()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    has_evidence BOOLEAN;
BEGIN
    IF TG_OP <> 'TRUNCATE' THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = TG_TABLE_NAME || ' is append-only (attempted ' || TG_OP || ')';
    END IF;

    EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I LIMIT 1)', TG_TABLE_NAME)
        INTO has_evidence;
    IF has_evidence THEN
        RAISE EXCEPTION USING
            ERRCODE = '55000',
            MESSAGE = TG_TABLE_NAME || ' contains committed evidence and cannot be truncated';
    END IF;
    RETURN NULL;
END;
$$;

CREATE TRIGGER global_candidate_source_identities_no_mutation
    BEFORE UPDATE OR DELETE ON global_candidate_source_identities
    FOR EACH ROW EXECUTE FUNCTION approved_provider_candidate_evidence_append_only();
CREATE TRIGGER global_candidate_source_identities_no_nonempty_truncate
    BEFORE TRUNCATE ON global_candidate_source_identities
    FOR EACH STATEMENT EXECUTE FUNCTION approved_provider_candidate_evidence_append_only();

CREATE TRIGGER global_candidate_source_observations_no_mutation
    BEFORE UPDATE OR DELETE ON global_candidate_source_observations
    FOR EACH ROW EXECUTE FUNCTION approved_provider_candidate_evidence_append_only();
CREATE TRIGGER global_candidate_source_observations_no_nonempty_truncate
    BEFORE TRUNCATE ON global_candidate_source_observations
    FOR EACH STATEMENT EXECUTE FUNCTION approved_provider_candidate_evidence_append_only();

CREATE TRIGGER global_candidate_ingest_receipts_no_mutation
    BEFORE UPDATE OR DELETE ON global_candidate_ingest_receipts
    FOR EACH ROW EXECUTE FUNCTION approved_provider_candidate_evidence_append_only();
CREATE TRIGGER global_candidate_ingest_receipts_no_nonempty_truncate
    BEFORE TRUNCATE ON global_candidate_ingest_receipts
    FOR EACH STATEMENT EXECUTE FUNCTION approved_provider_candidate_evidence_append_only();

REVOKE ALL ON global_candidate_source_identities FROM PUBLIC;
REVOKE ALL ON global_candidate_source_observations FROM PUBLIC;
REVOKE ALL ON global_candidate_ingest_receipts FROM PUBLIC;
REVOKE ALL ON FUNCTION approved_provider_candidate_evidence_append_only() FROM PUBLIC;

COMMENT ON TABLE global_candidate_source_identities IS
    'Immutable exact provider-person binding to the existing platform-global candidate id.';
COMMENT ON TABLE global_candidate_source_observations IS
    'Append-only provider-neutral professional-profile observation; no tenant, job, contact, or private evidence.';
COMMENT ON TABLE global_candidate_ingest_receipts IS
    'Append-only idempotent outcome authority for approved-provider candidate ingestion.';
