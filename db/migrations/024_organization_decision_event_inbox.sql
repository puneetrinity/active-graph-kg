-- Migration 024: tenant-isolated organization decision-event inbox.
--
-- This is an exact, minimized transport receipt for Flow decision events. It
-- contains no candidate identity or PII and does not touch the legacy
-- feedback-event or global-candidate paths.

CREATE TABLE organization_decision_event_inbox (
    tenant_id TEXT NOT NULL,
    source_system TEXT NOT NULL DEFAULT 'flow',
    event_id UUID PRIMARY KEY,
    delivery_sequence BIGINT NOT NULL UNIQUE,
    source_event_sequence BIGINT NOT NULL UNIQUE,
    payload_schema_version INTEGER NOT NULL,
    organization_id INTEGER NOT NULL,
    subject_type TEXT NOT NULL,
    subject_id INTEGER NOT NULL,
    job_id INTEGER NOT NULL,
    action_code TEXT NOT NULL,
    taxonomy_version INTEGER NOT NULL,
    rubric_id UUID,
    rubric_version INTEGER,
    rubric_approval_mode TEXT,
    jd_digest_version INTEGER,
    recommendation_action TEXT,
    reason_code TEXT,
    before_state JSONB NOT NULL,
    after_state JSONB NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL,
    payload_digest CHAR(64) NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),

    CONSTRAINT organization_decision_event_inbox_tenant_check CHECK (
        tenant_id = 'org_' || organization_id::text
        AND tenant_id ~ '^org_[1-9][0-9]*$'
    ),
    CONSTRAINT organization_decision_event_inbox_source_check CHECK (source_system = 'flow'),
    CONSTRAINT organization_decision_event_inbox_delivery_positive CHECK (delivery_sequence > 0),
    CONSTRAINT organization_decision_event_inbox_source_sequence_positive CHECK (source_event_sequence > 0),
    CONSTRAINT organization_decision_event_inbox_schema_v1 CHECK (payload_schema_version = 1),
    CONSTRAINT organization_decision_event_inbox_organization_positive CHECK (organization_id > 0),
    CONSTRAINT organization_decision_event_inbox_subject_v1 CHECK (
        subject_type = 'application' AND subject_id > 0 AND job_id > 0
    ),
    CONSTRAINT organization_decision_event_inbox_action_v1 CHECK (
        action_code = 'application_stage_moved'
    ),
    CONSTRAINT organization_decision_event_inbox_taxonomy_positive CHECK (taxonomy_version > 0),
    CONSTRAINT organization_decision_event_inbox_rubric_shape CHECK (
        (rubric_id IS NULL AND rubric_version IS NULL AND rubric_approval_mode IS NULL)
        OR (
            rubric_id IS NOT NULL
            AND rubric_version > 0
            AND rubric_approval_mode ~ '^[a-z0-9][a-z0-9_-]{0,79}$'
        )
    ),
    CONSTRAINT organization_decision_event_inbox_jd_digest_positive CHECK (
        jd_digest_version IS NULL OR jd_digest_version > 0
    ),
    CONSTRAINT organization_decision_event_inbox_recommendation_v1 CHECK (
        recommendation_action IS NULL OR recommendation_action IN ('advance','hold','reject')
    ),
    CONSTRAINT organization_decision_event_inbox_reason_bounded CHECK (
        reason_code IS NULL OR reason_code ~ '^[a-z0-9][a-z0-9_]{0,79}$'
    ),
    CONSTRAINT organization_decision_event_inbox_before_state_v1 CHECK (
        jsonb_typeof(before_state) = 'object'
        AND before_state ? 'stage_id'
        AND before_state = jsonb_build_object('stage_id', before_state->'stage_id')
        AND octet_length(before_state::text) <= 1024
        AND (
            before_state->'stage_id' = 'null'::jsonb
            OR (
                jsonb_typeof(before_state->'stage_id') = 'number'
                AND before_state->>'stage_id' ~ '^[1-9][0-9]*$'
                AND (before_state->>'stage_id')::numeric <= 2147483647
            )
        )
    ),
    CONSTRAINT organization_decision_event_inbox_after_state_v1 CHECK (
        jsonb_typeof(after_state) = 'object'
        AND after_state ? 'stage_id'
        AND after_state = jsonb_build_object('stage_id', after_state->'stage_id')
        AND octet_length(after_state::text) <= 1024
        AND jsonb_typeof(after_state->'stage_id') = 'number'
        AND after_state->>'stage_id' ~ '^[1-9][0-9]*$'
        AND (after_state->>'stage_id')::numeric <= 2147483647
    ),
    CONSTRAINT organization_decision_event_inbox_state_changed CHECK (
        before_state->'stage_id' IS DISTINCT FROM after_state->'stage_id'
    ),
    CONSTRAINT organization_decision_event_inbox_digest_check CHECK (
        payload_digest ~ '^[0-9a-f]{64}$'
    )
);

CREATE INDEX organization_decision_event_inbox_tenant_delivery_idx
    ON organization_decision_event_inbox (tenant_id, delivery_sequence);
CREATE INDEX organization_decision_event_inbox_tenant_source_idx
    ON organization_decision_event_inbox (tenant_id, source_event_sequence);

CREATE TABLE organization_decision_stream_state (
    tenant_id TEXT PRIMARY KEY CHECK (tenant_id ~ '^org_[1-9][0-9]*$'),
    state TEXT NOT NULL DEFAULT 'current' CHECK (state = 'current'),
    last_delivery_sequence BIGINT NOT NULL CHECK (last_delivery_sequence > 0),
    last_source_event_sequence BIGINT NOT NULL
        CONSTRAINT organization_decision_stream_source_sequence_check
        CHECK (last_source_event_sequence > 0),
    last_event_id UUID NOT NULL UNIQUE
        REFERENCES organization_decision_event_inbox(event_id) ON DELETE RESTRICT,
    last_received_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);

ALTER TABLE organization_decision_event_inbox ENABLE ROW LEVEL SECURITY;
ALTER TABLE organization_decision_event_inbox FORCE ROW LEVEL SECURITY;
ALTER TABLE organization_decision_stream_state ENABLE ROW LEVEL SECURITY;
ALTER TABLE organization_decision_stream_state FORCE ROW LEVEL SECURITY;

CREATE POLICY organization_decision_event_inbox_tenant
    ON organization_decision_event_inbox
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true));

CREATE POLICY organization_decision_stream_state_tenant
    ON organization_decision_stream_state
    FOR ALL TO PUBLIC
    USING (tenant_id = current_setting('app.current_tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true));

REVOKE ALL ON organization_decision_event_inbox FROM PUBLIC;
REVOKE ALL ON organization_decision_stream_state FROM PUBLIC;

COMMENT ON TABLE organization_decision_event_inbox IS
    'Tenant-private, PII-free receipt of one Flow organization decision event; not a candidate projection.';
COMMENT ON TABLE organization_decision_stream_state IS
    'Current per-tenant accepted sequence watermark for the organization decision inbox.';
