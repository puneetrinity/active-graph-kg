-- Migration 023: reversible candidate-privacy directive authority.
--
-- This migration is additive and deliberately does not inspect or mutate any
-- existing product row.  Privacy requests are an append-only event stream plus
-- a compare-and-set projection.  Runtime code can use only the narrowly scoped
-- SECURITY DEFINER command and lookup functions below.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE candidate_privacy_directive_events (
    cursor BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    directive_id UUID NOT NULL,
    directive_version BIGINT NOT NULL CHECK (directive_version > 0),
    request_id UUID NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN (
        'requested', 'verified', 'activated', 'review_required', 'released', 'superseded'
    )),
    action TEXT NOT NULL CHECK (action IN ('withdraw_global_matching', 'request_erasure')),
    scope TEXT NOT NULL CHECK (scope IN ('global_matching', 'active_profile')),
    resulting_state TEXT NOT NULL CHECK (resulting_state IN (
        'requested', 'verified', 'active_quarantine', 'needs_review',
        'released', 'superseded', 'hard_purge_eligible'
    )),
    authority_type TEXT NOT NULL CHECK (authority_type IN ('verified_candidate', 'privacy_operator')),
    evidence_ref UUID NOT NULL,
    reason_code TEXT NOT NULL CHECK (reason_code IN (
        'candidate_global_opt_out', 'candidate_erasure_request',
        'verified_support_request', 'identity_ambiguity', 'operator_correction'
    )),
    issuer TEXT NOT NULL CHECK (btrim(issuer) <> ''),
    actor_id TEXT NOT NULL CHECK (btrim(actor_id) <> ''),
    actor_type TEXT NOT NULL CHECK (actor_type = 'service'),
    global_candidate_id UUID REFERENCES global_candidates(id) ON DELETE RESTRICT,
    candidate_tenant_id TEXT,
    candidate_id UUID,
    key_version INTEGER NOT NULL CHECK (key_version > 0),
    schema_version INTEGER NOT NULL DEFAULT 1 CHECK (schema_version = 1),
    effective_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT candidate_privacy_event_action_scope_check CHECK (
        (action = 'withdraw_global_matching' AND scope = 'global_matching') OR
        (action = 'request_erasure' AND scope = 'active_profile')
    ),
    CONSTRAINT candidate_privacy_event_candidate_pair_check CHECK (
        (candidate_tenant_id IS NULL) = (candidate_id IS NULL)
    ),
    CONSTRAINT candidate_privacy_event_candidate_fkey
        FOREIGN KEY (candidate_tenant_id, candidate_id)
        REFERENCES candidates(tenant_id, candidate_id) ON DELETE RESTRICT,
    CONSTRAINT candidate_privacy_event_directive_version_unique
        UNIQUE (directive_id, directive_version),
    CONSTRAINT candidate_privacy_event_request_type_unique
        UNIQUE (issuer, request_id, event_type)
);

CREATE INDEX candidate_privacy_events_cursor_idx
    ON candidate_privacy_directive_events (cursor);
CREATE INDEX candidate_privacy_events_directive_idx
    ON candidate_privacy_directive_events (directive_id, directive_version);

CREATE TABLE candidate_privacy_directives (
    directive_id UUID PRIMARY KEY,
    action TEXT NOT NULL CHECK (action IN ('withdraw_global_matching', 'request_erasure')),
    scope TEXT NOT NULL CHECK (scope IN ('global_matching', 'active_profile')),
    state TEXT NOT NULL CHECK (state IN (
        'requested', 'verified', 'active_quarantine', 'needs_review',
        'released', 'superseded', 'hard_purge_eligible'
    )),
    version BIGINT NOT NULL CHECK (version > 0),
    authority_type TEXT NOT NULL CHECK (authority_type IN ('verified_candidate', 'privacy_operator')),
    reason_code TEXT NOT NULL CHECK (reason_code IN (
        'candidate_global_opt_out', 'candidate_erasure_request',
        'verified_support_request', 'identity_ambiguity', 'operator_correction'
    )),
    global_candidate_id UUID REFERENCES global_candidates(id) ON DELETE RESTRICT,
    candidate_tenant_id TEXT,
    candidate_id UUID,
    last_event_cursor BIGINT NOT NULL UNIQUE
        REFERENCES candidate_privacy_directive_events(cursor) ON DELETE RESTRICT,
    effective_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT candidate_privacy_directive_action_scope_check CHECK (
        (action = 'withdraw_global_matching' AND scope = 'global_matching') OR
        (action = 'request_erasure' AND scope = 'active_profile')
    ),
    CONSTRAINT candidate_privacy_directive_candidate_pair_check CHECK (
        (candidate_tenant_id IS NULL) = (candidate_id IS NULL)
    ),
    CONSTRAINT candidate_privacy_directive_candidate_fkey
        FOREIGN KEY (candidate_tenant_id, candidate_id)
        REFERENCES candidates(tenant_id, candidate_id) ON DELETE RESTRICT
);

CREATE INDEX candidate_privacy_directives_global_idx
    ON candidate_privacy_directives (global_candidate_id)
    WHERE global_candidate_id IS NOT NULL;
CREATE INDEX candidate_privacy_directives_candidate_idx
    ON candidate_privacy_directives (candidate_tenant_id, candidate_id)
    WHERE candidate_id IS NOT NULL;

CREATE TABLE candidate_privacy_identity_tokens (
    directive_id UUID NOT NULL
        REFERENCES candidate_privacy_directives(directive_id) ON DELETE RESTRICT,
    identifier_type TEXT NOT NULL CHECK (identifier_type IN (
        'email', 'phone', 'linkedin_url', 'github_url', 'signal_candidate_id',
        'vantahire_application_id', 'vantahire_resume_id'
    )),
    key_version INTEGER NOT NULL CHECK (key_version > 0),
    token BYTEA NOT NULL CHECK (octet_length(token) = 32),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (directive_id, identifier_type, key_version, token)
);

CREATE INDEX candidate_privacy_identity_tokens_lookup_idx
    ON candidate_privacy_identity_tokens (identifier_type, key_version, token);

CREATE FUNCTION candidate_privacy_append_only()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'candidate privacy authority is append-only (attempted %)', TG_OP;
END;
$$;

CREATE TRIGGER candidate_privacy_events_no_mutation
    BEFORE UPDATE OR DELETE ON candidate_privacy_directive_events
    FOR EACH ROW EXECUTE FUNCTION candidate_privacy_append_only();
CREATE TRIGGER candidate_privacy_events_no_truncate
    BEFORE TRUNCATE ON candidate_privacy_directive_events
    FOR EACH STATEMENT EXECUTE FUNCTION candidate_privacy_append_only();
CREATE TRIGGER candidate_privacy_tokens_no_mutation
    BEFORE UPDATE OR DELETE ON candidate_privacy_identity_tokens
    FOR EACH ROW EXECUTE FUNCTION candidate_privacy_append_only();
CREATE TRIGGER candidate_privacy_tokens_no_truncate
    BEFORE TRUNCATE ON candidate_privacy_identity_tokens
    FOR EACH STATEMENT EXECUTE FUNCTION candidate_privacy_append_only();

ALTER TABLE candidate_privacy_directive_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE candidate_privacy_directives ENABLE ROW LEVEL SECURITY;
ALTER TABLE candidate_privacy_identity_tokens ENABLE ROW LEVEL SECURITY;

REVOKE ALL ON candidate_privacy_directive_events FROM PUBLIC;
REVOKE ALL ON candidate_privacy_directives FROM PUBLIC;
REVOKE ALL ON candidate_privacy_identity_tokens FROM PUBLIC;
REVOKE ALL ON SEQUENCE candidate_privacy_directive_events_cursor_seq FROM PUBLIC;

CREATE POLICY candidate_privacy_events_runtime_read
    ON candidate_privacy_directive_events FOR SELECT TO PUBLIC USING (true);
CREATE POLICY candidate_privacy_directives_runtime_read
    ON candidate_privacy_directives FOR SELECT TO PUBLIC USING (true);

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'activekg_app') THEN
        -- The restricted role's owner-managed default privileges historically
        -- grant DML on new public tables.  Revoke that inherited posture
        -- explicitly before installing the narrower privacy authority grants.
        REVOKE ALL ON candidate_privacy_directive_events FROM activekg_app;
        REVOKE ALL ON candidate_privacy_directives FROM activekg_app;
        REVOKE ALL ON candidate_privacy_identity_tokens FROM activekg_app;
        REVOKE ALL ON SEQUENCE candidate_privacy_directive_events_cursor_seq
            FROM activekg_app;
        GRANT SELECT ON candidate_privacy_directive_events TO activekg_app;
        GRANT SELECT ON candidate_privacy_directives TO activekg_app;
    END IF;
END;
$$;

CREATE FUNCTION candidate_privacy_decision_for(
    p_global_candidate_id UUID,
    p_candidate_tenant_id TEXT,
    p_candidate_id UUID
)
RETURNS TEXT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT COALESCE((
        SELECT CASE
            WHEN d.state IN ('needs_review', 'hard_purge_eligible') THEN 'review'
            WHEN d.state = 'active_quarantine' AND d.action = 'request_erasure' THEN 'block_all'
            WHEN d.state = 'active_quarantine' AND d.action = 'withdraw_global_matching' THEN 'block_global'
            ELSE 'allow'
        END
        FROM public.candidate_privacy_directives d
        WHERE (
            p_global_candidate_id IS NOT NULL
            AND d.global_candidate_id = p_global_candidate_id
        ) OR (
            p_candidate_tenant_id IS NOT NULL AND p_candidate_id IS NOT NULL
            AND d.candidate_tenant_id = p_candidate_tenant_id
            AND d.candidate_id = p_candidate_id
        )
        ORDER BY
            CASE
                WHEN d.state IN ('needs_review', 'hard_purge_eligible') THEN 4
                WHEN d.state = 'active_quarantine' AND d.action = 'request_erasure' THEN 3
                WHEN d.state = 'active_quarantine' AND d.action = 'withdraw_global_matching' THEN 2
                ELSE 1
            END DESC,
            d.effective_at DESC,
            d.version DESC
        LIMIT 1
    ), 'allow');
$$;

CREATE FUNCTION candidate_privacy_global_decision(p_global_candidate_id UUID)
RETURNS TEXT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT public.candidate_privacy_decision_for(p_global_candidate_id, NULL, NULL);
$$;

CREATE FUNCTION candidate_privacy_candidate_decision(
    p_candidate_tenant_id TEXT,
    p_candidate_id UUID
)
RETURNS TEXT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT public.candidate_privacy_decision_for(
        (SELECT c.global_candidate_id FROM public.candidates c
         WHERE c.tenant_id IS NOT DISTINCT FROM p_candidate_tenant_id
           AND c.candidate_id = p_candidate_id),
        CASE WHEN p_candidate_tenant_id IS NOT NULL THEN p_candidate_tenant_id END,
        CASE WHEN p_candidate_tenant_id IS NOT NULL THEN p_candidate_id END
    );
$$;

CREATE FUNCTION candidate_privacy_node_decision(p_node_id UUID)
RETURNS TEXT
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    WITH linked AS (
        SELECT c.global_candidate_id,
               c.tenant_id AS candidate_tenant_id,
               CASE WHEN c.tenant_id IS NOT NULL THEN c.candidate_id END AS candidate_id
        FROM public.candidates c
        WHERE c.node_id = p_node_id
        UNION
        SELECT cp.global_candidate_id, NULL::text, NULL::uuid
        FROM public.candidate_provenance cp
        WHERE cp.source_detail->>'resume_node_id' = p_node_id::text
          AND cp.source_type IN ('platform_applicant', 'org_upload')
    )
    SELECT COALESCE((
        SELECT decision
        FROM (
            SELECT public.candidate_privacy_decision_for(
                l.global_candidate_id, l.candidate_tenant_id, l.candidate_id
            ) AS decision
            FROM linked l
        ) decisions
        ORDER BY CASE decision
            WHEN 'review' THEN 4 WHEN 'block_all' THEN 3
            WHEN 'block_global' THEN 2 ELSE 1 END DESC
        LIMIT 1
    ), 'allow');
$$;

CREATE FUNCTION candidate_privacy_resolve_subject(
    p_identifier_type TEXT,
    p_lookup_digest BYTEA
)
RETURNS TABLE(global_candidate_id UUID, candidate_tenant_id TEXT, candidate_id UUID)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT DISTINCT resolved.global_candidate_id, resolved.candidate_tenant_id, resolved.candidate_id
    FROM (
        SELECT c.global_candidate_id,
               c.tenant_id AS candidate_tenant_id,
               c.candidate_id
        FROM public.candidate_identifiers ci
        JOIN public.candidates c
          ON c.tenant_id IS NOT DISTINCT FROM ci.tenant_id
         AND c.candidate_id = ci.candidate_id
        WHERE ci.identifier_type = p_identifier_type
          AND digest(convert_to(ci.value_normalized, 'UTF8'), 'sha256') = p_lookup_digest

        UNION ALL

        SELECT c.global_candidate_id,
               c.tenant_id AS candidate_tenant_id,
               c.candidate_id
        FROM public.candidates c
        WHERE p_identifier_type = 'email'
          AND c.primary_email IS NOT NULL
          AND digest(convert_to(lower(btrim(c.primary_email)), 'UTF8'), 'sha256')
              = p_lookup_digest

        UNION ALL

        SELECT gc.id AS global_candidate_id,
               NULL::text AS candidate_tenant_id,
               NULL::uuid AS candidate_id
        FROM public.global_candidates gc
        WHERE (p_identifier_type = 'email' AND gc.email_hash = encode(p_lookup_digest, 'hex'))
           OR (p_identifier_type = 'linkedin_url' AND gc.linkedin_id IS NOT NULL
               AND digest(convert_to(
                   'https://linkedin.com/in/' || lower(btrim(gc.linkedin_id)), 'UTF8'
               ), 'sha256') = p_lookup_digest)
           OR (p_identifier_type = 'github_url' AND gc.github_id IS NOT NULL
               AND digest(convert_to(
                   'https://github.com/' || lower(btrim(gc.github_id)), 'UTF8'
               ), 'sha256') = p_lookup_digest)
    ) resolved;
$$;

CREATE FUNCTION candidate_privacy_resolve_canonical(
    p_global_candidate_id UUID,
    p_candidate_tenant_id TEXT,
    p_candidate_id UUID
)
RETURNS TABLE(
    global_candidate_id UUID,
    candidate_tenant_id TEXT,
    candidate_id UUID,
    needs_review BOOLEAN
)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    linked_global_candidate_id UUID;
BEGIN
    IF (p_candidate_tenant_id IS NULL) <> (p_candidate_id IS NULL) THEN
        RAISE EXCEPTION 'candidate privacy canonical reference is incomplete';
    END IF;
    IF p_global_candidate_id IS NOT NULL AND NOT EXISTS (
        SELECT 1 FROM public.global_candidates gc WHERE gc.id = p_global_candidate_id
    ) THEN
        RAISE EXCEPTION 'candidate privacy canonical reference is invalid';
    END IF;
    IF p_candidate_id IS NOT NULL THEN
        SELECT c.global_candidate_id INTO linked_global_candidate_id
        FROM public.candidates c
        WHERE c.tenant_id = p_candidate_tenant_id AND c.candidate_id = p_candidate_id;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'candidate privacy canonical reference is invalid';
        END IF;
    END IF;
    RETURN QUERY SELECT
        COALESCE(p_global_candidate_id, linked_global_candidate_id),
        p_candidate_tenant_id,
        p_candidate_id,
        p_global_candidate_id IS NOT NULL
            AND p_candidate_id IS NOT NULL
            AND linked_global_candidate_id IS DISTINCT FROM p_global_candidate_id;
END;
$$;

CREATE FUNCTION candidate_privacy_match(
    p_tokens JSONB,
    p_global_candidate_id UUID,
    p_candidate_tenant_id TEXT,
    p_candidate_id UUID
)
RETURNS TABLE(
    directive_id UUID,
    action TEXT,
    scope TEXT,
    state TEXT,
    version BIGINT,
    effective_at TIMESTAMPTZ,
    decision TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    WITH supplied AS (
        SELECT
            item->>'identifier_type' AS identifier_type,
            (item->>'key_version')::integer AS key_version,
            decode(item->>'token', 'hex') AS token
        FROM jsonb_array_elements(COALESCE(p_tokens, '[]'::jsonb)) item
        WHERE jsonb_typeof(item) = 'object'
    ), matched AS (
        SELECT DISTINCT d.*
        FROM public.candidate_privacy_directives d
        WHERE (p_global_candidate_id IS NOT NULL AND d.global_candidate_id = p_global_candidate_id)
           OR (p_candidate_tenant_id IS NOT NULL AND p_candidate_id IS NOT NULL
               AND d.candidate_tenant_id = p_candidate_tenant_id
               AND d.candidate_id = p_candidate_id)
           OR EXISTS (
               SELECT 1
               FROM public.candidate_privacy_identity_tokens t
               JOIN supplied s USING (identifier_type, key_version, token)
               WHERE t.directive_id = d.directive_id
           )
    )
    SELECT
        m.directive_id, m.action, m.scope, m.state, m.version, m.effective_at,
        CASE
            WHEN m.state IN ('needs_review', 'hard_purge_eligible') THEN 'review'
            WHEN m.state = 'active_quarantine' AND m.action = 'request_erasure' THEN 'block_all'
            WHEN m.state = 'active_quarantine' AND m.action = 'withdraw_global_matching' THEN 'block_global'
            ELSE 'allow'
        END AS decision
    FROM matched m
    ORDER BY
        CASE
            WHEN m.state IN ('needs_review', 'hard_purge_eligible') THEN 4
            WHEN m.state = 'active_quarantine' AND m.action = 'request_erasure' THEN 3
            WHEN m.state = 'active_quarantine' AND m.action = 'withdraw_global_matching' THEN 2
            ELSE 1
        END DESC,
        m.effective_at DESC,
        m.version DESC;
$$;

CREATE FUNCTION candidate_privacy_token_key_versions()
RETURNS TABLE(key_version INTEGER)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT DISTINCT t.key_version
    FROM public.candidate_privacy_identity_tokens t
    ORDER BY t.key_version;
$$;

CREATE FUNCTION candidate_privacy_create_directive(
    p_directive_id UUID,
    p_request_id UUID,
    p_action TEXT,
    p_scope TEXT,
    p_authority_type TEXT,
    p_evidence_ref UUID,
    p_reason_code TEXT,
    p_issuer TEXT,
    p_actor_id TEXT,
    p_global_candidate_id UUID,
    p_candidate_tenant_id TEXT,
    p_candidate_id UUID,
    p_key_version INTEGER,
    p_tokens JSONB,
    p_needs_review BOOLEAN,
    p_effective_at TIMESTAMPTZ
)
RETURNS SETOF candidate_privacy_directives
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    existing_event candidate_privacy_directive_events%ROWTYPE;
    final_cursor BIGINT;
    final_state TEXT;
    token_count INTEGER;
    token_item JSONB;
    replay_token_difference BOOLEAN;
    resolved_global_candidate_id UUID;
    resolved_candidate_tenant_id TEXT;
    resolved_candidate_id UUID;
    canonical_needs_review BOOLEAN;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(p_issuer || ':' || p_request_id::text, 0));

    -- Commands for the same exact subject are serialized independently of the
    -- caller's request id.  Sorting gives every transaction the same lock
    -- order and therefore avoids a multi-identifier deadlock.
    FOR token_item IN
        SELECT item
        FROM jsonb_array_elements(COALESCE(p_tokens, '[]'::jsonb)) item
        ORDER BY item->>'identifier_type', item->>'key_version', item->>'token'
    LOOP
        IF jsonb_typeof(token_item) <> 'object'
           OR token_item->>'identifier_type' NOT IN (
               'email', 'phone', 'linkedin_url', 'github_url', 'signal_candidate_id',
               'vantahire_application_id', 'vantahire_resume_id'
           )
           OR COALESCE(token_item->>'key_version', '') !~ '^[1-9][0-9]*$'
           OR COALESCE(token_item->>'token', '') !~ '^[0-9a-f]{64}$' THEN
            RAISE EXCEPTION 'candidate privacy token payload is invalid';
        END IF;
        IF (token_item->>'key_version')::integer IS DISTINCT FROM p_key_version THEN
            RAISE EXCEPTION 'candidate privacy token key version is invalid';
        END IF;
        PERFORM pg_advisory_xact_lock(hashtextextended(
            'candidate-privacy-token:' || (token_item->>'identifier_type') || ':' ||
            (token_item->>'key_version') || ':' || (token_item->>'token'),
            0
        ));
    END LOOP;
    SELECT resolved.global_candidate_id, resolved.candidate_tenant_id,
           resolved.candidate_id, resolved.needs_review
      INTO resolved_global_candidate_id, resolved_candidate_tenant_id,
           resolved_candidate_id, canonical_needs_review
    FROM public.candidate_privacy_resolve_canonical(
        p_global_candidate_id, p_candidate_tenant_id, p_candidate_id
    ) resolved;
    p_global_candidate_id := resolved_global_candidate_id;
    p_candidate_tenant_id := resolved_candidate_tenant_id;
    p_candidate_id := resolved_candidate_id;
    p_needs_review := p_needs_review OR canonical_needs_review;
    IF p_global_candidate_id IS NOT NULL THEN
        PERFORM pg_advisory_xact_lock(hashtextextended(
            'candidate-privacy-global:' || p_global_candidate_id::text, 0
        ));
    END IF;
    IF p_candidate_tenant_id IS NOT NULL AND p_candidate_id IS NOT NULL THEN
        PERFORM pg_advisory_xact_lock(hashtextextended(
            'candidate-privacy-candidate:' || p_candidate_tenant_id || ':' || p_candidate_id::text,
            0
        ));
    END IF;

    SELECT * INTO existing_event
    FROM public.candidate_privacy_directive_events
    WHERE issuer = p_issuer AND request_id = p_request_id AND event_type = 'requested';

    IF FOUND THEN
        IF existing_event.action IS DISTINCT FROM p_action
           OR existing_event.scope IS DISTINCT FROM p_scope
           OR existing_event.authority_type IS DISTINCT FROM p_authority_type
           OR existing_event.evidence_ref IS DISTINCT FROM p_evidence_ref
           OR existing_event.reason_code IS DISTINCT FROM p_reason_code
           OR existing_event.actor_id IS DISTINCT FROM p_actor_id
           OR existing_event.global_candidate_id IS DISTINCT FROM p_global_candidate_id
           OR existing_event.candidate_tenant_id IS DISTINCT FROM p_candidate_tenant_id
           OR existing_event.candidate_id IS DISTINCT FROM p_candidate_id
           OR existing_event.key_version IS DISTINCT FROM p_key_version THEN
            RAISE EXCEPTION 'candidate privacy request replay conflict';
        END IF;
        WITH supplied AS (
            SELECT item->>'identifier_type' AS identifier_type,
                   (item->>'key_version')::integer AS key_version,
                   decode(item->>'token', 'hex') AS token
            FROM jsonb_array_elements(COALESCE(p_tokens, '[]'::jsonb)) item
        ), stored AS (
            SELECT identifier_type, key_version, token
            FROM public.candidate_privacy_identity_tokens
            WHERE directive_id = existing_event.directive_id
        ), difference AS (
            (SELECT * FROM supplied EXCEPT SELECT * FROM stored)
            UNION ALL
            (SELECT * FROM stored EXCEPT SELECT * FROM supplied)
        )
        SELECT EXISTS (SELECT 1 FROM difference) INTO replay_token_difference;
        IF replay_token_difference OR (
            SELECT resulting_state = 'needs_review'
            FROM public.candidate_privacy_directive_events
            WHERE directive_id = existing_event.directive_id
              AND directive_version = 3
        ) IS DISTINCT FROM p_needs_review THEN
            RAISE EXCEPTION 'candidate privacy request replay conflict';
        END IF;
        RETURN QUERY SELECT * FROM public.candidate_privacy_directives
            WHERE directive_id = existing_event.directive_id;
        RETURN;
    END IF;

    IF EXISTS (
        SELECT 1 FROM public.candidate_privacy_directive_events
        WHERE issuer = p_issuer AND request_id = p_request_id
    ) THEN
        RAISE EXCEPTION 'candidate privacy request replay conflict';
    END IF;

    IF p_actor_id IS NULL OR btrim(p_actor_id) = '' OR p_issuer IS NULL OR btrim(p_issuer) = '' THEN
        RAISE EXCEPTION 'candidate privacy service authority is incomplete';
    END IF;
    IF p_action = 'withdraw_global_matching' AND p_scope <> 'global_matching' THEN
        RAISE EXCEPTION 'candidate privacy action/scope mismatch';
    ELSIF p_action = 'request_erasure' AND p_scope <> 'active_profile' THEN
        RAISE EXCEPTION 'candidate privacy action/scope mismatch';
    END IF;
    IF (p_action = 'withdraw_global_matching' AND p_reason_code NOT IN (
            'candidate_global_opt_out', 'verified_support_request'
        )) OR (p_action = 'request_erasure' AND p_reason_code NOT IN (
            'candidate_erasure_request', 'verified_support_request'
        )) THEN
        RAISE EXCEPTION 'candidate privacy action/reason mismatch';
    END IF;
    IF p_authority_type NOT IN ('verified_candidate', 'privacy_operator') THEN
        RAISE EXCEPTION 'candidate privacy authority type is invalid';
    END IF;
    IF (p_candidate_tenant_id IS NULL) <> (p_candidate_id IS NULL) THEN
        RAISE EXCEPTION 'candidate privacy candidate reference is incomplete';
    END IF;
    IF jsonb_typeof(COALESCE(p_tokens, '[]'::jsonb)) <> 'array' THEN
        RAISE EXCEPTION 'candidate privacy tokens must be an array';
    END IF;
    SELECT count(*) INTO token_count FROM jsonb_array_elements(COALESCE(p_tokens, '[]'::jsonb));
    IF token_count > 8 OR (token_count = 0 AND p_global_candidate_id IS NULL AND p_candidate_id IS NULL) THEN
        RAISE EXCEPTION 'candidate privacy subject is empty or exceeds its bound';
    END IF;

    INSERT INTO public.candidate_privacy_directive_events (
        directive_id, directive_version, request_id, event_type, action, scope,
        resulting_state, authority_type, evidence_ref, reason_code, issuer,
        actor_id, actor_type, global_candidate_id, candidate_tenant_id,
        candidate_id, key_version, effective_at
    ) VALUES
        (p_directive_id, 1, p_request_id, 'requested', p_action, p_scope,
         'requested', p_authority_type, p_evidence_ref, p_reason_code, p_issuer,
         p_actor_id, 'service', p_global_candidate_id, p_candidate_tenant_id,
         p_candidate_id, p_key_version, p_effective_at),
        (p_directive_id, 2, p_request_id, 'verified', p_action, p_scope,
         'verified', p_authority_type, p_evidence_ref, p_reason_code, p_issuer,
         p_actor_id, 'service', p_global_candidate_id, p_candidate_tenant_id,
         p_candidate_id, p_key_version, p_effective_at);

    final_state := CASE WHEN p_needs_review THEN 'needs_review' ELSE 'active_quarantine' END;
    INSERT INTO public.candidate_privacy_directive_events (
        directive_id, directive_version, request_id, event_type, action, scope,
        resulting_state, authority_type, evidence_ref, reason_code, issuer,
        actor_id, actor_type, global_candidate_id, candidate_tenant_id,
        candidate_id, key_version, effective_at
    ) VALUES (
        p_directive_id, 3, p_request_id,
        CASE WHEN p_needs_review THEN 'review_required' ELSE 'activated' END,
        p_action, p_scope, final_state, p_authority_type, p_evidence_ref,
        CASE WHEN p_needs_review THEN 'identity_ambiguity' ELSE p_reason_code END,
        p_issuer, p_actor_id, 'service', p_global_candidate_id,
        p_candidate_tenant_id, p_candidate_id, p_key_version, p_effective_at
    ) RETURNING cursor INTO final_cursor;

    INSERT INTO public.candidate_privacy_directives (
        directive_id, action, scope, state, version, authority_type, reason_code,
        global_candidate_id, candidate_tenant_id, candidate_id,
        last_event_cursor, effective_at
    ) VALUES (
        p_directive_id, p_action, p_scope, final_state, 3, p_authority_type,
        CASE WHEN p_needs_review THEN 'identity_ambiguity' ELSE p_reason_code END,
        p_global_candidate_id, p_candidate_tenant_id, p_candidate_id,
        final_cursor, p_effective_at
    );

    INSERT INTO public.candidate_privacy_identity_tokens
        (directive_id, identifier_type, key_version, token)
    SELECT DISTINCT
        p_directive_id,
        item->>'identifier_type',
        (item->>'key_version')::integer,
        decode(item->>'token', 'hex')
    FROM jsonb_array_elements(COALESCE(p_tokens, '[]'::jsonb)) item;

    RETURN QUERY SELECT * FROM public.candidate_privacy_directives
        WHERE directive_id = p_directive_id;
END;
$$;

CREATE FUNCTION candidate_privacy_transition_directive(
    p_directive_id UUID,
    p_expected_version BIGINT,
    p_request_id UUID,
    p_transition TEXT,
    p_evidence_ref UUID,
    p_reason_code TEXT,
    p_issuer TEXT,
    p_actor_id TEXT,
    p_effective_at TIMESTAMPTZ
)
RETURNS SETOF candidate_privacy_directives
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE
    current_row candidate_privacy_directives%ROWTYPE;
    existing_event candidate_privacy_directive_events%ROWTYPE;
    new_cursor BIGINT;
    new_state TEXT;
    new_event_type TEXT;
    current_key_version INTEGER;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(p_issuer || ':' || p_request_id::text, 0));
    SELECT * INTO current_row FROM public.candidate_privacy_directives
      WHERE directive_id = p_directive_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'candidate privacy directive not found';
    END IF;

    IF p_transition = 'release' THEN
        new_state := 'released';
        new_event_type := 'released';
    ELSIF p_transition = 'mark_needs_review' THEN
        new_state := 'needs_review';
        new_event_type := 'review_required';
    ELSE
        RAISE EXCEPTION 'candidate privacy transition is not allowed';
    END IF;

    SELECT * INTO existing_event
    FROM public.candidate_privacy_directive_events
    WHERE issuer = p_issuer AND request_id = p_request_id
      AND event_type IN ('released', 'review_required');
    IF FOUND THEN
        IF existing_event.event_type IS DISTINCT FROM new_event_type
           OR existing_event.directive_id IS DISTINCT FROM p_directive_id
           OR existing_event.directive_version IS DISTINCT FROM p_expected_version + 1
           OR existing_event.evidence_ref IS DISTINCT FROM p_evidence_ref
           OR existing_event.actor_id IS DISTINCT FROM p_actor_id
           OR existing_event.reason_code IS DISTINCT FROM p_reason_code THEN
            RAISE EXCEPTION 'candidate privacy transition replay conflict';
        END IF;
        RETURN QUERY SELECT * FROM public.candidate_privacy_directives
            WHERE directive_id = p_directive_id;
        RETURN;
    END IF;

    IF EXISTS (
        SELECT 1 FROM public.candidate_privacy_directive_events
        WHERE issuer = p_issuer AND request_id = p_request_id
    ) THEN
        RAISE EXCEPTION 'candidate privacy transition replay conflict';
    END IF;

    IF current_row.version <> p_expected_version THEN
        RAISE EXCEPTION 'candidate privacy directive version conflict';
    END IF;
    IF (p_transition = 'release' AND p_reason_code NOT IN (
            'operator_correction', 'verified_support_request'
        )) OR (p_transition = 'mark_needs_review' AND p_reason_code NOT IN (
            'identity_ambiguity', 'operator_correction'
        )) THEN
        RAISE EXCEPTION 'candidate privacy transition/reason mismatch';
    END IF;

    SELECT key_version INTO STRICT current_key_version
    FROM public.candidate_privacy_directive_events
    WHERE directive_id = p_directive_id
    ORDER BY directive_version DESC
    LIMIT 1;

    INSERT INTO public.candidate_privacy_directive_events (
        directive_id, directive_version, request_id, event_type, action, scope,
        resulting_state, authority_type, evidence_ref, reason_code, issuer,
        actor_id, actor_type, global_candidate_id, candidate_tenant_id,
        candidate_id, key_version, effective_at
    ) VALUES (
        p_directive_id, current_row.version + 1, p_request_id, new_event_type,
        current_row.action, current_row.scope, new_state, current_row.authority_type,
        p_evidence_ref, p_reason_code, p_issuer, p_actor_id, 'service',
        current_row.global_candidate_id, current_row.candidate_tenant_id,
        current_row.candidate_id, current_key_version, p_effective_at
    ) RETURNING cursor INTO new_cursor;

    UPDATE public.candidate_privacy_directives
    SET state = new_state,
        version = current_row.version + 1,
        reason_code = p_reason_code,
        last_event_cursor = new_cursor,
        effective_at = p_effective_at,
        updated_at = now()
    WHERE directive_id = p_directive_id AND version = p_expected_version;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'candidate privacy directive compare-and-set failed';
    END IF;

    RETURN QUERY SELECT * FROM public.candidate_privacy_directives
        WHERE directive_id = p_directive_id;
END;
$$;

REVOKE ALL ON FUNCTION candidate_privacy_append_only() FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_decision_for(UUID, TEXT, UUID) FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_global_decision(UUID) FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_candidate_decision(TEXT, UUID) FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_node_decision(UUID) FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_resolve_subject(TEXT, BYTEA) FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_resolve_canonical(UUID, TEXT, UUID) FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_match(JSONB, UUID, TEXT, UUID) FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_token_key_versions() FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_create_directive(
    UUID, UUID, TEXT, TEXT, TEXT, UUID, TEXT, TEXT, TEXT, UUID, TEXT, UUID,
    INTEGER, JSONB, BOOLEAN, TIMESTAMPTZ
) FROM PUBLIC;
REVOKE ALL ON FUNCTION candidate_privacy_transition_directive(
    UUID, BIGINT, UUID, TEXT, UUID, TEXT, TEXT, TEXT, TIMESTAMPTZ
) FROM PUBLIC;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'activekg_app') THEN
        GRANT EXECUTE ON FUNCTION candidate_privacy_decision_for(UUID, TEXT, UUID) TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_global_decision(UUID) TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_candidate_decision(TEXT, UUID) TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_node_decision(UUID) TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_resolve_subject(TEXT, BYTEA) TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_resolve_canonical(UUID, TEXT, UUID)
            TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_match(JSONB, UUID, TEXT, UUID) TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_token_key_versions() TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_create_directive(
            UUID, UUID, TEXT, TEXT, TEXT, UUID, TEXT, TEXT, TEXT, UUID, TEXT,
            UUID, INTEGER, JSONB, BOOLEAN, TIMESTAMPTZ
        ) TO activekg_app;
        GRANT EXECUTE ON FUNCTION candidate_privacy_transition_directive(
            UUID, BIGINT, UUID, TEXT, UUID, TEXT, TEXT, TEXT, TIMESTAMPTZ
        ) TO activekg_app;
    END IF;
END;
$$;
