-- Migration 021: public-memory projection and contact evidence
--
-- ``global_candidates`` is an identity layer, not a cross-tenant profile
-- surface: its historical fields and embedding may contain applicant or
-- uploader evidence. Shared retrieval must therefore use this separate,
-- Crustdata-only projection. Contact evidence is intentionally tenant scoped
-- until provider terms explicitly permit cross-organisation reuse.

ALTER TABLE global_candidates
    ADD COLUMN IF NOT EXISTS public_profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS public_profile_observed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS public_crustdata_person_id BIGINT,
    ADD COLUMN IF NOT EXISTS public_headline TEXT,
    ADD COLUMN IF NOT EXISTS public_location_city TEXT,
    ADD COLUMN IF NOT EXISTS public_location_country_code TEXT,
    ADD COLUMN IF NOT EXISTS public_role_family TEXT,
    ADD COLUMN IF NOT EXISTS public_seniority_band TEXT,
    ADD COLUMN IF NOT EXISTS public_skills_normalized TEXT[],
    ADD COLUMN IF NOT EXISTS public_embedding VECTOR(384),
    ADD COLUMN IF NOT EXISTS public_embedding_status TEXT NOT NULL DEFAULT 'queued',
    ADD COLUMN IF NOT EXISTS public_embed_version INT NOT NULL DEFAULT 0;

ALTER TABLE global_candidates
    DROP CONSTRAINT IF EXISTS global_candidates_public_embedding_status_check;
ALTER TABLE global_candidates
    ADD CONSTRAINT global_candidates_public_embedding_status_check
    CHECK (public_embedding_status IN ('queued', 'ready', 'failed', 'skipped_empty'));

CREATE INDEX IF NOT EXISTS idx_gc_public_embedding_status
    ON global_candidates (public_embedding_status);
CREATE INDEX IF NOT EXISTS idx_gc_public_embed_version
    ON global_candidates (public_embed_version)
    WHERE public_embedding_status IN ('ready', 'skipped_empty');
CREATE INDEX IF NOT EXISTS idx_gc_public_location
    ON global_candidates (public_location_country_code, public_location_city);
CREATE INDEX IF NOT EXISTS idx_gc_public_role_family
    ON global_candidates (public_role_family);

-- Select documented provider fields only. Unknown nested keys are discarded:
-- Signal's tenant blob may also contain recruiter/applicant/provider additions,
-- and a key-name blacklist is not a privacy boundary.
CREATE OR REPLACE FUNCTION activekg_redact_public_contact_text(value TEXT)
RETURNS TEXT
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT regexp_replace(
        regexp_replace(
            value,
            '[[:alnum:]._%+-]+@[[:alnum:].-]+\.[[:alpha:]]{2,}',
            '[redacted]',
            'gi'
        ),
        '(\+[0-9][0-9(). -]{5,}[0-9]|[(][0-9]{2,4}[)][ .-][0-9]{3,4}[ .-][0-9]{3,4}|[0-9]{2,4}[ .-][0-9]{3,4}[ .-][0-9]{3,4}|[0-9]{5}[ .-][0-9]{5}|[0-9]{9,15})',
        '[redacted]',
        'g'
    )
$$;

CREATE OR REPLACE FUNCTION activekg_pick_public_fields(
    value JSONB,
    scalar_allowed TEXT[],
    list_allowed TEXT[] DEFAULT ARRAY[]::TEXT[]
)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT COALESCE(jsonb_object_agg(key, sanitized), '{}'::jsonb)
      FROM (
          SELECT key,
                 CASE
                     WHEN key = ANY(scalar_allowed)
                      AND jsonb_typeof(child) = 'string'
                     THEN to_jsonb(activekg_redact_public_contact_text(child #>> '{}'))
                     WHEN key = ANY(scalar_allowed)
                      AND jsonb_typeof(child) IN ('number', 'boolean')
                     THEN child
                     WHEN key = ANY(list_allowed)
                      AND jsonb_typeof(child) = 'array'
                     THEN (
                         SELECT COALESCE(
                             jsonb_agg(
                                 CASE
                                     WHEN jsonb_typeof(element) = 'string'
                                     THEN to_jsonb(
                                         activekg_redact_public_contact_text(element #>> '{}')
                                     )
                                     ELSE element
                                 END
                             ),
                             '[]'::jsonb
                         )
                         FROM jsonb_array_elements(child) element
                         WHERE jsonb_typeof(element) IN ('string', 'number', 'boolean')
                     )
                     ELSE NULL
                 END AS sanitized
          FROM jsonb_each(
              CASE WHEN jsonb_typeof(value) = 'object' THEN value ELSE '{}'::jsonb END
          ) AS source_item(key, child)
      ) item
     WHERE sanitized IS NOT NULL
       AND sanitized <> '[]'::jsonb
$$;

CREATE OR REPLACE FUNCTION activekg_pick_public_rows(
    value JSONB,
    scalar_allowed TEXT[],
    list_allowed TEXT[] DEFAULT ARRAY[]::TEXT[]
)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT COALESCE(
        jsonb_agg(activekg_pick_public_fields(element, scalar_allowed, list_allowed))
            FILTER (
                WHERE activekg_pick_public_fields(
                    element, scalar_allowed, list_allowed
                ) <> '{}'::jsonb
            ),
        '[]'::jsonb
    )
    FROM jsonb_array_elements(
        CASE WHEN jsonb_typeof(value) = 'array' THEN value ELSE '[]'::jsonb END
    ) element
$$;

CREATE OR REPLACE FUNCTION activekg_public_crustdata_projection(value JSONB)
RETURNS JSONB
LANGUAGE sql
IMMUTABLE
AS $$
    WITH projected AS (
        SELECT
            activekg_pick_public_fields(
                value -> 'basic_profile',
                ARRAY['name','first_name','last_name','headline','current_title','summary'],
                ARRAY['languages']
            ) AS basic,
            activekg_pick_public_fields(
                value #> '{basic_profile,location}',
                ARRAY['city','state','country','continent','full_location','raw','country_code']
            ) AS basic_location,
            activekg_pick_public_fields(
                value -> 'professional_network',
                ARRAY['connections','followers','profile_picture_permalink'],
                ARRAY['open_to_cards']
            ) AS network,
            activekg_pick_public_fields(
                value #> '{professional_network,location}', ARRAY['raw']
            ) AS network_location,
            activekg_pick_public_fields(
                value #> '{professional_network,metadata}', ARRAY['last_scraped_source']
            ) AS network_metadata,
            activekg_pick_public_fields(
                value #> '{social_handles,professional_network_identifier}', ARRAY['profile_url']
            ) AS linkedin,
            activekg_pick_public_fields(
                value #> '{social_handles,twitter_identifier}', ARRAY['slug']
            ) AS twitter,
            activekg_pick_public_fields(
                value #> '{social_handles,dev_platform_identifier}', ARRAY['profile_url']
            ) AS developer,
            activekg_pick_public_rows(
                value #> '{experience,employment_details,current}',
                ARRAY[
                    'company_name','title','seniority_level','function_category',
                    'start_date','end_date','description','name','years_at_company_raw',
                    'company_headquarters_country',
                    'company_professional_network_industry','company_type',
                    'company_headcount_range'
                ],
                ARRAY['company_industries']
            ) AS current_jobs,
            activekg_pick_public_rows(
                value #> '{experience,employment_details,past}',
                ARRAY[
                    'company_name','title','seniority_level','function_category',
                    'start_date','end_date','description','name','years_at_company_raw',
                    'company_headquarters_country',
                    'company_professional_network_industry','company_type',
                    'company_headcount_range'
                ],
                ARRAY['company_industries']
            ) AS past_jobs,
            activekg_pick_public_rows(
                value #> '{education,schools}',
                ARRAY['school','degree','field_of_study','start_year','end_year']
            ) AS schools,
            activekg_pick_public_rows(
                value -> 'certifications',
                ARRAY['name','issuing_organization','issue_date','expiration_date']
            ) AS certifications,
            activekg_pick_public_rows(
                value -> 'honors', ARRAY['title','issuer','description']
            ) AS honors
    )
    SELECT jsonb_strip_nulls(jsonb_build_object(
        'crustdata_person_id', (
            activekg_pick_public_fields(
                value,
                ARRAY['crustdata_person_id','years_of_experience_raw','recently_changed_jobs']
            ) -> 'crustdata_person_id'
        ),
        'years_of_experience_raw', (
            activekg_pick_public_fields(
                value,
                ARRAY['crustdata_person_id','years_of_experience_raw','recently_changed_jobs']
            ) -> 'years_of_experience_raw'
        ),
        'recently_changed_jobs', (
            activekg_pick_public_fields(
                value,
                ARRAY['crustdata_person_id','years_of_experience_raw','recently_changed_jobs']
            ) -> 'recently_changed_jobs'
        ),
        'metadata', NULLIF(
            activekg_pick_public_fields(value -> 'metadata', ARRAY['updated_at']), '{}'::jsonb
        ),
        'basic_profile', NULLIF(
            jsonb_strip_nulls(
                projected.basic || jsonb_build_object(
                    'location', NULLIF(projected.basic_location, '{}'::jsonb)
                )
            ), '{}'::jsonb
        ),
        'professional_network', NULLIF(
            jsonb_strip_nulls(
                projected.network || jsonb_build_object(
                    'location', NULLIF(projected.network_location, '{}'::jsonb),
                    'metadata', NULLIF(projected.network_metadata, '{}'::jsonb)
                )
            ), '{}'::jsonb
        ),
        'social_handles', NULLIF(
            jsonb_strip_nulls(jsonb_build_object(
                'professional_network_identifier', NULLIF(projected.linkedin, '{}'::jsonb),
                'twitter_identifier', NULLIF(projected.twitter, '{}'::jsonb),
                'dev_platform_identifier', NULLIF(projected.developer, '{}'::jsonb)
            )), '{}'::jsonb
        ),
        'education', CASE WHEN projected.schools = '[]'::jsonb THEN NULL
            ELSE jsonb_build_object('schools', projected.schools) END,
        'experience', CASE
            WHEN projected.current_jobs = '[]'::jsonb AND projected.past_jobs = '[]'::jsonb
            THEN NULL
            ELSE jsonb_build_object(
                'employment_details',
                jsonb_strip_nulls(jsonb_build_object(
                    'current', NULLIF(projected.current_jobs, '[]'::jsonb),
                    'past', NULLIF(projected.past_jobs, '[]'::jsonb)
                ))
            )
        END,
        'skills', NULLIF(
            activekg_pick_public_fields(
                value -> 'skills',
                ARRAY[]::TEXT[],
                ARRAY['professional_network_skills']
            ), '{}'::jsonb
        ),
        'certifications', NULLIF(projected.certifications, '[]'::jsonb),
        'honors', NULLIF(projected.honors, '[]'::jsonb)
    ))
    FROM projected
$$;

-- Backfill only from the immutable Signal source payload, never from the
-- mutable tenant canonical profile. The latter may include applicant or
-- recruiter evidence and must not become a public profile by accident.
CREATE OR REPLACE FUNCTION activekg_assert_public_crustdata_backfill_safe()
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    conflicting_person_id BIGINT;
    conflicting_candidate_ids UUID[];
BEGIN
    WITH latest_public_source AS (
        SELECT DISTINCT ON (c.global_candidate_id)
            c.global_candidate_id,
            CASE
                WHEN (
                    activekg_public_crustdata_projection(csr.payload -> 'crustdata')
                    ->> 'crustdata_person_id'
                ) ~ '^[0-9]+$'
                THEN (
                    activekg_public_crustdata_projection(csr.payload -> 'crustdata')
                    ->> 'crustdata_person_id'
                )::bigint
                ELSE NULL
            END AS projected_person_id
        FROM candidates c
        JOIN candidate_source_records csr
          ON csr.candidate_id = c.candidate_id
         AND csr.tenant_id = c.tenant_id
        WHERE c.global_candidate_id IS NOT NULL
          AND csr.source = 'signal'
          AND jsonb_typeof(csr.payload -> 'crustdata') = 'object'
          AND csr.payload -> 'crustdata' <> '{}'::jsonb
        ORDER BY c.global_candidate_id, csr.updated_at DESC
    ),
    effective_ids AS (
        SELECT
            gc.id,
            CASE
                WHEN gc.public_profile = '{}'::jsonb
                    AND src.projected_person_id IS NOT NULL
                THEN src.projected_person_id
                ELSE gc.public_crustdata_person_id
            END AS person_id
        FROM global_candidates gc
        LEFT JOIN latest_public_source src ON src.global_candidate_id = gc.id
    )
    SELECT person_id, array_agg(id ORDER BY id)
      INTO conflicting_person_id, conflicting_candidate_ids
      FROM effective_ids
     WHERE person_id IS NOT NULL
     GROUP BY person_id
    HAVING count(*) > 1
     ORDER BY person_id
     LIMIT 1;

    IF conflicting_person_id IS NOT NULL THEN
        RAISE EXCEPTION USING
            ERRCODE = '23505',
            MESSAGE = format(
                'migration 021 refused: Crustdata person ID %s maps to multiple global candidates %s',
                conflicting_person_id,
                conflicting_candidate_ids
            ),
            HINT = 'Reconcile the conflicting global identities before retrying migration 021.';
    END IF;
END
$$;

SELECT activekg_assert_public_crustdata_backfill_safe();

WITH latest_public_source AS (
    SELECT DISTINCT ON (c.global_candidate_id)
        c.global_candidate_id,
        activekg_public_crustdata_projection(csr.payload -> 'crustdata') AS crustdata,
        COALESCE(csr.fetched_at, csr.updated_at, csr.created_at) AS observed_at
    FROM candidates c
    JOIN candidate_source_records csr
      ON csr.candidate_id = c.candidate_id
     AND csr.tenant_id = c.tenant_id
    WHERE c.global_candidate_id IS NOT NULL
      AND csr.source = 'signal'
      AND jsonb_typeof(csr.payload -> 'crustdata') = 'object'
      AND csr.payload -> 'crustdata' <> '{}'::jsonb
    ORDER BY c.global_candidate_id, csr.updated_at DESC
)
UPDATE global_candidates gc
SET public_profile = src.crustdata,
    public_profile_observed_at = src.observed_at,
    public_crustdata_person_id = CASE
        WHEN (src.crustdata ->> 'crustdata_person_id') ~ '^[0-9]+$'
        THEN (src.crustdata ->> 'crustdata_person_id')::bigint
        ELSE NULL
    END,
    public_headline = NULLIF(src.crustdata #>> '{basic_profile,headline}', ''),
    public_location_city = src.crustdata #>> '{basic_profile,location,city}',
    public_location_country_code = CASE
        WHEN length(src.crustdata #>> '{basic_profile,location,country_code}') = 2
            THEN upper(src.crustdata #>> '{basic_profile,location,country_code}')
        WHEN lower(src.crustdata #>> '{basic_profile,location,country}') = 'india' THEN 'IN'
        WHEN lower(src.crustdata #>> '{basic_profile,location,country}') IN
             ('united states', 'united states of america', 'usa') THEN 'US'
        WHEN lower(src.crustdata #>> '{basic_profile,location,country}') IN
             ('united kingdom', 'uk', 'great britain') THEN 'GB'
        WHEN lower(src.crustdata #>> '{basic_profile,location,country}') = 'canada' THEN 'CA'
        WHEN lower(src.crustdata #>> '{basic_profile,location,country}') = 'australia' THEN 'AU'
        WHEN lower(src.crustdata #>> '{basic_profile,location,country}') = 'germany' THEN 'DE'
        WHEN lower(src.crustdata #>> '{basic_profile,location,country}') = 'singapore' THEN 'SG'
        ELSE NULL
    END,
    public_seniority_band = src.crustdata #>> '{experience,employment_details,current,0,seniority_level}',
    public_skills_normalized = CASE
        WHEN jsonb_typeof(src.crustdata #> '{skills,professional_network_skills}') = 'array'
        THEN ARRAY(
            SELECT lower(trim(value))
            FROM jsonb_array_elements_text(src.crustdata #> '{skills,professional_network_skills}') value
            WHERE trim(value) <> ''
        )
        ELSE NULL
    END,
    public_embedding_status = 'queued',
    public_embed_version = 0
FROM latest_public_source src
WHERE gc.id = src.global_candidate_id
  AND gc.public_profile = '{}'::jsonb;

ALTER TABLE global_candidates
    DROP CONSTRAINT IF EXISTS global_candidates_public_headline_from_profile;
ALTER TABLE global_candidates
    ADD CONSTRAINT global_candidates_public_headline_from_profile
    CHECK (
        public_headline IS NOT DISTINCT FROM
        NULLIF(public_profile #>> '{basic_profile,headline}', '')
    );

CREATE UNIQUE INDEX IF NOT EXISTS idx_gc_public_crustdata_person_id
    ON global_candidates (public_crustdata_person_id)
    WHERE public_crustdata_person_id IS NOT NULL;

-- Public provenance may describe the public source class, never which tenant
-- paid for a search or which job/query caused it. Older Signal/web-discovery
-- rows embedded that private activity in a NULL-tenant (cross-tenant-readable)
-- source_detail; scrub it during the boundary migration.
UPDATE candidate_provenance
SET source_detail = '{}'::jsonb
WHERE tenant_id IS NULL
  AND source_type IN ('signal_sourced', 'web_discovery')
  AND source_detail <> '{}'::jsonb;

CREATE TABLE IF NOT EXISTS candidate_contact_evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    global_candidate_id UUID NOT NULL,
    tenant_id TEXT NOT NULL,
    email TEXT NOT NULL,
    email_hash TEXT NOT NULL,
    provider TEXT NOT NULL,
    provider_record_id TEXT,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    validated_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'found',
    suppressed_at TIMESTAMPTZ,
    bounce_reason TEXT,
    is_primary BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT candidate_contact_evidence_global_candidate_fkey
        FOREIGN KEY (global_candidate_id) REFERENCES global_candidates(id) ON DELETE CASCADE,
    CONSTRAINT candidate_contact_evidence_tenant_nonblank CHECK (btrim(tenant_id) <> ''),
    CONSTRAINT candidate_contact_evidence_email_nonblank CHECK (btrim(email) <> ''),
    CONSTRAINT candidate_contact_evidence_email_hash_nonblank CHECK (btrim(email_hash) <> ''),
    CONSTRAINT candidate_contact_evidence_provider_check
        CHECK (provider IN ('fullenrich', 'enrichlayer')),
    CONSTRAINT candidate_contact_evidence_confidence_check
        CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT candidate_contact_evidence_status_check
        CHECK (status IN ('found', 'verified', 'soft_bounce', 'hard_bounce', 'complaint', 'invalid')),
    CONSTRAINT candidate_contact_evidence_provider_record_required
        CHECK (provider_record_id IS NULL OR btrim(provider_record_id) <> ''),
    CONSTRAINT candidate_contact_evidence_primary_usable
        CHECK (NOT is_primary OR (status IN ('found', 'verified') AND suppressed_at IS NULL)),
    CONSTRAINT candidate_contact_evidence_unique
        UNIQUE (global_candidate_id, tenant_id, provider, email_hash)
);

CREATE INDEX IF NOT EXISTS idx_cce_tenant_global_primary
    ON candidate_contact_evidence (tenant_id, global_candidate_id, is_primary)
    WHERE status IN ('found', 'verified');
CREATE INDEX IF NOT EXISTS idx_cce_email_hash ON candidate_contact_evidence (email_hash);
CREATE UNIQUE INDEX IF NOT EXISTS idx_cce_one_primary
    ON candidate_contact_evidence (tenant_id, global_candidate_id)
    WHERE is_primary;

ALTER TABLE candidate_contact_evidence ENABLE ROW LEVEL SECURITY;
ALTER TABLE candidate_contact_evidence FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS tenant_isolation_candidate_contact_evidence ON candidate_contact_evidence;
CREATE POLICY tenant_isolation_candidate_contact_evidence
    ON candidate_contact_evidence FOR ALL TO PUBLIC
    USING (
        tenant_id = current_setting('app.current_tenant_id', true)::text
        AND tenant_id <> '__quarantine__'
    )
    WITH CHECK (
        tenant_id = current_setting('app.current_tenant_id', true)::text
        AND tenant_id <> '__quarantine__'
    );
DROP POLICY IF EXISTS admin_all_candidate_contact_evidence ON candidate_contact_evidence;
CREATE POLICY admin_all_candidate_contact_evidence
    ON candidate_contact_evidence FOR ALL TO admin_role
    USING (true)
    WITH CHECK (true);

-- A hard bounce or complaint is a fact about an address, not a recruiting
-- preference. Keep only its hash in the platform-wide tombstone so future
-- provider calls do not repurchase a known-bad address. Unsubscribe scope is
-- intentionally absent pending the product/legal decision.
CREATE TABLE IF NOT EXISTS contact_suppression_tombstones (
    email_hash TEXT PRIMARY KEY,
    global_candidate_id UUID,
    reason TEXT NOT NULL,
    first_observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_evidence_id UUID,
    provider_event_id TEXT,
    CONSTRAINT contact_suppression_source_evidence_fkey
        FOREIGN KEY (source_evidence_id) REFERENCES candidate_contact_evidence(id)
        ON DELETE SET NULL,
    CONSTRAINT contact_suppression_global_candidate_fkey
        FOREIGN KEY (global_candidate_id) REFERENCES global_candidates(id)
        ON DELETE SET NULL,
    CONSTRAINT contact_suppression_email_hash_nonblank CHECK (btrim(email_hash) <> ''),
    CONSTRAINT contact_suppression_reason_check CHECK (reason IN ('hard_bounce', 'complaint'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_contact_suppression_provider_event
    ON contact_suppression_tombstones (provider_event_id)
    WHERE provider_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_contact_suppression_global_candidate
    ON contact_suppression_tombstones (global_candidate_id)
    WHERE global_candidate_id IS NOT NULL;

-- Public market membership contains no tenant/job/activity fields. One public
-- person may be acquired under several canonical markets across time.
CREATE TABLE IF NOT EXISTS public_candidate_market_memberships (
    global_candidate_id UUID NOT NULL,
    coarse_market_key TEXT NOT NULL,
    role_family TEXT NOT NULL,
    location_city TEXT NOT NULL,
    location_country_code TEXT NOT NULL,
    seniority_band TEXT NOT NULL,
    first_observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT public_candidate_market_global_candidate_fkey
        FOREIGN KEY (global_candidate_id) REFERENCES global_candidates(id) ON DELETE CASCADE,
    CONSTRAINT public_candidate_market_key_nonblank CHECK (btrim(coarse_market_key) <> ''),
    CONSTRAINT public_candidate_market_role_nonblank CHECK (btrim(role_family) <> ''),
    CONSTRAINT public_candidate_market_city_nonblank CHECK (btrim(location_city) <> ''),
    CONSTRAINT public_candidate_market_country_code_check
        CHECK (location_country_code ~ '^[A-Z]{2}$'),
    CONSTRAINT public_candidate_market_seniority_nonblank CHECK (btrim(seniority_band) <> ''),
    CONSTRAINT public_candidate_market_memberships_pkey
        PRIMARY KEY (global_candidate_id, coarse_market_key)
);

CREATE INDEX IF NOT EXISTS idx_pcmm_market_last_observed
    ON public_candidate_market_memberships (coarse_market_key, last_observed_at DESC);
