"""Global candidate memory endpoints.

Provides CRUD operations for cross-tenant candidate memory:
global_candidates, candidate_provenance, tenant_candidate_access, feedback_events.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from activekg.api.auth import get_jwt_claims, require_scope
from activekg.common.logger import get_enhanced_logger

logger = get_enhanced_logger(__name__)

router = APIRouter(tags=["global-memory"])

_DSN = os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL", "")

GLOBAL_MEMORY_ENABLED = os.getenv("GLOBAL_MEMORY_ENABLED", "false").lower() == "true"


# ---------------------------------------------------------------------------
# Country name → ISO 3166-1 alpha-2 normalizer
# ---------------------------------------------------------------------------

_COUNTRY_NAME_TO_CODE: dict[str, str] = {
    "united states": "US", "united states of america": "US", "usa": "US", "us": "US",
    "united kingdom": "GB", "uk": "GB", "great britain": "GB", "england": "GB",
    "india": "IN", "canada": "CA", "australia": "AU", "germany": "DE", "deutschland": "DE",
    "france": "FR", "brazil": "BR", "brasil": "BR", "japan": "JP", "china": "CN",
    "south korea": "KR", "korea": "KR", "republic of korea": "KR",
    "israel": "IL", "singapore": "SG", "netherlands": "NL", "holland": "NL",
    "sweden": "SE", "norway": "NO", "denmark": "DK", "finland": "FI",
    "ireland": "IE", "switzerland": "CH", "austria": "AT", "belgium": "BE",
    "spain": "ES", "italy": "IT", "portugal": "PT", "poland": "PL",
    "czech republic": "CZ", "czechia": "CZ", "romania": "RO", "hungary": "HU",
    "turkey": "TR", "türkiye": "TR", "mexico": "MX", "argentina": "AR",
    "colombia": "CO", "chile": "CL", "peru": "PE",
    "south africa": "ZA", "nigeria": "NG", "kenya": "KE", "egypt": "EG",
    "united arab emirates": "AE", "uae": "AE", "saudi arabia": "SA",
    "indonesia": "ID", "malaysia": "MY", "philippines": "PH", "vietnam": "VN",
    "thailand": "TH", "taiwan": "TW", "hong kong": "HK",
    "new zealand": "NZ", "pakistan": "PK", "bangladesh": "BD", "sri lanka": "LK",
    "ukraine": "UA", "russia": "RU", "russian federation": "RU",
    "estonia": "EE", "latvia": "LV", "lithuania": "LT",
    "croatia": "HR", "serbia": "RS", "bulgaria": "BG", "greece": "GR",
    "luxembourg": "LU", "iceland": "IS", "costa rica": "CR", "uruguay": "UY",
    "ghana": "GH", "ethiopia": "ET", "morocco": "MA", "tunisia": "TN",
}

_ISO_ALPHA2 = re.compile(r"^[A-Z]{2}$")


# ---------------------------------------------------------------------------
# Extraction function tag → Signal canonical role_family normalizer
# ---------------------------------------------------------------------------

_EXTRACTION_TO_ROLE_FAMILY: dict[str, str] = {
    # Direct matches (case-normalized)
    "backend": "backend",
    "frontend": "frontend",
    "fullstack": "fullstack",
    "devops": "devops",
    "data": "data",
    "qa": "qa",
    "security": "security",
    "mobile": "mobile",
    # Extraction tags that map to Signal families
    "ml": "data",
    "machine learning": "data",
    "ai": "data",
    "analytics": "data",
    "data engineering": "data",
    "infrastructure": "devops",
    "sre": "devops",
    "platform": "devops",
    "ios": "mobile",
    "android": "mobile",
    "testing": "qa",
    "quality assurance": "qa",
}


def _normalize_role_family(raw: str | None) -> str | None:
    """Normalize extraction function tag to Signal's canonical role_family."""
    if not raw or not raw.strip():
        return None
    val = raw.strip().lower()
    mapped = _EXTRACTION_TO_ROLE_FAMILY.get(val)
    if mapped:
        return mapped
    # If already a valid Signal role family (e.g. non-tech families), pass through
    _SIGNAL_ROLE_FAMILIES = {
        "backend", "frontend", "fullstack", "devops", "data", "qa", "security", "mobile",
        "technical_account_manager", "sales_engineer", "customer_success",
        "account_executive", "business_development", "account_manager",
    }
    if val in _SIGNAL_ROLE_FAMILIES:
        return val
    # Unknown tag — store as-is but log for visibility
    logger.warning("Unmapped role_family tag, storing as-is", extra_fields={"raw_role_family": raw})
    return val


def _normalize_country_code(raw: str | None) -> str | None:
    """Convert free-text country name to ISO 3166-1 alpha-2 code. Returns None if unrecognized."""
    if not raw or not raw.strip():
        return None
    val = raw.strip()
    # Already a 2-letter ISO code
    if _ISO_ALPHA2.match(val.upper()):
        return val.upper()
    code = _COUNTRY_NAME_TO_CODE.get(val.lower())
    if code:
        return code
    logger.warning("Unrecognized country name, storing NULL", extra_fields={"raw_country": val})
    return None


def _get_conn():
    return psycopg.connect(_DSN, autocommit=True)


def _get_tenant_conn(tenant_id: str | None):
    """Get a transactional connection with RLS tenant context set.

    RLS on candidate_provenance, tenant_candidate_access, and feedback_events
    requires app.current_tenant_id. This helper creates a non-autocommit connection
    and sets the tenant context via set_config() (scoped to the transaction).
    Caller must commit/rollback and close.
    """
    conn = psycopg.connect(_DSN, autocommit=False)
    if tenant_id:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('app.current_tenant_id', %s, true)",
                (tenant_id,),
            )
    return conn


def _require_enabled():
    if not GLOBAL_MEMORY_ENABLED:
        raise HTTPException(status_code=503, detail="Global memory feature is disabled")


def _validate_tenant(claims, body_tenant_id: str | None) -> None:
    """Ensure body tenant_id matches JWT claims. Prevents cross-tenant writes."""
    if body_tenant_id is None:
        return  # public provenance (no tenant)
    if claims.tenant_id != body_tenant_id:
        raise HTTPException(
            status_code=403,
            detail=f"Tenant mismatch: JWT tenant_id={claims.tenant_id!r}, body tenant_id={body_tenant_id!r}",
        )


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class GlobalCandidateUpsert(BaseModel):
    linkedin_id: str | None = None
    linkedin_url: str | None = None
    github_id: str | None = None
    email_hash: str | None = None
    name: str | None = None
    headline: str | None = None
    location_city: str | None = None
    location_country_code: str | None = None
    location_confidence: float | None = None
    location_source: str | None = None
    role_family: str | None = None
    seniority_band: str | None = None
    skills_normalized: list[str] | None = None
    identity_confidence: float | None = None
    merge_status: str = "single"


class ProvenanceCreate(BaseModel):
    source_type: str
    tenant_id: str | None = None
    source_detail: dict = {}


class AccessUpsert(BaseModel):
    tenant_id: str
    visibility: str
    consent_state: str | None = None
    access_reason: str


class FeedbackEvent(BaseModel):
    tenant_id: str
    job_id: str
    recruiter_id: str | None = None
    global_candidate_id: str | None = None
    signal_candidate_id: str | None = None
    action: str
    rank_at_time: int | None = None
    fit_score_at_time: float | None = None
    source_type_at_time: str | None = None
    match_tier_at_time: str | None = None
    location_match_at_time: str | None = None
    role_family: str | None = None
    location_country_code: str | None = None
    seniority_band: str | None = None
    event_id: str


class FeedbackEventIngest(BaseModel):
    events: list[FeedbackEvent]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fields on global_candidates that can be set/updated (excluding id, timestamps, embedding).
_CANDIDATE_FIELDS = [
    "linkedin_id",
    "linkedin_url",
    "github_id",
    "email_hash",
    "name",
    "headline",
    "location_city",
    "location_country_code",
    "location_confidence",
    "location_source",
    "role_family",
    "seniority_band",
    "skills_normalized",
    "identity_confidence",
    "merge_status",
]

# Fields that always overwrite on update (identity anchors + merge control).
# All other _CANDIDATE_FIELDS use COALESCE (non-destructive merge).
_ALWAYS_OVERWRITE_FIELDS = {
    "linkedin_id", "linkedin_url", "github_id", "email_hash",
    "identity_confidence", "merge_status",
}


def _find_existing(
    cur: psycopg.Cursor, body: GlobalCandidateUpsert
) -> dict[str, Any] | None:
    """Lookup existing global_candidate by anchor priority: linkedin_id > github_id > email_hash."""
    for anchor, value in [
        ("linkedin_id", body.linkedin_id),
        ("github_id", body.github_id),
        ("email_hash", body.email_hash),
    ]:
        if value is None:
            continue
        cur.execute(
            f"SELECT * FROM global_candidates WHERE {anchor} = %s LIMIT 1",
            (value,),
        )
        row = cur.fetchone()
        if row:
            cols = [d.name for d in cur.description]
            return dict(zip(cols, row))
    return None


def _row_to_dict(cur: psycopg.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d.name for d in cur.description]
    result = dict(zip(cols, row))
    # Serialize non-JSON-native types for the response.
    for k, v in result.items():
        if hasattr(v, "isoformat"):
            result[k] = v.isoformat()
        elif isinstance(v, bytes):
            result[k] = v.hex()
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/global-candidates/upsert",
    dependencies=[Depends(require_scope("kg:write"))],
)
def upsert_global_candidate(
    body: GlobalCandidateUpsert,
    claims=Depends(get_jwt_claims),
):
    _require_enabled()

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            existing = _find_existing(cur, body)

            if existing:
                # Non-destructive merge: identity/merge-control fields always
                # overwrite; profile fields use COALESCE so richer data is not
                # clobbered by a sparser evidence stream.
                updates: list[str] = []
                params: list[Any] = []
                for field in _CANDIDATE_FIELDS:
                    val = getattr(body, field)
                    if val is not None:
                        if field in _ALWAYS_OVERWRITE_FIELDS:
                            updates.append(f"{field} = %s")
                        else:
                            updates.append(f"{field} = COALESCE({field}, %s)")
                        params.append(val)

                if updates:
                    updates.append("last_evidence_at = now()")
                    updates.append("updated_at = now()")
                    params.append(existing["id"])
                    cur.execute(
                        f"UPDATE global_candidates SET {', '.join(updates)} WHERE id = %s",
                        params,
                    )

                candidate_id = str(existing["id"])
                logger.info(
                    "Global candidate updated",
                    extra_fields={"global_candidate_id": candidate_id},
                )
                return {"global_candidate_id": candidate_id, "action": "updated"}
            else:
                # Insert new record.
                cols: list[str] = []
                placeholders: list[str] = []
                params = []
                for field in _CANDIDATE_FIELDS:
                    val = getattr(body, field)
                    if val is not None:
                        cols.append(field)
                        placeholders.append("%s")
                        params.append(val)

                cols_str = ", ".join(cols) if cols else ""
                ph_str = ", ".join(placeholders) if placeholders else ""

                if cols:
                    cur.execute(
                        f"INSERT INTO global_candidates ({cols_str}) VALUES ({ph_str}) RETURNING id",
                        params,
                    )
                else:
                    cur.execute(
                        "INSERT INTO global_candidates DEFAULT VALUES RETURNING id"
                    )

                new_id = str(cur.fetchone()[0])
                logger.info(
                    "Global candidate created",
                    extra_fields={"global_candidate_id": new_id},
                )
                return {"global_candidate_id": new_id, "action": "created"}
    finally:
        conn.close()


@router.get(
    "/global-candidates/by-anchor",
    dependencies=[Depends(require_scope("kg:read"))],
)
def get_by_anchor(
    linkedin_id: str | None = Query(None),
    github_id: str | None = Query(None),
    email_hash: str | None = Query(None),
    claims=Depends(get_jwt_claims),
):
    _require_enabled()

    if not any([linkedin_id, github_id, email_hash]):
        raise HTTPException(
            status_code=400,
            detail="At least one anchor query param required (linkedin_id, github_id, email_hash)",
        )

    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            for anchor, value in [
                ("linkedin_id", linkedin_id),
                ("github_id", github_id),
                ("email_hash", email_hash),
            ]:
                if value is None:
                    continue
                cur.execute(
                    f"SELECT * FROM global_candidates WHERE {anchor} = %s LIMIT 1",
                    (value,),
                )
                row = cur.fetchone()
                if row:
                    return _row_to_dict(cur, row)

            raise HTTPException(status_code=404, detail="Candidate not found")
    finally:
        conn.close()


@router.post(
    "/global-candidates/{candidate_id}/provenance",
    dependencies=[Depends(require_scope("kg:write"))],
)
def create_provenance(
    candidate_id: str,
    body: ProvenanceCreate,
    claims=Depends(get_jwt_claims),
):
    _require_enabled()
    _validate_tenant(claims, body.tenant_id)

    # Use JWT tenant_id for RLS context (not body.tenant_id, which may be None
    # for public provenance). The RLS policies allow tenant_id IS NULL rows
    # when the caller has a valid tenant context.
    rls_tenant = claims.tenant_id if claims else body.tenant_id
    conn = _get_tenant_conn(rls_tenant)
    try:
        with conn.cursor() as cur:
            if body.tenant_id is None:
                # NULL tenant_id: use partial unique index for idempotency
                cur.execute(
                    """
                    INSERT INTO candidate_provenance
                        (global_candidate_id, source_type, tenant_id, source_detail)
                    SELECT %s, %s, NULL, %s::jsonb
                    WHERE NOT EXISTS (
                        SELECT 1 FROM candidate_provenance
                        WHERE global_candidate_id = %s AND source_type = %s AND tenant_id IS NULL
                    )
                    RETURNING id
                    """,
                    (
                        candidate_id,
                        body.source_type,
                        json.dumps(body.source_detail),
                        candidate_id,
                        body.source_type,
                    ),
                )
                row = cur.fetchone()
                if row:
                    prov_id = str(row[0])
                else:
                    # Already exists — update source_detail
                    cur.execute(
                        """
                        UPDATE candidate_provenance
                        SET source_detail = %s::jsonb
                        WHERE global_candidate_id = %s AND source_type = %s AND tenant_id IS NULL
                        RETURNING id
                        """,
                        (json.dumps(body.source_detail), candidate_id, body.source_type),
                    )
                    prov_id = str(cur.fetchone()[0])
            else:
                cur.execute(
                    """
                    INSERT INTO candidate_provenance
                        (global_candidate_id, source_type, tenant_id, source_detail)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (global_candidate_id, source_type, tenant_id)
                    DO UPDATE SET source_detail = EXCLUDED.source_detail
                    RETURNING id
                    """,
                    (
                        candidate_id,
                        body.source_type,
                        body.tenant_id,
                        json.dumps(body.source_detail),
                    ),
                )
                prov_id = str(cur.fetchone()[0])
        conn.commit()
        logger.info(
            "Provenance upserted",
            extra_fields={
                "provenance_id": prov_id,
                "global_candidate_id": candidate_id,
                "source_type": body.source_type,
            },
        )
        return {"provenance_id": prov_id, "global_candidate_id": candidate_id}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post(
    "/global-candidates/{candidate_id}/access",
    dependencies=[Depends(require_scope("kg:write"))],
)
def upsert_access(
    candidate_id: str,
    body: AccessUpsert,
    claims=Depends(get_jwt_claims),
):
    _require_enabled()
    _validate_tenant(claims, body.tenant_id)

    conn = _get_tenant_conn(body.tenant_id)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tenant_candidate_access
                    (tenant_id, global_candidate_id, visibility, consent_state, access_reason)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (tenant_id, global_candidate_id)
                DO UPDATE SET
                    visibility = EXCLUDED.visibility,
                    consent_state = EXCLUDED.consent_state,
                    access_reason = EXCLUDED.access_reason
                RETURNING id
                """,
                (
                    body.tenant_id,
                    candidate_id,
                    body.visibility,
                    body.consent_state,
                    body.access_reason,
                ),
            )
            access_id = str(cur.fetchone()[0])
        conn.commit()
        logger.info(
            "Access upserted",
            extra_fields={
                "access_id": access_id,
                "tenant_id": body.tenant_id,
                "global_candidate_id": candidate_id,
            },
        )
        return {"access_id": access_id, "global_candidate_id": candidate_id}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post(
    "/feedback-events/ingest",
    dependencies=[Depends(require_scope("kg:write"))],
)
def ingest_feedback_events(
    body: FeedbackEventIngest,
    claims=Depends(get_jwt_claims),
):
    _require_enabled()

    if not body.events:
        return {"inserted": 0, "skipped": 0}

    # All events in a batch share the same tenant_id (Vanta forward-sync sends per-tenant batches)
    tenant_id = body.events[0].tenant_id
    _validate_tenant(claims, tenant_id)
    conn = _get_tenant_conn(tenant_id)
    inserted = 0
    skipped = 0
    try:
        with conn.cursor() as cur:
            for ev in body.events:
                cur.execute(
                    """
                    INSERT INTO feedback_events
                        (tenant_id, job_id, recruiter_id, global_candidate_id,
                         signal_candidate_id, action, rank_at_time, fit_score_at_time,
                         source_type_at_time, match_tier_at_time, location_match_at_time,
                         role_family, location_country_code, seniority_band, event_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (event_id) DO NOTHING
                    """,
                    (
                        ev.tenant_id,
                        ev.job_id,
                        ev.recruiter_id,
                        ev.global_candidate_id,
                        ev.signal_candidate_id,
                        ev.action,
                        ev.rank_at_time,
                        ev.fit_score_at_time,
                        ev.source_type_at_time,
                        ev.match_tier_at_time,
                        ev.location_match_at_time,
                        ev.role_family,
                        ev.location_country_code,
                        ev.seniority_band,
                        ev.event_id,
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1

        conn.commit()
        logger.info(
            "Feedback events ingested",
            extra_fields={"inserted": inserted, "skipped": skipped},
        )
        return {"inserted": inserted, "skipped": skipped}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Post-extraction hook: sync platform applicant to global_candidates
# Called from extraction worker, NOT from an HTTP endpoint.
# ---------------------------------------------------------------------------


def sync_applicant_to_global_memory(
    *,
    node_id: str,
    tenant_id: str | None,
    node_props: dict[str, Any],
    extracted_result: Any,
    metadata: dict[str, Any],
) -> None:
    """Sync a platform applicant resume node to global_candidates after extraction.

    Called by the extraction worker when:
    - GLOBAL_MEMORY_ENABLED is true
    - Node metadata has provenance_type = 'platform_applicant'
    """
    import hashlib

    # Build candidate fields from extraction result
    location = None
    if hasattr(extracted_result, "location") and extracted_result.location:
        location = extracted_result.location

    email_raw = node_props.get("applicant_email") or metadata.get("applicant_email")
    email_hash = None
    if email_raw and isinstance(email_raw, str):
        email_hash = hashlib.sha256(email_raw.strip().lower().encode()).hexdigest()

    name = node_props.get("applicant_name") or metadata.get("applicant_name")

    # Map extraction fields (normalize to Signal's canonical taxonomy)
    role_family = None
    if hasattr(extracted_result, "functions") and extracted_result.functions:
        role_family = _normalize_role_family(extracted_result.functions[0])

    seniority_band = None
    if hasattr(extracted_result, "seniority") and extracted_result.seniority:
        seniority_band = extracted_result.seniority

    skills: list[str] | None = None
    if hasattr(extracted_result, "skills_normalized") and extracted_result.skills_normalized:
        skills = list(extracted_result.skills_normalized)

    location_city = location.city if location and hasattr(location, "city") else None
    location_country_raw = location.country if location and hasattr(location, "country") else None
    location_country = _normalize_country_code(location_country_raw)

    conn = _get_tenant_conn(tenant_id)
    try:
        with conn.cursor() as cur:
            # Upsert global_candidates by email_hash (primary anchor for applicants)
            existing = None
            if email_hash:
                cur.execute(
                    "SELECT id FROM global_candidates WHERE email_hash = %s LIMIT 1",
                    (email_hash,),
                )
                row = cur.fetchone()
                if row:
                    existing = str(row[0])

            if existing:
                # Non-destructive merge: profile fields use COALESCE
                sets = ["last_evidence_at = now()", "updated_at = now()"]
                params: list[Any] = []
                for col, val in [
                    ("name", name),
                    ("role_family", role_family),
                    ("seniority_band", seniority_band),
                    ("skills_normalized", skills),
                    ("location_city", location_city),
                    ("location_country_code", location_country),
                ]:
                    if val is not None:
                        sets.append(f"{col} = COALESCE({col}, %s)")
                        params.append(val)

                params.append(existing)
                cur.execute(
                    f"UPDATE global_candidates SET {', '.join(sets)} WHERE id = %s",
                    params,
                )
                gc_id = existing
            else:
                # Insert new
                cur.execute(
                    """
                    INSERT INTO global_candidates
                        (email_hash, name, role_family, seniority_band, skills_normalized,
                         location_city, location_country_code, identity_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        email_hash,
                        name,
                        role_family,
                        seniority_band,
                        skills,
                        location_city,
                        location_country,
                        0.5,  # Moderate confidence for email-only anchor
                    ),
                )
                gc_id = str(cur.fetchone()[0])

            # Upsert provenance
            application_id = metadata.get("application_id")
            job_id = metadata.get("job_id")
            org_id = metadata.get("org_id")
            cur.execute(
                """
                INSERT INTO candidate_provenance
                    (global_candidate_id, source_type, tenant_id, source_detail)
                VALUES (%s, 'platform_applicant', %s, %s::jsonb)
                ON CONFLICT (global_candidate_id, source_type, tenant_id)
                DO UPDATE SET source_detail = EXCLUDED.source_detail
                """,
                (
                    gc_id,
                    tenant_id,
                    json.dumps({
                        "application_id": str(application_id) if application_id else None,
                        "job_id": str(job_id) if job_id else None,
                        "org_id": str(org_id) if org_id else None,
                        "resume_node_id": node_id,
                    }),
                ),
            )

            # Upsert tenant access
            consent_state = metadata.get("consent_state", "opted_out")
            visibility = metadata.get("visibility", "private")
            if tenant_id:
                cur.execute(
                    """
                    INSERT INTO tenant_candidate_access
                        (tenant_id, global_candidate_id, visibility, consent_state, access_reason)
                    VALUES (%s, %s, %s, %s, 'platform_applicant')
                    ON CONFLICT (tenant_id, global_candidate_id)
                    DO UPDATE SET
                        visibility = EXCLUDED.visibility,
                        consent_state = EXCLUDED.consent_state
                    """,
                    (tenant_id, gc_id, visibility, consent_state),
                )

        conn.commit()
        logger.info(
            "Applicant synced to global memory",
            extra_fields={
                "global_candidate_id": gc_id,
                "node_id": node_id,
                "tenant_id": tenant_id,
            },
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
