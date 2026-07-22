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
from activekg.embedding.global_candidates import (  # noqa: F401 — shared with the embedding producer
    build_candidate_embedding_text,
)

logger = get_enhanced_logger(__name__)

router = APIRouter(tags=["global-memory"])

_DSN = os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL", "")

GLOBAL_MEMORY_ENABLED = os.getenv("GLOBAL_MEMORY_ENABLED", "false").lower() == "true"

# Authoritative cap for /global-candidates/search. Callers may ask for less;
# asking for more is clamped and reported back via applied_limit.
_SEARCH_LIMIT_MAX = int(os.getenv("GLOBAL_SEARCH_LIMIT_MAX", "500"))


# ---------------------------------------------------------------------------
# Country name → ISO 3166-1 alpha-2 normalizer
# ---------------------------------------------------------------------------

_COUNTRY_NAME_TO_CODE: dict[str, str] = {
    "united states": "US",
    "united states of america": "US",
    "usa": "US",
    "us": "US",
    "united kingdom": "GB",
    "uk": "GB",
    "great britain": "GB",
    "england": "GB",
    "india": "IN",
    "canada": "CA",
    "australia": "AU",
    "germany": "DE",
    "deutschland": "DE",
    "france": "FR",
    "brazil": "BR",
    "brasil": "BR",
    "japan": "JP",
    "china": "CN",
    "south korea": "KR",
    "korea": "KR",
    "republic of korea": "KR",
    "israel": "IL",
    "singapore": "SG",
    "netherlands": "NL",
    "holland": "NL",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "ireland": "IE",
    "switzerland": "CH",
    "austria": "AT",
    "belgium": "BE",
    "spain": "ES",
    "italy": "IT",
    "portugal": "PT",
    "poland": "PL",
    "czech republic": "CZ",
    "czechia": "CZ",
    "romania": "RO",
    "hungary": "HU",
    "turkey": "TR",
    "türkiye": "TR",
    "mexico": "MX",
    "argentina": "AR",
    "colombia": "CO",
    "chile": "CL",
    "peru": "PE",
    "south africa": "ZA",
    "nigeria": "NG",
    "kenya": "KE",
    "egypt": "EG",
    "united arab emirates": "AE",
    "uae": "AE",
    "saudi arabia": "SA",
    "indonesia": "ID",
    "malaysia": "MY",
    "philippines": "PH",
    "vietnam": "VN",
    "thailand": "TH",
    "taiwan": "TW",
    "hong kong": "HK",
    "new zealand": "NZ",
    "pakistan": "PK",
    "bangladesh": "BD",
    "sri lanka": "LK",
    "ukraine": "UA",
    "russia": "RU",
    "russian federation": "RU",
    "estonia": "EE",
    "latvia": "LV",
    "lithuania": "LT",
    "croatia": "HR",
    "serbia": "RS",
    "bulgaria": "BG",
    "greece": "GR",
    "luxembourg": "LU",
    "iceland": "IS",
    "costa rica": "CR",
    "uruguay": "UY",
    "ghana": "GH",
    "ethiopia": "ET",
    "morocco": "MA",
    "tunisia": "TN",
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
        "backend",
        "frontend",
        "fullstack",
        "devops",
        "data",
        "qa",
        "security",
        "mobile",
        "technical_account_manager",
        "sales_engineer",
        "customer_success",
        "account_executive",
        "business_development",
        "account_manager",
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
    "linkedin_id",
    "linkedin_url",
    "github_id",
    "email_hash",
    "identity_confidence",
    "merge_status",
}


_LINKEDIN_SLUG_RE = re.compile(r"linkedin\.com/in/([^/?#]+)", re.IGNORECASE)


def linkedin_id_from_url(url: str | None) -> str | None:
    """Canonical linkedin_id = lowercased /in/ slug. ONE normalizer for every
    write path — tenant identifiers, applicant sync, and Signal ingest must
    agree on this value or the same person lands in different rows."""
    if not url:
        return None
    m = _LINKEDIN_SLUG_RE.search(url)
    return m.group(1).lower() if m else None


def _find_existing_all(
    cur: psycopg.Cursor,
    linkedin_id: str | None,
    github_id: str | None,
    email_hash: str | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Look up ALL anchor matches, priority-ordered linkedin > github > email.

    Returns (primary, extras). The old first-match-only lookup meant a person
    sourced via linkedin (row A) who later applied via email (row B) matched A,
    then stamping B's email_hash onto A violated the partial-unique anchor
    index — the sourced-then-applied flywheel case crashed the upsert.
    """
    seen_ids: set[str] = set()
    matches: list[dict[str, Any]] = []
    for anchor, value in [
        ("linkedin_id", linkedin_id),
        ("github_id", github_id),
        ("email_hash", email_hash),
    ]:
        if value is None:
            continue
        cur.execute(
            f"SELECT * FROM global_candidates WHERE {anchor} = %s LIMIT 1",  # noqa: S608 — anchor from fixed list
            (value,),
        )
        row = cur.fetchone()
        if row:
            cols = [d.name for d in cur.description]
            d = dict(zip(cols, row, strict=False))
            rid = str(d["id"])
            if rid not in seen_ids:
                seen_ids.add(rid)
                matches.append(d)
    if not matches:
        return None, []
    return matches[0], matches[1:]


def _enqueue_merge(
    cur: psycopg.Cursor,
    a_id: str,
    b_id: str | None,
    tenant_id: str | None,
    reason: str,
    details: dict[str, Any],
) -> None:
    """Persist an identity conflict as a durable work item (idempotent on the
    open (pair, reason) via the partial-unique index)."""
    cur.execute(
        """
        INSERT INTO candidate_merge_queue
            (global_candidate_id_a, global_candidate_id_b, tenant_id, reason, details)
        VALUES (%s, %s, %s, %s, %s::jsonb)
        ON CONFLICT DO NOTHING
        """,
        (a_id, b_id, tenant_id, reason, json.dumps(details)),
    )


def _names_conflict(existing_name: str | None, incoming_name: str | None) -> bool:
    """Sanity guard for weak-anchor (email-only) matches: shared/fake emails
    (careers@agency.com) must not silently COALESCE-merge different humans.
    Conservative: only flags when both names exist and share no token."""
    if not existing_name or not incoming_name:
        return False
    a = {t for t in re.split(r"[^a-z]+", existing_name.lower()) if len(t) > 1}
    b = {t for t in re.split(r"[^a-z]+", incoming_name.lower()) if len(t) > 1}
    if not a or not b:
        return False
    return not (a & b)


def _find_existing(cur: psycopg.Cursor, body: GlobalCandidateUpsert) -> dict[str, Any] | None:
    """Back-compat single-match lookup (primary anchor only)."""
    primary, _ = _find_existing_all(cur, body.linkedin_id, body.github_id, body.email_hash)
    return primary


def _row_to_dict(cur: psycopg.Cursor, row: tuple) -> dict[str, Any]:
    cols = [d.name for d in cur.description]
    result = dict(zip(cols, row, strict=False))
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
            existing, extras = _find_existing_all(
                cur, body.linkedin_id, body.github_id, body.email_hash
            )

            # Cross-anchor conflict: this evidence bridges >1 existing row
            # (e.g. sourced-by-linkedin row + applied-by-email row = same
            # human). Queue a needs_merge item and mark the primary; do NOT
            # stamp anchors owned by the other row — that violates the
            # partial-unique anchor indexes (crash) or steals identity.
            conflicted_anchors: set[str] = set()
            if existing and extras:
                for extra in extras:
                    for anchor in ("linkedin_id", "github_id", "email_hash"):
                        if extra.get(anchor) and getattr(body, anchor) == extra.get(anchor):
                            conflicted_anchors.add(anchor)
                    _enqueue_merge(
                        cur,
                        str(existing["id"]),
                        str(extra["id"]),
                        None,
                        "needs_merge",
                        {"bridging_anchors": sorted(conflicted_anchors), "source": "upsert"},
                    )
                cur.execute(
                    "UPDATE global_candidates SET merge_status = 'needs_merge' WHERE id IN (%s, %s)",
                    (existing["id"], extras[0]["id"]),
                )

            # Weak-anchor sanity guard: matched by email only + names disjoint
            # → likely a shared mailbox, not the same person. Queue for review
            # and skip profile fills so we never blend two humans.
            weak_match_suspect = bool(
                existing
                and not extras
                and body.email_hash
                and existing.get("email_hash") == body.email_hash
                and (body.linkedin_id is None or existing.get("linkedin_id") != body.linkedin_id)
                and (body.github_id is None or existing.get("github_id") != body.github_id)
                and _names_conflict(existing.get("name"), body.name)
            )
            if existing and weak_match_suspect:
                _enqueue_merge(
                    cur,
                    str(existing["id"]),
                    None,
                    None,
                    "review_required",
                    {
                        "existing_name": existing.get("name"),
                        "incoming_name": body.name,
                        "anchor": "email_hash",
                    },
                )

            if existing:
                # Non-destructive merge: identity/merge-control fields always
                # overwrite (except anchors owned by a conflicting row);
                # profile fields use COALESCE so richer data is not clobbered
                # by a sparser evidence stream. Suspect weak matches attach no
                # profile data at all.
                updates: list[str] = []
                params: list[Any] = []
                for field in _CANDIDATE_FIELDS:
                    val = getattr(body, field)
                    if val is not None:
                        if field in conflicted_anchors:
                            continue
                        if weak_match_suspect:
                            # Doubtful identity: record evidence timestamps only —
                            # neither profile fills nor new anchors attach.
                            continue
                        if field in _ALWAYS_OVERWRITE_FIELDS:
                            updates.append(f"{field} = %s")
                        else:
                            updates.append(f"{field} = COALESCE({field}, %s)")
                        params.append(val)

                if updates:
                    updates.append("last_evidence_at = now()")
                    updates.append("embedding_status = 'queued'")  # re-embed on new evidence
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
                    cur.execute("INSERT INTO global_candidates DEFAULT VALUES RETURNING id")

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

    # LinkedIn anchor from the resume links Flow already extracts. Without it,
    # applicants (email-anchored) and Signal-sourced rows (linkedin-anchored)
    # could never converge on one profile — guaranteed duplicates.
    linkedin_url_raw = node_props.get("linkedin_url") or metadata.get("linkedin_url")
    li_id = linkedin_id_from_url(linkedin_url_raw if isinstance(linkedin_url_raw, str) else None)

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
            # Multi-anchor lookup (linkedin > email). Cross-anchor hits queue a
            # needs_merge item — the sourced-then-applied case must converge,
            # not duplicate or crash on the unique anchor indexes.
            existing_row, extras = _find_existing_all(cur, li_id, None, email_hash)
            for extra in extras:
                _enqueue_merge(
                    cur,
                    str(existing_row["id"]),
                    str(extra["id"]),
                    tenant_id,
                    "needs_merge",
                    {"source": "applicant_sync", "node_id": node_id},
                )

            if existing_row:
                gc_id = str(existing_row["id"])
                # Non-destructive merge: profile fields use COALESCE; skills
                # are UNION-merged (a new resume adds evidence, COALESCE would
                # freeze the first-ever list); missing anchors are filled.
                sets = [
                    "last_evidence_at = now()",
                    "updated_at = now()",
                    "embedding_status = 'queued'",
                ]  # re-embed: profile evidence changed
                params: list[Any] = []
                for col, val in [
                    ("name", name),
                    ("role_family", role_family),
                    ("seniority_band", seniority_band),
                    ("location_city", location_city),
                    ("location_country_code", location_country),
                ]:
                    if val is not None:
                        sets.append(f"{col} = COALESCE({col}, %s)")
                        params.append(val)
                if skills:
                    sets.append(
                        "skills_normalized = ARRAY(SELECT DISTINCT unnest(COALESCE(skills_normalized, ARRAY[]::text[]) || %s::text[]))"
                    )
                    params.append(skills)
                if email_hash and not existing_row.get("email_hash"):
                    sets.append("email_hash = %s")
                    params.append(email_hash)
                if li_id and not existing_row.get("linkedin_id"):
                    sets.append("linkedin_id = %s")
                    params.append(li_id)
                    if isinstance(linkedin_url_raw, str):
                        sets.append("linkedin_url = COALESCE(linkedin_url, %s)")
                        params.append(linkedin_url_raw)

                params.append(gc_id)
                cur.execute(
                    f"UPDATE global_candidates SET {', '.join(sets)} WHERE id = %s",
                    params,
                )
            else:
                # Insert new
                cur.execute(
                    """
                    INSERT INTO global_candidates
                        (email_hash, linkedin_id, linkedin_url, name, role_family,
                         seniority_band, skills_normalized,
                         location_city, location_country_code, identity_confidence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        email_hash,
                        li_id,
                        linkedin_url_raw if isinstance(linkedin_url_raw, str) else None,
                        name,
                        role_family,
                        seniority_band,
                        skills,
                        location_city,
                        location_country,
                        0.7 if li_id else 0.5,  # linkedin anchor is stronger evidence
                    ),
                )
                gc_id = str(cur.fetchone()[0])

            # Upsert provenance. provenance_type passthrough: candidate-submitted
            # applications are 'platform_applicant'; recruiter/bulk uploads should
            # arrive as 'org_upload' so DI provenance stays honest.
            application_id = metadata.get("application_id")
            job_id = metadata.get("job_id")
            org_id = metadata.get("org_id")
            source_type = metadata.get("provenance_type") or "platform_applicant"
            if source_type not in ("platform_applicant", "org_upload"):
                source_type = "platform_applicant"
            cur.execute(
                """
                INSERT INTO candidate_provenance
                    (global_candidate_id, source_type, tenant_id, source_detail)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (global_candidate_id, source_type, tenant_id)
                DO UPDATE SET source_detail = EXCLUDED.source_detail
                """,
                (
                    gc_id,
                    source_type,
                    tenant_id,
                    json.dumps(
                        {
                            "application_id": str(application_id) if application_id else None,
                            "job_id": str(job_id) if job_id else None,
                            "org_id": str(org_id) if org_id else None,
                            "resume_node_id": node_id,
                        }
                    ),
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


# ---------------------------------------------------------------------------
# Signal-sourced candidates → global memory (#29 slice 2)
# ---------------------------------------------------------------------------


def upsert_signal_candidate_to_global(
    cur: psycopg.Cursor,
    *,
    tenant_id: str,
    linkedin_url: str | None,
    name: str | None,
    headline: str | None,
    location_city: str | None,
    location_country: str | None,
    seniority_band: str | None,
    skills: list[str] | None,
    signal_candidate_id: str,
) -> str | None:
    """Upsert a Crustdata/Signal-sourced candidate into global_candidates.

    Called from the tenant resolve path (same transaction/cursor) so the
    tenant row and the global row commit together. Provenance is PUBLIC
    (tenant_id NULL): sourced profiles are public-web data per the product
    scope rules. Returns the global_candidate_id for the tenant-side link,
    or None when no usable identity anchor exists.
    """
    li_id = linkedin_id_from_url(linkedin_url)
    if not li_id:
        return None  # no anchor — a global row without identity is merge debt

    existing, extras = _find_existing_all(cur, li_id, None, None)
    for extra in extras:
        _enqueue_merge(
            cur,
            str(existing["id"]),
            str(extra["id"]),
            None,
            "needs_merge",
            {"source": "signal_ingest", "signal_candidate_id": signal_candidate_id},
        )

    country_code = _normalize_country_code(location_country) if location_country else None

    if existing:
        gc_id = str(existing["id"])
        sets = [
            "last_evidence_at = now()",
            "updated_at = now()",
            "embedding_status = 'queued'",
        ]  # re-embed: profile evidence changed
        params: list[Any] = []
        for col, val in [
            ("name", name),
            ("headline", headline),
            ("seniority_band", seniority_band),
            ("location_city", location_city),
            ("location_country_code", country_code),
            ("linkedin_url", linkedin_url),
        ]:
            if val is not None:
                sets.append(f"{col} = COALESCE({col}, %s)")
                params.append(val)
        if skills:
            sets.append(
                "skills_normalized = ARRAY(SELECT DISTINCT unnest(COALESCE(skills_normalized, ARRAY[]::text[]) || %s::text[]))"
            )
            params.append([s.lower().strip() for s in skills if s and s.strip()])
        params.append(gc_id)
        cur.execute(
            f"UPDATE global_candidates SET {', '.join(sets)} WHERE id = %s",
            params,
        )
    else:
        cur.execute(
            """
            INSERT INTO global_candidates
                (linkedin_id, linkedin_url, name, headline, seniority_band,
                 skills_normalized, location_city, location_country_code,
                 identity_confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0.7)
            RETURNING id
            """,
            (
                li_id,
                linkedin_url,
                name,
                headline,
                seniority_band,
                [s.lower().strip() for s in skills if s and s.strip()] if skills else None,
                location_city,
                country_code,
            ),
        )
        gc_id = str(cur.fetchone()[0])

    # Public provenance: sourced = public-web data (tenant_id NULL).
    cur.execute(
        """
        INSERT INTO candidate_provenance
            (global_candidate_id, source_type, tenant_id, source_detail)
        VALUES (%s, 'signal_sourced', NULL, %s::jsonb)
        ON CONFLICT DO NOTHING
        """,
        (
            gc_id,
            json.dumps({"signal_candidate_id": signal_candidate_id, "sourcing_tenant": tenant_id}),
        ),
    )
    return gc_id


# ---------------------------------------------------------------------------
# Vector search over global candidates (#29 slice 5 — hybrid retrieval substrate)
# ---------------------------------------------------------------------------

_embedder = None


def set_embedder(embedder: Any) -> None:
    """Injected from API startup so search reuses the process-wide model."""
    global _embedder
    _embedder = embedder


class GlobalCandidateSearchRequest(BaseModel):
    query_text: str
    limit: int = 50
    location_city: str | None = None
    role_family: str | None = None
    seniority_band: str | None = None
    skills_any: list[str] | None = None


@router.post(
    "/global-candidates/search",
    dependencies=[Depends(require_scope("kg:read"))],
)
def search_global_candidates(
    body: GlobalCandidateSearchRequest,
    claims=Depends(get_jwt_claims),
):
    """Vector search over the platform pool.

    Visibility rule (product scope tiers): a row is visible to the requesting
    tenant iff it has PUBLIC provenance (tenant_id IS NULL) OR that tenant has
    an access row for it (their own private uploads / consented applicants).
    """
    _require_enabled()
    if _embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not initialized")

    tenant_id = getattr(claims, "tenant_id", None) if claims else None
    # Server-side cap is the authoritative binding limit (protects the ANN
    # scan); the response carries applied_limit so callers can log truncation.
    # The old hardcoded 200 silently truncated Signal's limit=300 requests —
    # over an 815-row Bengaluru segment that cost half the fit-top-100
    # (Stage-3 offline gate finding).
    limit = max(1, min(body.limit, _SEARCH_LIMIT_MAX))

    vec = _embedder.encode([body.query_text])[0]
    vec_literal = "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"

    filters: list[str] = ["gc.embedding_status = 'ready'", "gc.embedding IS NOT NULL"]
    params: list[Any] = []
    if body.location_city:
        # Substring match ('Greater Bengaluru Area' must pass 'Bengaluru'),
        # and NULL passes through: rows whose location simply failed parsing
        # stay retrievable — the caller's ranker already demotes unknown
        # locations and its country guard treats no-location as an escape,
        # so hiding them here silently shrank the pool instead.
        filters.append("(gc.location_city ILIKE '%%' || %s || '%%' OR gc.location_city IS NULL)")
        params.append(body.location_city)
    if body.role_family:
        filters.append("gc.role_family = %s")
        params.append(body.role_family)
    if body.seniority_band:
        filters.append("gc.seniority_band = %s")
        params.append(body.seniority_band)
    if body.skills_any:
        filters.append("gc.skills_normalized && %s::text[]")
        params.append([s.lower().strip() for s in body.skills_any])

    visibility = (
        "(EXISTS (SELECT 1 FROM candidate_provenance cp"
        " WHERE cp.global_candidate_id = gc.id AND cp.tenant_id IS NULL)"
        " OR EXISTS (SELECT 1 FROM tenant_candidate_access tca"
        " WHERE tca.global_candidate_id = gc.id AND tca.tenant_id = %s"
        " AND tca.revoked_at IS NULL))"
    )
    filters.append(visibility)
    params.append(tenant_id or "")

    # Hydrate the tenant-side crustdata blob via the #29 link column so the
    # caller's ranker gets full profiles. RLS on candidates limits the join to
    # the requesting tenant's own rows (tenant conn sets the GUC); cross-tenant
    # blob sharing for public rows is a follow-up (#12).
    sql = f"""
        SELECT gc.id, gc.name, gc.headline, gc.linkedin_url, gc.linkedin_id,
               gc.role_family, gc.seniority_band, gc.skills_normalized,
               gc.location_city, gc.location_country_code,
               1 - (gc.embedding <=> %s::vector) AS similarity,
               tc.profile AS crustdata_profile,
               tc.candidate_id AS tenant_candidate_id,
               tc.signal_candidate_id
        FROM global_candidates gc
        LEFT JOIN LATERAL (
            SELECT c.profile, c.candidate_id,
                   (SELECT ci.value_normalized FROM candidate_identifiers ci
                    WHERE ci.candidate_id = c.candidate_id
                      AND ci.tenant_id = c.tenant_id
                      AND ci.identifier_type = 'signal_candidate_id'
                    LIMIT 1) AS signal_candidate_id
            FROM candidates c
            WHERE c.global_candidate_id = gc.id
            LIMIT 1
        ) tc ON true
        WHERE {" AND ".join(filters)}
        ORDER BY gc.embedding <=> %s::vector
        LIMIT %s
    """

    conn = _get_tenant_conn(tenant_id)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, [vec_literal, *params, vec_literal, limit])
            rows = [_row_to_dict(cur, r) for r in cur.fetchall()]
        conn.commit()
        return {"results": rows, "count": len(rows), "applied_limit": limit}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Resume refs for sourced candidates (#Stage-5B — resume-to-recruiter)
# ---------------------------------------------------------------------------


class ResumeRefsRequest(BaseModel):
    linkedin_ids: list[str]


@router.post(
    "/global-candidates/resume-refs",
    dependencies=[Depends(require_scope("kg:read"))],
)
def resume_refs(
    body: ResumeRefsRequest,
    claims=Depends(get_jwt_claims),
):
    """Batch: which of these people have a resume the requesting tenant may see?

    Joins global identity (linkedin_id) -> applicant/upload provenance and
    returns the Flow application pointer so the caller can serve the resume
    through its existing permission-gated streamer. RLS on candidate_provenance
    scopes results to the requesting tenant's own applicant rows (cross-tenant
    resume visibility is deliberately deferred to the consent work, #12).
    """
    _require_enabled()
    tenant_id = getattr(claims, "tenant_id", None) if claims else None
    slugs = [s.strip().lower() for s in body.linkedin_ids if s and s.strip()][:200]
    if not slugs:
        return {"refs": {}}

    conn = _get_tenant_conn(tenant_id)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT gc.linkedin_id, cp.source_type, cp.source_detail
                FROM global_candidates gc
                JOIN candidate_provenance cp ON cp.global_candidate_id = gc.id
                WHERE gc.linkedin_id = ANY(%s)
                  AND cp.source_type IN ('platform_applicant', 'org_upload')
                """,
                (slugs,),
            )
            refs: dict[str, dict[str, Any]] = {}
            for slug, source_type, detail in cur.fetchall():
                if slug in refs:
                    continue
                detail = detail or {}
                application_id = detail.get("application_id")
                if not application_id:
                    continue
                refs[slug] = {
                    "application_id": application_id,
                    "org_id": detail.get("org_id"),
                    "resume_node_id": detail.get("resume_node_id"),
                    "source_type": source_type,
                }
        conn.commit()
        return {"refs": refs}
    finally:
        conn.close()
