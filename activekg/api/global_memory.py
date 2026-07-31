"""Global candidate memory endpoints.

Provides CRUD operations for cross-tenant candidate memory:
global_candidates, candidate_provenance, tenant_candidate_access, feedback_events.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from activekg.api.auth import get_jwt_claims, require_scope
from activekg.common.logger import get_enhanced_logger
from activekg.embedding.global_candidates import (  # noqa: F401 — shared with the embedding producer
    PUBLIC_EMBED_VERSION,
    build_candidate_embedding_text,
)
from activekg.graph.candidate_identifiers import (
    IdentifierNormalizationError,
    normalize_identifier,
)

logger = get_enhanced_logger(__name__)

router = APIRouter(tags=["global-memory"])

_DSN = os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL", "")

GLOBAL_MEMORY_ENABLED = os.getenv("GLOBAL_MEMORY_ENABLED", "false").lower() == "true"

# Cross-tenant reads remain dark until migration 021 has backfilled the public
# projection and the public embedding drain has passed the retrieval gate.
PUBLIC_PROFILE_SEARCH_ENABLED = (
    os.getenv("GLOBAL_PUBLIC_PROFILE_SEARCH_ENABLED", "false").lower() == "true"
)
# Transitional kill switch for the historical shared-canonical vector surface.
# It must be false before public_v1 is activated: legacy embeddings may contain
# tenant-private resume evidence after identity reconciliation.
LEGACY_GLOBAL_SEARCH_ENABLED = (
    os.getenv("GLOBAL_LEGACY_CANDIDATE_SEARCH_ENABLED", "true").lower() == "true"
)

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


def _require_public_enabled():
    _require_enabled()
    if not PUBLIC_PROFILE_SEARCH_ENABLED:
        raise HTTPException(status_code=503, detail="Public profile search is disabled")


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


class ContactEvidenceRecord(BaseModel):
    global_candidate_id: UUID
    email: str
    provider: Literal["fullenrich", "enrichlayer"]
    provider_record_id: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    observed_at: datetime | None = None
    validated_at: datetime | None = None
    status: Literal[
        "found",
        "verified",
        "soft_bounce",
        "hard_bounce",
        "complaint",
        "invalid",
    ] = "found"
    bounce_reason: str | None = None


class ContactEvidenceLookup(BaseModel):
    global_candidate_ids: list[UUID] = Field(min_length=1, max_length=200)


class ContactSuppressionRecord(BaseModel):
    email: str
    reason: Literal["hard_bounce", "complaint"]
    provider_event_id: str | None = None


class PublicMarketExclusionRequest(BaseModel):
    coarse_market_key: str
    fresh_days: int = Field(default=14, ge=1, le=365)
    limit: int = Field(default=2000, ge=1, le=10000)


class PublicIdentityLookupRequest(BaseModel):
    linkedin_urls: list[str] = Field(min_length=1, max_length=200)


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
    try:
        normalized = normalize_identifier("linkedin_url", url)
    except IdentifierNormalizationError:
        return None
    m = _LINKEDIN_SLUG_RE.search(normalized)
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


_CONTACT_PROVIDERS = {"fullenrich", "enrichlayer"}
_CONTACT_STATUSES = {"found", "verified", "soft_bounce", "hard_bounce", "complaint", "invalid"}
_PUBLIC_JOB_SCALAR_FIELDS = {
    "company_name",
    "title",
    "seniority_level",
    "function_category",
    "start_date",
    "end_date",
    "description",
    "name",
    "years_at_company_raw",
    "company_headquarters_country",
    "company_professional_network_industry",
    "company_type",
    "company_headcount_range",
}
_PUBLIC_JOB_LIST_FIELDS = {"company_industries"}


UNOWNED_SUPPRESSION_SCOPE: str = "contact:suppress_unowned"

# Issuers trusted to suppress an address they hold no evidence for. This is a
# Flow-only capability: Flow owns the provider webhook that observes bounces and
# complaints. Kept as an allowlist so no future issuer inherits it silently.
UNOWNED_SUPPRESSION_ISSUERS: frozenset[str] = frozenset(
    issuer for issuer in (os.getenv("FLOW_JWT_ISSUER", "vantahire"),) if issuer and issuer.strip()
)


def _require_unowned_suppression_authority(claims: Any) -> None:
    """Gate the evidence-absent suppression path.

    Suppressing an address the caller owns no evidence for can tombstone ANY
    address platform-wide, so `contact:write` (which every enrichment caller
    holds) is not sufficient. Require a dedicated scope AND a verified service
    issuer. The provider event id is audit data and is deliberately NOT treated
    as authorization: it is attacker-suppliable.
    """
    from activekg.api.auth import JWT_ENABLED

    if not JWT_ENABLED:
        return
    scopes = set(getattr(claims, "scopes", []) or [])
    if UNOWNED_SUPPRESSION_SCOPE not in scopes:
        raise HTTPException(
            status_code=403,
            detail=(
                "Insufficient permissions. Suppressing an address without "
                f"tenant-owned evidence requires scope: {UNOWNED_SUPPRESSION_SCOPE}"
            ),
        )
    if getattr(claims, "actor_type", None) != "service":
        raise HTTPException(
            status_code=403,
            detail="Only service identities may suppress without tenant-owned evidence",
        )
    issuer = getattr(claims, "issuer", None)
    if issuer not in UNOWNED_SUPPRESSION_ISSUERS:
        raise HTTPException(
            status_code=403,
            detail="Issuer is not trusted to suppress without tenant-owned evidence",
        )


def _tenant_from_claims(claims: Any) -> str:
    tenant_id = getattr(claims, "tenant_id", None) if claims else None
    if not tenant_id or not str(tenant_id).strip():
        raise HTTPException(status_code=400, detail="tenant_id claim is required")
    return str(tenant_id)


def _normalize_email(raw: str) -> tuple[str, str]:
    try:
        email = normalize_identifier("email", raw)
    except IdentifierNormalizationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid email: {exc}") from exc
    return email, hashlib.sha256(email.encode("utf-8")).hexdigest()


def _signal_id_normalized(raw: str) -> str:
    try:
        return normalize_identifier("signal_candidate_id", raw)
    except IdentifierNormalizationError as exc:
        raise HTTPException(status_code=400, detail=f"invalid signal_candidate_id: {exc}") from exc


_PUBLIC_EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
_PUBLIC_PHONE_PATTERN = re.compile(
    r"(?:"
    r"\+\d[\d().\s-]{5,}\d"
    r"|\(\d{2,4}\)[\s.-]\d{3,4}[\s.-]\d{3,4}"
    r"|\b\d{2,4}[\s.-]\d{3,4}[\s.-]\d{3,4}\b"
    r"|\b\d{5}[\s.-]\d{5}\b"
    r"|\b\d{9,15}\b"
    r")"
)


def _sanitize_public_scalar(value: Any) -> str | int | float | bool | None:
    if isinstance(value, str):
        without_email = _PUBLIC_EMAIL_PATTERN.sub("[redacted]", value)
        return _PUBLIC_PHONE_PATTERN.sub("[redacted]", without_email)
    if isinstance(value, (int, float, bool)) and not isinstance(value, bytes):
        return value
    return None


def _pick_public(
    source: Any,
    scalar_fields: set[str],
    list_fields: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    result = {}
    for key in scalar_fields:
        if key not in source:
            continue
        sanitized = _sanitize_public_scalar(source[key])
        if sanitized is not None:
            result[key] = sanitized
    for key in list_fields or set():
        value = source.get(key)
        if not isinstance(value, list):
            continue
        scalars = [
            sanitized for item in value if (sanitized := _sanitize_public_scalar(item)) is not None
        ]
        if scalars:
            result[key] = scalars
    return result


def _pick_public_rows(
    source: Any,
    scalar_fields: set[str],
    list_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(source, list):
        return []
    return [picked for row in source if (picked := _pick_public(row, scalar_fields, list_fields))]


def sanitize_public_profile(profile: Any) -> dict[str, Any]:
    """Project only documented public Crustdata fields into the shared surface."""
    if not isinstance(profile, dict):
        return {}

    basic = _pick_public(
        profile.get("basic_profile"),
        {
            "name",
            "first_name",
            "last_name",
            "headline",
            "current_title",
            "summary",
        },
        {"languages"},
    )
    location = _pick_public(
        (profile.get("basic_profile") or {}).get("location")
        if isinstance(profile.get("basic_profile"), dict)
        else None,
        {"city", "state", "country", "continent", "full_location", "raw", "country_code"},
    )
    if location:
        basic["location"] = location

    network = _pick_public(
        profile.get("professional_network"),
        {"connections", "followers", "profile_picture_permalink"},
        {"open_to_cards"},
    )
    network_source = profile.get("professional_network")
    if isinstance(network_source, dict):
        network_location = _pick_public(network_source.get("location"), {"raw"})
        network_metadata = _pick_public(network_source.get("metadata"), {"last_scraped_source"})
        if network_location:
            network["location"] = network_location
        if network_metadata:
            network["metadata"] = network_metadata

    social_source = profile.get("social_handles")
    social: dict[str, Any] = {}
    if isinstance(social_source, dict):
        for key, fields in {
            "professional_network_identifier": {"profile_url"},
            "twitter_identifier": {"slug"},
            "dev_platform_identifier": {"profile_url"},
        }.items():
            value = _pick_public(social_source.get(key), fields)
            if value:
                social[key] = value

    experience_source = profile.get("experience")
    employment = (
        experience_source.get("employment_details") if isinstance(experience_source, dict) else None
    )
    experience: dict[str, Any] = {}
    if isinstance(employment, dict):
        details = {
            key: rows
            for key in ("current", "past")
            if (
                rows := _pick_public_rows(
                    employment.get(key),
                    _PUBLIC_JOB_SCALAR_FIELDS,
                    _PUBLIC_JOB_LIST_FIELDS,
                )
            )
        }
        if details:
            experience["employment_details"] = details

    education_source = profile.get("education")
    education: dict[str, Any] = {}
    if isinstance(education_source, dict):
        schools = _pick_public_rows(
            education_source.get("schools"),
            {"school", "degree", "field_of_study", "start_year", "end_year"},
        )
        if schools:
            education["schools"] = schools

    result: dict[str, Any] = {}
    scalar_fields = {
        "crustdata_person_id",
        "years_of_experience_raw",
        "recently_changed_jobs",
    }
    result.update(_pick_public(profile, scalar_fields))
    metadata = _pick_public(profile.get("metadata"), {"updated_at"})
    skills = _pick_public(
        profile.get("skills"),
        set(),
        {"professional_network_skills"},
    )
    certifications = _pick_public_rows(
        profile.get("certifications"),
        {"name", "issuing_organization", "issue_date", "expiration_date"},
    )
    honors = _pick_public_rows(profile.get("honors"), {"title", "issuer", "description"})
    for key, value in (
        ("metadata", metadata),
        ("basic_profile", basic),
        ("professional_network", network),
        ("social_handles", social),
        ("education", education),
        ("experience", experience),
        ("skills", skills),
        ("certifications", certifications),
        ("honors", honors),
    ):
        if value:
            result[key] = value
    return result


def _merge_nonempty(existing: Any, incoming: Any) -> Any:
    """Deep merge an observation without allowing partial/empty wipes."""
    if incoming in (None, "", [], {}):
        return existing
    if isinstance(existing, dict) and isinstance(incoming, dict):
        merged = dict(existing)
        for key, value in incoming.items():
            merged[key] = _merge_nonempty(merged.get(key), value)
        return merged
    return incoming


def _validated_public_market(value: Any) -> dict[str, str] | None:
    """Validate Signal's versioned coarse-market key before shared indexing."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("public_market must be an object")
    if value.get("version") != 1:
        raise ValueError("unsupported public_market version")

    fields = {
        "role_family": str(value.get("role_family") or "").strip(),
        "location_city": str(value.get("location_city") or "").strip(),
        "location_country_code": str(value.get("location_country_code") or "").strip().upper(),
        "seniority_band": str(value.get("seniority_band") or "").strip(),
    }
    if not all(fields.values()) or _ISO_ALPHA2.fullmatch(fields["location_country_code"]) is None:
        raise ValueError("public_market canonical dimensions are incomplete")
    for field in ("role_family", "location_city", "seniority_band"):
        if fields[field] != fields[field].lower():
            raise ValueError(f"public_market {field} is not canonical")

    key_material = {
        "version": 1,
        "roleFamily": fields["role_family"],
        "locationCity": fields["location_city"],
        "locationCountryCode": fields["location_country_code"],
        "seniorityBand": fields["seniority_band"],
    }
    digest = hashlib.sha256(
        json.dumps(key_material, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    expected_key = f"public-market:v1:{digest}"
    if value.get("coarse_market_key") != expected_key:
        raise ValueError("public_market key does not match its canonical dimensions")
    return {**fields, "coarse_market_key": expected_key}


def _assert_public_candidate_visible(
    cur: psycopg.Cursor, *, tenant_id: str, global_candidate_id: str
) -> str:
    cur.execute(
        """
        SELECT gc.id
        FROM global_candidates gc
        WHERE gc.id = %s
          AND (
              EXISTS (
                  SELECT 1 FROM candidate_provenance cp
                  WHERE cp.global_candidate_id = gc.id
                    AND cp.source_type = 'signal_sourced'
                    AND cp.tenant_id IS NULL
              )
              OR EXISTS (
                  SELECT 1 FROM tenant_candidate_access tca
                  WHERE tca.global_candidate_id = gc.id
                    AND tca.tenant_id = %s
                    AND tca.revoked_at IS NULL
              )
          )
        LIMIT 1
        """,
        (global_candidate_id, tenant_id),
    )
    row = cur.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="candidate is not visible to this tenant")
    return str(row[0])


def _choose_primary_contact(cur: psycopg.Cursor, *, tenant_id: str, candidate_id: str) -> None:
    """Select one usable email without deleting conflicting evidence."""
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (f"{tenant_id}:{candidate_id}",),
    )
    cur.execute(
        """
        UPDATE candidate_contact_evidence
        SET is_primary = false, updated_at = now()
        WHERE tenant_id = %s AND global_candidate_id = %s AND is_primary
        """,
        (tenant_id, candidate_id),
    )
    cur.execute(
        """
        WITH winner AS (
            SELECT cce.id
            FROM candidate_contact_evidence cce
            LEFT JOIN contact_suppression_tombstones cst
              ON cst.email_hash = cce.email_hash
            LEFT JOIN contact_person_suppressions cps
              ON cps.global_candidate_id = cce.global_candidate_id
            WHERE cce.tenant_id = %s
              AND cce.global_candidate_id = %s
              AND cce.status IN ('found', 'verified')
              AND cce.suppressed_at IS NULL
              AND cst.email_hash IS NULL
              -- A person-terminal complaint blocks every address of this person.
              AND cps.global_candidate_id IS NULL
            ORDER BY
              CASE cce.provider WHEN 'fullenrich' THEN 2 WHEN 'enrichlayer' THEN 1 ELSE 0 END DESC,
              (cce.status = 'verified') DESC,
              cce.confidence DESC,
              cce.validated_at DESC NULLS LAST,
              cce.observed_at DESC,
              cce.id
            LIMIT 1
        )
        UPDATE candidate_contact_evidence cce
        SET is_primary = true, updated_at = now()
        FROM winner
        WHERE cce.id = winner.id
        """,
        (tenant_id, candidate_id),
    )


def _selected_contact_state(
    cur: psycopg.Cursor, *, tenant_id: str, candidate_id: str
) -> dict[str, Any]:
    # A complaint is person-terminal platform-wide: no address of this person is
    # selectable by ANY tenant, so there is no alternate to re-elect. Checked
    # before re-election so a different org's complaint cannot be worked around
    # by promoting another validated address.
    cur.execute(
        "SELECT reason FROM contact_person_suppressions WHERE global_candidate_id = %s",
        (candidate_id,),
    )
    person_suppressed = cur.fetchone()
    if person_suppressed:
        return {"state": "suppressed", "contact": None, "reason": person_suppressed[0]}

    # A platform tombstone may have been created by a different tenant since
    # this tenant last selected its primary. Re-elect on read so a usable
    # alternate is promoted without exposing or mutating the other tenant's
    # evidence row.
    _choose_primary_contact(cur, tenant_id=tenant_id, candidate_id=candidate_id)
    cur.execute(
        """
        SELECT cce.email, cce.provider, cce.provider_record_id,
               cce.confidence, cce.observed_at, cce.validated_at, cce.status
        FROM candidate_contact_evidence cce
        LEFT JOIN contact_suppression_tombstones cst
          ON cst.email_hash = cce.email_hash
        WHERE cce.tenant_id = %s
          AND cce.global_candidate_id = %s
          AND cce.is_primary
          AND cce.status IN ('found', 'verified')
          AND cce.suppressed_at IS NULL
          AND cst.email_hash IS NULL
        LIMIT 1
        """,
        (tenant_id, candidate_id),
    )
    row = cur.fetchone()
    if row:
        return {"state": "found", "contact": _row_to_dict(cur, row)}

    cur.execute(
        """
        SELECT COALESCE(cst.reason, cce.status)
        FROM candidate_contact_evidence cce
        LEFT JOIN contact_suppression_tombstones cst
          ON cst.email_hash = cce.email_hash
        WHERE cce.tenant_id = %s AND cce.global_candidate_id = %s
          AND (
              cce.status NOT IN ('found', 'verified')
              OR cce.suppressed_at IS NOT NULL
              OR cst.email_hash IS NOT NULL
          )
        ORDER BY cce.observed_at DESC
        LIMIT 1
        """,
        (tenant_id, candidate_id),
    )
    unusable = cur.fetchone()
    if unusable:
        return {"state": "suppressed", "contact": None, "reason": unusable[0]}
    return {"state": "miss", "contact": None}


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
    tenant_id = _tenant_from_claims(claims)

    if not any([linkedin_id, github_id, email_hash]):
        raise HTTPException(
            status_code=400,
            detail="At least one anchor query param required (linkedin_id, github_id, email_hash)",
        )

    conn = _get_tenant_conn(tenant_id)
    try:
        with conn.cursor() as cur:
            for anchor, value in [
                ("linkedin_id", linkedin_id),
                ("github_id", github_id),
                ("email_hash", email_hash),
            ]:
                if value is None:
                    continue
                normalized_value = value.lower() if anchor == "linkedin_id" else value
                anchor_sql = f"lower(gc.{anchor})" if anchor == "linkedin_id" else f"gc.{anchor}"
                public_allowed = anchor == "linkedin_id"
                cur.execute(
                    f"""
                    SELECT gc.*,
                           EXISTS (
                               SELECT 1 FROM candidate_provenance cp
                               WHERE cp.global_candidate_id = gc.id
                                 AND cp.source_type = 'signal_sourced'
                                 AND cp.tenant_id IS NULL
                           ) AS _public_visible,
                           EXISTS (
                               SELECT 1 FROM tenant_candidate_access tca
                               WHERE tca.global_candidate_id = gc.id
                                 AND tca.tenant_id = %s
                                 AND tca.revoked_at IS NULL
                           ) AS _tenant_visible
                    FROM global_candidates gc
                    WHERE {anchor_sql} = %s
                      AND (
                          EXISTS (
                              SELECT 1 FROM tenant_candidate_access tca
                              WHERE tca.global_candidate_id = gc.id
                                AND tca.tenant_id = %s
                                AND tca.revoked_at IS NULL
                          )
                          OR (
                              %s
                              AND EXISTS (
                                  SELECT 1 FROM candidate_provenance cp
                                  WHERE cp.global_candidate_id = gc.id
                                    AND cp.source_type = 'signal_sourced'
                                    AND cp.tenant_id IS NULL
                              )
                          )
                      )
                    LIMIT 1
                    """,
                    (tenant_id, normalized_value, tenant_id, public_allowed),
                )
                row = cur.fetchone()
                if row:
                    result = _row_to_dict(cur, row)
                    public_visible = bool(result.pop("_public_visible", False))
                    tenant_visible = bool(result.pop("_tenant_visible", False))
                    if tenant_visible:
                        # This endpoint resolves identity anchors; it must not
                        # double as a profile read. The historical canonical
                        # columns can contain evidence contributed by another
                        # tenant after two records reconcile to one identity.
                        # Return only public identity plus the exact private
                        # anchor the caller already supplied.
                        identity = {
                            "id": str(result["id"]),
                            "surface": "tenant_identity_v1",
                        }
                        if anchor == "linkedin_id":
                            identity["linkedin_id"] = normalized_value
                            identity["linkedin_url"] = f"https://linkedin.com/in/{normalized_value}"
                        elif anchor == "github_id":
                            identity["github_id"] = value
                        elif anchor == "email_hash":
                            identity["email_hash"] = value
                        conn.commit()
                        return identity
                    if public_visible:
                        # A public match never authorizes applicant-derived
                        # canonical fields such as email_hash or resume skills.
                        conn.commit()
                        return {
                            "id": str(result["id"]),
                            "linkedin_id": result.get("linkedin_id"),
                            "linkedin_url": result.get("linkedin_url"),
                            "name": (result.get("public_profile") or {})
                            .get("basic_profile", {})
                            .get("name"),
                            "headline": result.get("public_headline"),
                            "location_city": result.get("public_location_city"),
                            "location_country_code": result.get("public_location_country_code"),
                            "role_family": result.get("public_role_family"),
                            "seniority_band": result.get("public_seniority_band"),
                            "skills_normalized": result.get("public_skills_normalized"),
                            "surface": "public_v1",
                        }

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
    public_profile: dict[str, Any] | None = None,
    public_role_family: str | None = None,
    public_market: dict[str, Any] | None = None,
    profile_observed_at: datetime | None = None,
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

    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
        (f"global-signal:{li_id}",),
    )
    existing, extras = _find_existing_all(cur, li_id, None, None)
    if existing:
        # Serialize all public observations for one canonical identity, then
        # re-read under the lock. Without this, two workers that both observed a
        # legacy NULL provider ID could publish different Crustdata people and
        # silently let the last writer win.
        cur.execute(
            "SELECT * FROM global_candidates WHERE id = %s FOR UPDATE",
            (str(existing["id"]),),
        )
        locked_row = cur.fetchone()
        if locked_row is None:
            raise RuntimeError("global candidate disappeared during Signal ingest")
        existing = dict(zip((column.name for column in cur.description), locked_row, strict=False))

    observed_at = profile_observed_at
    if observed_at is not None:
        observed_at = (
            observed_at.replace(tzinfo=timezone.utc)
            if observed_at.tzinfo is None
            else observed_at.astimezone(timezone.utc)
        )

    existing_observed_at = existing.get("public_profile_observed_at") if existing else None
    if isinstance(existing_observed_at, str):
        existing_observed_at = datetime.fromisoformat(existing_observed_at.replace("Z", "+00:00"))
    if isinstance(existing_observed_at, datetime):
        existing_observed_at = (
            existing_observed_at.replace(tzinfo=timezone.utc)
            if existing_observed_at.tzinfo is None
            else existing_observed_at.astimezone(timezone.utc)
        )
    observation_is_stale = bool(
        existing
        and existing_observed_at is not None
        and (observed_at is None or observed_at <= existing_observed_at)
    )

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
    normalized_skills = [s.lower().strip() for s in skills if s and s.strip()] if skills else None
    sanitized_observation = sanitize_public_profile(public_profile)
    existing_public = existing.get("public_profile") if existing else {}
    incoming_person_id_raw = sanitized_observation.get("crustdata_person_id")
    existing_person_id_raw = (existing.get("public_crustdata_person_id") if existing else None) or (
        (existing_public or {}).get("crustdata_person_id")
    )
    try:
        incoming_person_id = (
            int(incoming_person_id_raw) if incoming_person_id_raw is not None else None
        )
    except (TypeError, ValueError):
        incoming_person_id = None
    try:
        existing_person_id = (
            int(existing_person_id_raw) if existing_person_id_raw is not None else None
        )
    except (TypeError, ValueError):
        existing_person_id = None
    if (
        existing
        and existing_person_id is not None
        and incoming_person_id is not None
        and existing_person_id != incoming_person_id
    ):
        # A stable LinkedIn identity switching provider IDs is ambiguous
        # evidence, not a refresh. Keep the published projection unchanged and
        # make the conflict countable instead of cross-attaching a new profile.
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"crustdata-switch:{existing['id']}",),
        )
        _enqueue_merge(
            cur,
            str(existing["id"]),
            None,
            None,
            "review_required",
            {
                "source": "signal_ingest",
                "anchor": "crustdata_person_id_switch",
                "linkedin_id": li_id,
                "existing_crustdata_person_id": existing_person_id,
                "incoming_crustdata_person_id": incoming_person_id,
            },
        )
        cur.execute(
            """
            UPDATE global_candidates
            SET merge_status = 'needs_merge', updated_at = now()
            WHERE id = %s
            """,
            (str(existing["id"]),),
        )
        sanitized_observation = {}
    if observation_is_stale:
        # Identity disagreements remain durable #12 evidence, but an older or
        # equal paid-batch replay cannot refresh profile, embedding, provenance,
        # or market timestamps.
        return str(existing["id"])
    shareable_profile = _merge_nonempty(existing_public or {}, sanitized_observation)
    crustdata_person_id_raw = shareable_profile.get("crustdata_person_id")
    try:
        crustdata_person_id = (
            int(crustdata_person_id_raw) if crustdata_person_id_raw is not None else None
        )
    except (TypeError, ValueError):
        crustdata_person_id = None

    basic = shareable_profile.get("basic_profile")
    basic = basic if isinstance(basic, dict) else {}
    public_location = basic.get("location")
    public_location = public_location if isinstance(public_location, dict) else {}
    # Cross-tenant fields come only from the sanitized Crustdata projection.
    # The flat Signal hints are retained on the tenant-private record, but are
    # not an alternate public-data lane when the provider profile is partial.
    public_headline = basic.get("headline") or None
    public_city = public_location.get("city") or None
    public_country_raw = public_location.get("country_code") or public_location.get("country")
    public_country = _normalize_country_code(public_country_raw)
    experience = shareable_profile.get("experience")
    experience = experience if isinstance(experience, dict) else {}
    employment = experience.get("employment_details")
    employment = employment if isinstance(employment, dict) else {}
    current = employment.get("current")
    current = current if isinstance(current, list) else []
    public_seniority = (
        current[0].get("seniority_level") if current and isinstance(current[0], dict) else None
    )
    public_skills_node = shareable_profile.get("skills")
    public_skills_node = public_skills_node if isinstance(public_skills_node, dict) else {}
    public_skills_raw = public_skills_node.get("professional_network_skills")
    public_skills = (
        [str(value).strip().lower() for value in public_skills_raw if str(value).strip()]
        if isinstance(public_skills_raw, list)
        else None
    )
    public_id_conflict: dict[str, Any] | None = None
    if crustdata_person_id is not None:
        # A provider ID is a platform-global anchor. Serialize competing writes
        # and persist disagreements for review instead of relying on the unique
        # index to roll back an otherwise valid tenant ingest.
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (f"crustdata:{crustdata_person_id}",),
        )
        cur.execute(
            """
            SELECT id, linkedin_id
            FROM global_candidates
            WHERE public_crustdata_person_id = %s
              AND (%s::uuid IS NULL OR id <> %s::uuid)
            LIMIT 1
            """,
            (
                crustdata_person_id,
                str(existing["id"]) if existing else None,
                str(existing["id"]) if existing else None,
            ),
        )
        conflict_row = cur.fetchone()
        if conflict_row:
            public_id_conflict = {
                "id": str(conflict_row[0]),
                "linkedin_id": conflict_row[1],
            }
            # Do not publish disputed provider evidence on either identity. The
            # canonical row/provenance can still be recorded and later merged.
            sanitized_observation = {}
            shareable_profile = existing_public or {}
            crustdata_person_id = None

    if existing:
        gc_id = str(existing["id"])
        sets = [
            (
                "last_evidence_at = "
                + ("now()" if observed_at is None else "GREATEST(last_evidence_at, %s)")
            ),
            "updated_at = now()",
            "embedding_status = 'queued'",
        ]  # re-embed: profile evidence changed
        params: list[Any] = [observed_at] if observed_at is not None else []
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
            params.append(normalized_skills)
        if sanitized_observation:
            # Public fields are a replace-on-observation projection of the raw
            # Crustdata evidence. They never derive from the mutable tenant
            # canonical, which can also contain private applicant evidence.
            sets.extend(
                [
                    "public_profile = %s::jsonb",
                    "public_profile_observed_at = COALESCE(%s, now())",
                    "public_crustdata_person_id = COALESCE(%s, public_crustdata_person_id)",
                    "public_headline = %s",
                    "public_location_city = %s",
                    "public_location_country_code = %s",
                    "public_role_family = COALESCE(%s, public_role_family)",
                    "public_seniority_band = %s",
                    "public_skills_normalized = %s::text[]",
                    "public_embedding = NULL",
                    "public_embedding_status = 'queued'",
                    "public_embed_version = 0",
                ]
            )
            params.extend(
                [
                    json.dumps(shareable_profile),
                    observed_at,
                    crustdata_person_id,
                    public_headline,
                    public_city,
                    public_country,
                    public_role_family,
                    public_seniority,
                    public_skills,
                ]
            )
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
                 identity_confidence, public_profile, public_profile_observed_at,
                 public_crustdata_person_id, public_headline,
                 public_location_city, public_location_country_code,
                 public_role_family, public_seniority_band, public_skills_normalized,
                 public_embedding_status, public_embed_version, last_evidence_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 0.7,
                    %s::jsonb,
                    CASE WHEN %s::jsonb <> '{}'::jsonb THEN COALESCE(%s, now()) ELSE NULL END,
                    %s, %s, %s, %s, %s, %s, %s, 'queued', 0, COALESCE(%s, now()))
            RETURNING id
            """,
            (
                li_id,
                linkedin_url,
                name,
                headline,
                seniority_band,
                normalized_skills,
                location_city,
                country_code,
                json.dumps(shareable_profile),
                json.dumps(shareable_profile),
                observed_at,
                crustdata_person_id if shareable_profile else None,
                public_headline if shareable_profile else None,
                public_city if shareable_profile else None,
                public_country if shareable_profile else None,
                public_role_family if shareable_profile else None,
                public_seniority if shareable_profile else None,
                public_skills if shareable_profile else None,
                observed_at,
            ),
        )
        gc_id = str(cur.fetchone()[0])

    if public_id_conflict:
        _enqueue_merge(
            cur,
            gc_id,
            public_id_conflict["id"],
            None,
            "review_required",
            {
                "source": "signal_ingest",
                "anchor": "crustdata_person_id",
                "incoming_linkedin_id": li_id,
                "existing_linkedin_id": public_id_conflict["linkedin_id"],
                "signal_candidate_id": signal_candidate_id,
            },
        )
        cur.execute(
            """
            UPDATE global_candidates
            SET merge_status = 'needs_merge', updated_at = now()
            WHERE id = ANY(%s::uuid[])
            """,
            ([gc_id, public_id_conflict["id"]],),
        )

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
            json.dumps({}),
        ),
    )
    # A market observation is evidence about the accepted Crustdata profile.
    # Never attach it when provider identity conflicted and the incoming public
    # observation was deliberately withheld.
    market = (
        _validated_public_market(public_market)
        if sanitized_observation and crustdata_person_id is not None
        else None
    )
    if market:
        cur.execute(
            """
            INSERT INTO public_candidate_market_memberships
                (global_candidate_id, coarse_market_key, role_family,
                 location_city, location_country_code, seniority_band,
                 first_observed_at, last_observed_at)
            VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, now()), COALESCE(%s, now()))
            ON CONFLICT (global_candidate_id, coarse_market_key)
            DO UPDATE SET
                role_family = EXCLUDED.role_family,
                location_city = EXCLUDED.location_city,
                location_country_code = EXCLUDED.location_country_code,
                seniority_band = EXCLUDED.seniority_band,
                last_observed_at = GREATEST(
                    public_candidate_market_memberships.last_observed_at,
                    EXCLUDED.last_observed_at
                )
            """,
            (
                gc_id,
                market["coarse_market_key"],
                market["role_family"],
                market["location_city"],
                market["location_country_code"],
                market["seniority_band"],
                observed_at,
                observed_at,
            ),
        )
    return gc_id


# ---------------------------------------------------------------------------
# Tenant-restricted provider contact evidence (#12)
# ---------------------------------------------------------------------------


@router.post(
    "/contact-evidence/record",
    dependencies=[Depends(require_scope("contact:write"))],
)
def record_contact_evidence(
    body: ContactEvidenceRecord,
    claims=Depends(get_jwt_claims),
):
    """Record provider evidence without turning it into a shared profile field."""
    _require_enabled()
    tenant_id = _tenant_from_claims(claims)
    provider = body.provider.strip().lower()
    status = body.status.strip().lower()
    if provider not in _CONTACT_PROVIDERS:
        raise HTTPException(status_code=400, detail="unsupported contact provider")
    if status not in _CONTACT_STATUSES:
        raise HTTPException(status_code=400, detail="unsupported contact status")
    email, email_hash = _normalize_email(body.email)

    conn = _get_tenant_conn(tenant_id)
    try:
        with conn.cursor() as cur:
            candidate_id = _assert_public_candidate_visible(
                cur,
                tenant_id=tenant_id,
                global_candidate_id=body.global_candidate_id,
            )
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"{tenant_id}:{candidate_id}",),
            )
            cur.execute(
                "SELECT reason FROM contact_suppression_tombstones WHERE email_hash = %s",
                (email_hash,),
            )
            existing_tombstone = cur.fetchone()
            suppressed_at = (
                datetime.now().astimezone()
                if status in {"hard_bounce", "complaint"} or existing_tombstone
                else None
            )
            cur.execute(
                """
                INSERT INTO candidate_contact_evidence
                    (global_candidate_id, tenant_id, email, email_hash, provider,
                     provider_record_id, confidence, observed_at, validated_at,
                     status, suppressed_at, bounce_reason)
                VALUES (%s, %s, %s, %s, %s, %s, %s, COALESCE(%s, now()), %s,
                        %s, %s, %s)
                ON CONFLICT (global_candidate_id, tenant_id, provider, email_hash)
                DO UPDATE SET
                    provider_record_id = COALESCE(EXCLUDED.provider_record_id,
                                                  candidate_contact_evidence.provider_record_id),
                    confidence = GREATEST(candidate_contact_evidence.confidence,
                                          EXCLUDED.confidence),
                    observed_at = GREATEST(candidate_contact_evidence.observed_at,
                                           EXCLUDED.observed_at),
                    validated_at = COALESCE(EXCLUDED.validated_at,
                                            candidate_contact_evidence.validated_at),
                    status = CASE
                        WHEN candidate_contact_evidence.status = 'verified'
                         AND EXCLUDED.status = 'found'
                        THEN candidate_contact_evidence.status
                        ELSE EXCLUDED.status
                    END,
                    suppressed_at = COALESCE(EXCLUDED.suppressed_at,
                                             candidate_contact_evidence.suppressed_at),
                    is_primary = CASE
                        WHEN EXCLUDED.status IN ('found', 'verified')
                         AND EXCLUDED.suppressed_at IS NULL
                         AND candidate_contact_evidence.suppressed_at IS NULL
                        THEN candidate_contact_evidence.is_primary
                        ELSE false
                    END,
                    bounce_reason = COALESCE(EXCLUDED.bounce_reason,
                                             candidate_contact_evidence.bounce_reason),
                    updated_at = now()
                RETURNING id
                """,
                (
                    candidate_id,
                    tenant_id,
                    email,
                    email_hash,
                    provider,
                    body.provider_record_id,
                    body.confidence,
                    body.observed_at,
                    body.validated_at,
                    status,
                    suppressed_at,
                    body.bounce_reason,
                ),
            )
            evidence_id = str(cur.fetchone()[0])
            if status in {"hard_bounce", "complaint"}:
                cur.execute(
                    """
                    INSERT INTO contact_suppression_tombstones
                        (email_hash, global_candidate_id, reason, source_evidence_id)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (email_hash) DO UPDATE SET
                        global_candidate_id = COALESCE(
                            contact_suppression_tombstones.global_candidate_id,
                            EXCLUDED.global_candidate_id
                        ),
                        reason = EXCLUDED.reason,
                        last_observed_at = now(),
                        source_evidence_id = EXCLUDED.source_evidence_id
                    """,
                    (email_hash, candidate_id, status, evidence_id),
                )
            _choose_primary_contact(cur, tenant_id=tenant_id, candidate_id=candidate_id)
            selected = _selected_contact_state(cur, tenant_id=tenant_id, candidate_id=candidate_id)
        conn.commit()
        return {
            "evidence_id": evidence_id,
            "global_candidate_id": candidate_id,
            "status": status,
            **selected,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post(
    "/contact-evidence/lookup",
    dependencies=[Depends(require_scope("contact:read"))],
)
def lookup_contact_evidence(
    body: ContactEvidenceLookup,
    claims=Depends(get_jwt_claims),
):
    """Return only this tenant's selected, unsuppressed provider evidence."""
    _require_enabled()
    tenant_id = _tenant_from_claims(claims)
    candidate_ids = list(dict.fromkeys(str(value) for value in body.global_candidate_ids))
    conn = _get_tenant_conn(tenant_id)
    try:
        with conn.cursor() as cur:
            visible_ids = [
                _assert_public_candidate_visible(
                    cur, tenant_id=tenant_id, global_candidate_id=candidate_id
                )
                for candidate_id in candidate_ids
            ]
            states = {
                candidate_id: _selected_contact_state(
                    cur, tenant_id=tenant_id, candidate_id=candidate_id
                )
                for candidate_id in visible_ids
            }
        conn.commit()
        return {
            "results": [{"global_candidate_id": value, **states[value]} for value in candidate_ids]
        }
    finally:
        conn.close()


@router.post(
    "/contact-evidence/suppress",
    dependencies=[Depends(require_scope("contact:write"))],
)
def suppress_contact_evidence(
    body: ContactSuppressionRecord,
    claims=Depends(get_jwt_claims),
):
    """Record a platform-wide hard-bounce/complaint suppression.

    Scope differs by reason, per locked policy:
      * hard_bounce - tombstones the ADDRESS. The person stays reachable at any
        other validated address.
      * complaint   - person-terminal AND platform-wide, because every org mails
        under one sender identity. Suppressing only the address would leave the
        same person reachable at a different address, including by another org.
    """
    _require_enabled()
    _tenant_from_claims(claims)
    reason = body.reason.strip().lower()
    if reason not in {"hard_bounce", "complaint"}:
        raise HTTPException(status_code=400, detail="unsupported suppression reason")
    provider_event_id = (body.provider_event_id or "").strip() or None
    _email, email_hash = _normalize_email(body.email)
    tenant_id = getattr(claims, "tenant_id", None)
    conn = _get_tenant_conn(tenant_id)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, global_candidate_id
                FROM candidate_contact_evidence
                WHERE tenant_id = %s AND email_hash = %s
                ORDER BY observed_at DESC
                LIMIT 1
                """,
                (tenant_id, email_hash),
            )
            owned_evidence = cur.fetchone()
            evidence_present = owned_evidence is not None

            if evidence_present:
                evidence_id, candidate_id = owned_evidence
            else:
                # Suppressing an address this tenant owns no evidence for is a
                # PRIVILEGED action: it can tombstone any address platform-wide
                # with no provable relationship to the caller. It therefore needs
                # a dedicated Flow-only scope from a verified service issuer.
                # provider_event_id is audit data, never authorization.
                _require_unowned_suppression_authority(claims)
                if provider_event_id is None:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "provider_event_id is required to suppress an address "
                            "with no tenant-owned contact evidence"
                        ),
                    )
                evidence_id, candidate_id = None, None
                logger.warning(
                    "contact suppression without tenant-owned evidence",
                    extra={
                        "tenant_id": tenant_id,
                        "issuer": getattr(claims, "issuer", None),
                        "actor_id": getattr(claims, "actor_id", None),
                        "reason": reason,
                        "provider_event_id": provider_event_id,
                    },
                )

            # Address tombstone. `reason` never DOWNGRADES: a later hard bounce
            # must not overwrite a recorded complaint, in either direction of
            # arrival, and the existing candidate/evidence references survive a
            # hash-only write.
            cur.execute(
                """
                INSERT INTO contact_suppression_tombstones
                    (email_hash, global_candidate_id, reason, source_evidence_id,
                     provider_event_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (email_hash) DO UPDATE SET
                    global_candidate_id = COALESCE(
                        contact_suppression_tombstones.global_candidate_id,
                        EXCLUDED.global_candidate_id
                    ),
                    reason = CASE
                        WHEN contact_suppression_tombstones.reason = 'complaint'
                          OR EXCLUDED.reason = 'complaint'
                        THEN 'complaint'
                        ELSE EXCLUDED.reason
                    END,
                    last_observed_at = now(),
                    source_evidence_id = COALESCE(
                        EXCLUDED.source_evidence_id,
                        contact_suppression_tombstones.source_evidence_id
                    ),
                    provider_event_id = COALESCE(EXCLUDED.provider_event_id,
                                                 contact_suppression_tombstones.provider_event_id)
                """,
                (
                    email_hash,
                    candidate_id,
                    reason,
                    evidence_id,
                    provider_event_id,
                ),
            )

            # A complaint is person-terminal platform-wide. Only possible when
            # the person is resolvable; see the receipt for the unresolved case.
            person_suppressed = False
            if reason == "complaint":
                if candidate_id is None:
                    cur.execute(
                        """
                        SELECT global_candidate_id
                        FROM contact_suppression_tombstones
                        WHERE email_hash = %s
                        """,
                        (email_hash,),
                    )
                    known = cur.fetchone()
                    candidate_id = known[0] if known else None
                if candidate_id is not None:
                    cur.execute(
                        """
                        INSERT INTO contact_person_suppressions
                            (global_candidate_id, reason, provider_event_id)
                        VALUES (%s, 'complaint', %s)
                        ON CONFLICT (global_candidate_id) DO UPDATE SET
                            last_observed_at = now(),
                            provider_event_id = COALESCE(
                                EXCLUDED.provider_event_id,
                                contact_person_suppressions.provider_event_id
                            )
                        """,
                        (candidate_id, provider_event_id),
                    )
                    person_suppressed = True

            # Append-only receipt: who suppressed what, under which issuer.
            cur.execute(
                """
                INSERT INTO contact_suppression_receipts
                    (email_hash, global_candidate_id, reason, scope, evidence_present,
                     tenant_id, issuer, actor_id, actor_type, provider_event_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    email_hash,
                    candidate_id,
                    reason,
                    "person" if person_suppressed else "address",
                    evidence_present,
                    tenant_id,
                    getattr(claims, "issuer", None),
                    getattr(claims, "actor_id", None),
                    getattr(claims, "actor_type", None),
                    provider_event_id,
                ),
            )

            if evidence_present:
                # RLS intentionally limits this maintenance update to the caller's
                # raw evidence; every tenant is nevertheless blocked by the shared
                # hash tombstone (and, for a complaint, the person suppression).
                cur.execute(
                    """
                    UPDATE candidate_contact_evidence
                    SET status = %s, suppressed_at = now(), is_primary = false,
                        bounce_reason = COALESCE(%s, bounce_reason), updated_at = now()
                    WHERE tenant_id = %s AND email_hash = %s
                    """,
                    (reason, reason, tenant_id, email_hash),
                )
                if candidate_id is not None:
                    _choose_primary_contact(
                        cur,
                        tenant_id=tenant_id,
                        candidate_id=str(candidate_id),
                    )
        conn.commit()
        return {
            "suppressed": True,
            "reason": reason,
            "evidence": "present" if evidence_present else "absent",
            "scope": "person" if person_suppressed else "address",
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


@router.post(
    "/public-candidates/exclusions",
    dependencies=[Depends(require_scope("kg:read"))],
)
def public_candidate_exclusions(
    body: PublicMarketExclusionRequest,
    claims=Depends(get_jwt_claims),
):
    """Return fresh, retrievable public Crustdata IDs for one coarse market.

    Rows acquired before migration 021 have no market membership. They are
    included after exact-market members as a bounded transition fallback:
    extra ``not_in`` IDs are harmless because Crustdata applies the actual
    query filters, while omitting a known public ID would repurchase it.
    """
    _require_public_enabled()
    _tenant_from_claims(claims)
    coarse_key = body.coarse_market_key.strip()
    if not coarse_key:
        raise HTTPException(status_code=400, detail="coarse_market_key is required")
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH eligible AS (
                    SELECT gc.id, gc.public_crustdata_person_id,
                           gc.public_profile_observed_at
                    FROM global_candidates gc
                    WHERE gc.public_crustdata_person_id IS NOT NULL
                      AND gc.public_profile_observed_at >=
                          now() - make_interval(days => %s)
                      AND gc.public_embedding_status = 'ready'
                      AND gc.public_embed_version = %s
                      AND gc.public_embedding IS NOT NULL
                      AND EXISTS (
                          SELECT 1 FROM candidate_provenance cp
                          WHERE cp.global_candidate_id = gc.id
                            AND cp.source_type = 'signal_sourced'
                            AND cp.tenant_id IS NULL
                      )
                ),
                candidates AS (
                    SELECT e.*, true AS classified
                    FROM eligible e
                    WHERE EXISTS (
                        SELECT 1 FROM public_candidate_market_memberships pcmm
                        WHERE pcmm.global_candidate_id = e.id
                          AND pcmm.coarse_market_key = %s
                    )
                    UNION ALL
                    SELECT e.*, false AS classified
                    FROM eligible e
                    WHERE NOT EXISTS (
                        SELECT 1 FROM public_candidate_market_memberships pcmm
                        WHERE pcmm.global_candidate_id = e.id
                    )
                )
                SELECT public_crustdata_person_id,
                       count(*) OVER() AS total_matched,
                       count(*) FILTER (WHERE classified) OVER() AS classified_matched,
                       count(*) FILTER (WHERE NOT classified) OVER() AS unclassified_matched,
                       classified
                FROM candidates
                ORDER BY classified DESC, public_profile_observed_at DESC,
                         public_crustdata_person_id
                LIMIT %s
                """,
                (body.fresh_days, PUBLIC_EMBED_VERSION, coarse_key, body.limit),
            )
            raw_rows = cur.fetchall()
        total_matched = int(raw_rows[0][1]) if raw_rows else 0
        classified_matched = int(raw_rows[0][2]) if raw_rows else 0
        unclassified_matched = int(raw_rows[0][3]) if raw_rows else 0
        person_ids = [int(row[0]) for row in raw_rows]
        return {
            "surface": "public_v1",
            "coarse_market_key": coarse_key,
            "crustdata_person_ids": person_ids,
            "total": len(person_ids),
            "total_matched": total_matched,
            "classified_matched": classified_matched,
            "unclassified_matched": unclassified_matched,
            "unclassified_returned": sum(1 for row in raw_rows if not row[4]),
            "truncated": total_matched > len(person_ids),
            "applied_limit": body.limit,
        }
    finally:
        conn.close()


@router.post(
    "/global-candidates/public-identities",
    dependencies=[Depends(require_scope("kg:read"))],
)
def resolve_public_identities(
    body: PublicIdentityLookupRequest,
    claims=Depends(get_jwt_claims),
):
    """Resolve LinkedIn anchors to public canonical IDs without exposing a profile."""
    _require_public_enabled()
    _tenant_from_claims(claims)

    normalized: dict[str, str] = {}
    for raw in body.linkedin_urls:
        try:
            normalized[raw] = normalize_identifier("linkedin_url", raw)
        except IdentifierNormalizationError:
            continue
    if not normalized:
        return {"surface": "public_v1", "results": []}

    normalized_slugs = {
        raw: canonical.rsplit("/", 1)[-1].lower() for raw, canonical in normalized.items()
    }
    slugs = list(dict.fromkeys(normalized_slugs.values()))
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT gc.id, lower(gc.linkedin_id)
                FROM global_candidates gc
                WHERE lower(gc.linkedin_id) = ANY(%s::text[])
                  AND gc.public_profile <> '{}'::jsonb
                  AND EXISTS (
                      SELECT 1 FROM candidate_provenance cp
                      WHERE cp.global_candidate_id = gc.id
                        AND cp.source_type = 'signal_sourced'
                        AND cp.tenant_id IS NULL
                )
                """,
                (slugs,),
            )
            by_slug = {str(row[1]): str(row[0]) for row in cur.fetchall()}
        return {
            "surface": "public_v1",
            "results": [
                {
                    "linkedin_url": raw,
                    "normalized_linkedin_url": canonical,
                    "global_candidate_id": by_slug[normalized_slugs[raw]],
                }
                for raw, canonical in normalized.items()
                if normalized_slugs[raw] in by_slug
            ],
        }
    finally:
        conn.close()


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
    surface: Literal["legacy_v0", "public_v1"] = "legacy_v0"
    location_city: str | None = None
    # Alias expansion from the caller (e.g. ['bengaluru', 'bangalore']) — the
    # caller owns city-alias knowledge; any one matching passes the filter.
    # Takes precedence over location_city when provided.
    location_cities: list[str] | None = None
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

    ``public_v1`` is intentionally public-only. Tenant-private applicants and
    uploads need a tenant-owned embedding/projection before they can safely be
    combined with this search: the historical shared canonical embedding may
    contain evidence contributed by a different tenant after identity
    reconciliation.
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

    public_surface = body.surface == "public_v1"
    if public_surface:
        _require_public_enabled()
    elif not LEGACY_GLOBAL_SEARCH_ENABLED:
        raise HTTPException(
            status_code=410,
            detail="legacy candidate search is disabled; use surface=public_v1",
        )
    embedding_col = "gc.public_embedding" if public_surface else "gc.embedding"
    location_col = "gc.public_location_city" if public_surface else "gc.location_city"
    role_col = "gc.public_role_family" if public_surface else "gc.role_family"
    seniority_col = "gc.public_seniority_band" if public_surface else "gc.seniority_band"
    skills_col = "gc.public_skills_normalized" if public_surface else "gc.skills_normalized"
    status_col = "gc.public_embedding_status" if public_surface else "gc.embedding_status"
    filters: list[str] = [f"{status_col} = 'ready'", f"{embedding_col} IS NOT NULL"]
    params: list[Any] = []
    if public_surface:
        filters.append("gc.public_embed_version = %s")
        params.append(PUBLIC_EMBED_VERSION)
    cities = [
        c.strip()
        for c in (body.location_cities or ([body.location_city] if body.location_city else []))
        if c and c.strip()
    ]
    if cities:
        # Substring match over ALL alias spellings ('Bangalore Urban' must
        # pass a 'Bengaluru' query — 5 of job-147's served members were alias
        # spellings), and NULL/empty passes through: rows whose location
        # simply failed parsing stay retrievable — the caller's ranker already
        # demotes unknown locations and its country guard treats no-location
        # as an escape, so hiding them here silently shrank the pool instead.
        filters.append(
            f"({location_col} IS NULL OR {location_col} = ''"
            f" OR {location_col} ILIKE ANY(%s::text[]))"
        )
        params.append([f"%{c}%" for c in cities])
    if body.role_family:
        filters.append(f"{role_col} = %s")
        params.append(body.role_family)
    if body.seniority_band:
        filters.append(f"{seniority_col} = %s")
        params.append(body.seniority_band)
    if body.skills_any:
        filters.append(f"{skills_col} && %s::text[]")
        params.append([s.lower().strip() for s in body.skills_any])

    if public_surface:
        # A public source payload is the complete authorization condition for
        # this endpoint. Tenant access rows do not qualify private applicants
        # or uploads for a cross-tenant response.
        filters.extend(
            [
                "gc.public_profile <> '{}'::jsonb",
                "EXISTS (SELECT 1 FROM candidate_provenance cp "
                "WHERE cp.global_candidate_id = gc.id "
                "AND cp.source_type = 'signal_sourced' AND cp.tenant_id IS NULL)",
            ]
        )
    else:
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
    if public_surface:
        sql = f"""
            SELECT gc.id,
                   COALESCE(gc.public_profile #>> '{{basic_profile,name}}',
                            gc.public_profile ->> 'name') AS name,
                   gc.public_headline AS headline,
                   gc.linkedin_url, gc.linkedin_id,
                   gc.public_role_family AS role_family,
                   gc.public_seniority_band AS seniority_band,
                   NULL::text[] AS skills_normalized,
                   gc.public_skills_normalized,
                   gc.public_location_city AS location_city,
                   gc.public_location_country_code AS location_country_code,
                   1 - (gc.public_embedding <=> %s::vector) AS similarity,
                   gc.public_profile AS crustdata_profile,
                   NULL::uuid AS tenant_candidate_id,
                   NULL::text AS signal_candidate_id,
                   'public'::text AS evidence_surface
            FROM global_candidates gc
            WHERE {" AND ".join(filters)}
            ORDER BY gc.public_embedding <=> %s::vector
            LIMIT %s
        """
    else:
        sql = f"""
            SELECT gc.id, gc.name, gc.headline, gc.linkedin_url, gc.linkedin_id,
                   gc.role_family, gc.seniority_band, gc.skills_normalized,
                   gc.location_city, gc.location_country_code,
                   1 - (gc.embedding <=> %s::vector) AS similarity,
                   tc.profile AS crustdata_profile,
                   tc.candidate_id AS tenant_candidate_id,
                   tc.signal_candidate_id,
                   'legacy'::text AS evidence_surface
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
        return {
            "surface": "public_v1" if public_surface else "legacy_v0",
            "results": rows,
            "count": len(rows),
            "applied_limit": limit,
        }
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
