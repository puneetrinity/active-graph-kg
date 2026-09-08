"""Approved-provider sourced-candidate ingestion into global Memory.

The endpoint accepts one strict, provider-neutral professional-profile
observation from the verified Signal service.  It intentionally does not
create tenant candidate state: ``global_candidates.id`` remains the canonical
platform person and the new migration-025 relations are its immutable source
authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Literal
from urllib.parse import urlparse
from uuid import UUID

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from psycopg.errors import UniqueViolation
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from typing_extensions import Self

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.common.logger import get_enhanced_logger
from activekg.graph.candidate_identifiers import IdentifierNormalizationError, normalize_identifier
from activekg.privacy.config import load_candidate_privacy_config
from activekg.privacy.identity import identity_token, normalize_privacy_identifier
from activekg.privacy.models import CandidatePrivacyDecision, decision_for
from activekg.privacy.repository import CandidatePrivacyRestricted, require_allowed

router = APIRouter(tags=["sourced-candidates"])
sourced_candidates_router = router
logger = get_enhanced_logger(__name__)

_MAX_BODY_BYTES = 256 * 1024
_MAX_PROFILE_BYTES = 192 * 1024
_TENANT_RE = re.compile(r"^org_[1-9][0-9]*$")
_RECEIPT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_EMAIL_RE = re.compile(r"(?i)(?<![a-z0-9._%+-])[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+[1-9][0-9 ()-]{7,20}|[0-9]{9,15})(?!\w)")
_FORBIDDEN_KEY_PARTS = (
    "email",
    "phone",
    "contact",
    "resume",
    "application",
    "shortlist",
    "campaign",
    "outreach",
    "interview",
    "decision",
    "recruiter",
    "tenant",
    "organization",
    "job_id",
    "query",
    "rank",
    "model_output",
    "raw_provider",
)

_ADAPTER = {("crustdata", "person", "crustdata_person_v1"): ("crustdata_person", 1)}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PublicLocation(_StrictModel):
    city: str | None = Field(default=None, min_length=1, max_length=160)
    state: str | None = Field(default=None, min_length=1, max_length=160)
    country: str | None = Field(default=None, min_length=1, max_length=160)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    continent: str | None = Field(default=None, min_length=1, max_length=80)
    full_location: str | None = Field(default=None, min_length=1, max_length=500)


class EmploymentEntry(_StrictModel):
    company_name: str | None = Field(default=None, min_length=1, max_length=300)
    title: str | None = Field(default=None, min_length=1, max_length=300)
    seniority_level: str | None = Field(default=None, min_length=1, max_length=100)
    function_category: str | None = Field(default=None, min_length=1, max_length=100)
    start_date: str | None = Field(default=None, min_length=1, max_length=32)
    end_date: str | None = Field(default=None, min_length=1, max_length=32)
    description: str | None = Field(default=None, min_length=1, max_length=4000)
    years_at_company: float | None = Field(default=None, ge=0, le=100)
    company_headquarters_country: str | None = Field(default=None, min_length=1, max_length=160)
    company_industries: list[str] = Field(default_factory=list, max_length=32)
    company_network_industry: str | None = Field(default=None, min_length=1, max_length=200)
    company_type: str | None = Field(default=None, min_length=1, max_length=100)
    company_headcount_range: str | None = Field(default=None, min_length=1, max_length=100)

    @field_validator("company_industries")
    @classmethod
    def validate_industries(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("company industry is invalid")
        return values


class EducationEntry(_StrictModel):
    school: str | None = Field(default=None, min_length=1, max_length=300)
    degree: str | None = Field(default=None, min_length=1, max_length=200)
    field_of_study: str | None = Field(default=None, min_length=1, max_length=200)
    start_year: int | None = Field(default=None, ge=1900, le=2200)
    end_year: int | None = Field(default=None, ge=1900, le=2200)


class CertificationEntry(_StrictModel):
    name: str = Field(min_length=1, max_length=300)
    issuing_organization: str | None = Field(default=None, min_length=1, max_length=300)
    issue_date: str | None = Field(default=None, min_length=1, max_length=32)
    expiration_date: str | None = Field(default=None, min_length=1, max_length=32)


class HonorEntry(_StrictModel):
    title: str = Field(min_length=1, max_length=300)
    issuer: str | None = Field(default=None, min_length=1, max_length=300)
    description: str | None = Field(default=None, min_length=1, max_length=2000)


class ProfessionalProfileReference(_StrictModel):
    platform: Literal["linkedin", "github", "twitter"]
    profile_url: str = Field(min_length=8, max_length=2048)

    @model_validator(mode="after")
    def validate_supported_host(self) -> Self:
        parsed = urlparse(self.profile_url)
        host = (parsed.hostname or "").lower().removeprefix("www.")
        expected = {
            "linkedin": ("linkedin.com",),
            "github": ("github.com",),
            "twitter": ("twitter.com", "x.com"),
        }[self.platform]
        if parsed.scheme != "https" or host not in expected or not parsed.path.strip("/"):
            raise ValueError("professional profile reference is invalid")
        return self


class NormalizedProfessionalProfile(_StrictModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=300)
    given_name: str | None = Field(default=None, min_length=1, max_length=160)
    family_name: str | None = Field(default=None, min_length=1, max_length=160)
    headline: str | None = Field(default=None, min_length=1, max_length=500)
    current_title: str | None = Field(default=None, min_length=1, max_length=300)
    public_picture_url: str | None = Field(default=None, min_length=8, max_length=2048)
    professional_summary: str | None = Field(default=None, min_length=1, max_length=8000)
    languages: list[str] = Field(default_factory=list, max_length=32)
    location: PublicLocation | None = None
    role_family: str | None = Field(
        default=None, min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9_-]{0,99}$"
    )
    seniority_band: str | None = Field(default=None, min_length=1, max_length=100)
    years_of_experience: float | None = Field(default=None, ge=0, le=100)
    recently_changed_jobs: bool | None = None
    skills: list[str] = Field(default_factory=list, max_length=256)
    current_employment: list[EmploymentEntry] = Field(default_factory=list, max_length=8)
    past_employment: list[EmploymentEntry] = Field(default_factory=list, max_length=64)
    education: list[EducationEntry] = Field(default_factory=list, max_length=32)
    certifications: list[CertificationEntry] = Field(default_factory=list, max_length=64)
    honors: list[HonorEntry] = Field(default_factory=list, max_length=64)
    public_profiles: list[ProfessionalProfileReference] = Field(default_factory=list, max_length=8)

    @field_validator("languages", "skills")
    @classmethod
    def validate_bounded_strings(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 200 for value in values):
            raise ValueError("bounded profile string is invalid")
        return values

    @field_validator("public_picture_url")
    @classmethod
    def validate_picture_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlparse(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("public picture URL is invalid")
        return value

    @model_validator(mode="after")
    def reject_private_evidence(self) -> Self:
        _assert_no_private_evidence(self.model_dump(mode="json"))
        size = len(_canonical_json(self.model_dump(mode="json")).encode("utf-8"))
        if size > _MAX_PROFILE_BYTES:
            raise ValueError("normalized profile is too large")
        return self


class PublicMarketObservation(_StrictModel):
    version: Literal[1]
    coarse_market_key: str = Field(pattern=r"^public-market:v1:[0-9a-f]{64}$")
    role_family: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9][a-z0-9_-]{0,99}$")
    location_city: str = Field(min_length=1, max_length=160)
    location_country_code: str = Field(pattern=r"^[A-Z]{2}$")
    seniority_band: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_key(self) -> Self:
        if self.location_city != self.location_city.lower():
            raise ValueError("public market city is not canonical")
        if self.seniority_band != self.seniority_band.lower():
            raise ValueError("public market seniority is not canonical")
        material = {
            "version": 1,
            "roleFamily": self.role_family,
            "locationCity": self.location_city,
            "locationCountryCode": self.location_country_code,
            "seniorityBand": self.seniority_band,
        }
        expected = hashlib.sha256(
            json.dumps(
                material, sort_keys=False, separators=(",", ":"), ensure_ascii=False
            ).encode()
        ).hexdigest()
        if self.coarse_market_key != f"public-market:v1:{expected}":
            raise ValueError("public market key is invalid")
        return self


class SourcedCandidateIngestRequest(_StrictModel):
    schema_version: Literal[1]
    provider_namespace: Literal["crustdata"]
    record_type: Literal["person"]
    adapter_version: Literal["crustdata_person_v1"]
    provider_record_id: str = Field(pattern=r"^[1-9][0-9]{0,18}$")
    linkedin_url: str = Field(min_length=1, max_length=2048)
    expected_global_candidate_id: UUID | None = None
    acquisition_receipt_id: str = Field(min_length=1, max_length=200)
    acquisition_generation: int = Field(ge=1, le=2_147_483_647)
    acquisition_slot: Literal["exact", "spill"]
    acquired_at: datetime
    provider_observed_at: datetime
    idempotency_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_profile: NormalizedProfessionalProfile
    public_market: PublicMarketObservation | None = None

    @field_validator("linkedin_url")
    @classmethod
    def validate_canonical_linkedin(cls, value: str) -> str:
        try:
            normalized = normalize_identifier("linkedin_url", value)
        except (IdentifierNormalizationError, ValueError) as exc:
            raise ValueError("LinkedIn URL is invalid") from exc
        if value != normalized or not value.startswith("https://linkedin.com/in/"):
            raise ValueError("LinkedIn URL is not canonical")
        return value

    @field_validator("acquisition_receipt_id")
    @classmethod
    def validate_receipt(cls, value: str) -> str:
        if _RECEIPT_RE.fullmatch(value) is None:
            raise ValueError("acquisition receipt is invalid")
        return value

    @model_validator(mode="after")
    def validate_times_and_idempotency(self) -> Self:
        for value in (self.acquired_at, self.provider_observed_at):
            if value.tzinfo is None:
                raise ValueError("observation timestamps require timezone")
            if value.astimezone(timezone.utc) > datetime.now(timezone.utc) + timedelta(minutes=5):
                raise ValueError("observation timestamp is in the future")
        if self.idempotency_key != compute_idempotency_key(self):
            raise ValueError("idempotency key does not match canonical acquisition identity")
        return self


def sourced_candidate_ingest_mode() -> Literal["off", "dual", "canonical_only"]:
    mode = os.getenv("SOURCED_CANDIDATE_INGEST_MODE", "off").strip()
    if mode not in {"off", "dual", "canonical_only"}:
        raise HTTPException(
            status_code=503, detail="sourced_candidate_ingest_configuration_invalid"
        )
    return mode  # type: ignore[return-value]


async def require_sourced_candidate_writer(
    claims: JWTClaims | None = Depends(get_jwt_claims),
) -> JWTClaims:
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(status_code=401, detail="sourced_candidate_service_auth_required")
    if (
        claims.issuer != auth.SIGNAL_JWT_ISSUER
        or claims.actor_type != "service"
        or claims.actor_id != "signal-service"
        or claims.scopes != ["candidate-source:write"]
        or _TENANT_RE.fullmatch(claims.tenant_id) is None
    ):
        raise HTTPException(status_code=403, detail="sourced_candidate_service_auth_denied")
    return claims


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalized_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", key.lower())


def _assert_no_private_evidence(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = _normalized_key(key)
            if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                raise ValueError("private profile field is forbidden")
            _assert_no_private_evidence(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_private_evidence(nested)
    elif isinstance(value, str):
        url_value = value.startswith(("https://", "http://"))
        if _EMAIL_RE.search(value) or (not url_value and _PHONE_RE.search(value)):
            raise ValueError("contact-like profile value is forbidden")


def compute_idempotency_key(payload: SourcedCandidateIngestRequest) -> str:
    components = (
        "v1",
        payload.provider_namespace,
        payload.record_type,
        payload.provider_record_id,
        payload.acquisition_receipt_id,
        str(payload.acquisition_generation),
        payload.acquisition_slot,
    )
    return hashlib.sha256("\0".join(components).encode("utf-8")).hexdigest()


def _canonical_input(payload: SourcedCandidateIngestRequest) -> tuple[str, str, str]:
    material = payload.model_dump(mode="json", exclude={"idempotency_key"})
    canonical = _canonical_json(material)
    profile = _canonical_json(payload.normalized_profile.model_dump(mode="json"))
    return (
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        profile,
        hashlib.sha256(profile.encode("utf-8")).hexdigest(),
    )


async def _parse_body(request: Request) -> SourcedCandidateIngestRequest:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > _MAX_BODY_BYTES:
                raise HTTPException(status_code=413, detail="sourced_candidate_request_too_large")
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="sourced_candidate_request_invalid"
            ) from exc
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > _MAX_BODY_BYTES:
            raise HTTPException(status_code=413, detail="sourced_candidate_request_too_large")
    try:
        return SourcedCandidateIngestRequest.model_validate_json(bytes(body))
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="sourced_candidate_request_invalid") from exc


def _connect() -> psycopg.Connection:
    dsn = (os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL") or "").strip()
    if not dsn:
        raise RuntimeError("sourced candidate database unavailable")
    return psycopg.connect(dsn, autocommit=False)


def _privacy_tokens(linkedin_url: str) -> list[dict[str, Any]]:
    config = load_candidate_privacy_config(require_hmac=True)
    _active, keys = config.require_hmac()
    normalized = normalize_privacy_identifier("linkedin_url", linkedin_url)
    return [
        {
            "identifier_type": normalized.identifier_type,
            "key_version": version,
            "token": identity_token(key, normalized).hex(),
        }
        for version, key in sorted(keys.items())
    ]


def _require_privacy_for_candidate(
    cur: psycopg.Cursor,
    *,
    linkedin_url: str,
    global_candidate_id: str | UUID | None,
) -> None:
    try:
        tokens = _privacy_tokens(linkedin_url)
        cur.execute(
            """
            SELECT action,state,decision
            FROM candidate_privacy_match(%s::jsonb,%s,NULL,NULL)
            ORDER BY CASE decision WHEN 'review' THEN 4 WHEN 'block_all' THEN 3
                     WHEN 'block_global' THEN 2 ELSE 1 END DESC,
                     effective_at DESC,version DESC
            LIMIT 1
            """,
            (json.dumps(tokens), global_candidate_id),
        )
        row = cur.fetchone()
        if row is None:
            decision = CandidatePrivacyDecision.ALLOW
        else:
            decision = decision_for(row[1], row[0])
            if decision is not CandidatePrivacyDecision(row[2]):
                raise RuntimeError("candidate privacy matcher is inconsistent")
        require_allowed(decision, global_use=True)
    except CandidatePrivacyRestricted as exc:
        raise HTTPException(status_code=451, detail="candidate_privacy_restricted") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail="candidate_privacy_unavailable") from exc


def _linkedin_slug(linkedin_url: str) -> str:
    return linkedin_url.removeprefix("https://linkedin.com/in/")


def _row_dict(cur: psycopg.Cursor, row: tuple[Any, ...] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(zip((item.name for item in cur.description), row, strict=True))


def _receipt_response(row: dict[str, Any], delivery_status: str) -> dict[str, Any]:
    return {
        "delivery_status": delivery_status,
        "resolution": row["resolution"],
        "provider_namespace": row["provider_namespace"],
        "record_type": row["record_type"],
        "provider_record_id": row["provider_record_id"],
        "acquisition_receipt_id": row["acquisition_receipt_id"],
        "acquisition_generation": int(row["acquisition_generation"]),
        "acquisition_slot": row["acquisition_slot"],
        "idempotency_key": row["idempotency_key"],
        "source_observation_id": str(row["source_observation_id"]),
        "ingest_receipt_id": row["idempotency_key"],
        "source_identity_id": (
            str(row["source_identity_id"]) if row["source_identity_id"] else None
        ),
        "global_candidate_id": (
            str(row["global_candidate_id"]) if row["global_candidate_id"] else None
        ),
    }


def _select_receipt(cur: psycopg.Cursor, idempotency_key: str) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT idempotency_key,input_digest,source_observation_id,source_identity_id,
               global_candidate_id,resolution,provider_namespace,record_type,
               provider_record_id,acquisition_receipt_id,acquisition_generation,
               acquisition_slot
        FROM global_candidate_ingest_receipts WHERE idempotency_key=%s
        """,
        (idempotency_key,),
    )
    return _row_dict(cur, cur.fetchone())


def _compatibility_profile(
    payload: SourcedCandidateIngestRequest,
) -> dict[str, Any]:
    p = payload.normalized_profile
    basic: dict[str, Any] = {
        key: value
        for key, value in {
            "name": p.display_name,
            "first_name": p.given_name,
            "last_name": p.family_name,
            "headline": p.headline,
            "current_title": p.current_title,
            "profile_picture_permalink": p.public_picture_url,
            "summary": p.professional_summary,
            "languages": p.languages or None,
            "location": p.location.model_dump(exclude_none=True) if p.location else None,
        }.items()
        if value not in (None, [], {})
    }
    network = {"profile_picture_permalink": p.public_picture_url} if p.public_picture_url else {}
    experience = {
        "employment_details": {
            "current": [entry.model_dump(exclude_none=True) for entry in p.current_employment],
            "past": [entry.model_dump(exclude_none=True) for entry in p.past_employment],
        }
    }
    social: dict[str, Any] = {}
    for reference in p.public_profiles:
        key = {
            "linkedin": "professional_network_identifier",
            "github": "dev_platform_identifier",
            "twitter": "twitter_identifier",
        }[reference.platform]
        social[key] = {"profile_url": reference.profile_url}
    result: dict[str, Any] = {
        "crustdata_person_id": int(payload.provider_record_id),
        "metadata": {"updated_at": payload.provider_observed_at.isoformat()},
        "basic_profile": basic,
        "professional_network": network,
        "experience": experience,
        "education": {"schools": [entry.model_dump(exclude_none=True) for entry in p.education]},
        "skills": {"professional_network_skills": p.skills},
        "certifications": [entry.model_dump(exclude_none=True) for entry in p.certifications],
        "honors": [entry.model_dump(exclude_none=True) for entry in p.honors],
        "social_handles": social,
        "years_of_experience_raw": p.years_of_experience,
        "recently_changed_jobs": p.recently_changed_jobs,
    }
    return {key: value for key, value in result.items() if value not in (None, [], {})}


def _conflict_code(
    source_id: str | None,
    linkedin_ids: list[str],
    expected_id: str | None,
    expected_exists: bool,
    identity_linkedin_url: str | None,
    incoming_linkedin_url: str,
) -> str | None:
    if len(linkedin_ids) > 1:
        return "linkedin_ambiguous"
    linkedin_id = linkedin_ids[0] if linkedin_ids else None
    if expected_id and not expected_exists:
        return "invalid_identity_state"
    if (
        source_id
        and identity_linkedin_url is not None
        and identity_linkedin_url != incoming_linkedin_url
    ):
        return "provider_linkedin_mismatch"
    if source_id and linkedin_id and source_id != linkedin_id:
        return "provider_linkedin_mismatch"
    if source_id and expected_id and source_id != expected_id:
        return "provider_expected_mismatch"
    if linkedin_id and expected_id and linkedin_id != expected_id:
        return "linkedin_expected_mismatch"
    return None


def _enqueue_conflict(
    cur: psycopg.Cursor,
    candidate_ids: list[str],
    code: str,
) -> None:
    ordered = sorted(set(candidate_ids))
    if not ordered:
        return
    a_id = ordered[0]
    b_id = ordered[1] if len(ordered) > 1 else None
    cur.execute(
        """
        INSERT INTO candidate_merge_queue
            (global_candidate_id_a,global_candidate_id_b,tenant_id,reason,details)
        VALUES (%s,%s,NULL,'anchor_conflict',%s::jsonb)
        ON CONFLICT DO NOTHING
        """,
        (a_id, b_id, json.dumps({"source": "approved_provider", "code": code})),
    )


def _store(
    payload: SourcedCandidateIngestRequest,
    claims: JWTClaims,
) -> dict[str, Any]:
    input_digest, profile_json, profile_digest = _canonical_input(payload)
    adapter_family, adapter_version = _ADAPTER[
        (payload.provider_namespace, payload.record_type, payload.adapter_version)
    ]
    conn = _connect()
    started = time.monotonic()

    def check_deadline() -> None:
        if time.monotonic() - started > 3:
            raise TimeoutError("sourced candidate transaction deadline exceeded")

    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='500ms'")
            cur.execute("SET LOCAL statement_timeout='2000ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='3000ms'")
            cur.execute("SELECT set_config('app.current_tenant_id', %s, true)", (claims.tenant_id,))

            receipt = _select_receipt(cur, payload.idempotency_key)
            if receipt is not None:
                if receipt["input_digest"] != input_digest:
                    raise HTTPException(
                        status_code=409, detail="sourced_candidate_idempotency_conflict"
                    )
                conn.rollback()
                return _receipt_response(receipt, "replayed")

            lock_keys = sorted(
                {
                    f"provider:{payload.provider_namespace}:{payload.record_type}:{payload.provider_record_id}",
                    f"linkedin:{payload.linkedin_url}",
                }
            )
            for key in lock_keys:
                cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (key,))
            check_deadline()

            receipt = _select_receipt(cur, payload.idempotency_key)
            if receipt is not None:
                if receipt["input_digest"] != input_digest:
                    raise HTTPException(
                        status_code=409, detail="sourced_candidate_idempotency_conflict"
                    )
                conn.rollback()
                return _receipt_response(receipt, "replayed")

            cur.execute(
                """
                SELECT id::text,global_candidate_id::text,canonical_linkedin_url
                FROM global_candidate_source_identities
                WHERE provider_namespace=%s AND record_type=%s AND provider_record_id=%s
                """,
                (payload.provider_namespace, payload.record_type, payload.provider_record_id),
            )
            source_row = cur.fetchone()
            source_identity_id = str(source_row[0]) if source_row else None
            source_candidate_id = str(source_row[1]) if source_row else None
            identity_linkedin_url = str(source_row[2]) if source_row else None
            if source_row is None:
                # Migration 021's exact Crustdata id is the pre-025
                # compatibility anchor. Adopt it once without manufacturing a
                # historical source observation or rewriting the old row.
                cur.execute(
                    """
                    SELECT id::text FROM global_candidates
                    WHERE public_crustdata_person_id=%s
                    """,
                    (int(payload.provider_record_id),),
                )
                legacy_provider_row = cur.fetchone()
                if legacy_provider_row is not None:
                    source_candidate_id = str(legacy_provider_row[0])

            slug = _linkedin_slug(payload.linkedin_url)
            cur.execute(
                """
                SELECT DISTINCT id::text FROM global_candidates
                WHERE linkedin_url=%s OR linkedin_id=%s ORDER BY id::text
                """,
                (payload.linkedin_url, slug),
            )
            linkedin_ids = [str(row[0]) for row in cur.fetchall()]

            expected_id = (
                str(payload.expected_global_candidate_id)
                if payload.expected_global_candidate_id
                else None
            )
            expected_exists = False
            expected_anchor_conflict = False
            if expected_id:
                cur.execute(
                    "SELECT linkedin_url,linkedin_id FROM global_candidates WHERE id=%s",
                    (expected_id,),
                )
                expected_row = cur.fetchone()
                expected_exists = expected_row is not None
                if expected_row is not None:
                    expected_anchors: set[str] = set()
                    for anchor in (
                        expected_row[0],
                        (f"https://linkedin.com/in/{expected_row[1]}" if expected_row[1] else None),
                    ):
                        if anchor is None:
                            continue
                        try:
                            expected_anchors.add(normalize_identifier("linkedin_url", str(anchor)))
                        except (IdentifierNormalizationError, ValueError):
                            expected_anchor_conflict = True
                    if len(expected_anchors) > 1 or (
                        expected_anchors and payload.linkedin_url not in expected_anchors
                    ):
                        expected_anchor_conflict = True

            code = _conflict_code(
                source_candidate_id,
                linkedin_ids,
                expected_id,
                expected_exists,
                identity_linkedin_url,
                payload.linkedin_url,
            )
            if code is None and expected_anchor_conflict:
                code = "linkedin_expected_mismatch"
            all_candidates = [
                value
                for value in (
                    source_candidate_id,
                    *linkedin_ids,
                    expected_id if expected_exists else None,
                )
                if value
            ]
            for candidate_id in sorted(set(all_candidates)) or [None]:
                _require_privacy_for_candidate(
                    cur,
                    linkedin_url=payload.linkedin_url,
                    global_candidate_id=candidate_id,
                )
            check_deadline()

            created = False
            candidate_id: str | None = None
            resolution: Literal[
                "created", "matched", "refreshed", "stale", "conflict_review_required"
            ]
            outcome: Literal["accepted", "stale", "conflict_review_required"]
            if code:
                resolution = "conflict_review_required"
                outcome = "conflict_review_required"
                source_identity_id = None
                _enqueue_conflict(cur, all_candidates, code)
            else:
                candidate_id = (
                    source_candidate_id
                    or (linkedin_ids[0] if linkedin_ids else None)
                    or expected_id
                )
                p = payload.normalized_profile
                compatibility = _compatibility_profile(payload)
                if candidate_id is None:
                    cur.execute(
                        """
                        INSERT INTO global_candidates (
                            linkedin_id,linkedin_url,name,headline,location_city,
                            location_country_code,role_family,seniority_band,skills_normalized,
                            identity_confidence,merge_status,embedding_status,first_seen_at,
                            last_evidence_at,created_at,updated_at,public_profile,
                            public_profile_observed_at,public_crustdata_person_id,public_headline,
                            public_location_city,public_location_country_code,public_role_family,
                            public_seniority_band,public_skills_normalized,public_embedding_status,
                            public_embed_version
                        ) VALUES (
                            %s,%s,%s,%s,%s,%s,%s,%s,%s,1.0,'single','queued',%s,%s,
                            clock_timestamp(),clock_timestamp(),%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s,
                            'queued',0
                        ) RETURNING id::text
                        """,
                        (
                            slug,
                            payload.linkedin_url,
                            p.display_name,
                            p.headline,
                            p.location.city if p.location else None,
                            p.location.country_code if p.location else None,
                            p.role_family,
                            p.seniority_band,
                            p.skills or None,
                            payload.acquired_at,
                            payload.provider_observed_at,
                            json.dumps(compatibility),
                            payload.provider_observed_at,
                            int(payload.provider_record_id),
                            p.headline,
                            p.location.city if p.location else None,
                            p.location.country_code if p.location else None,
                            p.role_family,
                            p.seniority_band,
                            p.skills or None,
                        ),
                    )
                    candidate_id = str(cur.fetchone()[0])
                    created = True

                if source_identity_id is None:
                    cur.execute(
                        """
                        INSERT INTO global_candidate_source_identities (
                            provider_namespace,record_type,adapter_family,adapter_version,
                            provider_record_id,canonical_linkedin_url,global_candidate_id,
                            verified_issuer,verified_actor_id,first_observed_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        RETURNING id::text
                        """,
                        (
                            payload.provider_namespace,
                            payload.record_type,
                            adapter_family,
                            adapter_version,
                            payload.provider_record_id,
                            payload.linkedin_url,
                            candidate_id,
                            claims.issuer,
                            claims.actor_id,
                            payload.acquired_at,
                        ),
                    )
                    source_identity_id = str(cur.fetchone()[0])

                cur.execute(
                    """
                    SELECT MAX(o.provider_observed_at),MAX(g.public_profile_observed_at)
                    FROM global_candidates g
                    LEFT JOIN global_candidate_source_observations o
                      ON o.global_candidate_id=g.id AND o.outcome <> 'conflict_review_required'
                    WHERE g.id=%s
                    """,
                    (candidate_id,),
                )
                observation_watermark, compatibility_watermark = cur.fetchone()
                watermarks = [
                    value
                    for value in (observation_watermark, compatibility_watermark)
                    if value is not None
                ]
                prior_watermark = max(watermarks) if watermarks else None
                incoming_newer = (
                    prior_watermark is None or payload.provider_observed_at > prior_watermark
                )
                if created:
                    resolution = "created"
                    outcome = "accepted"
                elif not incoming_newer:
                    resolution = "stale"
                    outcome = "stale"
                elif prior_watermark is None:
                    resolution = "matched"
                    outcome = "accepted"
                else:
                    resolution = "refreshed"
                    outcome = "accepted"

                if outcome == "accepted" and not created:
                    cur.execute(
                        """
                        UPDATE global_candidates SET
                            public_profile=%s::jsonb,
                            public_profile_observed_at=%s,
                            public_crustdata_person_id=%s,
                            public_headline=%s,
                            public_location_city=%s,
                            public_location_country_code=%s,
                            public_role_family=%s,
                            public_seniority_band=%s,
                            public_skills_normalized=%s,
                            public_embedding=NULL,
                            public_embedding_status='queued',
                            public_embed_version=0,
                            last_evidence_at=GREATEST(last_evidence_at,%s),
                            updated_at=clock_timestamp()
                        WHERE id=%s AND (
                            public_profile_observed_at IS NULL
                            OR public_profile_observed_at < %s
                        )
                        """,
                        (
                            json.dumps(compatibility),
                            payload.provider_observed_at,
                            int(payload.provider_record_id),
                            p.headline,
                            p.location.city if p.location else None,
                            p.location.country_code if p.location else None,
                            p.role_family,
                            p.seniority_band,
                            p.skills or None,
                            payload.provider_observed_at,
                            candidate_id,
                            payload.provider_observed_at,
                        ),
                    )

            cur.execute(
                """
                INSERT INTO global_candidate_source_observations (
                    idempotency_key,source_identity_id,global_candidate_id,
                    provider_namespace,record_type,adapter_family,adapter_version,
                    provider_record_id,canonical_linkedin_url,expected_global_candidate_id,
                    acquisition_receipt_id,acquisition_generation,acquisition_slot,acquired_at,
                    provider_observed_at,schema_version,normalized_profile,profile_digest,outcome,
                    conflict_code,verified_issuer,verified_actor_id
                ) VALUES (
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,%s::jsonb,%s,%s,%s,%s,%s
                ) RETURNING id::text
                """,
                (
                    payload.idempotency_key,
                    source_identity_id,
                    candidate_id,
                    payload.provider_namespace,
                    payload.record_type,
                    adapter_family,
                    adapter_version,
                    payload.provider_record_id,
                    payload.linkedin_url,
                    expected_id,
                    payload.acquisition_receipt_id,
                    payload.acquisition_generation,
                    payload.acquisition_slot,
                    payload.acquired_at,
                    payload.provider_observed_at,
                    profile_json,
                    profile_digest,
                    outcome,
                    code,
                    claims.issuer,
                    claims.actor_id,
                ),
            )
            observation_id = str(cur.fetchone()[0])

            if candidate_id is not None:
                cur.execute(
                    """
                    INSERT INTO candidate_provenance (
                        global_candidate_id,source_type,tenant_id,source_detail
                    ) VALUES (%s,'signal_sourced',NULL,'{}'::jsonb)
                    ON CONFLICT (global_candidate_id,source_type)
                        WHERE tenant_id IS NULL DO NOTHING
                    """,
                    (candidate_id,),
                )
                if outcome == "accepted" and payload.public_market is not None:
                    market = payload.public_market
                    cur.execute(
                        """
                        INSERT INTO public_candidate_market_memberships (
                            global_candidate_id,coarse_market_key,role_family,location_city,
                            location_country_code,seniority_band,first_observed_at,last_observed_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                        ON CONFLICT (global_candidate_id,coarse_market_key) DO UPDATE SET
                            last_observed_at=GREATEST(
                                public_candidate_market_memberships.last_observed_at,
                                EXCLUDED.last_observed_at
                            )
                        """,
                        (
                            candidate_id,
                            market.coarse_market_key,
                            market.role_family,
                            market.location_city,
                            market.location_country_code,
                            market.seniority_band,
                            payload.provider_observed_at,
                            payload.provider_observed_at,
                        ),
                    )

            cur.execute(
                """
                INSERT INTO global_candidate_ingest_receipts (
                    idempotency_key,input_digest,source_observation_id,source_identity_id,
                    global_candidate_id,resolution,provider_namespace,record_type,
                    provider_record_id,acquisition_receipt_id,acquisition_generation,
                    acquisition_slot,verified_issuer,verified_actor_id
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                RETURNING idempotency_key,input_digest,source_observation_id,
                          source_identity_id,global_candidate_id,resolution,
                          provider_namespace,record_type,provider_record_id,
                          acquisition_receipt_id,acquisition_generation,acquisition_slot
                """,
                (
                    payload.idempotency_key,
                    input_digest,
                    observation_id,
                    source_identity_id,
                    candidate_id,
                    resolution,
                    payload.provider_namespace,
                    payload.record_type,
                    payload.provider_record_id,
                    payload.acquisition_receipt_id,
                    payload.acquisition_generation,
                    payload.acquisition_slot,
                    claims.issuer,
                    claims.actor_id,
                ),
            )
            receipt = _row_dict(cur, cur.fetchone())
            assert receipt is not None
            check_deadline()
        conn.commit()
        return _receipt_response(receipt, "recorded")
    except HTTPException:
        conn.rollback()
        raise
    except UniqueViolation as exc:
        conn.rollback()
        raise HTTPException(status_code=409, detail="sourced_candidate_identity_conflict") from exc
    except Exception as exc:
        conn.rollback()
        logger.warning(
            "Sourced candidate ingest failed",
            extra_fields={"error_type": type(exc).__name__},
        )
        raise HTTPException(status_code=503, detail="sourced_candidate_ingest_unavailable") from exc
    finally:
        conn.close()


@router.post("/sourced-candidates/ingest", response_model=None)
async def ingest_sourced_candidate(
    request: Request,
    claims: Annotated[JWTClaims, Depends(require_sourced_candidate_writer)],
) -> dict[str, Any]:
    if sourced_candidate_ingest_mode() == "off":
        raise HTTPException(status_code=503, detail="sourced_candidate_ingest_disabled")
    payload = await _parse_body(request)
    if (payload.provider_namespace, payload.record_type, payload.adapter_version) not in _ADAPTER:
        raise HTTPException(status_code=422, detail="sourced_candidate_adapter_denied")
    return _store(payload, claims)
