"""Candidate-approved snapshots and ordered consent, never a search/embedding writer."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from datetime import datetime
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import psycopg
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from typing_extensions import Self

from activekg.api import auth
from activekg.api.auth import JWTClaims, get_jwt_claims
from activekg.privacy.config import load_candidate_privacy_config
from activekg.privacy.identity import identity_token, normalize_privacy_identifier
from activekg.privacy.models import CandidatePrivacyDecision, decision_for
from activekg.privacy.repository import CandidatePrivacyRestricted, require_allowed

router = APIRouter(tags=["candidate-consent"])
PURPOSE = "platform_professional_matching"
COPY = "I allow Ealana to use the professional profile shown here, and the resume version I select, for matching me with opportunities across organizations. This is optional and separate from my applications. I can withdraw this permission later. Applications already submitted stay with those organizations. Independently sourced public professional information is controlled separately through Privacy & Data."
COPY_SHA256 = hashlib.sha256(COPY.encode()).hexdigest()
_UUID = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
_TENANT = re.compile(r"^candidate_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_SAFE_INT = 9007199254740991
Uuid = Annotated[str, Field(pattern=_UUID)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Positive = Annotated[int, Field(ge=1, le=_SAFE_INT)]
Int4 = Annotated[int, Field(ge=1, le=2147483647)]


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _text(value: str, minimum: int, maximum: int) -> str:
    if any(ord(c) < 32 or 127 <= ord(c) <= 159 or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError("invalid scalar text")
    normalized = unicodedata.normalize("NFC", value).strip()
    if not minimum <= len(normalized) <= maximum:
        raise ValueError("invalid text length")
    return normalized


def _timestamp(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", value):
        raise ValueError("invalid timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z") != value:
        raise ValueError("invalid timestamp")
    return value


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Profile(StrictModel):
    display_name: str
    headline: str
    location: str
    skills: list[str] = Field(max_length=100)
    linkedin: str | None

    @field_validator("display_name", "headline", "location")
    @classmethod
    def bounded_text(cls, value: str, info: Any) -> str:
        bounds = {"display_name": (1, 200), "headline": (0, 300), "location": (0, 200)}
        return _text(value, *bounds[info.field_name])

    @field_validator("skills")
    @classmethod
    def bounded_skills(cls, value: list[str]) -> list[str]:
        values = [_text(skill, 1, 100) for skill in value]
        if len(set(values)) != len(values):
            raise ValueError("duplicate skill")
        return values

    @field_validator("linkedin")
    @classmethod
    def linkedin_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        url = urlsplit(_text(value, 1, 2048))
        if (
            url.scheme != "https"
            or url.hostname not in {"linkedin.com", "www.linkedin.com"}
            or url.username
            or url.password
            or url.port
            or url.query
            or url.fragment
            or not re.fullmatch(r"/in/[a-zA-Z0-9_%.-]+/?", url.path)
        ):
            raise ValueError("invalid professional URL")
        return "https://www.linkedin.com" + url.path.rstrip("/")

    @model_validator(mode="after")
    def bounded_object(self) -> Self:
        if len(compact(self.model_dump()).encode()) > 32768:
            raise ValueError("profile too large")
        return self


class Resume(StrictModel):
    reference_id: Uuid
    resume_version_id: Uuid
    organization_id: Int4
    application_id: Int4
    job_id: Int4
    content_sha256: Digest
    byte_count: int = Field(ge=1, le=5242880)
    media_type: Literal[
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    source_observed_at: str
    _time = field_validator("source_observed_at")(_timestamp)

    def ordered(self) -> list[Any]:
        return [
            self.reference_id,
            self.resume_version_id,
            self.organization_id,
            self.application_id,
            self.job_id,
            self.content_sha256,
            self.byte_count,
            self.media_type,
            self.source_observed_at,
        ]


class Source(StrictModel):
    source_id: Uuid
    source_version: Positive
    profile: Profile
    resume: Resume | None

    def ordered(self) -> list[Any]:
        p = self.profile
        return [
            self.source_id,
            self.source_version,
            p.display_name,
            p.headline,
            p.location,
            p.skills,
            p.linkedin,
            self.resume.ordered() if self.resume else None,
        ]


class PrivacyIdentifier(StrictModel):
    identifier_type: Literal["email", "vantahire_application_id", "vantahire_resume_id"]
    value: str = Field(min_length=1, max_length=2048)


class GrantProof(StrictModel):
    verified_email: str = Field(min_length=3, max_length=320)
    privacy_subject: list[PrivacyIdentifier] = Field(min_length=1, max_length=3)


class ConsentCommand(StrictModel):
    schema_version: Literal[1]
    subject_id: Uuid
    event_id: Uuid
    version: Positive
    action: Literal["grant", "withdraw"]
    purpose: Literal["platform_professional_matching"]
    purpose_version: Literal[1]
    copy_version: Literal[1]
    copy_sha256: Digest
    source: Source | None
    captured_at: str
    idempotency_key: Digest
    proof: GrantProof | None
    _time = field_validator("captured_at")(_timestamp)

    @model_validator(mode="after")
    def authority_shape(self) -> Self:
        if self.copy_sha256 != COPY_SHA256:
            raise ValueError("unknown consent copy")
        if (self.action == "grant") != (self.source is not None and self.proof is not None):
            raise ValueError("invalid consent action shape")
        if self.action == "withdraw" and (self.source is not None or self.proof is not None):
            raise ValueError("withdrawal carries no source or proof")
        if self.source and self.source.source_version != self.version:
            raise ValueError("source version mismatch")
        if self.proof:
            email = self.proof.verified_email
            if email != email.strip().lower() or not re.fullmatch(
                r"[^\s@]+@[^\s@]+\.[^\s@]+", email
            ):
                raise ValueError("invalid verified account")
            values = [(item.identifier_type, item.value) for item in self.proof.privacy_subject]
            expected = [("email", email)]
            if self.source and self.source.resume:
                expected.append(
                    ("vantahire_application_id", str(self.source.resume.application_id))
                )
            if values != expected:
                raise ValueError("privacy subject mismatch")
        return self


def canonical_bytes(payload: ConsentCommand) -> str:
    return compact(
        [
            1,
            payload.subject_id,
            payload.event_id,
            payload.version,
            payload.action,
            payload.purpose,
            payload.copy_version,
            payload.copy_sha256,
            payload.source.ordered() if payload.source else None,
            payload.captured_at,
        ]
    )


def command_identity(payload: ConsentCommand) -> tuple[str, str]:
    command_digest = digest(canonical_bytes(payload))
    key = digest(
        "\0".join(
            [
                "candidate-consent:v1",
                payload.subject_id,
                str(payload.version),
                payload.event_id,
                command_digest,
            ]
        )
    )
    return command_digest, key


def candidate_consent_intake_enabled() -> bool:
    return os.getenv("CANDIDATE_CONSENT_INTAKE_ENABLED", "false").strip().lower() == "true"


async def require_consent_writer(claims: JWTClaims | None = Depends(get_jwt_claims)) -> JWTClaims:
    if not auth.JWT_ENABLED or claims is None:
        raise HTTPException(401, "candidate_consent_service_auth_required")
    if (
        claims.issuer != "vantahire"
        or auth.JWT_AUDIENCE != "activekg"
        or claims.actor_id != "vantahire-backend"
        or claims.actor_type != "service"
        or claims.scopes != ["candidate-consent:write"]
        or not _TENANT.fullmatch(claims.tenant_id)
    ):
        raise HTTPException(403, "candidate_consent_service_auth_denied")
    return claims


def _connect() -> psycopg.Connection:
    dsn = os.getenv("ACTIVEKG_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError("candidate_consent_database_unavailable")
    return psycopg.connect(dsn, row_factory=dict_row, connect_timeout=3)


def _privacy_tokens(proof: GrantProof) -> list[dict[str, Any]]:
    _, keys = load_candidate_privacy_config(require_hmac=True).require_hmac()
    tokens = []
    for item in proof.privacy_subject:
        normalized = normalize_privacy_identifier(item.identifier_type, item.value)
        for version, key in sorted(keys.items()):
            tokens.append(
                {
                    "identifier_type": normalized.identifier_type,
                    "key_version": version,
                    "token": identity_token(key, normalized).hex(),
                }
            )
    return sorted(
        tokens, key=lambda item: (item["identifier_type"], item["key_version"], item["token"])
    )


def _require_global(
    cur: psycopg.Cursor, tokens: list[dict[str, Any]], global_id: str | None
) -> None:
    try:
        cur.execute(
            """SELECT action,state,decision FROM candidate_privacy_match(%s::jsonb,%s::uuid,NULL,NULL)
            ORDER BY CASE decision WHEN 'review' THEN 4 WHEN 'block_all' THEN 3 WHEN 'block_global' THEN 2 ELSE 1 END DESC,
            effective_at DESC,version DESC LIMIT 1""",
            (compact(tokens), global_id),
        )
        row = cur.fetchone()
        decision = (
            CandidatePrivacyDecision.ALLOW
            if row is None
            else decision_for(row["state"], row["action"])
        )
        if row and decision != CandidatePrivacyDecision(row["decision"]):
            raise RuntimeError("inconsistent privacy decision")
        if decision == CandidatePrivacyDecision.REVIEW:
            raise HTTPException(503, "candidate_consent_temporarily_unavailable")
        require_allowed(decision, global_use=True)
        if global_id:
            cur.execute(
                "SELECT candidate_privacy_global_decision(%s::uuid) AS decision", (global_id,)
            )
            global_decision = CandidatePrivacyDecision(cur.fetchone()["decision"])
            if global_decision == CandidatePrivacyDecision.REVIEW:
                raise HTTPException(503, "candidate_consent_temporarily_unavailable")
            require_allowed(global_decision, global_use=True)
    except CandidatePrivacyRestricted as exc:
        raise HTTPException(451, "candidate_privacy_restricted") from exc


def _check_resume(cur: psycopg.Cursor, tenant: str, resume: Resume) -> None:
    cur.execute(
        "SELECT set_config('app.current_tenant_id',%s,true)", (f"org_{resume.organization_id}",)
    )
    # Exactly one keyed application-data statement in organization context. Never return the private row.
    cur.execute(
        """SELECT EXISTS(SELECT 1 FROM organization_candidate_resume_evidence v
        JOIN organization_candidate_references r ON
          (r.tenant_id,r.reference_id,r.candidate_id)=(v.tenant_id,v.reference_id,v.candidate_id)
        JOIN organization_candidate_ingest_receipts i ON
          (i.tenant_id,i.reference_id,i.resume_version_id,i.candidate_id)=
          (v.tenant_id,v.reference_id,v.resume_version_id,v.candidate_id)
        WHERE v.resume_version_id=%s::uuid AND v.reference_id=%s::uuid AND v.tenant_id=%s
          AND r.application_id=%s AND r.job_id=%s AND r.origin_code='candidate_applied'
          AND v.content_sha256=%s AND v.byte_count=%s AND v.media_type=%s AND v.source_observed_at=%s::timestamptz
          AND i.resolution='created') AS matches""",
        (
            resume.resume_version_id,
            resume.reference_id,
            f"org_{resume.organization_id}",
            resume.application_id,
            resume.job_id,
            resume.content_sha256,
            resume.byte_count,
            resume.media_type,
            resume.source_observed_at,
        ),
    )
    matches = cur.fetchone()["matches"]
    cur.execute("SELECT set_config('app.current_tenant_id',%s,true)", (tenant,))
    if not matches:
        raise HTTPException(409, "candidate_consent_resume_unavailable")


def _resolve_identity(
    cur: psycopg.Cursor, payload: ConsentCommand, state: dict[str, Any] | None
) -> tuple[str | None, bool]:
    assert payload.proof and payload.source
    email_hash = digest(payload.proof.verified_email)
    linkedin = payload.source.profile.linkedin
    slug = urlsplit(linkedin).path.split("/in/", 1)[1] if linkedin else None
    cur.execute(
        """SELECT id,email_hash FROM global_candidates
        WHERE email_hash=%s OR (%s::text IS NOT NULL AND
          (linkedin_id=%s OR linkedin_url=%s OR linkedin_url=%s)) ORDER BY id FOR UPDATE""",
        (email_hash, linkedin, slug, linkedin, linkedin.replace("www.", "") if linkedin else None),
    )
    matches = cur.fetchall()
    bound = str(state["global_candidate_id"]) if state and state["global_candidate_id"] else None
    email_ids = [str(row["id"]) for row in matches if row["email_hash"] == email_hash]
    all_ids = {str(row["id"]) for row in matches}
    if bound:
        return bound, bool(all_ids - {bound})
    if len(all_ids) > 1 or (all_ids and not email_ids):
        return None, True
    return (email_ids[0] if email_ids else None), False


def _response(
    payload: ConsentCommand,
    command_digest: str,
    outcome: str,
    effective_version: int,
    effective_action: str | None,
) -> dict[str, Any]:
    return {
        "subject_id": payload.subject_id,
        "event_id": payload.event_id,
        "version": payload.version,
        "idempotency_key": payload.idempotency_key,
        "command_digest": command_digest,
        "outcome": outcome,
        "effective_version": effective_version,
        "effective_action": effective_action,
    }


def _store(payload: ConsentCommand, claims: JWTClaims) -> dict[str, Any]:
    if claims.tenant_id != f"candidate_{payload.subject_id}":
        raise HTTPException(403, "candidate_consent_tenant_mismatch")
    command_digest, expected_key = command_identity(payload)
    conn = None
    started = time.monotonic()
    try:
        conn = _connect()
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='3000ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='4000ms'")
            cur.execute("SELECT set_config('app.current_tenant_id',%s,true)", (claims.tenant_id,))
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                (f"candidate-consent:{payload.subject_id}",),
            )
            cur.execute(
                "SELECT * FROM candidate_consent_state WHERE subject_id=%s::uuid FOR UPDATE",
                (payload.subject_id,),
            )
            state = cur.fetchone()
            cur.execute(
                """SELECT * FROM candidate_consent_receipts WHERE event_id=%s::uuid OR idempotency_key=%s
                OR (subject_id=%s::uuid AND version=%s)""",
                (payload.event_id, payload.idempotency_key, payload.subject_id, payload.version),
            )
            receipts = cur.fetchall()
            if receipts:
                if (
                    len(receipts) != 1
                    or receipts[0]["command_digest"] != command_digest
                    or payload.idempotency_key != expected_key
                ):
                    raise HTTPException(409, "candidate_consent_replay_conflict")
                prior = receipts[0]
                outcome = (
                    "replayed" if prior["outcome"] in {"granted", "withdrawn"} else prior["outcome"]
                )
                if state and state["highest_version"] > payload.version:
                    outcome = "superseded"
                conn.rollback()
                return _response(
                    payload,
                    command_digest,
                    outcome,
                    prior["effective_version"],
                    prior["effective_action"],
                )
            if payload.idempotency_key != expected_key:
                raise HTTPException(422, "candidate_consent_idempotency_invalid")
            effective_version = int(state["effective_version"]) if state else 0
            effective_action = state["effective_action"] if state else None
            global_id = (
                str(state["global_candidate_id"])
                if state and state["global_candidate_id"]
                else None
            )
            source_id = None
            outcome = (
                "superseded"
                if state and payload.version < state["highest_version"]
                else payload.action
            )
            if outcome != "superseded" and payload.action == "grant":
                assert payload.proof and payload.source
                if payload.source.resume:
                    _check_resume(cur, claims.tenant_id, payload.source.resume)
                tokens = _privacy_tokens(payload.proof)
                # Share the frozen privacy writer's token lock ordering before canonical locks/admission.
                for token in tokens:
                    key = f"candidate-privacy-token:{token['identifier_type']}:{token['key_version']}:{token['token']}"
                    cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (key,))
                _require_global(cur, tokens, None)
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                    (f"candidate-consent-email:{digest(payload.proof.verified_email)}",),
                )
                global_id, conflict = _resolve_identity(cur, payload, state)
                if global_id:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
                        (f"candidate-privacy-global:{global_id}",),
                    )
                _require_global(cur, tokens, global_id)
                if conflict:
                    outcome = "identity_review_required"
                    global_id = (
                        str(state["global_candidate_id"])
                        if state and state["global_candidate_id"]
                        else None
                    )
                else:
                    if global_id is None:
                        cur.execute(
                            """INSERT INTO global_candidates(email_hash,embedding_status,embedding,public_profile)
                            VALUES(%s,'consent_pending',NULL,'{}'::jsonb) RETURNING id""",
                            (digest(payload.proof.verified_email),),
                        )
                        global_id = str(cur.fetchone()["id"])
                    outcome = "granted"
            elif outcome != "superseded":
                outcome = "withdrawn"
            if not state:
                cur.execute(
                    "INSERT INTO candidate_consent_state(subject_id,tenant_id) VALUES(%s::uuid,%s)",
                    (payload.subject_id, claims.tenant_id),
                )
            if outcome == "granted":
                assert payload.source
                source = payload.source
                source_id = source.source_id
                cur.execute(
                    """INSERT INTO candidate_consent_sources(source_id,subject_id,tenant_id,source_version,
                    global_candidate_id,profile,profile_sha256,resume,resume_sha256,purpose,purpose_version,
                    copy_version,copy_sha256,approved_at) VALUES(%s::uuid,%s::uuid,%s,%s,%s::uuid,%s::jsonb,%s,
                    %s::jsonb,%s,%s,1,1,%s,%s::timestamptz)""",
                    (
                        source_id,
                        payload.subject_id,
                        claims.tenant_id,
                        payload.version,
                        global_id,
                        compact(source.profile.model_dump()),
                        digest(compact(source.profile.model_dump())),
                        compact(source.resume.model_dump()) if source.resume else None,
                        digest(compact(source.resume.ordered())) if source.resume else None,
                        PURPOSE,
                        COPY_SHA256,
                        payload.captured_at,
                    ),
                )
            if outcome in {"granted", "withdrawn"}:
                effective_version, effective_action = payload.version, payload.action
                cur.execute(
                    """UPDATE candidate_consent_state SET highest_version=%s,last_action=%s,
                    global_candidate_id=%s::uuid,active_source_id=%s::uuid,effective_version=%s,effective_action=%s,
                    effective_at=clock_timestamp(),updated_at=clock_timestamp() WHERE subject_id=%s::uuid""",
                    (
                        payload.version,
                        payload.action,
                        global_id,
                        source_id,
                        effective_version,
                        effective_action,
                        payload.subject_id,
                    ),
                )
            elif outcome == "identity_review_required":
                cur.execute(
                    """UPDATE candidate_consent_state SET highest_version=%s,last_action=%s,updated_at=clock_timestamp()
                    WHERE subject_id=%s::uuid""",
                    (payload.version, payload.action, payload.subject_id),
                )
            cur.execute(
                """INSERT INTO candidate_consent_receipts(event_id,idempotency_key,command_digest,subject_id,
                tenant_id,version,action,outcome,source_id,global_candidate_id,effective_version,effective_action,
                verified_issuer,verified_actor_id) VALUES(%s::uuid,%s,%s,%s::uuid,%s,%s,%s,%s,%s::uuid,%s::uuid,%s,%s,
                'vantahire','vantahire-backend')""",
                (
                    payload.event_id,
                    payload.idempotency_key,
                    command_digest,
                    payload.subject_id,
                    claims.tenant_id,
                    payload.version,
                    payload.action,
                    outcome,
                    source_id,
                    global_id,
                    effective_version,
                    effective_action,
                ),
            )
        if time.monotonic() - started > 3:
            raise HTTPException(503, "candidate_consent_temporarily_unavailable")
        conn.commit()
        return _response(payload, command_digest, outcome, effective_version, effective_action)
    except HTTPException:
        if conn is not None:
            conn.rollback()
        raise
    except Exception as exc:
        if conn is not None:
            conn.rollback()
        # Never propagate driver messages, identities, profiles or transient proof to logs/responses.
        raise HTTPException(503, "candidate_consent_temporarily_unavailable") from exc
    finally:
        if conn is not None:
            conn.close()


@router.post("/candidate-consent/events", response_model=None)
async def ingest_candidate_consent(
    request: Request, claims: Annotated[JWTClaims, Depends(require_consent_writer)]
) -> dict[str, Any]:
    if not candidate_consent_intake_enabled():
        raise HTTPException(503, "candidate_consent_intake_disabled")
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 65536:
            raise HTTPException(413, "candidate_consent_request_too_large")
    try:
        payload = ConsentCommand.model_validate_json(bytes(body))
    except (ValidationError, ValueError, UnicodeError) as exc:
        raise HTTPException(422, "candidate_consent_request_invalid") from exc
    return await run_in_threadpool(_store, payload, claims)
