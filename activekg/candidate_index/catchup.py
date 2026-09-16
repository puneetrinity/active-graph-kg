"""Explicit, bounded catchup of still-effective candidate-approved sources.

Not a startup sweep, endpoint or provider adopter. Flow's immutable command and
live verified account supply the original proof; an email digest alone cannot.
The approved local operator adapter supplies fresh running posture, Flow's shipped
global admission and the bounded original-object check. No unchecked boolean file
or reconstructed profile is an implementation of those adapters. No model call
occurs here; normal workers reserve their own hard-bounded attempts afterward.
"""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import os
import stat
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import psycopg
from fastapi import HTTPException
from psycopg.rows import dict_row
from pydantic import BaseModel, ConfigDict, Field

from activekg.api import candidate_consent as consent
from activekg.candidate_index.admission import IndexConfiguration, configuration
from activekg.candidate_index.contracts import canonical_json, sha256
from activekg.candidate_index.repository import IndexRepository
from activekg.privacy.config import load_candidate_privacy_config

Uuid = Annotated[str, Field(pattern=consent._UUID)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ZERO = "00000000-0000-0000-0000-000000000000"
ATTEMPTS_PER_SOURCE = 5  # extraction2 + embedding3 SQL ceilings, not customer credits
Reason = Literal[
    "withdrawn_or_superseded",
    "source_orphaned",
    "binding_ambiguous",
    "source_mismatch",
    "account_changed",
    "intake_unacknowledged",
    "already_managed",
    "object_unavailable",
    "privacy_restricted",
    "privacy_unavailable",
    "busy",
    "captured",
]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Posture(Strict):
    flow_target: str = Field(min_length=1, max_length=200)
    memory_target: Uuid
    flow_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    memory_tree: str = Field(pattern=r"^[0-9a-f]{40}$")
    policy_sha256: Digest
    running: Literal[True]


class Entry(Strict):
    source_id: Uuid
    subject_id: Uuid
    fingerprint: Digest
    authority_sha256: Digest


class PlanBody(Strict):
    kind: Literal["candidate_consent_catchup"] = "candidate_consent_catchup"
    version: Literal[1] = 1
    nonce: Uuid
    issued_at: int = Field(ge=0)
    expires_at: int = Field(ge=0)
    posture: Posture
    cursor: Uuid
    next_cursor: Uuid
    max_rows: int = Field(ge=1, le=100)
    max_provider_attempts: int = Field(ge=0, le=500)
    scanned: int = Field(ge=0, le=100)
    entries: list[Entry] = Field(max_length=100)
    skipped: dict[Reason, Annotated[int, Field(ge=0, le=100)]]


class Plan(PlanBody):
    seal: Digest


class Record(Strict):
    seq: int = Field(ge=0, le=99)
    source_id: Uuid
    outcome: Reason
    previous: Digest
    seal: Digest


class CatchupRefused(RuntimeError):
    def __init__(self) -> None:
        super().__init__("candidate_index_catchup_refused")


class Skip(Exception):
    def __init__(self, reason: Reason) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(repr=False)
class Operations:
    flow_read_dsn: str
    memory_runtime_dsn: str
    observe_running: Callable[[], Posture]
    # Must call the shipped Flow candidate-user privacy gate, globalUse=true,
    # newGlobalOperation=true, not infer permission from the consent checkbox.
    admit_flow_user: Callable[[int], None]
    # Must hash/size/type-check the exact immutable original using the bounded
    # Flow downloader. No GCS credentials or arbitrary URL fetching on Memory.
    verify_original: Callable[[dict[str, Any]], bool]
    config: IndexConfiguration
    now: Callable[[], float] = time.time


FLOW_SQL = """
SELECT f.*,s.user_id,s.version,s.desired_action,s.acknowledged_version,s.acknowledged_action,
 s.current_source_id,s.effective_source_id,s.delivery_status,
 u.id AS live_user,u.username,u.role,u.email_verified,u.auth_version,
 e.event_id,e.action,e.user_id AS event_user_id,e.verified_email_sha256,e.account_auth_version,
 e.purpose,e.purpose_version,e.schema_version,e.copy_version,e.copy_sha256,e.command_sha256,e.captured_at,
 o.state AS outbox_state,o.idempotency_key,o.command_sha256 AS outbox_digest,
 v.gcs_locator,v.source_resume_id,v.extracted_text_sha256,
 (a.id IS NOT NULL AND j.id IS NOT NULL AND a.user_id=s.user_id AND r.origin_code='candidate_applied'
   AND b.state='acknowledged' AND b.acknowledged_at IS NOT NULL AND b.memory_candidate_id IS NOT NULL) AS resume_live,
 jsonb_build_object('reference_id',v.reference_id,'resume_version_id',v.resume_version_id,
   'organization_id',v.organization_id,'application_id',v.application_id,'job_id',v.job_id,
   'content_sha256',v.content_sha256,'byte_count',v.byte_count,'media_type',v.media_type,
   'source_observed_at',to_char(v.source_observed_at AT TIME ZONE 'UTC','YYYY-MM-DD"T"HH24:MI:SS.MS"Z"')) AS resume_pin
FROM public.candidate_consent_sources f
JOIN public.candidate_consent_subjects s ON s.subject_id=f.subject_id
LEFT JOIN public.users u ON u.id=s.user_id
LEFT JOIN public.candidate_consent_events e ON e.source_id=f.source_id AND e.subject_id=f.subject_id AND e.version=f.source_version
LEFT JOIN public.candidate_consent_outbox o ON o.event_id=e.event_id AND o.subject_id=e.subject_id AND o.version=e.version
LEFT JOIN public.application_resume_versions v ON v.resume_version_id=f.resume_version_id
LEFT JOIN public.applications a ON (a.id,a.organization_id,a.job_id)=(v.application_id,v.organization_id,v.job_id)
LEFT JOIN public.jobs j ON (j.id,j.organization_id)=(v.job_id,v.organization_id)
LEFT JOIN public.organization_candidate_references r ON
  (r.reference_id,r.organization_id,r.application_id,r.job_id)=(v.reference_id,v.organization_id,v.application_id,v.job_id)
LEFT JOIN public.organization_candidate_memory_outbox b ON
  (b.resume_version_id,b.reference_id,b.organization_id,b.application_id,b.job_id)=
  (v.resume_version_id,v.reference_id,v.organization_id,v.application_id,v.job_id)
"""


def _mac(key: bytes, value: Any) -> str:
    if type(key) is not bytes or len(key) != 32:
        raise CatchupRefused()
    return hmac.new(key, canonical_json(value).encode(), hashlib.sha256).hexdigest()


def verify_plan(value: Any, key: bytes, now: float) -> Plan:
    try:
        plan = Plan.model_validate(value.model_dump() if isinstance(value, Plan) else value)
        body = plan.model_dump(exclude={"seal"})
        if (
            not hmac.compare_digest(plan.seal, _mac(key, body))
            or not plan.issued_at <= now < plan.expires_at <= plan.issued_at + 7200
            or plan.scanned > plan.max_rows
            or plan.scanned != len(plan.entries) + sum(plan.skipped.values())
            or "captured" in plan.skipped
            or len(plan.entries) * ATTEMPTS_PER_SOURCE > plan.max_provider_attempts
            or plan.next_cursor < plan.cursor
        ):
            raise CatchupRefused()
        previous = plan.cursor
        for entry in plan.entries:
            if not previous < entry.source_id <= plan.next_cursor:
                raise CatchupRefused()
            previous = entry.source_id
        return plan
    except Exception:
        raise CatchupRefused() from None


def _posture(ops: Operations) -> Posture:
    value = ops.observe_running()
    if not isinstance(value, Posture) or value.policy_sha256 != ops.config.policy.digest:
        raise CatchupRefused()
    # Config edits cannot turn an old seal into approval to run a different policy.
    if configuration().policy.digest != value.policy_sha256:
        raise CatchupRefused()
    return value


@contextmanager
def _transaction(
    dsn: str, expected: str, *, memory: bool, readonly: bool, scope: str | None = None
) -> Iterator[psycopg.Cursor]:
    with psycopg.connect(dsn, connect_timeout=3, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            if readonly:
                cur.execute("SET TRANSACTION READ ONLY")
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='3s'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='4000ms'")
            table = (
                "activekg_schema_control.target_identity" if memory else "schema_control.identity"
            )
            singleton = "1" if memory else "true"
            cur.execute(
                f"SELECT target_id::text AS target FROM {table} WHERE singleton={singleton}"
            )
            if cur.fetchall() != [{"target": expected}]:
                raise CatchupRefused()
            if readonly:
                cur.execute("SHOW transaction_read_only")
                if cur.fetchone() != {"transaction_read_only": "on"}:
                    raise CatchupRefused()
            if scope:
                cur.execute("SELECT set_config('app.current_tenant_id',%s,true)", (scope,))
            yield cur


def _flow(ops: Operations, posture: Posture, clause: str, args: tuple) -> list[dict]:
    with _transaction(ops.flow_read_dsn, posture.flow_target, memory=False, readonly=True) as cur:
        cur.execute(FLOW_SQL + clause, args)
        return cur.fetchall()


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _command(row: dict) -> consent.ConsentCommand:
    if not row["live_user"] or not row["event_id"]:
        raise Skip("source_orphaned")
    if (
        row["desired_action"] != "grant"
        or row["acknowledged_action"] != "grant"
        or row["delivery_status"] != "delivered"
        or not row["version"] == row["source_version"] == row["acknowledged_version"]
        or not row["source_id"] == row["current_source_id"] == row["effective_source_id"]
    ):
        raise Skip("withdrawn_or_superseded")
    if row["outbox_state"] != "acknowledged":
        raise Skip("intake_unacknowledged")
    email = row["username"].strip().lower()
    if (
        row["role"] != "candidate"
        or row["email_verified"] is not True
        or not row["live_user"] == row["user_id"] == row["event_user_id"]
        or row["auth_version"] != row["account_auth_version"]
        or sha256(email) != row["verified_email_sha256"]
    ):
        raise Skip("account_changed")
    privacy = [{"identifier_type": "email", "value": email}]
    if row["resume"] is not None:
        if row["resume_live"] is not True or row["resume_pin"] != row["resume"]:
            raise Skip("source_orphaned")
        privacy.append(
            {
                "identifier_type": "vantahire_application_id",
                "value": str(row["resume"]["application_id"]),
            }
        )
    try:
        payload = consent.ConsentCommand.model_validate(
            {
                "schema_version": row["schema_version"],
                "subject_id": str(row["subject_id"]),
                "event_id": str(row["event_id"]),
                "version": row["source_version"],
                "action": row["action"],
                "purpose": row["purpose"],
                "purpose_version": row["purpose_version"],
                "copy_version": row["copy_version"],
                "copy_sha256": row["copy_sha256"],
                "captured_at": _iso(row["captured_at"]),
                "source": {
                    "source_id": str(row["source_id"]),
                    "source_version": row["source_version"],
                    "profile": row["profile"],
                    "resume": row["resume"],
                },
                "idempotency_key": row["idempotency_key"],
                "proof": {"verified_email": email, "privacy_subject": privacy},
            }
        )
        source = payload.source
        assert source is not None
        if (
            consent.command_identity(payload) != (row["command_sha256"], row["idempotency_key"])
            or row["outbox_digest"] != row["command_sha256"]
            or consent.digest(consent.compact(source.profile.model_dump())) != row["profile_sha256"]
            or (consent.digest(consent.compact(source.resume.ordered())) if source.resume else None)
            != row["resume_sha256"]
            or row["approved_at"] != row["captured_at"]
        ):
            raise ValueError()
        return payload
    except Exception:
        raise Skip("source_mismatch") from None


def _flow_admit(row: dict, ops: Operations) -> consent.ConsentCommand:
    payload = _command(row)
    try:
        ops.admit_flow_user(row["live_user"])
    except Exception as exc:
        raise Skip(
            "privacy_restricted"
            if getattr(exc, "code", None) == "candidate_privacy_restricted"
            else "privacy_unavailable"
        ) from None
    if payload.source.resume:
        original = {**row["resume_pin"], "gcs_locator": row["gcs_locator"]}
        try:
            if ops.verify_original(original) is not True:
                raise ValueError()
        except Exception:
            raise Skip("object_unavailable") from None
    return payload


def _memory(
    cur: psycopg.Cursor,
    payload: consent.ConsentCommand,
    row: dict,
    *,
    capture: bool,
    ops: Operations,
) -> Entry:
    source = payload.source
    assert source is not None and payload.proof is not None
    scope = f"candidate_{payload.subject_id}"
    # Exact shipped consent -> token -> email -> canonical lock ordering. No
    # name/LinkedIn binding, shell creation or source/receipt replay is performed.
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
        (f"candidate-consent:{payload.subject_id}",),
    )
    cur.execute(
        "SELECT * FROM candidate_consent_state WHERE subject_id=%s::uuid", (payload.subject_id,)
    )
    state = cur.fetchone()
    if (
        not state
        or state["effective_action"] != "grant"
        or state["last_action"] != "grant"
        or (
            state["highest_version"] != payload.version
            or state["effective_version"] != payload.version
            or str(state["active_source_id"]) != source.source_id
        )
    ):
        raise Skip("withdrawn_or_superseded")
    cur.execute(
        "SELECT * FROM candidate_consent_sources WHERE source_id=%s::uuid", (source.source_id,)
    )
    sources = cur.fetchall()
    cur.execute(
        "SELECT * FROM candidate_consent_receipts WHERE event_id=%s::uuid", (payload.event_id,)
    )
    receipts = cur.fetchall()
    if len(sources) != 1 or len(receipts) != 1:
        raise Skip("binding_ambiguous")
    saved, receipt = sources[0], receipts[0]
    global_id = str(state["global_candidate_id"])
    if (
        str(saved["subject_id"]) != payload.subject_id
        or saved["tenant_id"] != scope
        or saved["source_version"] != payload.version
        or str(saved["global_candidate_id"]) != global_id
        or saved["profile"] != source.profile.model_dump()
        or saved["profile_sha256"] != row["profile_sha256"]
        or saved["resume"] != row["resume"]
        or saved["resume_sha256"] != row["resume_sha256"]
        or saved["approved_at"] != row["approved_at"]
        or receipt["command_digest"] != row["command_sha256"]
        or receipt["idempotency_key"] != payload.idempotency_key
        or receipt["outcome"] != "granted"
        or receipt["action"] != "grant"
        or receipt["effective_action"] != "grant"
        or receipt["effective_version"] != payload.version
        or str(receipt["source_id"]) != source.source_id
        or str(receipt["global_candidate_id"]) != global_id
    ):
        raise Skip("source_mismatch")
    tokens = consent._privacy_tokens(payload.proof)
    for token in tokens:
        name = f"candidate-privacy-token:{token['identifier_type']}:{token['key_version']}:{token['token']}"
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (name,))
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
        (f"candidate-consent-email:{sha256(payload.proof.verified_email)}",),
    )
    cur.execute(
        "SELECT id FROM global_candidates WHERE email_hash=%s",
        (sha256(payload.proof.verified_email),),
    )
    if [str(r["id"]) for r in cur.fetchall()] != [global_id]:
        raise Skip("binding_ambiguous")
    cur.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s,0))",
        (f"candidate-privacy-global:{global_id}",),
    )
    consent._require_global(cur, tokens, global_id)
    cur.execute(
        "SELECT source_id FROM candidate_index_sources WHERE source_id=%s::uuid",
        (source.source_id,),
    )
    if cur.fetchone() is not None:
        raise Skip("already_managed")
    entry = Entry(
        source_id=source.source_id,
        subject_id=payload.subject_id,
        fingerprint=sha256(
            canonical_json(
                [
                    row["command_sha256"],
                    row["profile_sha256"],
                    row["resume_sha256"],
                    row["gcs_locator"],
                    str(row["source_resume_id"]) if row["source_resume_id"] else None,
                    row["extracted_text_sha256"],
                ]
            )
        ),
        authority_sha256=sha256(
            canonical_json(
                [
                    row["user_id"],
                    row["auth_version"],
                    row["verified_email_sha256"],
                    global_id,
                    payload.version,
                ]
            )
        ),
    )
    if capture:
        _, keys = load_candidate_privacy_config(require_hmac=True).require_hmac()
        IndexRepository.capture_on_cursor(
            cur,
            kind="candidate_consent",
            upstream_id=source.source_id,
            command_digest=row["command_sha256"],
            content_kind="approved_profile",
            content=None,
            tokens=tokens,
            key_versions=sorted(keys),
            policy=ops.config.policy,
            priority="maintenance",
            tunables=ops.config.tunables,
        )
    return entry


def _reason(exc: Exception) -> Reason:
    if isinstance(exc, Skip):
        return exc.reason
    if isinstance(exc, HTTPException) and exc.status_code in {451, 503}:
        return "privacy_restricted" if exc.status_code == 451 else "privacy_unavailable"
    if isinstance(exc, psycopg.Error) and exc.sqlstate in {"55P03", "57014"}:
        return "busy"
    raise CatchupRefused() from None


def census(
    ops: Operations,
    *,
    key: bytes,
    cursor: str = ZERO,
    max_rows: int = 100,
    max_provider_attempts: int,
    lifetime_seconds: int = 7200,
) -> Plan:
    try:
        started, posture = int(ops.now()), _posture(ops)
        # Validate bounds/key before any database, storage or admission IO.
        body = PlanBody(
            nonce=str(uuid4()),
            issued_at=started,
            expires_at=started + lifetime_seconds,
            posture=posture,
            cursor=cursor,
            next_cursor=cursor,
            max_rows=max_rows,
            max_provider_attempts=max_provider_attempts,
            scanned=0,
            entries=[],
            skipped={},
        )
        verify_plan({**body.model_dump(), "seal": _mac(key, body.model_dump())}, key, ops.now())
        rows = _flow(
            ops,
            posture,
            " WHERE f.source_id>%s::uuid ORDER BY f.source_id LIMIT %s",
            (cursor, max_rows),
        )
        entries, skipped = [], Counter()
        for row in rows:
            try:
                payload = _flow_admit(row, ops)
                with _transaction(
                    ops.memory_runtime_dsn,
                    posture.memory_target,
                    memory=True,
                    readonly=True,
                    scope=f"candidate_{payload.subject_id}",
                ) as cur:
                    entries.append(_memory(cur, payload, row, capture=False, ops=ops))
            except Exception as exc:
                skipped[_reason(exc)] += 1
        body = body.model_copy(
            update={
                "entries": entries,
                "skipped": dict(skipped),
                "scanned": len(rows),
                "next_cursor": str(rows[-1]["source_id"]) if rows else cursor,
            }
        )
        return verify_plan(
            {**body.model_dump(), "seal": _mac(key, body.model_dump())}, key, ops.now()
        )
    except Exception:
        raise CatchupRefused() from None


@contextmanager
def _private_file(path: Path, *, exclusive: bool = False):
    if not path.is_absolute() or path.parent.resolve() != path.parent:
        raise CatchupRefused()
    parent = path.parent.lstat()
    if (
        not stat.S_ISDIR(parent.st_mode)
        or stat.S_IMODE(parent.st_mode) != 0o700
        or parent.st_uid != os.getuid()
    ):
        raise CatchupRefused()
    fd = os.open(
        path,
        os.O_CREAT | os.O_RDWR | os.O_APPEND | os.O_NOFOLLOW | (os.O_EXCL if exclusive else 0),
        0o600,
    )
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.getuid()
            or info.st_nlink != 1
            or info.st_size > 128_000
        ):
            raise CatchupRefused()
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield fd
    finally:
        os.close(fd)


def _sync_parent(path: Path) -> None:
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _append(fd: int, value: dict) -> None:
    data = (canonical_json(value) + "\n").encode()
    while data:
        written = os.write(fd, data)
        if written <= 0:
            raise CatchupRefused()
        data = data[written:]
    os.fsync(fd)


def save_plan(path: Path, plan: Plan, key: bytes) -> None:
    verify_plan(plan, key, time.time())
    with _private_file(path, exclusive=True) as fd:
        _append(fd, plan.model_dump())
        _sync_parent(path)


def execute(
    ops: Operations, *, plan: Any, key: bytes, journal: Path, operator_approved: bool
) -> dict[str, int]:
    try:
        if operator_approved is not True:
            raise CatchupRefused()
        sealed = verify_plan(plan, key, ops.now())
        counts, seen, chain = Counter(), set(), sealed.seal
        with _private_file(journal) as fd:
            raw = os.read(fd, 128_001)
            end = raw.rfind(b"\n") + 1
            for index, line in enumerate(raw[:end].splitlines()):
                record = Record.model_validate_json(line)
                if (
                    record.seq != index
                    or record.previous != chain
                    or record.source_id in seen
                    or not any(e.source_id == record.source_id for e in sealed.entries)
                    or not hmac.compare_digest(
                        record.seal, _mac(key, record.model_dump(exclude={"seal"}))
                    )
                ):
                    raise CatchupRefused()
                seen.add(record.source_id)
                counts[record.outcome] += 1
                chain = record.seal
            # Validate the authenticated prefix before discarding only a torn tail.
            if end != len(raw):
                os.ftruncate(fd, end)
            os.fsync(fd)
            _sync_parent(journal)
            for entry in sealed.entries:
                verify_plan(sealed, key, ops.now())
                if _posture(ops) != sealed.posture:
                    raise CatchupRefused()
                if entry.source_id in seen:
                    continue
                outcome: Reason = "captured"
                try:
                    rows = _flow(
                        ops, sealed.posture, " WHERE f.source_id=%s::uuid", (entry.source_id,)
                    )
                    if len(rows) != 1:
                        raise Skip("binding_ambiguous" if rows else "source_orphaned")
                    row = rows[0]
                    payload = _flow_admit(row, ops)
                    # Storage/admission may wait; re-read the immutable command
                    # and live account afterward without holding a write txn.
                    fresh = _flow(
                        ops, sealed.posture, " WHERE f.source_id=%s::uuid", (entry.source_id,)
                    )
                    if len(fresh) != 1 or _command(fresh[0]) != payload:
                        raise Skip("source_mismatch")
                    try:
                        ops.admit_flow_user(fresh[0]["live_user"])
                    except Exception as exc:
                        raise Skip(
                            "privacy_restricted"
                            if getattr(exc, "code", None) == "candidate_privacy_restricted"
                            else "privacy_unavailable"
                        ) from None
                    verify_plan(sealed, key, ops.now())
                    with _transaction(
                        ops.memory_runtime_dsn,
                        sealed.posture.memory_target,
                        memory=True,
                        readonly=False,
                        scope=f"candidate_{entry.subject_id}",
                    ) as cur:
                        # Compare before capture. A later refusal rolls back the
                        # entire transaction; no historical receipt is created.
                        if _memory(cur, payload, fresh[0], capture=False, ops=ops) != entry:
                            raise Skip("source_mismatch")
                        _memory(cur, payload, fresh[0], capture=True, ops=ops)
                except Exception as exc:
                    outcome = _reason(exc)
                body = {
                    "seq": len(seen),
                    "source_id": entry.source_id,
                    "outcome": outcome,
                    "previous": chain,
                }
                record = {**body, "seal": _mac(key, body)}
                _append(fd, record)
                seen.add(entry.source_id)
                counts[outcome] += 1
                chain = record["seal"]
        return dict(counts)
    except Exception:
        raise CatchupRefused() from None
