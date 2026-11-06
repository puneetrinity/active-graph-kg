"""Routine-only private source receiver. No provider calls or full-text storage."""

import hashlib

import psycopg
from fastapi import HTTPException
from psycopg.types.json import Jsonb

from activekg.candidate_profiles.profile_authority import check_owner_privacy
from activekg.candidate_profiles.repository import _connect
from activekg.candidate_profiles.source_contracts import (
    parse_resume_transfer,
    proposal,
    validate_source_receipt,
)
from activekg.candidate_profiles.spark_contracts import (
    digest,
    interpretation_proposal,
    old,
    parse_exclusion_api,
    parse_interpretation_transfer,
    uid,
    validate_interpretation_receipt,
)


def preview_interpretation_exclusions(subject, binding, raw):
    try:
        uid(subject)
        uid(binding)
        parsed = parse_exclusion_api(raw, "exclusion_preview_request")
        old.need(parsed["canonical"].encode() == raw)
        request = parsed["value"]
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError):
        raise HTTPException(422, "invalid_request") from None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='5000ms'")
            cur.execute(
                "SELECT candidate_owned_status(%s::uuid,%s::uuid) AS result", (subject, binding)
            )
            if not cur.fetchone():
                raise HTTPException(404, "not_found")
            check_owner_privacy(cur, subject)
            cur.execute(
                "SELECT candidate_exclusion_preview(%s,%s,%s,%s,%s) AS result",
                (
                    subject,
                    binding,
                    request["idempotencyKey"],
                    request["expectedPreferenceDraftRevision"],
                    Jsonb(request["restrictions"]),
                ),
            )
            row = cur.fetchone()
            old.need(row is not None)
            result = parse_exclusion_api(
                old.canonical(row["result"]).encode(), "exclusion_preview"
            )["value"]
            old.need(
                result["expectedPreferenceDraftRevision"]
                == request["expectedPreferenceDraftRevision"]
            )
            old.need(
                [{k: v for k, v in rule.items() if k != "id"} for rule in result["restrictions"]]
                == request["restrictions"]
            )
            check_owner_privacy(cur, subject)
            return result
    except HTTPException:
        raise
    except psycopg.Error as exc:
        status = {"P0002": 404, "23505": 409, "40001": 409, "22023": 422}.get(exc.sqlstate, 503)
        if exc.sqlstate == "42501" and exc.diag.message_primary == "candidate_privacy_restricted":
            status = 451
        raise HTTPException(
            status, "authority_unavailable" if status == 503 else "exclusion_preview_refused"
        ) from None
    except Exception:
        raise HTTPException(503, "authority_unavailable") from None


def interpretation_storage(
    subject, binding, *, raw=None, command_id=None, command_hash=None, proposal_id=None
):
    """P6 storage only: no model call and no mutation of the canonical profile."""
    try:
        uid(subject)
        uid(binding)
        parsed = None
        if raw is None:
            uid(proposal_id)
        else:
            uid(command_id)
            digest(command_hash)
            parsed = parse_interpretation_transfer(raw)
            old.need(parsed["hash"] == command_hash)
            value = parsed["value"]
            if (value["subjectId"], value["bindingId"], value["commandId"]) != (
                subject,
                binding,
                command_id,
            ):
                raise HTTPException(403, "service_auth_denied")
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError):
        raise HTTPException(422, "invalid_request") from None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='5000ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='6000ms'")
            cur.execute(
                "SELECT candidate_owned_status(%s::uuid,%s::uuid) AS result", (subject, binding)
            )
            if not cur.fetchone():
                raise HTTPException(404, "not_found")
            check_owner_privacy(cur, subject)
            if parsed is None:
                cur.execute(
                    "SELECT candidate_profile_interpretation_read(%s,%s,%s) AS result",
                    (subject, binding, proposal_id),
                )
            else:
                cur.execute(
                    "SELECT candidate_profile_interpretation_receive(%s,%s,%s,%s,%s) AS result",
                    (subject, binding, command_id, command_hash, parsed["canonical"]),
                )
            row = cur.fetchone()
            old.need(row is not None and type(row["result"]) is dict)
            if parsed is None:
                result = interpretation_proposal(row["result"])
                old.need(result["proposalId"] == proposal_id)
            else:
                result = validate_interpretation_receipt(parsed["value"], row["result"])
            check_owner_privacy(cur, subject)
            return result
    except HTTPException:
        raise
    except psycopg.Error as exc:
        status = {"P0002": 404, "23505": 409, "22023": 422, "22P05": 422}.get(exc.sqlstate, 503)
        if exc.sqlstate == "42501" and exc.diag.message_primary == "candidate_privacy_restricted":
            status = 451
        raise HTTPException(
            status, "authority_unavailable" if status == 503 else "interpretation_refused"
        ) from None
    except Exception:
        raise HTTPException(503, "authority_unavailable") from None


def read_source(subject, binding, source_id, proposal_id):
    try:
        for value in (subject, binding, source_id, proposal_id):
            uid(value)
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout='5000ms'")
            cur.execute(
                "SELECT candidate_owned_status(%s::uuid,%s::uuid) AS result", (subject, binding)
            )
            if not cur.fetchone():
                raise HTTPException(404, "not_found")
            check_owner_privacy(cur, subject)
            cur.execute(
                "SELECT candidate_profile_source_read(%s,%s,%s,%s) AS result",
                (subject, binding, source_id, proposal_id),
            )
            row = cur.fetchone()
            old.need(row is not None)
            value = proposal(row["result"])
            old.need(value["sourceId"] == source_id and value["proposalId"] == proposal_id)
            body = {k: v for k, v in value.items() if k != "proposalHash"}
            old.need(
                hashlib.sha256(old.canonical(body).encode()).hexdigest() == value["proposalHash"]
            )
            check_owner_privacy(cur, subject)
            return value
    except HTTPException:
        raise
    except psycopg.Error as exc:
        status = 404 if exc.sqlstate == "P0002" else 503
        if exc.sqlstate == "42501" and exc.diag.message_primary == "candidate_privacy_restricted":
            status = 451
        raise HTTPException(
            status, "not_found" if status == 404 else "candidate_source_unavailable"
        ) from None
    except Exception:
        raise HTTPException(503, "candidate_source_unavailable") from None


def receive_source(subject, binding, command_id, command_hash, raw, kind="resume"):
    try:
        uid(subject)
        uid(binding)
        uid(command_id)
        digest(command_hash)
        parsed = parse_resume_transfer(raw, kind)
        value = parsed["value"]
        old.need(parsed["canonical"].encode() == raw and parsed["hash"] == command_hash)
        if (value["subjectId"], value["bindingId"], value["commandId"]) != (
            subject,
            binding,
            command_id,
        ):
            raise HTTPException(403, "service_auth_denied")
    except (ValueError, TypeError, KeyError, UnicodeError, OverflowError):
        raise HTTPException(422, "invalid_request") from None
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='5000ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='6000ms'")
            cur.execute(
                "SELECT candidate_owned_status(%s::uuid,%s::uuid) AS result", (subject, binding)
            )
            if not cur.fetchone():
                raise HTTPException(404, "not_found")
            check_owner_privacy(cur, subject)
            cur.execute(
                "SELECT candidate_profile_source_receive(%s::uuid,%s::uuid,%s::uuid,%s,%s) AS result",
                (subject, binding, command_id, command_hash, parsed["canonical"]),
            )
            row = cur.fetchone()
            old.need(row is not None and type(row["result"]) is dict)
            receipt = validate_source_receipt(value, row["result"], kind)
            check_owner_privacy(cur, subject)
            return receipt
    except HTTPException:
        raise
    except psycopg.Error as exc:
        # Missing EXECUTE is a deployment failure, not a candidate privacy decision.
        status = {"P0002": 404, "23505": 409, "22023": 422, "22P05": 422}.get(exc.sqlstate, 503)
        if exc.sqlstate == "42501" and exc.diag.message_primary == "candidate_privacy_restricted":
            status = 451
        raise HTTPException(
            status, "authority_unavailable" if status == 503 else "candidate_source_refused"
        ) from None
    except Exception:
        raise HTTPException(503, "authority_unavailable") from None
