"""Routine-only Spark storage adapter for separately gated product routes.

Authentication belongs to the route; this boundary still validates ownership,
privacy, canonical command identity and the exact database receipt itself.
"""

import psycopg
from fastapi import HTTPException

from activekg.candidate_profiles.profile_authority import check_owner_privacy
from activekg.candidate_profiles.profile_repository import checked_receipt
from activekg.candidate_profiles.repository import _connect
from activekg.candidate_profiles.spark_contracts import (
    digest as checked_hash,
)
from activekg.candidate_profiles.spark_contracts import (
    old,
    parse_command,
    parse_expiry,
    parse_response,
    uid,
)

_APPLY = {
    "accept_interpretation": "candidate_profile_interpretation_apply",
    "save_preference_section": "candidate_preference_draft_apply",
    "promote_preferences": "candidate_preference_draft_apply",
    "accept_exclusions": "candidate_exclusions_apply",
    "remove_exclusion": "candidate_exclusions_apply",
    "confirm": "candidate_profile_confirm_v3",
    "accept_source": "candidate_profile_source_apply",
    "reject_source": "candidate_profile_source_apply",
}


def execute_spark(subject, binding, *, raw=None, command_id=None, digest=None, expiry=False):
    try:
        uid(subject)
        uid(binding)
        parsed = None
        if raw is not None:
            uid(command_id)
            checked_hash(digest)
            parsed = parse_expiry(raw) if expiry else parse_command(raw)
            old.need(parsed["canonical"].encode() == raw and parsed["hash"] == digest)
            if expiry:
                old.need(
                    parsed["value"]["commandId"] == command_id
                    and parsed["value"]["bindingId"] == binding
                )
            routine = (
                (
                    "candidate_profile_expire_v3"
                    if parsed["value"]["schemaVersion"] == 3
                    else "candidate_profile_expire_command"
                )
                if expiry
                else _APPLY.get(parsed["value"]["action"])
            )
            if routine is None:
                # Unsupported actions cannot fall through into older v2 writes.
                raise HTTPException(409, "feature_unavailable")
    except (ValueError, TypeError, KeyError, OverflowError):
        raise HTTPException(422, "invalid_request") from None

    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='3000ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='4000ms'")
            cur.execute(
                "SELECT candidate_owned_status(%s::uuid,%s::uuid) AS result", (subject, binding)
            )
            if not cur.fetchone():
                raise HTTPException(404, "not_found")
            check_owner_privacy(cur, subject)
            if expiry and parsed is not None:
                cur.execute(
                    f"SELECT {routine}(%s::uuid,%s::uuid,%s::uuid,%s::uuid,%s) AS result",
                    (
                        subject,
                        binding,
                        command_id,
                        parsed["value"]["idempotencyKey"],
                        parsed["value"]["commandHash"],
                    ),
                )
            elif parsed is None:
                cur.execute(
                    "SELECT candidate_profile_v3_storage_read(%s::uuid,%s::uuid) AS result",
                    (subject, binding),
                )
            else:
                # Identifier is chosen exclusively from the fixed internal map.
                cur.execute(
                    f"SELECT {routine}(%s::uuid,%s::uuid,%s::uuid,%s,%s) AS result",
                    (subject, binding, command_id, digest, parsed["canonical"]),
                )
            row = cur.fetchone()
            old.need(row is not None and isinstance(row["result"], dict))
            value = (
                checked_receipt(
                    row["result"], subject, binding, command_id, parsed["value"]["commandHash"]
                )
                if expiry and parsed["value"]["schemaVersion"] == 2
                else parse_response(
                    old.canonical(row["result"]).encode(),
                    "storage_read" if parsed is None else "receipt",
                )["value"]
            )
            if parsed is not None:
                old.need(
                    (
                        value["subjectId"],
                        value["bindingId"],
                        value["commandId"],
                        value["commandHash"],
                    )
                    == (
                        subject,
                        binding,
                        command_id,
                        parsed["value"]["commandHash"] if expiry else digest,
                    ),
                    "identity_mismatch",
                )
            check_owner_privacy(cur, subject)
            return value
    except HTTPException:
        raise
    except psycopg.Error as exc:
        status = {"P0002": 404, "23505": 409, "42501": 451, "22023": 422}.get(exc.sqlstate, 503)
        raise HTTPException(
            status, "authority_unavailable" if status == 503 else "candidate_command_refused"
        ) from None
    except Exception:
        raise HTTPException(503, "authority_unavailable") from None
