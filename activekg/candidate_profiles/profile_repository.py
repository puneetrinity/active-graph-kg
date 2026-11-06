"""Routine-only storage for the manual, owner-private S2 checkpoint."""

import hashlib

import psycopg
from fastapi import HTTPException
from psycopg.types.json import Jsonb

from activekg.candidate_profiles.profile_authority import check_owner_privacy
from activekg.candidate_profiles.profile_contracts import (
    UUID,
    canonical,
    need,
    obj,
    parse_profile_command,
    preferences,
    profile,
    revision,
)
from activekg.candidate_profiles.repository import CATALOG_SQL as FOUNDATION_CATALOG_SQL
from activekg.candidate_profiles.repository import _connect

PROFILE_TABLES = (
    "candidate_profile_heads",
    "candidate_profile_drafts",
    "candidate_profile_versions",
    "candidate_preference_versions",
    "candidate_profile_receipts",
    "candidate_profile_erasure_audit",
    "candidate_preference_drafts",
    "candidate_profile_confirmations",
    "candidate_consent_copy_catalog",
    "candidate_public_employers",
    "candidate_exclusion_proposals",
    "candidate_profile_sources",
    "candidate_profile_proposals",
    "candidate_profile_source_receipts",
    "candidate_profile_interpretations",
    "candidate_profile_interpretation_receipts",
)
PROFILE_RUNTIME_FUNCTIONS = (
    "candidate_exclusion_preview(uuid,uuid,uuid,bigint,jsonb)",
    "candidate_profile_interpretation_receive(uuid,uuid,uuid,text,text)",
    "candidate_profile_interpretation_read(uuid,uuid,uuid)",
    "candidate_profile_interpretation_apply(uuid,uuid,uuid,text,text)",
    "candidate_profile_apply(uuid,uuid,uuid,text,jsonb)",
    "candidate_profile_read(uuid,uuid)",
    "candidate_consent_v2_project(uuid,uuid,uuid,bigint,text,integer,text,text[],text)",
    "candidate_preference_draft_apply(uuid,uuid,uuid,text,text)",
    "candidate_profile_v3_storage_read(uuid,uuid)",
    "candidate_profile_confirm_v3(uuid,uuid,uuid,text,text)",
    "candidate_exclusions_apply(uuid,uuid,uuid,text,text)",
    "candidate_profile_expire_v3(uuid,uuid,uuid,uuid,text)",
    "candidate_profile_expire_command(uuid,uuid,uuid,uuid,text)",
    "candidate_profile_source_receive(uuid,uuid,uuid,text,text)",
    "candidate_profile_source_read(uuid,uuid,uuid,uuid)",
    "candidate_profile_source_apply(uuid,uuid,uuid,text,text)",
)
# Only routines behind authenticated product routes are granted. Source/preview
# creation, erasure and canonicalization helpers remain owner-only.
PROFILE_OWNER_FUNCTIONS = (
    "candidate_profile_interpretations_scrub()",
    "candidate_profile_erase(uuid,uuid,uuid,text)",
    "candidate_profile_canonical(jsonb)",
    "candidate_preference_draft_scrub()",
    "candidate_company_choices(uuid,uuid,text)",
    "candidate_exclusion_proposals_scrub()",
    "candidate_profile_source_date_valid(jsonb)",
    "candidate_profile_source_fact_valid(jsonb)",
    "candidate_profile_sources_scrub()",
)
PROFILE_CATALOG_MD5 = "1cef6651171465a6d4092ce59e1573b6"
_PROFILE_TABLE_FILTER = "c.relname IN (" + ",".join("'" + t + "'" for t in PROFILE_TABLES) + ")"
_PROFILE_FUNCTION_FILTER = (
    "(starts_with(p.proname,'candidate_profile_')"
    " OR starts_with(p.proname,'candidate_preference_draft_')"
    " OR p.proname IN ('candidate_consent_v2_project','candidate_company_choices',"
    "'candidate_exclusion_preview','candidate_exclusion_proposals_scrub','candidate_exclusions_apply'))"
)
_PROFILE_RUNTIME_NAMES = ",".join(
    "'" + signature.split("(")[0] + "'" for signature in PROFILE_RUNTIME_FUNCTIONS
)
PROFILE_CATALOG_SQL = (
    FOUNDATION_CATALOG_SQL.replace(
        "starts_with(c.relname,'candidate_owned_')",
        _PROFILE_TABLE_FILTER,
    )
    .replace("starts_with(p.proname,'candidate_owned_')", _PROFILE_FUNCTION_FILTER)
    .replace("relname='candidate_owned_subjects'", "relname='candidate_profile_heads'")
    .replace(
        "NOT has_function_privilege(ch.role,p.oid,'EXECUTE')",
        "has_function_privilege(ch.role,p.oid,'EXECUTE') IS DISTINCT FROM (p.proname IN ("
        + _PROFILE_RUNTIME_NAMES
        + "))",
    )
    .replace(
        "OR a.is_grantable)))",
        "OR a.is_grantable OR p.proname NOT IN (" + _PROFILE_RUNTIME_NAMES + "))))",
    )
    .replace(
        " 'tables',",
        """ 'consent_shape',(SELECT jsonb_agg(jsonb_build_array(c.relname,
          (SELECT jsonb_agg(jsonb_build_array(a.attname,format_type(a.atttypid,a.atttypmod),a.attnotnull,
            pg_get_expr(d.adbin,d.adrelid)) ORDER BY a.attnum)
           FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid=a.attrelid AND d.adnum=a.attnum
           WHERE a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped),
          (SELECT jsonb_agg(jsonb_build_array(x.conname,pg_get_constraintdef(x.oid),x.convalidated)
            ORDER BY x.conname COLLATE "C") FROM pg_constraint x WHERE x.conrelid=c.oid)
        ) ORDER BY c.relname COLLATE "C") FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
         WHERE n.nspname='public' AND c.relname IN ('candidate_consent_state','candidate_consent_receipts')),
        'tables',""",
    )
)


def profile_catalog_evidence(cur, role=None):
    cur.execute(
        PROFILE_CATALOG_SQL.replace(
            "WHERE NOT t.tgisinternal AND",
            "WHERE NOT t.tgisinternal AND t.tgname NOT IN ('candidate_network_confirmed','candidate_network_profile_changed','candidate_network_operator_erased') AND",
        ),
        (role,),
    )
    return tuple(cur.fetchone())


def profile_catalog_ready(cur, role=None):
    digest, permissions = profile_catalog_evidence(cur, role)
    if digest != PROFILE_CATALOG_MD5 or permissions is not True:
        return False
    cur.execute("SELECT to_regclass('public.candidate_network_subjects')")
    row = cur.fetchone()
    if (row[0] if not isinstance(row, dict) else next(iter(row.values()))) is not None:
        from activekg.candidate_profiles.publication import network_catalog_ready

        return network_catalog_ready(cur, role)
    return True


def profile_view(value):
    obj(
        value,
        "schemaVersion draftVersion profileVersion preferenceVersion draft confirmedProfile preferences confirmedConsentVersion publicationActive",
    )
    need(type(value["schemaVersion"]) is int and value["schemaVersion"] == 2)
    need(
        value["publicationActive"] is False
        and type(value["confirmedConsentVersion"]) is int
        and value["confirmedConsentVersion"] == 0
    )
    for key in ("draftVersion", "profileVersion", "preferenceVersion"):
        value[key] = revision(value[key])
    for key in ("draft", "confirmedProfile"):
        if value[key] is not None:
            checked = profile(value[key])
            need(checked == value[key])
    if value["preferences"] is not None:
        need(preferences(value["preferences"]) == value["preferences"])
    draft = value["draft"]
    missing = []
    if draft is None:
        missing.append("profile")
    else:
        if not draft["displayName"]:
            missing.append("displayName")
        if draft["currentLocation"] is None:
            missing.extend(["currentLocation.city", "currentLocation.countryCode"])
        if not draft["skills"]:
            missing.append("skills")
    if value["preferences"] is None:
        missing.append("preferences")
    return {
        **value,
        "state": "confirmed"
        if value["confirmedProfile"] is not None and draft == value["confirmedProfile"]
        else "draft",
        "draftHash": hashlib.sha256(canonical(draft).encode()).hexdigest()
        if draft is not None
        else None,
        "missingFields": missing,
        "sources": [],
        "proposals": [],
        "capabilities": {
            "canEdit": True,
            "editBlockedReason": None,
            "canConfirm": not missing,
            "confirmBlockedReason": "incomplete_profile" if missing else None,
            "canEnrich": False,
            "enrichBlockedReason": "feature_unavailable",
            "canRequestResume": False,
            "resumeBlockedReason": "feature_unavailable",
        },
        "matchingState": "not_available",
    }


def checked_receipt(result, subject, binding, command_id, digest):
    obj(
        result,
        "schemaVersion commandId subjectId bindingId commandHash status draftVersion profileVersion preferenceVersion reason publicationActive",
    )
    need(type(result["schemaVersion"]) is int and result["schemaVersion"] == 2)
    need(
        (result["commandId"], result["subjectId"], result["bindingId"], result["commandHash"])
        == (command_id, subject, binding, digest)
    )
    need(result["publicationActive"] is False)
    need(result["status"] in ("applied", "rejected"))
    need((result["status"] == "applied") == (result["reason"] is None))
    need(
        result["reason"]
        in (
            None,
            "stale_edit",
            "confirmation_conflict",
            "incomplete_profile",
            "candidate_privacy_restricted",
            "invalid_request",
            "command_expired",
        )
    )
    for key in ("draftVersion", "profileVersion", "preferenceVersion"):
        revision(result[key])
    return result


def erase_owned_profile(subject, binding, request_id):
    """Erase an exactly bound owner; quarantine must not block deletion itself.

    Only the dedicated authenticated erasure route calls this function. The SQL
    routine serializes with writes and retains a permanent delayed-write fence.
    """
    for value in (subject, binding, request_id):
        if not isinstance(value, str) or not UUID.fullmatch(value):
            raise HTTPException(422, "invalid_request")
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='3000ms'")
            cur.execute(
                "SELECT candidate_profile_erase(%s::uuid,%s::uuid,%s::uuid,'candidate') AS result",
                (subject, binding, request_id),
            )
            row = cur.fetchone()
            expected = {"requestId": request_id, "status": "erased"}
            if row is None or row["result"] != expected:
                raise HTTPException(503, "authority_unavailable")
            return {**expected, "subjectId": subject, "bindingId": binding, "schemaVersion": 1}
    except HTTPException:
        raise
    except psycopg.Error as exc:
        if (exc.sqlstate, exc.diag.message_primary) == ("P0002", "not_found"):
            raise HTTPException(404, "not_found") from None
        raise HTTPException(503, "authority_unavailable") from None


def execute_profile(subject, binding, *, raw=None, command_id=None, digest=None):
    try:
        need(isinstance(subject, str) and UUID.fullmatch(subject) is not None)
        need(isinstance(binding, str) and UUID.fullmatch(binding) is not None)
        if raw is not None:
            need(isinstance(command_id, str) and UUID.fullmatch(command_id) is not None)
            _, wire, actual = parse_profile_command(raw)
            need(actual == digest and wire.encode() == raw)
        with _connect() as conn, conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='1500ms'")
            cur.execute("SET LOCAL statement_timeout='3000ms'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout='4000ms'")
            # Bind before any subject-dependent privacy response or data disclosure.
            cur.execute(
                "SELECT candidate_owned_status(%s::uuid,%s::uuid) AS result", (subject, binding)
            )
            if not cur.fetchone():
                raise HTTPException(404, "not_found")
            check_owner_privacy(cur, subject)
            if raw is None:
                cur.execute(
                    "SELECT candidate_profile_read(%s::uuid,%s::uuid) AS result", (subject, binding)
                )
            else:
                cur.execute(
                    "SELECT candidate_profile_apply(%s::uuid,%s::uuid,%s::uuid,%s,%s::jsonb) AS result",
                    (subject, binding, command_id, digest, Jsonb(wire)),
                )
            row = cur.fetchone()
            need(row is not None and isinstance(row["result"], dict))
            result = row["result"]
            # Recheck before commit/response. Unavailable authority rolls back writes.
            check_owner_privacy(cur, subject)
            return (
                profile_view(result)
                if raw is None
                else checked_receipt(result, subject, binding, command_id, digest)
            )
    except HTTPException:
        raise
    except psycopg.Error as exc:
        marker = exc.diag.message_primary
        mapping = {
            ("P0002", "not_found"): 404,
            ("23505", "idempotency_conflict"): 409,
            ("42501", "candidate_privacy_restricted"): 451,
            ("22023", "invalid_request"): 422,
        }
        status = mapping.get((exc.sqlstate, marker), 503)
        raise HTTPException(status, marker if status != 503 else "authority_unavailable") from None
    except Exception:
        raise HTTPException(503, "authority_unavailable") from None
