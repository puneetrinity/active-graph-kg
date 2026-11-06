"""P2 immutable source transfer. Validation only; no route or database activation."""

import hashlib
import json
import math
import re

from activekg.candidate_profiles import profile_contracts as old
from activekg.candidate_profiles import spark_contracts as spark


def raw_text(value, maximum, minimum=0):
    """Source evidence is never normalised; offsets use JavaScript UTF-16 units."""
    old.need(
        type(value) is str
        and "\x00" not in value
        and not any(0xD800 <= ord(c) <= 0xDFFF for c in value)
    )
    old.need(minimum <= len(value.encode("utf-16-le")) // 2 <= maximum)
    return value


def version(value):
    old.need(type(value) in (int, float) and value == 1)
    return 1


def timestamp(value):
    old.need(type(value) is str)
    match = re.fullmatch(
        r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2})(?::([0-9]{2})(?:\.[0-9]+)?)?Z",
        value,
    )
    old.need(match is not None)
    year, month, day, hour, minute = map(int, match.groups()[:5])
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    old.need(1 <= month <= 12 and 1 <= day <= days[month - 1])
    old.need(hour < 24 and minute < 60 and int(match[6] or 0) < 60)
    return value


def source_fact(value):
    old.obj(value, "factId kind targetRowId value evidence")
    # Reuse the existing field semantics, including row date/current consistency.
    if value["kind"] == "skill":
        corrected = {
            "factId": spark.uid(value["factId"]),
            "kind": "skill",
            "value": raw_text(value["value"], 100, 1),
        }
    else:
        corrected = spark.correction({k: value[k] for k in ("factId", "kind", "value")})
    target = value["targetRowId"]
    if target is not None:
        spark.uid(target)
        old.need(value["kind"] in ("employment", "education", "project"))
    evidence = []
    for item in old.array(value["evidence"], 8, 1):
        old.obj(item, "sourceId sourceVersion contentHash locator quote")
        evidence.append(
            {
                "sourceId": spark.uid(item["sourceId"]),
                "sourceVersion": old.revision(item["sourceVersion"]),
                "contentHash": spark.digest(item["contentHash"]),
                "locator": raw_text(item["locator"], 200, 1),
                "quote": raw_text(item["quote"], 2048, 1),
            }
        )
    return {**corrected, "targetRowId": target, "evidence": evidence}


def proposal(value):
    old.obj(
        value,
        "schemaVersion proposalId sourceId sourceVersion proposalHash originalDraftVersion "
        "originalProfileVersion originalPreferenceVersion originalPreferenceDraftRevision "
        "sourceKind facts reviewNotes expiresAt",
    )
    facts = [source_fact(v) for v in old.array(value["facts"], 400)]
    old.need(
        sum(len(e["quote"].encode()) for f in facts for e in f["evidence"]) <= 65536,
        "evidence_too_large",
    )
    old.distinct([f["factId"] for f in facts])
    result = {
        "schemaVersion": version(value["schemaVersion"]),
        "proposalId": spark.uid(value["proposalId"]),
        "sourceId": spark.uid(value["sourceId"]),
        "proposalHash": spark.digest(value["proposalHash"]),
        "sourceKind": old.enum(
            value["sourceKind"], "professional_lookup resume legacy_flow_profile"
        ),
        "facts": facts,
        "reviewNotes": [raw_text(v, 2000, 1) for v in old.array(value["reviewNotes"], 20)],
        "expiresAt": timestamp(value["expiresAt"]),
    }
    for key in (
        "sourceVersion",
        "originalDraftVersion",
        "originalProfileVersion",
        "originalPreferenceVersion",
        "originalPreferenceDraftRevision",
    ):
        result[key] = old.revision(value[key])
    return result


def _walk(value, depth=1):
    if type(value) in (dict, list):
        old.need(depth <= 12)
        if type(value) is dict:
            for key in value:
                raw_text(key, 2 * 1024 * 1024)
        for item in value.values() if type(value) is dict else value:
            _walk(item, depth + 1)
    elif type(value) is str:
        raw_text(value, 2 * 1024 * 1024)
    elif type(value) is float:
        old.need(math.isfinite(value))


def _validate_source_transfer(value, kind):
    _walk(value)
    old.obj(
        value,
        "schemaVersion commandId subjectId bindingId sourceId sourceVersion contentHash "
        "textHash policyVersion consentRevision text proposal",
    )
    result = {"schemaVersion": version(value["schemaVersion"])}
    for key in ("commandId", "subjectId", "bindingId", "sourceId"):
        result[key] = spark.uid(value[key])
    for key in ("contentHash", "textHash"):
        result[key] = spark.digest(value[key])
    for key in ("sourceVersion", "consentRevision"):
        result[key] = old.revision(value[key])
    old.need(result["consentRevision"] > 0)
    result["policyVersion"] = raw_text(value["policyVersion"], 80, 1)
    result["text"] = raw_text(value["text"], 512 * 1024, 1)
    result["proposal"] = p = proposal(value["proposal"])
    canonical = old.canonical(result)
    old.need(
        len(canonical.encode()) <= 2 * 1024 * 1024 and len(result["text"].encode()) <= 512 * 1024,
        "payload_too_large",
    )
    old.need(
        hashlib.sha256(result["text"].encode()).hexdigest() == result["textHash"],
        "text_hash_mismatch",
    )
    old.need(
        p["sourceKind"] == kind
        and p["sourceId"] == result["sourceId"]
        and p["sourceVersion"] == result["sourceVersion"],
        "source_mismatch",
    )
    old.need(
        kind == "professional_lookup" or all(f["kind"] != "photo" for f in p["facts"]),
        "source_mismatch",
    )
    if kind == "professional_lookup":
        old.need(
            result["policyVersion"] == "rc-profile-v1"
            and result["contentHash"] == result["textHash"],
            "source_mismatch",
        )
        entries = [{"kind": f["kind"], "value": f["value"]} for f in p["facts"]]
        old.need(old.canonical(entries) == result["text"], "lookup_evidence_mismatch")
        for fact, entry in zip(p["facts"], entries, strict=True):
            old.need(
                fact["targetRowId"] is None
                and len(fact["evidence"]) == 1
                and fact["evidence"][0]["quote"] == old.canonical(entry),
                "lookup_evidence_mismatch",
            )
    body = {k: v for k, v in p.items() if k != "proposalHash"}
    old.need(
        hashlib.sha256(old.canonical(body).encode()).hexdigest() == p["proposalHash"],
        "proposal_hash_mismatch",
    )
    utf16 = result["text"].encode("utf-16-le")
    for fact in p["facts"]:
        for evidence in fact["evidence"]:
            old.need(
                all(evidence[k] == result[k] for k in ("sourceId", "sourceVersion", "contentHash")),
                "evidence_mismatch",
            )
            match = re.fullmatch(r"text:(0|[1-9][0-9]*):(0|[1-9][0-9]*)", evidence["locator"])
            old.need(match is not None, "invalid_evidence_locator")
            start, length = map(int, match.groups())
            old.need(
                start <= old.MAX_SAFE
                and 1 <= length <= old.MAX_SAFE
                and start + length <= len(utf16) // 2,
                "quote_mismatch",
            )
            # Byte comparison also refuses an offset splitting a surrogate pair.
            old.need(
                utf16[start * 2 : (start + length) * 2] == evidence["quote"].encode("utf-16-le"),
                "quote_mismatch",
            )
    return {
        "value": result,
        "canonical": canonical,
        "hash": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def validate_resume_transfer(value):
    return _validate_source_transfer(value, "resume")


def validate_lookup_transfer(value):
    return _validate_source_transfer(value, "professional_lookup")


def parse_resume_transfer(raw, kind="resume"):
    old.need(type(raw) is bytes and len(raw) <= 2 * 1024 * 1024, "payload_too_large")
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=old._pairs, parse_constant=lambda _: old.need(False)
    )
    old.need(kind in ("resume", "professional_lookup"))
    return _validate_source_transfer(value, kind)


def validate_source_receipt(request, value, kind="resume"):
    old.need(kind in ("resume", "professional_lookup"))
    transfer = _validate_source_transfer(request, kind)
    old.obj(
        value,
        "schemaVersion commandId subjectId bindingId commandHash sourceId sourceVersion "
        "proposalId proposalHash outcome reason canonicalProfileChanged publicationActive",
    )
    result = {"schemaVersion": version(value["schemaVersion"])}
    for key in ("commandId", "subjectId", "bindingId", "sourceId", "proposalId"):
        result[key] = spark.uid(value[key])
    for key in ("commandHash", "proposalHash"):
        result[key] = spark.digest(value[key])
    result["sourceVersion"] = old.revision(value["sourceVersion"])
    result["outcome"] = old.enum(value["outcome"], "stored rejected")
    result["reason"] = value["reason"]
    if result["reason"] is not None:
        old.enum(result["reason"], "consent_changed source_conflict stale_edit proposal_expired")
    old.need((result["outcome"] == "stored") == (result["reason"] is None))
    for key in ("canonicalProfileChanged", "publicationActive"):
        old.need(value[key] is False)
        result[key] = False
    old.need(
        all(
            result[k] == transfer["value"][k]
            for k in ("commandId", "subjectId", "bindingId", "sourceId", "sourceVersion")
        ),
        "receipt_mismatch",
    )
    old.need(
        result["commandHash"] == transfer["hash"]
        and result["proposalId"] == transfer["value"]["proposal"]["proposalId"]
        and result["proposalHash"] == transfer["value"]["proposal"]["proposalHash"],
        "receipt_mismatch",
    )
    return result
