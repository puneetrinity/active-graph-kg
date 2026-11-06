"""Strict candidate Spark commands, receipts, consent and exclusion contracts."""

import hashlib
import json
import re
from datetime import datetime

from activekg.candidate_profiles import profile_contracts as old


def uid(value):
    old.need(type(value) is str and old.UUID.fullmatch(value) is not None)
    return value


def interpretation_proposal(value):
    # Import locally: source contracts already reuse the preference validators.
    from activekg.candidate_profiles.source_contracts import raw_text, source_fact, timestamp

    version_keys = (
        "expectedDraftVersion",
        "expectedProfileVersion",
        "expectedPreferenceVersion",
        "expectedPreferenceDraftRevision",
    )
    old.obj(
        value,
        "schemaVersion status summary changes proposalId proposalHash expiresAt "
        + " ".join(version_keys),
    )
    old.need(type(value["schemaVersion"]) is int and value["schemaVersion"] == 1)
    old.need(value["status"] == "proposal")
    result = {
        **value,
        "summary": raw_text(value["summary"], 2000, 1),
        "proposalId": uid(value["proposalId"]),
        "proposalHash": digest(value["proposalHash"]),
        "expiresAt": timestamp(value["expiresAt"]),
    }
    for key in version_keys:
        result[key] = old.revision(value[key])
    changes = []
    seen = set()
    for change in old.array(value["changes"], 20, 1):
        old.need(type(change) is dict)
        kind = old.enum(change.get("kind"), "source_correction preference_section exclusions")
        if kind == "source_correction":
            old.obj(change, "kind fact")
            fact = source_fact(change["fact"])
            old.need(fact["kind"] not in ("displayName", "linkedinUrl", "photo"))
            parsed = {"kind": kind, "fact": fact}
            keys = ["fact:" + fact["factId"]]
        elif kind == "preference_section":
            old.obj(change, "kind answer")
            answer = section(change["answer"])
            old.need(answer["section"] != "authorisation")
            parsed = {"kind": kind, "answer": answer}
            keys = ["section:" + answer["section"]]
        else:
            old.obj(change, "kind restrictions")
            restrictions = []
            for r in old.array(change["restrictions"], 20, 1):
                old.need(type(r) is dict)
                rule = old.enum(r.get("kind"), "remote_only no_travel exclude_company")
                old.obj(r, "kind companyId" if rule == "exclude_company" else "kind")
                restrictions.append(
                    {
                        "kind": rule,
                        **({"companyId": uid(r["companyId"])} if rule == "exclude_company" else {}),
                    }
                )
            parsed = {"kind": kind, "restrictions": restrictions}
            keys = ["restriction:" + old.canonical(r) for r in restrictions]
        old.need(len(set(keys)) == len(keys) and not seen.intersection(keys))
        seen.update(keys)
        changes.append(parsed)
    result["changes"] = changes
    body = {k: v for k, v in result.items() if k != "proposalHash"}
    old.need(hashlib.sha256(old.canonical(body).encode()).hexdigest() == result["proposalHash"])
    return result


def validate_interpretation_receipt(transfer, receipt):
    old.obj(
        receipt,
        "schemaVersion commandId subjectId bindingId commandHash proposalId proposalHash outcome reason",
    )
    old.need(type(receipt["schemaVersion"]) is int and receipt["schemaVersion"] == 1)
    for key in ("commandId", "subjectId", "bindingId"):
        old.need(receipt[key] == transfer[key])
    old.need(receipt["commandHash"] == hashlib.sha256(old.canonical(transfer).encode()).hexdigest())
    old.need(receipt["proposalId"] == transfer["proposal"]["proposalId"])
    old.need(receipt["proposalHash"] == transfer["proposal"]["proposalHash"])
    old.enum(receipt["outcome"], "stored rejected")
    if receipt["outcome"] == "stored":
        old.need(receipt["reason"] is None)
    else:
        old.enum(receipt["reason"], "consent_changed stale_edit proposal_expired source_conflict")
    return receipt


def parse_interpretation_transfer(raw):
    old.need(type(raw) is bytes and len(raw) <= 128 * 1024)
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=old._pairs, parse_constant=lambda _: old.need(False)
    )
    old._walk(value)
    old.obj(
        value,
        "schemaVersion commandId subjectId bindingId consentRevision use "
        "sourceProposalId sourceVersion targetRowId proposal",
    )
    old.need(type(value["schemaVersion"]) is int and value["schemaVersion"] == 1)
    for key in ("commandId", "subjectId", "bindingId"):
        uid(value[key])
    old.need(old.revision(value["consentRevision"]) > 0)
    use = old.enum(value["use"], "source_correction private_exclusions change_preferences")
    if use == "source_correction":
        uid(value["sourceProposalId"])
        old.revision(value["sourceVersion"])
        if value["targetRowId"] is not None:
            uid(value["targetRowId"])
    else:
        old.need(
            all(value[k] is None for k in ("sourceProposalId", "sourceVersion", "targetRowId"))
        )
    result = {**value, "proposal": interpretation_proposal(value["proposal"])}
    for change in result["proposal"]["changes"]:
        old.need(
            change["kind"]
            == {
                "source_correction": "source_correction",
                "change_preferences": "preference_section",
                "private_exclusions": "exclusions",
            }[use]
            or (
                use == "private_exclusions"
                and change["kind"] == "preference_section"
                and change["answer"]["section"] == "private_notes"
            )
        )
    wire = old.canonical(result)
    old.need(wire.encode() == raw)
    return {"value": result, "canonical": wire, "hash": hashlib.sha256(raw).hexdigest()}


def digest(value):
    import re

    old.need(type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None)
    return value


def ids(value):
    return old.distinct([uid(v) for v in old.array(value, 400)])


def section(payload):
    old.obj(payload, "section answered value")
    name = old.enum(
        payload["section"],
        "roles location arrangement relocation employment_types availability compensation authorisation private_notes",
    )
    old.need(type(payload["answered"]) is bool)
    v = payload["value"]
    if not payload["answered"]:
        old.need(v is None)
        return dict(payload)
    if name == "roles":
        old.obj(v, "desiredRoles preferredSeniority")
        v = {
            "desiredRoles": old.text_list(v["desiredRoles"], 10, 200, 1),
            "preferredSeniority": old.text_list(v["preferredSeniority"], 8, 100),
        }
    elif name == "location":
        old.need(type(v) is dict)
        if v.get("mode") == "anywhere":
            old.obj(v, "mode")
        else:
            old.obj(v, "mode places")
            old.need(v["mode"] == "specific")
            v = {
                "mode": "specific",
                "places": [old.place(p) for p in old.array(v["places"], 20, 1)],
            }
    elif name == "arrangement":
        old.obj(v, "workArrangements remoteRegions")
        choices = old.distinct(
            [old.enum(x, "remote hybrid onsite") for x in old.array(v["workArrangements"], 3, 1)]
        )
        regions = v["remoteRegions"]
        if regions is None:
            old.need("remote" not in choices)
        else:
            old.need(type(regions) is dict)
            if regions.get("mode") == "worldwide":
                old.obj(regions, "mode")
            else:
                old.obj(regions, "mode countries")
                old.need(regions["mode"] == "countries")
                old.distinct(old.array(regions["countries"], 249, 1))
                old.need(all(type(x) is str and x in old.COUNTRIES for x in regions["countries"]))
        v = {"workArrangements": choices, "remoteRegions": regions}
    elif name == "relocation":
        old.obj(v, "mode destinations")
        v = {
            "mode": old.enum(v["mode"], "yes no discuss"),
            "destinations": [old.place(x) for x in old.array(v["destinations"], 20)],
        }
    elif name == "employment_types":
        v = old.distinct(
            [
                old.enum(x, "permanent contract gig_freelance part_time internship")
                for x in old.array(v, 5, 1)
            ]
        )
    elif name == "availability":
        old.need(type(v) is dict)
        mode = old.enum(v.get("mode"), "unknown immediate date notice_days")
        old.obj(v, "mode" + {"date": " date", "notice_days": " days"}.get(mode, ""))
        if mode == "date":
            old.date({"value": v["date"], "precision": "day"})
        if mode == "notice_days":
            days = old.revision(v["days"])
            old.need(days <= 730)
            v = {"mode": mode, "days": days}
    elif name == "compensation":
        v = old.compensation(v)
    elif name == "authorisation":
        result = []
        for item in old.array(v, 249):
            old.obj(item, "countryCode status")
            old.need(type(item["countryCode"]) is str and item["countryCode"] in old.COUNTRIES)
            result.append(
                {
                    "countryCode": item["countryCode"],
                    "status": old.enum(item["status"], "allowed needs_visa not_sure"),
                }
            )
        old.distinct([x["countryCode"] for x in result])
        v = result
    elif name == "private_notes":
        v = old.distinct([old.text(x, 2000, 1) for x in old.array(v, 20)])
    return {"section": name, "answered": True, "value": v}


def correction(item):
    old.obj(item, "factId kind value")
    uid(item["factId"])
    kind = old.enum(
        item["kind"],
        "displayName headline linkedinUrl photo currentLocation skill employment education project",
    )
    value = item["value"]
    if kind in ("displayName", "headline", "skill"):
        value = old.text(
            value,
            {"displayName": 200, "headline": 300, "skill": 100}[kind],
            1 if kind == "skill" else 0,
        )
    elif kind == "photo":
        old.obj(value, "url")
        value = {"url": old.photo_url(value["url"])}
    elif kind == "currentLocation":
        value = None if value is None else old.place(value, True)
    elif kind == "linkedinUrl":
        base = {
            "displayName": "",
            "headline": "",
            "linkedinUrl": value,
            "currentLocation": None,
            "skills": [],
            "employment": [],
            "education": [],
            "projects": [],
        }
        value = old.profile(base)["linkedinUrl"]
    else:
        value = old.rows(
            [value],
            {"employment": "employment", "education": "education", "project": "projects"}[kind],
            1,
        )[0]
    return {"factId": item["factId"], "kind": kind, "value": value}


def parse_expiry(raw):
    old.need(type(raw) is bytes and len(raw) <= 4096)
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=old._pairs, parse_constant=lambda _: old.need(False)
    )
    old._walk(value)
    old.obj(value, "schemaVersion commandId bindingId idempotencyKey commandHash")
    old.need(type(value["schemaVersion"]) is int and value["schemaVersion"] in (2, 3))
    result = {
        "schemaVersion": value["schemaVersion"],
        "commandId": uid(value["commandId"]),
        "bindingId": uid(value["bindingId"]),
        "idempotencyKey": uid(value["idempotencyKey"]),
        "commandHash": digest(value["commandHash"]),
    }
    canonical = old.canonical(result)
    return {
        "value": result,
        "canonical": canonical,
        "hash": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def parse_command(raw):
    old.need(type(raw) is bytes and len(raw) <= 384 * 1024, "payload_too_large")
    value = json.loads(
        raw.decode("utf8"), object_pairs_hook=old._pairs, parse_constant=lambda _: old.need(False)
    )
    old._walk(value)
    old.obj(
        value,
        "schemaVersion idempotencyKey expectedDraftVersion expectedProfileVersion expectedPreferenceVersion expectedPreferenceDraftRevision action payload",
    )
    old.need(type(value["schemaVersion"]) in (int, float) and value["schemaVersion"] == 3)
    result = {"schemaVersion": 3, "idempotencyKey": uid(value["idempotencyKey"])}
    for key in (
        "expectedDraftVersion",
        "expectedProfileVersion",
        "expectedPreferenceVersion",
        "expectedPreferenceDraftRevision",
    ):
        result[key] = old.revision(value[key])
    action = old.enum(
        value["action"],
        "save_draft save_preference_section accept_source reject_source accept_interpretation accept_exclusions remove_exclusion promote_preferences confirm request_source",
    )
    p = value["payload"]
    if action == "save_draft":
        old.obj(p, "profile")
        p = {"profile": old.profile(p["profile"])}
        old.need(len(old.canonical(p["profile"]).encode()) <= 256 * 1024, "payload_too_large")
    elif action == "save_preference_section":
        p = section(p)
        old.need(len(old.canonical(p).encode()) <= 32 * 1024, "payload_too_large")
    elif action == "accept_source":
        old.obj(
            p, "proposalId sourceVersion proposalHash selectedFactIds replaceFactIds corrections"
        )
        p = {
            **p,
            "proposalId": uid(p["proposalId"]),
            "sourceVersion": old.revision(p["sourceVersion"]),
            "proposalHash": digest(p["proposalHash"]),
            "selectedFactIds": ids(p["selectedFactIds"]),
            "replaceFactIds": ids(p["replaceFactIds"]),
            "corrections": [correction(x) for x in old.array(p["corrections"], 400)],
        }
        old.need(
            len(p["selectedFactIds"]) > 0 and set(p["replaceFactIds"]) <= set(p["selectedFactIds"])
        )
        old.distinct([x["factId"] for x in p["corrections"]])
        old.need(all(x["factId"] in p["selectedFactIds"] for x in p["corrections"]))
    elif action == "reject_source":
        old.obj(p, "proposalId sourceVersion proposalHash reason")
        p = {
            **p,
            "proposalId": uid(p["proposalId"]),
            "sourceVersion": old.revision(p["sourceVersion"]),
            "proposalHash": digest(p["proposalHash"]),
            "reason": old.enum(p["reason"], "not_me incorrect not_needed"),
        }
    elif action in ("accept_interpretation", "accept_exclusions"):
        old.obj(p, "proposalId proposalHash")
        p = {"proposalId": uid(p["proposalId"]), "proposalHash": digest(p["proposalHash"])}
    elif action == "remove_exclusion":
        old.obj(p, "restrictionId")
        p = {"restrictionId": uid(p["restrictionId"])}
    elif action == "promote_preferences":
        old.obj(p, "preferenceDraftHash")
        p = {"preferenceDraftHash": digest(p["preferenceDraftHash"])}
    elif action == "confirm":
        old.obj(p, "draftHash preferenceDraftHash consentRevision")
        p = {
            "draftHash": digest(p["draftHash"]),
            "preferenceDraftHash": digest(p["preferenceDraftHash"]),
            "consentRevision": old.revision(p["consentRevision"]),
        }
    elif action == "request_source":
        old.need(type(p) is dict)
        kind = old.enum(p.get("kind"), "resume professional_lookup")
        if kind == "resume":
            old.obj(p, "kind sourceId sourceVersion")
            p = {
                "kind": kind,
                "sourceId": uid(p["sourceId"]),
                "sourceVersion": old.revision(p["sourceVersion"]),
            }
        else:
            old.obj(p, "kind attemptId")
            p = {"kind": kind, "attemptId": uid(p["attemptId"])}
    result.update(action=action, payload=p)
    canonical = old.canonical(result)
    old.need(len(canonical.encode()) <= 384 * 1024, "payload_too_large")
    return {
        "value": result,
        "canonical": canonical,
        "hash": hashlib.sha256(canonical.encode()).hexdigest(),
    }


def receipt(value):
    old.obj(
        value,
        "schemaVersion commandId subjectId bindingId commandHash status reason changed draftVersion profileVersion preferenceVersion preferenceDraftRevision preferenceDraftHash publicationActive",
    )
    old.need(
        type(value["schemaVersion"]) in (int, float)
        and value["schemaVersion"] == 3
        and value["publicationActive"] is False
    )
    result = dict(value, schemaVersion=3)
    for key in ("commandId", "subjectId", "bindingId"):
        result[key] = uid(value[key])
    result["commandHash"] = digest(value["commandHash"])
    result["status"] = old.enum(value["status"], "applied rejected")
    result["reason"] = (
        None
        if value["reason"] is None
        else old.enum(
            value["reason"],
            "stale_edit confirmation_conflict incomplete_preferences incomplete_profile legacy_preferences_review_required candidate_privacy_restricted command_expired consent_required consent_changed source_conflict proposal_expired exclusion_limit",
        )
    )
    old.need(
        type(value["changed"]) is bool
        and ((value["status"] == "applied") == (value["reason"] is None))
        and (value["status"] != "rejected" or not value["changed"])
    )
    for key in ("draftVersion", "profileVersion", "preferenceVersion", "preferenceDraftRevision"):
        result[key] = old.revision(value[key])
    result["preferenceDraftHash"] = (
        None if value["preferenceDraftHash"] is None else digest(value["preferenceDraftHash"])
    )
    return result


def preference_sections(value):
    old.need(type(value) is dict)
    result = {}
    for name, answer in value.items():
        old.obj(answer, "answered value")
        parsed = section({"section": name, **answer})
        result[name] = {"answered": parsed["answered"], "value": parsed["value"]}
    return result


def exclusions(value):
    old.need(type(value) is list and len(value) <= 20)
    result = []
    ids = set()
    meanings = set()
    for item in value:
        old.need(type(item) is dict)
        kind = old.enum(item.get("kind"), "remote_only no_travel exclude_company")
        old.obj(item, "id kind companyId" if kind == "exclude_company" else "id kind")
        parsed = {"id": uid(item["id"]), "kind": kind}
        if kind == "exclude_company":
            parsed["companyId"] = uid(item["companyId"])
        meaning = (kind, parsed.get("companyId"))
        old.need(parsed["id"] not in ids and meaning not in meanings)
        ids.add(parsed["id"])
        meanings.add(meaning)
        result.append(parsed)
    return result


def storage_read(value):
    from activekg.candidate_profiles.source_contracts import timestamp

    old.need(type(value) is dict)
    value = {
        "sources": [],
        "proposals": [],
        "hasMoreSources": False,
        "hasMoreProposals": False,
        **value,
    }
    old.obj(
        value,
        "schemaVersion draftVersion profileVersion preferenceVersion draft confirmedProfile preferences preferenceDraft publicationActive sources proposals hasMoreSources hasMoreProposals",
    )
    old.need(
        type(value["schemaVersion"]) in (int, float)
        and value["schemaVersion"] == 3
        and value["publicationActive"] is False
    )
    result = dict(value, schemaVersion=3)
    for key in ("hasMoreSources", "hasMoreProposals"):
        old.need(type(value[key]) is bool)
    for key in ("sources", "proposals"):
        old.need(type(value[key]) is list and len(value[key]) <= 50)
    for source in value["sources"]:
        old.obj(source, "sourceId sourceVersion")
        uid(source["sourceId"])
        old.revision(source["sourceVersion"])
    for proposal in value["proposals"]:
        old.obj(proposal, "proposalId proposalHash kind sourceId sourceVersion expiresAt")
        uid(proposal["proposalId"])
        digest(proposal["proposalHash"])
        timestamp(proposal["expiresAt"])
        old.need(proposal["kind"] in ("source", "interpretation"))
        if proposal["sourceId"] is not None:
            uid(proposal["sourceId"])
        if proposal["sourceVersion"] is not None:
            old.revision(proposal["sourceVersion"])
    for key in ("draftVersion", "profileVersion", "preferenceVersion"):
        result[key] = old.revision(value[key])
    for key in ("draft", "confirmedProfile"):
        result[key] = None if value[key] is None else old.profile(value[key])
    prefs = value["preferences"]
    if prefs is not None:
        old.need(type(prefs) is dict)
        if prefs.get("schemaVersion") == 3:
            old.obj(prefs, "schemaVersion sections exclusions")
            result["preferences"] = {
                "schemaVersion": 3,
                "sections": preference_sections(prefs["sections"]),
                "exclusions": exclusions(prefs["exclusions"]),
            }
        else:
            result["preferences"] = old.preferences(prefs)
    draft = value["preferenceDraft"]
    if draft is not None:
        old.obj(
            draft,
            "preferenceDraftRevision preferenceDraftHash sections exclusions promotedPreferenceVersion promotedDraftRevision",
        )
        d = {
            "preferenceDraftRevision": old.revision(draft["preferenceDraftRevision"]),
            "preferenceDraftHash": digest(draft["preferenceDraftHash"]),
            "sections": preference_sections(draft["sections"]),
            "exclusions": exclusions(draft["exclusions"]),
        }
        for key in ("promotedPreferenceVersion", "promotedDraftRevision"):
            d[key] = None if draft[key] is None else old.revision(draft[key])
        old.need((d["promotedPreferenceVersion"] is None) == (d["promotedDraftRevision"] is None))
        old.need(
            d["promotedDraftRevision"] is None
            or d["promotedDraftRevision"] <= d["preferenceDraftRevision"]
        )
        result["preferenceDraft"] = d
    return result


def parse_response(raw, kind):
    old.need(kind in ("receipt", "storage_read"))
    old.need(type(raw) is bytes and len(raw) <= 600 * 1024)
    value = json.loads(
        raw.decode("utf8"), object_pairs_hook=old._pairs, parse_constant=lambda _: old.need(False)
    )
    old._walk(value)
    result = receipt(value) if kind == "receipt" else storage_read(value)
    return {"value": result, "canonical": old.canonical(result)}


BUNDLED_PURPOSES = [
    "job_matching",
    "profile_enrichment",
    "ai_processing",
    "talent_network_discovery",
    "recruiter_contact",
]


def parse_consent_projection(raw):
    old.need(type(raw) is bytes and len(raw) <= 4096)
    value = json.loads(
        raw.decode("utf8"), object_pairs_hook=old._pairs, parse_constant=lambda _: old.need(False)
    )
    old._walk(value)
    old.obj(
        value,
        "schemaVersion subjectId bindingId eventId version action copyVersion copyHash purposes",
    )
    old.need(type(value["schemaVersion"]) in (int, float) and value["schemaVersion"] == 2)
    result = {**value, "schemaVersion": 2}
    for key in ("subjectId", "bindingId", "eventId"):
        result[key] = uid(value[key])
    result["version"] = old.revision(value["version"])
    old.need(result["version"] > 0)
    result["action"] = old.enum(value["action"], "grant withdraw")
    if result["action"] == "grant":
        result["copyVersion"] = old.revision(value["copyVersion"])
        old.need(0 < result["copyVersion"] <= 2147483647)
        result["copyHash"] = digest(value["copyHash"])
        old.need(value["purposes"] == BUNDLED_PURPOSES)
    else:
        old.need(
            value["copyVersion"] is None and value["copyHash"] is None and value["purposes"] == []
        )
    canonical = old.canonical(result)
    return {
        "value": result,
        "canonical": canonical,
        "hash": hashlib.sha256(canonical.encode()).hexdigest(),
    }


"""Strict counterpart to spark-exclusion-api.ts. No provider calls."""
EFFECTS = {
    "remote_only": "Exclude jobs explicitly requiring on-site or hybrid work.",
    "no_travel": "Exclude jobs explicitly requiring travel.",
    "exclude_company": "Exclude jobs from the resolved company. This does not hide your profile from its recruiters.",
}
DOMAIN = r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"


def company(value, snapshot=False):
    old.obj(
        value,
        "companyId displayName domain identityRevision"
        if snapshot
        else "companyId displayName domain",
    )
    import unicodedata

    name = value["displayName"]
    domain = value["domain"]
    old.need(
        type(name) is str
        and 0 < len(name) <= 200
        and name == name.strip()
        and name == unicodedata.normalize("NFC", name)
    )
    old.need(
        type(domain) is str and len(domain) <= 253 and re.fullmatch(DOMAIN, domain) is not None
    )
    result = {"companyId": uid(value["companyId"]), "displayName": name, "domain": domain}
    if snapshot:
        result["identityRevision"] = old.revision(value["identityRevision"])
        old.need(result["identityRevision"] > 0)
    return result


def parse_exclusion_api(raw, kind):
    old.need(type(raw) is bytes and len(raw) <= 32 * 1024)
    v = json.loads(
        raw.decode(), object_pairs_hook=old._pairs, parse_constant=lambda _: old.need(False)
    )
    old._walk(v)
    if kind == "company_choices":
        old.need(type(v) is list and len(v) <= 10)
        result = [company(c) for c in v]
        old.need(len({c["companyId"] for c in result}) == len(result))
    elif kind == "exclusion_preview_request":
        old.obj(v, "idempotencyKey expectedPreferenceDraftRevision restrictions")
        rules = v["restrictions"]
        old.need(type(rules) is list and 1 <= len(rules) <= 20)
        parsed = []
        seen = set()
        for r in rules:
            old.need(type(r) is dict)
            kind_value = old.enum(r.get("kind"), "remote_only no_travel exclude_company")
            old.obj(r, "kind companyId" if kind_value == "exclude_company" else "kind")
            item = {"kind": kind_value}
            if kind_value == "exclude_company":
                item["companyId"] = uid(r["companyId"])
            key = (kind_value, item.get("companyId"))
            old.need(key not in seen)
            seen.add(key)
            parsed.append(item)
        result = {
            "idempotencyKey": uid(v["idempotencyKey"]),
            "expectedPreferenceDraftRevision": old.revision(v["expectedPreferenceDraftRevision"]),
            "restrictions": parsed,
        }
    elif kind == "exclusion_preview":
        old.obj(
            v,
            "proposalId proposalHash expectedPreferenceDraftRevision restrictions effects companies unknownPolicy expiresAt",
        )
        rules = exclusions(v["restrictions"])
        old.need(len(rules) > 0)
        effects = v["effects"]
        old.need(type(effects) is list and len(effects) == len(rules))
        for r, e in zip(rules, effects, strict=False):
            old.obj(e, "restrictionId text")
            old.need(uid(e["restrictionId"]) == r["id"] and e["text"] == EFFECTS[r["kind"]])
        old.need(type(v["companies"]) is list and len(v["companies"]) <= 20)
        companies = [company(c, True) for c in v["companies"]]
        old.need(
            [r["companyId"] for r in rules if "companyId" in r]
            == [c["companyId"] for c in companies]
        )
        expires = v["expiresAt"]
        old.need(
            type(expires) is str
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z", expires) is not None
        )
        datetime.strptime(expires, "%Y-%m-%dT%H:%M:%S.%fZ")
        old.need(v["unknownPolicy"] == "show_with_warning")
        result = {
            **v,
            "proposalId": uid(v["proposalId"]),
            "proposalHash": digest(v["proposalHash"]),
            "expectedPreferenceDraftRevision": old.revision(v["expectedPreferenceDraftRevision"]),
            "restrictions": rules,
            "companies": companies,
        }
        payload = {k: x for k, x in result.items() if k != "proposalHash"}
        old.need(
            hashlib.sha256(old.canonical(payload).encode()).hexdigest() == result["proposalHash"]
        )
    else:
        raise ValueError("invalid_kind")
    return {"value": result, "canonical": old.canonical(result)}
