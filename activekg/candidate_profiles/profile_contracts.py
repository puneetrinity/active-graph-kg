"""S2A manual profile protocol. Source/provider actions remain gated out."""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
import unicodedata
from typing import Any
from urllib.parse import urlsplit

MAX_SAFE = 9007199254740991
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
COUNTRIES = frozenset(
    "AD AE AF AG AI AL AM AO AQ AR AS AT AU AW AX AZ BA BB BD BE BF BG BH BI BJ BL BM BN BO BQ BR BS BT BV BW BY BZ CA CC CD CF CG CH CI CK CL CM CN CO CR CU CV CW CX CY CZ DE DJ DK DM DO DZ EC EE EG EH ER ES ET FI FJ FK FM FO FR GA GB GD GE GF GG GH GI GL GM GN GP GQ GR GS GT GU GW GY HK HM HN HR HT HU ID IE IL IM IN IO IQ IR IS IT JE JM JO JP KE KG KH KI KM KN KP KR KW KY KZ LA LB LC LI LK LR LS LT LU LV LY MA MC MD ME MF MG MH MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ NA NC NE NF NG NI NL NO NP NR NU NZ OM PA PE PF PG PH PK PL PM PN PR PS PT PW PY QA RE RO RS RU RW SA SB SC SD SE SG SH SI SJ SK SL SM SN SO SR SS ST SV SX SY SZ TC TD TF TG TH TJ TK TL TM TN TO TR TT TV TW TZ UA UG UM US UY UZ VA VC VE VG VI VN VU WF WS YE YT ZA ZM ZW".split()
)
CURRENCIES = frozenset(
    "AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BGN BHD BIF BMD BND BOB BRL BSD BTN BWP BYN BZD CAD CDF CHF CLP CNY COP CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD UYU UZS VED VES VND VUV WST XAF XCD XCG XOF XPF YER ZAR ZMW ZWG".split()
)
JS_SPACE = " \t\r\n\v\f\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def need(ok: bool, reason: str = "invalid_request") -> None:
    if not ok:
        raise ValueError(reason)


def obj(value: Any, keys: str) -> dict:
    need(type(value) is dict and set(value) == set(keys.split()))
    return value


def text(value: Any, maximum: int, minimum: int = 0) -> str:
    need(type(value) is str and not any(0xD800 <= ord(c) <= 0xDFFF for c in value))
    result = unicodedata.normalize("NFC", value).strip(JS_SPACE)
    need(minimum <= len(result) <= maximum)
    return result


def revision(value: Any) -> int:
    need(
        type(value) in (int, float)
        and math.isfinite(value)
        and int(value) == value
        and 0 <= value <= MAX_SAFE
    )
    return int(value)


def enum(value: Any, choices: str) -> str:
    need(type(value) is str and value in choices.split())
    return value


def array(value: Any, maximum: int, minimum: int = 0) -> list:
    need(type(value) is list and minimum <= len(value) <= maximum)
    return value


def distinct(values: list) -> list:
    need(len(values) == len(set(values)))
    return values


def text_list(value: Any, maximum: int, length: int, minimum: int = 0) -> list:
    result = [text(v, length, 1) for v in array(value, maximum, minimum)]
    distinct([re.sub(f"[{JS_SPACE}]+", " ", v.lower()) for v in result])
    return result


def place(value: Any, display: bool = False) -> dict:
    obj(value, "city countryCode" + (" displayCountry" if display else ""))
    need(value["countryCode"] in COUNTRIES)
    result = {"city": text(value["city"], 200, 1), "countryCode": value["countryCode"]}
    if display:
        result["displayCountry"] = text(value["displayCountry"], 200, 1)
    return result


def date(value: Any) -> Any:
    if value is None:
        return None
    obj(value, "value precision")
    precision = enum(value["precision"], "year month day")
    v = value["value"]
    pattern = {
        "year": r"[0-9]{4}",
        "month": r"[0-9]{4}-[0-9]{2}",
        "day": r"[0-9]{4}-[0-9]{2}-[0-9]{2}",
    }[precision]
    need(type(v) is str and re.fullmatch(pattern, v) is not None)
    parts = list(map(int, v.split("-")))
    need(parts[0] > 0)
    if len(parts) > 1:
        need(1 <= parts[1] <= 12)
    if len(parts) > 2:
        need(1 <= parts[2] <= calendar.monthrange(parts[0], parts[1])[1])
    return dict(value)


def rows(value: Any, kind: str, maximum: int) -> list:
    fields = {
        "employment": {"employer": (200, 1), "title": (200, 1), "description": (10000, 0)},
        "education": {"institution": (200, 1), "qualification": (300, 0), "description": (5000, 0)},
        "projects": {"name": (200, 1), "description": (10000, 0)},
    }[kind]
    result = []
    for row in array(value, maximum):
        obj(row, "id start end current " + " ".join(fields))
        need(
            type(row["id"]) is str
            and UUID.fullmatch(row["id"]) is not None
            and type(row["current"]) is bool
        )
        start, end = date(row["start"]), date(row["end"])
        need(not row["current"] or end is None)
        if start and end:
            need((end["value"] + "-99-99")[:10] >= (start["value"] + "-01-01")[:10])
        result.append(
            {
                "id": row["id"],
                "start": start,
                "end": end,
                "current": row["current"],
                **{k: text(row[k], *limits) for k, limits in fields.items()},
            }
        )
    distinct([r["id"] for r in result])
    return result


def photo_url(value: Any) -> str:
    need(
        type(value) is str
        and len(value) <= 2048
        and re.fullmatch(r"https://media\.licdn\.com/[A-Za-z0-9_~.%/?=&+,:;@!$()*-]+", value)
        is not None
    )
    return value


def profile(value: Any) -> dict:
    need(type(value) is dict)
    obj(
        value,
        "displayName headline linkedinUrl currentLocation skills employment education projects"
        + (" photoUrl" if "photoUrl" in value else ""),
    )
    linkedin = value["linkedinUrl"]
    if linkedin is not None:
        need(
            type(linkedin) is str
            and re.fullmatch(r"https://(www\.)?linkedin\.com/in/[A-Za-z0-9_%~-]+/?", linkedin)
            is not None
        )
        u = urlsplit(linkedin)
        need(
            u.scheme == "https"
            and u.netloc in ("linkedin.com", "www.linkedin.com")
            and not u.query
            and not u.fragment
            and re.fullmatch(r"/in/[A-Za-z0-9_%~-]+/?", u.path) is not None
        )
        linkedin = linkedin.replace("https://www.linkedin.com/", "https://linkedin.com/").rstrip(
            "/"
        )
    return {
        "displayName": text(value["displayName"], 200),
        "headline": text(value["headline"], 300),
        "linkedinUrl": linkedin,
        **(
            {"photoUrl": None if value["photoUrl"] is None else photo_url(value["photoUrl"])}
            if "photoUrl" in value
            else {}
        ),
        "currentLocation": None
        if value["currentLocation"] is None
        else place(value["currentLocation"], True),
        "skills": text_list(value["skills"], 200, 100),
        **{
            k: rows(value[k], k, n)
            for k, n in (("employment", 200), ("education", 50), ("projects", 50))
        },
    }


def money(value: Any) -> str:
    need(
        type(value) is str
        and re.fullmatch(r"(0|[1-9][0-9]{0,11})(\.[0-9]{1,2})?", value) is not None
    )
    parts = value.split(".")
    return parts[0] + "." + (parts[1] if len(parts) > 1 else "").ljust(2, "0")


def compensation(value: Any) -> dict:
    need(type(value) is dict)
    if value.get("mode") == "open_to_discussion":
        return obj(value, "mode")
    obj(value, "mode currency period component targetMin targetMax hardMinimum")
    need(value["mode"] == "target" and value["currency"] in CURRENCIES)
    result = {
        "mode": "target",
        "currency": value["currency"],
        "period": enum(value["period"], "annual monthly hourly"),
        "component": enum(value["component"], "base total ctc"),
        "targetMin": money(value["targetMin"]),
        "targetMax": money(value["targetMax"]),
        "hardMinimum": None if value["hardMinimum"] is None else money(value["hardMinimum"]),
    }

    def cents(s):
        return int(s.replace(".", ""))

    need(cents(result["targetMin"]) <= cents(result["targetMax"]))
    need(
        result["hardMinimum"] is None or cents(result["hardMinimum"]) <= cents(result["targetMax"])
    )
    return result


def preferences(value: Any) -> dict:
    obj(
        value,
        "desiredRoles preferredSeniority workArrangements locations remoteRegions relocation employmentTypes availability compensation strengths",
    )
    result = {
        "desiredRoles": text_list(value["desiredRoles"], 10, 200, 1),
        "preferredSeniority": text_list(value["preferredSeniority"], 8, 100),
    }
    for key, choices in (
        ("workArrangements", "remote hybrid onsite"),
        ("employmentTypes", "permanent contract part_time internship"),
    ):
        result[key] = distinct(
            [enum(v, choices) for v in array(value[key], len(choices.split()), 1)]
        )
    locations = value["locations"]
    need(type(locations) is dict)
    if locations.get("mode") == "anywhere":
        result["locations"] = obj(locations, "mode")
    else:
        obj(locations, "mode places")
        need(locations["mode"] == "specific")
        result["locations"] = {
            "mode": "specific",
            "places": [place(p) for p in array(locations["places"], 20, 1)],
        }
    regions = value["remoteRegions"]
    if regions is None:
        need("remote" not in result["workArrangements"])
    else:
        need(type(regions) is dict)
        if regions.get("mode") == "worldwide":
            obj(regions, "mode")
        else:
            obj(regions, "mode countries")
            need(regions["mode"] == "countries")
            distinct(array(regions["countries"], 249, 1))
            need(all(type(v) is str and v in COUNTRIES for v in regions["countries"]))
    result["remoteRegions"] = regions
    relocation = obj(value["relocation"], "mode destinations")
    result["relocation"] = {
        "mode": enum(relocation["mode"], "yes no discuss"),
        "destinations": [place(p) for p in array(relocation["destinations"], 20)],
    }
    available = value["availability"]
    need(type(available) is dict)
    mode = enum(available.get("mode"), "unknown immediate date notice_days")
    obj(available, "mode" + ({"date": " date", "notice_days": " days"}.get(mode, "")))
    if mode == "date":
        date({"value": available["date"], "precision": "day"})
    if mode == "notice_days":
        need(revision(available["days"]) <= 730)
        available = {"mode": mode, "days": int(available["days"])}
    result["availability"] = available
    result["compensation"] = compensation(value["compensation"])
    strengths = obj(
        value["strengths"], "roles location workArrangement employmentType compensation"
    )
    result["strengths"] = {k: enum(v, "firm flexible") for k, v in strengths.items()}
    return result


def _pairs(pairs: list) -> dict:
    result = {}
    for k, v in pairs:
        need(k not in result)
        result[k] = v
    return result


def _walk(value: Any, depth: int = 1) -> None:
    if type(value) in (dict, list):
        need(depth <= 12)
        if type(value) is dict:
            for k in value:
                text(k, 384 * 1024)
            values = value.values()
        else:
            values = value
        for v in values:
            _walk(v, depth + 1)
    elif type(value) is str:
        text(value, 384 * 1024)
    elif type(value) is float:
        need(math.isfinite(value))


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _parse_profile_command(raw: bytes) -> tuple[dict, str, str]:
    need(type(raw) is bytes and len(raw) <= 384 * 1024, "payload_too_large")
    value = json.loads(
        raw.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=lambda _: need(False)
    )
    _walk(value)
    obj(
        value,
        "schemaVersion idempotencyKey expectedDraftVersion expectedProfileVersion expectedPreferenceVersion action payload",
    )
    need(type(value["schemaVersion"]) in (int, float) and value["schemaVersion"] == 2)
    need(
        type(value["idempotencyKey"]) is str and UUID.fullmatch(value["idempotencyKey"]) is not None
    )
    result = {
        "schemaVersion": 2,
        "idempotencyKey": value["idempotencyKey"],
        "action": enum(value["action"], "save_draft save_preferences confirm"),
    }
    for k in ("expectedDraftVersion", "expectedProfileVersion", "expectedPreferenceVersion"):
        result[k] = revision(value[k])
    payload = value["payload"]
    if result["action"] == "save_draft":
        obj(payload, "profile")
        p = profile(payload["profile"])
        need(len(canonical(p).encode()) <= 256 * 1024, "payload_too_large")
        result["payload"] = {"profile": p}
    elif result["action"] == "save_preferences":
        obj(payload, "preferences")
        p = preferences(payload["preferences"])
        need(len(canonical(p).encode()) <= 32 * 1024, "payload_too_large")
        result["payload"] = {"preferences": p}
    else:
        obj(payload, "draftHash consentVersion")
        need(
            type(payload["draftHash"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", payload["draftHash"]) is not None
        )
        result["payload"] = {
            "draftHash": payload["draftHash"],
            "consentVersion": revision(payload["consentVersion"]),
        }
    wire = canonical(result)
    return result, wire, hashlib.sha256(wire.encode()).hexdigest()


def parse_profile_command(raw: bytes) -> tuple[dict, str, str]:
    try:
        return _parse_profile_command(raw)
    except (TypeError, KeyError, OverflowError, RecursionError, UnicodeError) as exc:
        raise ValueError("invalid_request") from exc
