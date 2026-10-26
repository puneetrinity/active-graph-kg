"""Small, byte-exact foundation protocol shared with Flow."""

import hashlib
import json
import re
from typing import Any

UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
FIELDS = ("schemaVersion", "commandId", "subjectId", "bindingId", "sequence", "action")
TABLES = ("candidate_owned_subjects", "candidate_owned_command_receipts")
RUNTIME_FUNCTIONS = (
    "candidate_owned_ensure(uuid,uuid,uuid,bigint,text)",
    "candidate_owned_status(uuid,uuid)",
)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("invalid_request")
        result[key] = value
    return result


def parse_command(raw: bytes) -> tuple[dict[str, Any], str]:
    if len(raw) > 4096:
        raise ValueError("invalid_request")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    if not isinstance(value, dict) or tuple(value) != FIELDS:
        raise ValueError("invalid_request")
    if type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1:
        raise ValueError("invalid_request")
    if any(
        not isinstance(value[k], str) or not UUID.fullmatch(value[k])
        for k in ("commandId", "subjectId", "bindingId")
    ):
        raise ValueError("invalid_request")
    if type(value["sequence"]) is not int or not 1 <= value["sequence"] <= 9007199254740991:
        raise ValueError("invalid_request")
    if value["action"] != "ensure_subject":
        raise ValueError("invalid_request")
    if json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8") != raw:
        raise ValueError("invalid_request")
    return value, hashlib.sha256(raw).hexdigest()


def receipt(command: dict[str, Any], digest: str) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "commandId": command["commandId"],
        "subjectId": command["subjectId"],
        "bindingId": command["bindingId"],
        "sequence": command["sequence"],
        "wireSha256": digest,
        "outcome": "subject_ready",
        "profileVersion": 0,
        "preferenceVersion": 0,
        "publicationActive": False,
    }
