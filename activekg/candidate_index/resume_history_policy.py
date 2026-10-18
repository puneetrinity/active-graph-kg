"""Opt-in dated employment, layered on the frozen skill-reading policy.

Keep date precision and source excerpts. A stated total never creates a job interval.
Quotes are retained only inside private extraction evidence, not the search response.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from activekg.candidate_index import resume_policy as skills
from activekg.candidate_index.contracts import canonical_json, sha256
from activekg.candidate_index.providers import WorkRefused

OUTPUT_TOKENS = skills.OUTPUT_TOKENS
MAX_EMPLOYMENT = 20
MONTHS = (
    "january february march april may june july august september october november december"
).split()


def quoted_date(quote: str) -> str | None:
    """Only unambiguous written dates; never infer precision or a date order."""
    text = quote.strip().lower().rstrip(".")
    if re.fullmatch(r"(?:19|20|21)\d{2}(?:-\d{2}(?:-\d{2})?)?", text):
        return text
    match = re.fullmatch(r"(\d{1,2})/(\d{4})", text)
    if match:
        return f"{int(match[2]):04}-{int(match[1]):02}"
    for month, name in enumerate(MONTHS, 1):
        token = "(?:" + name + "|" + name[:3] + r"\.?)"
        for pattern, _day_first in [
            (token + r"\s+(\d{4})", False),
            (token + r"\s+(\d{1,2}),?\s+(\d{4})", False),
            (r"(\d{1,2})\s+" + token + r"\s+(\d{4})", True),
        ]:
            match = re.fullmatch(pattern, text)
            if match:
                values = match.groups()
                if len(values) == 1:
                    return f"{int(values[0]):04}-{month:02}"
                return f"{int(values[1]):04}-{month:02}-{int(values[0]):02}"
    return None


def prompt(text: str) -> tuple[str, str]:
    system, user = skills.prompt(text)
    return system + (
        "\nAlso return employment_history (at most 20 jobs, never pad): "
        "each item has title, employment_type, start, end, ongoing, start_quote, end_quote. "
        "Use only employment actually stated in the resume, not projects or education. "
        "Dates are YYYY, YYYY-MM or YYYY-MM-DD, preserving the precision written. "
        "Copy each date expression exactly into its quote field. Never invent a month or day. "
        "Use null for missing dates and their quotes; ongoing is true only for explicitly current employment "
        "(quote the present/current wording in end_quote, and use end=null). "
        "Keep internship labels. title/employment_type may be null when absent. "
        "Do not infer dates from years of experience, skills, age or other jobs. "
        "List every stated job even when its dates are missing; do not hide incomplete history."
    ), user


def policy_fields() -> dict[str, str]:
    descriptor = {
        "parent": skills.policy_fields(),
        "source": sha256(Path(__file__).read_bytes()),
        "output_tokens": OUTPUT_TOKENS,
        "max_employment": MAX_EMPLOYMENT,
    }
    return {
        "extraction_adapter_version": "candidate-index-extraction:v3:"
        + sha256(canonical_json(descriptor)),
        "extraction_schema_sha256": sha256(canonical_json(descriptor)),
        "extraction_prompt_sha256": sha256(canonical_json(prompt(""))),
    }


def _date(value: object, quote: object, text: str) -> str | None:
    if value is None:
        if quote is not None:
            raise WorkRefused("incomplete")
        return None
    if not isinstance(value, str) or not re.fullmatch(
        r"(?:19|20|21)\d{2}(?:-\d{2}(?:-\d{2})?)?", value
    ):
        raise WorkRefused("incomplete")
    parts = [int(part) for part in value.split("-")]
    try:
        date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
    except ValueError:
        raise WorkRefused("incomplete") from None
    if not isinstance(quote, str) or not quote.strip() or len(quote) > 120 or quote not in text:
        raise WorkRefused("incomplete")
    if quoted_date(quote) != value:
        raise WorkRefused("incomplete")
    return value


def validate(raw: dict, text: str, diagnostics: dict) -> dict:
    if not isinstance(raw, dict) or not isinstance(raw.get("employment_history"), list):
        raise WorkRefused("incomplete")
    history = raw["employment_history"]
    if len(history) > MAX_EMPLOYMENT:
        skills.refuse(diagnostics, "list_limit")
    parsed = skills.validate(
        {key: value for key, value in raw.items() if key != "employment_history"}, text, diagnostics
    )
    result = []
    for role in history:
        keys = {"title", "employment_type", "start", "end", "ongoing", "start_quote", "end_quote"}
        if not isinstance(role, dict) or set(role) != keys or type(role["ongoing"]) is not bool:
            raise WorkRefused("incomplete")
        for field in ("title", "employment_type"):
            value = role[field]
            if value is not None and (
                not isinstance(value, str)
                or not value.strip()
                or len(value) > (200 if field == "title" else 100)
                or value not in text
                or re.search(r"[\x00-\x1f\x7f@]|https?://", value)
            ):
                raise WorkRefused("incomplete")
        start = _date(role["start"], role["start_quote"], text)
        if role["ongoing"]:
            quote = role["end_quote"]
            if (
                role["end"] is not None
                or not isinstance(quote, str)
                or quote not in text
                or not re.fullmatch(
                    r"(?i)(?:present|current|currently|ongoing|till date|to date|now)",
                    quote.strip(),
                )
            ):
                raise WorkRefused("incomplete")
            end = None
        else:
            end = _date(role["end"], role["end_quote"], text)
        if start and end and start[: min(len(start), len(end))] > end[: min(len(start), len(end))]:
            raise WorkRefused("incomplete")
        result.append({**role, "start": start, "end": end})
    parsed["employment_history"] = result
    if len(json.dumps(parsed, ensure_ascii=False).encode()) > 65536:
        skills.refuse(diagnostics, "namespace_bytes")
    return parsed
