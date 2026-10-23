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


SCHEMA_ANCHOR = '  "confidence": 0.XX\n}'
HISTORY_SCHEMA = (
    '  "employment_history": [\n'
    '    {"title": "string|null", "employment_type": "string|null", "start": "YYYY|YYYY-MM|YYYY-MM-DD|null",\n'
    '     "end": "YYYY|YYYY-MM|YYYY-MM-DD|null", "ongoing": true|false,\n'
    '     "start_quote": "string|null", "end_quote": "string|null"}, ...\n'
    "  ],\n"
)


def prompt(text: str) -> tuple[str, str]:
    system, user = skills.prompt(text)
    if system.count(SCHEMA_ANCHOR) != 1:
        raise WorkRefused("policy_mismatch")
    system = system.replace(SCHEMA_ANCHOR, HISTORY_SCHEMA + SCHEMA_ANCHOR)
    return system + (
        "\nemployment_history is REQUIRED in the JSON object (use [] only when the resume states no job). "
        "List every job actually stated in the resume (at most 20, never pad), not projects or education. "
        "Dates are YYYY, YYYY-MM or YYYY-MM-DD, preserving the precision written. "
        "Copy each date expression exactly as written in the resume into its quote field (for example "
        '"Mar 2019" or "03/2019"); start/end are that same date in YYYY-MM form. Never invent a month or day. '
        "Use null for missing dates and their quotes; ongoing is true only for explicitly current employment "
        "(quote the present/current wording in end_quote, and use end=null). "
        "Copy title and employment_type exactly as written; keep internship labels; use null when absent. "
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


ISO_DATE = r"(?:19|20|21)\d{2}(?:-\d{2}(?:-\d{2})?)?"
ROLE_KEYS = ("title", "employment_type", "start", "end", "ongoing", "start_quote", "end_quote")
CURRENT = r"(?i)(?:present|current|currently|ongoing|till date|to date|now)"


def _calendar(value: str | None) -> str | None:
    if value is None or not re.fullmatch(ISO_DATE, value):
        return None
    parts = [int(part) for part in value.split("-")]
    try:
        date(parts[0], parts[1] if len(parts) > 1 else 1, parts[2] if len(parts) > 2 else 1)
    except ValueError:
        return None
    return value


def _verbatim(value: object, text: str) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 120 and value in text


def _date(value: object, quote: object, text: str) -> tuple[str | None, str | None]:
    """A date is kept only when resume text supports it; otherwise it is unknown (None).

    Grounding is never weakened: the kept date is the deterministic reading of a verbatim
    excerpt (the quote, or the value itself when the model copied the resume's own wording).
    A value that contradicts its quote, or adds precision the quote lacks, becomes unknown.
    """
    claimed = None
    if isinstance(value, str):
        claimed = (
            _calendar(value) if re.fullmatch(ISO_DATE, value) else _calendar(quoted_date(value))
        )
        if claimed is None:
            return None, None
    elif value is not None:
        return None, None
    if _verbatim(quote, text):
        read = _calendar(quoted_date(quote))
        if read is not None and (claimed is None or claimed == read):
            return read, quote
        if read is not None:
            return None, None  # value and quote disagree: unknown, never pick one
    if claimed is not None and _verbatim(value, text) and _calendar(quoted_date(value)) == claimed:
        return claimed, value
    return None, None


def _text_field(value: object, text: str, limit: int) -> str | None:
    if (
        isinstance(value, str)
        and value.strip()
        and len(value) <= limit
        and value in text
        and not re.search(r"[\x00-\x1f\x7f@]|https?://", value)
    ):
        return value
    return None


def validate(raw: dict, text: str, diagnostics: dict) -> dict:
    if not isinstance(raw, dict) or not isinstance(raw.get("employment_history"), list):
        raise WorkRefused("incomplete")
    history = raw["employment_history"]
    if len(history) > MAX_EMPLOYMENT:
        skills.refuse(diagnostics, "list_limit")
    parsed = skills.validate(
        {key: value for key, value in raw.items() if key != "employment_history"}, text, diagnostics
    )
    # Unsupported employment evidence is marked unknown for that job only; the rest of the
    # resume is kept. Nothing unsupported by the resume text is ever published.
    result = []
    for item in history:
        if not isinstance(item, dict):
            continue
        title = _text_field(item.get("title"), text, 200)
        employment_type = _text_field(item.get("employment_type"), text, 100)
        start, start_quote = _date(item.get("start"), item.get("start_quote"), text)
        ongoing = (
            item.get("ongoing") is True
            and item.get("end") is None
            and _verbatim(item.get("end_quote"), text)
            and re.fullmatch(CURRENT, item["end_quote"].strip()) is not None
        )
        if ongoing:
            end, end_quote = None, item["end_quote"]
        else:
            end, end_quote = _date(item.get("end"), item.get("end_quote"), text)
        if start and end and start[: min(len(start), len(end))] > end[: min(len(start), len(end))]:
            start = start_quote = end = end_quote = None  # impossible order: both unknown
        if title is None and start is None and end is None and not ongoing:
            continue  # nothing in this job is supported by the resume
        result.append(
            dict(
                zip(
                    ROLE_KEYS,
                    (title, employment_type, start, end, ongoing, start_quote, end_quote),
                    strict=True,
                )
            )
        )
    parsed["employment_history"] = result
    if len(json.dumps(parsed, ensure_ascii=False).encode()) > 65536:
        skills.refuse(diagnostics, "namespace_bytes")
    return parsed
