"""Opt-in resume extraction v2; no IO other than reading its own policy source.

Literal support is NOT evidence of proficiency, recency or affirmative context.
In particular, 'Python not required' still contains a literal Python mention.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path

from pydantic import ConfigDict, Field, create_model

from activekg.candidate_index.contracts import canonical_json, sha256
from activekg.candidate_index.providers import WorkRefused
from activekg.extraction.prompt import build_extraction_prompt
from activekg.extraction.schema import ExtractionResult, LocationInfo

OUTPUT_TOKENS = 4096
RETRY_POLICY = "two-reservations:medium:first;low:primary-on-cutoff;medium:fallback-otherwise"
LIST_LIMITS = {
    "skills_raw": 100,
    "skills_normalized": 100,
    "primary_skills": 12,
    "primary_titles": 3,
    "recent_job_titles": 3,
    "certifications": 10,
    "domains": 10,
    "functions": 10,
    "industries": 10,
}
YEARS_LIMIT = 10
SKILL_CHARACTERS = 120

# Copy types, not legacy validators: those silently truncate. Every list has
# one bound shared by the prompt, the strict precheck and this model.
_fields = {}
for _name, _field in ExtractionResult.model_fields.items():
    _definition = deepcopy(_field)
    if _name in LIST_LIMITS:
        _definition = Field(
            default=None if _name in {"industries", "certifications"} else [],
            max_length=LIST_LIMITS[_name],
        )
    _fields[_name] = (_field.annotation, _definition)
ResumeResult = create_model("ManagedResumeV2", __config__=ConfigDict(extra="forbid"), **_fields)


def prompt(text: str) -> tuple[str, str]:
    system, user = build_extraction_prompt(text)
    limits = ", ".join(f"{name}: {limit}" for name, limit in LIST_LIMITS.items())
    return system + (
        "\n\nLIST LIMITS (ceilings, never targets): "
        + limits
        + f"; years_by_skill: {YEARS_LIMIT} entries. Skill names: at most {SKILL_CHARACTERS} characters."
        + " Do not pad lists to their limits."
        + " List one skill per entry. Split grouped lists into separate skills, including the group head."
        + " Keep the resume's own names; do not prefix services with a vendor unless the resume does."
    ), user


def policy_fields() -> dict[str, str]:
    # Includes implementation, limits and output allowance; no unversioned
    # environment override can alter v2 extraction under an existing hash.
    descriptor = {
        "source": sha256(Path(__file__).read_bytes()),
        "output_tokens": OUTPUT_TOKENS,
        "retry_policy": RETRY_POLICY,
        "limits": LIST_LIMITS,
        "years_limit": YEARS_LIMIT,
        "skill_characters": SKILL_CHARACTERS,
    }
    return {
        "extraction_adapter_version": "candidate-index-extraction:v2:"
        + sha256(canonical_json(descriptor)),
        "extraction_schema_sha256": sha256(canonical_json(ResumeResult.model_json_schema())),
        "extraction_prompt_sha256": sha256(canonical_json(prompt(""))),
    }


def normalized(value: str) -> str:
    # Formatting only. Do not strip punctuation or invent equivalent names.
    return " ".join(value.split()).casefold()


def supported(skill: str, text: str) -> bool:
    needle, haystack = normalized(skill), normalized(text)
    # Token edges, not \b: C++, C#, .NET and Node.js retain their symbols.
    # Slash/comma/parentheses separate skills; +/#/. within names do not.
    for match in re.finditer(re.escape(needle), haystack):
        before = haystack[match.start() - 1] if match.start() else ""
        after = haystack[match.end()] if match.end() < len(haystack) else ""
        if before and (before.isalnum() or before in "_+#."):
            continue
        if after and (after.isalnum() or after in "_+#"):
            continue
        if after == "." and match.end() + 1 < len(haystack) and haystack[match.end() + 1].isalnum():
            continue
        return True
    return False


def refuse(diagnostics: dict, reason: str) -> None:
    diagnostics["validation_reason"] = reason
    raise WorkRefused("incomplete")


def _items(body: str) -> list[str]:
    """Split explicit comma/semicolon lists, never slash names or qualifiers."""
    depth, start, result = 0, 0, []
    for index, char in enumerate(body):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth < 0 or depth > 4:
            return []
        if char in ",;" and depth == 0:
            result.append(body[start:index].strip())
            start = index + 1
    result.append(body[start:].strip())
    return result if depth == 0 and all(result) else []


def _group(value: str) -> tuple[str, str] | None:
    value = value.strip()
    if "(" not in value or not value.endswith(")"):
        return None
    head, body = value.split("(", 1)
    head, body = head.strip(), body[:-1]
    if not head or any(c in head for c in ",;():") or not _items(body):
        return None
    return head, body


def _source_groups(head: str, text: str) -> list[str]:
    """Only the named parent's balanced parentheses or bounded colon list.

    Never consume a neighbouring provider's list. No provider dictionaries,
    inferred aliases, general substring equivalence or punctuation deletion.
    """
    pattern = r"\s+".join(re.escape(part) for part in head.casefold().split())
    bodies = []
    folded = text.casefold()
    for match in re.finditer(pattern + r"\s*([(:])", folded):
        before = folded[match.start() - 1] if match.start() else ""
        if before and (before.isalnum() or before in "_+#.-"):
            continue
        start = match.end()
        if match.group(1) == ":":
            # A subsequent label, semicolon or newline ends this group.
            body = re.split(r"[;\n\r:]", folded[start:], maxsplit=1)[0]
            if _items(body):
                bodies.append(body)
            continue
        depth = 1
        for index in range(start, len(folded)):
            if folded[index] == "(":
                depth += 1
            elif folded[index] == ")":
                depth -= 1
            if depth > 4:
                break
            if depth == 0:
                body = folded[start:index]
                if _items(body):
                    bodies.append(body)
                break
    return bodies


def grounded_skills(name: str, text: str, depth: int = 0) -> tuple[list[str], int]:
    """Returns grounded names and unsupported components; no silent clipping.

    A bare prefix is never inferred from an arbitrary multiword skill. Prefix
    expansion requires an explicit source group with the same head and member.
    Single parenthetical qualifiers remain intact, e.g. Python (programming language).
    """
    if depth > 4:
        return [], 1
    group = _group(name)
    if group and len(_items(group[1])) > 1:
        head, body = group
        sources = _source_groups(head, text)
        if not sources:
            return [], 1
        # Preserve the source association for each child. A child's occurrence
        # elsewhere (under another vendor) cannot validate this parent group.
        result, removed = [head], 0
        for child in _items(body):
            candidates = [grounded_skills(child, source, depth + 1) for source in sources]
            accepted = next((item for item in candidates if item[0] and item[1] == 0), None)
            if accepted is None:
                removed += 1
            else:
                result.extend(accepted[0])
        return result, removed
    # Prefer the longest explicit head (e.g. Google Cloud over Google).
    words = name.strip().split()
    for cut in range(len(words) - 1, 0, -1):
        head, member = " ".join(words[:cut]), " ".join(words[cut:])
        sources = _source_groups(head, text)
        if any(normalized(member) == normalized(item) for body in sources for item in _items(body)):
            return [name.strip(), head, member], 0
    return ([name.strip()], 0) if supported(name, text) else ([], 1)


def validate(raw: dict, text: str, diagnostics: dict) -> dict:
    from pydantic import ValidationError

    location = raw.get("location")
    if location is not None and (
        not isinstance(location, dict) or set(location) - set(LocationInfo.model_fields)
    ):
        refuse(diagnostics, "location_format")
    # Check input lengths BEFORE deduplication or deriving normalized names.
    for name, limit in LIST_LIMITS.items():
        values = raw.get(name)
        if values is None and name in {"industries", "certifications"}:
            continue
        if name not in raw:
            continue
        if not isinstance(values, list) or any(
            not isinstance(v, str) or not v.strip() for v in values
        ):
            refuse(diagnostics, "list_format")
        if len(values) > limit:
            refuse(diagnostics, "list_limit")
    years = raw.get("years_by_skill")
    if years is not None and (
        not isinstance(years, dict)
        or len(years) > YEARS_LIMIT
        or any(type(v) not in (int, float) or not 0 <= v <= 100 for v in years.values())
    ):
        refuse(diagnostics, "years_format_or_limit")
    result = deepcopy(raw)
    diagnostics.update(
        raw_skills=len(raw.get("skills_raw", [])),
        unsupported_skills=0,
        duplicate_skills=0,
        retained_skills=0,
        expanded_skill_entries=0,
        unsupported_group_components=0,
    )
    kept, seen = [], set()
    for name in raw.get("skills_raw", []):
        if len(name) > SKILL_CHARACTERS and _group(name) is None:
            refuse(diagnostics, "skill_length")
        expanded, unsupported = grounded_skills(name, text)
        diagnostics["unsupported_group_components"] += unsupported if expanded else 0
        if not expanded:
            diagnostics["unsupported_skills"] += 1
            continue
        diagnostics["expanded_skill_entries"] += max(0, len(expanded) - 1)
        for entry in expanded:
            key = normalized(entry)
            if len(entry) > SKILL_CHARACTERS or len(key) > SKILL_CHARACTERS:
                refuse(diagnostics, "skill_length")
            if key in seen:
                diagnostics["duplicate_skills"] += 1
                continue
            seen.add(key)
            kept.append(entry.strip())
            if len(kept) > LIST_LIMITS["skills_raw"]:
                refuse(diagnostics, "expanded_list_limit")
    result["skills_raw"] = kept
    result["skills_normalized"] = [normalized(name) for name in kept]
    # Neither the summary nor years-by-skill can reintroduce ungrounded skills.
    result["primary_skills"] = list(
        dict.fromkeys(
            name.strip() for name in raw.get("primary_skills", []) if normalized(name) in seen
        )
    )
    if years is not None:
        result["years_by_skill"] = {k: v for k, v in years.items() if normalized(k) in seen}
    diagnostics["summary_skills_removed"] = len(raw.get("primary_skills", [])) - len(
        result["primary_skills"]
    )
    diagnostics["years_entries_removed"] = len(years or {}) - len(
        result.get("years_by_skill") or {}
    )
    diagnostics["retained_skills"] = len(kept)
    try:
        parsed = ResumeResult.model_validate(result, strict=True).model_dump(mode="json")
    except (ValidationError, ValueError, TypeError):
        refuse(diagnostics, "schema_invalid")
    # PostgreSQL jsonb text includes spaces. Conservative UTF-8 check of the
    # complete namespace, not an ASCII-only per-skill multiplication.
    if len(json.dumps(parsed, ensure_ascii=False, allow_nan=False).encode()) > 65536:
        refuse(diagnostics, "namespace_bytes")
    diagnostics["validation_reason"] = "accepted"
    return parsed
