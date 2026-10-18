"""Closed, text-free extraction diagnostics. Unknown metrics stay absent."""

from __future__ import annotations

from typing import Any

from activekg.candidate_index.contracts import IndexContractError, canonical_json

EVENTS = frozenset(
    {
        "reserved",
        "input_checked",
        "dispatch_attempted",
        "response_received",
        "validation_finished",
        "settled",
    }
)
REASONS = frozenset(
    {
        "completed",
        "incomplete",
        "output_truncated",
        "low_confidence",
        "unsupported_format",
        "chunk_overflow",
        "provider_timeout",
        "provider_unavailable",
        "rate_limited",
        "dispatch_exhausted",
        "policy_mismatch",
        "privacy_restricted",
        "privacy_review",
        "privacy_unavailable",
        "source_missing",
        "consent_changed",
        "superseded",
        "not_validated",
    }
)
ENUMS = {
    "validation_reason": frozenset(
        {
            "not_validated",
            "accepted",
            "location_format",
            "list_format",
            "list_limit",
            "years_format_or_limit",
            "skill_length",
            "expanded_list_limit",
            "schema_invalid",
            "namespace_bytes",
        }
    ),
    "reason": REASONS,
    "finish_reason": frozenset(
        {"stop", "length", "content_filter", "tool_calls", "unlisted", "unavailable"}
    ),
    "model_slot": frozenset({"primary", "backup"}),
    "reasoning_effort": frozenset({"medium", "low"}),
}
BOUNDS = {
    "elapsed_ms": 45000,
    "input_characters": 2097152,
    "completion_tokens": 4096,
    "skills_kept": 100,
    "skills_dropped": 1000,
    "duplicates_removed": 1000,
}


def diagnostic_payload(values: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(values, dict) or set(values) - (set(ENUMS) | set(BOUNDS)):
        raise IndexContractError("candidate_index_diagnostic_invalid")
    result: dict[str, Any] = {"version": 1}
    for key, value in values.items():
        if value is None:
            continue
        if key in ENUMS:
            if not isinstance(value, str) or value not in ENUMS[key]:
                raise IndexContractError("candidate_index_diagnostic_invalid")
        elif type(value) is not int or not 0 <= value <= BOUNDS[key]:
            raise IndexContractError("candidate_index_diagnostic_invalid")
        result[key] = value
    if len(canonical_json(result).encode()) > 2048:
        raise IndexContractError("candidate_index_diagnostic_invalid")
    return result
