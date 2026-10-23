"""Synthetic dated-employment extraction; no network or production data."""

from dataclasses import replace

import pytest

from activekg.candidate_index import resume_history_policy as history
from activekg.candidate_index import resume_policy
from activekg.candidate_index.processing import resume_extraction
from activekg.candidate_index.providers import WorkRefused
from tests.test_candidate_index import POLICY

TEXT = "Python. Backend Engineer, 2019-03 to 2024-06. Intern, 2018 to 2019. Current role: 2024 to Present."


def role(**changes):
    return {
        "title": "Backend Engineer",
        "employment_type": None,
        "start": "2019-03",
        "end": "2024-06",
        "ongoing": False,
        "start_quote": "2019-03",
        "end_quote": "2024-06",
        **changes,
    }


def raw(roles=None):
    return {
        "skills_raw": ["Python"],
        "skills_normalized": [],
        "confidence": 0.9,
        "employment_history": [role()] if roles is None else roles,
    }


def test_history_uses_separate_version_and_preserves_the_v2_policy():
    assert history.policy_fields()["extraction_adapter_version"].startswith(
        "candidate-index-extraction:v3:"
    )
    assert resume_policy.policy_fields()["extraction_adapter_version"].startswith(
        "candidate-index-extraction:v2:"
    )
    assert history.OUTPUT_TOKENS == resume_policy.OUTPUT_TOKENS == 4096


def test_dated_history_does_not_enter_vector_text_or_public_evidence():
    result = resume_extraction(
        raw(), TEXT, replace(POLICY, **history.policy_fields()), "primary", {}
    )
    assert result.namespace["employment_history"] == [role()]
    assert "2019" not in result.professional_text
    assert all(value["field"] != "employment_history" for value in result.evidence)


def test_missing_dates_are_preserved_not_dropped_or_replaced_by_total_years():
    unknown = role(start=None, end=None, start_quote=None, end_quote=None)
    result = history.validate({**raw([unknown]), "total_years_experience": 8}, TEXT, {})
    assert result["employment_history"] == [unknown]


def test_year_precision_and_explicit_current_work_are_preserved():
    current = role(
        title=None, start="2024", start_quote="2024", end=None, ongoing=True, end_quote="Present"
    )
    intern = role(title="Intern", start="2018", start_quote="2018", end="2019", end_quote="2019")
    assert history.validate(raw([intern, current]), TEXT, {})["employment_history"] == [
        intern,
        current,
    ]


UNKNOWN_START = {"start": None, "start_quote": None}
UNKNOWN_END = {"end": None, "end_quote": None}


@pytest.mark.parametrize(
    "changes,published",
    [
        ({"start": "2019-13"}, UNKNOWN_START),
        ({"start": "2019-02-30"}, UNKNOWN_START),
        # Quote not in the text, but the value is the resume's own wording: kept from the value.
        ({"start_quote": "2019-04"}, {}),
        # Value contradicts its verbatim quote: unknown, never pick one.
        ({"start": "2020-03"}, UNKNOWN_START),
        # Value adds precision the quote does not have: unknown, never guess a month.
        ({"start": "2019-01", "start_quote": "2019"}, UNKNOWN_START),
        ({"title": "Invented title"}, {"title": None}),
        # "Current" claimed without present/current wording: not current; the dated end stays.
        ({"ongoing": True}, {"ongoing": False}),
        ({"ongoing": "true"}, {"ongoing": False}),
        # Unknown keys are dropped, never published.
        ({"extra": "unbounded"}, {}),
        # Value missing but the verbatim quote reads unambiguously: kept from the quote.
        ({"start": None}, {}),
        ({"start": "2019-04"}, UNKNOWN_START),
        ({"end": "2024-05"}, UNKNOWN_END),
        ({"end": None, "ongoing": True, "end_quote": "2024"}, {"end": "2024", "end_quote": "2024"}),
    ],
)
def test_unsupported_history_is_unknown_for_that_job_only(changes, published):
    other = role(title="Intern", start="2018", start_quote="2018", end="2019", end_quote="2019")
    result = history.validate(raw([role(**changes), other]), TEXT, {})
    assert result["employment_history"] == [{**role(), **published}, other]
    assert result["skills_raw"] == ["Python"]


def test_model_date_wording_is_read_only_from_verbatim_text():
    text = "Backend Engineer. Mar 2019 - Jun 2024. Python."
    written = role(start="Mar 2019", start_quote="Mar 2019", end="2024-06", end_quote="Jun 2024")
    assert history.validate(raw([written]), text, {})["employment_history"] == [
        role(start="2019-03", start_quote="Mar 2019", end="2024-06", end_quote="Jun 2024")
    ]


def test_impossible_order_makes_both_dates_unknown():
    backwards = role(start="2024-06", start_quote="2024-06", end="2019-03", end_quote="2019-03")
    assert history.validate(raw([backwards]), TEXT, {})["employment_history"] == [
        role(**UNKNOWN_START, **UNKNOWN_END)
    ]


def test_jobs_with_no_supported_evidence_are_not_published():
    empty = role(title="Invented", start="1999", start_quote="1999", end=None, end_quote=None)
    malformed = "Backend Engineer 2019"
    assert history.validate(raw([empty, malformed, role()]), TEXT, {})["employment_history"] == [
        role()
    ]


def test_history_is_part_of_the_required_json_schema():
    system, _ = history.prompt("resume")
    schema = system.split("EXTRACTION RULES")[0]
    assert '"employment_history"' in schema and schema.index('"employment_history"') < schema.index(
        '"confidence"'
    )
    assert "employment_history is REQUIRED" in system


def test_missing_history_key_is_still_refused():
    with pytest.raises(WorkRefused):
        history.validate({k: v for k, v in raw().items() if k != "employment_history"}, TEXT, {})


def test_list_overflow_is_recorded_and_never_truncated():
    diagnostics = {}
    with pytest.raises(WorkRefused):
        history.validate(raw([role()] * 21), TEXT, diagnostics)
    assert diagnostics["validation_reason"] == "list_limit"


def test_v3_cannot_be_run_under_v2_policy():
    with pytest.raises(WorkRefused):
        resume_extraction(
            raw(), TEXT, replace(POLICY, **resume_policy.policy_fields()), "primary", {}
        )


@pytest.mark.parametrize(
    "quote,expected",
    [
        ("Mar 2019", "2019-03"),
        ("March 4, 2019", "2019-03-04"),
        ("4 March 2019", "2019-03-04"),
        ("03/2019", "2019-03"),
        ("2019", "2019"),
        ("04/03/2019", None),
        ("2019-03", "2019-03"),
        ("Q1 2019", None),
        ("Mar '19", None),
    ],
)
def test_quote_precision_is_not_guessed(quote, expected):
    assert history.quoted_date(quote) == expected
