from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from activekg.rc_skills.config import rc_skills_enabled
from activekg.rc_skills.contracts import FinishRequest, Observation, ReadRequest, content_hash

# Same microsecond/Unicode vector as Discover research/fixtures/rc-skills-v1.json.
VECTOR = {
    "version": 1,
    "outcome": "success",
    "identityHash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "skills": ["AWS", "Café"],
    "upstreamAt": "2026-01-01T00:00:00.123456Z",
    "observedAt": "2026-10-08T00:00:00.000000Z",
    "validUntil": "2028-12-31T00:00:00.123456Z",
    "timestampSource": "check_last_update",
    "contentHash": "3d141131b41fcd6d5aed366e47d733225e37e8cdd8576ab6c823a5eb0d6d88fd",
}


def changed(**patch: object) -> dict:
    value = copy.deepcopy(VECTOR)
    value.update(patch)
    value.pop("contentHash")
    value["contentHash"] = content_hash(value)
    return value


def test_cross_system_hash_and_microseconds() -> None:
    parsed = Observation.model_validate(VECTOR)
    assert parsed.model_dump(by_alias=True) == VECTOR
    assert parsed.valid_until == "2028-12-31T00:00:00.123456Z"


def test_fetch_provenance_is_preserved_and_committed() -> None:
    value = changed(timestampSource="fetch_updated_at")
    assert Observation.model_validate(value).model_dump(by_alias=True) == value
    assert value["contentHash"] != VECTOR["contentHash"]
    with pytest.raises(ValidationError):
        Observation.model_validate({**VECTOR, "timestampSource": "fetch_updated_at"})


@pytest.mark.parametrize("extra", ["email", "linkedinUrl", "profile", "employer", "candidateId"])
def test_no_provider_payload_fields(extra: str) -> None:
    with pytest.raises(ValidationError):
        Observation.model_validate(changed(**{extra: "must never be stored"}))


@pytest.mark.parametrize(
    "patch",
    [
        {"skills": []},
        {"skills": [""]},
        {"skills": ["AWS\u200b"]},
        {"skills": ["x" * 501]},
        {"skills": ["AWS"] * 501},
        {"skills": ["x" * 400] * 200},
        {"upstreamAt": None},
        {"upstreamAt": "2027-10-08T00:00:00Z"},
        {"observedAt": "invalid"},
        {"observedAt": "2026-02-30T00:00:00Z"},
        {"upstreamAt": "2026-01-01T00:00:00.1234567Z"},
        {"validUntil": "2028-12-31T00:00:00.123000Z"},
        {"validUntil": "2026-10-08T00:00:00.000Z"},
        {"timestampSource": "fetch_time"},
    ],
)
def test_invalid_or_oversized_observation(patch: dict) -> None:
    with pytest.raises((ValidationError, ValueError)):
        Observation.model_validate(changed(**patch))


def test_hash_is_not_caller_supplied_authority() -> None:
    with pytest.raises(ValidationError):
        Observation.model_validate({**VECTOR, "skills": ["Kafka"]})


@pytest.mark.parametrize("outcome", ["no_skills", "not_found", "unusable_age"])
def test_no_data_is_explicit_not_positive(outcome: str) -> None:
    parsed = Observation.model_validate(changed(outcome=outcome, skills=[], validUntil=None))
    assert parsed.skills == []
    with pytest.raises(ValidationError):
        Observation.model_validate(changed(outcome=outcome))


def test_read_is_bounded_and_unique_case_insensitively() -> None:
    value = "abcdefab-1234-4123-8123-abcdefabcdef"
    assert ReadRequest.model_validate({"version": 1, "candidateIds": [value]}).candidate_ids == [
        value
    ]
    for ids in [[value, value.upper()], [value] * 101, ["invalid"]]:
        with pytest.raises(ValidationError):
            ReadRequest.model_validate({"version": 1, "candidateIds": ids})


def test_finish_is_minimal_and_does_not_admit_email() -> None:
    identity = "10000000-0000-4000-8000-000000000001"
    body = {
        "version": 1,
        "globalCandidateId": identity,
        "originRun": identity,
        "originAttempt": identity,
        "observation": VECTOR,
    }
    assert FinishRequest.model_validate(body).observation.content_hash == VECTOR["contentHash"]
    with pytest.raises(ValidationError):
        FinishRequest.model_validate({**body, "email": "private@fixture.invalid"})


def test_dark_gate_does_not_read_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RC_SKILLS_ENABLED", raising=False)
    monkeypatch.setenv("REVERSE_CONTACT_API_KEY", "not-read")
    assert rc_skills_enabled() is False
    monkeypatch.setenv("RC_SKILLS_ENABLED", "true")
    assert rc_skills_enabled() is True
    monkeypatch.setenv("RC_SKILLS_ENABLED", "perhaps")
    with pytest.raises(ValueError, match="RC_SKILLS_ENABLED"):
        rc_skills_enabled()
