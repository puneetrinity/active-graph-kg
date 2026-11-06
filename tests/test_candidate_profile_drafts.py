import json

import pytest

from activekg.candidate_profiles.profile_contracts import parse_profile_command


def draft():
    return {
        "schemaVersion": 2,
        "idempotencyKey": "10000000-0000-4000-8000-000000000001",
        "expectedDraftVersion": 0,
        "expectedProfileVersion": 0,
        "expectedPreferenceVersion": 0,
        "action": "save_draft",
        "payload": {
            "profile": {
                "displayName": "Synthetic",
                "headline": "",
                "linkedinUrl": None,
                "currentLocation": None,
                "skills": [],
                "employment": [],
                "education": [],
                "projects": [],
            }
        },
    }


def test_manual_draft_canonical():
    value, wire, digest = parse_profile_command(json.dumps(draft()).encode())
    assert value["payload"]["profile"]["linkedinUrl"] is None
    assert len(digest) == 64
    assert parse_profile_command(wire.encode())[2] == digest


def test_consent_v2_guard_detects_removed_subject_check(monkeypatch):
    from scripts import candidate_privacy_surface_guard as guard

    reference = next(
        r for r in guard.discover() if r.key == "activekg/candidate_profiles/consent_v2.py::receive"
    )
    source = guard._reference_source(reference)
    row = {"test_id": "privacy-candidate-consent-v2"}
    guard._validate_fence_anchor(reference, row)
    monkeypatch.setattr(
        guard,
        "_reference_source",
        lambda _: source.replace('subject != parsed["value"]["subjectId"]', "False"),
    )
    with pytest.raises(guard.GuardError):
        guard._validate_fence_anchor(reference, row)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}',
        b'{"a":1,"\\u0061":2}',
        b'{"a":"\\ud800"}',
        b'{"a":NaN}',
        b'{"a":1e999}',
        b"[" * 13 + b"0" + b"]" * 13,
    ],
)
def test_invalid_json(raw):
    with pytest.raises(ValueError):
        parse_profile_command(raw)


@pytest.mark.parametrize("key", ["userId", "subjectId", "bindingId", "provenance", "consent"])
def test_no_browser_authority(key):
    value = draft()
    value[key] = "foreign"
    with pytest.raises(ValueError):
        parse_profile_command(json.dumps(value).encode())


@pytest.mark.parametrize("value", [True, -1, 1.1, 9007199254740992, "1"])
def test_revision(value):
    request = draft()
    request["expectedDraftVersion"] = value
    with pytest.raises(ValueError):
        parse_profile_command(json.dumps(request).encode())


@pytest.mark.parametrize(
    "decision,code",
    [
        ("allow", None),
        ("block_global", None),
        ("block_all", 451),
        ("review", 503),
        ("unknown", 503),
        (None, 503),
    ],
)
def test_owner_privacy_is_not_network_permission(decision, code):
    from fastapi import HTTPException

    from activekg.candidate_profiles.profile_authority import check_owner_privacy

    class Cursor:
        def __init__(self):
            self.results = iter([{"global_candidate_id": "synthetic"}, {"decision": decision}])

        def execute(self, *args):
            pass

        def fetchone(self):
            return next(self.results)

    if code:
        with pytest.raises(HTTPException) as err:
            check_owner_privacy(Cursor(), "synthetic")
        assert err.value.status_code == code
    else:
        check_owner_privacy(Cursor(), "synthetic")


@pytest.mark.parametrize(
    "symbol,anchor",
    [
        ("save_profile", "require_owner(claims, write=True)"),
        ("read_profile", "require_owner(claims)"),
        ("execute_profile", "check_owner_privacy(cur, subject)"),
        ("check_owner_privacy", 'decision == "block_all"'),
    ],
)
def test_profile_privacy_guard_catches_missing_fence(monkeypatch, symbol, anchor):
    from scripts import candidate_privacy_surface_guard as guard

    references = guard.discover()
    reference = next(row for row in references if row.symbol == symbol)
    original = guard._reference_source(reference)
    assert anchor in original
    row = {"test_id": "privacy-candidate-profile-owner"}
    guard._validate_fence_anchor(reference, row)
    monkeypatch.setattr(
        guard, "_reference_source", lambda _: original.replace(anchor, "REMOVED_FENCE", 1)
    )
    with pytest.raises(guard.GuardError):
        guard._validate_fence_anchor(reference, row)
