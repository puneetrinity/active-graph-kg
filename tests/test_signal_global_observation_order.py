from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from activekg.api import global_memory

_GLOBAL_ID = "00000000-0000-0000-0000-000000000123"
_PROFILE = {
    "crustdata_person_id": 12345,
    "basic_profile": {
        "name": "Ordering Probe",
        "headline": "Senior Backend Engineer",
        "location": {"city": "Bengaluru", "country_code": "IN"},
    },
}


class RecordingCursor:
    def __init__(self, stored_observed_at: datetime):
        self.existing = {
            "id": _GLOBAL_ID,
            "public_profile_observed_at": stored_observed_at,
            "public_profile": _PROFILE,
            "public_crustdata_person_id": 12345,
        }
        self.statements: list[tuple[str, Any]] = []
        self.description: list[SimpleNamespace] = []
        self._next_row: tuple[Any, ...] | None = None

    def execute(self, sql: str, params: Any = None) -> None:
        normalized = " ".join(sql.split())
        self.statements.append((normalized, params))
        self._next_row = None
        if normalized.startswith("SELECT * FROM global_candidates"):
            self.description = [SimpleNamespace(name=key) for key in self.existing]
            self._next_row = tuple(self.existing.values())

    def fetchone(self) -> tuple[Any, ...] | None:
        row = self._next_row
        self._next_row = None
        return row


def _upsert(
    cursor: RecordingCursor,
    *,
    observed_at: datetime | None,
    person_id: int = 12345,
) -> str | None:
    return global_memory.upsert_signal_candidate_to_global(
        cursor,  # type: ignore[arg-type]
        tenant_id="org_ordering",
        linkedin_url="https://www.linkedin.com/in/ordering-probe/",
        name="Ordering Probe",
        headline="Backend Engineer",
        location_city="Bengaluru",
        location_country="India",
        seniority_band="Senior",
        skills=["Python"],
        signal_candidate_id="signal-ordering-probe",
        profile_observed_at=observed_at,
        public_profile={
            "crustdata_person_id": person_id,
            "basic_profile": {
                "name": "Incoming Probe",
                "headline": "Backend Engineer",
                "location": {"city": "Bengaluru", "country_code": "IN"},
            },
        },
    )


@pytest.mark.parametrize("delta", [timedelta(minutes=-1), timedelta(0)])
def test_global_signal_upsert_ignores_older_or_equal_public_observation(
    monkeypatch: pytest.MonkeyPatch,
    delta: timedelta,
) -> None:
    stored_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    cursor = RecordingCursor(stored_at)
    monkeypatch.setattr(
        global_memory,
        "_find_existing_all",
        lambda *_args, **_kwargs: (cursor.existing.copy(), []),
    )

    result = _upsert(cursor, observed_at=stored_at + delta)

    assert result == _GLOBAL_ID
    sql = [statement for statement, _params in cursor.statements]
    assert any("pg_advisory_xact_lock" in statement for statement in sql)
    assert not any(statement.startswith("UPDATE global_candidates") for statement in sql)
    assert not any(statement.startswith("INSERT INTO candidate_provenance") for statement in sql)
    assert not any("candidate_merge_queue" in statement for statement in sql)


def test_global_signal_upsert_ignores_undated_replay_after_dated_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    cursor = RecordingCursor(stored_at)
    monkeypatch.setattr(
        global_memory,
        "_find_existing_all",
        lambda *_args, **_kwargs: (cursor.existing.copy(), []),
    )

    assert _upsert(cursor, observed_at=None) == _GLOBAL_ID
    assert not any(
        statement.startswith("UPDATE global_candidates") for statement, _params in cursor.statements
    )


def test_global_signal_upsert_advances_public_projection_only_for_newer_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    incoming_at = stored_at + timedelta(minutes=1)
    cursor = RecordingCursor(stored_at)
    monkeypatch.setattr(
        global_memory,
        "_find_existing_all",
        lambda *_args, **_kwargs: (cursor.existing.copy(), []),
    )

    result = _upsert(cursor, observed_at=incoming_at)

    assert result == _GLOBAL_ID
    update = next(
        (statement, params)
        for statement, params in cursor.statements
        if statement.startswith("UPDATE global_candidates")
    )
    assert "last_evidence_at = GREATEST(last_evidence_at, %s)" in update[0]
    assert "public_profile_observed_at = COALESCE(%s, now())" in update[0]
    assert update[1][0] == incoming_at
    assert incoming_at in update[1]
    provenance = next(
        params
        for statement, params in cursor.statements
        if statement.startswith("INSERT INTO candidate_provenance")
    )
    assert provenance[1] == "{}"
