from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from activekg.api import global_memory


class RecordingCursor:
    def __init__(self, stored_observed_at: datetime | None):
        self.stored_observed_at = stored_observed_at
        self.statements: list[tuple[str, Any]] = []
        self._next_row: tuple[Any, ...] | None = None

    def execute(self, sql: str, params: Any = None) -> None:
        normalized = " ".join(sql.split())
        self.statements.append((normalized, params))
        if "SELECT source_detail ->> 'profile_observed_at'" in normalized:
            value = self.stored_observed_at.isoformat() if self.stored_observed_at else None
            self._next_row = (value,)
        else:
            self._next_row = None

    def fetchone(self) -> tuple[Any, ...] | None:
        row = self._next_row
        self._next_row = None
        return row


def _upsert(cursor: RecordingCursor, observed_at: datetime) -> str | None:
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
    )


@pytest.mark.parametrize("delta", [timedelta(minutes=-1), timedelta(0)])
def test_global_signal_upsert_ignores_older_or_equal_observation(
    monkeypatch: pytest.MonkeyPatch,
    delta: timedelta,
) -> None:
    stored_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    cursor = RecordingCursor(stored_at)
    monkeypatch.setattr(
        global_memory,
        "_find_existing_all",
        lambda *_args, **_kwargs: ({"id": "00000000-0000-0000-0000-000000000123"}, []),
    )

    result = _upsert(cursor, stored_at + delta)

    assert result == "00000000-0000-0000-0000-000000000123"
    sql = [statement for statement, _params in cursor.statements]
    assert not any(statement.startswith("UPDATE global_candidates") for statement in sql)
    assert not any(statement.startswith("INSERT INTO candidate_provenance") for statement in sql)


def test_global_signal_upsert_advances_only_to_newer_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_at = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
    incoming_at = stored_at + timedelta(minutes=1)
    cursor = RecordingCursor(stored_at)
    monkeypatch.setattr(
        global_memory,
        "_find_existing_all",
        lambda *_args, **_kwargs: ({"id": "00000000-0000-0000-0000-000000000123"}, []),
    )

    result = _upsert(cursor, incoming_at)

    assert result == "00000000-0000-0000-0000-000000000123"
    update = next(
        (statement, params)
        for statement, params in cursor.statements
        if statement.startswith("UPDATE global_candidates")
    )
    assert "last_evidence_at = GREATEST(last_evidence_at, %s)" in update[0]
    assert update[1][0] == incoming_at
    assert any(
        statement.startswith("INSERT INTO candidate_provenance")
        for statement, _params in cursor.statements
    )
