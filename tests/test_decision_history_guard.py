from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.decision_history_guard import GuardError, validate

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "activekg/api/organization_decision_events.py",
    "activekg/api/main.py",
    "activekg/api/global_memory.py",
    "activekg/embedding/worker.py",
    "activekg/extraction/worker.py",
    "db/migrations/012_global_memory.sql",
    "db/migrations/024_organization_decision_event_inbox.sql",
)


def _copy(tmp_path: Path) -> Path:
    for relative in FILES:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def _mutate(root: Path, relative: str, old: str, new: str) -> None:
    path = root / relative
    source = path.read_text(encoding="utf-8")
    assert old in source
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def test_decision_history_guard_accepts_locked_surface(tmp_path: Path) -> None:
    validate(_copy(tmp_path))


@pytest.mark.parametrize(
    ("relative", "old", "new"),
    [
        (
            "activekg/api/main.py",
            "app.include_router(organization_decision_events_router)",
            "# registration removed",
        ),
        (
            "activekg/api/organization_decision_events.py",
            "from activekg.api import auth",
            "from activekg.api import auth\nfrom activekg.graph.repository import GraphRepository",
        ),
        (
            "activekg/embedding/worker.py",
            "from __future__ import annotations",
            "from __future__ import annotations\nimport activekg.api.organization_decision_events",
        ),
        (
            "activekg/api/global_memory.py",
            "def ingest_feedback_events(",
            "def ingest_feedback_events(\n    # changed legacy authority",
        ),
        (
            "db/migrations/024_organization_decision_event_inbox.sql",
            "FORCE ROW LEVEL SECURITY",
            "NO FORCE ROW LEVEL SECURITY",
        ),
    ],
)
def test_guard_rejects_authority_and_legacy_drift(
    tmp_path: Path, relative: str, old: str, new: str
) -> None:
    root = _copy(tmp_path)
    _mutate(root, relative, old, new)
    with pytest.raises((GuardError, SyntaxError)):
        validate(root)
