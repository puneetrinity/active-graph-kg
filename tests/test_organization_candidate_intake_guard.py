from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts import organization_candidate_intake_guard as guard

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    *guard.FROZEN_HASHES,
    "db/migrations/026_organization_private_candidate_intake.sql",
    "activekg/api/organization_candidates.py",
    "activekg/api/main.py",
    "activekg/common/migration_manifest.py",
    "scripts/schema_control_callers.json",
)


def _copy_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    for relative in FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    return root


@pytest.mark.parametrize(
    ("relative", "old", "new", "code"),
    [
        (
            "activekg/api/organization_candidates.py",
            'claims.actor_id != "vantahire-backend"',
            'claims.actor_id != "any-service"',
            "receiver authority",
        ),
        (
            "activekg/api/organization_candidates.py",
            "require_allowed(decision, global_use=False)",
            "require_allowed(decision, global_use=True)",
            "receiver authority",
        ),
        (
            "activekg/api/organization_candidates.py",
            'exclude={"idempotency_key", "privacy_subject"}',
            'exclude={"idempotency_key"}',
            "transient privacy subject",
        ),
        (
            "db/migrations/026_organization_private_candidate_intake.sql",
            "BEFORE UPDATE OR DELETE",
            "BEFORE UPDATE",
            "trigger census",
        ),
        (
            "activekg/api/main.py",
            "app.include_router(organization_candidates_router)",
            "# route removed",
            "route or readiness",
        ),
        (
            "activekg/common/migration_manifest.py",
            '    "026_organization_private_candidate_intake.sql",',
            '    "026_removed_authority.sql",',
            "manifest",
        ),
    ],
)
def test_guard_refuses_contract_mutations(
    tmp_path: Path, relative: str, old: str, new: str, code: str
) -> None:
    root = _copy_root(tmp_path)
    path = root / relative
    source = path.read_text(encoding="utf-8")
    assert old in source
    path.write_text(source.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(guard.GuardError, match=code):
        guard.validate(root)


def test_guard_refuses_frozen_privacy_authority_drift(tmp_path: Path) -> None:
    root = _copy_root(tmp_path)
    path = root / "activekg/privacy/models.py"
    path.write_bytes(path.read_bytes() + b"\n# mutation\n")
    with pytest.raises(guard.GuardError, match="frozen authority drifted"):
        guard.validate(root)


def test_checked_in_guard_passes() -> None:
    guard.validate()
