from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts import sourced_candidate_ingest_guard as guard

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    *guard.FROZEN_HASHES,
    "db/migrations/025_approved_provider_candidate_ingest.sql",
    "activekg/api/sourced_candidates.py",
    "activekg/api/main.py",
    "activekg/api/global_memory.py",
    "activekg/common/migration_manifest.py",
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
            "activekg/api/sourced_candidates.py",
            'claims.actor_id != "signal-service"',
            'claims.actor_id != "any-service"',
            "receiver authority",
        ),
        (
            "activekg/api/sourced_candidates.py",
            "require_allowed(decision, global_use=True)",
            "require_allowed(decision, global_use=False)",
            "receiver authority",
        ),
        (
            "activekg/api/sourced_candidates.py",
            'Literal["crustdata_person_v1"]',
            "str",
            "receiver authority",
        ),
        (
            "activekg/api/main.py",
            'sourced_candidate_ingest_mode() == "canonical_only"',
            "False",
            "old Signal writer",
        ),
        (
            "activekg/api/global_memory.py",
            'raise HTTPException(status_code=410, detail="global_candidate_writer_retired")',
            "_require_enabled()",
            "retired global writer",
        ),
        (
            "db/migrations/025_approved_provider_candidate_ingest.sql",
            "BEFORE UPDATE OR DELETE",
            "BEFORE UPDATE",
            "append-only trigger census",
        ),
        (
            "db/migrations/025_approved_provider_candidate_ingest.sql",
            "    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),",
            "    tenant_id TEXT,\n    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),",
            "tenant/job/rank state",
        ),
        (
            "activekg/common/migration_manifest.py",
            "len(MIGRATIONS) != 25",
            "len(MIGRATIONS) != 24",
            "manifest",
        ),
        (
            "activekg/api/main.py",
            "app.include_router(sourced_candidates_router)",
            "# route removed",
            "route registration",
        ),
    ],
)
def test_guard_refuses_contract_mutations(
    tmp_path: Path,
    relative: str,
    old: str,
    new: str,
    code: str,
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
