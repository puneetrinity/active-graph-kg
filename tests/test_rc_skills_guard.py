import shutil
from pathlib import Path

import pytest

from scripts.rc_skills_guard import ROOT, GuardError, validate


def test_real_rc_boundary():
    validate()


@pytest.mark.parametrize(
    "path,before,after",
    [
        (
            "db/migrations/030_rc_skill_observations.sql",
            "FORCE ROW LEVEL SECURITY",
            "NO FORCE ROW LEVEL SECURITY",
        ),
        ("db/migrations/030_rc_skill_observations.sql", "ON DELETE RESTRICT", "ON DELETE CASCADE"),
        ("activekg/rc_skills/repository.py", "candidate_privacy_match(", "unchecked_match("),
        ("activekg/rc_skills/router.py", "claims.scopes != [scope]", "False"),
        ("activekg/rc_skills/router.py", "claims.issuer != auth.SIGNAL_JWT_ISSUER", "False"),
        ("activekg/rc_skills/contracts.py", 'extra="forbid"', 'extra="ignore"'),
    ],
)
def test_mutated_boundary_is_refused(tmp_path, path, before, after):
    for source in (ROOT / "activekg/rc_skills").glob("*.py"):
        target = tmp_path / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    migration = Path("db/migrations/030_rc_skill_observations.sql")
    (tmp_path / migration).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / migration, tmp_path / migration)
    tolerance = Path("db/migrations/031_rc_clock_tolerance.sql")
    shutil.copyfile(ROOT / tolerance, tmp_path / tolerance)
    target = tmp_path / path
    assert before in target.read_text()
    target.write_text(target.read_text().replace(before, after))
    with pytest.raises(GuardError):
        validate(tmp_path)
