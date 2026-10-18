import os
from uuid import uuid4

import pytest

from activekg.candidate_index.recovery import checked_pairs
from scripts.recover_candidate_resume_index import read_private


def test_selection_is_bounded_unique_and_explicit():
    pair = {"source_id": str(uuid4()), "generation_id": str(uuid4())}
    assert checked_pairs("org_32", [pair]) == [pair]
    for scope, pairs in (
        ("public", [pair]),
        ("org_32", []),
        ("org_32", [pair] * 2),
        ("org_32", [{**pair, "email": "secret"}]),
    ):
        with pytest.raises(ValueError):
            checked_pairs(scope, pairs)


def test_plan_files_private_no_symlinks(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text("{}")
    os.chmod(path, 0o600)
    assert read_private(str(path))[0] == {}
    os.chmod(path, 0o644)
    with pytest.raises(ValueError):
        read_private(str(path))
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        read_private(str(link))
