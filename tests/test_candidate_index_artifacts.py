"""Artifact corruption and build-boundary tests; no network or model needed."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import provision_candidate_index_artifacts as artifacts


def fixture(tmp_path):
    spec = copy.deepcopy(artifacts.manifest())
    for model in spec["models"]:
        model["files"] = [
            {"path": "config.json", "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
        ]
        root = artifacts.snapshot(tmp_path, model)
        root.mkdir(parents=True)
        (root / "config.json").write_bytes(b"{}")
        ref = root.parent.parent / "refs" / "main"
        ref.parent.mkdir()
        ref.write_text(model["revision"])
    return spec


def test_pinned_artifacts():
    spec = artifacts.manifest()
    assert [m["revision"] for m in spec["models"]] == [
        "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        "233902d25c440f23af6f7d6e94d2946bac0bee0a",
    ]
    assert len(spec["models"][0]["files"]) == 10
    assert len(spec["models"][1]["files"]) == 6


def test_complete(tmp_path):
    assert artifacts.verify(tmp_path, fixture(tmp_path)) == {
        "models": 2,
        "files": 2,
        "verified": True,
    }


@pytest.mark.parametrize(
    "mutation",
    ["missing", "corrupt", "ref", "blob", "outside"],
    ids=["missing", "corrupt", "wrong-revision", "broken-blob", "outside-cache"],
)
def test_corruption_refuses(tmp_path, mutation):
    cache = tmp_path / "cache"
    spec = fixture(cache)
    root = artifacts.snapshot(cache, spec["models"][0])
    file = root / "config.json"
    if mutation == "missing":
        file.unlink()
    elif mutation == "corrupt":
        file.write_bytes(b"[]")
    elif mutation == "ref":
        (root.parent.parent / "refs" / "main").write_text("0" * 40)
    else:
        file.unlink()
        target = tmp_path / "external"
        if mutation == "outside":
            target.write_bytes(b"{}")
        file.symlink_to(target)
    with pytest.raises(ValueError, match="candidate_index_artifact_invalid"):
        artifacts.verify(cache, spec)


def test_moving_revision_refuses(tmp_path):
    value = artifacts.manifest()
    value["models"][0]["revision"] = "main"
    file = tmp_path / "manifest.json"
    file.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        artifacts.manifest(file)


def test_acquisition_is_pinned_and_anonymous(tmp_path, monkeypatch):
    import huggingface_hub

    spec = fixture(tmp_path)
    calls = []

    def acquire(**kwargs):
        calls.append(kwargs)
        model = next(m for m in spec["models"] if m["repo"] == kwargs["repo_id"])
        return str(artifacts.snapshot(tmp_path, model))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", acquire)
    artifacts.provision(tmp_path, spec)
    assert len(calls) == 2
    assert all(c["token"] is False and c["local_files_only"] is False for c in calls)
    assert [c["revision"] for c in calls] == [m["revision"] for m in spec["models"]]


def test_docker_keeps_legacy_online_contract():
    source = (Path(__file__).parents[1] / "Dockerfile").read_text()
    assert "HF_HUB_OFFLINE" not in source and "TRANSFORMERS_OFFLINE" not in source
    assert "provision_candidate_index_artifacts.py --verify-only" in source
