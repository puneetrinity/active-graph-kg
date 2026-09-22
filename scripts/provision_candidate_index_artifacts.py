"""Build-only exact snapshot acquisition; no runtime download or credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

MANIFEST = Path(__file__).with_name("candidate_index_artifacts.json")


def require(condition: bool) -> None:
    if not condition:
        raise ValueError("candidate_index_artifact_invalid")


def manifest(path: Path = MANIFEST) -> dict:
    value = json.loads(path.read_text())
    require(set(value) == {"version", "cache_dir", "models"} and value["version"] == 1)
    require(value["cache_dir"] == "/opt/ealana-models/hub")
    require([m["kind"] for m in value["models"]] == ["embedding", "reranker"])
    for model in value["models"]:
        require(set(model) == {"kind", "repo", "revision", "files"})
        require(bool(re.fullmatch(r"[\w-]+/[\w.-]+", model["repo"])))
        require(bool(re.fullmatch(r"[a-f0-9]{40}", model["revision"])))
        paths = [f["path"] for f in model["files"]]
        require(len(paths) == len(set(paths)) and 1 <= len(paths) <= 32)
        for file in model["files"]:
            require(set(file) == {"path", "size", "sha256"})
            require(not Path(file["path"]).is_absolute() and ".." not in Path(file["path"]).parts)
            require(type(file["size"]) is int and 0 < file["size"] <= 1024**3)
            require(bool(re.fullmatch(r"[a-f0-9]{64}", file["sha256"])))
    return value


def snapshot(cache: Path, model: dict) -> Path:
    return cache / ("models--" + model["repo"].replace("/", "--")) / "snapshots" / model["revision"]


def verify(cache: Path, spec: dict, *, refs: bool = True) -> dict:
    """Full-byte verification, including blob targets inside the cache."""
    total = 0
    for model in spec["models"]:
        root = snapshot(cache, model)
        require(root.is_dir() and not root.is_symlink())
        for file in model["files"]:
            target = root / file["path"]
            require(target.is_file() and target.resolve().is_relative_to(cache.resolve()))
            require(target.stat().st_size == file["size"])
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            require(digest.hexdigest() == file["sha256"])
            total += 1
        if refs:
            ref = root.parent.parent / "refs" / "main"
            require(ref.is_file() and not ref.is_symlink() and ref.read_text() == model["revision"])
    return {"models": len(spec["models"]), "files": total, "verified": True}


def provision(cache: Path, spec: dict) -> dict:
    from huggingface_hub import snapshot_download

    # An exact public revision and file digest are required even on a warm cache.
    for model in spec["models"]:
        result = snapshot_download(
            repo_id=model["repo"],
            revision=model["revision"],
            cache_dir=str(cache),
            allow_patterns=[f["path"] for f in model["files"]],
            token=False,
            local_files_only=False,
            max_workers=2,
        )
        require(Path(result).resolve() == snapshot(cache, model).resolve())
    verify(cache, spec, refs=False)
    for model in spec["models"]:
        ref = snapshot(cache, model).parent.parent / "refs" / "main"
        require(not ref.is_symlink())
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(model["revision"])
    return verify(cache, spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    spec = manifest()
    cache = Path(spec["cache_dir"])
    require(os.getenv("HF_HUB_CACHE") == str(cache))
    try:
        result = verify(cache, spec) if args.verify_only else provision(cache, spec)
    except Exception:
        raise SystemExit("candidate_index_artifact_invalid") from None
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
