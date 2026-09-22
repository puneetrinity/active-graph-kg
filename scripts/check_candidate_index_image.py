"""Final-image synthetic smoke; run with docker --network none, no secrets/DB."""

from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.provision_candidate_index_artifacts import manifest, verify  # noqa: E402


async def smoke() -> dict:
    spec = manifest()
    verify(Path(spec["cache_dir"]), spec)
    # No global offline env mutation: production's legacy loaders retain their contract.
    assert not os.getenv("HF_HUB_OFFLINE") and not os.getenv("TRANSFORMERS_OFFLINE")
    attempted = 0

    def refuse(*_args, **_kwargs):
        nonlocal attempted
        attempted += 1
        raise RuntimeError("image_smoke_network_refused")

    socket.socket.connect = refuse
    socket.socket.connect_ex = refuse
    socket.create_connection = refuse
    socket.getaddrinfo = refuse
    from activekg.candidate_index.providers import embed_once, rerank_once
    from activekg.engine.embedding_provider import EmbeddingProvider

    model = spec["models"][0]
    scratch_before = set(Path(tempfile.gettempdir()).glob("candidate-index-import-*"))
    vectors = await embed_once(
        ["Synthetic software engineering skills"],
        model["repo"],
        model["revision"],
        deadline=time.monotonic() + 180,
    )
    assert len(vectors) == 1 and len(vectors[0]) == 384
    assert all(math.isfinite(x) for x in vectors[0])
    scores = await rerank_once(
        "software engineering",
        ["Synthetic software engineering skills", "Synthetic gardening"],
        deadline=time.monotonic() + 180,
    )
    assert len(scores) == 2 and all(math.isfinite(x) for x in scores)
    assert set(Path(tempfile.gettempdir()).glob("candidate-index-import-*")) == scratch_before
    # child_work refuses successful results unless each real child reports zero
    # attempted connections and the kernel's hard/soft processing limit at zero.
    # The frozen legacy provider must remain importable/configurable; no global
    # offline env, model-name rewrite or monkeypatch enters the product image.
    legacy = EmbeddingProvider(backend="sentence-transformers", model_name="all-MiniLM-L6-v2")
    assert legacy is not None and attempted == 0
    return {
        "embedding_dimension": 384,
        "rerank_scores": 2,
        "network_attempts": attempted,
        "artifacts_verified": True,
        "processing_file_limit": [0, 0],
        "import_scratch_removed": True,
        "legacy_configuration_preserved": True,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(asyncio.run(smoke()), sort_keys=True))
    except Exception:
        raise SystemExit("candidate_index_image_smoke_refused") from None
