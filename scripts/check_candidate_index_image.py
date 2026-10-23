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
    # v3 search chunks are bounded by the pinned tokenizer: dense and long accepted profiles must embed.
    from activekg.candidate_index import resume_history_policy as history
    from activekg.candidate_index.contracts import IndexPolicy
    from activekg.candidate_index.processing import resume_extraction

    v3 = IndexPolicy(
        **history.policy_fields(),
        primary_model_id="openai/gpt-oss-20b",
        fallback_model_id="openai/gpt-oss-120b",
        embedding_model_id="all-MiniLM-L6-v2",
        embedding_artifact_revision=model["revision"],
    )
    v3_chunks = 0
    for skills in (
        [f"C++/C#/.NET v{i}" for i in range(100)],
        [
            f"Enterprise distributed systems architecture and reliability practices level {i}"
            for i in range(100)
        ],
    ):
        raw = {
            "skills_raw": skills,
            "skills_normalized": [],
            "confidence": 0.9,
            "employment_history": [],
        }
        result = resume_extraction(raw, "; ".join(skills), v3, "primary", {})
        embedded = await embed_once(
            [c["text"] for c in result.chunks],
            model["repo"],
            model["revision"],
            deadline=time.monotonic() + 180,
        )
        assert len(embedded) == len(result.chunks)
        v3_chunks += len(embedded)
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
        "v3_token_bounded_chunks_embedded": v3_chunks,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(asyncio.run(smoke()), sort_keys=True))
    except Exception:
        raise SystemExit("candidate_index_image_smoke_refused") from None
