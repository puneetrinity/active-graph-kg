"""Chunking bounded by an extra fit rule; real-tokenizer proof where the pinned snapshot exists."""

import random
import re
from dataclasses import replace

import pytest

from activekg.candidate_index import resume_history_policy as history
from activekg.candidate_index.contracts import IndexContractError, chunk_manifest, sha256
from activekg.candidate_index.processing import policy_chunks, resume_extraction
from activekg.candidate_index.providers import WorkRefused, cached_embedding_path
from tests.test_candidate_index import POLICY


def legacy_chunk_manifest(text, policy):
    """Exact copy of the pre-change character-only chunker (v1/v2 must be unchanged)."""
    result, start = [], 0
    while start < len(text):
        if len(result) >= policy.max_chunks:
            raise IndexContractError("candidate_index_chunk_overflow")
        stop = min(start + policy.chunk_characters, len(text))
        if stop < len(text):
            section = text[start:stop]
            boundaries = list(re.finditer(r"[.!?](?:\s+)|\n+", section))
            if not boundaries:
                boundaries = list(re.finditer(r"\s+", section))
            if boundaries:
                stop = start + boundaries[-1].end()
        value = text[start:stop]
        result.append({"ordinal": len(result), "text": value, "sha256": sha256(value)})
        start = stop
    return result


def test_without_a_fit_rule_chunking_is_byte_identical_to_before():
    rng = random.Random(7)
    words = ["Python", "C++", "AWS.", "Kubernetes\n", "a", "data-engineering", "!", "  "]
    for _ in range(300):
        text = "".join(
            rng.choice(words) + rng.choice([" ", "", ". "]) for _ in range(rng.randint(1, 900))
        )
        if not text.strip():
            continue
        try:
            expected = legacy_chunk_manifest(text, POLICY)
        except IndexContractError as exc:
            with pytest.raises(IndexContractError, match=str(exc)):
                chunk_manifest(text, POLICY)
            continue
        assert chunk_manifest(text, POLICY) == expected


def test_fit_rule_bounds_every_chunk_and_stays_lossless():
    rng = random.Random(11)
    fits = lambda t: len(t.split()) + t.count("+") <= 40  # noqa: E731 - dense-token stand-in
    for _ in range(200):
        text = " ".join(
            rng.choice(["C++/C#", "Python.", "Go", "Kubernetes", "x" * 30])
            for _ in range(rng.randint(1, 600))
        )
        chunks = chunk_manifest(text, POLICY, fits)
        assert "".join(c["text"] for c in chunks) == text
        assert all(fits(c["text"]) and len(c["text"]) <= POLICY.chunk_characters for c in chunks)


def test_unbreakable_run_uses_a_character_split_and_refuses_when_nothing_fits():
    chunks = chunk_manifest("x" * 100, POLICY, lambda t: len(t) <= 30)
    assert [len(c["text"]) for c in chunks] == [30, 30, 30, 10]
    with pytest.raises(IndexContractError, match="chunk_overflow"):
        chunk_manifest("abc", POLICY, lambda t: False)


# ---- Real pinned tokenizer (local snapshot; CI proves it inside the offline image) ----
REAL = replace(
    POLICY,
    embedding_model_id="all-MiniLM-L6-v2",
    embedding_artifact_revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    **history.policy_fields(),
)


def real_tokens():
    try:
        path = cached_embedding_path(REAL.embedding_model_id, REAL.embedding_artifact_revision)
    except WorkRefused:
        pytest.skip("pinned embedding snapshot not available")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(path), device="cpu")
    return lambda t: len(
        model.tokenizer.encode(t, truncation=False, add_special_tokens=True)
    ), model.max_seq_length


@pytest.mark.parametrize(
    "skills",
    [
        [f"C++/C#/.NET v{i}" for i in range(100)],  # Codex R1: 600-char chunks reached 425 tokens
        [
            "Enterprise distributed systems architecture and performance engineering "
            f"with advanced reliability practices level {i}"
            for i in range(100)
        ],  # Codex R2: 600-char chunks exceeded the 32-chunk cap
    ],
)
def test_v3_chunks_fit_the_real_embedding_tokenizer(skills):
    count, limit = real_tokens()
    text = "; ".join(skills)
    raw = {
        "skills_raw": skills,
        "skills_normalized": [],
        "confidence": 0.9,
        "employment_history": [],
    }
    result = resume_extraction(raw, text, REAL, "primary", {})
    assert 1 <= len(result.chunks) <= REAL.max_chunks
    assert all(count(c["text"]) <= limit == 256 for c in result.chunks)
    assert "".join(c["text"] for c in result.chunks) == result.professional_text
    # The embed stage re-derives the same manifest before encoding.
    assert policy_chunks(result.professional_text, REAL) == result.chunks
