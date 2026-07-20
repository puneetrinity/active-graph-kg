from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from activekg.engine.embedding_provider import EmbeddingProvider
from activekg.graph.repository import GraphRepository


def test_sentence_transformer_initializes_once_under_concurrent_first_use(monkeypatch):
    init_calls = 0
    init_calls_lock = threading.Lock()

    class FakeSentenceTransformer:
        def __init__(self, model_name, *, device, model_kwargs):
            nonlocal init_calls
            with init_calls_lock:
                init_calls += 1
            time.sleep(0.05)
            assert model_name == "test-model"
            assert device == "cpu"
            assert model_kwargs["low_cpu_mem_usage"] is False

        def encode(self, texts, *, convert_to_numpy):
            assert convert_to_numpy is True
            return np.ones((len(texts), 3), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=np.float32))

    provider = EmbeddingProvider(model_name="test-model")
    workers = 8
    start = threading.Barrier(workers)

    def encode(index: int) -> np.ndarray:
        start.wait(timeout=2)
        return provider.encode([f"text-{index}"])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(encode, range(workers)))

    assert init_calls == 1
    assert all(result.shape == (1, 3) for result in results)


def test_hugging_face_loads_are_serialized_across_model_types(monkeypatch):
    active_loads = 0
    max_active_loads = 0
    load_counts = {"embedding": 0, "cross_encoder": 0}
    state_lock = threading.Lock()

    def simulate_model_load(model_type: str):
        nonlocal active_loads, max_active_loads
        with state_lock:
            active_loads += 1
            max_active_loads = max(max_active_loads, active_loads)
            load_counts[model_type] += 1
        time.sleep(0.05)
        with state_lock:
            active_loads -= 1

    class FakeSentenceTransformer:
        def __init__(self, model_name, *, device, model_kwargs):
            simulate_model_load("embedding")

        def encode(self, texts, *, convert_to_numpy):
            return np.ones((len(texts), 3), dtype=np.float32)

    class FakeCrossEncoder:
        def __init__(self, model_name, *, device, automodel_args):
            simulate_model_load("cross_encoder")

        def predict(self, pairs):
            return np.ones(len(pairs), dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            CrossEncoder=FakeCrossEncoder,
            SentenceTransformer=FakeSentenceTransformer,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32=np.float32))

    provider = EmbeddingProvider(model_name="test-model")
    repository = object.__new__(GraphRepository)
    repository._cross_encoder = None
    repository.logger = SimpleNamespace(error=lambda *args, **kwargs: None)
    candidate = SimpleNamespace(props={"text": "candidate text"})
    start = threading.Barrier(2)

    def load_embedding_model():
        start.wait(timeout=2)
        provider.encode(["query"])

    def load_cross_encoder():
        start.wait(timeout=2)
        repository._cross_encoder_rerank("query", [(candidate, 0.5)], top_k=1)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(load_embedding_model),
            executor.submit(load_cross_encoder),
        ]
        for future in futures:
            future.result()

    assert max_active_loads == 1
    assert load_counts == {"embedding": 1, "cross_encoder": 1}
