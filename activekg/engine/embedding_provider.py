from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from activekg.common.logger import get_enhanced_logger


class EmbeddingProvider:
    """Simple wrapper for embeddings. Supports 'sentence-transformers' or 'ollama' backends.

    This is a thin interface; wire in your model of choice.
    """

    def __init__(self, backend: str = "sentence-transformers", model_name: str | None = None):
        self.backend = backend
        self.model_name = model_name or (
            "all-MiniLM-L6-v2" if backend == "sentence-transformers" else "nomic-embed-text"
        )
        self.logger = get_enhanced_logger(__name__)
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        if self.backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as e:
                raise ImportError("sentence-transformers not installed") from e
            import torch
            # Fix for sentence-transformers 5.x meta tensor issue
            # Load without device parameter first, then move to CPU
            # This avoids the meta tensor initialization issue
            import os

            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            model_kwargs = {
                "device_map": None,
                "low_cpu_mem_usage": False,
                "torch_dtype": torch.float32,
            }
            self._model = SentenceTransformer(
                self.model_name, device="cpu", model_kwargs=model_kwargs
            )
        elif self.backend == "ollama":
            try:
                import ollama
            except Exception as e:
                raise ImportError("ollama client not installed") from e
            self._model = ollama
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure_model()
        assert self._model is not None, "Model should be initialized after _ensure_model()"
        if self.backend == "sentence-transformers":
            vecs = self._model.encode(texts, convert_to_numpy=True)
            vecs = vecs.astype(np.float32)
            # L2-normalize to stabilize cosine similarity and drift metrics
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            return (vecs / norms).astype(np.float32)
        elif self.backend == "ollama":
            # Minimalistic batch wrapper; consider streaming or batching
            vectors: list[list[float]] = []
            for t in texts:
                res = self._model.embeddings({"model": self.model_name, "prompt": t})
                vectors.append(res["embedding"])
            vecs = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0.0, 1.0, norms)
            return (vecs / norms).astype(np.float32)
        raise RuntimeError("Unreachable")
