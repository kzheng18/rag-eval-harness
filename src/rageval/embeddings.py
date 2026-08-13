"""Pluggable embedding backends.

- tfidf: offline, deterministic, zero heavy deps. Used by tests and CI.
- sentence-transformers: real dense embeddings for production use.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> "Embedder": ...
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class TfidfEmbedder:
    """TF-IDF vectors, L2-normalized so dot product == cosine similarity."""

    def __init__(self, max_features: int = 4096) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vec = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        self._vec.fit(texts)
        self._fitted = True
        return self

    def _transform(self, texts: list[str]) -> np.ndarray:
        mat = self._vec.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return self._transform(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._transform([text])[0]


class SentenceTransformerEmbedder:
    """Dense embeddings via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def fit(self, texts: list[str]) -> "SentenceTransformerEmbedder":
        return self  # nothing to fit

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True),
            dtype=np.float32,
        )

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedder(backend: str) -> Embedder:
    if backend == "tfidf":
        return TfidfEmbedder()
    if backend in ("sentence-transformers", "st"):
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r}")
