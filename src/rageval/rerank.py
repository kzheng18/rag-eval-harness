"""Rerankers reorder the candidate set from vector search.

The whole point: vector similarity gives roughly-relevant candidates but often in the
wrong order, because it compares two vectors independently. A reranker scores the query
and each candidate *together*, so it can tell "mentions the keywords" from "answers the
question".

Backends:
- bm25:          offline, term-weighted reordering of the candidate pool (CI default).
- cohere:        hosted Cohere Rerank (needs COHERE_API_KEY).
- cross-encoder: local cross-encoder via sentence-transformers.
- none:          passthrough baseline.
"""
from __future__ import annotations

import re
from typing import Protocol

from .vectorstore import Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...


class NoopReranker:
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        return hits[:top_n]


class BM25Reranker:
    """BM25 over just the candidate pool. IDF is computed on the candidates, which
    sharpens rare, discriminating query terms that a corpus-wide vector can dilute."""

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([_tokenize(h.chunk.text) for h in hits])
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(hits)), key=lambda i: -scores[i])
        return [Hit(hits[i].chunk, float(scores[i])) for i in order[:top_n]]


class CohereReranker:
    def __init__(self, api_key: str, model: str = "rerank-english-v3.0") -> None:
        import cohere

        self._client = cohere.Client(api_key)
        self._model = model

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        docs = [h.chunk.text for h in hits]
        res = self._client.rerank(query=query, documents=docs, top_n=min(top_n, len(docs)), model=self._model)
        return [Hit(hits[r.index].chunk, float(r.relevance_score)) for r in res.results]


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -float(scores[i]))
        return [Hit(hits[i].chunk, float(scores[i])) for i in order[:top_n]]


def build_reranker(backend: str, *, cohere_api_key: str | None = None) -> Reranker:
    if backend == "none":
        return NoopReranker()
    if backend == "bm25":
        return BM25Reranker()
    if backend == "cohere":
        if not cohere_api_key:
            raise ValueError("RERANK_BACKEND=cohere requires COHERE_API_KEY")
        return CohereReranker(cohere_api_key)
    if backend in ("cross-encoder", "cross_encoder"):
        return CrossEncoderReranker()
    raise ValueError(f"Unknown rerank backend: {backend!r}")
