"""Lexical (sparse) retrieval over the whole corpus, and rank fusion.

This is the missing half of hybrid search. The reranker in ``rerank.py`` only
*reorders* the candidate pool that dense vector search already returned -- so if the
right chunk never makes it into that pool, no reranker can save it. Dense embeddings
have a well-known blind spot here: a rare, discriminating token (an error code, a
symbol, an exact name) gets averaged away in the pooled vector, so the on-topic chunk
falls outside top_k and is gone.

A BM25 index over the full corpus retrieves on exact term overlap, so it catches those
cases. We then fuse the dense and lexical candidate lists with Reciprocal Rank Fusion,
which combines rankings by position instead of by score -- no need to make cosine
similarities and BM25 scores commensurable, which they are not.
"""
from __future__ import annotations

import re

from .chunking import Chunk
from .vectorstore import Hit

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class LexicalIndex:
    """BM25 over every chunk in the corpus (not just a candidate pool).

    Deliberately separate from ``BM25Reranker``: the reranker is scoped to the handful
    of hits dense search already found, whereas this indexes the whole corpus so it can
    surface a chunk dense search missed entirely.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def add(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        # BM25Okapi needs at least one document; guard empty corpora.
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in self._chunks]) if self._chunks else None

    def search(self, query: str, top_k: int) -> list[Hit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(self._chunks)), key=lambda i: -scores[i])[:top_k]
        return [Hit(self._chunks[i], float(scores[i])) for i in order]


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hit]], *, k: int = 60, top_n: int | None = None
) -> list[Hit]:
    """Combine several ranked hit lists into one by Reciprocal Rank Fusion.

    Each list contributes ``1 / (k + rank)`` for every chunk it ranks (rank starts at
    1). A chunk that ranks decently in *both* dense and lexical results beats one that
    ranks highly in only a single list -- which is exactly the agreement signal we want
    from hybrid search. ``k`` damps the influence of the very top ranks; 60 is the value
    from the original RRF paper (Cormack et al., 2009). The fused ``score`` is the RRF
    weight, so downstream code can still sort or threshold on it.
    """
    agg: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk.id
            agg[cid] = agg.get(cid, 0.0) + 1.0 / (k + rank)
            chunk_by_id.setdefault(cid, hit.chunk)

    ordered = sorted(agg.items(), key=lambda kv: -kv[1])
    if top_n is not None:
        ordered = ordered[:top_n]
    return [Hit(chunk_by_id[cid], score) for cid, score in ordered]
