"""Vector stores. numpy (offline default), Chroma and Pinecone (optional)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


class NumpyStore:
    """In-memory cosine search over a normalized matrix. No external deps."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._chunks = list(chunks)
        self._matrix = np.ascontiguousarray(vectors, dtype=np.float32)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[Hit]:
        if self._matrix is None:
            raise RuntimeError("Store is empty; call add() first.")
        scores = self._matrix @ query_vec.astype(np.float32)
        idx = np.argsort(-scores)[:top_k]
        return [Hit(self._chunks[i], float(scores[i])) for i in idx]


class ChromaStore:
    """Persistent vector store backed by ChromaDB (optional dependency)."""

    def __init__(self, path: str = ".chroma", collection: str = "rageval") -> None:
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )
        self._by_id: dict[str, Chunk] = {}

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self._by_id = {c.id: c for c in chunks}
        self._col.upsert(
            ids=[c.id for c in chunks],
            embeddings=[v.tolist() for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "ordinal": c.ordinal} for c in chunks],
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[Hit]:
        res = self._col.query(query_embeddings=[query_vec.tolist()], n_results=top_k)
        hits: list[Hit] = []
        for cid, dist in zip(res["ids"][0], res["distances"][0]):
            chunk = self._by_id.get(cid)
            if chunk is not None:
                hits.append(Hit(chunk, 1.0 - float(dist)))  # cosine distance -> similarity
        return hits


def build_store(kind: str) -> NumpyStore | ChromaStore:
    if kind == "numpy":
        return NumpyStore()
    if kind == "chroma":
        return ChromaStore()
    raise ValueError(f"Unknown vector store: {kind!r}")
