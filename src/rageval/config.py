"""Runtime configuration, read from environment with offline-friendly defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
GOLDEN_PATH = DATA_DIR / "golden.jsonl"
INDEX_DIR = REPO_ROOT / ".index"


@dataclass(frozen=True)
class Config:
    embedding_backend: str = "tfidf"
    rerank_backend: str = "bm25"
    vector_store: str = "numpy"

    chunk_size: int = 600
    chunk_overlap: int = 100

    top_k: int = 10   # candidates from vector search
    final_k: int = 4  # kept after rerank

    cohere_api_key: str | None = None
    openai_api_key: str | None = None
    pinecone_api_key: str | None = None


def load_config() -> Config:
    """Build config from the environment at call time (not import time)."""
    return Config(
        embedding_backend=os.getenv("EMBEDDING_BACKEND", "tfidf"),
        rerank_backend=os.getenv("RERANK_BACKEND", "bm25"),
        vector_store=os.getenv("VECTOR_STORE", "numpy"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "600")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "100")),
        top_k=int(os.getenv("TOP_K", "10")),
        final_k=int(os.getenv("FINAL_K", "4")),
        cohere_api_key=os.getenv("COHERE_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        pinecone_api_key=os.getenv("PINECONE_API_KEY") or None,
    )
