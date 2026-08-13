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
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "tfidf")
    rerank_backend: str = os.getenv("RERANK_BACKEND", "bm25")
    vector_store: str = os.getenv("VECTOR_STORE", "numpy")

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "600"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "100"))

    top_k: int = int(os.getenv("TOP_K", "10"))       # candidates from vector search
    final_k: int = int(os.getenv("FINAL_K", "4"))    # kept after rerank

    cohere_api_key: str | None = os.getenv("COHERE_API_KEY") or None
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY") or None


def load_config() -> Config:
    return Config()
