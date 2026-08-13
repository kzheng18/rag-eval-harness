"""Deterministic, word-aware chunking so the same corpus always yields the same
chunk ids -- important because the golden eval set references chunks by id."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    text: str
    ordinal: int


def _normalize(text: str) -> str:
    return _WS.sub(" ", text).strip()


def chunk_text(text: str, doc_id: str, chunk_size: int, overlap: int) -> list[Chunk]:
    """Sliding window over words, ~chunk_size characters per window with overlap.

    Chunk ids are content-addressed (doc_id + ordinal + short content hash): stable
    across runs, but they change if the text changes -- so a stale golden set fails
    loudly instead of silently pointing at the wrong chunk.
    """
    words = _normalize(text).split(" ")
    if words == [""]:
        return []

    # Group words into windows of roughly chunk_size characters.
    windows: list[list[str]] = []
    cur: list[str] = []
    length = 0
    for w in words:
        if cur and length + len(w) + 1 > chunk_size:
            windows.append(cur)
            cur, length = [], 0
        cur.append(w)
        length += len(w) + 1
    if cur:
        windows.append(cur)

    # Apply overlap by carrying trailing words of each window into the next.
    overlap_words = max(0, overlap // 6)  # ~6 chars/word heuristic
    chunks: list[Chunk] = []
    prev_tail: list[str] = []
    for ordinal, win in enumerate(windows):
        body_words = prev_tail + win
        body = " ".join(body_words)
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:8]
        chunks.append(Chunk(id=f"{doc_id}::{ordinal}::{digest}", doc_id=doc_id, text=body, ordinal=ordinal))
        prev_tail = win[-overlap_words:] if overlap_words else []
    return chunks


def load_corpus(corpus_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        doc_id = path.stem
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), doc_id, chunk_size, overlap))
    return chunks
