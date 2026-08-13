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
    """Split on word boundaries into ~chunk_size-char windows with overlap.

    Chunk ids are content-addressed (doc_id + ordinal + a short content hash) so
    they are stable across runs but change if the underlying text changes -- which
    is exactly what you want so a stale golden set fails loudly.
    """
    words = _normalize(text).split(" ")
    chunks: list[Chunk] = []
    step = max(1, chunk_size - overlap)
    ordinal = 0
    i = 0
    # Build by characters but respect word boundaries.
    while i < len(words):
        cur: list[str] = []
        length = 0
        j = i
        while j < len(words) and length + len(words[j]) + 1 <= chunk_size:
            cur.append(words[j])
            length += len(words[j]) + 1
            j += 1
        if not cur:  # single very long word
            cur = [words[i]]
            j = i + 1
        body = " ".join(cur)
        digest = hashlib.sha1(body.encode("utf-8")).hexdigest()[:8]
        chunks.append(Chunk(id=f"{doc_id}::{ordinal}::{digest}", doc_id=doc_id, text=body, ordinal=ordinal))
        ordinal += 1
        # advance by step words
        advance = max(1, min(len(cur), step // max(1, chunk_size // max(1, len(cur)))))
        i += advance if advance < (j - i) else (j - i)
    return chunks


def load_corpus(corpus_dir: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        doc_id = path.stem
        chunks.extend(chunk_text(path.read_text(encoding="utf-8"), doc_id, chunk_size, overlap))
    return chunks
