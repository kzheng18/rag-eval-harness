"""Shared helpers for the eval scripts: bootstrap path, load golden set, metrics."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Allow running the evals without installing the package (src/ layout).
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rageval.chunking import Chunk  # noqa: E402
from rageval.config import GOLDEN_PATH  # noqa: E402


@dataclass(frozen=True)
class GoldItem:
    question: str
    doc_id: str
    needle: str


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldItem]:
    items: list[GoldItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        items.append(GoldItem(row["question"], row["doc_id"], row["needle"]))
    return items


def resolve_gold_ids(item: GoldItem, chunks: list[Chunk]) -> set[str]:
    """Find the chunk id(s) that actually contain the needle for this question.

    Decoupling the golden set from content-hashed chunk ids means re-chunking never
    silently breaks the eval: if the needle no longer resolves, that is a hard error.
    """
    ids = {c.id for c in chunks if c.doc_id == item.doc_id and item.needle in c.text}
    if not ids:
        raise ValueError(
            f"Needle not found for question {item.question!r}: "
            f"no chunk in doc {item.doc_id!r} contains {item.needle!r}"
        )
    return ids


def reciprocal_rank(ranked_ids: list[str], gold_ids: set[str]) -> float:
    for rank, cid in enumerate(ranked_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


def hit_at_k(ranked_ids: list[str], gold_ids: set[str], k: int) -> float:
    return 1.0 if any(cid in gold_ids for cid in ranked_ids[:k]) else 0.0
