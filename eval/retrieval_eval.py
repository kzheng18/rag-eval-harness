"""Offline retrieval evaluation. Scores hit@k and MRR over the golden set and fails
(exit 1) if either falls below the thresholds in eval/thresholds.yaml.

Runs with no API keys and no heavy models, so it can gate every push in CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from eval.common import hit_at_k, load_golden, reciprocal_rank, resolve_gold_ids
from rageval.chunking import load_corpus
from rageval.config import CORPUS_DIR, load_config
from rageval.embeddings import build_embedder
from rageval.vectorstore import build_store

THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.yaml"


def evaluate() -> dict[str, float]:
    config = load_config()
    thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))["retrieval"]
    k = int(thresholds["k"])

    chunks = load_corpus(CORPUS_DIR, config.chunk_size, config.chunk_overlap)
    embedder = build_embedder(config.embedding_backend)
    store = build_store(config.vector_store)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))

    golden = load_golden()
    rrs: list[float] = []
    hits: list[float] = []
    for item in golden:
        gold_ids = resolve_gold_ids(item, chunks)
        ranked = store.search(embedder.embed_query(item.question), config.top_k)
        ranked_ids = [h.chunk.id for h in ranked]
        rrs.append(reciprocal_rank(ranked_ids, gold_ids))
        hits.append(hit_at_k(ranked_ids, gold_ids, k))

    return {
        "n": float(len(golden)),
        "k": float(k),
        "hit_at_k": sum(hits) / len(hits),
        "mrr": sum(rrs) / len(rrs),
        "min_hit_at_k": float(thresholds["min_hit_at_k"]),
        "min_mrr": float(thresholds["min_mrr"]),
    }


def main() -> int:
    m = evaluate()
    print(
        f"retrieval eval (n={int(m['n'])}, k={int(m['k'])}): "
        f"hit@k={m['hit_at_k']:.3f} (min {m['min_hit_at_k']:.2f})  "
        f"MRR={m['mrr']:.3f} (min {m['min_mrr']:.2f})"
    )
    ok = m["hit_at_k"] >= m["min_hit_at_k"] and m["mrr"] >= m["min_mrr"]
    if not ok:
        print("FAIL: retrieval quality below threshold", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
