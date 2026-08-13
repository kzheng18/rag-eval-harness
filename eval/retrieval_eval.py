"""Offline retrieval evaluation. Scores hit@k and MRR over the golden set, both for
raw vector search (baseline) and after reranking, and fails (exit 1) if the reranked
numbers fall below the thresholds in eval/thresholds.yaml -- or if reranking makes
things worse.

Runs with no API keys and no heavy models (tfidf + bm25), so it gates every push.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from eval.common import hit_at_k, load_golden, reciprocal_rank, resolve_gold_ids
from rageval.chunking import load_corpus
from rageval.config import CORPUS_DIR, load_config
from rageval.embeddings import build_embedder
from rageval.rerank import build_reranker
from rageval.vectorstore import build_store

THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.yaml"


def _score(ranked_ids_per_q, gold_per_q, k):
    rr = [reciprocal_rank(ids, g) for ids, g in zip(ranked_ids_per_q, gold_per_q)]
    hit = [hit_at_k(ids, g, k) for ids, g in zip(ranked_ids_per_q, gold_per_q)]
    return sum(hit) / len(hit), sum(rr) / len(rr)


def evaluate() -> dict:
    config = load_config()
    thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))["retrieval"]
    k = int(thresholds["k"])

    chunks = load_corpus(CORPUS_DIR, config.chunk_size, config.chunk_overlap)
    embedder = build_embedder(config.embedding_backend)
    store = build_store(config.vector_store)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    reranker = build_reranker(config.rerank_backend, cohere_api_key=config.cohere_api_key)

    golden = load_golden()
    gold_per_q = [resolve_gold_ids(it, chunks) for it in golden]

    base_ids, rr_ids = [], []
    for it in golden:
        candidates = store.search(embedder.embed_query(it.question), config.top_k)
        base_ids.append([h.chunk.id for h in candidates])
        reranked = reranker.rerank(it.question, candidates, config.top_k)
        rr_ids.append([h.chunk.id for h in reranked])

    base_hit, base_mrr = _score(base_ids, gold_per_q, k)
    rr_hit, rr_mrr = _score(rr_ids, gold_per_q, k)

    return {
        "n": len(golden),
        "k": k,
        "rerank_backend": config.rerank_backend,
        "baseline": {"hit_at_k": base_hit, "mrr": base_mrr},
        "reranked": {"hit_at_k": rr_hit, "mrr": rr_mrr},
        "min_hit_at_k": float(thresholds["min_hit_at_k"]),
        "min_mrr": float(thresholds["min_mrr"]),
    }


def main() -> int:
    m = evaluate()
    b, r = m["baseline"], m["reranked"]
    print(f"retrieval eval  (n={m['n']}, k={m['k']}, rerank={m['rerank_backend']})")
    print(f"  baseline   hit@k={b['hit_at_k']:.3f}  MRR={b['mrr']:.3f}")
    print(f"  reranked   hit@k={r['hit_at_k']:.3f}  MRR={r['mrr']:.3f}   "
          f"(dMRR={r['mrr'] - b['mrr']:+.3f})")
    print(f"  floor      hit@k>={m['min_hit_at_k']:.2f}  MRR>={m['min_mrr']:.2f}")

    failures = []
    if r["hit_at_k"] < m["min_hit_at_k"]:
        failures.append("hit@k below floor")
    if r["mrr"] < m["min_mrr"]:
        failures.append("MRR below floor")
    if r["mrr"] + 1e-9 < b["mrr"]:
        failures.append("reranking made MRR worse than baseline")

    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
