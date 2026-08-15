"""Offline retrieval evaluation. Scores hit@k and MRR over the golden set, both for
raw vector search (baseline) and after the configured retrieval + rerank path, and
fails (exit 1) if the final numbers fall below the thresholds in eval/thresholds.yaml
-- or if the pipeline ends up worse than the raw-vector baseline.

Baseline is always pure dense vector search. The "reranked" row reflects whatever
RETRIEVAL_MODE is set to: with "dense" it is rerank over the dense pool; with "hybrid"
it is rerank over the RRF-fused dense+lexical pool, so the measured effect of hybrid
search shows up right here in CI.

Runs with no API keys and no heavy models (tfidf + bm25), so it gates every push.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

from eval.common import hit_at_k, load_golden, reciprocal_rank, resolve_gold_ids
from rageval.config import load_config
from rageval.pipeline import RAGPipeline

THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.yaml"


def _score(ranked_ids_per_q, gold_per_q, k):
    rr = [reciprocal_rank(ids, g) for ids, g in zip(ranked_ids_per_q, gold_per_q)]
    hit = [hit_at_k(ids, g, k) for ids, g in zip(ranked_ids_per_q, gold_per_q)]
    return sum(hit) / len(hit), sum(rr) / len(rr)


def evaluate() -> dict:
    config = load_config()
    thresholds = yaml.safe_load(THRESHOLDS_PATH.read_text(encoding="utf-8"))["retrieval"]
    k = int(thresholds["k"])

    pipeline = RAGPipeline(config)
    chunks = pipeline.index_corpus()

    golden = load_golden()
    gold_per_q = [resolve_gold_ids(it, chunks) for it in golden]

    base_ids, rr_ids = [], []
    for it in golden:
        # Baseline is always pure dense search, regardless of retrieval_mode.
        dense = pipeline.store.search(pipeline.embedder.embed_query(it.question), config.top_k)
        base_ids.append([h.chunk.id for h in dense])
        # The pipeline path: hybrid fusion (if enabled) then rerank.
        candidates = pipeline._candidates(it.question)
        reranked = pipeline.reranker.rerank(it.question, candidates, config.top_k)
        rr_ids.append([h.chunk.id for h in reranked])

    base_hit, base_mrr = _score(base_ids, gold_per_q, k)
    rr_hit, rr_mrr = _score(rr_ids, gold_per_q, k)

    return {
        "n": len(golden),
        "k": k,
        "retrieval_mode": config.retrieval_mode,
        "rerank_backend": config.rerank_backend,
        "baseline": {"hit_at_k": base_hit, "mrr": base_mrr},
        "reranked": {"hit_at_k": rr_hit, "mrr": rr_mrr},
        "min_hit_at_k": float(thresholds["min_hit_at_k"]),
        "min_mrr": float(thresholds["min_mrr"]),
    }


def main() -> int:
    m = evaluate()
    b, r = m["baseline"], m["reranked"]
    print(f"retrieval eval  (n={m['n']}, k={m['k']}, "
          f"mode={m['retrieval_mode']}, rerank={m['rerank_backend']})")
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
