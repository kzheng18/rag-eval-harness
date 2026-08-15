# Evaluation results

## Retrieval (offline: TF-IDF embeddings + numpy cosine + BM25 rerank)

Reproducible with no API keys on a 7-document corpus and a 16-question golden set:

```
retrieval eval  (n=16, k=5, mode=dense, rerank=bm25)
  baseline   hit@k=0.875  MRR=0.644
  reranked   hit@k=0.938  MRR=0.646   (dMRR=+0.001)
  floor      hit@k>=0.80  MRR>=0.60
PASS
```

Reproduce:

```bash
pip install -e ".[dev]"
python -m eval.retrieval_eval
```

### Reading these numbers

- **hit@5** is whether the correct chunk appears in the top 5; **MRR** rewards putting
  it near the top, so MRR is the metric that moves when ordering improves.
- The golden questions are paraphrased so they do *not* reuse the wording of the
  passages they target -- a realistic test, and a hard one for a lexical embedder.
  Reranking alone barely helps here (dMRR=+0.001): BM25 can only reorder the candidates
  TF-IDF already returned, and it shares TF-IDF's blind spots.
- The thresholds in `thresholds.yaml` sit just under current performance, so a real
  regression fails CI while normal noise does not.

### Hybrid retrieval closes the gap

```
retrieval eval  (n=16, k=5, mode=hybrid, rerank=bm25)
  baseline   hit@k=0.875  MRR=0.644
  reranked   hit@k=1.000  MRR=0.729   (dMRR=+0.085)
  floor      hit@k>=0.80  MRR>=0.60
PASS
```

`RETRIEVAL_MODE=hybrid` adds a BM25 index over the *whole* corpus and fuses its ranking
with dense search via Reciprocal Rank Fusion before reranking. On this paraphrased set
it lifts hit@5 from 0.938 to a perfect 1.000 and MRR from 0.646 to 0.729 (+0.085 over
the dense baseline) -- a real gain, not noise, because the full-corpus lexical index
reaches chunks that carry the exact needle term even when the paraphrased query has
pulled the dense vector away from them. Reranking could never recover those: it only
reorders what dense already found. The mechanism is pinned by `tests/test_hybrid.py`.

## Retrieval (dense: sentence-transformers + Cohere rerank)

This path needs model downloads and a Cohere key, so the numbers are generated on your
own machine rather than checked in from a keyless environment:

```bash
pip install -e ".[local,providers]"
EMBEDDING_BACKEND=sentence-transformers VECTOR_STORE=chroma RERANK_BACKEND=cohere \
  python -m eval.retrieval_eval
```

Paste the output here once you've run it, so this file always reflects a real run.

## Generation (RAGAS: faithfulness + answer relevancy)

Gated on `OPENAI_API_KEY`. Skips cleanly without one.

```bash
pip install -e ".[evals,providers]"
OPENAI_API_KEY=... python -m eval.generation_eval
```
