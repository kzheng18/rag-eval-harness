# Evaluation results

## Retrieval (offline: TF-IDF embeddings + numpy cosine + BM25 rerank)

Reproducible with no API keys on a 7-document corpus and a 16-question golden set:

```
retrieval eval  (n=16, k=5, rerank=bm25)
  baseline   hit@k=0.875  MRR=0.644
  reranked   hit@k=0.938  MRR=0.656   (dMRR=+0.012)
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
- Reranking lifts hit@5 and gives a small MRR gain *on the offline lexical path*. The
  effect is modest here because BM25 and TF-IDF share a lexical signal. The larger win
  comes from a model-based reranker on dense embeddings (below), which is the case the
  offline path stands in for during CI.
- The thresholds in `thresholds.yaml` sit just under current performance, so a real
  regression fails CI while normal noise does not.

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
