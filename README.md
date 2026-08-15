# rageval

A small but complete RAG (retrieval-augmented generation) system built around one
idea: **you can't trust a RAG pipeline you can't measure.** The interesting part of
this repo isn't the retrieval, it's the evaluation harness around it and the fact
that it runs in CI, so a drop in quality fails the build like a broken test.

## Why this exists

It's easy to build a RAG demo that looks good when you type in a couple of questions
by hand. It's much harder to know whether it's *actually* good, or whether your last
change quietly made retrieval worse. This project treats retrieval quality as a
measurable, regression-tested property instead of something you eyeball.

Two layers of evaluation:

1. **Retrieval eval (offline, runs in CI with no API keys).** A golden set of
   `question -> relevant chunk` pairs, scored with `hit@k` and `MRR`. This is what
   catches the classic failure mode where vector similarity returns chunks that
   *look* relevant but are wrong. It also measures the effect of reranking.
2. **Generation eval (gated on API keys).** RAGAS / DeepEval metrics
   (faithfulness, answer relevancy, context precision) over the same golden set,
   run when an LLM judge is available.

## Hybrid retrieval (dense + lexical)

Reranking can only reorder the candidates dense vector search already returned. If the
right chunk never makes that pool -- the classic failure where a rare, discriminating
token (an error code, a symbol, an exact name) gets averaged away in the pooled
embedding -- no reranker can recover it.

`RETRIEVAL_MODE=hybrid` adds the missing half: a BM25 index over the *whole* corpus
retrieves on exact term overlap, and its ranking is fused with the dense ranking using
Reciprocal Rank Fusion (RRF), which combines by position rather than by raw score so
cosine similarities and BM25 scores never have to be made commensurable. The fused pool
then feeds the same reranker.

```bash
RETRIEVAL_MODE=hybrid python -m eval.retrieval_eval
```

It is off by default so `dense` stays the measured baseline. On the paraphrased golden
set -- where questions deliberately avoid reusing passage wording -- turning it on lifts
offline hit@5 from 0.938 to 1.000 and MRR from 0.646 to 0.729 (see `eval/RESULTS.md`),
because the full-corpus BM25 index reaches chunks carrying the exact needle term that
the paraphrased query pulls the dense vector away from. The gain grows on dense
embeddings (`sentence-transformers`), whose rare-token blind spot is sharper than
TF-IDF's. The fusion mechanism itself is regression-tested in `tests/test_hybrid.py`.

## Architecture

```
query
  -> embed            (TF-IDF offline  |  sentence-transformers)
  -> retrieve         dense vector search (numpy cosine | Chroma | Pinecone)
                      + optional hybrid: fuse with full-corpus lexical BM25 (RRF)
  -> rerank           (BM25 offline    |  Cohere  |  cross-encoder)
  -> generate         (extractive stub |  OpenAI / Cohere)
  -> answer + cited context
```

Every stage is pluggable via env vars so the same pipeline runs fully offline for
tests/CI and with hosted models in production. See `.env.example`.

## Quickstart (offline, no keys)

```bash
pip install -e ".[dev]"
python -m rageval.ingest            # build the index from data/corpus
python -m eval.retrieval_eval       # score retrieval, fail on regression
uvicorn rageval.api:app --reload    # serve /query
```

## Running with real models

```bash
pip install -e ".[local,providers,evals]"
cp .env.example .env                # set EMBEDDING_BACKEND=sentence-transformers, RERANK_BACKEND=cohere, etc.
python -m rageval.ingest
python -m eval.generation_eval      # RAGAS / DeepEval, needs an LLM key
```

## Results

See [`eval/RESULTS.md`](eval/RESULTS.md) for current retrieval numbers and the
measured effect of reranking.

## Layout

```
src/rageval/     pipeline: chunking, embeddings, vector store, lexical+RRF, rerank, generate, api
data/corpus/     sample knowledge base
data/golden.jsonl  evaluation set: questions + relevant chunk ids
eval/            retrieval + generation eval harness and thresholds
tests/           unit tests
.github/workflows/ci.yml   runs tests + retrieval eval on every push
```
