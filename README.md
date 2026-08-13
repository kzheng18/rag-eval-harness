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

## Architecture

```
query
  -> embed            (TF-IDF offline  |  sentence-transformers)
  -> vector search    (numpy cosine    |  Chroma  |  Pinecone)
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
src/rageval/     pipeline: chunking, embeddings, vector store, rerank, generate, api
data/corpus/     sample knowledge base
data/golden.jsonl  evaluation set: questions + relevant chunk ids
eval/            retrieval + generation eval harness and thresholds
tests/           unit tests
.github/workflows/ci.yml   runs tests + retrieval eval on every push
```
