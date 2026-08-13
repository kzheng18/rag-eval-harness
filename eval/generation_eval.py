"""Generation-quality evaluation with RAGAS (faithfulness, answer relevancy).

Unlike the retrieval eval, this needs a language model to act as judge, so it is gated
on OPENAI_API_KEY. With no key it prints a clear skip and exits 0, so it never blocks a
PR that has no secrets -- CI only runs it on main where the key is available.
"""
from __future__ import annotations

import os
import sys

from eval.common import load_golden
from rageval.config import load_config
from rageval.pipeline import RAGPipeline

MIN_FAITHFULNESS = 0.85
MIN_ANSWER_RELEVANCY = 0.80


def main() -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("SKIP: generation eval needs OPENAI_API_KEY (LLM judge). Nothing to do.")
        return 0

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        print("SKIP: install eval extras first:  pip install -e \".[evals]\"", file=sys.stderr)
        return 0

    config = load_config()
    pipeline = RAGPipeline(config)
    pipeline.index_corpus()

    rows = {"question": [], "answer": [], "contexts": []}
    for item in load_golden():
        result = pipeline.answer(item.question)
        rows["question"].append(item.question)
        rows["answer"].append(result.answer)
        rows["contexts"].append([h.chunk.text for h in result.contexts])

    scores = evaluate(Dataset.from_dict(rows), metrics=[faithfulness, answer_relevancy])
    df = scores.to_pandas()
    faith = float(df["faithfulness"].mean())
    rel = float(df["answer_relevancy"].mean())
    print(f"generation eval: faithfulness={faith:.3f} (min {MIN_FAITHFULNESS:.2f})  "
          f"answer_relevancy={rel:.3f} (min {MIN_ANSWER_RELEVANCY:.2f})")

    if faith < MIN_FAITHFULNESS or rel < MIN_ANSWER_RELEVANCY:
        print("FAIL: generation quality below threshold", file=sys.stderr)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
