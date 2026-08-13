"""The eval harness itself is regression-tested: current retrieval must clear the
floor defined in thresholds.yaml. If someone breaks retrieval, this test fails too."""
from eval.retrieval_eval import evaluate


def test_retrieval_clears_thresholds():
    m = evaluate()
    assert m["reranked"]["hit_at_k"] >= m["min_hit_at_k"]
    assert m["reranked"]["mrr"] >= m["min_mrr"]


def test_reranking_does_not_hurt():
    m = evaluate()
    assert m["reranked"]["mrr"] + 1e-9 >= m["baseline"]["mrr"]
