from rageval.chunking import Chunk
from rageval.rerank import BM25Reranker, NoopReranker, build_reranker
from rageval.vectorstore import Hit


def _hits():
    texts = [
        "caching avoids recomputing embeddings for unchanged documents",
        "a cross encoder scores the query and passage together for precise ranking",
        "vector databases store embeddings and answer nearest neighbor queries",
    ]
    return [Hit(Chunk(id=f"c{i}", doc_id="d", text=t, ordinal=i), score=0.1) for i, t in enumerate(texts)]


def test_bm25_puts_the_on_topic_passage_first():
    hits = _hits()
    out = BM25Reranker().rerank("how does a cross encoder rank a passage?", hits, top_n=3)
    assert out[0].chunk.id == "c1"


def test_noop_is_passthrough_and_respects_top_n():
    hits = _hits()
    out = NoopReranker().rerank("anything", hits, top_n=2)
    assert [h.chunk.id for h in out] == ["c0", "c1"]


def test_build_reranker_rejects_unknown():
    try:
        build_reranker("nope")
        assert False, "expected ValueError"
    except ValueError:
        pass
