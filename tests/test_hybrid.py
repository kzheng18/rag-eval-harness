"""Hybrid retrieval: full-corpus lexical BM25 + Reciprocal Rank Fusion.

The point of hybrid search is to reach a chunk that dense vector search never puts in
its candidate pool -- the classic rare-token blind spot. A reranker cannot fix that,
because it only reorders what dense already returned. These tests pin the mechanism
that makes hybrid able to fix it.
"""
from rageval.chunking import Chunk
from rageval.config import load_config
from rageval.lexical import LexicalIndex, reciprocal_rank_fusion
from rageval.pipeline import RAGPipeline
from rageval.vectorstore import Hit


def _chunk(i, text):
    return Chunk(id=f"c{i}", doc_id="d", text=text, ordinal=i)


def _corpus():
    return [
        _chunk(0, "caching avoids recomputing embeddings for unchanged documents"),
        _chunk(1, "the retriever raised error code E4093 on a malformed request"),
        _chunk(2, "vector databases store embeddings and answer nearest neighbor queries"),
    ]


def test_lexical_index_finds_rare_token():
    idx = LexicalIndex()
    idx.add(_corpus())
    hits = idx.search("what causes E4093?", top_k=2)
    assert hits[0].chunk.id == "c1"


def test_lexical_index_empty_corpus_returns_nothing():
    idx = LexicalIndex()
    idx.add([])
    assert idx.search("anything", top_k=5) == []


def test_rrf_rewards_agreement_across_lists():
    a, b, c = _corpus()
    # 'b' is ranked in both lists; 'a' only tops the dense list; 'c' only tops lexical.
    dense = [Hit(a, 9.0), Hit(b, 1.0)]
    lexical = [Hit(c, 9.0), Hit(b, 1.0)]
    fused = reciprocal_rank_fusion([dense, lexical], k=60)
    assert fused[0].chunk.id == "c1"  # 'b' wins on agreement
    assert {h.chunk.id for h in fused} == {"c0", "c1", "c2"}


def test_rrf_recovers_a_chunk_dense_missed():
    a, b, c = _corpus()
    # Dense misses the rare-token chunk 'b' entirely; lexical ranks it first.
    dense_pool = [Hit(a, 0.8), Hit(c, 0.7)]
    lexical_pool = [Hit(b, 5.0), Hit(a, 0.1)]
    fused = reciprocal_rank_fusion([dense_pool, lexical_pool], k=60, top_n=3)
    assert "c1" in {h.chunk.id for h in fused}  # unreachable without hybrid


def test_rrf_top_n_truncates():
    a, b, c = _corpus()
    fused = reciprocal_rank_fusion([[Hit(a, 1), Hit(b, 1), Hit(c, 1)]], k=60, top_n=2)
    assert len(fused) == 2


def test_hybrid_pipeline_indexes_and_retrieves():
    import os

    os.environ["RETRIEVAL_MODE"] = "hybrid"
    try:
        pipeline = RAGPipeline(load_config())
        assert pipeline.hybrid and pipeline.lexical is not None
        pipeline.index_corpus()
        hits = pipeline.retrieve("what structure speeds up nearest neighbor search?")
        assert any(h.chunk.doc_id == "vector_databases" for h in hits)
        assert 1 <= len(hits) <= pipeline.config.final_k
    finally:
        del os.environ["RETRIEVAL_MODE"]


def test_dense_mode_builds_no_lexical_index():
    pipeline = RAGPipeline(load_config())  # default RETRIEVAL_MODE=dense
    assert not pipeline.hybrid
    assert pipeline.lexical is None
