from rageval.config import load_config
from rageval.pipeline import RAGPipeline


def test_pipeline_indexes_and_answers():
    pipeline = RAGPipeline(load_config())
    pipeline.index_corpus()
    result = pipeline.answer("what does mean reciprocal rank reward?")
    assert result.answer  # extractive stub returns something
    assert 1 <= len(result.contexts) <= load_config().final_k
    assert all(h.chunk.text for h in result.contexts)


def test_relevant_chunk_is_retrieved():
    pipeline = RAGPipeline(load_config())
    pipeline.index_corpus()
    result = pipeline.answer("what structure speeds up nearest neighbor search?")
    assert any(h.chunk.doc_id == "vector_databases" for h in result.contexts)
