"""The RAG pipeline: retrieve -> rerank -> generate.

Retrieval has two modes (config.retrieval_mode):

- "dense":  raw vector similarity over the store. The baseline.
- "hybrid": dense vector search *and* full-corpus lexical BM25, fused with Reciprocal
            Rank Fusion before reranking. This widens the candidate pool so a chunk
            that dense search misses on a rare token can still be reached -- something
            reranking alone can never fix, since it only reorders what dense returned.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk, load_corpus
from .config import CORPUS_DIR, Config
from .embeddings import build_embedder
from .generate import build_generator
from .lexical import LexicalIndex, reciprocal_rank_fusion
from .rerank import build_reranker
from .vectorstore import Hit, build_store


@dataclass
class RAGAnswer:
    question: str
    answer: str
    contexts: list[Hit]


class RAGPipeline:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.embedder = build_embedder(config.embedding_backend)
        self.store = build_store(config.vector_store)
        self.reranker = build_reranker(config.rerank_backend, cohere_api_key=config.cohere_api_key)
        self.generator = build_generator(config.openai_api_key)
        self.hybrid = config.retrieval_mode == "hybrid"
        self.lexical = LexicalIndex() if self.hybrid else None

    def index(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)
        if self.lexical is not None:
            self.lexical.add(chunks)

    def index_corpus(self) -> list[Chunk]:
        chunks = load_corpus(CORPUS_DIR, self.config.chunk_size, self.config.chunk_overlap)
        self.index(chunks)
        return chunks

    def _candidates(self, question: str) -> list[Hit]:
        """Candidate pool handed to the reranker.

        Dense-only unless retrieval_mode="hybrid", in which case dense and lexical
        candidate lists are fused with RRF so both signals get a vote.
        """
        dense = self.store.search(self.embedder.embed_query(question), self.config.top_k)
        if self.lexical is None:
            return dense
        lexical = self.lexical.search(question, self.config.top_k)
        return reciprocal_rank_fusion([dense, lexical], k=self.config.rrf_k)

    def retrieve(self, question: str) -> list[Hit]:
        candidates = self._candidates(question)
        return self.reranker.rerank(question, candidates, self.config.final_k)

    def answer(self, question: str) -> RAGAnswer:
        hits = self.retrieve(question)
        text = self.generator.generate(question, hits)
        return RAGAnswer(question=question, answer=text, contexts=hits)
