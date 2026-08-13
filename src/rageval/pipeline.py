"""The RAG pipeline: embed -> vector search -> generate.

Reranking is added in a later step; for now retrieval is raw vector similarity so
we have a baseline to measure against.
"""
from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk, load_corpus
from .config import CORPUS_DIR, Config
from .embeddings import build_embedder
from .generate import build_generator
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

    def index(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, vectors)

    def index_corpus(self) -> list[Chunk]:
        chunks = load_corpus(CORPUS_DIR, self.config.chunk_size, self.config.chunk_overlap)
        self.index(chunks)
        return chunks

    def retrieve(self, question: str) -> list[Hit]:
        qv = self.embedder.embed_query(question)
        candidates = self.store.search(qv, self.config.top_k)  # over-retrieve
        return self.reranker.rerank(question, candidates, self.config.final_k)

    def answer(self, question: str) -> RAGAnswer:
        hits = self.retrieve(question)
        text = self.generator.generate(question, hits)
        return RAGAnswer(question=question, answer=text, contexts=hits)
