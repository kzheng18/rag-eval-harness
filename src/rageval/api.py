"""Minimal FastAPI surface over the pipeline: POST /query -> answer + contexts."""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .config import load_config
from .pipeline import RAGPipeline

app = FastAPI(title="rageval", version="0.1.0")
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(load_config())
        _pipeline.index_corpus()
    return _pipeline


class QueryRequest(BaseModel):
    question: str


class ContextItem(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[ContextItem]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    result = get_pipeline().answer(req.question)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        contexts=[
            ContextItem(chunk_id=h.chunk.id, doc_id=h.chunk.doc_id, score=h.score, text=h.chunk.text)
            for h in result.contexts
        ],
    )
