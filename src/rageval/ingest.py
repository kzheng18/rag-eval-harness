"""Build the index from data/corpus and report basic stats."""
from __future__ import annotations

from .config import load_config
from .pipeline import RAGPipeline


def main() -> None:
    config = load_config()
    pipeline = RAGPipeline(config)
    chunks = pipeline.index_corpus()
    docs = len({c.doc_id for c in chunks})
    print(
        f"Indexed {len(chunks)} chunks from {docs} documents "
        f"(embedding={config.embedding_backend}, store={config.vector_store})."
    )


if __name__ == "__main__":
    main()
