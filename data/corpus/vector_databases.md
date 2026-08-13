# Vector databases

A vector database stores embeddings and answers nearest-neighbor queries over them.
Exact search compares the query against every stored vector, which is fine for a few
thousand chunks but does not scale. Approximate nearest neighbor indexes such as HNSW
trade a small amount of recall for a large speedup by navigating a graph of vectors
instead of scanning all of them.

Choosing a distance metric matters. Cosine similarity is standard for normalized text
embeddings; if you forget to normalize, an L2 index will rank longer passages
differently than you expect. The specific failure mode to watch for is high-similarity
neighbors that share surface vocabulary with the query but come from the wrong
document, because the index only knows about geometry, not about whether the passage
actually answers anything.

Persistence and metadata filtering are the other axes. Being able to attach a document
id and restrict search to a subset is what lets you scope retrieval to, say, a single
user's files without re-indexing.
