# Reranking

Vector search is fast but coarse. It returns a candidate set that is roughly relevant,
but the ordering inside that set is often wrong, especially when several chunks share
vocabulary with the query. A reranker takes the query and each candidate together and
scores them jointly, which is far more precise than comparing two independent vectors.

A cross-encoder reranker concatenates the query and a candidate passage and runs them
through a model that outputs a single relevance score. Because the model sees both
texts at once, it can tell that a passage merely mentions the query's keywords versus
actually answering the question. The tradeoff is cost: you run the model once per
candidate, so you only rerank the top handful from vector search rather than the whole
corpus.

Hosted rerank APIs such as Cohere Rerank give most of the benefit without hosting a
model yourself. The practical recipe is: over-retrieve with cheap vector search, then
rerank the top candidates down to the final few you feed the generator. In this repo,
turning on reranking is the change that moves mean reciprocal rank the most, because it
fixes exactly the case where the right chunk was retrieved but buried below lookalikes.
