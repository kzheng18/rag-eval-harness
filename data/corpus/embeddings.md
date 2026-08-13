# Embeddings

An embedding maps text into a dense vector so that semantically similar text lands
close together in vector space. In a retrieval system you embed every chunk of your
corpus once at index time, and embed the user's query at request time, then compare
them with cosine similarity.

The quality of retrieval is bounded by the quality of the embedding model. A weak
model produces vectors where unrelated passages sit close together, which shows up
downstream as retrieved context that looks on-topic but does not actually answer the
question. Normalizing vectors to unit length lets you use a plain dot product as
cosine similarity, which is cheaper than recomputing norms per query.

Dense embeddings capture meaning but lose exact-term signal. A query that hinges on a
rare identifier or an exact code symbol can retrieve poorly with dense vectors alone,
because that rare token is averaged away. This is the single biggest reason to keep a
lexical or reranking stage in the pipeline rather than trusting vector search on its
own.
