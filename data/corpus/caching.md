# Caching

Caching cuts latency and cost by reusing prior work. An embedding cache keyed by the
hash of the input text avoids re-embedding chunks that have not changed between index
runs, which matters when re-indexing a large corpus after editing a few documents.

A response cache keyed by the normalized query can short-circuit the whole pipeline for
repeated questions, but it is dangerous when the underlying corpus changes, because you
can serve a stale answer that no longer matches the documents. Cache invalidation
should be tied to the index version so that re-indexing transparently busts responses
built against the old index.
