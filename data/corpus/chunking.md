# Chunking

Chunking splits documents into passages small enough to embed and retrieve precisely.
Chunks that are too large dilute the embedding with unrelated content, so the vector no
longer points cleanly at any one idea. Chunks that are too small lose the surrounding
context needed to answer a question, so the generator sees fragments.

Overlap between adjacent chunks reduces the chance that the sentence answering a
question is split across a boundary. A common recipe is a few hundred characters per
chunk with a modest overlap. Stable chunk identifiers matter more than people expect:
if a chunk id changes every time you re-index, your evaluation set silently starts
pointing at the wrong passage. Content-addressed ids that change only when the text
changes make a stale golden set fail loudly instead.
