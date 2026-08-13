from rageval.chunking import chunk_text, load_corpus
from rageval.config import CORPUS_DIR


def test_chunk_ids_are_stable_and_deterministic():
    text = "alpha beta gamma delta epsilon zeta eta theta " * 40
    a = chunk_text(text, "doc", chunk_size=200, overlap=40)
    b = chunk_text(text, "doc", chunk_size=200, overlap=40)
    assert [c.id for c in a] == [c.id for c in b]
    assert len(a) > 1


def test_chunk_id_changes_when_text_changes():
    a = chunk_text("the quick brown fox", "d", 200, 0)[0]
    b = chunk_text("the quick brown cat", "d", 200, 0)[0]
    assert a.id != b.id  # content-addressed


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", "d", 200, 0) == []


def test_corpus_loads():
    chunks = load_corpus(CORPUS_DIR, 600, 100)
    assert len(chunks) > 5
    assert all(c.text for c in chunks)
