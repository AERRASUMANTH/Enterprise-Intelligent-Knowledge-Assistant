import pytest

from app.services.chunker import chunk_pages, split_text
from app.services.document_loader import PageText


def test_short_text_is_single_chunk():
    assert split_text("hello world", chunk_size=100, chunk_overlap=10) == ["hello world"]


def test_empty_text_gives_no_chunks():
    assert split_text("   ", chunk_size=100, chunk_overlap=10) == []


def test_chunks_respect_size_and_overlap():
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    chunks = split_text(text, chunk_size=100, chunk_overlap=30)

    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # Consecutive chunks share text (the overlap).
    for first, second in zip(chunks, chunks[1:]):
        assert first.split()[-1] in second
    # No word is lost.
    assert set(words) <= set(" ".join(chunks).split())


def test_words_are_not_cut_in_half():
    text = " ".join(["abcdefghij"] * 50)
    for chunk in split_text(text, chunk_size=64, chunk_overlap=16):
        assert all(word == "abcdefghij" for word in chunk.split())


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_settings_raise(size, overlap):
    with pytest.raises(ValueError):
        split_text("some text", chunk_size=size, chunk_overlap=overlap)


def test_chunk_pages_keeps_metadata():
    pages = [PageText("a " * 300, page=1), PageText("b " * 300, page=2)]
    chunks = chunk_pages(pages, "report.pdf", chunk_size=200, chunk_overlap=20)

    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "report.pdf" for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Chunks never mix text from two pages.
    assert all(("a" in c.text) != ("b" in c.text) for c in chunks)
