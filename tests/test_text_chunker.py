
import pytest

from pdf_agent.capabilities.pdf.chunker import chunk_pages
from pdf_agent.capabilities.pdf.reader import PDFPage


def test_chunk_pages_combines_pages_within_character_limit():
    pages = [
        PDFPage(page_number=1, text="A" * 4),
        PDFPage(page_number=2, text="B" * 4),
    ]

    chunks = chunk_pages(pages, max_chars=10)

    assert len(chunks) == 1
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 2
    assert "AAAA" in chunks[0].text
    assert "BBBB" in chunks[0].text


def test_chunk_pages_splits_when_character_limit_is_exceeded():
    pages = [
        PDFPage(page_number=1, text="A" * 6),
        PDFPage(page_number=2, text="B" * 6),
    ]

    chunks = chunk_pages(pages, max_chars=10)

    assert len(chunks) == 2
    assert chunks[0].start_page == 1
    assert chunks[0].end_page == 1
    assert chunks[1].start_page == 2
    assert chunks[1].end_page == 2


def test_chunk_pages_splits_a_single_oversized_page():
    pages = [
        PDFPage(page_number=3, text="A" * 12),
    ]

    chunks = chunk_pages(pages, max_chars=5)

    assert [len(chunk.text) for chunk in chunks] == [5, 5, 2]
    assert all(chunk.start_page == 3 for chunk in chunks)
    assert all(chunk.end_page == 3 for chunk in chunks)


def test_chunk_pages_skips_empty_pages():
    pages = [
        PDFPage(page_number=1, text=""),
        PDFPage(page_number=2, text="Useful text"),
        PDFPage(page_number=3, text="   "),
    ]

    chunks = chunk_pages(pages, max_chars=20)

    assert len(chunks) == 1
    assert chunks[0].start_page == 2
    assert chunks[0].end_page == 2
    assert chunks[0].text == "Useful text"


@pytest.mark.parametrize("max_chars", [0, -1])
def test_chunk_pages_rejects_non_positive_limit(max_chars):
    with pytest.raises(ValueError):
        chunk_pages([], max_chars=max_chars)
