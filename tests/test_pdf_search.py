import fitz
import pytest

from pdf_agent.pdf_search import PDFSearch, SearchResult


@pytest.fixture
def sample_pdf(tmp_path):
    file_path = tmp_path / "sample.pdf"

    with fitz.open() as document:
        first_page = document.new_page()
        first_page.insert_text(
            (72, 72),
            "Machine learning is useful.\nPython is popular.",
        )

        second_page = document.new_page()
        second_page.insert_text(
            (72, 72),
            "Deep learning is a branch of machine learning.",
        )

        third_page = document.new_page()
        third_page.insert_text(
            (72, 72),
            "Data analysis uses Python.",
        )

        document.save(file_path)

    return file_path


def test_search_keyword(sample_pdf):
    search = PDFSearch(sample_pdf)

    results = search.search("machine learning")

    assert len(results) == 2
    assert results[0].page_number == 1
    assert results[1].page_number == 2


def test_search_is_case_insensitive(sample_pdf):
    search = PDFSearch(sample_pdf)

    results = search.search("MACHINE LEARNING")

    assert len(results) == 2


def test_search_returns_page_text(sample_pdf):
    search = PDFSearch(sample_pdf)

    results = search.search("Python")

    assert len(results) == 2
    assert "Python is popular." in results[0].text
    assert "Data analysis uses Python." in results[1].text


def test_search_returns_search_result(sample_pdf):
    search = PDFSearch(sample_pdf)

    results = search.search("machine learning")

    assert isinstance(results[0], SearchResult)


def test_search_keyword_not_found(sample_pdf):
    search = PDFSearch(sample_pdf)

    results = search.search("artificial intelligence")

    assert results == []


def test_search_empty_keyword(sample_pdf):
    search = PDFSearch(sample_pdf)

    with pytest.raises(ValueError):
        search.search("")


def test_search_whitespace_keyword(sample_pdf):
    search = PDFSearch(sample_pdf)

    with pytest.raises(ValueError):
        search.search("   ")


@pytest.mark.parametrize("keyword", [123, None, True])
def test_search_keyword_must_be_string(sample_pdf, keyword):
    search = PDFSearch(sample_pdf)

    with pytest.raises(TypeError):
        search.search(keyword)