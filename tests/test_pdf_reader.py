import fitz
import pytest

from pdf_agent.pdf_reader import (
    PDFReader,
    PDFReaderError,
    PageNumberOutOfRangeError,
)


@pytest.fixture
def sample_pdf(tmp_path):
    file_path = tmp_path / "sample.pdf"

    with fitz.open() as document:
        first_page = document.new_page()
        first_page.insert_text((72, 72), "First page")

        second_page = document.new_page()
        second_page.insert_text((72, 72), "Second page")

        document.save(file_path)

    return file_path


def test_get_page_count(sample_pdf):
    reader = PDFReader(sample_pdf)

    assert reader.get_page_count() == 2


def test_read_page_text(sample_pdf):
    reader = PDFReader(sample_pdf)

    assert "First page" in reader.read_page_text(1)
    assert "Second page" in reader.read_page_text(2)


def test_read_all_text(sample_pdf):
    reader = PDFReader(sample_pdf)

    text = reader.read_all_text()

    assert "First page" in text
    assert "Second page" in text


def test_file_not_found(tmp_path):
    reader = PDFReader(tmp_path / "missing.pdf")

    with pytest.raises(FileNotFoundError):
        reader.get_page_count()


@pytest.mark.parametrize("page_number", [0, 3])
def test_page_number_out_of_range(sample_pdf, page_number):
    reader = PDFReader(sample_pdf)

    with pytest.raises(PageNumberOutOfRangeError):
        reader.read_page_text(page_number)


@pytest.mark.parametrize("page_number", [1.5, "1", True])
def test_page_number_must_be_integer(sample_pdf, page_number):
    reader = PDFReader(sample_pdf)

    with pytest.raises(TypeError):
        reader.read_page_text(page_number)


def test_invalid_pdf(tmp_path):
    file_path = tmp_path / "invalid.pdf"
    file_path.write_text("This is not a PDF file.", encoding="utf-8")

    reader = PDFReader(file_path)

    with pytest.raises(PDFReaderError):
        reader.get_page_count()