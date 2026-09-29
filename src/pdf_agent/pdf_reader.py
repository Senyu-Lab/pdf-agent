from pathlib import Path

import fitz


class PDFReaderError(Exception):
    pass


class PageNumberOutOfRangeError(PDFReaderError):
    pass


class PDFReader:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def _open_document(self) -> fitz.Document:
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {self.file_path}")

        if not self.file_path.is_file():
            raise PDFReaderError(f"Path is not a file: {self.file_path}")

        try:
            return fitz.open(self.file_path)
        except fitz.FileDataError as error:
            raise PDFReaderError(
                f"Unable to open PDF file: {self.file_path}"
            ) from error

    def get_page_count(self) -> int:
        with self._open_document() as document:
            return document.page_count

    def read_page_text(self, page_number: int) -> str:
        if isinstance(page_number, bool) or not isinstance(page_number, int):
            raise TypeError("Page number must be an integer.")

        with self._open_document() as document:
            if page_number < 1 or page_number > document.page_count:
                raise PageNumberOutOfRangeError(
                    f"Page number must be between 1 and {document.page_count}."
                )

            page = document.load_page(page_number - 1)
            return page.get_text()

    def read_all_text(self) -> str:
        with self._open_document() as document:
            return "\n".join(
                page.get_text()
                for page in document
            )