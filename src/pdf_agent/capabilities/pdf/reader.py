from dataclasses import dataclass
from pathlib import Path

import fitz


@dataclass
class PDFPage:
    page_number: int
    text: str

class PDFReaderError(Exception):
    pass


class PageNumberOutOfRangeError(PDFReaderError):
    pass


class PDFReader:
    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def _open_document(self) -> fitz.Document:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"PDF file not found: {self.file_path}"
            )

        if not self.file_path.is_file():
            raise PDFReaderError(
                f"Path is not a file: {self.file_path}"
            )

        try:
            return fitz.open(self.file_path)
        except fitz.FileDataError as error:
            raise PDFReaderError(
                f"Unable to open PDF file: {self.file_path}"
            ) from error

    def get_page_count(self) -> int:
        with self._open_document() as document:
            return document.page_count

    def get_metadata(self) -> dict[str, str | int]:
        with self._open_document() as document:
            metadata = document.metadata or {}

            return {
                "title": metadata.get("title") or "",
                "author": metadata.get("author") or "",
                "subject": metadata.get("subject") or "",
                "creator": metadata.get("creator") or "",
                "producer": metadata.get("producer") or "",
                "creation_date": metadata.get("creationDate") or "",
                "modification_date": metadata.get("modDate") or "",
                "page_count": document.page_count,
            }

    def read_page_text(self, page_number: int) -> str:
        if isinstance(page_number, bool) or not isinstance(
            page_number, int
        ):
            raise TypeError("Page number must be an integer.")

        with self._open_document() as document:
            if page_number < 1 or page_number > document.page_count:
                raise PageNumberOutOfRangeError(
                    f"Page number must be between 1 and {document.page_count}."
                )

            page = document.load_page(page_number - 1)
            return page.get_text()

    def read_page_range(
            self,
            start_page: int,
            end_page: int,
    ) -> list[PDFPage]:
        if (
                isinstance(start_page, bool)
                or not isinstance(start_page, int)
                or isinstance(end_page, bool)
                or not isinstance(end_page, int)
        ):
            raise TypeError("Page numbers must be integers.")

        with self._open_document() as document:
            if (
                    start_page < 1
                    or end_page > document.page_count
                    or start_page > end_page
            ):
                raise PageNumberOutOfRangeError(
                    f"Page range must be between 1 and "
                    f"{document.page_count}, with start_page "
                    f"less than or equal to end_page."
                )

            return [
                PDFPage(
                    page_number=page_number,
                    text=document.load_page(page_number - 1).get_text(),
                )
                for page_number in range(start_page, end_page + 1)
            ]

    def read_pages(self) -> list[PDFPage]:
        with self._open_document() as document:
            return [
                PDFPage(
                    page_number=page_number,
                    text=page.get_text(),
                )
                for page_number, page in enumerate(document, start=1)
            ]

    def read_all_text(self) -> str:
        with self._open_document() as document:
            return "\n".join(
                page.get_text()
                for page in document
            )
