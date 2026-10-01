from dataclasses import dataclass

from pdf_agent.pdf_reader import PDFReader


@dataclass
class SearchResult:
    page_number: int
    text: str


class PDFSearch:
    def __init__(self, file_path: str):
        self.reader = PDFReader(file_path)

    def search(self, keyword: str) -> list[SearchResult]:
        if not isinstance(keyword, str):
            raise TypeError("Keyword must be a string.")

        if not keyword.strip():
            raise ValueError("Keyword must not be empty.")

        results = []

        for page_number in range(1, self.reader.get_page_count() + 1):
            text = self.reader.read_page_text(page_number)

            if keyword.casefold() in text.casefold():
                results.append(
                    SearchResult(
                        page_number=page_number,
                        text=text,
                    )
                )

        return results