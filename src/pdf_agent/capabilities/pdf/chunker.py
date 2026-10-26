
from dataclasses import dataclass

from pdf_agent.capabilities.pdf.reader import PDFPage


@dataclass
class TextChunk:
    # 保存文本块内容及其来源页码
    text: str
    start_page: int
    end_page: int


def chunk_pages(
    pages: list[PDFPage],
    max_chars: int = 4000,
) -> list[TextChunk]:
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero.")

    chunks: list[TextChunk] = []
    current_text = ""
    start_page: int | None = None
    end_page: int | None = None

    def save_chunk() -> None:
        nonlocal current_text, start_page, end_page

        if current_text:
            chunks.append(
                TextChunk(
                    text=current_text,
                    start_page=start_page,
                    end_page=end_page,
                )
            )

        current_text = ""
        start_page = None
        end_page = None

    for page in pages:
        text = page.text.strip()

        if not text:
            continue

        # 超长单页单独切分，避免突破字符数限制
        if len(text) > max_chars:
            save_chunk()

            for offset in range(0, len(text), max_chars):
                chunks.append(
                    TextChunk(
                        text=text[offset:offset + max_chars],
                        start_page=page.page_number,
                        end_page=page.page_number,
                    )
                )

            continue

        combined_text = (
            f"{current_text}\n{text}" if current_text else text
        )

        if len(combined_text) > max_chars:
            save_chunk()
            current_text = text
            start_page = page.page_number
        else:
            if not current_text:
                start_page = page.page_number
            current_text = combined_text

        end_page = page.page_number

    save_chunk()

    return chunks
