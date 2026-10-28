
from collections.abc import Callable
from dataclasses import dataclass

from pdf_agent.capabilities.pdf.chunker import TextChunk, chunk_pages
from pdf_agent.capabilities.pdf.reader import PDFPage


@dataclass
class ChunkSummary:
    # 保存单个文本块的摘要及其来源页码
    summary: str
    start_page: int
    end_page: int



def summarize_chunks(
    chunks: list[TextChunk],
    summarize_func: Callable[[TextChunk], str],
    max_retries: int = 2,
) -> list[ChunkSummary]:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    summaries: list[ChunkSummary] = []

    for chunk in chunks:
        for attempt in range(max_retries + 1):
            try:
                summary = summarize_func(chunk)
                break
            except Exception as error:
                if attempt == max_retries:
                    raise ChunkSummaryError(
                        start_page=chunk.start_page,
                        end_page=chunk.end_page,
                        cause=error,
                    ) from error

        summaries.append(
            ChunkSummary(
                summary=summary,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
            )
        )

    return summaries

def reduce_summaries(
        summaries: list[ChunkSummary],
        combine_func: Callable[[str], str],
) -> str:
    if not summaries:
        raise ValueError("Cannot reduce an empty list of summaries.")

    # 保留每段摘要的来源页码，方便追溯原文
    parts = [
        f"Pages {item.start_page}-{item.end_page}:\n{item.summary}"
        for item in summaries
    ]

    combined_text = "\n\n".join(parts)

    return combine_func(combined_text)


class ChunkSummaryError(Exception):
    # 记录摘要失败的页码范围和原始异常
    def __init__(
        self,
        start_page: int,
        end_page: int,
        cause: Exception,
    ):
        self.start_page = start_page
        self.end_page = end_page
        self.cause = cause

        super().__init__(
            f"Failed to summarize pages {start_page}-{end_page}: {cause}"
        )



def summarize_document(
    pages: list[PDFPage],
    summarize_func: Callable[[TextChunk], str],
    combine_func: Callable[[str], str],
    max_chars: int = 4000,
    max_retries: int = 2,
) -> str:
    # 第一步：将页面文本切分成多个文本块
    chunks = chunk_pages(pages, max_chars=max_chars)

    if not chunks:
        raise ValueError("Cannot summarize a document with no text.")

    # 第二步：逐块生成摘要
    summaries = summarize_chunks(
        chunks,
        summarize_func,
        max_retries=max_retries,
    )

    # 第三步：合并所有局部摘要
    return reduce_summaries(summaries, combine_func)
