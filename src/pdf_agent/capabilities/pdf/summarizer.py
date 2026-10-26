
from dataclasses import dataclass
from typing import Callable

from pdf_agent.capabilities.pdf.chunker import TextChunk


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
