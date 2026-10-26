
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
) -> list[ChunkSummary]:
    summaries: list[ChunkSummary] = []

    for chunk in chunks:
        try:
            summary = summarize_func(chunk)
        except Exception as error:
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
