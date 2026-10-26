
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
        summary = summarize_func(chunk)

        summaries.append(
            ChunkSummary(
                summary=summary,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
            )
        )

    return summaries
