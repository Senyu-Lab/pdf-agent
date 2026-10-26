
from pdf_agent.capabilities.pdf.chunker import TextChunk
from pdf_agent.capabilities.pdf.summarizer import (
    ChunkSummary,
    summarize_chunks,
)


def test_summarize_chunks_summarizes_each_chunk_in_order():
    chunks = [
        TextChunk(
            text="First section content",
            start_page=1,
            end_page=2,
        ),
        TextChunk(
            text="Second section content",
            start_page=3,
            end_page=4,
        ),
    ]

    def fake_summarize(chunk):
        return f"Summary: {chunk.text}"

    summaries = summarize_chunks(chunks, fake_summarize)

    assert len(summaries) == 2
    assert summaries[0] == ChunkSummary(
        summary="Summary: First section content",
        start_page=1,
        end_page=2,
    )
    assert summaries[1] == ChunkSummary(
        summary="Summary: Second section content",
        start_page=3,
        end_page=4,
    )


def test_summarize_chunks_returns_empty_list_for_no_chunks():
    summaries = summarize_chunks([], lambda chunk: "Summary")

    assert summaries == []
