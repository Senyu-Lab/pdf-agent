
from pdf_agent.capabilities.pdf.chunker import TextChunk
from pdf_agent.capabilities.pdf.reader import PDFPage
from pdf_agent.capabilities.pdf.summarizer import (
    ChunkSummary,
    summarize_chunks,
    ChunkSummaryError,
    reduce_summaries,
    summarize_document,
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


import pytest


def test_summarize_chunks_reports_page_range_when_summary_fails():
    chunks = [
        TextChunk(
            text="First section",
            start_page=1,
            end_page=2,
        ),
        TextChunk(
            text="Second section",
            start_page=3,
            end_page=4,
        ),
    ]

    def fake_summarize(chunk):
        if chunk.start_page == 3:
            raise RuntimeError("Simulated LLM failure")
        return f"Summary: {chunk.text}"

    with pytest.raises(ChunkSummaryError) as exc_info:
        summarize_chunks(chunks, fake_summarize)

    assert exc_info.value.start_page == 3
    assert exc_info.value.end_page == 4
    assert "Simulated LLM failure" in str(exc_info.value)


def test_summarize_chunks_retries_after_temporary_failure():
    chunks = [
        TextChunk(
            text="Section content",
            start_page=1,
            end_page=2,
        ),
    ]
    attempts = 0

    def flaky_summarize(chunk):
        nonlocal attempts
        attempts += 1

        if attempts < 3:
            raise RuntimeError("Temporary failure")

        return f"Summary: {chunk.text}"

    summaries = summarize_chunks(
        chunks,
        flaky_summarize,
        max_retries=2,
    )

    assert attempts == 3
    assert len(summaries) == 1
    assert summaries[0].summary == "Summary: Section content"


def test_summarize_chunks_fails_after_retries_are_exhausted():
    chunks = [
        TextChunk(
            text="Section content",
            start_page=5,
            end_page=6,
        ),
    ]
    attempts = 0

    def always_fails(chunk):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("Persistent failure")

    with pytest.raises(ChunkSummaryError) as exc_info:
        summarize_chunks(
            chunks,
            always_fails,
            max_retries=2,
        )

    assert attempts == 3
    assert exc_info.value.start_page == 5
    assert exc_info.value.end_page == 6
    assert "Persistent failure" in str(exc_info.value)




def test_reduce_summaries_combines_chunk_summaries_in_order():
    summaries = [
        ChunkSummary(
            summary="Introduction and background",
            start_page=1,
            end_page=3,
        ),
        ChunkSummary(
            summary="Methods and experiments",
            start_page=4,
            end_page=7,
        ),
        ChunkSummary(
            summary="Results and conclusions",
            start_page=8,
            end_page=10,
        ),
    ]

    received_text = ""

    def fake_combine(text):
        nonlocal received_text
        received_text = text
        return "Final document summary"

    result = reduce_summaries(summaries, fake_combine)

    assert result == "Final document summary"
    assert "Pages 1-3" in received_text
    assert "Introduction and background" in received_text
    assert "Pages 4-7" in received_text
    assert "Methods and experiments" in received_text
    assert "Pages 8-10" in received_text
    assert "Results and conclusions" in received_text


def test_reduce_summaries_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        reduce_summaries([], lambda text: "Final summary")




def test_summarize_document_runs_complete_pipeline():
    pages = [
        PDFPage(page_number=1, text="Introduction content"),
        PDFPage(page_number=2, text="Methods content"),
        PDFPage(page_number=3, text="Results content"),
    ]

    summarized_chunks = []

    def fake_summarize(chunk):
        summarized_chunks.append(chunk)
        return f"Summary of: {chunk.text}"

    def fake_combine(text):
        return f"Final summary:\n{text}"

    result = summarize_document(
        pages=pages,
        summarize_func=fake_summarize,
        combine_func=fake_combine,
        max_chars=30,
    )

    assert result.startswith("Final summary:")
    assert len(summarized_chunks) == 3
    assert "Introduction content" in result
    assert "Methods content" in result
    assert "Results content" in result
    assert "Pages 1-1" in result
    assert "Pages 2-2" in result
    assert "Pages 3-3" in result

