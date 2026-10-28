
from unittest.mock import Mock

from pdf_agent.capabilities.pdf.chunker import TextChunk
from pdf_agent.capabilities.pdf.llm_summarizer import (
    combine_summaries_with_llm,
    summarize_chunk_with_llm,
)
from pdf_agent.capabilities.pdf.reader import PDFPage
from pdf_agent.capabilities.pdf.summarizer import summarize_document


def test_summarize_chunk_with_llm():
    llm = Mock()
    llm.generate_text.return_value = "第一段摘要"

    chunk = TextChunk(
        text="这是 PDF 第一页的原文内容。",
        start_page=1,
        end_page=1,
    )

    result = summarize_chunk_with_llm(llm, chunk)

    assert result == "第一段摘要"
    llm.generate_text.assert_called_once()

    prompt = llm.generate_text.call_args.args[0]
    assert "这是 PDF 第一页的原文内容。" in prompt
    assert "1" in prompt


def test_combine_summaries_with_llm():
    llm = Mock()
    llm.generate_text.return_value = "整份文档的最终摘要"

    text = "Pages 1-2:\n第一段摘要"

    result = combine_summaries_with_llm(llm, text)

    assert result == "整份文档的最终摘要"
    llm.generate_text.assert_called_once()

    prompt = llm.generate_text.call_args.args[0]
    assert text in prompt


def test_summarize_document_with_llm_adapters():
    llm = Mock()
    llm.generate_text.side_effect = [
        "第一段摘要",
        "第二段摘要",
        "整份文档的最终摘要",
    ]

    pages = [
        PDFPage(page_number=1, text="第一段原文内容"),
        PDFPage(page_number=2, text="第二段原文内容"),
    ]

    result = summarize_document(
        pages=pages,
        summarize_func=lambda chunk: summarize_chunk_with_llm(llm, chunk),
        combine_func=lambda text: combine_summaries_with_llm(llm, text),
        max_chars=10,
    )

    assert result == "整份文档的最终摘要"
    assert llm.generate_text.call_count == 3

    final_prompt = llm.generate_text.call_args_list[-1].args[0]
    assert "第一段摘要" in final_prompt
    assert "第二段摘要" in final_prompt
