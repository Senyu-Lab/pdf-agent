from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pdf_agent.capabilities.pdf.reader import PDFReader
from pdf_agent.capabilities.pdf.search import PDFSearch
from pdf_agent.capabilities.pdf.llm_summarizer import (
    combine_summaries_with_llm,
    summarize_chunk_with_llm,
)
from pdf_agent.capabilities.pdf.summarizer import summarize_document

@dataclass
class Tool:
    # Tool 的名称，LLM 会通过这个名称选择工具
    name: str

    # Tool 的功能描述，帮助 LLM 判断什么时候使用它
    description: str

    # Tool 的参数定义
    parameters: dict[str, Any]

    # 实际执行 Tool 的 Python 函数
    function: Callable[..., Any]

    def to_schema(self) -> dict[str, Any]:
        # 将 Tool 转换为 OpenAI Responses API 使用的 Schema
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_chat_schema(self) -> dict[str, Any]:
        # 将 Tool 转换为 OpenAI Chat Completions 和 Groq 使用的 Schema
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

def create_get_page_count_tool(file_path: str) -> Tool:
    reader = PDFReader(file_path)

    return Tool(
        name="get_page_count",
        description="Get the total number of pages in the PDF.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=reader.get_page_count,
    )

def create_extract_pdf_metadata_tool(file_path: str) -> Tool:
    reader = PDFReader(file_path)

    return Tool(
        name="extract_pdf_metadata",
        description=(
            "Extract metadata from the PDF, including title, author, "
            "subject, creator, producer, creation date, modification "
            "date, and page count."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        function=reader.get_metadata,
    )

def create_search_pdf_tool(file_path: str) -> Tool:
    search = PDFSearch(file_path)

    return Tool(
        name="search_pdf",
        description="Search for a keyword in the PDF and return matching pages.",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                    "description": "The keyword to search for in the PDF.",
                }
            },
            "required": ["keyword"],
        },
        function=search.search,
    )


def create_read_pdf_page_tool(file_path: str) -> Tool:
    reader = PDFReader(file_path)

    return Tool(
        name="read_pdf_page",
        description=(
            "Read the text content of a specific page in the PDF. "
            "Use this tool when the user asks to read or analyze a page."
        ),
        parameters={
            "type": "object",
            "properties": {
                "page_number": {
                    "type": "integer",
                    "description": "The page number to read, starting from 1.",
                }
            },
            "required": ["page_number"],
            "additionalProperties": False,
        },
        function=reader.read_page_text,
    )


def create_read_all_pdf_text_tool(file_path: str) -> Tool:
    reader = PDFReader(file_path)

    return Tool(
        name="read_all_pdf_text",
        description=(
            "Read the full text content of the PDF document. "
            "Use this tool when the user asks to summarize or analyze "
            "the entire document."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        function=reader.read_all_text,
    )

def create_summarize_pdf_tool(file_path: str, llm) -> Tool:
    reader = PDFReader(file_path)

    def summarize():
        pages = reader.read_pages()

        return summarize_document(
            pages=pages,
            summarize_func=lambda chunk: summarize_chunk_with_llm(
                llm, chunk
            ),
            combine_func=lambda summaries_text: combine_summaries_with_llm(
                llm, summaries_text
            ),
        )

    return Tool(
        name="summarize_pdf",
        description=(
            "Summarize the entire PDF document. "
            "Use this tool when the user asks for a document summary."
        ),
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        function=summarize,
    )
