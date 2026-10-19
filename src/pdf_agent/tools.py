from dataclasses import dataclass
from typing import Any, Callable
from pdf_agent.capabilities.pdf.reader import PDFReader
from pdf_agent.capabilities.pdf.search import PDFSearch


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


