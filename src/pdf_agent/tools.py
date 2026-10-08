from dataclasses import dataclass
from typing import Any, Callable
from pdf_agent.pdf_reader import PDFReader
from pdf_agent.pdf_search import PDFSearch


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


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools = {}

        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        # 防止重复注册相同名称的 Tool
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as error:
            raise KeyError(f"Tool not found: {name}") from error

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())