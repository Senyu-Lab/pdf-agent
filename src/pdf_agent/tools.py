from dataclasses import dataclass
from typing import Any, Callable
from pdf_agent.pdf_reader import PDFReader

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