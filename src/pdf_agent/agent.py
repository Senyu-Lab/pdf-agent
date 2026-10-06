from dataclasses import dataclass
from typing import Any

from pdf_agent.pdf_reader import PDFReader


@dataclass
class AgentResult:
    # 保存 Agent 最终给用户的回答
    answer: str

    # 记录 Agent 使用了哪个工具
    tool_name: str

    # 保存工具实际返回的数据
    tool_result: Any


class PDFAgent:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def run(self, question: str) -> AgentResult:
        # 创建 PDF 阅读工具
        reader = PDFReader(self.file_path)

        # 第一版 Agent 只处理 PDF 页数查询
        if "多少页" in question or "几页" in question:
            page_count = reader.get_page_count()

            return AgentResult(
                answer=f"这个 PDF 一共有 {page_count} 页。",
                tool_name="get_page_count",
                tool_result=page_count,
            )

        raise ValueError("Unsupported question.")