from dataclasses import dataclass
from typing import Any

from pdf_agent.mock_llm_client import MockLLMClient
from pdf_agent.llm_client import LLMClient
from pdf_agent.tool_registry import ToolRegistry
from pdf_agent.tools import (
    create_get_page_count_tool,
    create_search_pdf_tool,
)


@dataclass
class AgentResult:
    # 保存 Agent 最终给用户的回答
    answer: str

    # 记录 Agent 使用了哪个工具
    tool_name: str

    # 保存工具实际返回的数据
    tool_result: Any


class PDFAgent:
    def __init__(
            self,
            file_path: str,
            llm: LLMClient | None = None,
    ):
        self.file_path = file_path

        self.registry = ToolRegistry([
            create_get_page_count_tool(file_path),
            create_search_pdf_tool(file_path),
        ])

        # 允许外部传入 LLM，未指定时使用 Mock LLM
        self.llm = llm or MockLLMClient()

    def run(self, question: str) -> AgentResult:
        # 将用户问题交给 LLM，由 LLM 决定调用哪个 Tool
        tool_call = self.llm.chat(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            tools=self.registry.get_schemas(),
        )

        # 根据 LLM 返回的 Tool 名称找到对应 Tool
        tool = self.registry.get(tool_call.tool_name)

        # 执行 Tool，并传入 LLM 提供的参数
        tool_result = tool.function(**tool_call.arguments)

        # 将 Tool Result 交给 LLM 生成最终回答
        answer = self.llm.generate_answer(
            messages=[
                {
                    "role": "user",
                    "content": question,
                }
            ],
            tool_name=tool_call.tool_name,
            tool_result=tool_result,
        )

        return AgentResult(
            answer=answer,
            tool_name=tool_call.tool_name,
            tool_result=tool_result,
        )
