from typing import Any

from pdf_agent.llm_client import LLMClient, ToolCall


class MockLLMClient(LLMClient):
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> ToolCall:
        # Mock LLM 根据问题模拟 Tool Selection
        question = messages[-1]["content"]

        if "多少页" in question or "几页" in question:
            return ToolCall(
                tool_name="get_page_count",
                arguments={},
            )

        if "有没有" in question or "是否有" in question:
            return ToolCall(
                tool_name="search_pdf",
                arguments={
                    "keyword": "Python",
                },
            )

        raise ValueError(
            "Mock LLM cannot determine which tool to use."
        )

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        tool_name: str,
        tool_result: Any,
    ) -> str:
        # Mock LLM 根据 Tool Result 模拟最终回答
        if tool_name == "get_page_count":
            return f"这个 PDF 一共有 {tool_result} 页。"

        if tool_name == "search_pdf":
            if not tool_result:
                return "没有找到相关内容。"

            pages = [
                result.page_number
                for result in tool_result
            ]

            page_text = "、".join(
                str(page)
                for page in pages
            )

            return f"在第 {page_text} 页找到了相关内容。"

        raise ValueError(
            f"Unsupported tool result: {tool_name}"
        )