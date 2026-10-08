from dataclasses import dataclass
from typing import Any
import json

from openai import OpenAI


@dataclass
class ToolCall:
    # LLM 决定调用的 Tool 名称
    tool_name: str

    # LLM 提供给 Tool 的参数
    arguments: dict[str, Any]


class LLMClient:
    def __init__(self):
        # OpenAI SDK 会自动读取 OPENAI_API_KEY 环境变量
        self.client = OpenAI()

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> ToolCall:
        # 调用 OpenAI Responses API，让模型决定使用哪个 Tool
        response = self.client.responses.create(
            model="gpt-5-mini",
            input=messages,
            tools=tools,
        )

        # 查找模型返回的 function call
        for item in response.output:
            if item.type == "function_call":
                return ToolCall(
                    tool_name=item.name,
                    arguments=json.loads(item.arguments),
                )

        raise ValueError("LLM did not return a tool call.")

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        tool_name: str,
        tool_result: Any,
    ) -> str:
        # 将 Tool Result 转换为最终用户回答
        response = self.client.responses.create(
            model="gpt-5-mini",
            input=[
                *messages,
                {
                    "role": "developer",
                    "content": (
                        f"The tool '{tool_name}' was executed "
                        f"and returned this result: {tool_result}"
                    ),
                },
            ],
        )

        return response.output_text