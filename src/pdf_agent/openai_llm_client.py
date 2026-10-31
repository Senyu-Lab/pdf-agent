
import json
from typing import Any

from openai import OpenAI

from pdf_agent.llm_client import LLMClient, ToolCall


class OpenAILLMClient(LLMClient):
    tool_schema_format = "responses"

    def __init__(self, model: str = "gpt-5-mini"):
        # OpenAI SDK 自动读取 OPENAI_API_KEY
        self.client = OpenAI()
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> ToolCall:
        response = self.client.responses.create(
            model=self.model,
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
        response = self.client.responses.create(
            model=self.model,
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

    def generate_text(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        return response.output_text
