
import json
from typing import Any

from groq import Groq

from pdf_agent.llm_client import ToolCall


class GroqLLMClient:
    # Groq Chat Completions API 使用的 Tool Schema 格式
    tool_schema_format = "chat_completions"

    def __init__(
            self,
            model: str = "openai/gpt-oss-20b",
    ):
        # Groq SDK 会自动读取 GROQ_API_KEY 环境变量
        self.client = Groq()
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> ToolCall:
        # 使用 Chat Completions API，让模型选择需要调用的 Tool
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            raise ValueError("LLM did not return a tool call.")

        # 当前 Agent 一次只执行一个 Tool，因此先处理第一个调用
        tool_call = message.tool_calls[0]

        return ToolCall(
            tool_name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )

    def generate_answer(
        self,
        messages: list[dict[str, str]],
        tool_name: str,
        tool_result: Any,
    ) -> str:
        # 将工具执行结果交给模型，生成面向用户的自然语言回答
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                *messages,
                {
                    "role": "assistant",
                    "content": (
                        f"Tool '{tool_name}' returned this result: "
                        f"{json.dumps(tool_result, ensure_ascii=False, default=str)}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请根据上面的工具执行结果，直接回答我最初的问题。"
                        "如果结果为空，请如实说明没有找到相关内容。"
                    ),
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        if answer is None:
            raise ValueError("LLM did not return a text answer.")

        return answer


    def generate_text(self, prompt: str) -> str:
        # 根据提示词调用模型生成文本
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        result = response.choices[0].message.content

        # 防止模型没有返回文本
        if result is None:
            raise ValueError("LLM did not return a text answer.")

        return result

