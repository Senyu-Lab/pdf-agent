import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class ToolCall:
    # LLM 决定调用的 Tool 名称
    tool_name: str

    # LLM 提供给 Tool 的参数
    arguments: dict[str, Any]


class LLMClient(ABC):
    tool_schema_format = "responses"

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> ToolCall:
        pass

    @abstractmethod
    def generate_answer(
        self,
        messages: list[dict[str, str]],
        tool_name: str,
        tool_result: Any,
    ) -> str:
        pass

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        pass