
import pytest

from pdf_agent.llm_client import LLMClient


def test_llm_client_cannot_be_instantiated():
    with pytest.raises(TypeError):
        LLMClient()


def test_llm_client_requires_chat():
    class IncompleteClient(LLMClient):
        def generate_answer(self, messages, tool_name, tool_result):
            return "answer"

        def generate_text(self, prompt):
            return "text"

    with pytest.raises(TypeError):
        IncompleteClient()


def test_llm_client_requires_generate_answer():
    class IncompleteClient(LLMClient):
        def chat(self, messages, tools):
            return None

        def generate_text(self, prompt):
            return "text"

    with pytest.raises(TypeError):
        IncompleteClient()


def test_llm_client_requires_generate_text():
    class IncompleteClient(LLMClient):
        def chat(self, messages, tools):
            return None

        def generate_answer(self, messages, tool_name, tool_result):
            return "answer"

    with pytest.raises(TypeError):
        IncompleteClient()
