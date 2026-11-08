from unittest.mock import patch
import pytest
from unittest.mock import MagicMock, patch

import json
from pdf_agent.llm_client import ToolCall
from pdf_agent.deepseek_llm_client import DeepSeekLLMClient


def test_client_uses_default_configuration(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("pdf_agent.deepseek_llm_client.OpenAI") as mock_openai:
        llm = DeepSeekLLMClient()

    mock_openai.assert_called_once_with(
        api_key="test-deepseek-key",
        base_url="https://api.deepseek.com",
    )

    assert llm.model == "deepseek-flash"

def test_client_accepts_custom_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("pdf_agent.deepseek_llm_client.OpenAI"):
        llm = DeepSeekLLMClient(model="deepseek-chat")

    assert llm.model == "deepseek-chat"

def test_generate_text_returns_text(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("pdf_agent.deepseek_llm_client.OpenAI") as mock_openai:
        mock_api = mock_openai.return_value
        mock_api.chat.completions.create.return_value.choices[
            0
        ].message.content = "这是 DeepSeek 的回答"

        llm = DeepSeekLLMClient()

        result = llm.generate_text("请介绍 Python")

    assert result == "这是 DeepSeek 的回答"

    mock_api.chat.completions.create.assert_called_once_with(
        model="deepseek-flash",
        messages=[
            {"role": "user", "content": "请介绍 Python"},
        ],
        temperature=0,
    )

def test_generate_text_raises_error_when_response_is_empty(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("pdf_agent.deepseek_llm_client.OpenAI") as mock_openai:
        mock_api = mock_openai.return_value
        mock_api.chat.completions.create.return_value.choices[
            0
        ].message.content = None

        llm = DeepSeekLLMClient()

        with pytest.raises(
            ValueError,
            match="LLM did not return a text answer.",
        ):
            llm.generate_text("请介绍 Python")

def test_chat_returns_tool_call(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("pdf_agent.deepseek_llm_client.OpenAI") as mock_openai:
        mock_api = mock_openai.return_value

        tool_call = MagicMock()
        tool_call.function.name = "search_pdf"
        tool_call.function.arguments = '{"query": "Python"}'

        mock_api.chat.completions.create.return_value.choices[
            0
        ].message.tool_calls = [tool_call]

        llm = DeepSeekLLMClient()

        result = llm.chat(
            messages=[
                {"role": "user", "content": "搜索 Python"},
            ],
            tools=[{"type": "function", "function": {"name": "search_pdf"}}],
        )

    assert result.tool_name == "search_pdf"
    assert result.arguments == {"query": "Python"}

def test_chat_raises_error_when_no_tool_call(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("pdf_agent.deepseek_llm_client.OpenAI") as mock_openai:
        mock_api = mock_openai.return_value
        mock_api.chat.completions.create.return_value.choices[
            0
        ].message.tool_calls = None

        llm = DeepSeekLLMClient()

        with pytest.raises(
            ValueError,
            match="LLM did not return a tool call.",
        ):
            llm.chat(
                messages=[
                    {"role": "user", "content": "搜索 Python"},
                ],
                tools=[],
            )

def test_chat_raises_error_when_arguments_are_invalid_json(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("pdf_agent.deepseek_llm_client.OpenAI") as mock_openai:
        mock_api = mock_openai.return_value

        tool_call = MagicMock()
        tool_call.function.name = "search_pdf"
        tool_call.function.arguments = "{invalid json}"

        mock_api.chat.completions.create.return_value.choices[
            0
        ].message.tool_calls = [tool_call]

        llm = DeepSeekLLMClient()

        with pytest.raises(json.JSONDecodeError):
            llm.chat(
                messages=[
                    {"role": "user", "content": "搜索 Python"},
                ],
                tools=[],
            )

def test_generate_answer_returns_text(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    with patch("pdf_agent.deepseek_llm_client.OpenAI") as mock_openai:
        mock_api = mock_openai.return_value
        mock_api.chat.completions.create.return_value.choices[
            0
        ].message.content = "Python 出现在第 2 页。"

        llm = DeepSeekLLMClient()

        result = llm.generate_answer(
            messages=[
                {"role": "user", "content": "搜索 Python"},
            ],
            tool_name="search_pdf",
            tool_result={"matches": ["Python"]},
        )

    assert result == "Python 出现在第 2 页。"