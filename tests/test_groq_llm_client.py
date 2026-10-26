
import json
from unittest.mock import MagicMock, patch

import pytest

from pdf_agent.groq_llm_client import GroqLLMClient
from pdf_agent.llm_client import ToolCall


@pytest.fixture
def client():
    # 模拟 Groq SDK，避免测试时发起真实 API 请求
    with patch("pdf_agent.groq_llm_client.Groq") as mock_groq:
        llm = GroqLLMClient()
        yield llm, mock_groq.return_value


def test_chat_returns_tool_call(client):
    llm, mock_api = client

    # 模拟模型选择 search_pdf 工具

    # 模拟模型选择 search_pdf 工具
    mock_response = MagicMock()
    mock_message = mock_response.choices[0].message

    mock_tool_call = MagicMock()
    mock_tool_call.function.name = "search_pdf"
    mock_tool_call.function.arguments = '{"keyword": "Python"}'

    mock_message.tool_calls = [mock_tool_call]
    mock_api.chat.completions.create.return_value = mock_response


    result = llm.chat(
        messages=[{"role": "user", "content": "搜索 Python"}],
        tools=[],
    )

    assert result == ToolCall(
        tool_name="search_pdf",
        arguments={"keyword": "Python"},
    )
    mock_api.chat.completions.create.assert_called_once()


def test_chat_raises_when_no_tool_call(client):
    llm, mock_api = client

    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = None
    mock_api.chat.completions.create.return_value = mock_response

    with pytest.raises(ValueError, match="LLM did not return a tool call"):
        llm.chat(
            messages=[{"role": "user", "content": "你好"}],
            tools=[],
        )


def test_chat_raises_when_arguments_are_invalid_json(client):
    llm, mock_api = client

    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = [
        MagicMock(
            function=MagicMock(
                name="search_pdf",
                arguments="{invalid json}",
            )
        )
    ]
    mock_api.chat.completions.create.return_value = mock_response

    with pytest.raises(json.JSONDecodeError):
        llm.chat(
            messages=[{"role": "user", "content": "搜索 Python"}],
            tools=[],
        )


def test_generate_answer_returns_text(client):
    llm, mock_api = client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "找到了 Python 相关内容。"
    mock_api.chat.completions.create.return_value = mock_response

    result = llm.generate_answer(
        messages=[{"role": "user", "content": "搜索 Python"}],
        tool_name="search_pdf",
        tool_result={"matches": ["Python"]},
    )

    assert result == "找到了 Python 相关内容。"
    mock_api.chat.completions.create.assert_called_once()


def test_generate_answer_raises_when_content_is_none(client):
    llm, mock_api = client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_api.chat.completions.create.return_value = mock_response

    with pytest.raises(ValueError, match="LLM did not return a text answer"):
        llm.generate_answer(
            messages=[{"role": "user", "content": "搜索 Python"}],
            tool_name="search_pdf",
            tool_result={},
        )


def test_generate_text_returns_model_response(client):
    llm, mock_api = client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = "这是一段测试摘要。"
    mock_api.chat.completions.create.return_value = mock_response

    result = llm.generate_text("请总结这段文本。")

    assert result == "这是一段测试摘要。"
    mock_api.chat.completions.create.assert_called_once()


def test_generate_text_raises_when_content_is_none(client):
    llm, mock_api = client

    mock_response = MagicMock()
    mock_response.choices[0].message.content = None
    mock_api.chat.completions.create.return_value = mock_response

    with pytest.raises(ValueError, match="LLM did not return a text answer"):
        llm.generate_text("请总结这段文本。")
