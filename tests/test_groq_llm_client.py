
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


from pdf_agent.llm_client import LLMClient


def test_groq_client_implements_llm_client():
    with patch("pdf_agent.groq_llm_client.Groq"):
        client = GroqLLMClient()

    assert isinstance(client, LLMClient)

def test_groq_client_uses_configured_model_and_tool_schema_format():
    with patch("pdf_agent.groq_llm_client.Groq"):
        client = GroqLLMClient(model="llama-3.3-70b-versatile")

    assert client.model == "llama-3.3-70b-versatile"
    assert client.tool_schema_format == "chat_completions"

def test_chat_sends_expected_api_parameters():
    with patch("pdf_agent.groq_llm_client.Groq") as mock_groq:
        client = GroqLLMClient(model="test-model")

        mock_groq.return_value.chat.completions.create.return_value.choices = [
            MagicMock(
                message=MagicMock(
                    tool_calls=[
                        MagicMock(
                            function=MagicMock(
                                name="search_pdf",
                                arguments='{"query": "test"}',
                            )
                        )
                    ]
                )
            )
        ]

        client.chat(messages=[], tools=[])

    mock_groq.return_value.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=[],
        tools=[],
        tool_choice="auto",
        temperature=0,
    )

def test_chat_returns_only_first_tool_call():
    with patch("pdf_agent.groq_llm_client.Groq"):
        client = GroqLLMClient()

    first_call = MagicMock()
    first_call.function.name = "search_pdf"
    first_call.function.arguments = '{"query": "first"}'

    second_call = MagicMock()
    second_call.function.name = "summarize_pdf"
    second_call.function.arguments = '{"query": "second"}'

    response = MagicMock()
    response.choices[0].message.tool_calls = [
        first_call,
        second_call,
    ]

    client.client.chat.completions.create.return_value = response

    result = client.chat(messages=[], tools=[])

    assert result.tool_name == "search_pdf"
    assert result.arguments == {"query": "first"}