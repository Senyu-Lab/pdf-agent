
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pdf_agent.llm_client import ToolCall
from pdf_agent.openai_llm_client import OpenAILLMClient


@patch("pdf_agent.openai_llm_client.OpenAI")
def test_chat_returns_tool_call(mock_openai):
    mock_client = mock_openai.return_value
    mock_client.responses.create.return_value = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="search_pdf",
                arguments='{"keyword": "Python"}',
            )
        ]
    )

    client = OpenAILLMClient()
    result = client.chat([], [])

    assert result == ToolCall(
        tool_name="search_pdf",
        arguments={"keyword": "Python"},
    )


@patch("pdf_agent.openai_llm_client.OpenAI")
def test_chat_raises_when_no_tool_call(mock_openai):
    mock_client = mock_openai.return_value
    mock_client.responses.create.return_value = SimpleNamespace(
        output=[
            SimpleNamespace(type="message")
        ]
    )

    client = OpenAILLMClient()

    with pytest.raises(ValueError, match="LLM did not return a tool call"):
        client.chat([], [])


@patch("pdf_agent.openai_llm_client.OpenAI")
def test_chat_raises_when_arguments_are_invalid_json(mock_openai):
    mock_client = mock_openai.return_value
    mock_client.responses.create.return_value = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="search_pdf",
                arguments="invalid json",
            )
        ]
    )

    client = OpenAILLMClient()

    with pytest.raises(ValueError):
        client.chat([], [])


@patch("pdf_agent.openai_llm_client.OpenAI")
def test_generate_answer_returns_text(mock_openai):
    mock_client = mock_openai.return_value
    mock_client.responses.create.return_value = SimpleNamespace(
        output_text="The PDF contains Python content."
    )

    client = OpenAILLMClient()
    result = client.generate_answer(
        messages=[],
        tool_name="search_pdf",
        tool_result={"matches": ["Python"]},
    )

    assert result == "The PDF contains Python content."


@patch("pdf_agent.openai_llm_client.OpenAI")
def test_generate_text_returns_model_response(mock_openai):
    mock_client = mock_openai.return_value
    mock_client.responses.create.return_value = SimpleNamespace(
        output_text="This is a generated summary."
    )

    client = OpenAILLMClient()
    result = client.generate_text("Summarize this PDF")

    assert result == "This is a generated summary."
