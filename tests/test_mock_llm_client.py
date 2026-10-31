
from pdf_agent.mock_llm_client import MockLLMClient


def test_generate_text_returns_mock_response():
    client = MockLLMClient()

    result = client.generate_text("Summarize this PDF")

    assert result == "Mock generated text for: Summarize this PDF"
