import fitz

from pdf_agent.agent import AgentResult, PDFAgent
from pdf_agent.llm_client import ToolCall
from pdf_agent.mock_llm_client import MockLLMClient
from pdf_agent.tool_registry import ToolRegistry
from pdf_agent.tools import (
    Tool,
    create_get_page_count_tool,
    create_search_pdf_tool,
)


def test_agent_can_get_page_count(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建一个包含两页的测试 PDF
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    agent = PDFAgent(pdf_path)

    result = agent.run("这个 PDF 有多少页？")

    assert isinstance(result, AgentResult)
    assert result.tool_name == "get_page_count"
    assert result.tool_result == 2
    assert result.answer == "这个 PDF 一共有 2 页。"


def test_agent_can_search_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建一个包含关键词的测试 PDF
    document = fitz.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Python is a programming language.",
    )

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Machine learning is an important field.",
    )

    document.save(pdf_path)
    document.close()

    agent = PDFAgent(pdf_path)

    result = agent.run("PDF 里面有没有 Python？")

    assert isinstance(result, AgentResult)
    assert result.tool_name == "search_pdf"
    assert len(result.tool_result) == 1
    assert result.tool_result[0].page_number == 1
    assert "Python" in result.tool_result[0].text
    assert result.answer == "在第 1 页找到了相关内容。"


def test_llm_client_returns_page_count_tool_call():
    client = MockLLMClient()

    result = client.chat(
        messages=[
            {
                "role": "user",
                "content": "这个 PDF 有多少页？",
            }
        ],
        tools=[],
    )

    assert isinstance(result, ToolCall)
    assert result.tool_name == "get_page_count"
    assert result.arguments == {}


def test_llm_client_selects_search_tool():
    client = MockLLMClient()

    result = client.chat(
        messages=[
            {
                "role": "user",
                "content": "PDF 里面有没有 Python？",
            }
        ],
        tools=[],
    )

    assert isinstance(result, ToolCall)
    assert result.tool_name == "search_pdf"
    assert result.arguments == {
        "keyword": "Python",
    }


def test_llm_client_generates_page_count_answer():
    client = MockLLMClient()

    result = client.generate_answer(
        messages=[
            {
                "role": "user",
                "content": "这个 PDF 有多少页？",
            }
        ],
        tool_name="get_page_count",
        tool_result=2,
    )

    assert result == "这个 PDF 一共有 2 页。"


def test_llm_client_generates_search_answer():
    client = MockLLMClient()

    search_result = [
        type(
            "SearchResult",
            (),
            {
                "page_number": 1,
                "text": "Python is a programming language.",
            },
        )()
    ]

    result = client.generate_answer(
        messages=[
            {
                "role": "user",
                "content": "PDF 里面有没有 Python？",
            }
        ],
        tool_name="search_pdf",
        tool_result=search_result,
    )

    assert result == "在第 1 页找到了相关内容。"


def test_llm_client_handles_empty_search_result():
    client = MockLLMClient()

    result = client.generate_answer(
        messages=[
            {
                "role": "user",
                "content": "PDF 里面有没有 Python？",
            }
        ],
        tool_name="search_pdf",
        tool_result=[],
    )

    assert result == "没有找到相关内容。"


def test_get_page_count_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建一个包含三页的测试 PDF
    document = fitz.open()
    document.new_page()
    document.new_page()
    document.new_page()
    document.save(pdf_path)
    document.close()

    tool = create_get_page_count_tool(pdf_path)

    assert isinstance(tool, Tool)
    assert tool.name == "get_page_count"
    assert tool.description == "Get the total number of pages in the PDF."
    assert tool.parameters["type"] == "object"
    assert tool.function() == 3


def test_search_pdf_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建测试 PDF
    document = fitz.open()

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Python is useful.",
    )

    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Machine learning is useful.",
    )

    document.save(pdf_path)
    document.close()

    tool = create_search_pdf_tool(pdf_path)

    assert isinstance(tool, Tool)
    assert tool.name == "search_pdf"
    assert tool.parameters["type"] == "object"
    assert "keyword" in tool.parameters["properties"]

    results = tool.function("Python")

    assert len(results) == 1
    assert results[0].page_number == 1
    assert "Python" in results[0].text


def test_tool_to_schema():
    tool = Tool(
        name="test_tool",
        description="A test tool.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=lambda: "result",
    )

    schema = tool.to_schema()

    assert schema["type"] == "function"
    assert schema["name"] == "test_tool"
    assert schema["description"] == "A test tool."
    assert schema["parameters"]["type"] == "object"


def test_tool_registry():
    tool = Tool(
        name="test_tool",
        description="A test tool.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=lambda: "result",
    )

    registry = ToolRegistry()

    registry.register(tool)

    assert registry.get("test_tool") is tool
    assert registry.list_tools() == [tool]


def test_tool_registry_get_schemas():
    tool = Tool(
        name="test_tool",
        description="A test tool.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=lambda: "result",
    )

    registry = ToolRegistry([tool])

    schemas = registry.get_schemas()

    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["name"] == "test_tool"
    assert schemas[0]["description"] == "A test tool."
    assert schemas[0]["parameters"]["type"] == "object"