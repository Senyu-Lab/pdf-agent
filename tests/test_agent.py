import fitz

from pdf_agent.agent import AgentResult, PDFAgent
from pdf_agent.tools import (
    Tool,
    create_get_page_count_tool,
    create_search_pdf_tool,
)
from pdf_agent.tool_registry import ToolRegistry
from pdf_agent.llm_client import LLMClient, ToolCall


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
    assert tool.description
    assert tool.parameters["type"] == "object"

    assert tool.function() == 3

def test_search_pdf_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建一个包含关键词的测试 PDF
    document = fitz.open()

    page = document.new_page()
    page.insert_text((72, 72), "Python is useful for data analysis.")

    page = document.new_page()
    page.insert_text((72, 72), "Machine learning is another topic.")

    document.save(pdf_path)
    document.close()

    tool = create_search_pdf_tool(pdf_path)

    assert isinstance(tool, Tool)
    assert tool.name == "search_pdf"
    assert tool.description
    assert tool.parameters["type"] == "object"

    assert "keyword" in tool.parameters["properties"]
    assert tool.parameters["properties"]["keyword"]["type"] == "string"
    assert tool.parameters["required"] == ["keyword"]

    results = tool.function("Python")

    assert len(results) == 1
    assert results[0].page_number == 1
    assert "Python" in results[0].text

def test_tool_registry():
    page_count_tool = Tool(
        name="get_page_count",
        description="Get the total number of pages in the PDF.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        function=lambda: 10,
    )

    search_tool = Tool(
        name="search_pdf",
        description="Search for a keyword in the PDF.",
        parameters={
            "type": "object",
            "properties": {
                "keyword": {
                    "type": "string",
                }
            },
            "required": ["keyword"],
        },
        function=lambda keyword: keyword,
    )

    registry = ToolRegistry([
        page_count_tool,
        search_tool,
    ])

    assert registry.get("get_page_count") is page_count_tool
    assert registry.get("search_pdf") is search_tool
    assert len(registry.list_tools()) == 2

def test_tool_registry_get_schemas():
    tool = Tool(
        name="example_tool",
        description="An example tool.",
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
    assert schemas[0]["function"]["name"] == "example_tool"

def test_tool_to_schema():
    tool = Tool(
        name="example_tool",
        description="An example tool.",
        parameters={
            "type": "object",
            "properties": {
                "value": {
                    "type": "string",
                }
            },
            "required": ["value"],
        },
        function=lambda value: value,
    )

    schema = tool.to_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "example_tool"
    assert schema["function"]["description"] == "An example tool."
    assert schema["function"]["parameters"] == tool.parameters

def test_llm_client_returns_tool_call():
    client = LLMClient()

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
    client = LLMClient()

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
def test_llm_client_generates_final_answer():
    client = LLMClient()

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