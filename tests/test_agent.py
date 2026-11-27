import fitz
import pytest

from pdf_agent.capabilities.pdf.reader import (
    PDFReader,
    PageNumberOutOfRangeError,
)
from pdf_agent.capabilities.pdf.reader import PDFReader
from unittest.mock import Mock
from pdf_agent.agent import AgentResult, PDFAgent
from pdf_agent.llm_client import ToolCall
from pdf_agent.mock_llm_client import MockLLMClient
from pdf_agent.tool_registry import ToolRegistry
from pdf_agent.tools import (
    Tool,
    create_get_page_count_tool,
    create_read_all_pdf_text_tool,
    create_read_pdf_page_tool,
    create_search_pdf_tool,
    create_summarize_pdf_tool,
    create_extract_pdf_metadata_tool,
    create_get_page_range_text_tool,
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


def test_llm_client_selects_summarize_tool():
    client = MockLLMClient()

    result = client.chat(
        messages=[
            {
                "role": "user",
                "content": "请总结这份 PDF。",
            }
        ],
        tools=[],
    )

    assert isinstance(result, ToolCall)
    assert result.tool_name == "summarize_pdf"
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

def test_extract_pdf_metadata_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建包含元数据的测试 PDF
    document = fitz.open()
    document.set_metadata({
        "title": "PDF Agent Test",
        "author": "Senyu",
        "subject": "Metadata testing",
    })
    document.new_page()
    document.save(pdf_path)
    document.close()

    tool = create_extract_pdf_metadata_tool(pdf_path)

    assert isinstance(tool, Tool)
    assert tool.name == "extract_pdf_metadata"
    assert tool.parameters["type"] == "object"
    assert tool.parameters["required"] == []
    assert tool.parameters["additionalProperties"] is False

    result = tool.function()

    assert result["title"] == "PDF Agent Test"
    assert result["author"] == "Senyu"
    assert result["subject"] == "Metadata testing"
    assert result["page_count"] == 1

    schema = tool.to_schema()
    assert schema["name"] == "extract_pdf_metadata"
    assert schema["parameters"]["properties"] == {}
    assert schema["parameters"]["required"] == []

    chat_schema = tool.to_chat_schema()
    assert chat_schema["function"]["name"] == "extract_pdf_metadata"

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


def test_tool_to_chat_schema():
    tool = Tool(
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

    schema = tool.to_chat_schema()

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search_pdf"
    assert schema["function"]["description"] == (
        "Search for a keyword in the PDF."
    )
    assert schema["function"]["parameters"]["type"] == "object"
    assert "keyword" in schema["function"]["parameters"]["properties"]


def test_tool_registry_get_chat_schemas():
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

    schemas = registry.get_chat_schemas()

    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "test_tool"

def test_get_page_range_text_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()
    for text in ["First page", "Second page", "Third page"]:
        page = document.new_page()
        page.insert_text((72, 72), text)

    document.save(pdf_path)
    document.close()

    tool = create_get_page_range_text_tool(pdf_path)

    assert isinstance(tool, Tool)
    assert tool.name == "get_page_range_text"
    assert tool.parameters["required"] == ["start_page", "end_page"]

    result = tool.function(start_page=2, end_page=3)

    assert len(result) == 2
    assert result[0]["page_number"] == 2
    assert result[0]["text"].strip() == "Second page"
    assert result[1]["page_number"] == 3
    assert result[1]["text"].strip() == "Third page"

def test_read_pdf_page_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建包含两页文本的测试 PDF
    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text((72, 72), "First page content.")

    second_page = document.new_page()
    second_page.insert_text((72, 72), "Second page content.")

    document.save(pdf_path)
    document.close()

    tool = create_read_pdf_page_tool(pdf_path)

    assert isinstance(tool, Tool)
    assert tool.name == "read_pdf_page"
    assert tool.parameters["type"] == "object"
    assert "page_number" in tool.parameters["properties"]

    result = tool.function(page_number=2)

    assert "Second page content." in result
    assert "First page content." not in result



def test_read_all_pdf_text_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建包含两页文本的测试 PDF
    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text((72, 72), "First page content.")

    second_page = document.new_page()
    second_page.insert_text((72, 72), "Second page content.")

    document.save(pdf_path)
    document.close()

    tool = create_read_all_pdf_text_tool(pdf_path)

    assert isinstance(tool, Tool)
    assert tool.name == "read_all_pdf_text"
    assert tool.parameters["type"] == "object"
    assert tool.parameters["required"] == []
    assert tool.parameters["additionalProperties"] is False

    result = tool.function()

    assert "First page content." in result
    assert "Second page content." in result

def test_agent_registers_read_all_pdf_text_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    agent = PDFAgent(str(pdf_path))

    tool = agent.registry.get("read_all_pdf_text")

    assert tool.name == "read_all_pdf_text"

def test_summarize_pdf_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text((72, 72), "Python is a programming language.")

    second_page = document.new_page()
    second_page.insert_text((72, 72), "Python supports data analysis.")

    document.save(pdf_path)
    document.close()

    llm = Mock()
    llm.generate_text.side_effect = [
        "Python is a programming language.",
        "This document introduces Python and data analysis.",
    ]

    tool = create_summarize_pdf_tool(pdf_path, llm)

    result = tool.function()

    assert isinstance(tool, Tool)
    assert tool.name == "summarize_pdf"
    assert result == "This document introduces Python and data analysis."
    assert llm.generate_text.call_count == 2

def test_agent_registers_summarize_pdf_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    agent = PDFAgent(str(pdf_path))
    tool = agent.registry.get("summarize_pdf")

    assert tool.name == "summarize_pdf"

def test_agent_registers_extract_pdf_metadata_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    agent = PDFAgent(str(pdf_path))

    tool = agent.registry.get("extract_pdf_metadata")

    assert isinstance(tool, Tool)
    assert tool.name == "extract_pdf_metadata"

def test_agent_can_summarize_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建包含两页文本的测试 PDF
    document = fitz.open()

    first_page = document.new_page()
    first_page.insert_text(
        (72, 72),
        "Python is a programming language.",
    )

    second_page = document.new_page()
    second_page.insert_text(
        (72, 72),
        "Python supports data analysis.",
    )

    document.save(pdf_path)
    document.close()

    llm = Mock()

    llm.tool_schema_format = "responses"
    llm.chat.return_value = ToolCall(
        tool_name="summarize_pdf",
        arguments={},
    )

    llm.generate_text.side_effect = [
        "Python is a programming language.",
        "This document introduces Python and data analysis.",
    ]

    llm.generate_answer.return_value = (
        "这份文档介绍了 Python 编程语言及其数据分析应用。"
    )

    agent = PDFAgent(str(pdf_path), llm=llm)

    result = agent.run("请总结这份 PDF。")

    assert isinstance(result, AgentResult)
    assert result.tool_name == "summarize_pdf"
    assert result.tool_result == (
        "This document introduces Python and data analysis."
    )
    assert result.answer == (
        "这份文档介绍了 Python 编程语言及其数据分析应用。"
    )

    llm.chat.assert_called_once()
    llm.generate_text.assert_called()
    llm.generate_answer.assert_called_once()

def test_pdf_reader_get_metadata(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # 创建包含元数据的测试 PDF
    document = fitz.open()

    document.set_metadata({
        "title": "PDF Agent Test",
        "author": "Senyu",
        "subject": "PDF metadata testing",
        "creator": "PyCharm",
        "producer": "PyMuPDF",
        "creationDate": "D:20261113000000",
        "modDate": "D:20261113010000",
    })

    document.new_page()
    document.new_page()

    document.save(pdf_path)
    document.close()

    reader = PDFReader(pdf_path)

    metadata = reader.get_metadata()

    assert metadata["title"] == "PDF Agent Test"
    assert metadata["author"] == "Senyu"
    assert metadata["subject"] == "PDF metadata testing"
    assert metadata["creator"] == "PyCharm"
    assert metadata["producer"] == "PyMuPDF"
    assert metadata["page_count"] == 2

def test_pdf_reader_get_metadata_with_empty_fields(tmp_path):
    pdf_path = tmp_path / "empty_metadata.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    reader = PDFReader(pdf_path)

    metadata = reader.get_metadata()

    assert metadata["title"] == ""
    assert metadata["author"] == ""
    assert metadata["subject"] == ""
    assert metadata["creator"] == ""
    assert metadata["producer"] == ""
    assert metadata["creation_date"] == ""
    assert metadata["modification_date"] == ""
    assert metadata["page_count"] == 1

def test_pdf_reader_read_page_range(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()

    for text in ["First page", "Second page", "Third page"]:
        page = document.new_page()
        page.insert_text((72, 72), text)

    document.save(pdf_path)
    document.close()

    reader = PDFReader(pdf_path)

    pages = reader.read_page_range(2, 3)

    assert len(pages) == 2

    assert pages[0].page_number == 2
    assert pages[0].text.strip() == "Second page"

    assert pages[1].page_number == 3
    assert pages[1].text.strip() == "Third page"

@pytest.mark.parametrize(
    ("start_page", "end_page"),
    [
        (0, 2),   # 起始页小于 1
        (1, 4),   # 结束页超过总页数
        (3, 2),   # 起始页大于结束页
    ],
)
def test_pdf_reader_read_page_range_out_of_range(
    tmp_path,
    start_page,
    end_page,
):
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()
    for _ in range(3):
        document.new_page()
    document.save(pdf_path)
    document.close()

    reader = PDFReader(pdf_path)

    with pytest.raises(PageNumberOutOfRangeError):
        reader.read_page_range(start_page, end_page)

def test_agent_registers_get_page_range_text_tool(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    agent = PDFAgent(str(pdf_path))

    tool = agent.registry.get("get_page_range_text")

    assert isinstance(tool, Tool)
    assert tool.name == "get_page_range_text"