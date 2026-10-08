import fitz

from pdf_agent.agent import AgentResult, PDFAgent
from pdf_agent.tools import Tool, create_get_page_count_tool


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