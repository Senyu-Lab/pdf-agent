import fitz

from pdf_agent import cli


def test_cli_summarize_pdf(tmp_path, monkeypatch, capsys):
    pdf_path = tmp_path / "sample.pdf"

    # 创建一个有效的临时 PDF
    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    # 模拟 Groq 客户端，避免真实 API 调用
    monkeypatch.setattr(
        cli,
        "GroqLLMClient",
        lambda: object(),
    )

    # 模拟 Agent，隔离真实的 LLM 和工具执行
    class FakeAgent:
        def __init__(self, file_path, llm=None):
            assert file_path == str(pdf_path)
            assert llm is not None

        def run(self, question):
            assert "总结" in question
            assert "核心观点" in question

            return type(
                "FakeResult",
                (),
                {"answer": "这是测试摘要。"},
            )()

    monkeypatch.setattr(cli, "PDFAgent", FakeAgent)

    # 模拟命令行参数
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", str(pdf_path), "--summarize"],
    )

    cli.main()

    output = capsys.readouterr().out

    assert f"PDF: {pdf_path}" in output
    assert "Pages: 1" in output
    assert "PDF Summary" in output
    assert "这是测试摘要。" in output