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


def test_cli_search_pdf(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from pdf_agent import cli

    pdf_path = tmp_path / "sample.pdf"

    # 创建一个有效的临时 PDF
    import fitz

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    # 模拟搜索结果，避免依赖真实 PDF 内容
    fake_results = [
        SimpleNamespace(
            page_number=1,
            text="Python is useful for data analysis.",
        ),
        SimpleNamespace(
            page_number=2,
            text="Python supports automation.",
        ),
    ]

    class FakePDFSearch:
        def __init__(self, file_path):
            assert file_path == str(pdf_path)

        def search(self, keyword):
            assert keyword == "Python"
            return fake_results

    monkeypatch.setattr(cli, "PDFSearch", FakePDFSearch)
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", str(pdf_path), "--search", "Python"],
    )

    cli.main()

    output = capsys.readouterr().out

    assert f"PDF: {pdf_path}" in output
    assert "Pages: 1" in output
    assert 'Search: "Python"' in output
    assert "Found 2 result(s)." in output
    assert "Page 1" in output
    assert "Python is useful for data analysis." in output
    assert "Page 2" in output
    assert "Python supports automation." in output
