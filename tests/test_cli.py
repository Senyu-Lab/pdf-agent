import fitz

from pdf_agent import cli



def test_cli_summarize_pdf(tmp_path, monkeypatch, capsys):
    from pdf_agent import cli
    from pdf_agent.capabilities.pdf.reader import PDFPage

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

    # 提供模拟页面文本，避免空白 PDF 无法生成摘要
    monkeypatch.setattr(
        cli.PDFReader,
        "read_pages",
        lambda self: [
            PDFPage(
                page_number=1,
                text="这是用于测试的 PDF 内容。",
            )
        ],
    )

    # 模拟完整摘要流程，避免真实 LLM 调用
    def fake_summarize_document(
        pages,
        summarize_func,
        combine_func,
    ):
        assert len(pages) == 1
        assert pages[0].text == "这是用于测试的 PDF 内容。"
        assert callable(summarize_func)
        assert callable(combine_func)

        return "这是测试摘要。"

    monkeypatch.setattr(
        cli,
        "summarize_document",
        fake_summarize_document,
    )

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


import pytest


def test_cli_help(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", "--help"],
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0

    output = capsys.readouterr().out

    assert "usage:" in output.lower()
    assert "--summarize" in output
    assert "--search" in output
    assert "Path to the PDF file." in output


def test_cli_summarize_uses_chunked_pipeline(
    tmp_path, monkeypatch, capsys
):
    from pdf_agent import cli
    from pdf_agent.capabilities.pdf.reader import PDFPage

    pdf_path = tmp_path / "sample.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    pages = [
        PDFPage(page_number=1, text="测试 PDF 内容"),
    ]
    calls = {}

    class FakeReader:
        def __init__(self, file_path):
            pass

        def get_page_count(self):
            return 1

        def read_pages(self):
            return pages

    class FakeLLM:
        pass

    def fake_summarize_document(
        pages,
        summarize_func,
        combine_func,
    ):
        calls["pages"] = pages
        calls["summarize_func"] = summarize_func
        calls["combine_func"] = combine_func
        return "分块摘要流程生成的结果"

    def unexpected_agent(*args, **kwargs):
        raise AssertionError(
            "CLI should not use PDFAgent for this summary flow."
        )

    monkeypatch.setattr(cli, "PDFReader", FakeReader)
    monkeypatch.setattr(cli, "GroqLLMClient", FakeLLM)
    monkeypatch.setattr(
        cli,
        "summarize_document",
        fake_summarize_document,
        raising=False,
    )
    monkeypatch.setattr(cli, "PDFAgent", unexpected_agent)
    monkeypatch.setattr(
        "sys.argv",
        ["cli.py", str(pdf_path), "--summarize"],
    )

    cli.main()

    output = capsys.readouterr().out

    assert calls["pages"] == pages
    assert callable(calls["summarize_func"])
    assert callable(calls["combine_func"])
    assert "分块摘要流程生成的结果" in output
