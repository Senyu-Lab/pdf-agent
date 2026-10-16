
import argparse
import tempfile
from pathlib import Path

import fitz

from pdf_agent.agent import PDFAgent
from pdf_agent.groq_llm_client import GroqLLMClient


def create_sample_pdf(file_path: Path) -> None:
    # 创建一份固定内容的 PDF，方便重复验证 Agent 的实际表现
    with fitz.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Python is a programming language. "
            "It is widely used in data analysis and machine learning.",
        )
        page.insert_text(
            (72, 110),
            "PDF Agent test document.",
        )
        document.save(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PDF Agent with Groq."
    )
    parser.add_argument(
        "--question",
        default="这个 PDF 有多少页？",
        help="A natural-language question about the sample PDF.",
    )
    args = parser.parse_args()

    # 使用临时文件，避免在项目目录留下测试 PDF
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = Path(temp_dir) / "sample.pdf"
        create_sample_pdf(pdf_path)

        # 初始化真实 Groq 客户端并运行 Agent
        agent = PDFAgent(
            file_path=str(pdf_path),
            llm=GroqLLMClient(),
        )

        result = agent.run(args.question)

        print(f"Question: {args.question}")
        print(f"Tool: {result.tool_name}")
        print(f"Tool result: {result.tool_result}")
        print(f"Answer: {result.answer}")


if __name__ == "__main__":
    main()
