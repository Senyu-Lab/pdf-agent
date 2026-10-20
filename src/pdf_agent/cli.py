import argparse

from pdf_agent.capabilities.pdf.reader import PDFReader
from pdf_agent.capabilities.pdf.search import PDFSearch
from pdf_agent.agent import PDFAgent
from pdf_agent.groq_llm_client import GroqLLMClient


def main():
    parser = argparse.ArgumentParser(
        description="A lightweight PDF agent."
    )

    parser.add_argument(
        "pdf",
        help="Path to the PDF file.",
    )

    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Summarize the PDF using an LLM.",
    )

    parser.add_argument(
        "--search",
        help="Search for a keyword in the PDF.",
    )


    args = parser.parse_args()

    reader = PDFReader(args.pdf)

    print(f"PDF: {args.pdf}")
    print(f"Pages: {reader.get_page_count()}")

    if args.search:
        search = PDFSearch(args.pdf)
        results = search.search(args.search)

        print(f'\nSearch: "{args.search}"')
        print(f"Found {len(results)} result(s).\n")

        for result in results:
            print(f"Page {result.page_number}")
            print(result.text.strip())
            print()

    if args.summarize:
        agent = PDFAgent(
            args.pdf,
            llm=GroqLLMClient(),
        )

        result = agent.run(
            "请阅读并总结这份 PDF 文档的主要内容，"
            "提炼核心观点、重要信息和结论。"
        )

        print("\nPDF Summary")
        print("=" * 40)
        print(result.answer)


if __name__ == "__main__":
    main()