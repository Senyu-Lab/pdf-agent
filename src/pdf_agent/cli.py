import argparse

from pdf_agent.pdf_reader import PDFReader
from pdf_agent.pdf_search import PDFSearch


def main():
    parser = argparse.ArgumentParser(
        description="A lightweight PDF agent."
    )

    parser.add_argument(
        "pdf",
        help="Path to the PDF file.",
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


if __name__ == "__main__":
    main()