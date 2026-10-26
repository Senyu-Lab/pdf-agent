# PDF Agent

A lightweight PDF processing agent built with Python. It supports PDF text extraction, keyword search, and LLM-based summarization with a chunk-and-reduce workflow for long documents.

## Features

- **PDF text extraction** with PyMuPDF
- **Keyword search** across PDF pages
- **Page-based text processing** with page number tracking
- **Long-document summarization** by splitting document text into chunks
- **Summary reduction** to combine chunk summaries into a coherent final summary
- **Retry handling** for failed chunk summarization
- **LLM integration** through Groq
- **Automated tests** with pytest and mocked dependencies

## Tech Stack

- Python 3.12+
- PyMuPDF
- Groq API
- OpenAI SDK
- pytest

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Senyu-Lab/pdf-agent.git
cd pdf-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the project

Install the project and its development dependencies:

```bash
pip install -e ".[dev]"
```

### 4. Configure the Groq API key

LLM-based summarization requires a Groq API key. Set `GROQ_API_KEY` in your environment before running the summarization command.

For Windows PowerShell, set it for the current terminal session:

```powershell
$env:GROQ_API_KEY = "your_groq_api_key"
```

Replace the placeholder with your own API key. Do not commit API keys to the repository.

## CLI Usage

Run commands from the project root.

### Display PDF information

```bash
python -m pdf_agent.cli path/to/document.pdf
```

### Search for a keyword

```bash
python -m pdf_agent.cli path/to/document.pdf --search "keyword"
```

### Summarize a PDF

```bash
python -m pdf_agent.cli path/to/document.pdf --summarize
```

The summarization workflow:

1. Extracts text from PDF pages.
2. Splits the extracted text into manageable chunks while preserving page numbers.
3. Generates a summary for each chunk using the LLM.
4. Retries failed chunk summarization attempts.
5. Combines the chunk summaries into a final document summary.

**Current limitation:** The reader extracts selectable text from PDFs. Scanned pages that contain only images require OCR, which is planned for a future iteration.

## Architecture

```text
PDF Document
     |
     v
   PDFReader
     |
     v
  Page Text
     |
     v
    Chunker
     |
     v
Chunk Summarization
     |
     v
 Summary Reduction
     |
     v
 Final Summary
```

The project also includes a PDF keyword-search capability and an agent/tool-calling foundation.

## Running Tests

Run the test suite:

```bash
pytest -q
```

Tests use mocks where appropriate so core logic can be tested without making real LLM API calls.


## License

MIT License
