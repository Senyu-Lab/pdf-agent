# PDF Agent

A lightweight PDF agent with document tools and LLM-based tool calling.

## Features

* PDF text extraction with PyMuPDF
* Keyword search across PDF pages
* Tool abstraction and registry
* Mock LLM for local development
* Basic Agent tool-calling workflow
* Automated tests with pytest

## Architecture

```text
User
 ↓
LLM
 ↓
Tool Call
 ↓
Tool Registry
 ↓
PDF Tool
 ↓
Tool Result
 ↓
Final Answer
```

## Test

```bash
pytest
```

The project uses a Mock LLM, so the core Agent workflow can be tested without an API key.

## Tech Stack

* Python 3.12+
* PyMuPDF
* pytest
* OpenAI SDK
* LLM Tool Calling

## Roadmap

* [x] PDF reading
* [x] PDF search
* [x] Tool system
* [x] Mock LLM
* [x] Basic Agent workflow
* [ ] Tool execution layer
* [ ] Multi-step Agent workflow
* [ ] Conversation memory
* [ ] RAG
* [ ] Production LLM integration

## License

MIT License
