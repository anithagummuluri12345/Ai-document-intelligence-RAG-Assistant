# AI Document Intelligence & RAG Assistant

An AI-powered document question-answering system that uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from PDF documents and generate evidence-based answers with source citations.

The project explores semantic search, document processing, vector similarity search, and grounded response generation to make document analysis more useful and reliable.

## Overview

Searching lengthy documents manually can be time-consuming. This project provides a web-based interface where users can upload a PDF and ask questions in natural language.

The system retrieves relevant document passages and uses them to generate answers. When the available content is insufficient, it is designed to avoid unsupported responses.

## Features

- **PDF Question Answering:** Upload documents and ask questions in natural language.
- **Semantic Search:** Retrieve relevant passages using vector embeddings.
- **Retrieval-Augmented Generation:** Ground responses in retrieved document content.
- **Source Citations:** Display source filenames and page references alongside answers.
- **Insufficient-Context Handling:** Support responses when the documents do not contain enough information.
- **Flexible Generation:** Support a fallback answer-generation path and optional LLM API integration.
- **Evaluation:** Include an evaluation workflow for assessing retrieval and answer behavior.

## How It Works

1. **Document ingestion:** Extract text and page information from uploaded PDF files.
2. **Text chunking:** Split document text into smaller, overlapping passages.
3. **Embedding generation:** Convert text passages into vector representations.
4. **Vector indexing:** Store embeddings and associated metadata for similarity search.
5. **Retrieval:** Find passages relevant to the user's question.
6. **Answer generation:** Generate an answer using the retrieved context.
7. **Citation display:** Present the response with relevant document references.

## Technology Stack

| Component              | Technology                      |
| ---------------------- | ------------------------------- |
| Language               | Python                          |
| Document processing    | PyMuPDF                         |
| Text chunking          | LangChain Text Splitters        |
| Embeddings             | Sentence Transformers           |
| Vector search          | FAISS                           |
| Optional LLM providers | OpenAI API, Groq API            |
| Interface and server   | Python HTTP server              |
| Evaluation             | Python-based evaluation scripts |

## Getting Started

### Prerequisites

- Python 3.10 or later
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/anithagummuluri12345/Ai-document-intelligence-RAG-Assistant.git
cd Ai-document-intelligence-RAG-Assistant
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the environment's Python executable directly instead.

**macOS/Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

If you want to use an optional LLM provider, copy `.env.example` to `.env` and configure the supported API key.

Keep API keys private. Never commit your actual `.env` file to GitHub.

The exact environment variable names should match those used by the application.

### 5. Run the application

```bash
python app.py
```

Open the local URL printed by the application in your browser. If the server uses port 8000, visit:

```text
http://localhost:8000
```

### 6. Ask questions about a document

1. Upload a PDF through the interface.
2. Enter a question related to the document.
3. Review the generated answer.
4. Check the source and page citations to verify the supporting evidence.

## Evaluation

The repository includes an evaluation workflow for testing document retrieval and answer behavior.

Run the evaluation script from the project root:

```bash
python evaluation/evaluate_rag.py
```

Evaluation results depend on the test dataset, document content, retrieval settings, and selected answer-generation method. Use representative test cases to assess retrieval quality, citation correctness, and the handling of questions that cannot be answered from the source material.

## Current Scope and Limitations

- Retrieval quality depends on the document content, text extraction, embeddings, and search configuration.
- Answers generated without an external LLM may be extractive rather than fully synthesized.
- Scanned or image-heavy PDFs may require OCR support.
- The application should be treated as a local development project unless authentication, access controls, and deployment security have been implemented and tested.

## Potential Improvements

- Multi-document collections and isolated document sessions.
- Hybrid keyword and semantic retrieval.
- Cross-encoder reranking for improved retrieval relevance.
- More comprehensive RAG evaluation datasets and automated regression tests.
- Streaming responses and persistent chat history.
- Docker-based deployment and automated testing.
- Support for additional document formats.
- Tool-calling workflows for more advanced document analysis.

## Learning Objectives

This project provides practical exposure to:

- Retrieval-Augmented Generation (RAG).
- Embedding models and vector similarity search.
- Document ingestion and text preprocessing.
- Grounded LLM response generation.
- Source attribution and evaluation of AI applications.
- Python application development and API integration.

## Author

**Anitha Gummuluri**

Computer Science Engineering Student | AI Engineering | Full-Stack Development

- GitHub: [anithagummuluri12345](https://github.com/anithagummuluri12345)

---

_This project is intended for learning and experimentation with AI-powered document intelligence. See the repository's license and dependency licenses for applicable usage terms._
