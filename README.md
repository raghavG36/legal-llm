# Legal RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for legal documents, built with Python, FastAPI, and PyTorch.

## Features

- **Legal Document Ingestion**: Extract text from PDF legal documents (acts, judgements, contracts)
- **Legal-Aware Chunking**: Intelligent chunking by sections, clauses, and paragraphs
- **Local Embeddings**: Sentence-Transformer models for semantic embeddings
- **In-Memory Vector Store**: PyTorch tensor-based vector store with cosine similarity search
- **Local LLM**: HuggingFace causal language models for answer generation
- **Legal-Safe Prompts**: RAG prompts with safety guidelines to prevent hallucination
- **REST API**: FastAPI endpoints for document upload and legal queries

## Tech Stack

- **Python**: 3.10+
- **Framework**: FastAPI
- **ML/DL**: PyTorch, Sentence-Transformers, HuggingFace Transformers
- **PDF Processing**: pypdf
- **Configuration**: Pydantic Settings

## Installation

### Prerequisites

- Python 3.10 or higher
- pip or conda

### Setup

1. **Clone the repository** (or navigate to the project directory)

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings
   ```

## Quick Start

### 1. Build the Index

First, place your legal PDF documents in a folder (default: `./legal_docs`), then build the index:

```bash
python scripts/build_index.py --folder ./legal_docs
```

This will:
- Extract text from all PDFs
- Chunk the text using legal-aware heuristics
- Generate embeddings for each chunk
- Store everything in an in-memory vector store

### 2. Start the API Server

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Query the System

**Using curl**:
```bash
curl -X POST "http://localhost:8000/api/legal-query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the penalty for theft?", "top_k": 5}'
```

**Using httpie**:
```bash
http POST http://localhost:8000/api/legal-query question="What is the penalty for theft?" top_k=5
```

**Using Python**:
```python
import requests

response = requests.post(
    "http://localhost:8000/api/legal-query",
    json={"question": "What is the penalty for theft?", "top_k": 5}
)
print(response.json())
```

## API Endpoints

### POST `/api/legal-query`

Query the RAG system with a legal question.

**Request Body**:
```json
{
  "question": "What is the penalty for theft?",
  "top_k": 5
}
```

**Response**:
```json
{
  "answer": "According to Section 302...",
  "context": [
    {
      "score": 0.87,
      "doc_id": "BNS",
      "chunk_text": "Section 302: Whoever commits...",
      "metadata": {}
    }
  ]
}
```

### POST `/api/upload-pdf`

Upload a new legal PDF and update the index.

**Request**: Multipart form data with `file` field

**Response**:
```json
{
  "doc_id": "document_name",
  "chunks_added": 42,
  "message": "PDF processed successfully"
}
```

## Configuration

Configuration is managed through environment variables (see `.env.example`). Key settings:

- `EMBEDDING_MODEL_NAME`: SentenceTransformer model (default: `all-mpnet-base-v2`)
- `LLM_MODEL_NAME`: HuggingFace model (default: `gpt2`)
- `DEVICE`: `auto`, `cpu`, or `cuda`
- `DEFAULT_TOP_K`: Number of chunks to retrieve (default: 5)
- `MAX_CHUNK_CHARS`: Maximum characters per chunk (default: 600)

## Project Structure

```
legal-llm/
├── app/
│   ├── api/           # FastAPI routes and schemas
│   ├── ingestion/     # PDF loading and index building
│   ├── models/        # Embedding and LLM wrappers
│   ├── rag/           # RAG components (chunking, vector store, retrieval)
│   ├── config.py      # Configuration management
│   └── logging_config.py
├── tests/             # Unit and integration tests
├── scripts/           # CLI scripts
├── requirements.txt
└── README.md
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

The codebase follows:
- Type hints throughout
- Comprehensive docstrings
- Modular architecture
- Separation of concerns

## License

[Add your license here]

## Contributing

[Add contribution guidelines if needed]

