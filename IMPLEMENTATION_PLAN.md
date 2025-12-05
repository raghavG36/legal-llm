# Legal RAG System - Implementation Plan

## Overview
This document outlines a phased approach to building a production-ready RAG system for legal documents. Each phase is designed to be independently testable and builds upon previous phases.

---

## Phase 1: Project Foundation & Configuration
**Goal**: Set up project structure, dependencies, configuration, and logging

**Deliverables**:
- Project directory structure
- `requirements.txt` with all dependencies
- `.env.example` file
- `.gitignore`
- `app/config.py` - Pydantic-based configuration
- `app/logging_config.py` - Structured logging setup
- Basic `README.md` with setup instructions

**Testing**: Verify config loads from .env, logging works correctly

**Dependencies**: None

---

## Phase 2: PDF Ingestion Module
**Goal**: Extract text from legal PDF documents

**Deliverables**:
- `app/ingestion/pdf_loader.py` - PDF text extraction using pypdf
- Handle edge cases (empty pages, malformed PDFs)
- Function to load multiple PDFs from a folder
- Basic error handling and logging

**Testing**: 
- `tests/test_pdf_loader.py` - Test with sample PDFs
- Test empty/malformed PDF handling

**Dependencies**: Phase 1

---

## Phase 3: Legal-Aware Chunking
**Goal**: Implement intelligent chunking for legal documents

**Deliverables**:
- `app/rag/chunking.py` - Legal-aware chunking functions
- Heuristics for section/clause detection
- Target chunk size management (300-600 chars)
- No mid-sentence breaks

**Testing**:
- `tests/test_chunking.py` - Test various legal document formats
- Test chunk size boundaries
- Test section/clause detection

**Dependencies**: Phase 2

---

## Phase 4: Embedding Model Wrapper
**Goal**: Create abstraction for sentence embeddings

**Deliverables**:
- `app/models/embeddings.py` - EmbeddingModel class
- Load SentenceTransformer model (default: all-mpnet-base-v2)
- Device management (CPU/CUDA auto-detection)
- Batch encoding support
- Type hints and docstrings

**Testing**:
- Test encoding with sample texts
- Test device placement
- Test batch processing

**Dependencies**: Phase 1

---

## Phase 5: Vector Store (PyTorch Tensors)
**Goal**: In-memory vector store using PyTorch tensors

**Deliverables**:
- `app/rag/vector_store.py` - TensorVectorStore class
- PyTorch tensor storage for embeddings
- Metadata storage (doc_id, chunk_text, etc.)
- Cosine similarity search implementation
- Top-k retrieval

**Testing**:
- `tests/test_vector_store.py` - Test add/search operations
- Test cosine similarity correctness
- Test edge cases (empty store, k > num_items)

**Dependencies**: Phase 4

---

## Phase 6: Retriever Module
**Goal**: Query embedding and retrieval logic

**Deliverables**:
- `app/rag/retriever.py` - Retriever class
- Query encoding
- Integration with vector store
- Return formatted results with scores

**Testing**:
- `tests/test_retriever.py` - Test retrieval with sample queries
- Test top-k behavior

**Dependencies**: Phase 4, Phase 5

---

## Phase 7: LLM Wrapper
**Goal**: Local LLM integration for generation

**Deliverables**:
- `app/models/llm.py` - LLMClient class
- HuggingFace transformers integration
- Tokenization and generation
- Sequence length handling
- Temperature and max_tokens configuration

**Testing**:
- Test generation with sample prompts
- Test sequence truncation
- Test device placement

**Dependencies**: Phase 1

---

## Phase 8: Legal Prompt Builder
**Goal**: Create legal-safe RAG prompts

**Deliverables**:
- `app/rag/prompt_builder.py` - Prompt building functions
- Legal-specific instructions
- Context formatting
- Safety guidelines (no hallucination, cite sources)

**Testing**:
- Test prompt structure
- Test context inclusion
- Test safety instructions

**Dependencies**: None (standalone)

---

## Phase 9: RAG Pipeline Orchestrator
**Goal**: Wire all components together

**Deliverables**:
- `app/rag/pipeline.py` - RAGPipeline class
- Integration of retriever, prompt builder, and LLM
- Complete answer generation flow
- Response formatting with context

**Testing**:
- `tests/test_rag_pipeline.py` - End-to-end pipeline test
- Mock LLM for faster testing
- Test response structure

**Dependencies**: Phase 6, Phase 7, Phase 8

---

## Phase 10: Index Builder
**Goal**: Build and manage document indices

**Deliverables**:
- `app/ingestion/index_builder.py` - Index building functions
- Process folder of PDFs → chunks → embeddings → store
- Batch processing for efficiency
- Metadata generation
- `scripts/build_index.py` - CLI script for index building
- Optional: Save/load index to disk

**Testing**:
- Test index building from folder
- Test batch processing
- Test metadata correctness

**Dependencies**: Phase 2, Phase 3, Phase 4, Phase 5

---

## Phase 11: FastAPI API Layer
**Goal**: HTTP API for upload and query

**Deliverables**:
- `app/api/schemas.py` - Pydantic request/response models
- `app/api/routers.py` - API route handlers
- `app/api/main.py` - FastAPI app factory
- POST `/api/upload-pdf` - PDF upload endpoint
- POST `/api/legal-query` - Query endpoint
- Error handling and validation
- Integration with RAG pipeline

**Testing**:
- Test API endpoints with HTTP requests
- Test error cases
- Test validation

**Dependencies**: Phase 9, Phase 10

---

## Phase 12: Integration Testing & Documentation
**Goal**: Final polish and documentation

**Deliverables**:
- Comprehensive README.md
- Example usage and curl commands
- Integration tests
- Code review and cleanup
- Type hint verification
- Docstring completeness check

**Testing**:
- End-to-end integration test
- Manual testing with real legal PDFs
- Performance validation

**Dependencies**: All previous phases

---

## Summary

**Total Phases**: 12

**Estimated Complexity**:
- Phases 1-3: Foundation (Low-Medium)
- Phases 4-6: Core ML components (Medium)
- Phases 7-9: Generation & Pipeline (Medium)
- Phases 10-11: Integration (Medium-High)
- Phase 12: Polish (Low)

**Critical Path**: 1 → 2 → 3 → 4 → 5 → 6 → 9 → 11

**Parallelizable**: 
- Phase 7 (LLM) can be done in parallel with Phase 4-6
- Phase 8 (Prompt Builder) is independent

---

## Next Steps

1. Review this plan
2. Confirm Phase 1 to begin
3. After each phase completion, review and confirm next phase

