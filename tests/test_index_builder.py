"""Tests for index builder."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from app.ingestion.index_builder import (
    _extract_section_info,
    build_index_from_folder,
    load_index,
    save_index,
)
from app.models.embeddings import EmbeddingModel
from app.rag.vector_store import TensorVectorStore


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock(spec=EmbeddingModel)
    embedder.get_embedding_dim.return_value = 768
    embedder.device = "cpu"
    return embedder


@pytest.fixture
def sample_pdf_folder(tmp_path):
    """Create a temporary folder with mock PDF files."""
    # Create mock PDF files (just text files for testing)
    pdf1 = tmp_path / "doc1.pdf"
    pdf2 = tmp_path / "doc2.pdf"
    pdf1.write_text("Section 1: First document content. " * 100)
    pdf2.write_text("Section 2: Second document content. " * 100)
    return tmp_path


def test_extract_section_info():
    """Test extracting section information from chunk text."""
    # Test with section
    chunk = "Section 302: This is the content about theft."
    info = _extract_section_info(chunk)
    assert "section" in info
    assert info["section"] == "302"

    # Test with clause
    chunk = "Clause 5: This is about obligations."
    info = _extract_section_info(chunk)
    assert "clause" in info
    assert info["clause"] == "5"

    # Test with no markers
    chunk = "This is plain text without markers."
    info = _extract_section_info(chunk)
    assert info == {}


def test_build_index_from_folder(mock_embedder, sample_pdf_folder):
    """Test building index from folder."""
    # Mock the PDF loader and embedder
    with patch("app.ingestion.index_builder.load_legal_corpus_from_folder") as mock_loader:
        mock_loader.return_value = [
            ("doc1", "Section 1: First document. " * 50),
            ("doc2", "Section 2: Second document. " * 50),
        ]

        # Mock chunking
        with patch("app.ingestion.index_builder.chunk_legal_text") as mock_chunk:
            mock_chunk.return_value = ["Chunk 1", "Chunk 2"]

            # Mock embedding
            mock_embeddings = torch.randn(4, 768)  # 2 docs * 2 chunks
            mock_embeddings = mock_embeddings / mock_embeddings.norm(dim=1, keepdim=True)
            mock_embedder.encode.return_value = mock_embeddings

            vector_store = build_index_from_folder(
                folder_path=str(sample_pdf_folder),
                embedder=mock_embedder,
            )

            assert len(vector_store) == 4
            assert vector_store.dim == 768
            # Verify embedder was called
            assert mock_embedder.encode.called


def test_build_index_from_folder_existing_store(mock_embedder, sample_pdf_folder):
    """Test building index with existing vector store."""
    existing_store = TensorVectorStore(dim=768, device="cpu")

    with patch("app.ingestion.index_builder.load_legal_corpus_from_folder") as mock_loader:
        mock_loader.return_value = [("doc1", "Content " * 50)]

        with patch("app.ingestion.index_builder.chunk_legal_text") as mock_chunk:
            mock_chunk.return_value = ["Chunk 1"]

            mock_embeddings = torch.randn(1, 768)
            mock_embeddings = mock_embeddings / mock_embeddings.norm(dim=1, keepdim=True)
            mock_embedder.encode.return_value = mock_embeddings

            vector_store = build_index_from_folder(
                folder_path=str(sample_pdf_folder),
                embedder=mock_embedder,
                vector_store=existing_store,
            )

            assert vector_store == existing_store
            assert len(vector_store) == 1


def test_build_index_from_folder_dimension_mismatch(mock_embedder):
    """Test building index with dimension mismatch."""
    existing_store = TensorVectorStore(dim=512, device="cpu")  # Different dimension

    with pytest.raises(ValueError, match="dimension mismatch"):
        build_index_from_folder(
            folder_path="./test",
            embedder=mock_embedder,
            vector_store=existing_store,
        )


def test_build_index_from_folder_no_pdfs(mock_embedder, tmp_path):
    """Test building index with no PDFs."""
    with patch("app.ingestion.index_builder.load_legal_corpus_from_folder") as mock_loader:
        mock_loader.return_value = []

        with pytest.raises(ValueError, match="No valid PDFs found"):
            build_index_from_folder(
                folder_path=str(tmp_path),
                embedder=mock_embedder,
            )


def test_build_index_from_folder_batch_processing(mock_embedder):
    """Test that embeddings are generated in batches."""
    with patch("app.ingestion.index_builder.load_legal_corpus_from_folder") as mock_loader:
        mock_loader.return_value = [("doc1", "Content " * 100)]

        with patch("app.ingestion.index_builder.chunk_legal_text") as mock_chunk:
            # Create many chunks
            mock_chunk.return_value = [f"Chunk {i}" for i in range(50)]

            # Mock encode to return batch embeddings
            def mock_encode(texts, **kwargs):
                batch_size = len(texts)
                return torch.randn(batch_size, 768)

            mock_embedder.encode.side_effect = mock_encode

            vector_store = build_index_from_folder(
                folder_path="./test",
                embedder=mock_embedder,
                batch_size=10,
            )

            # Verify encode was called multiple times (batches)
            assert mock_embedder.encode.call_count >= 5  # 50 chunks / 10 batch_size


def test_save_index(tmp_path):
    """Test saving index to disk."""
    # Create a vector store with sample data
    store = TensorVectorStore(dim=768, device="cpu")
    embeddings = torch.randn(5, 768)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    metadatas = [
        {"doc_id": "doc1", "chunk_text": "chunk1"},
        {"doc_id": "doc2", "chunk_text": "chunk2"},
    ] * 2 + [{"doc_id": "doc3", "chunk_text": "chunk5"}]
    store.add(embeddings, metadatas)

    save_path = tmp_path / "index"
    save_index(store, str(save_path))

    # Verify files were created
    assert (save_path / "embeddings.pt").exists()
    assert (save_path / "metadata.json").exists()
    assert (save_path / "store_info.json").exists()

    # Verify metadata file content
    with open(save_path / "metadata.json", "r") as f:
        loaded_metadatas = json.load(f)
    assert len(loaded_metadatas) == 5

    # Verify store info
    with open(save_path / "store_info.json", "r") as f:
        store_info = json.load(f)
    assert store_info["dim"] == 768
    assert store_info["num_vectors"] == 5


def test_load_index(tmp_path):
    """Test loading index from disk."""
    # Create and save an index first
    store = TensorVectorStore(dim=768, device="cpu")
    embeddings = torch.randn(3, 768)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    metadatas = [
        {"doc_id": "doc1", "chunk_text": "chunk1"},
        {"doc_id": "doc2", "chunk_text": "chunk2"},
        {"doc_id": "doc3", "chunk_text": "chunk3"},
    ]
    store.add(embeddings, metadatas)

    save_path = tmp_path / "index"
    save_index(store, str(save_path))

    # Load it back
    loaded_store = load_index(str(save_path), device="cpu")

    assert len(loaded_store) == 3
    assert loaded_store.dim == 768
    assert len(loaded_store.metadatas) == 3
    assert loaded_store.metadatas[0]["doc_id"] == "doc1"


def test_load_index_not_found(tmp_path):
    """Test loading non-existent index."""
    with pytest.raises(FileNotFoundError):
        load_index(str(tmp_path / "nonexistent"))


def test_build_index_metadata_structure(mock_embedder):
    """Test that metadata has correct structure."""
    with patch("app.ingestion.index_builder.load_legal_corpus_from_folder") as mock_loader:
        mock_loader.return_value = [("doc1", "Section 302: Content here.")]

        with patch("app.ingestion.index_builder.chunk_legal_text") as mock_chunk:
            mock_chunk.return_value = ["Section 302: Content here."]

            mock_embeddings = torch.randn(1, 768)
            mock_embeddings = mock_embeddings / mock_embeddings.norm(dim=1, keepdim=True)
            mock_embedder.encode.return_value = mock_embeddings

            vector_store = build_index_from_folder(
                folder_path="./test",
                embedder=mock_embedder,
            )

            # Check metadata structure
            metadata = vector_store.metadatas[0]
            assert "doc_id" in metadata
            assert "chunk_text" in metadata
            assert "chunk_index" in metadata
            assert "total_chunks" in metadata
            assert metadata["doc_id"] == "doc1"
            # Should have extracted section info
            assert "section" in metadata

