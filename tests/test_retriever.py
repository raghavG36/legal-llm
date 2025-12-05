"""Tests for retriever module."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from app.models.embeddings import EmbeddingModel
from app.rag.retriever import Retriever
from app.rag.vector_store import TensorVectorStore


@pytest.fixture
def mock_embedder():
    """Create a mock EmbeddingModel."""
    embedder = MagicMock(spec=EmbeddingModel)
    embedder.get_embedding_dim.return_value = 768
    embedder.model_name = "test-model"
    return embedder


@pytest.fixture
def mock_vector_store():
    """Create a mock TensorVectorStore."""
    store = MagicMock(spec=TensorVectorStore)
    store.dim = 768
    store.__len__ = lambda x: 10
    return store


@pytest.fixture
def real_vector_store():
    """Create a real TensorVectorStore with sample data."""
    store = TensorVectorStore(dim=768, device="cpu")
    # Add some sample embeddings
    embeddings = torch.randn(5, 768)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    metadatas = [
        {"doc_id": "doc1", "chunk_text": "First chunk about legal matters"},
        {"doc_id": "doc1", "chunk_text": "Second chunk about contracts"},
        {"doc_id": "doc2", "chunk_text": "Third chunk about penalties"},
        {"doc_id": "doc2", "chunk_text": "Fourth chunk about obligations"},
        {"doc_id": "doc3", "chunk_text": "Fifth chunk about rights"},
    ]
    store.add(embeddings, metadatas)
    return store


def test_retriever_init(mock_embedder, mock_vector_store):
    """Test retriever initialization."""
    retriever = Retriever(mock_embedder, mock_vector_store)
    assert retriever.embedder == mock_embedder
    assert retriever.vector_store == mock_vector_store


def test_retriever_init_dimension_mismatch():
    """Test retriever initialization with dimension mismatch."""
    embedder = MagicMock(spec=EmbeddingModel)
    embedder.get_embedding_dim.return_value = 512

    store = MagicMock(spec=TensorVectorStore)
    store.dim = 768

    with pytest.raises(ValueError, match="dimension mismatch"):
        Retriever(embedder, store)


def test_retriever_retrieve(mock_embedder, mock_vector_store):
    """Test retrieving chunks for a query."""
    # Mock encode to return a tensor
    query_embedding = torch.randn(1, 768)
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    mock_embedder.encode.return_value = query_embedding

    # Mock search to return results
    mock_vector_store.search.return_value = [
        {
            "score": 0.9,
            "metadata": {"doc_id": "doc1", "chunk_text": "Relevant chunk", "section": "1"},
        },
        {
            "score": 0.8,
            "metadata": {"doc_id": "doc2", "chunk_text": "Another chunk", "section": "2"},
        },
    ]

    retriever = Retriever(mock_embedder, mock_vector_store)
    results = retriever.retrieve("test query", k=2)

    assert len(results) == 2
    assert results[0]["score"] == 0.9
    assert results[0]["chunk_text"] == "Relevant chunk"
    assert results[0]["doc_id"] == "doc1"
    assert "metadata" in results[0]

    # Verify encode was called
    mock_embedder.encode.assert_called_once()
    # Verify search was called
    mock_vector_store.search.assert_called_once()


def test_retriever_retrieve_empty_query(mock_embedder, mock_vector_store):
    """Test retrieving with empty query."""
    retriever = Retriever(mock_embedder, mock_vector_store)

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("")

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve("   ")


def test_retriever_retrieve_custom_k(mock_embedder, mock_vector_store):
    """Test retrieving with custom k value."""
    query_embedding = torch.randn(1, 768)
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    mock_embedder.encode.return_value = query_embedding

    mock_vector_store.search.return_value = [
        {"score": 0.9, "metadata": {"doc_id": "doc1", "chunk_text": "chunk1"}},
    ] * 5

    retriever = Retriever(mock_embedder, mock_vector_store)
    results = retriever.retrieve("test query", k=3)

    # Verify k was passed to search
    call_kwargs = mock_vector_store.search.call_args[1]
    assert call_kwargs["k"] == 3


def test_retriever_retrieve_min_score(mock_embedder, mock_vector_store):
    """Test retrieving with minimum score threshold."""
    query_embedding = torch.randn(1, 768)
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    mock_embedder.encode.return_value = query_embedding

    mock_vector_store.search.return_value = [
        {"score": 0.9, "metadata": {"doc_id": "doc1", "chunk_text": "chunk1"}},
    ]

    retriever = Retriever(mock_embedder, mock_vector_store)
    results = retriever.retrieve("test query", min_score=0.5)

    # Verify min_score was passed to search
    call_kwargs = mock_vector_store.search.call_args[1]
    assert call_kwargs["min_score"] == 0.5


def test_retriever_retrieve_format(mock_embedder, mock_vector_store):
    """Test that results are properly formatted."""
    query_embedding = torch.randn(1, 768)
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    mock_embedder.encode.return_value = query_embedding

    mock_vector_store.search.return_value = [
        {
            "score": 0.85,
            "metadata": {
                "doc_id": "doc1",
                "chunk_text": "Test chunk text",
                "section": "Section 1",
                "page": 5,
            },
        },
    ]

    retriever = Retriever(mock_embedder, mock_vector_store)
    results = retriever.retrieve("test query")

    assert len(results) == 1
    result = results[0]
    assert result["score"] == 0.85
    assert result["chunk_text"] == "Test chunk text"
    assert result["doc_id"] == "doc1"
    assert "metadata" in result
    # Verify full metadata is preserved
    assert result["metadata"]["section"] == "Section 1"
    assert result["metadata"]["page"] == 5


def test_retriever_retrieve_batch(mock_embedder, mock_vector_store):
    """Test batch retrieval."""
    # Mock encode to return batch embeddings
    batch_embeddings = torch.randn(3, 768)
    batch_embeddings = batch_embeddings / batch_embeddings.norm(dim=1, keepdim=True)
    mock_embedder.encode.return_value = batch_embeddings

    # Mock search to return different results for each query
    def mock_search(**kwargs):
        query_idx = kwargs.get("query_embedding", None)
        if query_idx is not None:
            # Return different number of results based on some logic
            return [
                {
                    "score": 0.9 - i * 0.1,
                    "metadata": {"doc_id": f"doc{i+1}", "chunk_text": f"chunk{i+1}"},
                }
                for i in range(2)
            ]
        return []

    mock_vector_store.search.side_effect = mock_search

    retriever = Retriever(mock_embedder, mock_vector_store)
    queries = ["query 1", "query 2", "query 3"]
    all_results = retriever.retrieve_batch(queries, k=2)

    assert len(all_results) == 3
    assert all(len(results) == 2 for results in all_results)
    # Verify encode was called once with all queries
    mock_embedder.encode.assert_called_once_with(queries, normalize_embeddings=True)


def test_retriever_retrieve_batch_empty(mock_embedder, mock_vector_store):
    """Test batch retrieval with empty query list."""
    retriever = Retriever(mock_embedder, mock_vector_store)
    results = retriever.retrieve_batch([])

    assert results == []


def test_retriever_retrieve_encoding_error(mock_embedder, mock_vector_store):
    """Test error handling during query encoding."""
    mock_embedder.encode.side_effect = Exception("Encoding error")

    retriever = Retriever(mock_embedder, mock_vector_store)

    with pytest.raises(Exception, match="Encoding error"):
        retriever.retrieve("test query")


def test_retriever_retrieve_search_error(mock_embedder, mock_vector_store):
    """Test error handling during vector store search."""
    query_embedding = torch.randn(1, 768)
    query_embedding = query_embedding / query_embedding.norm(dim=1, keepdim=True)
    mock_embedder.encode.return_value = query_embedding

    mock_vector_store.search.side_effect = Exception("Search error")

    retriever = Retriever(mock_embedder, mock_vector_store)

    with pytest.raises(Exception, match="Search error"):
        retriever.retrieve("test query")


def test_retriever_repr(mock_embedder, mock_vector_store):
    """Test string representation."""
    retriever = Retriever(mock_embedder, mock_vector_store)
    repr_str = repr(retriever)

    assert "Retriever" in repr_str
    assert "test-model" in repr_str


def test_retriever_integration(real_vector_store):
    """Integration test with real vector store and mock embedder."""
    embedder = MagicMock(spec=EmbeddingModel)
    embedder.get_embedding_dim.return_value = 768
    embedder.model_name = "test-model"

    # Use one of the stored embeddings as query
    query_embedding = real_vector_store.embeddings[0:1]  # Shape: (1, 768)
    embedder.encode.return_value = query_embedding

    retriever = Retriever(embedder, real_vector_store)
    results = retriever.retrieve("test query", k=3)

    assert len(results) > 0
    assert all("score" in r for r in results)
    assert all("chunk_text" in r for r in results)
    assert all("doc_id" in r for r in results)
    assert all("metadata" in r for r in results)
    # First result should have high score (query matches stored embedding)
    assert results[0]["score"] > 0.9

