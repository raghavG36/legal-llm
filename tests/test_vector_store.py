"""Tests for vector store."""

import pytest
import torch

from app.rag.vector_store import TensorVectorStore


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    # Create normalized embeddings (for cosine similarity)
    embeddings = torch.randn(5, 768)
    # Normalize to unit length
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings


@pytest.fixture
def sample_metadatas():
    """Create sample metadata for testing."""
    return [
        {"doc_id": "doc1", "chunk_text": "First chunk", "section": "1"},
        {"doc_id": "doc1", "chunk_text": "Second chunk", "section": "2"},
        {"doc_id": "doc2", "chunk_text": "Third chunk", "section": "1"},
        {"doc_id": "doc2", "chunk_text": "Fourth chunk", "section": "2"},
        {"doc_id": "doc3", "chunk_text": "Fifth chunk", "section": "1"},
    ]


def test_vector_store_init():
    """Test vector store initialization."""
    store = TensorVectorStore(dim=768, device="cpu")
    assert store.dim == 768
    assert store.device == "cpu"
    assert store.num_vectors == 0
    assert len(store) == 0


def test_vector_store_add(sample_embeddings, sample_metadatas):
    """Test adding embeddings to the store."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    assert store.num_vectors == 5
    assert len(store) == 5
    assert store.embeddings.shape == (5, 768)
    assert len(store.metadatas) == 5


def test_vector_store_add_multiple_batches(sample_embeddings, sample_metadatas):
    """Test adding embeddings in multiple batches."""
    store = TensorVectorStore(dim=768, device="cpu")

    # Add first batch
    store.add(sample_embeddings[:3], sample_metadatas[:3])
    assert store.num_vectors == 3

    # Add second batch
    store.add(sample_embeddings[3:], sample_metadatas[3:])
    assert store.num_vectors == 5
    assert store.embeddings.shape == (5, 768)


def test_vector_store_add_dimension_mismatch():
    """Test adding embeddings with wrong dimension."""
    store = TensorVectorStore(dim=768, device="cpu")
    wrong_embeddings = torch.randn(2, 512)  # Wrong dimension
    metadatas = [{"doc_id": "doc1", "chunk_text": "text"} for _ in range(2)]

    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add(wrong_embeddings, metadatas)


def test_vector_store_add_metadata_length_mismatch(sample_embeddings):
    """Test adding embeddings with mismatched metadata length."""
    store = TensorVectorStore(dim=768, device="cpu")
    metadatas = [{"doc_id": "doc1", "chunk_text": "text"}]  # Only 1 metadata for 5 embeddings

    with pytest.raises(ValueError, match="Metadata length mismatch"):
        store.add(sample_embeddings, metadatas)


def test_vector_store_add_missing_chunk_text(sample_embeddings):
    """Test adding metadata without required 'chunk_text' field."""
    store = TensorVectorStore(dim=768, device="cpu")
    metadatas = [{"doc_id": "doc1"}]  # Missing chunk_text

    with pytest.raises(ValueError, match="missing required field 'chunk_text'"):
        store.add(sample_embeddings[:1], metadatas)


def test_vector_store_add_missing_doc_id(sample_embeddings):
    """Test adding metadata without required 'doc_id' field."""
    store = TensorVectorStore(dim=768, device="cpu")
    metadatas = [{"chunk_text": "text"}]  # Missing doc_id

    with pytest.raises(ValueError, match="missing required field 'doc_id'"):
        store.add(sample_embeddings[:1], metadatas)


def test_vector_store_search(sample_embeddings, sample_metadatas):
    """Test searching the vector store."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    # Create a query embedding (normalized)
    query = torch.randn(1, 768)
    query = query / query.norm(dim=1, keepdim=True)

    results = store.search(query, k=3)

    assert len(results) == 3
    assert all("score" in r for r in results)
    assert all("metadata" in r for r in results)
    # Scores should be in descending order
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_vector_store_search_empty():
    """Test searching an empty vector store."""
    store = TensorVectorStore(dim=768, device="cpu")
    query = torch.randn(1, 768)

    results = store.search(query, k=5)
    assert results == []


def test_vector_store_search_k_larger_than_vectors(sample_embeddings, sample_metadatas):
    """Test searching with k larger than number of vectors."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    query = torch.randn(1, 768)
    query = query / query.norm(dim=1, keepdim=True)

    results = store.search(query, k=100)  # More than available

    assert len(results) == 5  # Should return all available


def test_vector_store_search_min_score(sample_embeddings, sample_metadatas):
    """Test searching with minimum score threshold."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    # Use one of the stored embeddings as query (should have high similarity)
    query = sample_embeddings[0:1]  # Shape: (1, 768)

    results = store.search(query, k=5, min_score=0.5)
    # The first result should have very high similarity (close to 1.0)
    assert len(results) > 0
    assert results[0]["score"] >= 0.5


def test_vector_store_search_query_shape_1d(sample_embeddings, sample_metadatas):
    """Test searching with 1D query tensor."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    query = torch.randn(768)  # 1D tensor
    query = query / query.norm()

    results = store.search(query, k=3)
    assert len(results) == 3


def test_vector_store_search_query_dimension_mismatch(sample_embeddings, sample_metadatas):
    """Test searching with query of wrong dimension."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    query = torch.randn(1, 512)  # Wrong dimension

    with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
        store.search(query, k=3)


def test_vector_store_clear(sample_embeddings, sample_metadatas):
    """Test clearing the vector store."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    assert store.num_vectors == 5
    store.clear()

    assert store.num_vectors == 0
    assert store.embeddings is None
    assert len(store.metadatas) == 0


def test_vector_store_repr(sample_embeddings, sample_metadatas):
    """Test string representation."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    repr_str = repr(store)
    assert "TensorVectorStore" in repr_str
    assert "768" in repr_str
    assert "5" in repr_str


def test_vector_store_get_stats(sample_embeddings, sample_metadatas):
    """Test getting statistics."""
    store = TensorVectorStore(dim=768, device="cpu")
    store.add(sample_embeddings, sample_metadatas)

    stats = store.get_stats()
    assert stats["num_vectors"] == 5
    assert stats["dim"] == 768
    assert stats["device"] == "cpu"
    assert "memory_usage_mb" in stats


def test_vector_store_cosine_similarity_correctness():
    """Test that cosine similarity is computed correctly."""
    store = TensorVectorStore(dim=3, device="cpu")

    # Create known embeddings
    # Embedding 1: [1, 0, 0]
    # Embedding 2: [0, 1, 0]
    # Embedding 3: [1, 1, 0] / sqrt(2) (normalized)
    embeddings = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0 / (2**0.5), 1.0 / (2**0.5), 0.0]]
    )
    # Normalize
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    metadatas = [
        {"doc_id": "doc1", "chunk_text": "chunk1"},
        {"doc_id": "doc2", "chunk_text": "chunk2"},
        {"doc_id": "doc3", "chunk_text": "chunk3"},
    ]

    store.add(embeddings, metadatas)

    # Query with embedding 1
    query = torch.tensor([[1.0, 0.0, 0.0]])
    query = query / query.norm(dim=1, keepdim=True)

    results = store.search(query, k=3)

    # First result should be embedding 1 (cosine similarity = 1.0)
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert results[0]["metadata"]["chunk_text"] == "chunk1"

    # Second result should be embedding 3 (cosine similarity = 1/sqrt(2) ≈ 0.707)
    assert results[1]["score"] == pytest.approx(1.0 / (2**0.5), abs=1e-5)

    # Third result should be embedding 2 (cosine similarity = 0.0)
    assert results[2]["score"] == pytest.approx(0.0, abs=1e-5)


def test_vector_store_device_placement():
    """Test that embeddings are stored on correct device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    store = TensorVectorStore(dim=768, device="cuda")
    embeddings = torch.randn(2, 768)
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    metadatas = [{"doc_id": "doc1", "chunk_text": "text1"}, {"doc_id": "doc2", "chunk_text": "text2"}]

    store.add(embeddings, metadatas)

    assert store.embeddings.device.type == "cuda"

    # Search should work with CPU query (will be moved)
    query = torch.randn(1, 768)
    query = query / query.norm(dim=1, keepdim=True)
    results = store.search(query, k=2)
    assert len(results) == 2

