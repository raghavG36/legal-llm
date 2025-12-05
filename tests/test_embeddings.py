"""Tests for embedding model wrapper."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from sentence_transformers import SentenceTransformer

from app.models.embeddings import EmbeddingModel


@pytest.fixture
def mock_sentence_transformer():
    """Create a mock SentenceTransformer for testing."""
    mock_model = MagicMock(spec=SentenceTransformer)
    # Mock the get_sentence_embedding_dimension method
    mock_model.get_sentence_embedding_dimension.return_value = 768
    return mock_model


def test_embedding_model_init_default():
    """Test initializing EmbeddingModel with default settings."""
    with patch("app.models.embeddings.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model

        embedder = EmbeddingModel()
        assert embedder.model_name == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.device in ["cpu", "cuda"]


def test_embedding_model_init_custom():
    """Test initializing EmbeddingModel with custom settings."""
    with patch("app.models.embeddings.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        assert embedder.model_name == "test-model"
        assert embedder.device == "cpu"


def test_embedding_model_init_auto_device():
    """Test device auto-detection."""
    with patch("app.models.embeddings.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        mock_st.return_value = mock_model

        with patch("torch.cuda.is_available", return_value=False):
            embedder = EmbeddingModel(device="auto")
            assert embedder.device == "cpu"

        with patch("torch.cuda.is_available", return_value=True):
            embedder = EmbeddingModel(device="auto")
            assert embedder.device == "cuda"


def test_encode_single_text(mock_sentence_transformer):
    """Test encoding a single text string."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        # Mock encode to return a tensor
        mock_embedding = torch.randn(1, 768)
        mock_sentence_transformer.encode.return_value = mock_embedding

        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        result = embedder.encode("This is a test sentence.")

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 768)
        mock_sentence_transformer.encode.assert_called_once()


def test_encode_list_of_texts(mock_sentence_transformer):
    """Test encoding a list of texts."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        # Mock encode to return a tensor for multiple texts
        mock_embedding = torch.randn(3, 768)
        mock_sentence_transformer.encode.return_value = mock_embedding

        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        texts = ["Text 1", "Text 2", "Text 3"]
        result = embedder.encode(texts)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 768)
        mock_sentence_transformer.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_tensor=True,
            device="cpu",
        )


def test_encode_empty_list(mock_sentence_transformer):
    """Test encoding an empty list."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        result = embedder.encode([])

        assert isinstance(result, torch.Tensor)
        assert result.shape == (0, 768)
        # Should not call model.encode for empty list
        mock_sentence_transformer.encode.assert_not_called()


def test_encode_batch_size(mock_sentence_transformer):
    """Test encoding with custom batch size."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        mock_embedding = torch.randn(5, 768)
        mock_sentence_transformer.encode.return_value = mock_embedding

        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        texts = ["Text"] * 5
        embedder.encode(texts, batch_size=2)

        # Verify batch_size was passed
        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["batch_size"] == 2


def test_encode_normalize_embeddings(mock_sentence_transformer):
    """Test encoding with normalization control."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        mock_embedding = torch.randn(1, 768)
        mock_sentence_transformer.encode.return_value = mock_embedding

        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        embedder.encode("Test", normalize_embeddings=False)

        call_kwargs = mock_sentence_transformer.encode.call_args[1]
        assert call_kwargs["normalize_embeddings"] is False


def test_get_embedding_dim(mock_sentence_transformer):
    """Test getting embedding dimension."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        dim = embedder.get_embedding_dim()

        assert dim == 768
        mock_sentence_transformer.get_sentence_embedding_dimension.assert_called_once()


def test_get_embedding_dim_fallback(mock_sentence_transformer):
    """Test getting embedding dimension when method not available."""
    # Remove the method
    del mock_sentence_transformer.get_sentence_embedding_dimension

    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        mock_embedding = torch.randn(1, 512)
        mock_sentence_transformer.encode.return_value = mock_embedding

        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        dim = embedder.get_embedding_dim()

        assert dim == 512


def test_encode_device_placement(mock_sentence_transformer):
    """Test that embeddings are placed on correct device."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        # Create embedding on different device
        mock_embedding = torch.randn(1, 768)
        if torch.cuda.is_available():
            mock_embedding = mock_embedding.cuda()
        mock_sentence_transformer.encode.return_value = mock_embedding

        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        result = embedder.encode("Test")

        # Should be moved to CPU
        assert result.device.type == "cpu"


def test_encode_error_handling(mock_sentence_transformer):
    """Test error handling during encoding."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        mock_sentence_transformer.encode.side_effect = Exception("Encoding error")

        embedder = EmbeddingModel(model_name="test-model", device="cpu")

        with pytest.raises(Exception, match="Encoding error"):
            embedder.encode("Test")


def test_repr(mock_sentence_transformer):
    """Test string representation."""
    with patch("app.models.embeddings.SentenceTransformer", return_value=mock_sentence_transformer):
        embedder = EmbeddingModel(model_name="test-model", device="cpu")
        repr_str = repr(embedder)

        assert "EmbeddingModel" in repr_str
        assert "test-model" in repr_str
        assert "cpu" in repr_str
        assert "768" in repr_str

