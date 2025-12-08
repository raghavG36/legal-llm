"""Tests for LLM client wrapper."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.models.llm import LLMClient


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock(spec=AutoTokenizer)
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.model_max_length = 1024
    tokenizer.__len__ = lambda x: 50257  # GPT-2 vocab size
    return tokenizer


@pytest.fixture
def mock_model():
    """Create a mock model."""
    model = MagicMock(spec=AutoModelForCausalLM)
    model.config.max_position_embeddings = 1024
    model.config.model_type = "gpt2"
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    return model


def test_llm_client_init_default(mock_tokenizer, mock_model):
    """Test initializing LLMClient with default settings."""
    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                assert client.model_name == "gpt2"
                assert client.device == "cpu"
                assert client.max_seq_length == 1024


def test_llm_client_init_custom(mock_tokenizer, mock_model):
    """Test initializing LLMClient with custom settings."""
    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient(model_name="custom-model", device="cpu")

                assert client.model_name == "custom-model"
                assert client.device == "cpu"


def test_llm_client_init_auto_device(mock_tokenizer, mock_model):
    """Test device auto-detection."""
    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient(device="auto")
                assert client.device == "cpu"

            with patch("torch.cuda.is_available", return_value=True):
                client = LLMClient(device="auto")
                assert client.device == "cuda"


def test_llm_client_set_pad_token(mock_tokenizer, mock_model):
    """Test that pad token is set if missing."""
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<|endoftext|>"

    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                assert mock_tokenizer.pad_token == "<|endoftext|>"


def test_llm_client_generate(mock_tokenizer, mock_model):
    """Test text generation."""
    # Mock tokenizer behavior
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    mock_tokenizer.decode.return_value = "Generated text here."

    # Mock model generation
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])

    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                # Mock the tokenizer call
                def mock_tokenize(*args, **kwargs):
                    return {
                        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
                    }

                mock_tokenizer.__call__ = MagicMock(side_effect=mock_tokenize)

                result = client.generate("Test prompt", max_new_tokens=10)

                assert result == "Generated text here."
                mock_model.generate.assert_called_once()


def test_llm_client_generate_empty_prompt(mock_tokenizer, mock_model):
    """Test generation with empty prompt."""
    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                with pytest.raises(ValueError, match="Prompt cannot be empty"):
                    client.generate("")

                with pytest.raises(ValueError, match="Prompt cannot be empty"):
                    client.generate("   ")


def test_llm_client_generate_custom_params(mock_tokenizer, mock_model):
    """Test generation with custom parameters."""
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.decode.return_value = "Generated"

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11]])

    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                def mock_tokenize(*args, **kwargs):
                    return {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.tensor([[1, 1, 1]]),
                    }

                mock_tokenizer.__call__ = MagicMock(side_effect=mock_tokenize)

                client.generate(
                    "Test",
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                )

                # Verify parameters were passed
                call_kwargs = mock_model.generate.call_args[1]
                assert call_kwargs["max_new_tokens"] == 50
                assert call_kwargs["temperature"] == 0.7
                assert call_kwargs["do_sample"] is True


def test_llm_client_generate_greedy(mock_tokenizer, mock_model):
    """Test greedy generation (no sampling)."""
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    mock_tokenizer.decode.return_value = "Generated"

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11]])

    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                def mock_tokenize(*args, **kwargs):
                    return {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.tensor([[1, 1, 1]]),
                    }

                mock_tokenizer.__call__ = MagicMock(side_effect=mock_tokenize)

                client.generate("Test", do_sample=False)

                call_kwargs = mock_model.generate.call_args[1]
                assert call_kwargs["do_sample"] is False
                assert "temperature" not in call_kwargs


def test_llm_client_generate_stop_sequences(mock_tokenizer, mock_model):
    """Test generation with stop sequences."""
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    # Return text that contains stop sequence
    mock_tokenizer.decode.return_value = "Generated text. Stop here. More text."

    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 10, 11]])

    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                def mock_tokenize(*args, **kwargs):
                    return {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.tensor([[1, 1, 1]]),
                    }

                mock_tokenizer.__call__ = MagicMock(side_effect=mock_tokenize)

                result = client.generate("Test", stop_sequences=["Stop here"])

                # Should stop at the stop sequence
                assert result == "Generated text. "


def test_llm_client_generate_truncation(mock_tokenizer, mock_model):
    """Test that long prompts are truncated."""
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1] * 500]),  # Long input
        "attention_mask": torch.tensor([[1] * 500]),
    }
    mock_tokenizer.decode.return_value = "Generated"

    mock_model.generate.return_value = torch.tensor([[1] * 510])

    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                def mock_tokenize(*args, **kwargs):
                    # Simulate truncation
                    max_len = kwargs.get("max_length", 1024)
                    return {
                        "input_ids": torch.tensor([[1] * min(500, max_len)]),
                        "attention_mask": torch.tensor([[1] * min(500, max_len)]),
                    }

                mock_tokenizer.__call__ = MagicMock(side_effect=mock_tokenize)

                # Should not raise error even with long prompt
                result = client.generate("A" * 10000)
                assert result == "Generated"


def test_llm_client_get_model_info(mock_tokenizer, mock_model):
    """Test getting model information."""
    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                info = client.get_model_info()

                assert info["model_name"] == "gpt2"
                assert info["device"] == "cpu"
                assert info["max_seq_length"] == 1024
                assert "vocab_size" in info
                assert "model_type" in info


def test_llm_client_repr(mock_tokenizer, mock_model):
    """Test string representation."""
    with patch("app.models.llm.AutoTokenizer.from_pretrained", return_value=mock_tokenizer):
        with patch("app.models.llm.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
            with patch("torch.cuda.is_available", return_value=False):
                client = LLMClient()

                repr_str = repr(client)

                assert "LLMClient" in repr_str
                assert "gpt2" in repr_str
                assert "cpu" in repr_str
                assert "1024" in repr_str


def test_llm_client_error_handling():
    """Test error handling during model loading."""
    with patch("app.models.llm.AutoTokenizer.from_pretrained") as mock_tokenizer:
        mock_tokenizer.side_effect = Exception("Model not found")

        with pytest.raises(Exception, match="Model not found"):
            LLMClient(model_name="nonexistent-model")

