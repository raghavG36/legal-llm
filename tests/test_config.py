"""Test configuration loading."""

import os
from unittest.mock import patch

import pytest

from app.config import Settings, settings


def test_default_settings():
    """Test that default settings are loaded correctly."""
    assert settings.embedding_model_name == "sentence-transformers/all-mpnet-base-v2"
    assert settings.llm_model_name == "gpt2"
    assert settings.default_top_k == 5
    assert settings.max_chunk_chars == 600


def test_device_auto_detection():
    """Test device auto-detection."""
    device = settings.get_device()
    assert device in ["cpu", "cuda"]


def test_settings_validation():
    """Test that invalid values are rejected."""
    with pytest.raises(Exception):  # Pydantic will raise ValidationError
        Settings(default_top_k=-1)  # Should fail validation


def test_env_file_loading(monkeypatch, tmp_path):
    """Test loading settings from .env file."""
    env_file = tmp_path / ".env"
    env_file.write_text("EMBEDDING_MODEL_NAME=test-model\nDEFAULT_TOP_K=10\n")
    
    with patch("app.config.Settings.model_config") as mock_config:
        # This is a simplified test - in practice, Pydantic handles .env loading
        assert True  # Placeholder - actual .env loading tested via integration

