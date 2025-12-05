"""Configuration management using Pydantic Settings."""

import os
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Model Configuration
    embedding_model_name: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="SentenceTransformer model for embeddings",
    )
    llm_model_name: str = Field(
        default="gpt2",
        description="HuggingFace causal LM model name",
    )

    # Device Configuration
    device: str = Field(
        default="auto",
        description="Device to use: 'auto', 'cpu', or 'cuda'. Auto-detects if 'auto'",
    )

    # RAG Configuration
    default_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Default number of chunks to retrieve",
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Maximum tokens to generate",
    )
    temperature: float = Field(
        default=0.4,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM generation (lower = more deterministic)",
    )

    # Chunking Configuration
    max_chunk_chars: int = Field(
        default=600,
        ge=100,
        le=2000,
        description="Maximum characters per chunk",
    )
    min_chunk_chars: int = Field(
        default=300,
        ge=50,
        le=1000,
        description="Minimum characters per chunk (target)",
    )

    # Paths
    legal_docs_folder: str = Field(
        default="./legal_docs",
        description="Default folder path for legal documents",
    )
    index_save_path: str = Field(
        default="./index",
        description="Path to save/load index files",
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI host",
    )
    api_port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="FastAPI port",
    )
    api_reload: bool = Field(
        default=False,
        description="Enable auto-reload for development",
    )

    # Logging Configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    )
    log_format: str = Field(
        default="json",
        description="Log format: 'json' or 'text'",
    )

    def get_device(self) -> Literal["cpu", "cuda"]:
        """Determine the device to use for computation."""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device.lower()


# Global settings instance
settings = Settings()

