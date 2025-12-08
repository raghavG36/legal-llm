"""Index builder for processing legal documents and building vector store."""

import os
from pathlib import Path
from typing import List, Optional

import torch

from app.config import settings
from app.logging_config import get_logger
from app.ingestion.pdf_loader import load_legal_corpus_from_folder
from app.models.embeddings import EmbeddingModel
from app.rag.chunking import chunk_legal_text
from app.rag.vector_store import TensorVectorStore

logger = get_logger(__name__)


def build_index_from_folder(
    folder_path: str,
    embedder: EmbeddingModel,
    vector_store: Optional[TensorVectorStore] = None,
    batch_size: int = 32,
    max_chunk_chars: int = None,
    min_chunk_chars: int = None,
) -> TensorVectorStore:
    """
    Build a vector store index from a folder of legal PDF documents.

    Process:
    1. Load all PDFs from the folder
    2. Chunk each document using legal-aware chunking
    3. Generate embeddings for all chunks in batches
    4. Create metadata for each chunk
    5. Add embeddings and metadata to vector store

    Args:
        folder_path: Path to folder containing PDF files.
        embedder: EmbeddingModel instance for generating embeddings.
        vector_store: Optional existing TensorVectorStore to add to.
                    If None, creates a new one.
        batch_size: Batch size for embedding generation.
        max_chunk_chars: Maximum characters per chunk. Defaults to config.
        min_chunk_chars: Minimum characters per chunk. Defaults to config.

    Returns:
        Populated TensorVectorStore instance.

    Raises:
        FileNotFoundError: If folder doesn't exist.
        ValueError: If no valid PDFs found or embedding dimension mismatch.
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    logger.info(
        f"Building index from folder: {folder_path}",
        extra={"folder": str(folder_path)},
    )

    # Step 1: Load PDFs
    logger.info("Loading PDFs from folder...")
    corpus = load_legal_corpus_from_folder(str(folder_path))

    if not corpus:
        raise ValueError(f"No valid PDFs found in {folder_path}")

    logger.info(f"Loaded {len(corpus)} document(s)")

    # Step 2: Initialize or use existing vector store
    if vector_store is None:
        embedding_dim = embedder.get_embedding_dim()
        device = embedder.device
        vector_store = TensorVectorStore(dim=embedding_dim, device=device)
        logger.info(f"Created new vector store (dim={embedding_dim}, device={device})")
    else:
        # Verify dimension match
        if vector_store.dim != embedder.get_embedding_dim():
            raise ValueError(
                f"Embedding dimension mismatch: embedder={embedder.get_embedding_dim()}, "
                f"vector_store={vector_store.dim}"
            )
        logger.info(f"Using existing vector store (dim={vector_store.dim})")

    # Step 3: Process each document
    all_chunks = []
    all_metadatas = []

    for doc_id, raw_text in corpus:
        logger.debug(f"Processing document: {doc_id}")

        # Chunk the document
        chunks = chunk_legal_text(
            raw_text,
            max_chunk_chars=max_chunk_chars,
            min_chunk_chars=min_chunk_chars,
        )

        if not chunks:
            logger.warning(f"No chunks created for document: {doc_id}")
            continue

        logger.debug(
            f"Created {len(chunks)} chunk(s) for {doc_id}",
            extra={"doc_id": doc_id, "num_chunks": len(chunks)},
        )

        # Create metadata for each chunk
        for i, chunk_text in enumerate(chunks):
            metadata = {
                "doc_id": doc_id,
                "chunk_text": chunk_text,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }

            # Try to extract section/clause info from chunk text
            section_info = _extract_section_info(chunk_text)
            if section_info:
                metadata.update(section_info)

            all_chunks.append(chunk_text)
            all_metadatas.append(metadata)

    if not all_chunks:
        raise ValueError("No chunks created from any documents")

    logger.info(
        f"Total chunks created: {len(all_chunks)}",
        extra={"total_chunks": len(all_chunks), "num_documents": len(corpus)},
    )

    # Step 4: Generate embeddings in batches
    logger.info(f"Generating embeddings for {len(all_chunks)} chunks (batch_size={batch_size})...")

    all_embeddings = []
    for i in range(0, len(all_chunks), batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_embeddings = embedder.encode(
            batch_chunks,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        all_embeddings.append(batch_embeddings)
        logger.debug(
            f"Embedded batch {i//batch_size + 1}/{(len(all_chunks) + batch_size - 1)//batch_size}",
            extra={"batch_start": i, "batch_size": len(batch_chunks)},
        )

    # Concatenate all embeddings
    embeddings_tensor = torch.cat(all_embeddings, dim=0)

    logger.info(
        f"Generated embeddings: shape {list(embeddings_tensor.shape)}",
        extra={"embedding_shape": list(embeddings_tensor.shape)},
    )

    # Step 5: Add to vector store
    logger.info("Adding embeddings to vector store...")
    vector_store.add(embeddings_tensor, all_metadatas)

    logger.info(
        f"Index building complete",
        extra={
            "total_vectors": len(vector_store),
            "num_documents": len(corpus),
            "total_chunks": len(all_chunks),
        },
    )

    return vector_store


def _extract_section_info(chunk_text: str) -> dict:
    """
    Extract section/clause information from chunk text.

    Args:
        chunk_text: Text chunk to analyze.

    Returns:
        Dictionary with section/clause info if found, empty dict otherwise.
    """
    import re

    info = {}

    # Look for section markers
    section_match = re.search(
        r"(?:Section|Sec\.?)\s+(\d+|[IVX]+)",
        chunk_text[:200],  # Check first 200 chars
        re.IGNORECASE,
    )
    if section_match:
        info["section"] = section_match.group(1)

    # Look for clause markers
    clause_match = re.search(
        r"(?:Clause|Cl\.?)\s+(\d+|[IVX]+)",
        chunk_text[:200],
        re.IGNORECASE,
    )
    if clause_match:
        info["clause"] = clause_match.group(1)

    return info


def save_index(vector_store: TensorVectorStore, save_path: str) -> None:
    """
    Save vector store to disk.

    Args:
        vector_store: TensorVectorStore to save.
        save_path: Directory path to save index files.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save embeddings tensor
    embeddings_path = save_path / "embeddings.pt"
    torch.save(vector_store.embeddings, embeddings_path)

    # Save metadata (as a list of dicts - can be saved as JSON or pickle)
    import json

    metadata_path = save_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(vector_store.metadatas, f, indent=2, ensure_ascii=False)

    # Save metadata about the store
    store_info = {
        "dim": vector_store.dim,
        "device": vector_store.device,
        "num_vectors": vector_store.num_vectors,
    }
    info_path = save_path / "store_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(store_info, f, indent=2)

    logger.info(f"Index saved to {save_path}", extra={"num_vectors": vector_store.num_vectors})


def load_index(load_path: str, device: str = "cpu") -> TensorVectorStore:
    """
    Load vector store from disk.

    Args:
        load_path: Directory path containing index files.
        device: Device to load tensors on.

    Returns:
        Loaded TensorVectorStore instance.

    Raises:
        FileNotFoundError: If index files don't exist.
    """
    load_path = Path(load_path)

    # Load store info
    info_path = load_path / "store_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Index info file not found: {info_path}")

    import json

    with open(info_path, "r", encoding="utf-8") as f:
        store_info = json.load(f)

    # Create vector store
    vector_store = TensorVectorStore(dim=store_info["dim"], device=device)

    # Load embeddings
    embeddings_path = load_path / "embeddings.pt"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    embeddings = torch.load(embeddings_path, map_location=device)
    embeddings = embeddings.to(device)

    # Load metadata
    metadata_path = load_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadatas = json.load(f)

    # Add to store
    vector_store.add(embeddings, metadatas)

    logger.info(
        f"Index loaded from {load_path}",
        extra={"num_vectors": len(vector_store), "dim": vector_store.dim},
    )

    return vector_store

