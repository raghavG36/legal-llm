#!/usr/bin/env python3
"""CLI script to build and manage document indices."""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.logging_config import setup_logging
from app.ingestion.index_builder import build_index_from_folder, load_index, save_index
from app.models.embeddings import EmbeddingModel
from app.rag.vector_store import TensorVectorStore


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Build or load a vector store index from legal PDF documents"
    )
    parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help=f"Folder path containing PDF files (default: {settings.legal_docs_folder})",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help=f"Path to save index (default: {settings.index_save_path})",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Path to load existing index from",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Embedding model name (default: {settings.embedding_model_name})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=f"Device to use: auto, cpu, or cuda (default: {settings.device})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--max-chunk-chars",
        type=int,
        default=None,
        help=f"Maximum characters per chunk (default: {settings.max_chunk_chars})",
    )
    parser.add_argument(
        "--min-chunk-chars",
        type=int,
        default=None,
        help=f"Minimum characters per chunk (default: {settings.min_chunk_chars})",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        # Load existing index if specified
        if args.load:
            print(f"Loading index from {args.load}...")
            vector_store = load_index(args.load, device=args.device or settings.get_device())
            print(f"✓ Loaded index with {len(vector_store)} vectors")
            return

        # Build new index
        folder_path = args.folder or settings.legal_docs_folder
        if not Path(folder_path).exists():
            print(f"Error: Folder not found: {folder_path}")
            sys.exit(1)

        print(f"Building index from folder: {folder_path}")

        # Initialize embedder
        print(f"Loading embedding model: {args.model or settings.embedding_model_name}...")
        embedder = EmbeddingModel(
            model_name=args.model,
            device=args.device,
        )
        print(f"✓ Embedding model loaded (dim={embedder.get_embedding_dim()}, device={embedder.device})")

        # Build index
        vector_store = build_index_from_folder(
            folder_path=folder_path,
            embedder=embedder,
            batch_size=args.batch_size,
            max_chunk_chars=args.max_chunk_chars,
            min_chunk_chars=args.min_chunk_chars,
        )

        print(f"✓ Index built successfully: {len(vector_store)} vectors")

        # Save if specified
        if args.save:
            save_path = args.save
            print(f"Saving index to {save_path}...")
            save_index(vector_store, save_path)
            print(f"✓ Index saved to {save_path}")

        # Print stats
        stats = vector_store.get_stats()
        print("\nIndex Statistics:")
        print(f"  Total vectors: {stats['num_vectors']}")
        print(f"  Embedding dimension: {stats['dim']}")
        print(f"  Device: {stats['device']}")
        print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

