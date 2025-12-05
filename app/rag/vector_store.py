"""In-memory vector store using PyTorch tensors."""

from typing import Dict, List, Optional

import torch

from app.logging_config import get_logger

logger = get_logger(__name__)


class TensorVectorStore:
    """
    In-memory vector store using PyTorch tensors.

    Stores embeddings and associated metadata, with cosine similarity search.
    Designed to be easily swappable with FAISS/Chroma/Pinecone later.
    """

    def __init__(self, dim: int, device: str = "cpu"):
        """
        Initialize the vector store.

        Args:
            dim: Embedding dimension (e.g., 768 for all-mpnet-base-v2).
            device: Device to store tensors on ('cpu' or 'cuda').
        """
        self.dim = dim
        self.device = device

        # Storage: PyTorch tensor for embeddings
        self.embeddings: Optional[torch.Tensor] = None

        # Metadata: List of dictionaries, one per embedding
        self.metadatas: List[Dict] = []

        # Track number of vectors
        self.num_vectors = 0

        logger.info(
            f"Initialized TensorVectorStore",
            extra={"dim": dim, "device": device},
        )

    def add(self, embeddings: torch.Tensor, metadatas: List[Dict]) -> None:
        """
        Add embeddings and their metadata to the store.

        Args:
            embeddings: Tensor of shape (N, dim) where N is the number of embeddings.
            metadatas: List of N metadata dictionaries. Each dict should contain at least
                      'chunk_text' and 'doc_id'. Can include additional fields.

        Raises:
            ValueError: If embeddings and metadatas have mismatched lengths or
                       embeddings have wrong dimension.
        """
        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dim}, got {embeddings.shape[1]}"
            )

        num_new = embeddings.shape[0]
        if len(metadatas) != num_new:
            raise ValueError(
                f"Metadata length mismatch: got {len(metadatas)} metadata entries "
                f"for {num_new} embeddings"
            )

        # Ensure embeddings are on the correct device
        embeddings = embeddings.to(self.device)

        # Validate metadata structure
        for i, metadata in enumerate(metadatas):
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata at index {i} is not a dictionary")
            if "chunk_text" not in metadata:
                raise ValueError(f"Metadata at index {i} missing required field 'chunk_text'")
            if "doc_id" not in metadata:
                raise ValueError(f"Metadata at index {i} missing required field 'doc_id'")

        # Append to storage
        if self.embeddings is None:
            # First addition
            self.embeddings = embeddings
        else:
            # Concatenate with existing embeddings
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)

        # Append metadata
        self.metadatas.extend(metadatas)
        self.num_vectors += num_new

        logger.info(
            f"Added {num_new} embedding(s) to vector store",
            extra={
                "num_new": num_new,
                "total_vectors": self.num_vectors,
                "device": self.device,
            },
        )

    def search(
        self, query_embedding: torch.Tensor, k: int = 5, min_score: float = 0.0
    ) -> List[Dict]:
        """
        Search for top-k most similar embeddings using cosine similarity.

        Args:
            query_embedding: Query embedding tensor of shape (1, dim) or (dim,).
            k: Number of top results to return.
            min_score: Minimum similarity score threshold (0.0 to 1.0).

        Returns:
            List of dictionaries, each containing:
            {
                "score": float,  # Cosine similarity score
                "metadata": {...}  # Original metadata dict
            }
            Results are sorted by score (descending).

        Raises:
            ValueError: If store is empty or query embedding has wrong dimension.
        """
        if self.num_vectors == 0:
            logger.warning("Search called on empty vector store")
            return []

        if self.embeddings is None:
            return []

        # Ensure query is on correct device and has correct shape
        query_embedding = query_embedding.to(self.device)
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)  # (dim,) -> (1, dim)

        if query_embedding.shape[1] != self.dim:
            raise ValueError(
                f"Query embedding dimension mismatch: expected {self.dim}, "
                f"got {query_embedding.shape[1]}"
            )

        # Compute cosine similarity
        # For normalized embeddings: cosine_sim = query @ embeddings.T
        # Shape: (1, num_vectors)
        similarities = torch.mm(query_embedding, self.embeddings.T)

        # Clamp to [0, 1] range (in case of floating point errors)
        similarities = torch.clamp(similarities, -1.0, 1.0)

        # Get top-k indices
        k = min(k, self.num_vectors)  # Don't request more than available
        top_k_values, top_k_indices = torch.topk(similarities.squeeze(0), k)

        # Filter by min_score
        mask = top_k_values >= min_score
        top_k_values = top_k_values[mask]
        top_k_indices = top_k_indices[mask]

        # Convert to list of results
        results = []
        for score, idx in zip(top_k_values, top_k_indices):
            idx_int = idx.item()
            results.append(
                {
                    "score": score.item(),
                    "metadata": self.metadatas[idx_int].copy(),  # Return a copy
                }
            )

        logger.debug(
            f"Search completed: {len(results)} result(s) returned",
            extra={"k": k, "min_score": min_score, "num_results": len(results)},
        )

        return results

    def clear(self) -> None:
        """Clear all embeddings and metadata from the store."""
        self.embeddings = None
        self.metadatas = []
        self.num_vectors = 0
        logger.info("Vector store cleared")

    def __len__(self) -> int:
        """Return the number of vectors in the store."""
        return self.num_vectors

    def __repr__(self) -> str:
        """String representation of the vector store."""
        return (
            f"TensorVectorStore(dim={self.dim}, num_vectors={self.num_vectors}, "
            f"device='{self.device}')"
        )

    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics.
        """
        return {
            "num_vectors": self.num_vectors,
            "dim": self.dim,
            "device": self.device,
            "memory_usage_mb": (
                self.embeddings.element_size() * self.embeddings.nelement() / (1024 * 1024)
                if self.embeddings is not None
                else 0
            ),
        }

