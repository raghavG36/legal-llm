"""Retriever for querying the vector store."""

from typing import Dict, List, Optional

import torch

from app.config import settings
from app.logging_config import get_logger
from app.models.embeddings import EmbeddingModel
from app.rag.vector_store import TensorVectorStore

logger = get_logger(__name__)


class Retriever:
    """
    Retriever for querying the vector store with natural language queries.

    Encodes queries and retrieves relevant chunks from the vector store.
    """

    def __init__(self, embedder: EmbeddingModel, vector_store: TensorVectorStore):
        """
        Initialize the retriever.

        Args:
            embedder: EmbeddingModel instance for encoding queries.
            vector_store: TensorVectorStore instance to search.
        """
        self.embedder = embedder
        self.vector_store = vector_store

        # Verify embedding dimensions match
        embedder_dim = embedder.get_embedding_dim()
        if embedder_dim != vector_store.dim:
            raise ValueError(
                f"Embedding dimension mismatch: embedder dim={embedder_dim}, "
                f"vector_store dim={vector_store.dim}"
            )

        logger.info(
            "Retriever initialized",
            extra={
                "embedding_dim": embedder_dim,
                "num_vectors": len(vector_store),
            },
        )

    def retrieve(
        self,
        query: str,
        k: int = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: Natural language query string.
            k: Number of results to return. Defaults to config.default_top_k.
            min_score: Minimum similarity score threshold (0.0 to 1.0).

        Returns:
            List of dictionaries, each containing:
            {
                "score": float,           # Cosine similarity score
                "chunk_text": str,        # The text chunk
                "doc_id": str,            # Document ID
                "metadata": dict,         # Full metadata dictionary
            }
            Results are sorted by score (descending).

        Raises:
            ValueError: If query is empty or vector store is empty.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        k = k or settings.default_top_k

        logger.debug(
            f"Retrieving chunks for query",
            extra={"query_length": len(query), "k": k, "min_score": min_score},
        )

        # Encode the query
        try:
            query_embedding = self.embedder.encode(query, normalize_embeddings=True)
            # Ensure shape is (1, dim)
            if query_embedding.dim() == 1:
                query_embedding = query_embedding.unsqueeze(0)
        except Exception as e:
            logger.error(
                f"Error encoding query",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise

        # Search the vector store
        try:
            results = self.vector_store.search(
                query_embedding=query_embedding,
                k=k,
                min_score=min_score,
            )
        except Exception as e:
            logger.error(
                f"Error searching vector store",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise

        # Format results for easier access
        formatted_results = []
        for result in results:
            metadata = result["metadata"]
            formatted_result = {
                "score": result["score"],
                "chunk_text": metadata.get("chunk_text", ""),
                "doc_id": metadata.get("doc_id", ""),
                "metadata": metadata,  # Include full metadata for flexibility
            }
            formatted_results.append(formatted_result)

        logger.info(
            f"Retrieved {len(formatted_results)} chunk(s) for query",
            extra={
                "num_results": len(formatted_results),
                "k": k,
                "min_score": min_score,
            },
        )

        return formatted_results

    def retrieve_batch(
        self,
        queries: List[str],
        k: int = None,
        min_score: float = 0.0,
    ) -> List[List[Dict]]:
        """
        Retrieve results for multiple queries in batch.

        Args:
            queries: List of query strings.
            k: Number of results per query. Defaults to config.default_top_k.
            min_score: Minimum similarity score threshold.

        Returns:
            List of result lists, one per query. Each result list has the same
            format as retrieve().
        """
        k = k or settings.default_top_k

        logger.debug(
            f"Batch retrieving chunks for {len(queries)} queries",
            extra={"num_queries": len(queries), "k": k},
        )

        # Encode all queries in batch
        try:
            query_embeddings = self.embedder.encode(
                queries,
                normalize_embeddings=True,
            )
        except Exception as e:
            logger.error(
                f"Error encoding batch queries",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise

        # Search for each query
        all_results = []
        for i, query_embedding in enumerate(query_embeddings):
            # Ensure shape is (1, dim)
            query_embedding = query_embedding.unsqueeze(0) if query_embedding.dim() == 1 else query_embedding

            try:
                results = self.vector_store.search(
                    query_embedding=query_embedding,
                    k=k,
                    min_score=min_score,
                )

                # Format results
                formatted_results = []
                for result in results:
                    metadata = result["metadata"]
                    formatted_result = {
                        "score": result["score"],
                        "chunk_text": metadata.get("chunk_text", ""),
                        "doc_id": metadata.get("doc_id", ""),
                        "metadata": metadata,
                    }
                    formatted_results.append(formatted_result)

                all_results.append(formatted_results)

            except Exception as e:
                logger.error(
                    f"Error searching for query {i}",
                    extra={"error": str(e), "query_index": i},
                )
                # Add empty results for this query
                all_results.append([])

        logger.info(
            f"Batch retrieval complete: {len(all_results)} query results",
            extra={"num_queries": len(queries), "num_results": sum(len(r) for r in all_results)},
        )

        return all_results

    def __repr__(self) -> str:
        """String representation of the retriever."""
        return (
            f"Retriever(embedder={self.embedder.model_name}, "
            f"vector_store_size={len(self.vector_store)})"
        )

