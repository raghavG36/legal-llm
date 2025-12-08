"""RAG pipeline orchestrator - wires all components together."""

from typing import Dict, List, Optional

from app.config import settings
from app.logging_config import get_logger
from app.models.llm import LLMClient
from app.rag.prompt_builder import build_legal_prompt
from app.rag.retriever import Retriever

logger = get_logger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline that orchestrates retrieval and generation.

    Combines retriever, prompt builder, and LLM to answer legal questions
    based on retrieved context.
    """

    def __init__(self, retriever: Retriever, llm: LLMClient):
        """
        Initialize the RAG pipeline.

        Args:
            retriever: Retriever instance for finding relevant chunks.
            llm: LLMClient instance for generating answers.
        """
        self.retriever = retriever
        self.llm = llm

        logger.info("RAGPipeline initialized", extra={"llm_model": llm.model_name})

    def answer(
        self,
        question: str,
        k: int = None,
        min_score: float = 0.0,
        use_simple_prompt: bool = False,
        max_new_tokens: int = None,
        temperature: float = None,
    ) -> Dict:
        """
        Answer a legal question using RAG.

        The pipeline:
        1. Retrieves top-k relevant chunks using the retriever
        2. Builds a legal-safe prompt with the retrieved context
        3. Generates an answer using the LLM
        4. Returns the answer along with the retrieved context

        Args:
            question: The legal question to answer.
            k: Number of chunks to retrieve. Defaults to config.default_top_k.
            min_score: Minimum similarity score for retrieved chunks.
            use_simple_prompt: If True, uses simpler prompt format (better for smaller models).
            max_new_tokens: Maximum tokens to generate. Defaults to config.max_new_tokens.
            temperature: Sampling temperature. Defaults to config.temperature.

        Returns:
            Dictionary containing:
            {
                "answer": str,              # Generated answer
                "context": List[Dict],      # Retrieved chunks with scores
                "num_chunks": int,         # Number of chunks used
            }
            Each context item contains:
            {
                "score": float,
                "chunk_text": str,
                "doc_id": str,
                "metadata": dict
            }

        Raises:
            ValueError: If question is empty or retrieval fails.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        k = k or settings.default_top_k
        max_new_tokens = max_new_tokens or settings.max_new_tokens
        temperature = temperature if temperature is not None else settings.temperature

        logger.info(
            f"Processing question through RAG pipeline",
            extra={
                "question_length": len(question),
                "k": k,
                "min_score": min_score,
            },
        )

        # Step 1: Retrieve relevant chunks
        try:
            retrieved_chunks = self.retriever.retrieve(
                query=question,
                k=k,
                min_score=min_score,
            )

            if not retrieved_chunks:
                logger.warning("No chunks retrieved for question")
                return {
                    "answer": "Not found in the provided legal text.",
                    "context": [],
                    "num_chunks": 0,
                }

        except Exception as e:
            logger.error(
                f"Error during retrieval",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise

        # Step 2: Build legal-safe prompt
        try:
            if use_simple_prompt:
                from app.rag.prompt_builder import build_legal_prompt_simple

                prompt = build_legal_prompt_simple(question, retrieved_chunks)
            else:
                prompt = build_legal_prompt(question, retrieved_chunks)

            logger.debug(
                f"Prompt built: {len(prompt)} characters",
                extra={"num_chunks": len(retrieved_chunks)},
            )

        except Exception as e:
            logger.error(
                f"Error building prompt",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise

        # Step 3: Generate answer using LLM
        try:
            answer = self.llm.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            logger.info(
                f"Answer generated: {len(answer)} characters",
                extra={"answer_length": len(answer)},
            )

        except Exception as e:
            logger.error(
                f"Error during LLM generation",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            raise

        # Step 4: Format and return response
        return {
            "answer": answer.strip(),
            "context": retrieved_chunks,
            "num_chunks": len(retrieved_chunks),
        }

    def answer_batch(
        self,
        questions: List[str],
        k: int = None,
        min_score: float = 0.0,
        use_simple_prompt: bool = False,
    ) -> List[Dict]:
        """
        Answer multiple questions in batch.

        Args:
            questions: List of questions to answer.
            k: Number of chunks to retrieve per question.
            min_score: Minimum similarity score.
            use_simple_prompt: Whether to use simple prompt format.

        Returns:
            List of answer dictionaries, one per question.
        """
        logger.info(f"Processing batch of {len(questions)} questions")

        results = []
        for i, question in enumerate(questions):
            try:
                result = self.answer(
                    question=question,
                    k=k,
                    min_score=min_score,
                    use_simple_prompt=use_simple_prompt,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Error processing question {i}",
                    extra={"error": str(e), "question_index": i},
                )
                # Add error result
                results.append(
                    {
                        "answer": f"Error processing question: {str(e)}",
                        "context": [],
                        "num_chunks": 0,
                    }
                )

        return results

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return f"RAGPipeline(retriever={self.retriever}, llm={self.llm.model_name})"

