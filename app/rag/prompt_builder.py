"""Legal-safe RAG prompt builder."""

from typing import List

from app.logging_config import get_logger

logger = get_logger(__name__)


def build_legal_prompt(question: str, retrieved_chunks: List[dict]) -> str:
    """
    Build a legal-safe RAG prompt with strict instructions.

    The prompt includes:
    - Instructions to use ONLY the provided legal text
    - Requirement to cite sources verbatim
    - Prohibition against hallucination
    - Instruction to respond "Not found" if answer isn't in context

    Args:
        question: The user's legal question.
        retrieved_chunks: List of retrieved chunks, each containing:
            {
                "score": float,
                "chunk_text": str,
                "doc_id": str,
                "metadata": dict
            }

    Returns:
        Formatted prompt string ready for LLM generation.

    Raises:
        ValueError: If question is empty or no chunks provided.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not retrieved_chunks:
        raise ValueError("No retrieved chunks provided")

    logger.debug(
        f"Building legal prompt",
        extra={"question_length": len(question), "num_chunks": len(retrieved_chunks)},
    )

    # Build the instructions section
    instructions = _build_instructions()

    # Build the context section from retrieved chunks
    context = _build_context(retrieved_chunks)

    # Build the question section
    question_section = _build_question_section(question)

    # Combine all sections
    prompt = f"""{instructions}

{context}

{question_section}"""

    logger.debug(f"Legal prompt built: {len(prompt)} characters")

    return prompt


def _build_instructions() -> str:
    """
    Build the instructions section of the prompt.

    Returns:
        Instructions string with legal safety guidelines.
    """
    return """You are a legal assistant that answers questions based ONLY on the provided legal text. Follow these strict guidelines:

1. USE ONLY THE PROVIDED LEGAL TEXT: Base your answer exclusively on the legal text provided below. Do not use any external knowledge, general legal principles, or information not explicitly stated in the provided text.

2. VERBATIM CITATIONS: When referencing legal provisions, cite them exactly as they appear in the provided text. Include section numbers, clause numbers, or other identifiers when available.

3. NO HALLUCINATION: Do NOT invent, assume, or infer legal provisions, case law, penalties, or procedures that are not explicitly stated in the provided text. If you are uncertain, state that explicitly.

4. NOT FOUND RESPONSE: If the answer to the question is not clearly found in the provided legal text, you MUST respond with: "Not found in the provided legal text." Do not attempt to provide an answer based on general knowledge or assumptions.

5. ACCURACY: Ensure all facts, numbers, dates, and legal provisions are quoted accurately from the provided text.

6. STRUCTURE: Provide a clear, structured answer. If multiple relevant sections exist, organize them logically."""


def _build_context(retrieved_chunks: List[dict]) -> str:
    """
    Build the context section from retrieved chunks.

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries.

    Returns:
        Formatted context string.
    """
    context_parts = ["LEGAL TEXT CONTEXT:\n"]

    for i, chunk in enumerate(retrieved_chunks, start=1):
        chunk_text = chunk.get("chunk_text", "")
        doc_id = chunk.get("doc_id", "Unknown")
        score = chunk.get("score", 0.0)
        metadata = chunk.get("metadata", {})

        # Build chunk header with metadata
        chunk_header = f"[Context {i}]"
        if doc_id:
            chunk_header += f" Document: {doc_id}"
        if "section" in metadata:
            chunk_header += f" | Section: {metadata['section']}"
        if "page" in metadata:
            chunk_header += f" | Page: {metadata['page']}"
        chunk_header += f" | Relevance: {score:.2f}"

        context_parts.append(f"{chunk_header}\n{chunk_text}\n")

    return "\n".join(context_parts)


def _build_question_section(question: str) -> str:
    """
    Build the question section of the prompt.

    Args:
        question: The user's question.

    Returns:
        Formatted question section.
    """
    return f"""QUESTION:
{question}

ANSWER (based ONLY on the provided legal text above):"""


def build_legal_prompt_simple(question: str, retrieved_chunks: List[dict]) -> str:
    """
    Build a simpler, more concise legal prompt.

    This is an alternative prompt format that may work better with smaller models.

    Args:
        question: The user's legal question.
        retrieved_chunks: List of retrieved chunks.

    Returns:
        Formatted prompt string.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")

    if not retrieved_chunks:
        raise ValueError("No retrieved chunks provided")

    # Build context
    context_parts = []
    for chunk in retrieved_chunks:
        chunk_text = chunk.get("chunk_text", "")
        doc_id = chunk.get("doc_id", "Unknown")
        context_parts.append(f"[{doc_id}]\n{chunk_text}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""Answer the question using ONLY the legal text below. If the answer is not in the text, say "Not found in the provided legal text."

Legal Text:
{context}

Question: {question}

Answer:"""

    return prompt


def format_chunks_for_display(retrieved_chunks: List[dict]) -> str:
    """
    Format retrieved chunks for display purposes (e.g., in API responses).

    Args:
        retrieved_chunks: List of retrieved chunk dictionaries.

    Returns:
        Human-readable formatted string.
    """
    if not retrieved_chunks:
        return "No chunks retrieved."

    formatted_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        chunk_text = chunk.get("chunk_text", "")
        doc_id = chunk.get("doc_id", "Unknown")
        score = chunk.get("score", 0.0)
        metadata = chunk.get("metadata", {})

        part = f"Chunk {i} (Score: {score:.3f}, Document: {doc_id})"
        if "section" in metadata:
            part += f", Section: {metadata['section']}"
        part += f":\n{chunk_text[:200]}..." if len(chunk_text) > 200 else f":\n{chunk_text}"

        formatted_parts.append(part)

    return "\n\n".join(formatted_parts)

