"""Tests for legal prompt builder."""

import pytest

from app.rag.prompt_builder import (
    build_legal_prompt,
    build_legal_prompt_simple,
    format_chunks_for_display,
)


def test_build_legal_prompt_basic():
    """Test building a basic legal prompt."""
    question = "What is the penalty for theft?"
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "Section 302: Whoever commits theft shall be punished with imprisonment.",
            "doc_id": "BNS",
            "metadata": {"section": "302", "page": 45},
        }
    ]

    prompt = build_legal_prompt(question, chunks)

    assert question in prompt
    assert "Section 302" in prompt
    assert "LEGAL TEXT CONTEXT" in prompt
    assert "QUESTION" in prompt
    assert "ANSWER" in prompt


def test_build_legal_prompt_multiple_chunks():
    """Test building prompt with multiple chunks."""
    question = "What are the obligations?"
    chunks = [
        {
            "score": 0.85,
            "chunk_text": "Section 1: First obligation.",
            "doc_id": "doc1",
            "metadata": {"section": "1"},
        },
        {
            "score": 0.80,
            "chunk_text": "Section 2: Second obligation.",
            "doc_id": "doc1",
            "metadata": {"section": "2"},
        },
    ]

    prompt = build_legal_prompt(question, chunks)

    assert "Section 1" in prompt
    assert "Section 2" in prompt
    assert "[Context 1]" in prompt
    assert "[Context 2]" in prompt


def test_build_legal_prompt_instructions():
    """Test that prompt includes safety instructions."""
    question = "Test question"
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "Test chunk",
            "doc_id": "doc1",
            "metadata": {},
        }
    ]

    prompt = build_legal_prompt(question, chunks)

    # Check for key safety instructions
    assert "USE ONLY THE PROVIDED LEGAL TEXT" in prompt
    assert "NO HALLUCINATION" in prompt
    assert "Not found in the provided legal text" in prompt
    assert "VERBATIM CITATIONS" in prompt


def test_build_legal_prompt_empty_question():
    """Test building prompt with empty question."""
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "Test chunk",
            "doc_id": "doc1",
            "metadata": {},
        }
    ]

    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_legal_prompt("", chunks)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_legal_prompt("   ", chunks)


def test_build_legal_prompt_no_chunks():
    """Test building prompt with no chunks."""
    question = "Test question"

    with pytest.raises(ValueError, match="No retrieved chunks provided"):
        build_legal_prompt(question, [])


def test_build_legal_prompt_metadata_included():
    """Test that metadata is included in the prompt."""
    question = "Test question"
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "Test chunk text",
            "doc_id": "BNS",
            "metadata": {"section": "302", "page": 45, "clause": "1"},
        }
    ]

    prompt = build_legal_prompt(question, chunks)

    assert "Document: BNS" in prompt
    assert "Section: 302" in prompt
    assert "Page: 45" in prompt
    assert "Test chunk text" in prompt


def test_build_legal_prompt_score_included():
    """Test that relevance scores are included."""
    question = "Test question"
    chunks = [
        {
            "score": 0.87,
            "chunk_text": "Test chunk",
            "doc_id": "doc1",
            "metadata": {},
        }
    ]

    prompt = build_legal_prompt(question, chunks)

    assert "0.87" in prompt or "Relevance" in prompt


def test_build_legal_prompt_simple():
    """Test building simple prompt format."""
    question = "What is the penalty?"
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "Section 302: Penalty is imprisonment.",
            "doc_id": "BNS",
            "metadata": {},
        }
    ]

    prompt = build_legal_prompt_simple(question, chunks)

    assert question in prompt
    assert "Section 302" in prompt
    assert "Not found in the provided legal text" in prompt
    assert "Legal Text" in prompt


def test_build_legal_prompt_simple_empty_question():
    """Test simple prompt with empty question."""
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "Test chunk",
            "doc_id": "doc1",
            "metadata": {},
        }
    ]

    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_legal_prompt_simple("", chunks)


def test_build_legal_prompt_simple_no_chunks():
    """Test simple prompt with no chunks."""
    question = "Test question"

    with pytest.raises(ValueError, match="No retrieved chunks provided"):
        build_legal_prompt_simple(question, [])


def test_build_legal_prompt_simple_multiple_chunks():
    """Test simple prompt with multiple chunks."""
    question = "Test question"
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "Chunk 1",
            "doc_id": "doc1",
            "metadata": {},
        },
        {
            "score": 0.8,
            "chunk_text": "Chunk 2",
            "doc_id": "doc2",
            "metadata": {},
        },
    ]

    prompt = build_legal_prompt_simple(question, chunks)

    assert "Chunk 1" in prompt
    assert "Chunk 2" in prompt
    assert "[doc1]" in prompt
    assert "[doc2]" in prompt
    assert "---" in prompt  # Separator


def test_format_chunks_for_display():
    """Test formatting chunks for display."""
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "This is a test chunk with some content.",
            "doc_id": "doc1",
            "metadata": {"section": "1"},
        },
        {
            "score": 0.8,
            "chunk_text": "Another chunk.",
            "doc_id": "doc2",
            "metadata": {},
        },
    ]

    formatted = format_chunks_for_display(chunks)

    assert "Chunk 1" in formatted
    assert "Chunk 2" in formatted
    assert "doc1" in formatted
    assert "doc2" in formatted
    assert "0.900" in formatted or "0.9" in formatted
    assert "This is a test chunk" in formatted


def test_format_chunks_for_display_empty():
    """Test formatting empty chunks list."""
    formatted = format_chunks_for_display([])
    assert formatted == "No chunks retrieved."


def test_format_chunks_for_display_long_text():
    """Test formatting chunks with long text (truncation)."""
    long_text = "A" * 300
    chunks = [
        {
            "score": 0.9,
            "chunk_text": long_text,
            "doc_id": "doc1",
            "metadata": {},
        }
    ]

    formatted = format_chunks_for_display(chunks)

    # Should be truncated to 200 chars
    assert len(formatted.split(":\n")[1]) < len(long_text)
    assert "..." in formatted


def test_build_legal_prompt_missing_fields():
    """Test prompt building with missing optional fields."""
    question = "Test question"
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "Test chunk",
            # Missing doc_id and metadata
        }
    ]

    # Should not raise error, but handle gracefully
    prompt = build_legal_prompt(question, chunks)
    assert "Test chunk" in prompt


def test_build_legal_prompt_structure():
    """Test that prompt has correct structure."""
    question = "What is the law?"
    chunks = [
        {
            "score": 0.9,
            "chunk_text": "The law states...",
            "doc_id": "doc1",
            "metadata": {},
        }
    ]

    prompt = build_legal_prompt(question, chunks)

    # Check structure: instructions, context, question
    parts = prompt.split("\n\n")
    assert len(parts) >= 3  # At least instructions, context, question

    # Question should be near the end
    assert "QUESTION:" in prompt
    assert "ANSWER" in prompt

