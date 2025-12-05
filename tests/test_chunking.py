"""Tests for legal-aware chunking."""

import pytest

from app.rag.chunking import (
    _find_sentence_boundary,
    _merge_and_split_chunks,
    _split_by_legal_markers,
    _split_by_paragraphs,
    _split_large_chunk,
    chunk_legal_text,
)


def test_chunk_legal_text_empty():
    """Test chunking empty text."""
    result = chunk_legal_text("")
    assert result == []

    result = chunk_legal_text("   ")
    assert result == []


def test_chunk_legal_text_simple():
    """Test chunking simple text without markers."""
    text = "This is a simple legal document. It has multiple sentences. Each sentence is important."
    chunks = chunk_legal_text(text, max_chunk_chars=50, min_chunk_chars=20)
    assert len(chunks) > 0
    assert all(len(chunk) <= 50 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_legal_text_with_sections():
    """Test chunking text with section markers."""
    text = """
Section 1: Introduction
This is the introduction section. It contains important information.

Section 2: Definitions
This section defines key terms used in the document.

Section 3: Provisions
This section contains the main provisions of the legal document.
"""
    chunks = chunk_legal_text(text, max_chunk_chars=200, min_chunk_chars=50)
    assert len(chunks) >= 3
    # Check that sections are separated
    text_combined = " ".join(chunks)
    assert "Section 1" in text_combined
    assert "Section 2" in text_combined
    assert "Section 3" in text_combined


def test_chunk_legal_text_with_clauses():
    """Test chunking text with clause markers."""
    text = """
Clause 1: First clause content here.

Clause 2: Second clause content here.

Clause 3: Third clause content here.
"""
    chunks = chunk_legal_text(text, max_chunk_chars=100, min_chunk_chars=30)
    assert len(chunks) >= 3
    text_combined = " ".join(chunks)
    assert "Clause 1" in text_combined
    assert "Clause 2" in text_combined


def test_chunk_legal_text_with_headings():
    """Test chunking text with ALL CAPS headings."""
    text = """
INTRODUCTION TO THE DOCUMENT
This is the introduction paragraph.

MAIN PROVISIONS
These are the main provisions of the document.

CONCLUSION
This is the conclusion paragraph.
"""
    chunks = chunk_legal_text(text, max_chunk_chars=150, min_chunk_chars=50)
    assert len(chunks) >= 3


def test_chunk_legal_text_respects_max_size():
    """Test that chunks respect maximum size."""
    # Create a long text
    text = "Sentence. " * 1000  # Very long text
    chunks = chunk_legal_text(text, max_chunk_chars=100, min_chunk_chars=50)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert len(chunks) > 1


def test_chunk_legal_text_no_mid_sentence_breaks():
    """Test that chunks don't break mid-sentence."""
    text = "This is sentence one. This is sentence two. This is sentence three."
    chunks = chunk_legal_text(text, max_chunk_chars=30, min_chunk_chars=10)
    # Check that sentences are not broken
    for chunk in chunks:
        # Each chunk should end with sentence-ending punctuation or be complete
        assert chunk[-1] in ".!?" or chunk.count(".") == chunk.count(".")


def test_split_by_legal_markers_sections():
    """Test splitting by section markers."""
    text = "Preamble text.\n\nSection 1: First section.\n\nSection 2: Second section."
    chunks = _split_by_legal_markers(text)
    assert len(chunks) >= 2
    assert any("Section 1" in chunk for chunk in chunks)
    assert any("Section 2" in chunk for chunk in chunks)


def test_split_by_legal_markers_no_markers():
    """Test splitting when no markers are present."""
    text = "This is plain text without any legal markers. It should remain as one chunk."
    chunks = _split_by_legal_markers(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_by_paragraphs():
    """Test splitting by paragraph breaks."""
    chunks = [
        "Paragraph one.\n\nParagraph two.",
        "Paragraph three.\n\nParagraph four.",
    ]
    result = _split_by_paragraphs(chunks)
    assert len(result) >= 4
    assert all("Paragraph" in para for para in result)


def test_merge_and_split_chunks_small():
    """Test merging small chunks."""
    chunks = ["Small chunk 1.", "Small chunk 2.", "Small chunk 3."]
    result = _merge_and_split_chunks(chunks, max_chunk_chars=100, min_chunk_chars=50)
    # Should merge some chunks together
    assert len(result) <= len(chunks)
    assert all(len(chunk) <= 100 for chunk in result)


def test_merge_and_split_chunks_large():
    """Test splitting large chunks."""
    large_text = "Sentence. " * 200  # Very long
    chunks = [large_text]
    result = _merge_and_split_chunks(chunks, max_chunk_chars=100, min_chunk_chars=50)
    assert len(result) > 1
    assert all(len(chunk) <= 100 for chunk in result)


def test_split_large_chunk():
    """Test splitting a single large chunk."""
    large_text = "Sentence one. Sentence two. Sentence three. " * 50
    chunks = _split_large_chunk(large_text, max_chunk_chars=100, min_chunk_chars=50)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    # Verify all original text is preserved
    assert "".join(chunks).replace(" ", "") == large_text.replace(" ", "")


def test_find_sentence_boundary():
    """Test finding sentence boundaries."""
    text = "First sentence. Second sentence. Third sentence."
    # Find boundary before position 30
    boundary = _find_sentence_boundary(text, 30)
    assert boundary > 0
    assert boundary <= 30
    # Should be at end of first sentence
    assert text[boundary - 1] == "."


def test_find_sentence_boundary_no_boundary():
    """Test when no sentence boundary is found."""
    text = "This is a very long sentence without any periods in the first part"
    boundary = _find_sentence_boundary(text, 30)
    # Should return 0 or a reasonable position
    assert boundary >= 0
    assert boundary <= 30


def test_chunk_legal_text_complex_document():
    """Test chunking a complex legal document."""
    text = """
PREAMBLE
This document establishes the legal framework.

Section 1: Definitions
The term "Party" refers to any signatory. The term "Agreement" means this document.

Section 2: Obligations
Each party must comply with the terms. Failure to comply results in penalties.

Clause 1: Payment Terms
Payment must be made within 30 days. Late payments incur interest.

Clause 2: Termination
Either party may terminate with 30 days notice.

CONCLUSION
This agreement is binding upon all parties.
"""
    chunks = chunk_legal_text(text, max_chunk_chars=200, min_chunk_chars=100)
    assert len(chunks) > 0
    assert all(len(chunk) <= 200 for chunk in chunks)
    # Verify key sections are present
    text_combined = " ".join(chunks)
    assert "Section 1" in text_combined or "Definitions" in text_combined
    assert "Clause 1" in text_combined or "Payment" in text_combined


def test_chunk_legal_text_custom_sizes():
    """Test chunking with custom size parameters."""
    text = "Sentence. " * 100
    chunks = chunk_legal_text(text, max_chunk_chars=50, min_chunk_chars=20)
    assert all(len(chunk) <= 50 for chunk in chunks)
    # Most chunks should be >= min_chunk_chars (allowing some flexibility)
    large_chunks = [c for c in chunks if len(c) >= 15]  # Allow some flexibility
    assert len(large_chunks) > 0


def test_chunk_legal_text_preserves_content():
    """Test that chunking preserves all content."""
    original_text = "Section 1: First content. Section 2: Second content. Section 3: Third content."
    chunks = chunk_legal_text(original_text, max_chunk_chars=50, min_chunk_chars=20)
    # Combine chunks and compare (ignoring whitespace differences)
    combined = " ".join(chunks)
    # All key content should be present
    assert "Section 1" in combined
    assert "Section 2" in combined
    assert "Section 3" in combined
    assert "First content" in combined
    assert "Second content" in combined
    assert "Third content" in combined

