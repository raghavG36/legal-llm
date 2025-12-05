"""Legal-aware text chunking for legal documents."""

import re
from typing import List

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


def chunk_legal_text(text: str, max_chunk_chars: int = None, min_chunk_chars: int = None) -> List[str]:
    """
    Chunk legal text using legal-aware heuristics.

    Splits text by:
    - Section markers (Section 1, Sec. 1, etc.)
    - Clause markers (Clause 1, etc.)
    - Headings (ALL CAPS lines)
    - Double newlines (paragraph breaks)

    Then groups chunks to approximate target size without breaking mid-sentence.

    Args:
        text: Raw text to chunk.
        max_chunk_chars: Maximum characters per chunk. Defaults to config value.
        min_chunk_chars: Minimum target characters per chunk. Defaults to config value.

    Returns:
        List of text chunks. No empty strings are returned.
    """
    if not text or not text.strip():
        logger.warning("Empty text provided for chunking")
        return []

    max_chunk_chars = max_chunk_chars or settings.max_chunk_chars
    min_chunk_chars = min_chunk_chars or settings.min_chunk_chars

    if max_chunk_chars < min_chunk_chars:
        logger.warning(
            f"max_chunk_chars ({max_chunk_chars}) < min_chunk_chars ({min_chunk_chars}), "
            f"using min_chunk_chars for both"
        )
        max_chunk_chars = min_chunk_chars

    logger.debug(
        f"Chunking text with max_chunk_chars={max_chunk_chars}, min_chunk_chars={min_chunk_chars}",
        extra={"text_length": len(text)},
    )

    # Step 1: Split by legal document markers
    initial_chunks = _split_by_legal_markers(text)

    # Step 2: Further split by paragraphs (double newlines)
    paragraph_chunks = _split_by_paragraphs(initial_chunks)

    # Step 3: Merge small chunks and split large ones to target size
    final_chunks = _merge_and_split_chunks(paragraph_chunks, max_chunk_chars, min_chunk_chars)

    # Step 4: Clean and validate chunks
    final_chunks = [chunk.strip() for chunk in final_chunks if chunk.strip()]

    logger.info(
        f"Chunking complete: {len(final_chunks)} chunks created",
        extra={"original_length": len(text), "num_chunks": len(final_chunks)},
    )

    return final_chunks


def _split_by_legal_markers(text: str) -> List[str]:
    """
    Split text by legal document markers (sections, clauses, headings).

    Returns:
        List of text segments split at legal markers.
    """
    # Pattern for section markers: "Section 1", "Sec. 1", "Section I", etc.
    section_pattern = re.compile(
        r"\n\s*(?:Section|Sec\.?)\s+(?:\d+|[IVX]+|[ivx]+)[\.:]?\s*\n",
        re.IGNORECASE | re.MULTILINE,
    )

    # Pattern for clause markers: "Clause 1", "Cl. 1", etc.
    clause_pattern = re.compile(
        r"\n\s*(?:Clause|Cl\.?)\s+(?:\d+|[IVX]+|[ivx]+)[\.:]?\s*\n",
        re.IGNORECASE | re.MULTILINE,
    )

    # Pattern for ALL CAPS headings (at least 3 words, all caps, on their own line)
    heading_pattern = re.compile(r"\n\s*[A-Z][A-Z\s]{10,}\n", re.MULTILINE)

    # Combine all patterns
    all_patterns = [section_pattern, clause_pattern, heading_pattern]

    # Find all split points
    split_points = [0]  # Start of text

    for pattern in all_patterns:
        for match in pattern.finditer(text):
            # Use the start of the match as a split point
            split_points.append(match.start())

    # Remove duplicates and sort
    split_points = sorted(set(split_points))

    # Add end of text
    if split_points[-1] != len(text):
        split_points.append(len(text))

    # Extract chunks
    chunks = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

    # If no markers found, return the whole text as one chunk
    if not chunks:
        chunks = [text]

    logger.debug(f"Split by legal markers: {len(chunks)} initial chunks")
    return chunks


def _split_by_paragraphs(chunks: List[str]) -> List[str]:
    """
    Further split chunks by paragraph breaks (double newlines).

    Args:
        chunks: List of text chunks.

    Returns:
        List of chunks split by paragraphs.
    """
    paragraph_chunks = []

    for chunk in chunks:
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r"\n\s*\n+", chunk)
        # Also split by single newline if it's followed by indentation (new paragraph)
        further_split = []
        for para in paragraphs:
            # Check if paragraph is very long and might need splitting
            if len(para) > 2000:  # Very long paragraph, split by single newlines
                lines = para.split("\n")
                current = []
                for line in lines:
                    if current and len(" ".join(current)) > 1000:
                        further_split.append("\n".join(current))
                        current = [line]
                    else:
                        current.append(line)
                if current:
                    further_split.append("\n".join(current))
            else:
                further_split.append(para)

        paragraph_chunks.extend([p.strip() for p in further_split if p.strip()])

    logger.debug(f"Split by paragraphs: {len(paragraph_chunks)} chunks")
    return paragraph_chunks


def _merge_and_split_chunks(chunks: List[str], max_chunk_chars: int, min_chunk_chars: int) -> List[str]:
    """
    Merge small chunks and split large ones to approximate target size.

    Ensures no mid-sentence breaks.

    Args:
        chunks: List of text chunks.
        max_chunk_chars: Maximum characters per chunk.
        min_chunk_chars: Minimum target characters per chunk.

    Returns:
        List of chunks sized appropriately.
    """
    if not chunks:
        return []

    final_chunks = []
    current_chunk = ""

    for chunk in chunks:
        # If current chunk is empty, start with this chunk
        if not current_chunk:
            current_chunk = chunk
            continue

        # Check if adding this chunk would exceed max
        combined = current_chunk + "\n\n" + chunk
        combined_length = len(combined)

        if combined_length <= max_chunk_chars:
            # Safe to merge
            current_chunk = combined
        else:
            # Would exceed max, so finalize current chunk
            if current_chunk:
                # If current chunk is too small, try to add part of the next chunk
                if len(current_chunk) < min_chunk_chars and combined_length > min_chunk_chars:
                    # Try to split the next chunk at a sentence boundary
                    split_point = _find_sentence_boundary(chunk, max_chunk_chars - len(current_chunk) - 2)
                    if split_point > 0:
                        # Add part of the chunk
                        current_chunk = current_chunk + "\n\n" + chunk[:split_point].strip()
                        final_chunks.append(current_chunk)
                        current_chunk = chunk[split_point:].strip()
                    else:
                        # Can't split, just add current chunk
                        final_chunks.append(current_chunk)
                        current_chunk = chunk
                else:
                    # Current chunk is fine, add it
                    final_chunks.append(current_chunk)
                    current_chunk = chunk
            else:
                current_chunk = chunk

        # If current chunk is very large, split it
        if len(current_chunk) > max_chunk_chars:
            split_chunks = _split_large_chunk(current_chunk, max_chunk_chars, min_chunk_chars)
            # Add all but the last one
            final_chunks.extend(split_chunks[:-1])
            # Keep the last one as current
            current_chunk = split_chunks[-1] if split_chunks else ""

    # Add the last chunk
    if current_chunk:
        final_chunks.append(current_chunk)

    return final_chunks


def _split_large_chunk(chunk: str, max_chunk_chars: int, min_chunk_chars: int) -> List[str]:
    """
    Split a large chunk into smaller chunks at sentence boundaries.

    Args:
        chunk: Text chunk to split.
        max_chunk_chars: Maximum characters per resulting chunk.
        min_chunk_chars: Minimum target characters per chunk.

    Returns:
        List of smaller chunks.
    """
    if len(chunk) <= max_chunk_chars:
        return [chunk]

    chunks = []
    remaining = chunk
    current_pos = 0

    while len(remaining) > max_chunk_chars:
        # Find a good split point (sentence boundary) near max_chunk_chars
        split_point = _find_sentence_boundary(remaining, max_chunk_chars)

        if split_point > 0:
            chunks.append(remaining[:split_point].strip())
            remaining = remaining[split_point:].strip()
        else:
            # No sentence boundary found, split at last space before max
            split_point = remaining.rfind(" ", 0, max_chunk_chars)
            if split_point > min_chunk_chars:
                chunks.append(remaining[:split_point].strip())
                remaining = remaining[split_point:].strip()
            else:
                # Force split at max_chunk_chars
                chunks.append(remaining[:max_chunk_chars].strip())
                remaining = remaining[max_chunk_chars:].strip()

    if remaining:
        chunks.append(remaining)

    return chunks


def _find_sentence_boundary(text: str, max_length: int) -> int:
    """
    Find the best sentence boundary before max_length.

    Looks for sentence endings (. ! ?) followed by space and capital letter.

    Args:
        text: Text to search.
        max_length: Maximum position to search up to.

    Returns:
        Position of sentence boundary, or 0 if not found.
    """
    if len(text) <= max_length:
        return len(text)

    # Look for sentence endings in the last 20% of the allowed range
    search_start = max(0, int(max_length * 0.8))
    search_end = min(len(text), max_length)

    # Pattern: sentence ending (. ! ?) followed by space and capital letter or newline
    pattern = re.compile(r"[.!?]\s+[A-Z]|[.!?]\s*\n")

    best_match = 0
    for match in pattern.finditer(text[search_start:search_end]):
        # Position relative to start of text
        pos = search_start + match.end() - 1  # End of sentence
        if pos > best_match and pos <= max_length:
            best_match = pos

    # If no sentence boundary found, try to find a paragraph break
    if best_match == 0:
        para_break = text.rfind("\n\n", 0, max_length)
        if para_break > 0:
            best_match = para_break + 2  # Include the newlines

    return best_match

