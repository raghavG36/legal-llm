"""Tests for PDF loader module."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pypdf.errors import PdfReadError

from app.ingestion.pdf_loader import load_legal_corpus_from_folder, load_pdf


def test_load_pdf_nonexistent_file():
    """Test that loading a non-existent PDF raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_pdf("nonexistent_file.pdf")


def test_load_pdf_empty_file(tmp_path):
    """Test handling of empty or corrupted PDF files."""
    # Create an empty file
    empty_pdf = tmp_path / "empty.pdf"
    empty_pdf.write_bytes(b"")

    with pytest.raises(ValueError, match="Cannot read PDF|PDF stream error|Unexpected error"):
        load_pdf(str(empty_pdf))


def test_load_pdf_with_mock_reader(tmp_path):
    """Test PDF loading with mocked PdfReader."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf content")

    # Mock PdfReader to return pages with text
    mock_reader = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Page 1 content\n\nSection 1: Test"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "Page 2 content\n\nSection 2: More test"

    mock_reader.pages = [mock_page1, mock_page2]
    mock_reader.__len__ = lambda x: 2

    with patch("app.ingestion.pdf_loader.PdfReader", return_value=mock_reader):
        text = load_pdf(str(pdf_path))

    assert "Page 1 content" in text
    assert "Page 2 content" in text
    assert "Section 1: Test" in text


def test_load_pdf_with_empty_pages(tmp_path):
    """Test handling of PDFs with empty pages."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf content")

    mock_reader = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Valid content"
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = ""  # Empty page
    mock_page3 = MagicMock()
    mock_page3.extract_text.return_value = "   "  # Blank page

    mock_reader.pages = [mock_page1, mock_page2, mock_page3]
    mock_reader.__len__ = lambda x: 3

    with patch("app.ingestion.pdf_loader.PdfReader", return_value=mock_reader):
        text = load_pdf(str(pdf_path))

    assert "Valid content" in text
    # Empty pages should be skipped but not cause errors


def test_load_pdf_with_page_extraction_error(tmp_path):
    """Test handling of errors during page extraction."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf content")

    mock_reader = MagicMock()
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Valid content"
    mock_page2 = MagicMock()
    mock_page2.extract_text.side_effect = Exception("Extraction error")

    mock_reader.pages = [mock_page1, mock_page2]
    mock_reader.__len__ = lambda x: 2

    with patch("app.ingestion.pdf_loader.PdfReader", return_value=mock_reader):
        text = load_pdf(str(pdf_path))

    # Should still extract valid pages
    assert "Valid content" in text


def test_load_pdf_corrupted_file(tmp_path):
    """Test handling of corrupted PDF files."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.write_bytes(b"not a valid pdf")

    with patch("app.ingestion.pdf_loader.PdfReader") as mock_reader_class:
        mock_reader_class.side_effect = PdfReadError("Corrupted PDF")

        with pytest.raises(ValueError, match="Cannot read PDF file"):
            load_pdf(str(pdf_path))


def test_load_legal_corpus_from_folder_nonexistent():
    """Test loading from non-existent folder."""
    with pytest.raises(FileNotFoundError):
        load_legal_corpus_from_folder("nonexistent_folder")


def test_load_legal_corpus_from_folder_empty(tmp_path):
    """Test loading from empty folder."""
    result = load_legal_corpus_from_folder(str(tmp_path))
    assert result == []


def test_load_legal_corpus_from_folder_not_directory(tmp_path):
    """Test that providing a file path instead of directory raises error."""
    file_path = tmp_path / "not_a_folder.txt"
    file_path.write_text("test")

    with pytest.raises(ValueError, match="not a directory"):
        load_legal_corpus_from_folder(str(file_path))


def test_load_legal_corpus_from_folder_success(tmp_path):
    """Test successful loading of multiple PDFs from folder."""
    # Create mock PDF files
    pdf1 = tmp_path / "document1.pdf"
    pdf2 = tmp_path / "document2.pdf"
    pdf1.write_bytes(b"fake pdf 1")
    pdf2.write_bytes(b"fake pdf 2")

    # Mock the load_pdf function
    def mock_load_pdf(path: str) -> str:
        if "document1" in path:
            return "Content from document 1\n\nSection A: Legal text"
        elif "document2" in path:
            return "Content from document 2\n\nSection B: More legal text"
        return ""

    with patch("app.ingestion.pdf_loader.load_pdf", side_effect=mock_load_pdf):
        corpus = load_legal_corpus_from_folder(str(tmp_path))

    assert len(corpus) == 2
    doc_ids = [doc_id for doc_id, _ in corpus]
    assert "document1" in doc_ids
    assert "document2" in doc_ids

    # Check content
    texts = {doc_id: text for doc_id, text in corpus}
    assert "Content from document 1" in texts["document1"]
    assert "Content from document 2" in texts["document2"]


def test_load_legal_corpus_from_folder_with_failures(tmp_path):
    """Test loading when some PDFs fail."""
    pdf1 = tmp_path / "good.pdf"
    pdf2 = tmp_path / "bad.pdf"
    pdf1.write_bytes(b"fake pdf 1")
    pdf2.write_bytes(b"fake pdf 2")

    def mock_load_pdf(path: str) -> str:
        if "good" in path:
            return "Valid content"
        elif "bad" in path:
            raise ValueError("Corrupted PDF")
        return ""

    with patch("app.ingestion.pdf_loader.load_pdf", side_effect=mock_load_pdf):
        corpus = load_legal_corpus_from_folder(str(tmp_path))

    # Should still load the good PDF
    assert len(corpus) == 1
    assert corpus[0][0] == "good"


def test_load_legal_corpus_from_folder_all_fail(tmp_path):
    """Test when all PDFs fail to load."""
    pdf1 = tmp_path / "bad1.pdf"
    pdf2 = tmp_path / "bad2.pdf"
    pdf1.write_bytes(b"fake pdf 1")
    pdf2.write_bytes(b"fake pdf 2")

    def mock_load_pdf(path: str) -> str:
        raise ValueError("Corrupted PDF")

    with patch("app.ingestion.pdf_loader.load_pdf", side_effect=mock_load_pdf):
        with pytest.raises(ValueError, match="No valid PDFs could be loaded"):
            load_legal_corpus_from_folder(str(tmp_path))


def test_load_legal_corpus_from_folder_empty_pdfs(tmp_path):
    """Test handling of PDFs that extract to empty text."""
    pdf1 = tmp_path / "empty.pdf"
    pdf1.write_bytes(b"fake pdf")

    def mock_load_pdf(path: str) -> str:
        return ""  # Empty text

    with patch("app.ingestion.pdf_loader.load_pdf", side_effect=mock_load_pdf):
        corpus = load_legal_corpus_from_folder(str(tmp_path))

    # Empty PDFs should be skipped
    assert len(corpus) == 0


def test_load_legal_corpus_case_insensitive(tmp_path):
    """Test that both .pdf and .PDF extensions are found."""
    pdf1 = tmp_path / "doc1.pdf"
    pdf2 = tmp_path / "doc2.PDF"
    pdf1.write_bytes(b"fake pdf 1")
    pdf2.write_bytes(b"fake pdf 2")

    def mock_load_pdf(path: str) -> str:
        return f"Content from {Path(path).stem}"

    with patch("app.ingestion.pdf_loader.load_pdf", side_effect=mock_load_pdf):
        corpus = load_legal_corpus_from_folder(str(tmp_path))

    assert len(corpus) == 2
    doc_ids = [doc_id for doc_id, _ in corpus]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids

