"""PDF loading and text extraction for legal documents."""

import os
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

from app.logging_config import get_logger

logger = get_logger(__name__)


def load_pdf(path: str) -> str:
    """
    Extract text from a PDF file.

    Args:
        path: Path to the PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If the PDF cannot be read or is corrupted.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF file not found: {path}")

    try:
        reader = PdfReader(path)
        text_parts: List[str] = []
        total_pages = len(reader.pages)

        logger.info(f"Processing PDF: {path}", extra={"total_pages": total_pages})

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(page_text)
                else:
                    logger.warning(
                        f"Empty or blank page {page_num} in {path}",
                        extra={"page": page_num, "file": os.path.basename(path)},
                    )
            except Exception as e:
                logger.warning(
                    f"Error extracting text from page {page_num} in {path}: {e}",
                    extra={"page": page_num, "file": os.path.basename(path), "error": str(e)},
                )
                # Continue processing other pages
                continue

        if not text_parts:
            logger.warning(f"No text extracted from PDF: {path}")
            return ""

        full_text = "\n\n".join(text_parts)
        logger.info(
            f"Successfully extracted text from {path}",
            extra={
                "pages_processed": len(text_parts),
                "total_pages": total_pages,
                "text_length": len(full_text),
            },
        )

        return full_text

    except PdfReadError as e:
        error_msg = f"Cannot read PDF file (corrupted or invalid): {path}"
        logger.error(error_msg, extra={"error": str(e), "file": os.path.basename(path)})
        raise ValueError(error_msg) from e
    except PdfStreamError as e:
        error_msg = f"PDF stream error (file may be incomplete): {path}"
        logger.error(error_msg, extra={"error": str(e), "file": os.path.basename(path)})
        raise ValueError(error_msg) from e
    except Exception as e:
        error_msg = f"Unexpected error reading PDF: {path}"
        logger.error(error_msg, extra={"error": str(e), "file": os.path.basename(path)})
        raise ValueError(error_msg) from e


def load_legal_corpus_from_folder(folder_path: str) -> List[Tuple[str, str]]:
    """
    Load all PDF files from a folder and extract their text.

    Args:
        folder_path: Path to the folder containing PDF files.

    Returns:
        List of tuples (doc_id, raw_text) for each PDF.
        doc_id is the filename without extension.

    Raises:
        FileNotFoundError: If the folder does not exist.
        ValueError: If the folder is empty or contains no PDFs.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    # Find all PDF files
    pdf_files = list(folder.glob("*.pdf")) + list(folder.glob("*.PDF"))

    if not pdf_files:
        logger.warning(f"No PDF files found in folder: {folder_path}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {folder_path}")

    corpus: List[Tuple[str, str]] = []
    failed_files: List[str] = []

    for pdf_file in pdf_files:
        try:
            # Extract doc_id from filename (without extension)
            doc_id = pdf_file.stem

            logger.debug(f"Loading PDF: {pdf_file.name}", extra={"doc_id": doc_id})

            text = load_pdf(str(pdf_file))

            if text.strip():  # Only add non-empty documents
                corpus.append((doc_id, text))
                logger.info(
                    f"Successfully loaded: {pdf_file.name}",
                    extra={"doc_id": doc_id, "text_length": len(text)},
                )
            else:
                logger.warning(f"Skipping empty PDF: {pdf_file.name}", extra={"doc_id": doc_id})
                failed_files.append(pdf_file.name)

        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Failed to load PDF: {pdf_file.name}", extra={"error": str(e)})
            failed_files.append(pdf_file.name)
        except Exception as e:
            logger.error(
                f"Unexpected error loading PDF: {pdf_file.name}",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            failed_files.append(pdf_file.name)

    if failed_files:
        logger.warning(
            f"Failed to load {len(failed_files)} PDF file(s)",
            extra={"failed_files": failed_files},
        )

    if not corpus:
        raise ValueError(
            f"No valid PDFs could be loaded from {folder_path}. "
            f"All {len(pdf_files)} file(s) failed to load or were empty."
        )

    logger.info(
        f"Successfully loaded {len(corpus)} document(s) from {folder_path}",
        extra={"total_files": len(pdf_files), "successful": len(corpus), "failed": len(failed_files)},
    )

    return corpus

