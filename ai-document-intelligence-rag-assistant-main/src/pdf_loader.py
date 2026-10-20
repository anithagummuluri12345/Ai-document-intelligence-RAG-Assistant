"""
PDF Loader Component for AI Document Intelligence.

This module handles safe loading and page-by-page text extraction
from PDF documents using PyMuPDF (fitz).
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Union
import fitz  # PyMuPDF

# Configure module-level logger
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Safely extracts text page-by-page from a PDF file.

    Args:
        pdf_path (Union[str, Path]): Path to the target PDF file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the 1-based page number
                              and the extracted text for each page.
                              Example:
                              [
                                  {"page": 1, "text": "..."},
                                  {"page": 2, "text": "..."}
                              ]
                              Returns an empty list if the file is missing or invalid.
    """
    file_path = Path(pdf_path)

    # Gracefully handle missing file
    if not file_path.exists():
        logger.error("PDF file not found: %s", file_path)
        return []

    # Gracefully handle path pointing to a directory instead of a file
    if not file_path.is_file():
        logger.error("Specified path is not a file: %s", file_path)
        return []

    pages_data: List[Dict[str, Any]] = []

    try:
        # Safely open the PDF document using a context manager to ensure proper cleanup
        with fitz.open(file_path) as doc:
            # Check if the opened document is a valid PDF
            if not doc.is_pdf:
                logger.error("File is not a valid PDF: %s", file_path)
                return []

            # Iterate through pages, preserving 1-based page numbers
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text()  # Extract plain text from the page
                pages_data.append({
                    "page": page_number,
                    "text": text
                })

    except fitz.FileDataError as err:
        # Gracefully handle corrupted, damaged, or unreadable PDF files
        logger.error("Corrupted or invalid PDF data in '%s': %s", file_path, err)
        return []
    except Exception as err:
        # Gracefully handle any unexpected I/O or parsing error
        logger.error("Failed to read PDF file '%s': %s", file_path, err)
        return []

    return pages_data
