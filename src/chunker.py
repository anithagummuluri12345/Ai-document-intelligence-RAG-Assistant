"""
Text Chunking Component for AI Document Intelligence.

This module splits extracted document pages into smaller, overlapping chunks
using LangChain's RecursiveCharacterTextSplitter while preserving document metadata.
"""

import logging
from typing import Any, Dict, List

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure module-level logger
logger = logging.getLogger(__name__)


def chunk_pages(
    pages: List[Dict[str, Any]],
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Splits page-by-page document text into smaller overlapping chunks while preserving metadata.

    Args:
        pages (List[Dict[str, Any]]): Extracted page data from extract_text_from_pdf(),
                                      where each entry contains {"page": int, "text": str}.
        source (str): The filename or identifier of the source document.
        chunk_size (int, optional): Maximum characters per chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlapping characters between consecutive chunks.
                                       Defaults to 200.

    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries containing:
                              [
                                  {
                                      "text": "...",
                                      "page": 1,
                                      "source": "filename.pdf"
                                  },
                                  ...
                              ]
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
        )

    if not pages:
        logger.warning("No pages provided for chunking.")
        return []

    # Initialize LangChain's RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunked_documents: List[Dict[str, Any]] = []

    for item in pages:
        page_number = item.get("page")
        raw_text = item.get("text", "")
        doc_source = item.get("source", source)

        # Ignore empty or whitespace-only text
        if not raw_text or not raw_text.strip():
            continue

        # Split the page text into overlapping chunks
        split_chunks = splitter.split_text(raw_text)

        for chunk in split_chunks:
            cleaned_text = chunk.strip()
            # Ignore empty chunks
            if not cleaned_text:
                continue

            chunked_documents.append({
                "text": cleaned_text,
                "page": page_number,
                "source": doc_source,
            })

    return chunked_documents


# Alias for backwards compatibility
chunk_documents = chunk_pages

__all__ = ["chunk_pages", "chunk_documents"]
