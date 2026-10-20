"""
Embedding Component for AI Document Intelligence.

This module handles loading the sentence-transformers embedding model
and generating dense vector embeddings from chunked text documents.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

# Configure module-level logger
logger = logging.getLogger(__name__)

# Default model identifier
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """
    Loads and initializes the SentenceTransformer embedding model.

    Args:
        model_name (str): HuggingFace model repository identifier.
                          Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.

    Returns:
        SentenceTransformer: The loaded embedding model instance.

    Raises:
        RuntimeError: If the model fails to download or load into memory.
    """
    try:
        logger.info("Loading embedding model: %s", model_name)
        model = SentenceTransformer(model_name)
        return model
    except Exception as err:
        logger.error("Failed to load embedding model '%s': %s", model_name, err)
        raise RuntimeError(
            f"Failed to load embedding model '{model_name}'. "
            f"Please verify your network connection and model repository name. "
            f"Original error: {err}"
        ) from err


def create_embeddings(
    chunks: List[Dict[str, Any]],
    model: Optional[SentenceTransformer] = None,
) -> np.ndarray:
    """
    Generates dense vector embeddings for the 'text' field of each provided chunk.

    The original chunks and their metadata (page, source) remain completely untouched.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries produced by src/chunker.py,
                                       each containing at least the 'text' key.
        model (Optional[SentenceTransformer], optional): A pre-loaded model instance.
                                                         If None, load_embedding_model() will be used.

    Returns:
        np.ndarray: A 2D NumPy array of shape (N, embedding_dim) containing the vector representations.
                    Returns an empty NumPy array of shape (0, 384) if chunks is empty.

    Raises:
        RuntimeError: If embedding generation fails.
    """
    if not chunks:
        logger.warning("Empty chunks list received. Returning empty numpy array.")
        return np.empty((0, 384), dtype=np.float32)

    # Load the model if not provided by the caller
    if model is None:
        model = load_embedding_model()

    # Extract only the text content without mutating the input chunk dictionaries
    texts: List[str] = [chunk.get("text", "") for chunk in chunks]

    try:
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # Ensure return type is explicitly a numpy array
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype=np.float32)

        return embeddings

    except Exception as err:
        logger.error("Failed to generate embeddings: %s", err)
        raise RuntimeError(f"Error during embedding generation: {err}") from err
