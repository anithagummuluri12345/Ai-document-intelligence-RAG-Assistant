"""
Retrieval Component for AI Document Intelligence.

This module takes user search queries, embeds them using the 384-dimensional
sentence-transformers model, queries the persistent FAISS vector database,
and returns the top-k most relevant document chunks with metadata.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Ensure project root is on sys.path for both module and direct script execution
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from src.embedder import load_embedding_model, DEFAULT_MODEL_NAME
    from src.vector_db import VectorDB, DEFAULT_DB_DIR
except ImportError:
    from embedder import load_embedding_model, DEFAULT_MODEL_NAME
    from vector_db import VectorDB, DEFAULT_DB_DIR


# Configure module-level logger
logger = logging.getLogger(__name__)


class Retriever:
    """
    Encapsulates vector retrieval logic using FAISS and SentenceTransformer.
    """

    def __init__(
        self,
        db_dir: Union[str, Path] = DEFAULT_DB_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.db_dir = Path(db_dir)
        self.model_name = model_name
        self._model = None
        self._vector_db = None

    @property
    def model(self):
        """Lazy-loaded SentenceTransformer model."""
        if self._model is None:
            self._model = load_embedding_model(self.model_name)
        return self._model

    @property
    def vector_db(self) -> VectorDB:
        """Lazy-loaded VectorDB instance from disk."""
        if self._vector_db is None:
            self._vector_db = VectorDB.load(self.db_dir)
        return self._vector_db

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant chunks for a given text query.

        Args:
            query (str): The search query text.
            top_k (int): Number of most relevant results to retrieve. Defaults to 3.

        Returns:
            List[Dict[str, Any]]: List of retrieved chunks, each containing:
                                  - 'text': Chunk text content
                                  - 'page': Originating page number
                                  - 'source': Source document filename
                                  - 'score': Similarity distance (lower L2 = closer match)
        """
        if not query or not query.strip():
            logger.warning("Empty query received.")
            return []

        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k}")

        vdb = self.vector_db
        if vdb.ntotal == 0:
            logger.warning("Vector database is empty.")
            return []

        actual_k = min(top_k, vdb.ntotal)

        # Generate 384-dimensional embedding for query
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

        # Search FAISS index for top-k closest vectors
        distances, indices = vdb.index.search(query_embedding, actual_k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(vdb.chunks):
                continue

            chunk = vdb.chunks[idx]
            results.append({
                "text": chunk.get("text", ""),
                "page": chunk.get("page"),
                "source": chunk.get("source"),
                "score": float(dist),
            })

        return results


# Global singleton cache for functional API
_cached_retriever: Optional[Retriever] = None


def retrieve(
    query: str,
    top_k: int = 3,
    db_dir: Union[str, Path] = DEFAULT_DB_DIR,
    retriever: Optional[Retriever] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves top-k most relevant chunks matching the user query.

    Args:
        query (str): The search query text.
        top_k (int, optional): Number of results to return. Defaults to 3.
        db_dir (Union[str, Path], optional): Vector database directory path. Defaults to "vector_db".
        retriever (Optional[Retriever], optional): Pre-existing Retriever instance.

    Returns:
        List[Dict[str, Any]]: List of result dictionaries containing 'text', 'page', 'source', 'score'.
    """
    global _cached_retriever
    if retriever is not None:
        return retriever.retrieve(query, top_k=top_k)

    if _cached_retriever is None or _cached_retriever.db_dir != Path(db_dir):
        _cached_retriever = Retriever(db_dir=db_dir)

    return _cached_retriever.retrieve(query, top_k=top_k)


# Functional aliases for compatibility
retrieve_chunks = retrieve
search = retrieve

__all__ = ["Retriever", "retrieve", "retrieve_chunks", "search"]


if __name__ == "__main__":
    import sys

    # Ensure unicode characters (e.g. PDF ligatures like 'fi', 'fl') don't crash Windows console
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("Testing Retrieval Component")
    print("=" * 70)

    test_query = "How are deep neural networks used to detect steganographic threats in images?"
    print(f"\nQuery: '{test_query}'\n")

    try:
        retrieved_chunks = retrieve(test_query, top_k=3)
        print(f"Retrieved {len(retrieved_chunks)} relevant chunk(s):\n")

        for idx, res in enumerate(retrieved_chunks, start=1):
            print(f"--- [Result #{idx}] ---")
            print(f"Page   : {res.get('page')}")
            print(f"Source : {res.get('source')}")
            print(f"Score  : {res.get('score'):.4f} (L2 distance)")
            print(f"Text   : {res.get('text')[:300]}...")
            print("-" * 70)

        print("\n[SUCCESS] Retrieval component executed and returned relevant chunks successfully!")
    except Exception as exc:
        print(f"\n[ERROR] Retrieval failed: {exc}", file=sys.stderr)

