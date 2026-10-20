"""
Vector Database & Indexing Component for AI Document Intelligence.

This module handles indexing dense vector embeddings (384-dimensional)
into a FAISS vector index, associating each vector with its chunk text,
page number, and source filename metadata, and persisting the database to disk.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import faiss

# Configure module-level logger
logger = logging.getLogger(__name__)

DEFAULT_DIMENSION = 384
DEFAULT_DB_DIR = "vector_db"


class VectorDB:
    """
    Vector database manager wrapping FAISS index and chunk metadata.
    """

    def __init__(self, dimension: int = DEFAULT_DIMENSION):
        """
        Initializes an empty FAISS L2 index with the specified vector dimension.

        Args:
            dimension (int): Vector dimension (default: 384 for all-MiniLM-L6-v2).
        """
        self.dimension = dimension
        self.index: faiss.Index = faiss.IndexFlatL2(dimension)
        self.chunks: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def ntotal(self) -> int:
        """Returns the total number of vectors in the FAISS index."""
        return self.index.ntotal

    def __getattr__(self, name: str) -> Any:
        """Delegates attribute lookups directly to the underlying FAISS index."""
        return getattr(self.index, name)

    def __iter__(self):
        """Allows unpacking as: index, chunks = vector_db"""
        yield self.index
        yield self.chunks

    def add(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Adds text chunks and their corresponding embeddings into the vector database.

        Args:
            chunks (List[Dict[str, Any]]): List of chunk dictionaries containing 'text', 'page', and 'source'.
            embeddings (np.ndarray): 2D NumPy array of shape (N, 384) with float32 vectors.

        Raises:
            ValueError: If the number of chunks does not match the number of embeddings,
                        or if embedding dimensions are inconsistent.
        """
        if not chunks:
            logger.warning("No chunks provided to add.")
            return

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        num_chunks = len(chunks)
        num_embeddings = embeddings.shape[0]

        if num_chunks != num_embeddings:
            raise ValueError(
                f"Count mismatch: received {num_chunks} chunks but {num_embeddings} embeddings."
            )

        emb_dim = embeddings.shape[1]
        if emb_dim != self.dimension:
            logger.warning(
                "Embedding dimension %d does not match index dimension %d. Updating index.",
                emb_dim,
                self.dimension,
            )
            self.dimension = emb_dim
            if self.index.ntotal == 0:
                self.index = faiss.IndexFlatL2(emb_dim)

        # FAISS requires contiguous float32 arrays
        embeddings_f32 = np.ascontiguousarray(embeddings, dtype=np.float32)
        current_offset = self.index.ntotal

        # Add vectors to the FAISS index
        self.index.add(embeddings_f32)

        # Store chunk text, page, and source metadata aligned with FAISS index IDs
        for idx, chunk in enumerate(chunks, start=current_offset):
            metadata_entry = {
                "id": idx,
                "text": chunk.get("text", ""),
                "page": chunk.get("page"),
                "source": chunk.get("source"),
            }
            # Preserve any additional custom keys without mutation
            for k, v in chunk.items():
                if k not in metadata_entry:
                    metadata_entry[k] = v

            self.chunks.append(metadata_entry)

        logger.info(
            "Successfully indexed %d vectors. Total vectors in index: %d",
            num_chunks,
            self.index.ntotal,
        )

    def save(self, db_dir: Union[str, Path] = DEFAULT_DB_DIR) -> None:
        """
        Persists the FAISS index and chunk metadata to the specified directory.

        Args:
            db_dir (Union[str, Path]): Directory path to store vector_db files.
        """
        target_dir = Path(db_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        json_file = target_dir / "metadata.json"
        pickle_file = target_dir / "metadata.pkl"

        # Write the FAISS binary index
        faiss.write_index(self.index, str(index_file))

        # Save metadata in human-readable JSON format
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2, ensure_ascii=False)

        # Save metadata in fast binary pickle format
        with open(pickle_file, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("Vector database successfully persisted to: %s", target_dir)

    @classmethod
    def load(cls, db_dir: Union[str, Path] = DEFAULT_DB_DIR) -> "VectorDB":
        """
        Loads an existing FAISS index and chunk metadata from disk.

        Args:
            db_dir (Union[str, Path]): Path to directory containing index.faiss and metadata.

        Returns:
            VectorDB: Reconstructed VectorDB instance with populated index and chunk metadata.

        Raises:
            FileNotFoundError: If the index file or metadata file is missing.
        """
        target_dir = Path(db_dir)
        index_file = target_dir / "index.faiss"
        pickle_file = target_dir / "metadata.pkl"
        json_file = target_dir / "metadata.json"

        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index file not found at: {index_file}")

        # Load FAISS index
        faiss_index = faiss.read_index(str(index_file))

        # Load metadata
        if pickle_file.exists():
            with open(pickle_file, "rb") as f:
                chunks = pickle.load(f)
        elif json_file.exists():
            with open(json_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found in: {target_dir}")

        instance = cls(dimension=faiss_index.d)
        instance.index = faiss_index
        instance.chunks = chunks
        return instance


def build_vector_db(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    db_dir: Union[str, Path] = DEFAULT_DB_DIR,
) -> VectorDB:
    """
    Builds, indexes, and saves the vector database from chunks and embeddings.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries containing 'text', 'page', and 'source'.
        embeddings (np.ndarray): 2D NumPy array of 384-dimensional embeddings.
        db_dir (Union[str, Path]): Directory to save the FAISS index and metadata files.

    Returns:
        VectorDB: The populated and persisted VectorDB instance.
    """
    dimension = embeddings.shape[1] if hasattr(embeddings, "shape") and embeddings.ndim > 1 else DEFAULT_DIMENSION
    vector_db = VectorDB(dimension=dimension)
    vector_db.add(chunks, embeddings)
    vector_db.save(db_dir)
    return vector_db


def save_vector_db(
    db: Union[VectorDB, Tuple[faiss.Index, List[Dict[str, Any]]]],
    db_dir: Union[str, Path] = DEFAULT_DB_DIR,
) -> None:
    """
    Helper function to save an existing VectorDB instance or (index, chunks) tuple.
    """
    if isinstance(db, VectorDB):
        db.save(db_dir)
    elif isinstance(db, tuple) and len(db) == 2:
        index, chunks = db
        target_dir = Path(db_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(target_dir / "index.faiss"))
        with open(target_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        with open(target_dir / "metadata.pkl", "wb") as f:
            pickle.dump(chunks, f)
    else:
        raise TypeError("Expected VectorDB instance or (index, chunks) tuple.")


def load_vector_db(db_dir: Union[str, Path] = DEFAULT_DB_DIR) -> VectorDB:
    """
    Loads the VectorDB from the specified directory.
    """
    return VectorDB.load(db_dir)


# Standard functional aliases for interoperability
create_vector_db = build_vector_db
create_index = build_vector_db
build_index = build_vector_db
index_chunks = build_vector_db
load_index = load_vector_db
save_index = save_vector_db
VectorStore = VectorDB

__all__ = [
    "VectorDB",
    "VectorStore",
    "build_vector_db",
    "create_vector_db",
    "build_index",
    "create_index",
    "index_chunks",
    "save_vector_db",
    "load_vector_db",
    "load_index",
    "save_index",
]


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Vector Database & Indexing Component")
    print("=" * 60)

    # 1. Create sample chunk data preserving text, page, and source metadata
    sample_chunks = [
        {"text": "Artificial Intelligence is transforming document processing.", "page": 1, "source": "ai_overview.pdf"},
        {"text": "Retrieval-Augmented Generation combines retrieval with LLMs.", "page": 2, "source": "rag_guide.pdf"},
        {"text": "Vector databases store dense representations for semantic search.", "page": 3, "source": "vector_db.pdf"},
    ]

    # 2. Generate sample 384-dimensional embeddings (float32)
    rng = np.random.default_rng(42)
    sample_embeddings = rng.random((len(sample_chunks), DEFAULT_DIMENSION), dtype=np.float32)

    # 3. Build and persist vector database
    test_db_dir = "vector_db"
    print(f"\n[+] Building index with {len(sample_chunks)} chunks into '{test_db_dir}'...")
    db = build_vector_db(sample_chunks, sample_embeddings, db_dir=test_db_dir)
    print(f"    Index vector count : {db.ntotal}")
    print(f"    Vector dimension   : {db.dimension}")

    # 4. Reload from disk to verify persistence
    print("\n[+] Verifying persistence by loading index and metadata from disk...")
    loaded_db = load_vector_db(db_dir=test_db_dir)
    print(f"    Loaded vectors     : {loaded_db.ntotal}")
    print(f"    Loaded metadata len: {len(loaded_db.chunks)}")

    # 5. Verify metadata integrity
    assert loaded_db.ntotal == len(sample_chunks), "Vector count mismatch!"
    assert len(loaded_db.chunks) == len(sample_chunks), "Metadata count mismatch!"
    for i, chunk in enumerate(loaded_db.chunks):
        assert "text" in chunk and "page" in chunk and "source" in chunk, f"Missing metadata in chunk {i}"
        print(f"    Chunk {i}: page={chunk['page']}, source={chunk['source']}, text_preview='{chunk['text'][:40]}...'")

    print("\n[SUCCESS] Vector database indexing and persistence verified successfully!")
    print("=" * 60)
