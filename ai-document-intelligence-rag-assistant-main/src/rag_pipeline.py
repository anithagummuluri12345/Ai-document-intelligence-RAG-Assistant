"""
RAG Pipeline Orchestration Component for AI Document Intelligence.

This module integrates PDF loading, text chunking, embedding generation,
vector indexing, retrieval, and LLM answer generation into a unified,
end-to-end RAG pipeline.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from src.pdf_loader import extract_text_from_pdf
    from src.chunker import chunk_pages
    from src.embedder import load_embedding_model, create_embeddings
    from src.vector_db import VectorDB, build_vector_db, DEFAULT_DB_DIR
    from src.retriever import Retriever
    from src.generator import LLMGenerator, NO_INFO_MESSAGE
except ImportError:
    from pdf_loader import extract_text_from_pdf
    from chunker import chunk_pages
    from embedder import load_embedding_model, create_embeddings
    from vector_db import VectorDB, build_vector_db, DEFAULT_DB_DIR
    from retriever import Retriever
    from generator import LLMGenerator, NO_INFO_MESSAGE

# Configure module-level logger
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-End Retrieval-Augmented Generation (RAG) Pipeline.
    """

    def __init__(
        self,
        db_dir: Union[str, Path] = DEFAULT_DB_DIR,
        top_k: int = 3,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        """
        Initializes the complete RAG pipeline.

        Args:
            db_dir (Union[str, Path]): Path to persistent vector database directory.
            top_k (int): Default number of top chunks to retrieve. Defaults to 3.
            openai_api_key (Optional[str]): OpenAI API key.
            groq_api_key (Optional[str]): Groq API key.
            llm_model (Optional[str]): Model name for LLM generation.
        """
        self.db_dir = Path(db_dir)
        self.top_k = top_k

        # Initialize Retriever (loads model and vector database)
        self.retriever = Retriever(db_dir=self.db_dir)

        # Initialize Generator
        self.generator = LLMGenerator(
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            model_name=llm_model,
        )

    def ingest_pdf(self, pdf_path: Union[str, Path]) -> int:
        """
        Ingests a new PDF document into the pipeline:
        1. Extracts page-by-page text.
        2. Chunks pages into 1000-char overlapping blocks.
        3. Generates 384-dimensional dense embeddings.
        4. Indexes chunks and embeddings into the persistent FAISS vector database.

        Args:
            pdf_path (Union[str, Path]): Path to target PDF file.

        Returns:
            int: Total number of chunks indexed.
        """
        path_obj = Path(pdf_path)
        logger.info("Ingesting PDF: %s", path_obj)

        # Step 1: Extract text page by page
        pages = extract_text_from_pdf(path_obj)
        if not pages:
            logger.warning("No pages extracted from: %s", path_obj)
            return 0

        # Step 2: Chunk text with metadata preservation
        chunks = chunk_pages(pages, source=path_obj.name)
        if not chunks:
            logger.warning("No valid chunks produced for: %s", path_obj)
            return 0

        # Step 3: Compute 384-dim embeddings using pipeline's shared model
        embeddings = create_embeddings(chunks, model=self.retriever.model)

        # Step 4: Add to vector database and persist
        try:
            # If database already exists, load and append
            vdb = VectorDB.load(self.db_dir)
            vdb.add(chunks, embeddings)
        except Exception:
            # Otherwise create a fresh vector database
            vdb = build_vector_db(chunks, embeddings, db_dir=self.db_dir)

        vdb.save(self.db_dir)
        self.retriever._vector_db = vdb

        logger.info(
            "Successfully ingested '%s'. Indexed %d new chunks (Total in DB: %d).",
            path_obj.name,
            len(chunks),
            vdb.ntotal,
        )
        return len(chunks)

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes an end-to-end RAG query:
        1. Retrieves the top-k most relevant chunks matching the question.
        2. Feeds retrieved chunks into the grounded LLM generator.
        3. Returns the final answer with citations.

        Args:
            question (str): The user's query or question.
            top_k (Optional[int]): Number of chunks to retrieve (defaults to pipeline's top_k).

        Returns:
            Dict[str, Any]: Dictionary containing:
                            - 'question': Original user question
                            - 'answer': Generated answer with inline citations
                            - 'citations': List of unique {'source': ..., 'page': ...} pairs
                            - 'source_chunks': List of retrieved chunk dictionaries
                            - 'num_chunks_retrieved': Count of retrieved chunks
        """
        if not question or not question.strip():
            return {
                "question": question,
                "answer": NO_INFO_MESSAGE,
                "citations": [],
                "source_chunks": [],
                "num_chunks_retrieved": 0,
            }

        k = top_k if top_k is not None else self.top_k

        # 1. Retrieve most relevant context chunks
        retrieved_chunks = self.retriever.retrieve(question, top_k=k)

        # 2. Generate grounded answer strictly based on retrieved context
        gen_result = self.generator.generate(question, retrieved_chunks)

        return {
            "question": question,
            "answer": gen_result["answer"],
            "citations": gen_result["citations"],
            "source_chunks": retrieved_chunks,
            "num_chunks_retrieved": len(retrieved_chunks),
        }

    # Alias for natural conversational invocation
    ask = query


def answer_query(
    question: str,
    top_k: int = 3,
    db_dir: Union[str, Path] = DEFAULT_DB_DIR,
) -> Dict[str, Any]:
    """
    Convenience functional interface to query the RAG pipeline.

    Args:
        question (str): User question.
        top_k (int): Number of context chunks. Defaults to 3.
        db_dir (Union[str, Path]): Vector DB storage path. Defaults to 'vector_db'.

    Returns:
        Dict[str, Any]: Complete RAG response dictionary.
    """
    pipeline = RAGPipeline(db_dir=db_dir, top_k=top_k)
    return pipeline.query(question)


# Aliases for compatibility
run_pipeline = answer_query
query_rag = answer_query

__all__ = ["RAGPipeline", "answer_query", "run_pipeline", "query_rag"]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 75)
    print("AI Document Intelligence & RAG Assistant - End-to-End Pipeline Test")
    print("=" * 75)

    pipeline = RAGPipeline(top_k=3)

    # Optional: If test.pdf.pdf is present, ensure full document is ingested
    sample_pdf = Path("test.pdf.pdf")
    if sample_pdf.exists():
        print(f"\n[+] Ingesting full document '{sample_pdf.name}' into vector database...")
        total_chunks = pipeline.ingest_pdf(sample_pdf)
        print(f"    Indexed chunks: {total_chunks} (Total in DB: {pipeline.retriever.vector_db.ntotal})")

    # Sample End-to-End Question
    sample_question = "What architecture or classifier is proposed to detect steganographic threats in images?"
    print(f"\n[?] User Question:\n    \"{sample_question}\"")

    print("\n[*] Running End-to-End Pipeline (Retrieve -> Generate -> Cite)...")
    response = pipeline.query(sample_question)

    print("\n[+] Final Answer:")
    print("-" * 75)
    print(response["answer"])
    print("-" * 75)

    print("\n[+] Source Citations:")
    for cit in response["citations"]:
        print(f"    • Document: {cit['source']} | Page: {cit['page']}")

    print("\n[+] Retrieved Context Details:")
    for idx, chk in enumerate(response["source_chunks"], 1):
        print(f"    Chunk #{idx} [Page {chk['page']}, {chk['source']}] (L2 Distance: {chk['score']:.4f})")

    print("\n" + "=" * 75)
    print("[SUCCESS] End-to-end RAG pipeline completed successfully!")
    print("=" * 75)
