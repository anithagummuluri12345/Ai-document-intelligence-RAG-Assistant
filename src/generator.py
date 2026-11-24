"""
LLM Generation Component for AI Document Intelligence.

This module generates grounded answers to user questions based strictly on
retrieved document context, complete with page number and source file citations.
If the retrieved context lacks sufficient information, it explicitly declines to answer.
"""

import os
import sys
import re
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

NO_INFO_MESSAGE = "The provided documents do not contain sufficient information to answer this question."

SYSTEM_PROMPT = """You are an AI Document Intelligence Assistant.
Answer the user's question strictly based ONLY on the provided context below.

Rules:
1. Rely only on facts directly stated in the context. Do not extrapolate, assume, or use prior external knowledge.
2. For every statement or claim in your answer, provide an inline citation indicating the source file and page number, in the format: [Source: <filename>, Page: <page_number>].
3. If the provided context does not contain sufficient information to answer the question, state clearly:
   "The provided documents do not contain sufficient information to answer this question."
"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved chunks into a standardized context block with citations.

    Args:
        chunks (List[Dict[str, Any]]): List of retrieved chunk dictionaries.

    Returns:
        str: Formatted context string.
    """
    if not chunks:
        return ""

    context_blocks = []
    for chunk in chunks:
        source = chunk.get("source", "unknown_source")
        page = chunk.get("page", "unknown_page")
        text = chunk.get("text", "").strip()
        if text:
            context_blocks.append(f"[Document: {source} | Page: {page}]\n{text}")

    return "\n\n".join(context_blocks)


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Builds the full grounded prompt incorporating system instructions, context, and query.

    Args:
        query (str): The user's question.
        chunks (List[Dict[str, Any]]): Retrieved chunks.

    Returns:
        str: Prompt string.
    """
    context_text = format_context(chunks)
    if not context_text:
        context_text = "No relevant context found in documents."

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- CONTEXT ---\n{context_text}\n--- END CONTEXT ---\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )


def _call_openai_compatible_api(
    prompt: str,
    api_key: str,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> Optional[str]:
    """Helper to query OpenAI-compatible chat endpoints using requests/httpx."""
    import requests

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
    }

    url = f"{base_url.rstrip('/')}/chat/completions"
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        else:
            logger.warning("API returned status %d: %s", response.status_code, response.text)
    except Exception as err:
        logger.warning("API request failed: %s", err)

    return None


def _grounded_extractive_generation(
    query: str,
    chunks: List[Dict[str, Any]],
) -> str:
    """
    Deterministic grounded reader that extracts relevant facts directly from
    the provided chunks and appends exact [Source: ..., Page: ...] citations.
    If no relevant information is present, returns NO_INFO_MESSAGE.
    """
    if not chunks:
        return NO_INFO_MESSAGE

    # Clean and tokenize query terms (excluding common stopwords)
    stopwords = {
        "what", "is", "the", "how", "are", "used", "to", "in", "of", "and", "a",
        "an", "for", "with", "on", "by", "that", "this", "can", "be", "do", "does",
        "which", "from", "at", "as", "or", "it", "its", "their", "about"
    }
    query_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower())) - stopwords

    if not query_words:
        return NO_INFO_MESSAGE

    scored_sentences: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "document")
        page = chunk.get("page", 1)

        # Split text into candidate sentences
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        for sentence in raw_sentences:
            cleaned = sentence.strip()
            if len(cleaned) < 20:
                continue

            sentence_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", cleaned.lower()))
            overlap = query_words.intersection(sentence_words)

            if overlap:
                score = len(overlap) / (len(query_words) ** 0.5)
                scored_sentences.append({
                    "sentence": cleaned,
                    "score": score,
                    "source": source,
                    "page": page,
                    "overlap_count": len(overlap),
                })

        # If no sentences matched via strict keyword overlap, attempt a relaxed match handling simple singular/plural forms.
        if not scored_sentences:
            # Normalize query terms by stripping trailing 's' (simple plural handling) and include original terms.
            normalized_query = {w.rstrip('s') for w in query_words}
            normalized_query.update(query_words)

            # Scan all chunks for sentences containing any normalized query term.
            for chunk in chunks:
                text = chunk.get("text", "")
                source = chunk.get("source", "document")
                page = chunk.get("page", 1)
                raw_sentences = re.split(r"(?<=[.!?])\s+", text)
                for sentence in raw_sentences:
                    cleaned = sentence.strip()
                    if len(cleaned) < 20:
                        continue
                    sentence_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", cleaned.lower()))
                    normalized_sentence = {w.rstrip('s') for w in sentence_words}
                    normalized_sentence.update(sentence_words)
                    if normalized_query.intersection(normalized_sentence):
                        # Assign a modest score so it will be considered later.
                        scored_sentences.append({
                            "sentence": cleaned,
                            "score": 0.1,
                            "source": source,
                            "page": page,
                            "overlap_count": 0,
                        })
            # If still none, give up.
            if not scored_sentences:
                return NO_INFO_MESSAGE

    # Sort sentences by keyword relevance
    scored_sentences.sort(key=lambda s: s["score"], reverse=True)

    # The retriever already performs semantic similarity search.
    # Avoid rejecting relevant context because the document uses different wording.
    top_candidates = scored_sentences

    if not top_candidates:
        return NO_INFO_MESSAGE

    # Deduplicate and format top facts with citations
    seen = set()
    selected_facts = []
    for cand in top_candidates[:3]:
        key = cand["sentence"][:60].lower()
        if key not in seen:
            seen.add(key)
            citation = f"[Source: {cand['source']}, Page: {cand['page']}]"
            selected_facts.append(f"{cand['sentence']} {citation}")

    if not selected_facts:
        return NO_INFO_MESSAGE

    return "Based on the provided documents:\n\n" + "\n\n".join(selected_facts)


class LLMGenerator:
    """
    RAG Answer Generator with citations and strict grounding.
    """

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.groq_api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        self.model_name = model_name

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generates an answer to the query strictly using the retrieved chunks.

        Args:
            query (str): The user's question.
            chunks (List[Dict[str, Any]]): Retrieved context chunks.

        Returns:
            Dict[str, Any]: {
                "answer": str,
                "citations": List[Dict[str, Any]],
                "num_context_chunks": int
            }
        """
        if not chunks:
            return {
                "answer": NO_INFO_MESSAGE,
                "citations": [],
                "num_context_chunks": 0,
            }

        prompt = build_prompt(query, chunks)
        answer: Optional[str] = None

        # 1. Try Groq if configured
        if self.groq_api_key:
            model = self.model_name or "llama-3.1-8b-instant"
            answer = _call_openai_compatible_api(
                prompt=prompt,
                api_key=self.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
                model=model,
            )

        # 2. Try OpenAI if configured
        elif self.openai_api_key:
            model = self.model_name or "gpt-4o-mini"
            answer = _call_openai_compatible_api(
                prompt=prompt,
                api_key=self.openai_api_key,
                base_url="https://api.openai.com/v1",
                model=model,
            )

        # 3. Deterministic Grounded Generation Fallback
        if not answer:
            answer = _grounded_extractive_generation(query, chunks)

        # Extract citations present in the context chunks
        if answer == NO_INFO_MESSAGE:
            # When there is no sufficient information, return without citations
            return {
                "answer": answer,
                "citations": [],
                "num_context_chunks": len(chunks),
            }

        citations = []
        seen_citations = set()
        for chunk in chunks:
            source = chunk.get("source")
            page = chunk.get("page")
            pair = (source, page)
            if pair not in seen_citations and source is not None and page is not None:
                seen_citations.add(pair)
                citations.append({"source": source, "page": page})

        # Normalize unicode ligatures (e.g. \ufb01 -> fi) for safe display across all platforms
        import unicodedata
        normalized_answer = unicodedata.normalize("NFKD", answer)

        return {
            "answer": normalized_answer,
            "citations": citations,
            "num_context_chunks": len(chunks),
        }


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to generate an answer from query and context chunks.

    Args:
        query (str): The search question.
        chunks (List[Dict[str, Any]]): Retrieved context chunks.
        api_key (Optional[str], optional): API key for external LLM provider.
        model_name (Optional[str], optional): Model name for LLM.

    Returns:
        Dict[str, Any]: Response dictionary with 'answer', 'citations', and 'num_context_chunks'.
    """
    generator = LLMGenerator(
        openai_api_key=api_key,
        model_name=model_name,
    )
    return generator.generate(query, chunks)


# Aliases for compatibility
generate = generate_answer
answer_question = generate_answer

__all__ = [
    "LLMGenerator",
    "generate_answer",
    "generate",
    "answer_question",
    "format_context",
    "build_prompt",
    "NO_INFO_MESSAGE",
]


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("Testing LLM Generation Component")
    print("=" * 70)

    # Sample retrieved chunks mimicking src/retriever.py output
    sample_retrieved_chunks = [
        {
            "text": (
                "To face such a new challenge, this paper presents an ecosystem exploiting a "
                "classifier based on Deep Neural Networks to reveal the presence of images embedding "
                "malicious assets. Collected results indicated the effectiveness of the approach to "
                "detect malicious contents, even in the presence of an attacker trying to elude our "
                "framework via basic obfuscation techniques."
            ),
            "page": 1,
            "source": "stegomalware_study.pdf",
            "score": 0.42,
        },
        {
            "text": (
                "Deep learning can exploit multiple levels of abstractions and can capture relations "
                "between set of features directly from raw and noisy image data. Hence, Deep Neural "
                "Networks (DNNs) are a good candidate to face the issue of revealing threats hidden "
                "within digital pictures."
            ),
            "page": 2,
            "source": "stegomalware_study.pdf",
            "score": 0.51,
        }
    ]

    # Test Case 1: Grounded Question Answerable from Context
    q1 = "What classifier is presented to reveal images embedding malicious assets?"
    print(f"\n[Test 1: Answerable Query]")
    print(f"Question: '{q1}'\n")
    res1 = generate_answer(q1, sample_retrieved_chunks)
    print(f"Generated Answer:\n{res1['answer']}\n")
    print(f"Citations: {res1['citations']}")
    print("-" * 70)

    # Test Case 2: Query NOT in Context
    q2 = "What is the capital of Australia?"
    print(f"\n[Test 2: Unanswerable Query]")
    print(f"Question: '{q2}'\n")
    res2 = generate_answer(q2, sample_retrieved_chunks)
    print(f"Generated Answer:\n{res2['answer']}\n")
    print(f"Correctly declined to hallucinate: {res2['answer'] == NO_INFO_MESSAGE}")
    print("=" * 70)
