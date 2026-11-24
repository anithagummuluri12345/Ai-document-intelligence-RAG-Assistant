"""
Evaluation Framework for the AI Document Intelligence & RAG Assistant.

Runs a fixed set of questions against the existing RAG pipeline, measures
four reproducible metrics, prints a formatted report, and saves results to
evaluation/results.json.

Usage:
    python evaluation/evaluate_rag.py

Requirements:
    - The vector database must already be populated (run the app and ingest
      test.pdf.pdf first, or let this script ingest it automatically).
    - No external API keys or internet access are needed; the deterministic
      fallback generator is used when no LLM key is configured.
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so 'src.*' imports work correctly
# regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # evaluation/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # project root
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Reconfigure stdout for clean Unicode output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.rag_pipeline import RAGPipeline
from src.generator import NO_INFO_MESSAGE

# ---------------------------------------------------------------------------
# Logging - keep INFO-level messages to a minimum so the report is readable
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation Dataset
#
# expected_pages: the page numbers (1-indexed) that should ideally appear
#   among the retrieved chunks for a correct retrieval hit.
#   Set to None for out-of-document questions.
# answerable: True if the question can be answered from the document.
#   False questions are used to test the "no-info" refusal behaviour.
# ---------------------------------------------------------------------------
EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "question": "What dataset was used in the experiments?",
        # Page 11 contains dataset details (8,000 images, Base64 Test, ZIP Test).
        "expected_pages": [11],
        "answerable": True,
    },
    {
        "question": "Why are Deep Neural Networks good candidates to reveal hidden threats?",
        # The introduction/related-work section (pages 1-3) discusses DNNs.
        "expected_pages": [1, 2, 3],
        "answerable": True,
    },
    {
        "question": "What classifier is proposed to detect steganographic threats in images?",
        # The main contribution is described early in the paper (pages 1-4).
        "expected_pages": [1, 2, 3, 4],
        "answerable": True,
    },
    {
        "question": "What is the main objective of this research paper?",
        # Objective is stated in the abstract/introduction (pages 1-2).
        "expected_pages": [1, 2],
        "answerable": True,
    },
    {
        "question": "What is the capital of Australia?",
        # This question has nothing to do with the document.
        "expected_pages": None,
        "answerable": False,
    },
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _pages_from_result(result: Dict[str, Any]) -> List[int]:
    """Return a sorted list of unique page numbers in the retrieved chunks."""
    pages = []
    for chunk in result.get("source_chunks", []):
        p = chunk.get("page")
        if p is not None:
            try:
                pages.append(int(p))
            except (ValueError, TypeError):
                pass
    return sorted(set(pages))


def _has_citations(result: Dict[str, Any]) -> bool:
    """Return True when the result contains at least one citation."""
    return bool(result.get("citations"))


def _retrieval_hit(result: Dict[str, Any], expected_pages: Optional[List[int]]) -> bool:
    """
    Return True when at least one retrieved page is in the expected list.
    For out-of-document questions (expected_pages is None) always returns True
    (the concept of "hit" does not apply to these questions).
    """
    if expected_pages is None:
        return True
    retrieved_pages = _pages_from_result(result)
    return bool(set(retrieved_pages).intersection(expected_pages))


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    pipeline: RAGPipeline,
    dataset: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run every question in dataset through pipeline and collect metrics.

    Args:
        pipeline: An initialised RAGPipeline instance.
        dataset:  List of evaluation records (see EVAL_DATASET).
        top_k:    Number of chunks to retrieve per question.

    Returns:
        A dict containing 'per_question' results and aggregate 'metrics'.
    """
    per_question: List[Dict[str, Any]] = []

    for entry in dataset:
        question = entry["question"]
        answerable = entry["answerable"]
        expected_pages = entry.get("expected_pages")

        print(f"\n  [Q] {question}")
        t0 = time.perf_counter()
        result = pipeline.query(question, top_k=top_k)
        elapsed = time.perf_counter() - t0

        retrieved_pages = _pages_from_result(result)
        hit = _retrieval_hit(result, expected_pages)
        answer_is_no_info = result["answer"].strip() == NO_INFO_MESSAGE.strip()
        has_cit = _has_citations(result)

        record = {
            "question": question,
            "answerable": answerable,
            "expected_pages": expected_pages,
            "retrieved_pages": retrieved_pages,
            "num_chunks_retrieved": result.get("num_chunks_retrieved", 0),
            "retrieval_hit": hit,
            "answer_is_no_info": answer_is_no_info,
            "has_citations": has_cit,
            "citations": result.get("citations", []),
            "answer": result["answer"],
            "latency_seconds": round(elapsed, 3),
        }
        per_question.append(record)

        # Brief inline progress
        hit_label = "HIT" if hit else "MISS"
        cit_label = "CITED" if has_cit else "NO-CITE"
        no_info_label = "NO-INFO" if answer_is_no_info else "ANSWERED"
        print(
            f"       Pages retrieved: {retrieved_pages} | {hit_label} | "
            f"{cit_label} | {no_info_label} | {elapsed:.2f}s"
        )

    # -----------------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------------
    answerable_qs = [r for r in per_question if r["answerable"]]
    unanswerable_qs = [r for r in per_question if not r["answerable"]]

    # Metric 1: Retrieval Hit Rate
    retrieval_hits = sum(1 for r in answerable_qs if r["retrieval_hit"])
    retrieval_hit_rate = (
        retrieval_hits / len(answerable_qs) if answerable_qs else 0.0
    )

    # Metric 2: No-Info Accuracy
    no_info_correct = sum(1 for r in unanswerable_qs if r["answer_is_no_info"])
    no_info_accuracy = (
        no_info_correct / len(unanswerable_qs) if unanswerable_qs else 0.0
    )

    # Metric 3: Citation Presence Rate
    cited = sum(1 for r in answerable_qs if r["has_citations"])
    citation_presence_rate = (
        cited / len(answerable_qs) if answerable_qs else 0.0
    )

    # Metric 4: Average chunks retrieved
    avg_chunks = (
        sum(r["num_chunks_retrieved"] for r in per_question) / len(per_question)
        if per_question
        else 0.0
    )

    metrics = {
        "total_questions": len(per_question),
        "answerable_questions": len(answerable_qs),
        "unanswerable_questions": len(unanswerable_qs),
        "retrieval_hit_rate": round(retrieval_hit_rate, 4),
        "retrieval_hits": retrieval_hits,
        "no_info_accuracy": round(no_info_accuracy, 4),
        "no_info_correct": no_info_correct,
        "citation_presence_rate": round(citation_presence_rate, 4),
        "citations_present": cited,
        "avg_chunks_retrieved": round(avg_chunks, 2),
    }

    return {"per_question": per_question, "metrics": metrics}


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def print_report(eval_results: Dict[str, Any]) -> None:
    """Pretty-print the evaluation report to stdout."""
    metrics = eval_results["metrics"]
    per_question = eval_results["per_question"]

    w = 72
    bar = "=" * w

    print(f"\n{bar}")
    print(" AI Document Intelligence & RAG Evaluation Report".center(w))
    print(bar)

    # ---- Per-question detail ----
    print("\n DETAILED RESULTS\n" + "-" * w)
    for i, r in enumerate(per_question, 1):
        status = "ANSWERABLE" if r["answerable"] else "OUT-OF-DOC"
        print(f"\n  Q{i} [{status}]: {r['question']}")
        print(f"      Retrieved pages : {r['retrieved_pages']}")
        print(f"      Expected pages  : {r['expected_pages']}")
        hit_str = "YES" if r["retrieval_hit"] else "NO"
        print(f"      Retrieval hit   : {hit_str}")
        cit_str = "YES" if r["has_citations"] else "NO"
        print(f"      Has citations   : {cit_str}  {r['citations']}")
        ni_str = "YES" if r["answer_is_no_info"] else "NO"
        print(f"      Returned no-info: {ni_str}")
        preview = r["answer"][:200].replace("\n", " ")
        if len(r["answer"]) > 200:
            preview += "..."
        print(f"      Answer preview  : {preview}")
        print(f"      Latency         : {r['latency_seconds']}s")

    # ---- Aggregate metrics ----
    print(f"\n{bar}")
    print(" AGGREGATE METRICS".center(w))
    print(bar)
    print(
        f"  Total questions            : {metrics['total_questions']}"
        f"  (answerable: {metrics['answerable_questions']},"
        f" out-of-doc: {metrics['unanswerable_questions']})"
    )
    print()

    def pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    m1_hits = metrics["retrieval_hits"]
    m1_total = metrics["answerable_questions"]
    print(
        f"  1. Retrieval Hit Rate      : {pct(metrics['retrieval_hit_rate'])}"
        f"  ({m1_hits}/{m1_total} answerable questions"
        " had >=1 expected page retrieved)"
    )

    m2_correct = metrics["no_info_correct"]
    m2_total = metrics["unanswerable_questions"]
    print(
        f"  2. No-Info Accuracy        : {pct(metrics['no_info_accuracy'])}"
        f"  ({m2_correct}/{m2_total} out-of-doc questions"
        " correctly refused)"
    )

    m3_cited = metrics["citations_present"]
    m3_total = metrics["answerable_questions"]
    print(
        f"  3. Citation Presence Rate  : {pct(metrics['citation_presence_rate'])}"
        f"  ({m3_cited}/{m3_total} answerable questions"
        " returned >=1 citation)"
    )

    print(
        f"  4. Avg Chunks Retrieved    : {metrics['avg_chunks_retrieved']}"
        " chunks / question"
    )

    print(f"\n{bar}\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results_path = _SCRIPT_DIR / "results.json"

    print("\n" + "=" * 72)
    print(" AI Document Intelligence - RAG Evaluation".center(72))
    print("=" * 72)

    # ---- Initialise the pipeline ----
    print("\n[*] Initialising RAG pipeline ...")
    pipeline = RAGPipeline(top_k=5)

    # ---- Auto-ingest test.pdf.pdf if the vector DB appears empty ----
    pdf_path = _PROJECT_ROOT / "test.pdf.pdf"
    try:
        total_in_db = pipeline.retriever.vector_db.ntotal
    except Exception:
        total_in_db = 0

    if total_in_db == 0:
        if pdf_path.exists():
            print(f"[*] Vector database is empty - ingesting '{pdf_path.name}' ...")
            n = pipeline.ingest_pdf(pdf_path)
            print(f"    Indexed {n} chunks.")
        else:
            print(
                "[!] WARNING: vector database is empty and 'test.pdf.pdf' was not found.\n"
                "    Please ingest a PDF before running evaluation."
            )
    else:
        print(f"[*] Using existing vector database ({total_in_db} vectors).")

    # ---- Run evaluation ----
    print(f"\n[*] Running {len(EVAL_DATASET)} evaluation questions ...")
    eval_results = run_evaluation(pipeline, EVAL_DATASET, top_k=5)

    # ---- Print report ----
    print_report(eval_results)

    # ---- Save results ----
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(eval_results, fh, indent=2, ensure_ascii=False)
    print(f"[+] Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
