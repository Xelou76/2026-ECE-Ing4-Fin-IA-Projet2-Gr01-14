"""RAGAS evaluation benchmarks for the BrightVest Financial RAG system.

Three benchmarks are provided:

* ``ragas``      – Faithfulness, answer relevancy, context precision via RAGAS.
* ``retrieval``  – Dense vs Hybrid vs Hybrid+Rerank retrieval comparison.
* ``ablation``   – RAG vs LLM-only to quantify context contribution.

Usage::

    python -m src.evaluation.eval_ragas --benchmark ragas
    python -m src.evaluation.eval_ragas --benchmark retrieval
    python -m src.evaluation.eval_ragas --benchmark ablation
    python -m src.evaluation.eval_ragas --benchmark all
    python -m src.evaluation.eval_ragas --benchmark ragas --questions 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from groq import Groq

from src.config import (
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
)
from src.generation.agent import FinancialAgent
from src.retrieval.dense import DenseRetriever, RetrievedDocument
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_EVAL_DIR = Path(__file__).parent
_QUESTIONS_FILE = _EVAL_DIR / "test_questions.json"
_RESULTS_DIR = _EVAL_DIR / "results"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_questions(limit: int | None = None) -> list[dict[str, Any]]:
    """Load test questions from JSON file.

    Args:
        limit: Optional cap on the number of questions returned.

    Returns:
        List of question dicts with keys: question, ground_truth, query_type, tickers.
    """
    with open(_QUESTIONS_FILE, "r", encoding="utf-8") as fh:
        questions: list[dict[str, Any]] = json.load(fh)
    if limit is not None:
        questions = questions[:limit]
    return questions


def _ensure_results_dir() -> Path:
    """Create the results directory if it does not already exist.

    Returns:
        Path to the results directory.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return _RESULTS_DIR


def _timestamp() -> str:
    """Return an ISO-8601-style timestamp string safe for file names.

    Returns:
        String like ``20260320T143022``.
    """
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


def _save_results(data: Any, filename: str) -> Path:
    """Serialise *data* as JSON and write it to the results directory.

    Numpy scalar values are converted to plain Python floats/ints before
    serialisation so that json.dumps does not raise a TypeError.

    Args:
        data: JSON-serialisable object (dicts, lists, scalars).
        filename: File name (without directory prefix).

    Returns:
        Absolute path of the written file.
    """
    results_dir = _ensure_results_dir()
    out_path = results_dir / filename

    def _default(obj: Any) -> Any:
        # Convert numpy scalars and similar objects to native Python types.
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=_default)

    logger.info("Results saved to %s", out_path)
    return out_path


def _extract_ragas_scores(result: Any) -> dict[str, float]:
    """Extract numeric metric scores from a RAGAS EvaluationResult.

    Handles both RAGAS v0.1.x (dict-like ``result.items()``) and newer
    versions that expose results via ``result.to_pandas()``.

    Args:
        result: Object returned by ``ragas.evaluate()``.

    Returns:
        Dict mapping metric name to float score. Empty dict on failure.
    """
    # RAGAS v0.1.x: EvaluationResult behaves like a dict.
    if hasattr(result, "items"):
        try:
            return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
        except Exception:
            pass
    # Newer RAGAS API: scores accessible via pandas DataFrame.
    if hasattr(result, "to_pandas"):
        try:
            df = result.to_pandas()
            means = df.select_dtypes(include="number").mean().to_dict()
            return {k: float(v) for k, v in means.items()}
        except Exception:
            pass
    return {}


def _format_context(docs: list[RetrievedDocument]) -> str:
    """Format retrieved docs the same way as RAGGenerator._format_context.

    Args:
        docs: List of RetrievedDocument instances from the retrieval layer.

    Returns:
        Multi-line string with one labelled block per document.
    """
    if not docs:
        return "No relevant documents found."
    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        doc_type = meta.get("doc_type", "unknown")
        ticker = meta.get("ticker", "N/A")
        date = (
            meta.get("published_at")
            or meta.get("fiscal_date")
            or meta.get("date")
            or "N/A"
        )
        header = f"[Source {idx}] {doc_type} | {ticker} | {date}"
        parts.append(f"{header}\n{doc.content}\n")
    return "\n".join(parts)


def _groq_generate(prompt: str, client: Groq) -> str:
    """Call Groq chat completion and return the assistant message text.

    Args:
        prompt: Full prompt string passed as the user message.
        client: Initialised Groq client.

    Returns:
        Raw text content of the model's reply, stripped of leading/trailing
        whitespace.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=LLM_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Benchmark 1 – RAGAS
# ---------------------------------------------------------------------------


def run_ragas_benchmark(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the RAGAS faithfulness / answer-relevancy / context-precision benchmark.

    For each question the function:
    1. Runs :class:`FinancialAgent` to retrieve relevant documents and live
       context.
    2. Builds a context string from the retrieved documents.
    3. Generates an answer via Groq using the context.
    4. Collects the quadruple (question, answer, contexts, ground_truth).

    After collecting all data it attempts to run RAGAS ``evaluate`` with the
    ``faithfulness``, ``answer_relevancy`` and ``context_precision`` metrics.
    If RAGAS (or the ``datasets`` library) is not installed the raw collected
    data is returned instead so it can still be saved.

    Args:
        questions: List of question dicts from the test-questions file.

    Returns:
        Dict with keys ``ragas_scores`` (if RAGAS succeeded) or ``raw_data``
        (fallback), plus ``questions``, ``answers``, ``contexts`` and
        ``ground_truths`` always present.
    """
    groq_client = Groq(api_key=GROQ_API_KEY)
    agent = FinancialAgent()

    collected_questions: list[str] = []
    collected_answers: list[str] = []
    collected_contexts: list[list[str]] = []
    collected_ground_truths: list[str] = []

    for item in questions:
        question: str = item["question"]
        ground_truth: str = item.get("ground_truth", "")

        try:
            docs, live_context, _plan = agent.run(question, top_k=10)

            context_str = _format_context(docs)
            if live_context:
                context_str = context_str + "\n\n" + live_context

            prompt = (
                "You are a professional financial analyst. "
                "Answer the following question using ONLY the provided context.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                "Answer concisely and factually."
            )
            answer = _groq_generate(prompt, groq_client)

            collected_questions.append(question)
            collected_answers.append(answer)
            collected_contexts.append([doc.content for doc in docs])
            collected_ground_truths.append(ground_truth)

            logger.info("RAGAS data collected for: %s", question[:60])

        except Exception as exc:  # noqa: BLE001
            logger.warning("Skipping question due to error: %s — %s", question[:60], exc)

        time.sleep(2)

    raw_data: dict[str, Any] = {
        "questions": collected_questions,
        "answers": collected_answers,
        "contexts": collected_contexts,
        "ground_truths": collected_ground_truths,
    }

    # Attempt RAGAS evaluation -----------------------------------------------
    try:
        from datasets import Dataset  # type: ignore[import]
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import (  # type: ignore[import]
            answer_relevancy,
            context_precision,
            faithfulness,
        )
    except ImportError as exc:
        logger.warning(
            "RAGAS / datasets library not available — returning raw data only. (%s)", exc
        )
        return {"status": "raw_only", **raw_data}

    try:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore[import]
        from langchain_groq import ChatGroq  # type: ignore[import]
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import]
        from ragas.llms import LangchainLLMWrapper  # type: ignore[import]

        ragas_llm = LangchainLLMWrapper(
            ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY)
        )
        ragas_embeddings = LangchainEmbeddingsWrapper(
            OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        )

        faithfulness.llm = ragas_llm
        answer_relevancy.llm = ragas_llm
        context_precision.llm = ragas_llm
        answer_relevancy.embeddings = ragas_embeddings

        dataset = Dataset.from_dict(
            {
                "question": collected_questions,
                "answer": collected_answers,
                "contexts": collected_contexts,
                "ground_truth": collected_ground_truths,
            }
        )

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        scores: dict[str, float] = _extract_ragas_scores(result)
        logger.info("RAGAS scores: %s", scores)

        return {
            "status": "ok",
            "ragas_scores": scores,
            **raw_data,
        }

    except Exception as exc:  # noqa: BLE001
        logger.error("RAGAS evaluation failed: %s — returning raw data.", exc)
        return {"status": "ragas_failed", "error": str(exc), **raw_data}


# ---------------------------------------------------------------------------
# Benchmark 2 – Retrieval comparison
# ---------------------------------------------------------------------------


def run_retrieval_benchmark(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare Dense, Hybrid, and Hybrid+Rerank retrieval on the news collection.

    For each question the function retrieves documents using three strategies
    and records summary statistics (average score, doc count, top-1 score).

    Args:
        questions: List of question dicts from the test-questions file.

    Returns:
        Dict with keys ``per_question`` (list of per-question dicts) and
        ``summary`` (aggregate averages per strategy).
    """
    dense_retriever = DenseRetriever(collection_name="news")
    hybrid_retriever = HybridRetriever(collection_name="news")
    reranker = Reranker()

    per_question: list[dict[str, Any]] = []

    for item in questions:
        question: str = item["question"]
        row: dict[str, Any] = {"question": question}

        # --- Dense only ---
        try:
            dense_docs = dense_retriever.retrieve(question, top_k=10)
            scores_dense = [doc.score for doc in dense_docs]
            row["dense"] = {
                "avg_score": float(sum(scores_dense) / len(scores_dense)) if scores_dense else 0.0,
                "docs_retrieved": len(dense_docs),
                "top1_score": float(scores_dense[0]) if scores_dense else 0.0,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Dense retrieval failed for '%s': %s", question[:50], exc)
            row["dense"] = {"avg_score": 0.0, "docs_retrieved": 0, "top1_score": 0.0, "error": str(exc)}

        # --- Hybrid ---
        try:
            hybrid_docs = hybrid_retriever.retrieve(question, final_top_k=10)
            scores_hybrid = [doc.score for doc in hybrid_docs]
            row["hybrid"] = {
                "avg_score": float(sum(scores_hybrid) / len(scores_hybrid)) if scores_hybrid else 0.0,
                "docs_retrieved": len(hybrid_docs),
                "top1_score": float(scores_hybrid[0]) if scores_hybrid else 0.0,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Hybrid retrieval failed for '%s': %s", question[:50], exc)
            row["hybrid"] = {"avg_score": 0.0, "docs_retrieved": 0, "top1_score": 0.0, "error": str(exc)}
            hybrid_docs = []

        # --- Hybrid + Rerank ---
        try:
            if hybrid_docs:
                reranked_docs = reranker.rerank(question, hybrid_docs, top_k=5)
            else:
                reranked_docs = []
            scores_rerank = [doc.score for doc in reranked_docs]
            row["hybrid_rerank"] = {
                "avg_score": float(sum(scores_rerank) / len(scores_rerank)) if scores_rerank else 0.0,
                "docs_retrieved": len(reranked_docs),
                "top1_score": float(scores_rerank[0]) if scores_rerank else 0.0,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Reranking failed for '%s': %s", question[:50], exc)
            row["hybrid_rerank"] = {"avg_score": 0.0, "docs_retrieved": 0, "top1_score": 0.0, "error": str(exc)}

        per_question.append(row)
        logger.info("Retrieval comparison done for: %s", question[:60])

    # Build summary (aggregate averages per strategy) -------------------------
    strategies = ("dense", "hybrid", "hybrid_rerank")
    summary: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        valid_rows = [r[strategy] for r in per_question if strategy in r and "error" not in r[strategy]]
        if valid_rows:
            summary[strategy] = {
                "avg_score": float(sum(r["avg_score"] for r in valid_rows) / len(valid_rows)),
                "avg_docs_retrieved": float(sum(r["docs_retrieved"] for r in valid_rows) / len(valid_rows)),
                "avg_top1_score": float(sum(r["top1_score"] for r in valid_rows) / len(valid_rows)),
            }
        else:
            summary[strategy] = {"avg_score": 0.0, "avg_docs_retrieved": 0.0, "avg_top1_score": 0.0}

    return {"per_question": per_question, "summary": summary}


# ---------------------------------------------------------------------------
# Benchmark 3 – RAG vs LLM-only ablation
# ---------------------------------------------------------------------------


def run_ablation_benchmark(questions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare RAG-augmented generation against a plain LLM-only baseline.

    For each question the function:
    1. Runs the full RAG pipeline (FinancialAgent + Groq generation with context).
    2. Runs Groq generation *without* any retrieved context.

    A simplified comparison record is returned for each question.

    Args:
        questions: List of question dicts from the test-questions file.

    Returns:
        Dict with keys ``per_question`` (list of per-question result dicts) and
        ``summary`` (aggregate statistics across all questions).
    """
    groq_client = Groq(api_key=GROQ_API_KEY)
    agent = FinancialAgent()

    per_question: list[dict[str, Any]] = []

    for item in questions:
        question: str = item["question"]
        row: dict[str, Any] = {"question": question}

        # --- RAG answer ---
        rag_answer = ""
        rag_confidence = "low"
        rag_sources_count = 0
        rag_verified_sources = 0
        try:
            docs, live_context, _plan = agent.run(question, top_k=10)

            context_str = _format_context(docs)
            if live_context:
                context_str = context_str + "\n\n" + live_context

            rag_prompt = (
                "You are a professional financial analyst. "
                "Answer the following question using ONLY the provided context. "
                "At the end of your answer state your confidence level as one of: "
                "high, medium, or low.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            rag_answer = _groq_generate(rag_prompt, groq_client)

            # Heuristic confidence extraction
            lower_answer = rag_answer.lower()
            if "confidence: high" in lower_answer or "high confidence" in lower_answer:
                rag_confidence = "high"
            elif "confidence: medium" in lower_answer or "medium confidence" in lower_answer:
                rag_confidence = "medium"
            else:
                rag_confidence = "low"

            rag_sources_count = len(docs)
            # Count docs that carry a specific ticker symbol (vs generic macro/news).
            rag_verified_sources = sum(
                1 for d in docs
                if d.metadata.get("ticker") not in (None, "", "N/A")
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("RAG generation failed for '%s': %s", question[:50], exc)
            rag_answer = f"[ERROR: {exc}]"

        # --- LLM-only answer ---
        llm_only_answer = ""
        try:
            llm_only_prompt = (
                f"You are a financial analyst. "
                f"Answer this question based on your knowledge: {question}"
            )
            llm_only_answer = _groq_generate(llm_only_prompt, groq_client)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM-only generation failed for '%s': %s", question[:50], exc)
            llm_only_answer = f"[ERROR: {exc}]"

        row = {
            "question": question,
            "rag_answer": rag_answer,
            "llm_only_answer": llm_only_answer,
            "rag_confidence": rag_confidence,
            "rag_sources_count": rag_sources_count,
            "rag_verified_sources": rag_verified_sources,
        }
        per_question.append(row)
        logger.info("Ablation done for: %s", question[:60])

        time.sleep(2)

    # Summary -----------------------------------------------------------------
    total = len(per_question)
    avg_sources = (
        sum(r["rag_sources_count"] for r in per_question) / total if total else 0.0
    )
    avg_verified = (
        sum(r["rag_verified_sources"] for r in per_question) / total if total else 0.0
    )
    avg_rag_answer_len = (
        sum(len(r["rag_answer"]) for r in per_question) / total if total else 0.0
    )
    avg_llm_answer_len = (
        sum(len(r["llm_only_answer"]) for r in per_question) / total if total else 0.0
    )
    confidence_dist: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in per_question:
        confidence_dist[r["rag_confidence"]] = confidence_dist.get(r["rag_confidence"], 0) + 1

    summary: dict[str, Any] = {
        "total_questions": total,
        "avg_rag_sources_count": float(avg_sources),
        "avg_rag_verified_sources": float(avg_verified),
        "avg_rag_answer_length_chars": float(avg_rag_answer_len),
        "avg_llm_only_answer_length_chars": float(avg_llm_answer_len),
        "rag_confidence_distribution": confidence_dist,
    }

    return {"per_question": per_question, "summary": summary}


# ---------------------------------------------------------------------------
# Human-readable summary printers
# ---------------------------------------------------------------------------


def _print_ragas_summary(results: dict[str, Any]) -> None:
    """Print a concise RAGAS benchmark summary to stdout.

    Args:
        results: Output dict from :func:`run_ragas_benchmark`.
    """
    print("\n" + "=" * 60)
    print("RAGAS BENCHMARK SUMMARY")
    print("=" * 60)
    status = results.get("status", "unknown")
    print(f"Status : {status}")
    n = len(results.get("questions", []))
    print(f"Questions evaluated : {n}")
    if "ragas_scores" in results:
        print("\nScores:")
        for metric, score in results["ragas_scores"].items():
            print(f"  {metric:<30} {score:.4f}")
    else:
        print("\nRAGAS scores not available (library missing or evaluation failed).")
    print("=" * 60)


def _print_retrieval_summary(results: dict[str, Any]) -> None:
    """Print a concise retrieval comparison summary to stdout.

    Args:
        results: Output dict from :func:`run_retrieval_benchmark`.
    """
    print("\n" + "=" * 60)
    print("RETRIEVAL BENCHMARK SUMMARY")
    print("=" * 60)
    summary = results.get("summary", {})
    header = f"{'Strategy':<20} {'Avg Score':>12} {'Avg Docs':>10} {'Avg Top-1':>12}"
    print(header)
    print("-" * 56)
    for strategy, stats in summary.items():
        print(
            f"{strategy:<20} "
            f"{stats.get('avg_score', 0.0):>12.4f} "
            f"{stats.get('avg_docs_retrieved', 0.0):>10.1f} "
            f"{stats.get('avg_top1_score', 0.0):>12.4f}"
        )
    print("=" * 60)


def _print_ablation_summary(results: dict[str, Any]) -> None:
    """Print a concise RAG vs LLM-only ablation summary to stdout.

    Args:
        results: Output dict from :func:`run_ablation_benchmark`.
    """
    print("\n" + "=" * 60)
    print("ABLATION BENCHMARK SUMMARY (RAG vs LLM-only)")
    print("=" * 60)
    s = results.get("summary", {})
    print(f"Total questions           : {s.get('total_questions', 0)}")
    print(f"Avg RAG sources / query   : {s.get('avg_rag_sources_count', 0.0):.1f}")
    print(f"Avg ticker-specific docs  : {s.get('avg_rag_verified_sources', 0.0):.1f}")
    print(f"Avg RAG answer length     : {s.get('avg_rag_answer_length_chars', 0.0):.0f} chars")
    print(f"Avg LLM-only answer len   : {s.get('avg_llm_only_answer_length_chars', 0.0):.0f} chars")
    dist = s.get("rag_confidence_distribution", {})
    print(f"RAG confidence (high/med/low): {dist.get('high',0)}/{dist.get('medium',0)}/{dist.get('low',0)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="BrightVest RAG Evaluation")
    parser.add_argument(
        "--benchmark",
        choices=["ragas", "retrieval", "ablation", "all"],
        default="all",
        help="Which benchmark to run.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Limit to first N questions (for quick testing).",
    )
    args = parser.parse_args()

    questions_data = _load_questions(limit=args.questions)
    logger.info(
        "Loaded %d test questions (limit=%s).", len(questions_data), args.questions
    )

    ts = _timestamp()

    run_ragas = args.benchmark in ("ragas", "all")
    run_retrieval = args.benchmark in ("retrieval", "all")
    run_ablation = args.benchmark in ("ablation", "all")

    if run_ragas:
        logger.info("Starting RAGAS benchmark …")
        ragas_results = run_ragas_benchmark(questions_data)
        path = _save_results(ragas_results, f"ragas_{ts}.json")
        _print_ragas_summary(ragas_results)
        print(f"Saved → {path}")

    if run_retrieval:
        logger.info("Starting retrieval comparison benchmark …")
        retrieval_results = run_retrieval_benchmark(questions_data)
        path = _save_results(retrieval_results, f"retrieval_{ts}.json")
        _print_retrieval_summary(retrieval_results)
        print(f"Saved → {path}")

    if run_ablation:
        logger.info("Starting ablation benchmark …")
        ablation_results = run_ablation_benchmark(questions_data)
        path = _save_results(ablation_results, f"ablation_{ts}.json")
        _print_ablation_summary(ablation_results)
        print(f"Saved → {path}")
