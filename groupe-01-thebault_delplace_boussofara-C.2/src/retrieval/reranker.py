"""Cross-encoder reranker for the BrightVest Financial RAG pipeline.

Takes the top candidates produced by the hybrid retriever and scores each
(query, document) pair with a cross-encoder model.  The cross-encoder reads
both texts jointly, yielding a more accurate relevance signal than the
bi-encoder scores used during first-stage retrieval.

Model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
  - ~22 MB download on first use, then cached in ``~/.cache/huggingface/``
  - Runs well on Apple Silicon (M-series) via sentence-transformers MPS
    auto-detection — no explicit device override needed.

Example::

    from src.retrieval.reranker import Reranker
    from src.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever()
    candidates = retriever.retrieve("Is NVDA overbought?", top_k=30)

    reranker = Reranker()
    top_docs = reranker.rerank("Is NVDA overbought?", candidates, top_k=5)
    for doc in top_docs:
        print(f"[{doc.score:.3f}] {doc.content[:80]}")
"""

from __future__ import annotations

import logging
from typing import Any

from sentence_transformers import CrossEncoder

from src.config import RERANK_TOP_K
from src.retrieval.dense import RetrievedDocument

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------

CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------


class Reranker:
    """Cross-encoder reranker that re-scores a candidate document list.

    Uses ``cross-encoder/ms-marco-MiniLM-L-6-v2`` from sentence-transformers
    to jointly encode (query, document) pairs and produce a relevance logit
    for each pair.  Documents are then sorted by this logit in descending order
    and the top-*k* are returned.

    The cross-encoder logits are *not* bounded — they can be negative for
    irrelevant documents and positive for highly relevant ones.  The raw logit
    is stored directly in :attr:`RetrievedDocument.score` so that downstream
    consumers can interpret the magnitude as a confidence signal.

    Attributes:
        _model: Loaded :class:`~sentence_transformers.CrossEncoder` instance.
    """

    def __init__(self) -> None:
        """Load the cross-encoder model.

        On first call this downloads ~22 MB from the Hugging Face Hub into
        ``~/.cache/huggingface/``.  Subsequent instantiations load from the
        local cache and are fast.

        sentence-transformers automatically selects the best available device
        (MPS on Apple Silicon, CUDA on Nvidia, CPU otherwise) — no manual
        ``device`` override is required.
        """
        logger.info(
            "Reranker: loading cross-encoder model '%s' …",
            CROSS_ENCODER_MODEL,
        )
        self._model: CrossEncoder = CrossEncoder(CROSS_ENCODER_MODEL)
        logger.info(
            "Reranker: model '%s' loaded successfully.",
            CROSS_ENCODER_MODEL,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = RERANK_TOP_K,
    ) -> list[RetrievedDocument]:
        """Rerank *documents* with respect to *query* using the cross-encoder.

        Each document is paired with the query and scored jointly.  Documents
        are then sorted by cross-encoder score (descending) and the top-*k*
        are returned as new :class:`~src.retrieval.dense.RetrievedDocument`
        instances whose ``score`` field holds the cross-encoder logit.

        Args:
            query: Natural-language query string used during retrieval.
            documents: Candidate documents from the hybrid retriever.
                Typically the top-30 results from
                :class:`~src.retrieval.hybrid.HybridRetriever`.
            top_k: Number of documents to return after reranking.  If
                ``top_k`` exceeds the number of input documents all documents
                are returned, sorted by cross-encoder score.

        Returns:
            List of at most *top_k* :class:`~src.retrieval.dense.RetrievedDocument`
            objects ordered by descending cross-encoder score.  Returns an
            empty list when *documents* is empty.
        """
        if not documents:
            logger.debug("Reranker.rerank: received empty document list — returning [].")
            return []

        # Build (query, content) pairs for the cross-encoder.
        pairs: list[tuple[str, str]] = [
            (query, doc.content) for doc in documents
        ]

        logger.debug(
            "Reranker.rerank: scoring %d pairs for query '%s'…",
            len(pairs),
            query[:60],
        )

        # predict() returns a numpy array of logits, one per pair.
        scores: Any = self._model.predict(pairs)

        # Attach cross-encoder scores and sort descending.
        scored: list[tuple[float, RetrievedDocument]] = [
            (float(score), doc)
            for score, doc in zip(scores, documents)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Rebuild RetrievedDocument objects with updated scores.
        # NOTE: ms-marco-MiniLM-L-6-v2 was trained on web passages; financial text
        # scores systematically negative (-6 to -9) even when highly relevant.
        # We rely only on relative ranking (descending), not absolute score thresholds.
        effective_k = min(top_k, len(scored))
        reranked: list[RetrievedDocument] = [
            RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                score=ce_score,
                doc_id=doc.doc_id,
            )
            for ce_score, doc in scored[:effective_k]
        ]

        logger.debug(
            "Reranker.rerank: returning %d/%d documents "
            "(top score=%.4f, bottom score=%.4f).",
            len(reranked),
            len(documents),
            reranked[0].score if reranked else float("nan"),
            reranked[-1].score if reranked else float("nan"),
        )

        return reranked

    def get_model_name(self) -> str:
        """Return the identifier of the loaded cross-encoder model.

        Returns:
            The Hugging Face model ID string, e.g.
            ``"cross-encoder/ms-marco-MiniLM-L-6-v2"``.
        """
        return CROSS_ENCODER_MODEL
