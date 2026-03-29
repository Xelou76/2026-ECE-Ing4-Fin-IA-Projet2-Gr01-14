"""Hybrid retrieval via Reciprocal Rank Fusion (RRF) of dense and sparse results.

This module combines :class:`~src.retrieval.dense.DenseRetriever` (ChromaDB
vector search) and :class:`~src.retrieval.sparse.SparseRetriever` (BM25
lexical search) into a single :class:`HybridRetriever`.  The two ranked lists
are merged with the standard Reciprocal Rank Fusion algorithm:

    RRF_score(doc) = Σ_i  1 / (k + rank_i)

where ``rank_i`` is the 1-based position of the document in list *i* and ``k``
is a smoothing constant (default **60**, as recommended in the original RRF
paper by Cormack et al., 2009).  Documents absent from a list contribute 0 for
that list.  The fused list is sorted in descending RRF score order.

Dense and sparse calls are issued **in parallel** via a
:class:`~concurrent.futures.ThreadPoolExecutor` to minimise wall-clock latency.

Example::

    from src.retrieval.hybrid import HybridRetriever

    retriever = HybridRetriever(collection_name="news")
    docs = retriever.retrieve("NVDA earnings beat Q4 2025", final_top_k=10)
    for doc in docs:
        print(f"{doc.score:.4f}  {doc.content[:80]}")
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from src.config import DENSE_TOP_K, HYBRID_TOP_K
from src.retrieval.dense import DenseRetriever, RetrievedDocument
from src.retrieval.sparse import SparseRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Fuse dense vector search and BM25 lexical search via Reciprocal Rank Fusion.

    On construction, both a :class:`~src.retrieval.dense.DenseRetriever` and a
    :class:`~src.retrieval.sparse.SparseRetriever` are instantiated against the
    same ChromaDB *collection_name*.  At retrieval time the two ranked lists are
    combined with RRF (k=60) and the top ``final_top_k`` results are returned.

    Args:
        collection_name: Name of the existing ChromaDB collection to query.
            Must already have been created by the ingestion pipeline.

    Raises:
        ValueError: If the named collection does not exist in the ChromaDB store.
    """

    def __init__(self, collection_name: str = "news") -> None:
        self._collection_name = collection_name
        self._dense = DenseRetriever(collection_name=collection_name)
        self._sparse = SparseRetriever(collection_name=collection_name)

        logger.info(
            "HybridRetriever: initialised on collection '%s' "
            "(dense docs=%d, sparse corpus=%d).",
            collection_name,
            self._dense.get_collection_count(),
            self._sparse.get_corpus_size(),
        )

    # ------------------------------------------------------------------
    # RRF core
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievedDocument],
        sparse_results: list[RetrievedDocument],
        k: int = 60,
    ) -> list[RetrievedDocument]:
        """Merge two ranked lists into one via Reciprocal Rank Fusion.

        Each document is identified by its ``doc_id``.  For every list in which
        the document appears its RRF contribution is ``1 / (k + rank)`` where
        *rank* is 1-based.  Contributions from both lists are summed.  Documents
        appearing in only one list receive a contribution only from that list.

        The document content and metadata are taken from *dense_results* when
        the ``doc_id`` appears in both lists (dense search is considered slightly
        more reliable for text content); otherwise the sparse entry is used.

        Args:
            dense_results: Ranked list from dense retrieval (index 0 = best).
            sparse_results: Ranked list from sparse retrieval (index 0 = best).
            k: RRF smoothing constant.  The standard value of 60 (Cormack
                et al., 2009) is used by default.

        Returns:
            Merged list of :class:`~src.retrieval.dense.RetrievedDocument`
            objects sorted by **descending** RRF score.  The ``score`` field
            holds the computed RRF value (typically in (0, ~0.033] range per
            list).
        """
        # Build lookup: doc_id → document, preferring dense content.
        doc_registry: dict[str, RetrievedDocument] = {}

        # Register sparse first so dense overwrites on collision.
        for doc in sparse_results:
            doc_registry[doc.doc_id] = doc
        for doc in dense_results:
            doc_registry[doc.doc_id] = doc

        # Accumulate RRF scores.
        rrf_scores: dict[str, float] = {}

        for rank_0, doc in enumerate(dense_results):
            rank_1based = rank_0 + 1
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank_1based)

        for rank_0, doc in enumerate(sparse_results):
            rank_1based = rank_0 + 1
            rrf_scores[doc.doc_id] = rrf_scores.get(doc.doc_id, 0.0) + 1.0 / (k + rank_1based)

        # Build result list with the fused score.
        fused: list[RetrievedDocument] = []
        for doc_id, rrf_score in rrf_scores.items():
            base = doc_registry[doc_id]
            fused.append(
                RetrievedDocument(
                    content=base.content,
                    metadata=base.metadata,
                    score=rrf_score,
                    doc_id=doc_id,
                )
            )

        fused.sort(key=lambda d: d.score, reverse=True)

        logger.debug(
            "_reciprocal_rank_fusion: dense=%d, sparse=%d → fused=%d unique docs.",
            len(dense_results),
            len(sparse_results),
            len(fused),
        )
        return fused

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        dense_top_k: int = DENSE_TOP_K,
        sparse_top_k: int = DENSE_TOP_K,
        final_top_k: int = HYBRID_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve and fuse documents from dense and sparse retrievers.

        Dense and sparse calls are issued in parallel via a thread pool to
        minimise latency (the dense call involves a network round-trip to Ollama
        whereas the sparse call is CPU-only, so they overlap well).

        Args:
            query: Natural-language query string.
            dense_top_k: Number of candidates to fetch from the dense retriever.
            sparse_top_k: Number of candidates to fetch from the sparse retriever.
            final_top_k: Maximum number of documents to return after RRF fusion.
            filters: Optional filter dict forwarded to both retrievers.
                - Dense: ChromaDB ``where`` clause syntax.
                - Sparse: flat ``{key: value}`` equality filters.
                Pass ``None`` to disable filtering.

        Returns:
            Ranked list of :class:`~src.retrieval.dense.RetrievedDocument`
            objects (at most *final_top_k*), sorted by descending RRF score.

        Raises:
            RuntimeError: If the dense embedding step fails (Ollama unavailable).
        """
        dense_results: list[RetrievedDocument] = []
        sparse_results: list[RetrievedDocument] = []

        def _dense() -> list[RetrievedDocument]:
            return self._dense.retrieve(query=query, top_k=dense_top_k, filters=filters)

        def _sparse() -> list[RetrievedDocument]:
            return self._sparse.retrieve(query=query, top_k=sparse_top_k, filters=filters)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(_dense)
            future_sparse = executor.submit(_sparse)
            for future in as_completed([future_dense, future_sparse]):
                try:
                    result_list = future.result()
                except Exception as exc:
                    logger.warning("Retriever failed, falling back to empty list: %s", type(exc).__name__)
                    result_list = []
                if future is future_dense:
                    dense_results = result_list
                else:
                    sparse_results = result_list

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        result = fused[:final_top_k]

        logger.debug(
            "HybridRetriever.retrieve('%s'): dense=%d, sparse=%d → "
            "fused=%d, returned=%d.",
            query[:60],
            len(dense_results),
            len(sparse_results),
            len(fused),
            len(result),
        )
        return result

    def retrieve_with_metadata_filter(
        self,
        query: str,
        ticker: str | None = None,
        doc_type: str | None = None,
        date_from: str | None = None,
        dense_top_k: int = DENSE_TOP_K,
        sparse_top_k: int = DENSE_TOP_K,
        final_top_k: int = HYBRID_TOP_K,
    ) -> list[RetrievedDocument]:
        """Retrieve with typed metadata filters, fused via RRF.

        Calls :meth:`~src.retrieval.dense.DenseRetriever.retrieve_with_metadata_filter`
        (which supports *date_from* via the ``published_at`` field) and
        :meth:`~src.retrieval.sparse.SparseRetriever.retrieve_with_metadata_filter`
        (which does **not** support date filtering) in parallel, then merges the
        results with RRF.

        Args:
            query: Natural-language query string.
            ticker: If provided, restrict both retrievers to this ticker symbol.
            doc_type: If provided, restrict both retrievers to this document
                type (``"news"``, ``"earnings"``, ``"macro"``).
            date_from: If provided, restrict the **dense** retriever to
                documents with ``published_at >= date_from``.  The sparse
                retriever does not support date filtering and ignores this
                argument.
            dense_top_k: Candidates fetched from the dense retriever.
            sparse_top_k: Candidates fetched from the sparse retriever.
            final_top_k: Maximum number of documents to return after fusion.

        Returns:
            Ranked list of :class:`~src.retrieval.dense.RetrievedDocument`
            objects (at most *final_top_k*), sorted by descending RRF score.

        Raises:
            RuntimeError: If the dense embedding step fails (Ollama unavailable).
        """
        dense_results: list[RetrievedDocument] = []
        sparse_results: list[RetrievedDocument] = []

        def _dense() -> list[RetrievedDocument]:
            return self._dense.retrieve_with_metadata_filter(
                query=query,
                ticker=ticker,
                doc_type=doc_type,
                date_from=date_from,
                top_k=dense_top_k,
            )

        def _sparse() -> list[RetrievedDocument]:
            return self._sparse.retrieve_with_metadata_filter(
                query=query,
                ticker=ticker,
                doc_type=doc_type,
                top_k=sparse_top_k,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(_dense)
            future_sparse = executor.submit(_sparse)
            for future in as_completed([future_dense, future_sparse]):
                try:
                    result_list = future.result()
                except Exception as exc:
                    logger.warning("Retriever failed, falling back to empty list: %s", type(exc).__name__)
                    result_list = []
                if future is future_dense:
                    dense_results = result_list
                else:
                    sparse_results = result_list

        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        result = fused[:final_top_k]

        logger.debug(
            "HybridRetriever.retrieve_with_metadata_filter('%s', ticker=%s, "
            "doc_type=%s, date_from=%s): dense=%d, sparse=%d → "
            "fused=%d, returned=%d.",
            query[:60],
            ticker,
            doc_type,
            date_from,
            len(dense_results),
            len(sparse_results),
            len(fused),
            len(result),
        )
        return result

    def get_retrievers_info(self) -> dict[str, Any]:
        """Return diagnostic information about the underlying retrievers.

        Returns:
            Dictionary with the following keys:

            - ``"dense_collection"`` (:class:`str`): ChromaDB collection name
              used by the dense retriever.
            - ``"sparse_corpus_size"`` (:class:`int`): Number of documents
              currently indexed in the BM25 corpus.
        """
        return {
            "dense_collection": self._collection_name,
            "sparse_corpus_size": self._sparse.get_corpus_size(),
        }
