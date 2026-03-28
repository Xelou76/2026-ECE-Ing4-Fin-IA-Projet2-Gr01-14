"""Sparse (BM25) lexical retrieval over a ChromaDB collection.

This module provides :class:`SparseRetriever`, which loads all documents from a
ChromaDB collection into memory, builds a BM25Okapi index via *rank-bm25*, and
exposes a ``retrieve`` API consistent with :class:`~src.retrieval.dense.DenseRetriever`.

BM25 complements dense vector search particularly well for financial text, where
exact term matching matters: ticker symbols (NVDA, AAPL), accounting metrics
(EBITDA, EPS, P/E), and numeric values (e.g. "Q3 revenue $2.1B") are often
missed by semantic embeddings but scored reliably by BM25.

Example::

    from src.retrieval.sparse import SparseRetriever

    retriever = SparseRetriever(collection_name="news")
    docs = retriever.retrieve("NVDA EPS beat Q4 2025", top_k=10)
    for doc in docs:
        print(f"{doc.score:.4f}  {doc.content[:80]}")
"""

from __future__ import annotations

import logging
import re
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from src.config import CHROMA_DB_PATH, DENSE_TOP_K
from src.retrieval.dense import RetrievedDocument

logger = logging.getLogger(__name__)

# Punctuation characters stripped during tokenisation.
# We intentionally preserve digits, '$', '%', '.' (used in numeric tokens)
# and '-' / '_' (used in tickers, series IDs, field names).
_STRIP_PUNCT = re.compile(r"[^\w\s$%.\-]", re.ASCII)


class SparseRetriever:
    """BM25-based lexical retriever backed by a ChromaDB persistent store.

    On construction the retriever loads the entire collection into memory and
    builds a :class:`~rank_bm25.BM25Okapi` index.  Subsequent calls to
    :meth:`retrieve` are CPU-only and do **not** hit the database again unless
    :meth:`rebuild_index` is called explicitly.

    Args:
        collection_name: Name of the ChromaDB collection to index.  The
            collection must already exist (created by the ingestion pipeline).

    Raises:
        ValueError: If the named collection does not exist in the ChromaDB store.
    """

    def __init__(self, collection_name: str = "news") -> None:
        self._collection_name = collection_name

        self._chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            self._collection = self._chroma_client.get_collection(collection_name)
        except Exception as exc:
            raise ValueError(
                f"ChromaDB collection '{collection_name}' does not exist at "
                f"'{CHROMA_DB_PATH}'. Run the ingestion pipeline first."
            ) from exc

        # In-memory state populated by _build_index()
        self._documents: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

        self._build_index()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[str]:
        """Tokenise *text* for BM25 indexing or query scoring.

        The strategy is deliberately lightweight:

        1. Strip most punctuation while preserving characters meaningful in
           financial text: ``$``, ``%``, ``.``, ``-``, ``_``.
        2. Lowercase the result.
        3. Split on whitespace.
        4. Drop empty tokens.

        Digits are **not** removed — numeric values are semantically important
        in financial documents (e.g. "2.1B revenue", "P/E 35.2").

        Args:
            text: Raw document content or query string.

        Returns:
            List of lowercase token strings.
        """
        cleaned = _STRIP_PUNCT.sub(" ", text)
        tokens = [t for t in cleaned.lower().split() if t]
        return tokens

    def _build_index(self) -> None:
        """Load all documents from ChromaDB and (re)build the BM25 index.

        Fetches every document in the collection in a single batch request,
        tokenises each one, and constructs a fresh :class:`~rank_bm25.BM25Okapi`
        instance.  If the collection is empty the index is left as ``None`` and
        :attr:`_documents` is set to an empty list.
        """
        count = self._collection.count()
        if count == 0:
            logger.warning(
                "SparseRetriever: collection '%s' is empty — BM25 index not built.",
                self._collection_name,
            )
            self._documents = []
            self._bm25 = None
            return

        logger.info(
            "SparseRetriever: loading %d documents from '%s' for BM25 indexing…",
            count,
            self._collection_name,
        )
        result = self._collection.get(include=["documents", "metadatas"])

        ids: list[str] = result.get("ids") or []
        texts: list[str] = result.get("documents") or []
        metadatas: list[dict[str, Any]] = result.get("metadatas") or []

        self._documents = [
            {"id": doc_id, "content": content, "metadata": meta or {}}
            for doc_id, content, meta in zip(ids, texts, metadatas)
        ]

        tokenized_corpus = [self._tokenize(doc["content"]) for doc in self._documents]
        self._bm25 = BM25Okapi(tokenized_corpus)

        logger.info(
            "SparseRetriever: BM25 index built over %d documents.",
            len(self._documents),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve the *top_k* most lexically relevant documents for *query*.

        Scoring is performed over the entire in-memory BM25 index.  If
        *filters* is provided, documents whose metadata does not satisfy all
        filter conditions are removed **after** scoring before the top-k
        selection.  This keeps the implementation simple and avoids maintaining
        per-subset indexes.

        Filter dict format::

            # Single field equality
            {"ticker": "AAPL"}

            # Multiple fields (all conditions must match — implicit AND)
            {"ticker": "NVDA", "doc_type": "news"}

        Args:
            query: Natural-language or keyword query string.
            top_k: Maximum number of documents to return.
            filters: Optional flat dict of ``{metadata_key: expected_value}``
                pairs.  All conditions are combined with logical AND.

        Returns:
            List of :class:`~src.retrieval.dense.RetrievedDocument` objects
            sorted by descending BM25 score (highest relevance first).
            Returns an empty list when the collection is empty or no documents
            match the filters.
        """
        if self._bm25 is None or not self._documents:
            logger.warning(
                "SparseRetriever.retrieve: index is empty — returning no results."
            )
            return []

        query_tokens = self._tokenize(query)
        scores: list[float] = self._bm25.get_scores(query_tokens).tolist()

        # Build unsorted candidate list
        candidates: list[RetrievedDocument] = [
            RetrievedDocument(
                content=doc["content"],
                metadata=doc["metadata"],
                score=score,
                doc_id=doc["id"],
            )
            for doc, score in zip(self._documents, scores)
        ]

        # Apply metadata filters (post-scoring, flat AND logic)
        if filters:
            def _matches(doc: RetrievedDocument) -> bool:
                for key, expected in filters.items():
                    actual = doc.metadata.get(key)
                    if actual == expected:
                        continue  # exact match
                    # Ticker special-case: European tickers are stored with exchange
                    # suffixes (ASML.AS, LVMH.PA, SAP.DE…).  Match if the stored
                    # ticker starts with the expected base ticker followed by ".".
                    if key == "ticker" and actual is not None and isinstance(expected, str):
                        if actual.startswith(expected + "."):
                            continue
                    return False
                return True

            candidates = [d for d in candidates if _matches(d)]

        # Sort by descending BM25 score and truncate
        candidates.sort(key=lambda d: d.score, reverse=True)
        result = candidates[:top_k]

        logger.debug(
            "SparseRetriever.retrieve('%s'): %d candidates after filter, returning %d "
            "(top score=%.4f).",
            query[:60],
            len(candidates),
            len(result),
            result[0].score if result else float("nan"),
        )
        return result

    def retrieve_with_metadata_filter(
        self,
        query: str,
        ticker: str | None = None,
        doc_type: str | None = None,
        top_k: int = DENSE_TOP_K,
    ) -> list[RetrievedDocument]:
        """Retrieve documents with typed convenience filters.

        Thin wrapper around :meth:`retrieve` that builds the *filters* dict
        from explicit keyword arguments instead of requiring callers to
        construct it manually.

        Args:
            query: Natural-language or keyword query string.
            ticker: If provided, restrict results to this ticker symbol
                (matched against the ``"ticker"`` metadata field), e.g.
                ``"AAPL"``.
            doc_type: If provided, restrict results to this document type
                (``"news"``, ``"earnings"``, ``"macro"``).
            top_k: Maximum number of documents to return.

        Returns:
            List of :class:`~src.retrieval.dense.RetrievedDocument` objects
            sorted by descending BM25 score.  Returns an empty list when the
            collection is empty or no documents match.
        """
        filters: dict[str, Any] = {}
        if ticker is not None:
            filters["ticker"] = ticker
        if doc_type is not None:
            filters["doc_type"] = doc_type

        return self.retrieve(query=query, top_k=top_k, filters=filters or None)

    def rebuild_index(self) -> None:
        """Force a full reload from ChromaDB and rebuild the BM25 index.

        Call this method after running the ingestion pipeline to pick up
        newly indexed documents without restarting the process.
        """
        logger.info(
            "SparseRetriever.rebuild_index: rebuilding index for collection '%s'.",
            self._collection_name,
        )
        self._build_index()

    def get_corpus_size(self) -> int:
        """Return the number of documents currently held in the BM25 index.

        Returns:
            Integer count of indexed documents.  Zero when the collection was
            empty at construction time or after the last :meth:`rebuild_index`
            call.
        """
        return len(self._documents)
