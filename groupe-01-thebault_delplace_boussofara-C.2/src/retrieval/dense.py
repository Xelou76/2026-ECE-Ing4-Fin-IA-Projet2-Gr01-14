"""Dense vector retrieval using ChromaDB and Ollama embeddings.

This module provides the :class:`DenseRetriever` class which embeds a natural-language
query via Ollama (nomic-embed-text) and performs an approximate-nearest-neighbour
search against a persistent ChromaDB collection.

Example::

    from src.retrieval.dense import DenseRetriever

    retriever = DenseRetriever(collection_name="news")
    docs = retriever.retrieve("AAPL earnings beat Q4 2025", top_k=10)
    for doc in docs:
        print(doc.score, doc.content[:80])
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
import requests
from pydantic import BaseModel

from src.config import (
    CHROMA_DB_PATH,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class RetrievedDocument(BaseModel):
    """A single document returned by the dense retriever.

    Attributes:
        content: Raw text content of the chunk.
        metadata: Arbitrary key/value metadata stored alongside the chunk.
        score: ChromaDB L2 distance (lower = more similar).
        doc_id: Unique document identifier inside the ChromaDB collection.
    """

    content: str
    metadata: dict[str, Any]
    score: float
    doc_id: str


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class DenseRetriever:
    """Retrieve semantically relevant documents from a ChromaDB collection.

    The retriever embeds the incoming query with Ollama's *nomic-embed-text*
    model (running locally) and then issues a nearest-neighbour query against
    the specified ChromaDB persistent collection.

    Args:
        collection_name: Name of the existing ChromaDB collection to query.
            The collection must already exist (created during ingestion).

    Raises:
        RuntimeError: If Ollama is not reachable at startup *or* at query time.
        ValueError: If the collection does not exist in the ChromaDB store.
    """

    def __init__(self, collection_name: str = "news") -> None:
        self._collection_name = collection_name

        # -- ChromaDB ----------------------------------------------------------
        self._chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            self._collection = self._chroma_client.get_collection(collection_name)
        except Exception as exc:
            raise ValueError(
                f"ChromaDB collection '{collection_name}' does not exist at "
                f"'{CHROMA_DB_PATH}'. Run the ingestion pipeline first."
            ) from exc

        logger.info(
            "DenseRetriever: collection '%s' loaded (%d documents).",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_query(self, query: str) -> list[float]:
        """Embed *query* using the Ollama REST API.

        Args:
            query: Natural-language query string.

        Returns:
            Dense embedding vector as a list of floats.

        Raises:
            RuntimeError: If Ollama is unreachable or returns an error.
        """
        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/embeddings"
        try:
            response = requests.post(
                url,
                json={"model": EMBEDDING_MODEL, "prompt": query},
                timeout=30,
            )
            response.raise_for_status()
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at '{OLLAMA_BASE_URL}'. "
                "Make sure the Ollama server is running (`ollama serve`)."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise RuntimeError(
                f"Ollama embedding request timed out after 30 s (model='{EMBEDDING_MODEL}')."
            ) from exc
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(
                f"Ollama returned HTTP {response.status_code}: {response.text}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        payload = response.json()
        embedding: list[float] = payload.get("embedding", [])
        if not embedding:
            raise RuntimeError(
                f"Ollama returned an empty embedding for model '{EMBEDDING_MODEL}'. "
                "Check that the model is pulled: `ollama pull nomic-embed-text`."
            )
        return embedding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        """Retrieve the *top_k* most relevant documents for *query*.

        Args:
            query: Natural-language query string.
            top_k: Maximum number of documents to return.
            filters: Optional ChromaDB ``where`` filter dict.  Use the
                ChromaDB filter syntax, e.g.
                ``{"ticker": {"$eq": "AAPL"}}`` or
                ``{"$and": [{"ticker": {"$eq": "AAPL"}}, {"doc_type": {"$eq": "news"}}]}``.

        Returns:
            Ranked list of :class:`RetrievedDocument` objects, ordered by
            ascending L2 distance (most similar first).  Returns an empty
            list when the collection is empty.

        Raises:
            RuntimeError: If the embedding step fails (Ollama unavailable).
        """
        count = self._collection.count()
        if count == 0:
            logger.warning("Collection '%s' is empty — returning no results.", self._collection_name)
            return []

        query_embedding = self._embed_query(query)

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if filters:
            query_kwargs["where"] = filters

        results = self._collection.query(**query_kwargs)

        documents: list[RetrievedDocument] = []
        ids = results.get("ids", [[]])[0]
        texts = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, text, meta, dist in zip(ids, texts, metadatas, distances):
            documents.append(
                RetrievedDocument(
                    content=text,
                    metadata=meta or {},
                    score=float(dist),
                    doc_id=doc_id,
                )
            )

        logger.debug(
            "retrieve('%s'): returned %d/%d documents (top score=%.4f).",
            query[:60],
            len(documents),
            top_k,
            documents[0].score if documents else float("nan"),
        )
        return documents

    def retrieve_with_metadata_filter(
        self,
        query: str,
        ticker: str | None = None,
        doc_type: str | None = None,
        date_from: str | None = None,
        top_k: int = DENSE_TOP_K,
    ) -> list[RetrievedDocument]:
        """Retrieve documents with typed metadata filters.

        Convenience wrapper around :meth:`retrieve` that builds the ChromaDB
        ``where`` clause from typed keyword arguments instead of requiring
        callers to hand-craft the filter dict.

        Args:
            query: Natural-language query string.
            ticker: If provided, restrict results to this ticker symbol,
                e.g. ``"AAPL"``.
            doc_type: If provided, restrict to this document type
                (``"news"``, ``"earnings"``, ``"macro"``).
            date_from: If provided, restrict to documents whose
                ``published_at`` metadata field is *greater than or equal to*
                this ISO-8601 date string, e.g. ``"2025-01-01"``.
            top_k: Maximum number of documents to return.

        Returns:
            Ranked list of :class:`RetrievedDocument` objects.  Returns an
            empty list when the collection is empty or no documents match.

        Raises:
            RuntimeError: If the embedding step fails (Ollama unavailable).
        """
        clauses: list[dict[str, Any]] = []

        if ticker is not None:
            # European tickers are stored with exchange suffixes (ASML.AS, LVMH.PA, SAP.DE…).
            # When the user types just "ASML", we must also match "ASML.AS", "ASML.MI", etc.
            # Use $in with the base ticker + all known EU exchange suffix variants.
            _EU_SUFFIXES = [
                ".L", ".PA", ".DE", ".AS", ".MI", ".SW", ".BR", ".MC",
                ".HE", ".OL", ".ST", ".CO", ".LS", ".WA", ".BU", ".AT",
            ]
            ticker_variants = [ticker] + [ticker + s for s in _EU_SUFFIXES]
            clauses.append({"ticker": {"$in": ticker_variants}})
        if doc_type is not None:
            clauses.append({"doc_type": {"$eq": doc_type}})
        # date_from filtering is handled at Python level by the caller
        # (ChromaDB 1.x only supports $gte for int/float, not strings)

        where: dict[str, Any] | None
        if len(clauses) == 0:
            where = None
        elif len(clauses) == 1:
            where = clauses[0]
        else:
            where = {"$and": clauses}

        return self.retrieve(query=query, top_k=top_k, filters=where)

    def get_collection_count(self) -> int:
        """Return the total number of documents in the ChromaDB collection.

        Returns:
            Integer document count.
        """
        return self._collection.count()
