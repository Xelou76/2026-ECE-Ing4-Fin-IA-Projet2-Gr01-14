"""News article indexer for the BrightVest Financial RAG system.

Fetches articles from Supabase via ``SupabaseClient``, converts each article
to a text chunk, computes embeddings with Ollama (nomic-embed-text), and
stores the result in a persistent ChromaDB collection named ``"news"``.

Deduplication is handled through Supabase's ``dedup_hash`` field, which is
used as the ChromaDB document ID.  Articles already present in the collection
are skipped automatically by ChromaDB's upsert semantics (when
``force_reindex=False``).

Example:
    from src.ingestion.news_indexer import NewsIndexer

    indexer = NewsIndexer()
    count = indexer.index_articles(limit=5000)
    print(f"Indexed {count} new chunks.")
    print(indexer.get_collection_stats())
"""

import hashlib
import logging
from datetime import datetime

import chromadb
from langchain_community.embeddings import OllamaEmbeddings

from src.config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    NEWS_FETCH_LIMIT,
    OLLAMA_BASE_URL,
)
from src.data.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "news"
_EMBED_BATCH_SIZE = 100  # max articles per Ollama embedding call


class NewsIndexer:
    """Indexes Supabase news articles into ChromaDB using Ollama embeddings.

    Each article becomes one chunk whose content is
    ``"{headline}. {summary}"`` and whose ChromaDB ID is the article's
    ``dedup_hash`` (or a deterministic fallback when the hash is absent).

    Attributes:
        _chroma_client: Persistent ChromaDB client.
        _embeddings: LangChain OllamaEmbeddings wrapper.
        _collection: ChromaDB collection object for ``"news"``.
        _supabase: Read-only Supabase client.
    """

    def __init__(self) -> None:
        """Initialise ChromaDB, Ollama embeddings, and Supabase client.

        The ChromaDB collection ``"news"`` is created if it does not yet
        exist; otherwise the existing collection is opened.

        Note:
            Ollama must be running on ``OLLAMA_BASE_URL`` before any
            embedding calls are made (i.e. before ``index_articles`` is
            called).  The constructor itself does **not** contact Ollama.
        """
        self._chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

        self._embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL,
        )

        # get_or_create_collection is idempotent — safe to call every time.
        self._collection = self._chroma_client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        self._supabase = SupabaseClient()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_chunk(self, article: dict) -> dict | None:
        """Convert a raw Supabase article dict into a RAG chunk.

        Args:
            article: A dict returned by ``SupabaseClient.fetch_all_articles_paginated``,
                containing at minimum ``headline``.  All other fields may be
                ``None``.

        Returns:
            A dict with keys ``"id"``, ``"content"``, and ``"metadata"``, or
            ``None`` when the headline is empty/``None`` (the article cannot
            produce a meaningful chunk without a headline).
        """
        headline: str | None = article.get("headline")
        if not headline:
            return None

        summary: str = article.get("summary") or ""
        content: str = f"{headline}. {summary}".strip()

        ticker: str = article.get("ticker") or "UNKNOWN"
        sector: str = article.get("sector") or ""
        source: str = article.get("source") or ""
        category: str = article.get("category") or ""
        sentiment_final: str = article.get("sentiment_final") or ""

        # published_at may arrive as a datetime object or an ISO string.
        raw_published_at = article.get("published_at")
        if isinstance(raw_published_at, datetime):
            published_at: str = raw_published_at.isoformat()
        else:
            published_at = str(raw_published_at) if raw_published_at is not None else ""

        # Build the ChromaDB document ID.
        dedup_hash: str | None = article.get("dedup_hash")
        if dedup_hash:
            doc_id = dedup_hash
        else:
            # Deterministic fallback: combine ticker, date, and a hash of
            # the content so that re-runs produce the same ID.
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            doc_id = f"news_{ticker}_{published_at}_{content_hash}"

        # ChromaDB does not accept None values in metadata — replace with "".
        metadata: dict[str, str] = {
            "doc_type": "news",
            "ticker": ticker,
            "sector": sector,
            "source": source,
            "published_at": published_at,
            "sentiment_final": sentiment_final,
            "category": category,
        }

        return {"id": doc_id, "content": content, "metadata": metadata}

    def index_articles(
        self,
        limit: int = NEWS_FETCH_LIMIT,
        force_reindex: bool = False,
    ) -> int:
        """Fetch articles from Supabase and index them into ChromaDB.

        Already-indexed articles are skipped via ChromaDB's ``upsert``
        semantics (the ``dedup_hash`` acts as a stable document ID).  When
        ``force_reindex`` is ``True`` the entire collection is deleted and
        rebuilt from scratch.

        Args:
            limit: Maximum number of articles to fetch from Supabase.
            force_reindex: When ``True``, drop and recreate the ChromaDB
                ``"news"`` collection before indexing.  Defaults to
                ``False``.

        Returns:
            The number of newly indexed chunks (0 when all chunks were
            already present and ``force_reindex`` is ``False``).
        """
        if force_reindex:
            logger.info("force_reindex=True — deleting existing 'news' collection.")
            self._chroma_client.delete_collection(_COLLECTION_NAME)
            self._collection = self._chroma_client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        logger.info("Fetching up to %d articles from Supabase…", limit)
        articles = self._supabase.fetch_all_articles_paginated(limit=limit)
        logger.info("Fetched %d articles.", len(articles))

        # Build chunks, dropping articles with no headline.
        chunks: list[dict] = []
        for article in articles:
            chunk = self.build_chunk(article)
            if chunk is not None:
                chunks.append(chunk)

        skipped_headline = len(articles) - len(chunks)
        if skipped_headline:
            logger.warning(
                "Skipped %d articles with empty/missing headline.",
                skipped_headline,
            )

        if not chunks:
            logger.info("No valid chunks to index.")
            return 0

        # When not force-reindexing, find which IDs are already in ChromaDB
        # so we only embed and store truly new documents.
        if not force_reindex:
            existing_ids: set[str] = self._get_existing_ids(
                [c["id"] for c in chunks]
            )
            new_chunks = [c for c in chunks if c["id"] not in existing_ids]
            logger.info(
                "%d chunks already indexed, %d new to add.",
                len(existing_ids),
                len(new_chunks),
            )
        else:
            new_chunks = chunks

        if not new_chunks:
            logger.info("Nothing new to index.")
            return 0

        # Index in batches to avoid Ollama timeouts.
        total_indexed = 0
        for batch_start in range(0, len(new_chunks), _EMBED_BATCH_SIZE):
            batch = new_chunks[batch_start : batch_start + _EMBED_BATCH_SIZE]
            ids = [c["id"] for c in batch]
            documents = [c["content"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            try:
                embeddings = self._embeddings.embed_documents(documents)
            except Exception as exc:
                logger.error(
                    "Embedding failed for batch starting at index %d: %s",
                    batch_start,
                    exc,
                )
                continue

            try:
                self._collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas,
                    embeddings=embeddings,
                )
            except Exception as exc:
                logger.error(
                    "ChromaDB add failed for batch starting at index %d: %s",
                    batch_start,
                    exc,
                )
                continue

            total_indexed += len(batch)

            # Progress log every ~500 articles.
            processed = batch_start + len(batch)
            if processed % 500 < _EMBED_BATCH_SIZE or processed == len(new_chunks):
                logger.info(
                    "Progress: %d / %d chunks indexed.",
                    total_indexed,
                    len(new_chunks),
                )

        logger.info(
            "Indexing complete. %d new chunks added to collection '%s'.",
            total_indexed,
            _COLLECTION_NAME,
        )
        return total_indexed

    def index_eu_articles(self) -> int:
        """Fetch and index all EU-region articles from Supabase.

        EU articles are stored with exchange-suffix tickers (e.g. ``SAP.DE``,
        ``LVMH.PA``, ``ASML.AS``).  Because they are older than the main 5,000
        US articles, they are outside the ``NEWS_FETCH_LIMIT`` window of the
        standard :meth:`index_articles` run.  This method fetches them directly
        via a ``region='EU'`` filter so they are always available regardless of
        the main fetch limit.

        Already-indexed articles (same ``dedup_hash``) are skipped
        automatically.

        Returns:
            The number of newly indexed EU chunks.
        """
        logger.info("Fetching all EU-region articles from Supabase…")
        try:
            resp = self._supabase._client.table("articles").select(
                "ticker, headline, summary, source, published_at, "
                "sector, sentiment_final, category, dedup_hash"
            ).eq("region", "EU").order("published_at", desc=True).execute()
            articles = resp.data or []
        except Exception as exc:
            logger.error("Failed to fetch EU articles: %s", exc)
            return 0

        logger.info("Fetched %d EU articles.", len(articles))

        chunks: list[dict] = []
        for article in articles:
            chunk = self.build_chunk(article)
            if chunk is not None:
                chunks.append(chunk)

        if not chunks:
            logger.info("No valid EU chunks to index.")
            return 0

        existing_ids = self._get_existing_ids([c["id"] for c in chunks])
        new_chunks = [c for c in chunks if c["id"] not in existing_ids]
        logger.info("%d EU chunks already indexed, %d new.", len(existing_ids), len(new_chunks))

        if not new_chunks:
            return 0

        total_indexed = 0
        for batch_start in range(0, len(new_chunks), _EMBED_BATCH_SIZE):
            batch = new_chunks[batch_start: batch_start + _EMBED_BATCH_SIZE]
            try:
                embeddings = self._embeddings.embed_documents([c["content"] for c in batch])
                self._collection.add(
                    ids=[c["id"] for c in batch],
                    documents=[c["content"] for c in batch],
                    metadatas=[c["metadata"] for c in batch],
                    embeddings=embeddings,
                )
                total_indexed += len(batch)
            except Exception as exc:
                logger.error("EU batch at index %d failed: %s", batch_start, exc)

        logger.info("EU indexing complete: %d new chunks added.", total_indexed)
        return total_indexed

    def get_collection_stats(self) -> dict[str, object]:
        """Return basic statistics about the ChromaDB ``"news"`` collection.

        Returns:
            A dict with keys ``"collection"`` (str) and ``"count"`` (int).
        """
        return {
            "collection": _COLLECTION_NAME,
            "count": self._collection.count(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_existing_ids(self, candidate_ids: list[str]) -> set[str]:
        """Return the subset of ``candidate_ids`` already present in ChromaDB.

        Queries ChromaDB in a single ``get`` call to avoid N+1 overhead.

        Args:
            candidate_ids: List of document IDs to check.

        Returns:
            A set of IDs that are already stored in the collection.
        """
        if not candidate_ids:
            return set()
        try:
            result = self._collection.get(
                ids=candidate_ids,
                include=[],  # only IDs, no documents or embeddings
            )
            return set(result["ids"])
        except Exception as exc:
            logger.warning(
                "Could not fetch existing IDs from ChromaDB (%s). "
                "Treating all as new — duplicates may be added.",
                exc,
            )
            return set()
