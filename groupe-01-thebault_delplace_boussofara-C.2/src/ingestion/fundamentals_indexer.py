"""Fundamentals indexer for the BrightVest Financial RAG system.

Fetches financial fundamentals from Supabase via ``SupabaseClient``, converts
each row to a natural-language text chunk, computes embeddings with Ollama
(nomic-embed-text), and stores the result in a persistent ChromaDB collection
named ``"earnings"``.

Deduplication is performed via a deterministic document ID built from
``symbol``, ``period_type``, and ``fiscal_date_ending``.  Rows already present
in the collection are skipped unless ``force_reindex=True``.

Many numeric columns can be NULL depending on the upstream data source (FMP,
Alpha Vantage, legacy).  All None values are handled gracefully — the chunk
content simply omits those fields rather than raising a formatting error.

Example:
    from src.ingestion.fundamentals_indexer import FundamentalsIndexer

    indexer = FundamentalsIndexer()
    count = indexer.index_fundamentals(period_type="annual")
    print(f"Indexed {count} new chunks.")
    print(indexer.get_collection_stats())
"""

import logging
from datetime import date, datetime

import chromadb
from langchain_community.embeddings import OllamaEmbeddings

from src.config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from src.data.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "earnings"
_EMBED_BATCH_SIZE = 50  # fundamentals chunks are longer than news → smaller batches
# Minimum extra characters beyond the header line to consider a chunk meaningful.
_MIN_CONTENT_EXTRA = 20


class FundamentalsIndexer:
    """Indexes Supabase fundamentals data into ChromaDB using Ollama embeddings.

    Each row in ``fundamentals_serving`` becomes one chunk whose content is a
    natural-language summary of the available financial metrics.  Fields that
    are NULL in the database are simply omitted from the text.

    The ChromaDB document ID is deterministic:
    ``f"fundamentals_{symbol}_{period_type}_{fiscal_date_ending}"``

    Attributes:
        _chroma_client: Persistent ChromaDB client.
        _embeddings: LangChain OllamaEmbeddings wrapper.
        _collection: ChromaDB collection object for ``"earnings"``.
        _supabase: Read-only Supabase client.
    """

    def __init__(self) -> None:
        """Initialise ChromaDB, Ollama embeddings, and Supabase client.

        The ChromaDB collection ``"earnings"`` is created if it does not yet
        exist; otherwise the existing collection is opened.

        Note:
            Ollama must be running on ``OLLAMA_BASE_URL`` before any embedding
            calls are made (i.e. before ``index_fundamentals`` is called).
            The constructor itself does **not** contact Ollama.
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

    def build_chunk(self, row: dict) -> dict | None:
        """Convert a raw Supabase fundamentals row into a RAG chunk.

        Builds a natural-language text summary of all non-NULL financial
        metrics.  Fields that are ``None`` in the database are silently
        omitted from the content string.

        Args:
            row: A dict returned by ``SupabaseClient.fetch_fundamentals``,
                containing at minimum ``symbol``.  All numeric columns may be
                ``None``.

        Returns:
            A dict with keys ``"id"``, ``"content"``, and ``"metadata"``, or
            ``None`` when ``symbol`` is ``None``/empty, or when the generated
            content body is shorter than ``_MIN_CONTENT_EXTRA`` characters
            (meaning no useful metrics are available).
        """
        symbol: str | None = row.get("symbol")
        if not symbol:
            return None

        period_type: str = row.get("period_type") or "unknown"
        fiscal_date_ending_raw = row.get("fiscal_date_ending")
        fiscal_date: str = self._normalise_date(fiscal_date_ending_raw)

        # --- Extract numeric fields (all may be None) ---
        revenue: float | None = row.get("revenue")
        net_income: float | None = row.get("net_income")
        eps: float | None = row.get("eps")
        gross_margin: float | None = row.get("gross_margin")
        operating_margin: float | None = row.get("operating_margin")
        net_margin: float | None = row.get("net_margin")
        revenue_growth_yoy: float | None = row.get("revenue_growth_yoy")
        pe_ratio_ttm: float | None = row.get("pe_ratio_ttm")
        ev_to_ebitda_ttm: float | None = row.get("ev_to_ebitda_ttm")
        debt_to_equity: float | None = row.get("debt_to_equity")
        free_cash_flow: float | None = row.get("free_cash_flow")
        market_cap: float | None = row.get("market_cap")
        beta: float | None = row.get("beta")

        # --- Build header (always present) ---
        header = f"{symbol} {period_type} results ending {fiscal_date}: "

        # --- Build body conditionally —- omit NULL fields ---
        body_parts: list[str] = []

        if revenue is not None:
            body_parts.append(f"Revenue: ${revenue / 1e6:,.1f}M.")
        if net_income is not None:
            body_parts.append(f"Net income: ${net_income / 1e6:,.1f}M.")
        if eps is not None:
            body_parts.append(f"EPS: ${eps:.2f}.")

        margin_parts: list[str] = []
        if gross_margin is not None:
            margin_parts.append(f"gross {gross_margin:.1%}")
        if operating_margin is not None:
            margin_parts.append(f"operating {operating_margin:.1%}")
        if net_margin is not None:
            margin_parts.append(f"net {net_margin:.1%}")
        if margin_parts:
            body_parts.append("Margins: " + ", ".join(margin_parts) + ".")

        if revenue_growth_yoy is not None:
            body_parts.append(f"Growth: revenue YoY {revenue_growth_yoy:.1%}.")

        valuation_parts: list[str] = []
        if pe_ratio_ttm is not None:
            valuation_parts.append(f"P/E {pe_ratio_ttm:.1f}")
        if ev_to_ebitda_ttm is not None:
            valuation_parts.append(f"EV/EBITDA {ev_to_ebitda_ttm:.1f}")
        if valuation_parts:
            body_parts.append("Valuation: " + ", ".join(valuation_parts) + ".")

        balance_parts: list[str] = []
        if debt_to_equity is not None:
            balance_parts.append(f"debt/equity {debt_to_equity:.2f}")
        if free_cash_flow is not None:
            balance_parts.append(f"FCF: ${free_cash_flow / 1e6:,.1f}M")
        if balance_parts:
            body_parts.append("Balance sheet: " + ", ".join(balance_parts) + ".")

        if market_cap is not None:
            body_parts.append(f"Market cap: ${market_cap / 1e9:,.2f}B.")
        if beta is not None:
            body_parts.append(f"Beta: {beta:.2f}.")

        body = " ".join(body_parts)

        # Reject rows with too little useful information.
        if len(body) < _MIN_CONTENT_EXTRA:
            return None

        content = header + body

        doc_id = self._make_doc_id(row)

        metadata: dict[str, str] = {
            "doc_type": "earnings",
            "ticker": symbol,
            "period": period_type,
            "fiscal_date": fiscal_date,
        }

        return {"id": doc_id, "content": content, "metadata": metadata}

    def _make_doc_id(self, row: dict) -> str:
        """Build a deterministic ChromaDB document ID for a fundamentals row.

        Args:
            row: A dict returned by ``SupabaseClient.fetch_fundamentals``.
                Must contain ``symbol``.  ``period_type`` and
                ``fiscal_date_ending`` default to ``"unknown"`` when absent.

        Returns:
            A string of the form
            ``"fundamentals_{symbol}_{period_type}_{fiscal_date_ending}"``.
        """
        symbol: str = row.get("symbol") or "unknown"
        period_type: str = row.get("period_type") or "unknown"
        fiscal_date: str = self._normalise_date(row.get("fiscal_date_ending"))
        return f"fundamentals_{symbol}_{period_type}_{fiscal_date}"

    def index_fundamentals(
        self,
        symbol: str | None = None,
        period_type: str | None = None,
        force_reindex: bool = False,
    ) -> int:
        """Fetch fundamentals from Supabase and index them into ChromaDB.

        Already-indexed rows are skipped via a single ``get`` call to check
        existing IDs.  When ``force_reindex=True`` the entire collection is
        dropped and rebuilt from scratch.

        Args:
            symbol: Optional ticker symbol to restrict indexing to a single
                company (e.g. ``"NVDA"``).  When ``None``, all tickers are
                processed.
            period_type: Optional period filter — ``"annual"`` or
                ``"quarterly"``.  When ``None``, both period types are indexed.
            force_reindex: When ``True``, the ``"earnings"`` ChromaDB
                collection is deleted and recreated before indexing.
                Defaults to ``False``.

        Returns:
            The number of newly indexed chunks.  Returns ``0`` when all chunks
            were already present and ``force_reindex`` is ``False``.
        """
        if force_reindex:
            logger.info(
                "force_reindex=True — deleting existing 'earnings' collection."
            )
            self._chroma_client.delete_collection(_COLLECTION_NAME)
            self._collection = self._chroma_client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        logger.info(
            "Fetching fundamentals from Supabase (symbol=%r, period_type=%r)…",
            symbol,
            period_type,
        )
        rows = self._supabase.fetch_fundamentals(
            symbol=symbol,
            period_type=period_type,
        )
        logger.info("Fetched %d fundamentals rows.", len(rows))

        # Build chunks, dropping rows with no symbol or insufficient content.
        chunks: list[dict] = []
        skipped = 0
        for row in rows:
            chunk = self.build_chunk(row)
            if chunk is not None:
                chunks.append(chunk)
            else:
                skipped += 1

        if skipped:
            logger.warning(
                "Skipped %d rows with missing symbol or insufficient data.",
                skipped,
            )

        if not chunks:
            logger.info("No valid chunks to index.")
            return 0

        # Deduplicate by doc_id within the batch.
        # Rows with null fiscal_date_ending all produce the same "unknown" date
        # component, so multiple rows for the same ticker+period would collide.
        seen_batch_ids: set[str] = set()
        unique_chunks: list[dict] = []
        for chunk in chunks:
            if chunk["id"] not in seen_batch_ids:
                seen_batch_ids.add(chunk["id"])
                unique_chunks.append(chunk)
        dup_count = len(chunks) - len(unique_chunks)
        if dup_count:
            logger.warning(
                "Dropped %d duplicate doc_ids in batch (likely null fiscal_date_ending).",
                dup_count,
            )
        chunks = unique_chunks

        # Dedup: find which document IDs are already stored in ChromaDB.
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

        # Embed and store in batches of _EMBED_BATCH_SIZE.
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

            processed = batch_start + len(batch)
            if processed % 200 < _EMBED_BATCH_SIZE or processed == len(new_chunks):
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

    def get_collection_stats(self) -> dict[str, object]:
        """Return basic statistics about the ChromaDB ``"earnings"`` collection.

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
                include=[],  # only IDs — no documents or embeddings
            )
            return set(result["ids"])
        except Exception as exc:
            logger.warning(
                "Could not fetch existing IDs from ChromaDB (%s). "
                "Treating all as new — duplicates may be added.",
                exc,
            )
            return set()

    @staticmethod
    def _normalise_date(value: object) -> str:
        """Normalise a date value to an ISO-format string.

        Handles ``datetime.date`` objects, ISO strings, and ``None``.

        Args:
            value: The raw ``fiscal_date_ending`` value from Supabase.  May be
                a ``datetime.date`` instance, an ISO date string, or ``None``.

        Returns:
            An ISO date string (e.g. ``"2024-12-31"``), or ``"unknown"`` when
            the value is ``None`` or cannot be converted.
        """
        if value is None:
            return "unknown"
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        try:
            return str(value)
        except Exception:
            return "unknown"
