"""Macro indicator indexer for the BrightVest Financial RAG system.

Fetches macro-economic time series from Supabase via ``SupabaseClient``,
converts each observation to a text chunk, computes embeddings with Ollama
(nomic-embed-text), and stores the result in a persistent ChromaDB collection
named ``"macro"``.

Only the ``_VALUES_PER_SERIES`` most recent observations per series are indexed
to avoid embedding tens of thousands of legacy VIX daily values while still
keeping the context window rich for recent macro analysis.

Deduplication is handled through deterministic document IDs of the form
``"macro_{series_id}_{date}"``.  Observations already present in the collection
are skipped when ``force_reindex=False``.

Example:
    from src.ingestion.macro_indexer import MacroIndexer

    indexer = MacroIndexer()
    count = indexer.index_macro(values_per_series=50)
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

_COLLECTION_NAME = "macro"
_EMBED_BATCH_SIZE = 100
# Number of most-recent observations to index per series.
# Prevents embedding all 9,108 legacy VIX rows on every run.
_VALUES_PER_SERIES = 50

# All known series — used when the caller passes series_ids=None.
_DEFAULT_SERIES: list[str] = [
    "VIX",
    "FEDFUNDS",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "ICSA",
    "VIXCLS",
]


class MacroIndexer:
    """Indexes Supabase macro indicators into ChromaDB using Ollama embeddings.

    Each observation becomes one chunk whose content follows the template::

        "Macro indicator {name} ({series_id}), category: {category}. Value: {value} as of {date}."

    The ChromaDB document ID is ``f"macro_{series_id}_{date}"``, which is
    fully deterministic and enables safe re-runs without duplicates.

    Attributes:
        _chroma_client: Persistent ChromaDB client.
        _embeddings: LangChain OllamaEmbeddings wrapper.
        _collection: ChromaDB collection object for ``"macro"``.
        _supabase: Read-only Supabase client.
    """

    def __init__(self) -> None:
        """Initialise ChromaDB, Ollama embeddings, and Supabase client.

        The ChromaDB collection ``"macro"`` is created if it does not yet
        exist; otherwise the existing collection is opened.

        Note:
            Ollama must be running on ``OLLAMA_BASE_URL`` before any
            embedding calls are made (i.e. before ``index_macro`` is
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

    def build_chunk(self, row: dict) -> dict | None:
        """Convert a raw Supabase macro indicator row into a RAG chunk.

        Args:
            row: A dict returned by ``SupabaseClient.fetch_macro_indicators``,
                containing at minimum ``series_id`` and ``value``.  Other
                fields (``name``, ``category``, ``date``) may be ``None``.

        Returns:
            A dict with keys ``"id"``, ``"content"``, and ``"metadata"``, or
            ``None`` when ``value`` or ``series_id`` is ``None`` (the row
            cannot produce a meaningful chunk without these two fields).
        """
        series_id: str | None = row.get("series_id")
        value = row.get("value")

        # A chunk without a series identifier or a numeric value is meaningless.
        if series_id is None or value is None:
            return None

        name: str = row.get("name") or ""
        category: str = row.get("category") or ""

        # date may arrive as a datetime.date object, a datetime.datetime, or
        # an ISO string — normalise to a plain ISO date string.
        raw_date = row.get("date")
        if isinstance(raw_date, (date, datetime)):
            date_str: str = raw_date.isoformat()
        else:
            date_str = str(raw_date) if raw_date is not None else ""

        content: str = (
            f"Macro indicator {name} ({series_id}), category: {category}. "
            f"Value: {value} as of {date_str}."
        )

        # ChromaDB does not accept None values in metadata — replace with "".
        metadata: dict[str, str] = {
            "doc_type": "macro",
            "series_id": series_id,
            "category": category,
            "date": date_str,
        }

        return {
            "id": self._make_doc_id(row),
            "content": content,
            "metadata": metadata,
        }

    def _make_doc_id(self, row: dict) -> str:
        """Build a deterministic ChromaDB document ID for a macro observation.

        The ID combines the series identifier and the observation date so that
        re-runs produce the same ID and ChromaDB can detect duplicates.

        Args:
            row: A dict containing at least ``series_id`` and ``date``.

        Returns:
            A string of the form ``"macro_{series_id}_{date}"``.
        """
        series_id: str = row.get("series_id") or "UNKNOWN"

        raw_date = row.get("date")
        if isinstance(raw_date, (date, datetime)):
            date_str: str = raw_date.isoformat()
        else:
            date_str = str(raw_date) if raw_date is not None else "nodate"

        return f"macro_{series_id}_{date_str}"

    def index_macro(
        self,
        series_ids: list[str] | None = None,
        values_per_series: int = _VALUES_PER_SERIES,
        force_reindex: bool = False,
    ) -> int:
        """Fetch macro observations from Supabase and index them into ChromaDB.

        Iterates over each series individually so that the ``values_per_series``
        cap is applied **per series** rather than globally.  This prevents a
        single large series (e.g. VIX with 9,108 rows) from crowding out
        smaller but equally important series like T10Y2Y or ICSA.

        Already-indexed observations are skipped via deterministic document IDs
        unless ``force_reindex=True``, in which case the entire collection is
        dropped and rebuilt.

        Args:
            series_ids: List of series identifiers to index.  When ``None``
                all series in ``_DEFAULT_SERIES`` are processed.
            values_per_series: Maximum number of most-recent observations to
                index per series.  Defaults to ``_VALUES_PER_SERIES`` (50).
            force_reindex: When ``True``, drop and recreate the ChromaDB
                ``"macro"`` collection before indexing.  Defaults to
                ``False``.

        Returns:
            The total number of newly indexed chunks across all series.
        """
        target_series: list[str] = series_ids if series_ids is not None else _DEFAULT_SERIES

        if force_reindex:
            logger.info("force_reindex=True — deleting existing 'macro' collection.")
            self._chroma_client.delete_collection(_COLLECTION_NAME)
            self._collection = self._chroma_client.get_or_create_collection(
                name=_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Collect all chunks across series, then deduplicate by doc ID.
        all_chunks: list[dict] = []
        for series_id in target_series:
            logger.info(
                "Fetching up to %d observations for series '%s'…",
                values_per_series,
                series_id,
            )
            rows = self._supabase.fetch_macro_indicators(
                series_ids=[series_id],
                limit=values_per_series,
            )
            logger.info(
                "Fetched %d rows for series '%s'.",
                len(rows),
                series_id,
            )

            for row in rows:
                chunk = self.build_chunk(row)
                if chunk is not None:
                    all_chunks.append(chunk)

        if not all_chunks:
            logger.info("No valid chunks to index.")
            return 0

        # Deduplicate by doc ID (in case the same observation appears in
        # multiple series fetches, which should not happen in practice but
        # is a safe guard).
        seen_ids: set[str] = set()
        unique_chunks: list[dict] = []
        for chunk in all_chunks:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                unique_chunks.append(chunk)

        skipped_dup = len(all_chunks) - len(unique_chunks)
        if skipped_dup:
            logger.warning(
                "Dropped %d duplicate chunks (same series_id + date).",
                skipped_dup,
            )

        # When not force-reindexing, find which IDs are already in ChromaDB
        # so we only embed and store truly new documents.
        if not force_reindex:
            existing_ids: set[str] = self._get_existing_ids(
                [c["id"] for c in unique_chunks]
            )
            new_chunks = [c for c in unique_chunks if c["id"] not in existing_ids]
            logger.info(
                "%d chunks already indexed, %d new to add.",
                len(existing_ids),
                len(new_chunks),
            )
        else:
            new_chunks = unique_chunks

        if not new_chunks:
            logger.info("Nothing new to index.")
            return 0

        # Embed and store in batches to avoid Ollama timeouts.
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
        """Return statistics about the ChromaDB ``"macro"`` collection.

        Queries the collection metadata to enumerate which series are
        currently indexed.

        Returns:
            A dict with keys:

            * ``"collection"`` (str): Always ``"macro"``.
            * ``"count"`` (int): Total number of documents in the collection.
            * ``"series_indexed"`` (list[str]): Sorted list of unique
              ``series_id`` values present in the collection.
        """
        count: int = self._collection.count()

        series_indexed: list[str] = []
        if count > 0:
            try:
                # Retrieve all stored metadata to extract unique series_id values.
                result = self._collection.get(include=["metadatas"])
                metadatas: list[dict] = result.get("metadatas") or []
                series_set: set[str] = set()
                for meta in metadatas:
                    sid = meta.get("series_id")
                    if sid:
                        series_set.add(sid)
                series_indexed = sorted(series_set)
            except Exception as exc:
                logger.warning(
                    "Could not retrieve metadata for stats (%s). "
                    "series_indexed will be empty.",
                    exc,
                )

        return {
            "collection": _COLLECTION_NAME,
            "count": count,
            "series_indexed": series_indexed,
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
