"""Ingestion pipeline orchestrator for the BrightVest Financial RAG system.

Coordinates the three indexers — :class:`NewsIndexer`,
:class:`FundamentalsIndexer`, and :class:`MacroIndexer` — into a single
sequential pipeline.  Each source is run independently so that a failure in
one does not abort the others.

Example (programmatic)::

    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    results = pipeline.run(sources=["news", "fundamentals"], force_reindex=False)
    print(results)  # {"news": 312, "fundamentals": 0, "macro": -1}

Example (CLI)::

    python -m src.ingestion.pipeline --sources news fundamentals
    python -m src.ingestion.pipeline --force-reindex
    python -m src.ingestion.pipeline --stats-only
"""

import logging
import time

from src.config import NEWS_FETCH_LIMIT
from src.ingestion.fundamentals_indexer import FundamentalsIndexer
from src.ingestion.macro_indexer import MacroIndexer
from src.ingestion.news_indexer import NewsIndexer

logger = logging.getLogger(__name__)

# Sentinel value written to a source key when indexing fails.
_SOURCE_FAILED: int = -1

# Valid source names exposed via the public API and the CLI.
_ALL_SOURCES: list[str] = ["news", "fundamentals", "macro"]


class IngestionPipeline:
    """Orchestrates the full BrightVest RAG ingestion pipeline.

    Initialises all three indexers on construction and exposes a :meth:`run`
    method that executes each source sequentially.  A per-source
    ``try/except`` block ensures that a failure in one indexer is logged and
    recorded as ``-1`` in the results dict rather than propagating to the
    caller.

    Attributes:
        _news_indexer: Indexes Supabase ``articles`` into ChromaDB ``"news"``.
        _fundamentals_indexer: Indexes ``fundamentals_serving`` into ``"earnings"``.
        _macro_indexer: Indexes ``macro_indicators`` into ``"macro"``.
    """

    def __init__(self) -> None:
        """Initialise all three indexers.

        Each indexer opens (or creates) its own ChromaDB collection and
        prepares a Supabase client.  Ollama does not need to be running at
        this point — it is only contacted when embeddings are computed.
        """
        logger.debug("Initialising NewsIndexer…")
        self._news_indexer = NewsIndexer()

        logger.debug("Initialising FundamentalsIndexer…")
        self._fundamentals_indexer = FundamentalsIndexer()

        logger.debug("Initialising MacroIndexer…")
        self._macro_indexer = MacroIndexer()

        logger.info("IngestionPipeline ready (sources: %s).", _ALL_SOURCES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        sources: list[str] | None = None,
        force_reindex: bool = False,
    ) -> dict[str, int]:
        """Run the ingestion pipeline for the requested sources.

        Sources are executed sequentially.  If a source raises an unhandled
        exception, the error is logged and ``-1`` is stored in the result
        dict for that source; remaining sources are still processed.

        Args:
            sources: List of source names to index.  Valid values are
                ``"news"``, ``"fundamentals"``, and ``"macro"``.  When
                ``None`` (the default) all three sources are run.
            force_reindex: When ``True``, each indexer drops and recreates
                its ChromaDB collection before indexing.  Defaults to
                ``False``.

        Returns:
            A dict mapping each requested source name to the number of newly
            indexed chunks, or ``-1`` when that source failed.

            Example::

                {"news": 1523, "fundamentals": 42, "macro": 0}
        """
        target: list[str] = _ALL_SOURCES if sources is None else sources

        # Validate source names early so the caller gets a clear error.
        unknown = [s for s in target if s not in _ALL_SOURCES]
        if unknown:
            raise ValueError(
                f"Unknown source(s): {unknown!r}. Valid choices: {_ALL_SOURCES!r}."
            )

        logger.info(
            "Pipeline starting — sources=%r, force_reindex=%s.",
            target,
            force_reindex,
        )
        pipeline_start = time.monotonic()

        results: dict[str, int] = {}

        for source in target:
            logger.info("=== [%s] Starting indexation ===", source.upper())
            source_start = time.monotonic()

            try:
                count = self._run_source(source, force_reindex=force_reindex)
                elapsed = time.monotonic() - source_start
                logger.info(
                    "=== [%s] Done — %d chunks indexed in %.1fs ===",
                    source.upper(),
                    count,
                    elapsed,
                )
                results[source] = count

            except Exception as exc:  # noqa: BLE001
                elapsed = time.monotonic() - source_start
                logger.error(
                    "=== [%s] FAILED after %.1fs: %s ===",
                    source.upper(),
                    elapsed,
                    exc,
                    exc_info=True,
                )
                results[source] = _SOURCE_FAILED

        total_elapsed = time.monotonic() - pipeline_start
        total_indexed = sum(v for v in results.values() if v != _SOURCE_FAILED)
        logger.info(
            "Pipeline complete in %.1fs — %d chunks indexed total. Results: %s",
            total_elapsed,
            total_indexed,
            results,
        )

        return results

    def get_stats(self) -> dict[str, dict]:
        """Return ChromaDB collection statistics for all three sources.

        Queries each indexer's :meth:`get_collection_stats` method and
        wraps the results in a top-level dict keyed by source name.

        Returns:
            A dict with keys ``"news"``, ``"fundamentals"``, and ``"macro"``,
            each mapping to the stats dict returned by the corresponding
            indexer.

            Example::

                {
                    "news":         {"collection": "news",     "count": 4823},
                    "fundamentals": {"collection": "earnings", "count": 312},
                    "macro":        {"collection": "macro",    "count": 350,
                                     "series_indexed": ["DGS10", "FEDFUNDS", ...]},
                }
        """
        stats: dict[str, dict] = {}

        for source, indexer in self._indexer_map().items():
            try:
                stats[source] = indexer.get_collection_stats()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Could not retrieve stats for source '%s': %s", source, exc
                )
                stats[source] = {"error": str(exc)}

        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_source(self, source: str, *, force_reindex: bool) -> int:
        """Dispatch a single source to its indexer method.

        Args:
            source: One of ``"news"``, ``"fundamentals"``, or ``"macro"``.
            force_reindex: Passed through to the underlying indexer.

        Returns:
            Number of newly indexed chunks.

        Raises:
            ValueError: When ``source`` is not a known source name.
            Exception: Any exception raised by the underlying indexer is
                re-raised so that :meth:`run` can catch and log it.
        """
        if source == "news":
            count = self._news_indexer.index_articles(
                limit=NEWS_FETCH_LIMIT,
                force_reindex=force_reindex,
            )
            # EU articles pre-date the NEWS_FETCH_LIMIT window — index them separately.
            eu_count = self._news_indexer.index_eu_articles()
            return count + eu_count
        if source == "fundamentals":
            return self._fundamentals_indexer.index_fundamentals(
                force_reindex=force_reindex,
            )
        if source == "macro":
            return self._macro_indexer.index_macro(
                force_reindex=force_reindex,
            )

        raise ValueError(f"Unknown source: {source!r}")

    def _indexer_map(self) -> dict[str, object]:
        """Return an ordered mapping of source name → indexer instance.

        Used by :meth:`get_stats` to iterate over all indexers without
        duplicating the source list.

        Returns:
            An ordered dict whose keys follow the canonical source order
            ``["news", "fundamentals", "macro"]``.
        """
        return {
            "news": self._news_indexer,
            "fundamentals": self._fundamentals_indexer,
            "macro": self._macro_indexer,
        }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="BrightVest RAG Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.ingestion.pipeline\n"
            "  python -m src.ingestion.pipeline --sources news fundamentals\n"
            "  python -m src.ingestion.pipeline --force-reindex\n"
            "  python -m src.ingestion.pipeline --stats-only\n"
        ),
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["news", "fundamentals", "macro", "all"],
        default=["all"],
        metavar="{news,fundamentals,macro,all}",
        help="Sources to index (default: all).",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Force full reindex — drops and rebuilds each ChromaDB collection.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Display collection stats without running ingestion.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.stats_only:
        # Lightweight stats path: open ChromaDB directly, no Ollama needed.
        import chromadb
        from src.config import CHROMA_DB_PATH
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        print("\n--- ChromaDB Collection Stats ---")
        for coll_name in ("news", "earnings", "macro"):
            try:
                coll = client.get_collection(coll_name)
                print(f"\n[{coll_name.upper()}]  count: {coll.count()}")
            except Exception:
                print(f"\n[{coll_name.upper()}]  not found")
        print()

    else:
        pipeline = IngestionPipeline()

        # Resolve "all" → None so that IngestionPipeline.run uses all sources.
        resolved_sources: list[str] | None = (
            None if "all" in args.sources else args.sources
        )

        results = pipeline.run(
            sources=resolved_sources,
            force_reindex=args.force_reindex,
        )

        print("\n--- Ingestion Summary ---")
        for source_name, chunk_count in results.items():
            if chunk_count == _SOURCE_FAILED:
                status = "FAILED"
            else:
                status = f"{chunk_count} chunks indexed"
            print(f"  {source_name:<16} {status}")

        failed_sources = [s for s, c in results.items() if c == _SOURCE_FAILED]
        if failed_sources:
            print(f"\nWARNING: {len(failed_sources)} source(s) failed: {failed_sources}")
        else:
            total = sum(results.values())
            print(f"\nTotal: {total} new chunks indexed across all sources.")
        print()
