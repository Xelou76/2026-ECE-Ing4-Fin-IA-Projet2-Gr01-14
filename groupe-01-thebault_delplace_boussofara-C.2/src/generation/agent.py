"""Agentic query planner for the BrightVest Financial RAG pipeline.

This module implements :class:`FinancialAgent`, which orchestrates retrieval
across multiple ChromaDB collections and live-enrichment sources.  Given a raw
user query the agent:

1. Detects ticker symbols and classifies the query type.
2. Builds a :class:`RetrievalPlan` that selects the appropriate ChromaDB
   collections and determines whether live Supabase enrichment is needed.
3. Executes the plan: runs :class:`~src.retrieval.hybrid.HybridRetriever` for
   each collection, deduplicates the union, reranks with a cross-encoder, and
   optionally appends live price/technical context.

Example::

    from src.generation.agent import FinancialAgent

    agent = FinancialAgent()
    docs, live_ctx, plan = agent.run("Compare NVDA vs AAPL earnings growth")

    print(plan.query_type)       # "comparison"
    print(plan.tickers)          # ["NVDA", "AAPL"]
    print(plan.collections)      # ["earnings", "news"]
    for doc in docs:
        print(f"[{doc.score:.3f}] {doc.content[:80]}")
    if live_ctx:
        print(live_ctx)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from src.config import DENSE_TOP_K, HYBRID_TOP_K, RERANK_TOP_K
from src.retrieval.context_builder import ContextBuilder
from src.retrieval.dense import RetrievedDocument
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)


def _dedup_queries(queries: list[str]) -> list[str]:
    """Deduplicate a list of query strings while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            result.append(q)
    return result


# ---------------------------------------------------------------------------
# Query-type → data-source routing table
# ---------------------------------------------------------------------------

QUERY_TYPE_MAP: dict[str, list[str]] = {
    "news": ["news"],
    "earnings": ["earnings"],
    "comparison": ["earnings", "news"],
    "price": ["prices_daily", "technical_indicators"],
    "momentum": ["technical_indicators", "news"],
    "macro": ["macro", "news"],
    "portfolio": ["positions", "news", "earnings"],
    "recommendation": ["earnings", "news", "technical_indicators"],
}

# ---------------------------------------------------------------------------
# ChromaDB collection registry
# ---------------------------------------------------------------------------

#: Sources that map to a ChromaDB collection indexed by the ingestion pipeline.
CHROMA_COLLECTION_MAP: dict[str, str] = {
    "news": "news",
    "earnings": "earnings",
    "macro": "macro",
}

#: Sources that are served live from Supabase via :class:`~src.retrieval.context_builder.ContextBuilder`.
LIVE_ENRICHMENT_TYPES: frozenset[str] = frozenset(
    {"prices_daily", "technical_indicators", "positions"}
)

#: Maps context_builder query types to agent QUERY_TYPE_MAP keys.
_CB_TO_AGENT_TYPE: dict[str, str] = {
    "price": "price",
    "technical": "momentum",
    "momentum": "momentum",
    "fundamental": "earnings",
    "news": "news",
    "general": "news",
}

# ---------------------------------------------------------------------------
# Keyword mapping: query_type → descriptive keyword used in sub-query strings
# ---------------------------------------------------------------------------

_QUERY_TYPE_KEYWORDS: dict[str, str] = {
    "news": "news sentiment",
    "earnings": "fundamentals earnings revenue growth",
    "comparison": "fundamentals earnings",
    "price": "price performance",
    "momentum": "momentum trend technical",
    "macro": "macro economic indicators",
    "portfolio": "portfolio analysis",
    "recommendation": "investment analysis recommendation",
}


# ---------------------------------------------------------------------------
# Retrieval plan
# ---------------------------------------------------------------------------


class RetrievalPlan(BaseModel):
    """Structured description of what the agent will retrieve.

    Attributes:
        query_type: High-level category of the user query (e.g. ``"comparison"``).
        sub_queries: Expanded list of queries to run against each collection.
            Always ends with the original query to ensure global coverage.
        tickers: Ticker symbols detected in the original query.
        collections: ChromaDB collection names to query (subset of
            :data:`CHROMA_COLLECTION_MAP` keys).
        use_live_enrichment: Whether :class:`~src.retrieval.context_builder.ContextBuilder`
            should be called to add live price/technical context.
        enrichment_query_type: Label forwarded to
            :meth:`~src.retrieval.context_builder.ContextBuilder.enrich`.
            Meaningful only when *use_live_enrichment* is ``True``.
    """

    query_type: str
    sub_queries: list[str] = Field(default_factory=list)
    tickers: list[str] = Field(default_factory=list)
    collections: list[str] = Field(default_factory=list)
    use_live_enrichment: bool = False
    enrichment_query_type: str = "general"


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class FinancialAgent:
    """Orchestrate retrieval across ChromaDB collections and live Supabase data.

    The agent follows a three-step pipeline for every incoming query:

    1. **Plan** (:meth:`plan`): detect tickers, classify query type, select
       data sources, and expand to sub-queries for multi-ticker comparisons.
    2. **Retrieve** (:meth:`retrieve`): fan out to each ChromaDB collection
       using :class:`~src.retrieval.hybrid.HybridRetriever`, deduplicate by
       ``doc_id``, rerank with :class:`~src.retrieval.reranker.Reranker`, and
       optionally attach live enrichment via
       :class:`~src.retrieval.context_builder.ContextBuilder`.
    3. **Run** (:meth:`run`): thin wrapper that chains plan → retrieve and
       returns all three artefacts (documents, live context, plan) to the
       caller.

    :class:`HybridRetriever` instances are created lazily and cached so that
    multiple calls within a single agent session reuse the same BM25 corpus
    and ChromaDB connection.

    Attributes:
        _retrievers: Cache of :class:`~src.retrieval.hybrid.HybridRetriever`
            instances keyed by collection name.
        _reranker: Shared :class:`~src.retrieval.reranker.Reranker` instance.
        _ctx_builder: Shared :class:`~src.retrieval.context_builder.ContextBuilder`
            instance.
    """

    def __init__(self) -> None:
        """Initialise the agent with lazy-loaded retrievers, a reranker, and a context builder."""
        self._retrievers: dict[str, HybridRetriever] = {}
        self._reranker: Reranker = Reranker()
        self._ctx_builder: ContextBuilder = ContextBuilder()
        logger.info("FinancialAgent initialised (retrievers loaded lazily).")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_retriever(self, collection_name: str) -> HybridRetriever:
        """Return a cached :class:`~src.retrieval.hybrid.HybridRetriever` for *collection_name*.

        The retriever is created on first access and stored in
        :attr:`_retrievers` for subsequent calls.

        Args:
            collection_name: Name of the ChromaDB collection (e.g. ``"news"``).

        Returns:
            The :class:`~src.retrieval.hybrid.HybridRetriever` for this collection.

        Raises:
            ValueError: Propagated from :class:`~src.retrieval.dense.DenseRetriever`
                if the collection does not exist (ingestion not yet run).
        """
        if collection_name not in self._retrievers:
            logger.info(
                "FinancialAgent: creating HybridRetriever for collection '%s'.",
                collection_name,
            )
            self._retrievers[collection_name] = HybridRetriever(
                collection_name=collection_name
            )
        return self._retrievers[collection_name]

    def _generate_sub_queries(
        self,
        query: str,
        tickers: list[str],
        query_type: str,
    ) -> list[str]:
        """Expand *query* into focused sub-queries for multi-ticker or complex cases.

        Strategy:

        - If *query_type* is ``"comparison"`` or ``"recommendation"`` **and** at
          least two tickers are detected, generate one per-ticker sub-query for
          each ticker (using the type-specific keyword from
          :data:`_QUERY_TYPE_KEYWORDS`) followed by a per-ticker news sub-query,
          and append the original query as a global fallback.
        - For all other cases (0 or 1 ticker, or a query type that does not
          benefit from decomposition), return ``[query]`` unchanged.

        Args:
            query: Original user query string.
            tickers: Ticker symbols detected by the context builder.
            query_type: Classified query type.

        Returns:
            Non-empty list of query strings; the last element is always the
            original *query*.
        """
        if query_type in {"comparison", "recommendation"} and len(tickers) >= 2:
            type_kw = _QUERY_TYPE_KEYWORDS.get(query_type, query_type)
            sub_queries: list[str] = []
            for ticker in tickers:
                sub_queries.append(f"{ticker} {type_kw}")
            for ticker in tickers:
                sub_queries.append(f"{ticker} recent news sentiment")
            # Always add the original query as the global anchor.
            sub_queries.append(query)
            return _dedup_queries(sub_queries)

        if query_type == "macro":
            # Expand macro queries into series-specific sub-queries so the dense
            # retriever finds data beyond the most obvious series (e.g. VIX).
            lower = query.lower()
            sub_queries = [query]
            if any(kw in lower for kw in ("rate", "fed", "interest", "fedfunds")):
                sub_queries.append("Federal Funds Rate FEDFUNDS interest rate")
            if any(kw in lower for kw in ("yield", "treasury", "dgs10", "dgs2", "curve")):
                sub_queries.append("10-year 2-year treasury yield DGS10 DGS2 yield curve T10Y2Y")
            if any(kw in lower for kw in ("vix", "volatility", "fear")):
                sub_queries.append("VIX VIXCLS CBOE volatility index market risk")
            if any(kw in lower for kw in ("unemployment", "jobless", "claims", "icsa")):
                sub_queries.append("initial jobless claims ICSA unemployment")
            # If query is a broad macro question, include all key series
            if len(sub_queries) == 1:
                sub_queries += [
                    "Federal Funds Rate FEDFUNDS interest rate monetary policy",
                    "VIX VIXCLS volatility index market risk",
                    "treasury yield curve DGS10 DGS2 T10Y2Y",
                ]
            return _dedup_queries(sub_queries)

        # Single-ticker or non-decomposable types → no expansion needed.
        return [query]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, query: str) -> RetrievalPlan:
        """Build a :class:`RetrievalPlan` from a raw user query.

        Steps:

        1. Extract ticker symbols via
           :meth:`~src.retrieval.context_builder.ContextBuilder.extract_tickers`.
        2. Classify the query type via
           :meth:`~src.retrieval.context_builder.ContextBuilder.classify_query`.
        3. Map the classified type to data sources using :data:`QUERY_TYPE_MAP`.
        4. Separate ChromaDB collections from live-enrichment sources.
        5. Generate sub-queries via :meth:`_generate_sub_queries`.

        The ``enrichment_query_type`` in the returned plan is set to the
        classified query type so that :class:`~src.retrieval.context_builder.ContextBuilder`
        knows which Supabase tables to hit.

        Args:
            query: Raw natural-language question from the user.

        Returns:
            A fully populated :class:`RetrievalPlan`.
        """
        # Step 1 — Ticker detection
        tickers = self._ctx_builder.extract_tickers(query)
        logger.debug("plan: tickers detected = %s", tickers)

        # Step 2 — Query classification (uses context_builder's keyword rules)
        cb_query_type = self._ctx_builder.classify_query(query)

        # Map context_builder types to agent QUERY_TYPE_MAP keys.
        query_type = _CB_TO_AGENT_TYPE.get(cb_query_type, "news")

        # Heuristic overrides — evaluated in priority order (highest last wins via elif chain).
        # Priority: portfolio > recommendation > macro > comparison > base type
        lower_query = query.lower()
        if any(kw in lower_query for kw in ("portfolio", "my holdings", "my positions")):
            query_type = "portfolio"
        elif any(kw in lower_query for kw in ("recommend", "should i buy", "invest in", "investment")):
            query_type = "recommendation"
        elif any(kw in lower_query for kw in ("vix", "fed", "gdp", "cpi", "rate", "yield", "macro", "inflation")):
            query_type = "macro"
        elif len(tickers) >= 2:
            query_type = "comparison"
        elif query_type == "news" and tickers and any(
            kw in lower_query for kw in ("fundamentals", "fundamental", "analyze", "analysis", "financial")
        ):
            # Generic single-ticker analysis → include earnings collection
            query_type = "earnings"

        logger.debug(
            "plan: cb_query_type='%s' → agent query_type='%s'",
            cb_query_type,
            query_type,
        )

        # Step 3 — Map query_type to data sources
        sources: list[str] = QUERY_TYPE_MAP.get(query_type, ["news"])

        # Step 4 — Split into ChromaDB collections vs live-enrichment sources
        collections: list[str] = [
            CHROMA_COLLECTION_MAP[src]
            for src in sources
            if src in CHROMA_COLLECTION_MAP
        ]
        # Deduplicate while preserving order
        seen_collections: set[str] = set()
        unique_collections: list[str] = []
        for col in collections:
            if col not in seen_collections:
                seen_collections.add(col)
                unique_collections.append(col)

        use_live_enrichment = any(src in LIVE_ENRICHMENT_TYPES for src in sources)

        # Step 5 — Generate sub-queries
        sub_queries = self._generate_sub_queries(query, tickers, query_type)

        plan = RetrievalPlan(
            query_type=query_type,
            sub_queries=sub_queries,
            tickers=tickers,
            collections=unique_collections,
            use_live_enrichment=use_live_enrichment,
            enrichment_query_type=cb_query_type,
        )

        logger.info(
            "plan: query_type='%s', tickers=%s, collections=%s, "
            "live_enrichment=%s, sub_queries=%d.",
            plan.query_type,
            plan.tickers,
            plan.collections,
            plan.use_live_enrichment,
            len(plan.sub_queries),
        )
        return plan

    def retrieve(
        self,
        plan: RetrievalPlan,
        top_k_per_collection: int = RERANK_TOP_K,
    ) -> tuple[list[RetrievedDocument], str]:
        """Execute a :class:`RetrievalPlan` and return ranked documents plus live context.

        Execution steps:

        1. For each collection in *plan.collections*:

           a. For each sub-query in *plan.sub_queries*, call
              :meth:`~src.retrieval.hybrid.HybridRetriever.retrieve_with_metadata_filter`
              with the first detected ticker (if any) to bias results toward
              relevant assets.
           b. Merge results from all sub-queries, deduplicating by ``doc_id``.

        2. Rerank the union with :class:`~src.retrieval.reranker.Reranker`
           using the **last** sub-query (the original user query) as the
           reranking signal.
        3. If *plan.use_live_enrichment* is ``True``, call
           :meth:`~src.retrieval.context_builder.ContextBuilder.enrich` with
           the detected tickers and enrichment query type.

        Collections that do not exist (ingestion not yet run) are silently
        skipped with a warning so that a partially-indexed system can still
        serve results.

        Args:
            plan: A :class:`RetrievalPlan` produced by :meth:`plan`.
            top_k_per_collection: Maximum number of documents to keep after
                reranking. Defaults to :data:`~src.config.RERANK_TOP_K`.

        Returns:
            A two-tuple ``(documents, live_context)`` where:

            - *documents* is a list of up to *top_k_per_collection*
              :class:`~src.retrieval.dense.RetrievedDocument` objects sorted
              by descending cross-encoder score.
            - *live_context* is a formatted string (possibly empty) with live
              price / technical-indicator data from Supabase.
        """
        # Accumulate all retrieved documents across collections and sub-queries.
        # Use an ordered dict keyed by doc_id for deduplication; first occurrence wins.
        doc_registry: dict[str, RetrievedDocument] = {}

        for collection_name in plan.collections:
            try:
                retriever = self._get_retriever(collection_name)
            except (ValueError, Exception) as exc:
                logger.warning(
                    "retrieve: skipping collection '%s' — %s: %s",
                    collection_name,
                    type(exc).__name__,
                    exc,
                )
                continue

            for sub_query in plan.sub_queries:
                # Macro collection docs use series_id, not ticker — never filter by ticker there.
                if collection_name == "macro":
                    ticker_filter: str | None = None
                # For single-ticker queries: always filter by ticker.
                # For multi-ticker queries: if the sub-query starts with a known ticker,
                # filter by that ticker to reduce noise; otherwise no filter (global sub-query).
                elif len(plan.tickers) == 1:
                    ticker_filter = plan.tickers[0]
                else:
                    # Check if sub-query is ticker-specific (starts with a known ticker)
                    first_word = sub_query.split()[0] if sub_query.split() else ""
                    ticker_filter = first_word if first_word in plan.tickers else None

                # Fetch more candidates for earnings to survive the recency sort.
                _dense_k = DENSE_TOP_K * 2 if collection_name == "earnings" else DENSE_TOP_K
                _hybrid_k = HYBRID_TOP_K * 2 if collection_name == "earnings" else HYBRID_TOP_K

                try:
                    results: list[RetrievedDocument] = retriever.retrieve_with_metadata_filter(
                        query=sub_query,
                        ticker=ticker_filter,
                        dense_top_k=_dense_k,
                        sparse_top_k=_dense_k,
                        final_top_k=_hybrid_k,
                    )
                    # For earnings, sort to prefer post-2019 data (Python-level, since
                    # ChromaDB 1.x only supports numeric $gte, not string dates).
                    if collection_name == "earnings":
                        results = sorted(
                            results,
                            key=lambda d: (d.metadata.get("fiscal_date") or "") >= "2020-01-01",
                            reverse=True,
                        )
                except Exception as exc:
                    logger.warning(
                        "retrieve: sub-query '%s' on collection '%s' failed — %s: %s",
                        sub_query[:60],
                        collection_name,
                        type(exc).__name__,
                        exc,
                    )
                    results = []

                # Deduplicate: first occurrence by doc_id wins.
                for doc in results:
                    if doc.doc_id not in doc_registry:
                        doc_registry[doc.doc_id] = doc

            logger.debug(
                "retrieve: collection '%s' done. Total unique docs so far: %d.",
                collection_name,
                len(doc_registry),
            )

        all_docs: list[RetrievedDocument] = list(doc_registry.values())
        logger.info(
            "retrieve: %d unique documents collected across %d collection(s). "
            "Starting rerank (top_k=%d).",
            len(all_docs),
            len(plan.collections),
            top_k_per_collection,
        )

        # Rerank using the last sub_query, which is always the original query.
        rerank_query = plan.sub_queries[-1] if plan.sub_queries else ""

        # For comparison/macro queries, pre-select a balanced set BEFORE the
        # cross-encoder to prevent it from systematically preferring one series.
        docs_to_rerank = all_docs
        if plan.query_type == "macro":
            # Balance by series_id: max 2 docs per series so all series are visible.
            series_slots: dict[str, list[RetrievedDocument]] = {}
            leftover_macro: list[RetrievedDocument] = []
            for doc in all_docs:
                sid = doc.metadata.get("series_id", "?")
                if len(series_slots.get(sid, [])) < 2:
                    series_slots.setdefault(sid, []).append(doc)
                else:
                    leftover_macro.append(doc)
            docs_to_rerank = [d for bucket in series_slots.values() for d in bucket]
            docs_to_rerank.extend(leftover_macro[: max(0, top_k_per_collection * 2 - len(docs_to_rerank))])

        elif plan.query_type == "comparison" and len(plan.tickers) >= 2:
            slots_per_ticker = max(3, top_k_per_collection // len(plan.tickers) + 1)
            pre_ticker: dict[str, list[RetrievedDocument]] = {t: [] for t in plan.tickers}
            leftover: list[RetrievedDocument] = []
            for doc in all_docs:
                t = doc.metadata.get("ticker", "")
                if t in pre_ticker and len(pre_ticker[t]) < slots_per_ticker:
                    pre_ticker[t].append(doc)
                else:
                    leftover.append(doc)
            docs_to_rerank = [d for bucket in pre_ticker.values() for d in bucket]
            # Top up with leftover docs (news, multi-ticker) if needed
            docs_to_rerank.extend(leftover[: max(0, top_k_per_collection * 2 - len(docs_to_rerank))])

        reranked_docs = self._reranker.rerank(
            query=rerank_query,
            documents=docs_to_rerank,
            top_k=top_k_per_collection * 3,  # over-fetch for post-diversity
        )

        # Post-rerank diversity for macro: at most 1 doc per series_id.
        if plan.query_type == "macro":
            seen_series: set[str] = set()
            diverse_macro: list[RetrievedDocument] = []
            for doc in reranked_docs:
                sid = doc.metadata.get("series_id") or doc.metadata.get("doc_type", "?")
                if sid not in seen_series:
                    seen_series.add(sid)
                    diverse_macro.append(doc)
                if len(diverse_macro) >= top_k_per_collection:
                    break
            reranked_docs = diverse_macro

        # Live enrichment from Supabase (prices + technicals)
        live_context = ""
        if plan.use_live_enrichment and plan.tickers:
            try:
                live_context = self._ctx_builder.enrich(
                    tickers=plan.tickers,
                    query_type=plan.enrichment_query_type,
                )
                logger.debug(
                    "retrieve: live enrichment returned %d chars.",
                    len(live_context),
                )
            except Exception as exc:
                logger.warning(
                    "retrieve: live enrichment failed — %s: %s",
                    type(exc).__name__,
                    exc,
                )

        return reranked_docs, live_context

    def run(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
    ) -> tuple[list[RetrievedDocument], str, RetrievalPlan]:
        """Full pipeline: plan → retrieve → return results.

        Convenience method that chains :meth:`plan` and :meth:`retrieve` in a
        single call and returns all three artefacts so that the downstream
        generator has everything it needs.

        Args:
            query: Raw natural-language question from the user.
            top_k: Maximum number of documents to return after reranking.
                Defaults to :data:`~src.config.RERANK_TOP_K`.

        Returns:
            A three-tuple ``(documents, live_context, plan)`` where:

            - *documents* is a list of up to *top_k*
              :class:`~src.retrieval.dense.RetrievedDocument` objects.
            - *live_context* is a formatted string with live market data
              (empty when not applicable).
            - *plan* is the :class:`RetrievalPlan` built for this query,
              useful for logging, evaluation, and UI display.

        Example::

            agent = FinancialAgent()
            docs, ctx, plan = agent.run("Is NVDA overbought?")
        """
        logger.info("FinancialAgent.run: query='%s'", query[:120])
        plan = self.plan(query)
        docs, live_context = self.retrieve(plan, top_k_per_collection=top_k)
        logger.info(
            "FinancialAgent.run: done — %d docs, live_context=%d chars.",
            len(docs),
            len(live_context),
        )
        return docs, live_context, plan
