"""Supabase client for the BrightVest Financial RAG system.

Provides typed, read-only access to all Supabase tables used by the RAG
pipeline.  Each method wraps a supabase-py query with consistent error
handling: exceptions are caught, logged to stderr, and an empty list is
returned so callers never have to guard against None.

Example:
    from src.data.supabase_client import SupabaseClient

    client = SupabaseClient()
    articles = client.fetch_articles(ticker="AAPL", limit=200)
    fundamentals = client.fetch_fundamentals(symbol="NVDA", period_type="annual")
"""

import logging
from datetime import date, timedelta

from supabase import Client, create_client

from src.config import SUPABASE_KEY, SUPABASE_URL

logger = logging.getLogger(__name__)

_ARTICLES_COLUMNS = (
    "ticker, headline, summary, source, published_at, "
    "sector, sentiment_final, category, dedup_hash"
)

_VALID_PERIODS = frozenset({"annual", "quarterly"})


class SupabaseClient:
    """Read-only Supabase client for the BrightVest RAG pipeline.

    Wraps supabase-py and exposes typed query methods for every table that
    the RAG system consumes.  All writes are intentionally absent — this
    client is strictly read-only.

    Attributes:
        _client: The underlying supabase-py ``Client`` instance.
    """

    def __init__(self) -> None:
        """Initialise the Supabase connection.

        Raises:
            ValueError: If SUPABASE_URL or SUPABASE_KEY are empty strings,
                which usually means the .env file is missing or the env vars
                have not been exported.
        """
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set. "
                "Check your .env file for NEXT_PUBLIC_SUPABASE_URL and "
                "NEXT_PUBLIC_SUPABASE_ANON_KEY."
            )
        self._client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ------------------------------------------------------------------
    # articles
    # ------------------------------------------------------------------

    def fetch_articles(
        self,
        ticker: str | None = None,
        date_from: str | None = None,
        limit: int = 1000,
    ) -> list[dict]:
        """Fetch news articles from the ``articles`` table.

        Args:
            ticker: Optional ticker symbol to filter on (exact match).
            date_from: Optional ISO date string (e.g. ``"2025-01-01"``).
                Only articles with ``published_at >= date_from`` are returned.
            limit: Maximum number of rows to return.  Defaults to 1 000.

        Returns:
            A list of dicts, each containing: ``ticker``, ``headline``,
            ``summary``, ``source``, ``published_at``, ``sector``,
            ``sentiment_final``, ``category``, ``dedup_hash``.
            Returns an empty list on error.
        """
        try:
            query = (
                self._client.table("articles")
                .select(_ARTICLES_COLUMNS)
                .order("published_at", desc=True)
                .limit(limit)
            )
            if ticker is not None:
                query = query.eq("ticker", ticker)
            if date_from is not None:
                query = query.gte("published_at", date_from)

            response = query.execute()
            return response.data or []
        except Exception as exc:
            logger.error("fetch_articles failed: %s", exc)
            return []

    def fetch_all_articles_paginated(
        self,
        limit: int = 5000,
        batch_size: int = 1000,
    ) -> list[dict]:
        """Fetch articles in batches to avoid gateway timeouts.

        Iterates over the ``articles`` table in pages of ``batch_size`` rows,
        ordered by ``published_at DESC``, until ``limit`` rows have been
        collected or the table is exhausted.

        Args:
            limit: Total maximum number of articles to return.
            batch_size: Number of rows fetched per Supabase request.

        Returns:
            A flat list of article dicts (most recent first), capped at
            ``limit`` rows.  Returns whatever was collected before the error
            if a batch fails mid-way.
        """
        results: list[dict] = []
        offset = 0

        while len(results) < limit:
            batch_limit = min(batch_size, limit - len(results))
            try:
                response = (
                    self._client.table("articles")
                    .select(_ARTICLES_COLUMNS)
                    .order("published_at", desc=True)
                    .range(offset, offset + batch_limit - 1)
                    .execute()
                )
                batch = response.data or []
            except Exception as exc:
                logger.error(
                    "fetch_all_articles_paginated failed at offset %d: %s",
                    offset,
                    exc,
                )
                break

            results.extend(batch)

            if len(batch) < batch_limit:
                # Last page — no more rows in the table.
                break

            offset += batch_limit

        return results

    # ------------------------------------------------------------------
    # fundamentals_serving
    # ------------------------------------------------------------------

    def fetch_fundamentals(
        self,
        symbol: str | None = None,
        period_type: str | None = None,
        limit: int = 5000,
        batch_size: int = 1000,
    ) -> list[dict]:
        """Fetch financial fundamentals from ``fundamentals_serving``.

        Paginates to overcome the default Supabase 1000-row limit.  Results
        are ordered by ``fiscal_date_ending DESC`` so that the most recent
        data is returned first when ``limit`` is smaller than the full table.

        Args:
            symbol: Optional ticker symbol to filter on (exact match).
            period_type: Optional period filter — ``"annual"`` or
                ``"quarterly"``.
            limit: Maximum total rows to return.  Defaults to 5000.
            batch_size: Rows fetched per Supabase request.

        Returns:
            A list of dicts with all columns from ``fundamentals_serving``.
            Returns an empty list on error.
        """
        if period_type is not None and period_type not in _VALID_PERIODS:
            raise ValueError(
                f"period_type must be 'annual' or 'quarterly', got {period_type!r}"
            )

        results: list[dict] = []
        offset = 0

        while len(results) < limit:
            batch_limit = min(batch_size, limit - len(results))
            try:
                query = (
                    self._client.table("fundamentals_serving")
                    .select("*")
                    .order("fiscal_date_ending", desc=True)
                    .range(offset, offset + batch_limit - 1)
                )
                if symbol is not None:
                    query = query.eq("symbol", symbol)
                if period_type is not None:
                    query = query.eq("period_type", period_type)

                response = query.execute()
                batch = response.data or []
            except Exception as exc:
                logger.error("fetch_fundamentals failed at offset %d: %s", offset, exc)
                break

            results.extend(batch)

            if len(batch) < batch_limit:
                break  # Last page

            offset += batch_limit

        return results

    # ------------------------------------------------------------------
    # macro_indicators
    # ------------------------------------------------------------------

    def fetch_macro_indicators(
        self,
        series_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch macro-economic indicators from ``macro_indicators``.

        Args:
            series_ids: Optional list of FRED/legacy series identifiers
                (e.g. ``["FEDFUNDS", "DGS10"]``).  If provided, only rows
                whose ``series_id`` is in this list are returned.
            limit: Maximum number of rows to return per series (applied after
                ordering by ``date DESC``).

        Returns:
            A list of dicts containing: ``series_id``, ``date``, ``value``,
            ``name``, ``category``.  Returns an empty list on error.
        """
        try:
            query = (
                self._client.table("macro_indicators")
                .select("series_id, date, value, name, category")
                .order("date", desc=True)
                .limit(limit)
            )
            if series_ids is not None:
                query = query.in_("series_id", series_ids)

            response = query.execute()
            return response.data or []
        except Exception as exc:
            logger.error("fetch_macro_indicators failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # prices_daily
    # ------------------------------------------------------------------

    def fetch_recent_prices(
        self,
        symbols: list[str],
        days: int = 30,
    ) -> list[dict]:
        """Fetch recent OHLCV prices from ``prices_daily``.

        Args:
            symbols: List of ticker symbols to fetch prices for.
            days: Look-back window in calendar days.  Rows with
                ``trade_date >= today - days`` are returned.

        Returns:
            A list of dicts containing: ``symbol``, ``trade_date``,
            ``open_price``, ``high_price``, ``low_price``, ``close_price``,
            ``adj_close``, ``volume``.  Returns an empty list on error.
        """
        if not symbols:
            return []

        cutoff: str = (date.today() - timedelta(days=days)).isoformat()

        try:
            response = (
                self._client.table("prices_daily")
                .select(
                    "symbol, trade_date, open_price, high_price, "
                    "low_price, close_price, adj_close, volume"
                )
                .in_("symbol", symbols)
                .gte("trade_date", cutoff)
                .order("trade_date", desc=True)
                .execute()
            )
            return response.data or []
        except Exception as exc:
            logger.error("fetch_recent_prices failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # technical_indicators
    # ------------------------------------------------------------------

    def fetch_latest_technicals(
        self,
        symbols: list[str],
    ) -> list[dict]:
        """Fetch the most recent technical indicators for each symbol.

        Queries ``technical_indicators`` and returns exactly one row per
        symbol — the row with the latest ``date``.

        Args:
            symbols: List of ticker symbols.

        Returns:
            A list of dicts containing: ``symbol``, ``date``,
            ``return_1d``, ``return_5d``, ``return_20d``,
            ``volatility_20d``, ``rsi_14``, ``macd``, ``macd_signal``,
            ``macd_histogram``, ``max_drawdown_rolling_1y``,
            ``volume_avg_20d``, ``turnover_ratio``.
            One dict per symbol at most.  Returns an empty list on error.
        """
        if not symbols:
            return []

        try:
            response = (
                self._client.table("technical_indicators")
                .select(
                    "symbol, date, return_1d, return_5d, return_20d, "
                    "volatility_20d, rsi_14, macd, macd_signal, macd_histogram, "
                    "max_drawdown_rolling_1y, volume_avg_20d, turnover_ratio"
                )
                .in_("symbol", symbols)
                .order("symbol")
                .order("date", desc=True)
                .execute()
            )
            seen: set[str] = set()
            results: list[dict] = []
            for row in response.data or []:
                if row["symbol"] not in seen:
                    seen.add(row["symbol"])
                    results.append(row)
            return results
        except Exception as exc:
            logger.error("fetch_latest_technicals failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # portfolios / positions
    # ------------------------------------------------------------------

    def fetch_positions(
        self,
        user_id: str | None = None,
    ) -> list[dict]:
        """Fetch portfolio positions, optionally filtered by user.

        Joins ``positions`` with ``portfolios`` so that a ``user_id`` filter
        can be applied.  Only position-level fields are returned.

        Args:
            user_id: Optional Supabase auth user UUID.  When provided, only
                positions belonging to portfolios owned by this user are
                returned.

        Returns:
            A list of dicts containing: ``ticker``, ``name``, ``quantity``,
            ``pru``, ``currency``, ``purchase_date``.
            Returns an empty list on error.
        """
        try:
            query = (
                self._client.table("positions")
                .select(
                    "ticker, name, quantity, pru, currency, purchase_date, "
                    "portfolios(user_id)"
                )
            )
            if user_id is not None:
                # Note: server-side join filter may vary by supabase-py version
                query = query.eq("portfolios.user_id", user_id)

            response = query.execute()
            rows = response.data or []

            # Strip the nested portfolios dict — callers don't need it.
            cleaned: list[dict] = []
            for row in rows:
                portfolio_info = row.get("portfolios")
                if user_id is not None:
                    # When filtering by user_id, supabase-py returns rows where
                    # the joined portfolio matched; portfolios key may be None
                    # for non-matching rows — skip those.
                    if portfolio_info is None:
                        continue
                    # Client-side fallback filter in case server-side join
                    # filter did not apply (behaviour varies by supabase-py
                    # version).
                    portfolio_user_id = (
                        portfolio_info.get("user_id")
                        if isinstance(portfolio_info, dict)
                        else None
                    )
                    if portfolio_user_id != user_id:
                        continue
                position = {
                    k: v for k, v in row.items() if k != "portfolios"
                }
                cleaned.append(position)

            return cleaned
        except Exception as exc:
            logger.error("fetch_positions failed: %s", exc)
            return []
