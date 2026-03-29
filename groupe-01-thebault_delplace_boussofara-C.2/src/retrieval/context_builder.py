"""Live enrichment context builder for the BrightVest Financial RAG system.

Fetches real-time price and technical-indicator data from Supabase at query
time. This module is intentionally NOT backed by ChromaDB — data is pulled
fresh from ``prices_daily`` and ``technical_indicators`` on every request so
that the LLM always sees up-to-date market context.

Typical usage::

    builder = ContextBuilder()

    # High-level helper: detect tickers + classify + enrich in one call.
    context = builder.enrich_from_query("Is NVDA overbought right now?")

    # Or drive it manually when tickers / query_type are already known.
    context = builder.enrich(["AAPL", "MSFT"], query_type="momentum")
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from src.data.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword maps for query classification
# ---------------------------------------------------------------------------

_QUERY_TYPE_KEYWORDS: dict[str, list[str]] = {
    "price": [
        "price", "drop", "rise", "rally", "crash", "trading", "close",
        "open", "high", "low", "fell", "gained", "lost",
        "performance", "return",
        # NOTE: "up"/"down" intentionally removed — they appear as substrings in
        # "support", "upcoming", "breakdown", "markdown" etc., causing false positives.
        # "rise"/"drop"/"fell"/"lost"/"gained" cover these use-cases without ambiguity.
    ],
    "technical": [
        "rsi", "macd", "overbought", "oversold", "signal", "indicator",
        "crossover", "divergence", "stochastic", "bollinger",
    ],
    "momentum": [
        "momentum", "trend", "moving average", "breakout", "breakdowns",
        "ema", "sma", "50-day", "200-day",
        # NOTE: "ma" intentionally removed — it matches "market", "macro",
        # "margin", "management" etc. as a substring, causing false positives.
        # "moving average" already covers the MA use-case.
    ],
    "fundamental": [
        "earnings", "revenue", "profit", "eps", "pe ratio", "margin",
        "balance sheet", "cash flow", "debt", "equity", "valuation",
        "ebitda", "growth", "fundamentals", "fundamental", "analyze",
        "analysis", "financial", "income", "net income", "quarterly",
        "annual", "fiscal", "roic", "roe", "roa", "dividend", "peg",
    ],
    "news": [
        "news", "announcement", "report", "headline", "press release",
        "article", "sentiment", "coverage", "media",
    ],
}

# Common English words that look like tickers but are not.
_COMMON_WORDS: frozenset[str] = frozenset({
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO",
    "TO", "UP", "US", "WE", "AND", "ARE", "BUT", "CAN", "DID", "FOR",
    "GET", "GOT", "HAD", "HAS", "HIM", "HIS", "HOW", "ITS", "LET",
    "MAY", "NOT", "NOW", "OFF", "OUR", "OUT", "OWN", "PUT", "SAY",
    "SHE", "THE", "TOO", "TWO", "WAS", "WHO", "WHY", "YET", "YOU",
    "ALSO", "BEEN", "FROM", "HAVE", "HERE", "INTO", "JUST", "KNOW",
    "LIKE", "MADE", "MORE", "MUCH", "NEED", "OVER", "SAID", "SAME",
    "SOME", "SUCH", "THAN", "THAT", "THEM", "THEN", "THERE", "THEY",
    "THIS", "TIME", "VERY", "WANT", "WELL", "WENT", "WERE", "WHAT",
    "WHEN", "WITH", "WOULD", "YOUR", "ABOUT", "AFTER", "AGAIN",
    "COULD", "EVERY", "FIRST", "FOUND", "GREAT", "THEIR", "THESE",
    "THOSE", "THREE", "UNDER", "WHICH", "WHILE", "STILL",
    # Financial / common acronyms that are not tickers
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "ETF", "IPO", "SEC",
    "FED", "GDP", "CPI", "PPI", "PMI", "QE", "YOY", "QOQ", "TTM",
    "CEO", "CFO", "COO", "CTO", "ESG", "P&L", "ROE", "ROA", "EPS",
    "RSI", "EMA", "SMA", "ATH", "ATL", "IPO", "SPX", "NDX", "DOW",
    # Macro indices / rate series — not stock tickers
    "VIX", "VIXCLS", "FRED", "FOMC", "ECB", "BOJ", "SNB", "RBA",
    "FEDFUNDS", "DGS10", "DGS2", "T10Y2Y", "ICSA", "PCE", "NFP",
})

# Regex: uppercase word of 1–5 letters, anchored by non-alpha-or-start/end.
_TICKER_PATTERN: re.Pattern[str] = re.compile(r"(?<![A-Z])([A-Z]{1,5})(?![A-Z])")


class ContextBuilder:
    """Fetches live data from ``prices_daily`` and ``technical_indicators``.

    All heavy I/O goes through :class:`~src.data.supabase_client.SupabaseClient`.
    This class adds the orchestration logic (ticker extraction, query
    classification, formatting) on top of those raw queries.

    Attributes:
        _db: The shared :class:`~src.data.supabase_client.SupabaseClient`
            instance used for all Supabase queries.
    """

    def __init__(self) -> None:
        """Initialise the ContextBuilder and its Supabase connection."""
        self._db: SupabaseClient = SupabaseClient()

    # ------------------------------------------------------------------
    # Ticker detection
    # ------------------------------------------------------------------

    def extract_tickers(self, text: str) -> list[str]:
        """Detect likely ticker symbols inside *text*.

        Applies a simple heuristic:
        - Finds all uppercase sequences of 1–5 letters not surrounded by
          other uppercase letters (so that e.g. "NVIDIA" is not split into
          sub-sequences).
        - Removes entries found in :data:`_COMMON_WORDS`.
        - Deduplicates while preserving first-seen order.

        Args:
            text: Raw query string from the user.

        Returns:
            Ordered, deduplicated list of candidate ticker strings (uppercase).
            May be empty if no candidates are found.

        Example:
            >>> cb = ContextBuilder()
            >>> cb.extract_tickers("Is AAPL overbought compared to MSFT?")
            ['AAPL', 'MSFT']
        """
        candidates = _TICKER_PATTERN.findall(text)
        seen: set[str] = set()
        tickers: list[str] = []
        for candidate in candidates:
            # Skip common words; skip single-letter matches (false positives from sentence starts)
            if candidate in _COMMON_WORDS or len(candidate) < 2:
                continue
            if candidate not in seen:
                seen.add(candidate)
                tickers.append(candidate)
        return tickers

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def classify_query(self, query: str) -> str:
        """Classify *query* into a single retrieval category.

        Uses keyword matching against :data:`_QUERY_TYPE_KEYWORDS`.  The
        first category whose keywords appear (case-insensitive substring
        match) in the query wins.  Falls back to ``"general"`` when no
        keywords match.

        Priority order: ``price`` → ``technical`` → ``momentum`` →
        ``fundamental`` → ``news`` → ``"general"``.

        Args:
            query: The user's natural-language question.

        Returns:
            One of: ``"price"``, ``"technical"``, ``"momentum"``,
            ``"fundamental"``, ``"news"``, ``"general"``.

        Example:
            >>> cb = ContextBuilder()
            >>> cb.classify_query("Is NVDA overbought based on RSI?")
            'technical'
        """
        lower = query.lower()
        for query_type, keywords in _QUERY_TYPE_KEYWORDS.items():
            if any(kw in lower for kw in keywords):
                return query_type
        return "general"

    # ------------------------------------------------------------------
    # Raw data fetchers (thin wrappers over SupabaseClient)
    # ------------------------------------------------------------------

    def get_recent_prices(
        self,
        tickers: list[str],
        days: int = 30,
    ) -> list[dict]:
        """Fetch recent OHLCV rows for *tickers* from Supabase.

        Args:
            tickers: Ticker symbols to fetch.
            days: Look-back window in calendar days. Defaults to 30.

        Returns:
            List of price dicts as returned by
            :meth:`~src.data.supabase_client.SupabaseClient.fetch_recent_prices`.
            Empty list if *tickers* is empty or a Supabase error occurs.
        """
        if not tickers:
            return []
        return self._db.fetch_recent_prices(tickers, days=days)

    def get_latest_technicals(self, tickers: list[str]) -> list[dict]:
        """Fetch the most recent technical-indicator row for each ticker.

        Args:
            tickers: Ticker symbols to fetch.

        Returns:
            List of technical-indicator dicts (one per ticker at most) as
            returned by
            :meth:`~src.data.supabase_client.SupabaseClient.fetch_latest_technicals`.
            Empty list if *tickers* is empty or a Supabase error occurs.
        """
        if not tickers:
            return []
        return self._db.fetch_latest_technicals(tickers)

    # ------------------------------------------------------------------
    # Formatters
    # ------------------------------------------------------------------

    def format_price_summary(self, prices: list[dict]) -> str:
        """Format a list of OHLCV rows into a human-readable price summary.

        Groups rows by ``symbol`` and computes:
        - Latest closing price (most recent ``trade_date``).
        - Period high and low across all returned rows.
        - Percentage change from the oldest close to the latest close.
        - Average daily volume.

        Args:
            prices: Raw price dicts from
                :meth:`get_recent_prices`.  May be empty.

        Returns:
            Multi-line string starting with
            ``"=== Price Data (last 30 days) ==="``, one line per symbol.
            Returns an empty string if *prices* is empty.

        Example output::

            === Price Data (last 30 days) ===
            AAPL: Close $182.50 (latest), Range $178.20-$189.40,
                  30d change: +3.2%, Volume avg: 58.3M
        """
        if not prices:
            return ""

        # Group rows by symbol; rows are ordered desc by trade_date from DB.
        by_symbol: dict[str, list[dict]] = defaultdict(list)
        for row in prices:
            sym = row.get("symbol") or "UNKNOWN"
            by_symbol[sym].append(row)

        lines: list[str] = ["=== Price Data (last 30 days) ==="]

        for symbol, rows in sorted(by_symbol.items()):
            # Rows come in desc order → index 0 is the most recent.
            latest_row = rows[0]
            oldest_row = rows[-1]

            latest_close: float | None = latest_row.get("close_price") or latest_row.get("adj_close")
            oldest_close: float | None = oldest_row.get("close_price") or oldest_row.get("adj_close")

            # Range across the period
            highs = [r["high_price"] for r in rows if r.get("high_price") is not None]
            lows = [r["low_price"] for r in rows if r.get("low_price") is not None]
            period_high: float | None = max(highs) if highs else None
            period_low: float | None = min(lows) if lows else None

            # 30-day percentage change
            change_str = "N/A"
            if latest_close is not None and oldest_close is not None and oldest_close != 0:
                change_pct = (latest_close - oldest_close) / oldest_close * 100
                sign = "+" if change_pct >= 0 else ""
                change_str = f"{sign}{change_pct:.1f}%"

            # Average volume (in millions for readability)
            volumes = [r["volume"] for r in rows if r.get("volume") is not None]
            vol_str = "N/A"
            if volumes:
                avg_vol = sum(volumes) / len(volumes)
                vol_str = f"{avg_vol / 1_000_000:.1f}M"

            close_str = f"${latest_close:,.2f}" if latest_close is not None else "N/A"
            range_str = (
                f"${period_low:,.2f}-${period_high:,.2f}"
                if period_low is not None and period_high is not None
                else "N/A"
            )

            lines.append(
                f"  {symbol}: Close {close_str} (latest), Range {range_str}, "
                f"30d change: {change_str}, Volume avg: {vol_str}"
            )

        return "\n".join(lines)

    def format_technicals(self, technicals: list[dict]) -> str:
        """Format a list of technical-indicator rows into a readable summary.

        For each row, displays RSI (with overbought/oversold interpretation),
        MACD line and signal, short-/medium-term returns, 20-day volatility,
        and 1-year rolling max drawdown.

        RSI interpretation thresholds:
        - ``< 30`` → ``"oversold"``
        - ``> 70`` → ``"overbought"``
        - otherwise → ``"neutral"``

        Args:
            technicals: Raw technical-indicator dicts from
                :meth:`get_latest_technicals`.  May be empty.

        Returns:
            Multi-line string starting with
            ``"=== Technical Indicators ==="``, one line per symbol.
            Returns an empty string if *technicals* is empty.
            ``None`` values are shown as ``"N/A"``.

        Example output::

            === Technical Indicators ===
            NVDA: RSI(14)=62.5 [neutral], MACD=+2.30 (signal=+1.80),
                  Return 1d=-0.8%, 5d=+4.2%, 20d=+12.1%,
                  Volatility(20d)=28.5%, Max Drawdown(1Y)=-15.2%
        """
        if not technicals:
            return ""

        lines: list[str] = ["=== Technical Indicators ==="]

        for row in sorted(technicals, key=lambda r: r.get("symbol", "")):
            symbol: str = row.get("symbol", "UNKNOWN")

            # RSI
            rsi: float | None = row.get("rsi_14")
            if rsi is not None:
                if rsi < 30:
                    rsi_label = "oversold"
                elif rsi > 70:
                    rsi_label = "overbought"
                else:
                    rsi_label = "neutral"
                rsi_str = f"{rsi:.1f} [{rsi_label}]"
            else:
                rsi_str = "N/A"

            # MACD
            macd: float | None = row.get("macd")
            macd_signal: float | None = row.get("macd_signal")

            def _fmt_signed(val: float | None) -> str:
                if val is None:
                    return "N/A"
                sign = "+" if val >= 0 else ""
                return f"{sign}{val:.2f}"

            macd_str = _fmt_signed(macd)
            sig_str = _fmt_signed(macd_signal)

            # Returns
            def _fmt_pct(val: float | None) -> str:
                if val is None:
                    return "N/A"
                sign = "+" if val >= 0 else ""
                return f"{sign}{val * 100:.1f}%"

            ret_1d = _fmt_pct(row.get("return_1d"))
            ret_5d = _fmt_pct(row.get("return_5d"))
            ret_20d = _fmt_pct(row.get("return_20d"))

            # Volatility
            vol_20d: float | None = row.get("volatility_20d")
            vol_str = f"{vol_20d * 100:.1f}%" if vol_20d is not None else "N/A"

            # Max drawdown
            mdd: float | None = row.get("max_drawdown_rolling_1y")
            mdd_str = f"{mdd * 100:.1f}%" if mdd is not None else "N/A"

            lines.append(
                f"  {symbol}: RSI(14)={rsi_str}, MACD={macd_str} (signal={sig_str}), "
                f"Return 1d={ret_1d}, 5d={ret_5d}, 20d={ret_20d}, "
                f"Volatility(20d)={vol_str}, Max Drawdown(1Y)={mdd_str}"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def enrich(self, tickers: list[str], query_type: str) -> str:
        """Fetch and format live market context for *tickers*.

        Decides which data sources to hit based on *query_type*:

        - ``"price"``, ``"performance"``, ``"momentum"``, ``"drop"``,
          ``"rise"`` → fetches recent prices.
        - ``"technical"``, ``"overbought"``, ``"momentum"``, ``"trend"`` →
          fetches latest technical indicators.
        - Some types (e.g. ``"momentum"``) trigger **both** sources.

        Args:
            tickers: Ticker symbols to enrich.  Returns ``""`` immediately
                if the list is empty.
            query_type: Category string as returned by
                :meth:`classify_query`.

        Returns:
            A formatted multi-section string ready to be injected into a
            prompt as additional context.  Sections are separated by a blank
            line.  Returns ``""`` when no data is available or *tickers* is
            empty.
        """
        if not tickers:
            return ""

        context_parts: list[str] = []

        # query_type may come from classify_query() ("price", "technical",
        # "momentum", "general"…) OR from the agentic planner's QUERY_TYPE_MAP
        # which uses finer-grained labels ("performance", "drop", "rise", etc.).
        # Both sources are handled here deliberately.
        price_types = {"price", "performance", "momentum", "drop", "rise"}
        technical_types = {"technical", "overbought", "momentum", "trend"}

        if query_type in price_types:
            prices = self.get_recent_prices(tickers, days=30)
            price_section = self.format_price_summary(prices)
            if price_section:
                context_parts.append(price_section)

        if query_type in technical_types:
            technicals = self.get_latest_technicals(tickers)
            tech_section = self.format_technicals(technicals)
            if tech_section:
                context_parts.append(tech_section)

        return "\n\n".join(context_parts)

    def enrich_from_query(self, query: str) -> str:
        """Convenience wrapper that runs the full enrichment pipeline.

        Chains :meth:`extract_tickers` → :meth:`classify_query` →
        :meth:`enrich` in a single call.

        Args:
            query: Raw user query string.

        Returns:
            Formatted context string (possibly multi-section), or ``""`` if
            no tickers are detected or no data is available.

        Example::

            builder = ContextBuilder()
            ctx = builder.enrich_from_query("Is NVDA overbought right now?")
            # Returns something like:
            # === Technical Indicators ===
            #   NVDA: RSI(14)=62.5 [neutral], MACD=+2.30 ...
        """
        tickers = self.extract_tickers(query)
        if not tickers:
            logger.debug("enrich_from_query: no tickers detected in %r", query)
            return ""

        query_type = self.classify_query(query)
        logger.debug(
            "enrich_from_query: tickers=%s query_type=%s", tickers, query_type
        )
        return self.enrich(tickers, query_type)
