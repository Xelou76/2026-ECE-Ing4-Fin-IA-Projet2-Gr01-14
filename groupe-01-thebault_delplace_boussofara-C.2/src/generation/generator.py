"""LLM generation layer for the BrightVest Financial RAG system.

This module exposes :class:`RAGGenerator`, which orchestrates document
retrieval (via :class:`~src.retrieval.dense.DenseRetriever`), context
formatting, and Groq LLM calls to produce structured financial answers.

Two answer modes are supported:

* ``"simple"``  – Returns a :class:`SimpleAnswer` (plain text + sources).
* ``"analyst"`` – Returns an :class:`AnalystAnswer` (bull/bear JSON analysis).

Example::

    from src.generation.generator import RAGGenerator

    gen = RAGGenerator()
    answer = gen.answer("What are NVDA's latest earnings?", mode="analyst", ticker="NVDA")
    print(answer.signal)          # "bullish" | "bearish" | "neutral"
    print(answer.bull_case)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, computed_field

from src.config import (
    DENSE_TOP_K,
    GROQ_API_KEY,
    GROQ_RPM_LIMIT,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from src.generation.prompts import (
    ANALYST_SYSTEM_PROMPT,
    ANALYST_USER_TEMPLATE,
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
    QUERY_REWRITE_TEMPLATE,
    STANDALONE_QUESTION_TEMPLATE,
)
from src.data.supabase_client import SupabaseClient
from src.generation.agent import FinancialAgent
from src.retrieval.dense import DenseRetriever, RetrievedDocument

logger = logging.getLogger(__name__)

# Minimum seconds between consecutive Groq API calls to stay under 30 req/min.
_MIN_INTERVAL: float = 60.0 / GROQ_RPM_LIMIT  # 2.0 s

MAX_RETRIES: int = 3
RETRY_WAIT_SECONDS: int = 60


# ---------------------------------------------------------------------------
# Output Pydantic models
# ---------------------------------------------------------------------------


class SourceReference(BaseModel):
    """A single bibliographic reference extracted from retrieved documents.

    Attributes:
        type: Document category – ``"news"``, ``"earnings"``, or ``"macro"``.
        ticker: Equity ticker symbol when applicable, e.g. ``"AAPL"``.
        date: ISO-8601 date string associated with the source document.
        detail: Short human-readable summary of the referenced fact.
        verified: True if the LLM actually cited this source in its answer.
    """

    type: str  # "news" | "earnings" | "macro"
    ticker: str | None = None
    date: str | None = None
    detail: str | None = None
    verified: bool = True

    @computed_field
    @property
    def badge(self) -> str:
        return "✅" if self.verified else "⚠️"


class SimpleAnswer(BaseModel):
    """Output of :meth:`RAGGenerator.answer` in ``"simple"`` mode.

    Attributes:
        answer: Plain-text answer to the user's question.
        sources: List of source references supporting the answer.
        confidence: Self-assessed confidence – ``"high"``, ``"medium"``, or ``"low"``.
    """

    answer: str
    sources: list[SourceReference]
    confidence: str  # "high" | "medium" | "low"


class AnalystAnswer(BaseModel):
    """Output of :meth:`RAGGenerator.answer` in ``"analyst"`` mode.

    Attributes:
        answer: 2-3 sentence executive summary.
        bull_case: Key bullish arguments grounded in retrieved context.
        bear_case: Key bearish arguments or risks grounded in retrieved context.
        risks: List of specific risk factors.
        catalysts: List of potential positive catalysts.
        key_metrics: Mapping of metric name to value/description string.
        sources: List of source references used in the analysis.
        confidence: Self-assessed confidence – ``"high"``, ``"medium"``, or ``"low"``.
        signal: Overall directional view – ``"bullish"``, ``"bearish"``, or ``"neutral"``.
    """

    answer: str
    bull_case: str
    bear_case: str
    risks: list[str]
    catalysts: list[str]
    key_metrics: dict[str, str]
    sources: list[SourceReference]
    confidence: str  # "high" | "medium" | "low"
    signal: str  # "bullish" | "bearish" | "neutral"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class RAGGenerator:
    """Orchestrate retrieval and LLM generation for financial questions.

    On construction the class initialises a :class:`~langchain_groq.ChatGroq`
    client and a :class:`~src.retrieval.dense.DenseRetriever`.  A lightweight
    in-process rate limiter ensures the free-tier Groq limit (30 req/min) is
    respected without external state.

    Args:
        collection_name: ChromaDB collection to query during retrieval.
            Defaults to ``"news"`` – pass ``"fundamentals"`` or ``"macro"``
            to target those collections instead.

    Raises:
        RuntimeError: Propagated from :class:`~langchain_groq.ChatGroq` if
            ``GROQ_API_KEY`` is empty or invalid.
        ValueError: Propagated from :class:`~src.retrieval.dense.DenseRetriever`
            if the ChromaDB collection does not exist.
    """

    def __init__(self, collection_name: str = "news") -> None:
        self._llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            api_key=GROQ_API_KEY,
        )
        self._retriever = DenseRetriever(collection_name=collection_name)
        # Timestamp of the most recent Groq API call (epoch seconds).
        self._last_request_time: float = 0.0
        self._agent: FinancialAgent | None = None
        logger.info(
            "RAGGenerator initialised (model=%s, collection=%s).",
            LLM_MODEL,
            collection_name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_context(self, docs: list[RetrievedDocument]) -> str:
        """Format retrieved documents into an LLM-readable context block.

        Each document is rendered as a numbered source block containing the
        document type, ticker, date and full content.

        Args:
            docs: Ordered list of retrieved documents (most relevant first).

        Returns:
            Multi-line string ready to be injected into a prompt template.
        """
        if not docs:
            return "No relevant documents found."

        parts: list[str] = []
        for idx, doc in enumerate(docs, start=1):
            meta = doc.metadata
            doc_type: str = meta.get("doc_type", "unknown")
            ticker: str = meta.get("ticker", "N/A")

            # Prefer published_at for news, fiscal_date for earnings, date for macro.
            date: str = (
                meta.get("published_at")
                or meta.get("fiscal_date")
                or meta.get("date")
                or "N/A"
            )

            header = f"[Source {idx}] {doc_type} | {ticker} | {date}"
            parts.append(f"{header}\n{doc.content}\n")

        return "\n".join(parts)

    def _extract_sources(self, docs: list[RetrievedDocument]) -> list[SourceReference]:
        """Build :class:`SourceReference` objects from document metadata.

        Args:
            docs: Retrieved documents whose metadata contains source info.

        Returns:
            List of :class:`SourceReference` instances, one per document.
        """
        sources: list[SourceReference] = []
        for doc in docs:
            meta = doc.metadata
            doc_type: str = meta.get("doc_type", "unknown")
            ticker: str | None = meta.get("ticker") or None
            date: str | None = (
                meta.get("published_at")
                or meta.get("fiscal_date")
                or meta.get("date")
                or None
            )
            # Use the first 120 characters of content as a human-readable detail.
            detail: str | None = doc.content[:120].strip() or None
            sources.append(
                SourceReference(type=doc_type, ticker=ticker, date=date, detail=detail)
            )
        return sources

    def _rate_limit(self) -> None:
        """Enforce the Groq free-tier rate limit (30 req/min).

        Sleeps for the remaining fraction of :data:`_MIN_INTERVAL` (2 s) when
        the previous request was issued less than 2 seconds ago.
        """
        elapsed: float = time.monotonic() - self._last_request_time
        if elapsed < _MIN_INTERVAL:
            wait: float = _MIN_INTERVAL - elapsed
            logger.debug("Rate limiter: sleeping %.2f s before Groq call.", wait)
            time.sleep(wait)

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        """Send a chat completion request to Groq and return the raw text.

        Retries up to :data:`MAX_RETRIES` times with increasing backoff when
        a 429 rate-limit error is returned by the Groq API.

        Args:
            system_prompt: Content for the ``SystemMessage``.
            user_message: Content for the ``HumanMessage``.

        Returns:
            Raw string content of the LLM response.

        Raises:
            RuntimeError: If the Groq API call fails for any non-rate-limit
                reason, or if the rate limit is still exceeded after all
                retries.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._rate_limit()
            try:
                response = self._llm.invoke(messages)
                self._last_request_time = time.monotonic()
                content: Any = response.content
                return str(content).strip()
            except Exception as exc:
                if "ratelimit" in type(exc).__name__.lower() or "429" in str(exc):
                    wait = RETRY_WAIT_SECONDS * (attempt + 1)
                    logger.warning(
                        "Groq rate limit hit (attempt %d/%d), waiting %ds...",
                        attempt + 1,
                        MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    last_exc = exc
                    continue
                raise RuntimeError(
                    f"Groq API call failed ({LLM_MODEL}): {type(exc).__name__}"
                ) from exc
        raise RuntimeError(
            f"Groq rate limit exceeded after {MAX_RETRIES} retries"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        mode: str = "simple",
        ticker: str | None = None,
        top_k: int = DENSE_TOP_K,
    ) -> SimpleAnswer | AnalystAnswer:
        """Retrieve context and generate a structured financial answer.

        The method performs the following steps:

        1. Retrieve the *top_k* most relevant documents from ChromaDB,
           optionally filtered to a single *ticker*.
        2. Format the documents into a context block.
        3. Apply the Groq rate limiter.
        4. Call the Groq LLM with the appropriate prompt template.
        5. Parse the response into a typed Pydantic model.

        Args:
            question: Natural-language question from the user.
            mode: Answer mode – ``"simple"`` (plain Q&A) or ``"analyst"``
                (structured bull/bear JSON analysis).
            ticker: Optional equity ticker to narrow retrieval, e.g. ``"AAPL"``.
            top_k: Maximum number of documents to retrieve from ChromaDB.

        Returns:
            A :class:`SimpleAnswer` when *mode* is ``"simple"``, or an
            :class:`AnalystAnswer` when *mode* is ``"analyst"``.

        Raises:
            RuntimeError: If the Groq API call fails.
            ValueError: If *mode* is not ``"simple"`` or ``"analyst"``.
        """
        if mode not in {"simple", "analyst"}:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'simple' or 'analyst'.")

        # 1. Retrieve relevant documents.
        docs: list[RetrievedDocument] = self._retriever.retrieve_with_metadata_filter(
            query=question,
            ticker=ticker,
            top_k=top_k,
        )
        logger.info(
            "answer(mode=%s, ticker=%s): retrieved %d documents.",
            mode,
            ticker,
            len(docs),
        )

        # 2. Format context and extract source references.
        context: str = self._format_context(docs)
        sources: list[SourceReference] = self._extract_sources(docs)

        # 3 & 4. Choose prompt template and call the LLM.
        if mode == "simple":
            user_message = QA_USER_TEMPLATE.format(context=context, question=question)
            raw_response: str = self._call_llm(QA_SYSTEM_PROMPT, user_message)
            verified_sources = self._verify_citations(raw_response, sources)
            return SimpleAnswer(
                answer=raw_response,
                sources=verified_sources,
                confidence=self._infer_confidence(docs, verified_sources),
            )

        # mode == "analyst"
        user_message = ANALYST_USER_TEMPLATE.format(context=context, question=question)
        raw_response = self._call_llm(ANALYST_SYSTEM_PROMPT, user_message)

        # 5. Parse JSON response → AnalystAnswer, with graceful fallback.
        return self._parse_analyst_response(raw_response, sources, docs, answer_text=raw_response)

    def _verify_citations(
        self,
        answer_text: str,
        sources: list[SourceReference],
    ) -> list[SourceReference]:
        """Mark sources as verified if they are explicitly cited in the answer.

        Parses [Source N] references from the answer text and sets
        verified=True only for sources actually cited by the LLM.
        Sources not cited receive verified=False (potential hallucination risk).

        Args:
            answer_text: Raw text of the LLM answer.
            sources: Source references (one per retrieved document, 1-indexed).

        Returns:
            Updated list with verified field set on each source.
        """
        # Match both "[Source N]" and "[..., Source N, ...]" formats
        cited_indices: set[int] = {
            int(n)
            for n in re.findall(r"\bSource\s+(\d+)\b", answer_text, re.IGNORECASE)
        }
        return [
            src.model_copy(update={"verified": (i + 1) in cited_indices})
            for i, src in enumerate(sources)
        ]

    def _infer_confidence(
        self,
        docs: list[RetrievedDocument],
        verified_sources: list[SourceReference] | None = None,
    ) -> str:
        """Heuristically infer answer confidence from retrieved document count.

        When verified_sources are available (citation tracking active):
        - ≥4 verified citations → "high"
        - ≥2 verified citations → "medium"
        - 1 verified citation   → "low"
        - 0 verified citations  → doc-count heuristic but capped at "medium"
          (LLM may not have used [Source N] markers, so we can't fully trust it)

        Without citation tracking, pure doc-count heuristic is used.

        Args:
            docs: Retrieved documents used to build the answer.
            verified_sources: Optional list of sources with verified field set.

        Returns:
            ``"high"``, ``"medium"``, or ``"low"``.
        """
        if verified_sources is not None:
            cited_count = sum(1 for s in verified_sources if s.verified)
            if cited_count >= 4:
                return "high"
            if cited_count >= 2:
                return "medium"
            if cited_count > 0:
                return "low"
            # cited_count == 0: LLM didn't use [Source N] markers.
            # Fall back to doc count but cap at "medium" — we cannot verify
            # which sources were actually used.
            return "medium" if len(docs) >= 2 else "low"

        # No citation info available — pure doc-count heuristic.
        n = len(docs)
        if n >= 5:
            return "high"
        if n >= 2:
            return "medium"
        return "low"

    def _parse_analyst_response(
        self,
        raw_response: str,
        sources: list[SourceReference],
        docs: list[RetrievedDocument],
        answer_text: str = "",
    ) -> AnalystAnswer:
        """Parse a raw LLM string into an :class:`AnalystAnswer`.

        The LLM is instructed to produce valid JSON.  When parsing fails the
        method falls back to a minimal :class:`AnalystAnswer` whose *answer*
        field contains the raw LLM output so the information is never silently
        discarded.

        Args:
            raw_response: Raw text returned by the Groq LLM.
            sources: Pre-extracted source references.
            docs: Retrieved documents (used for confidence inference).
            answer_text: Raw LLM answer text used for citation verification.

        Returns:
            A populated :class:`AnalystAnswer`.
        """
        verified_sources = self._verify_citations(answer_text or raw_response, sources)

        # Try to extract JSON from markdown code fences first.
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_response)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = raw_response.strip()
            # If no code fences, try to extract a bare JSON object by finding
            # the outermost { ... } block (handles "Here is the JSON: {...}" responses).
            bare_json_match = re.search(r"\{[\s\S]*\}", cleaned)
            if bare_json_match:
                cleaned = bare_json_match.group(0)

        try:
            data: dict[str, Any] = json.loads(cleaned)

            def _to_list(val: Any) -> list[str]:
                """Coerce LLM output to list[str] — handles string 'N/A' gracefully."""
                if isinstance(val, list):
                    return [str(x) for x in val]
                if not val or val == "N/A":
                    return []
                return [str(val)]

            return AnalystAnswer(
                answer=data.get("answer", "N/A"),
                bull_case=data.get("bull_case", "N/A"),
                bear_case=data.get("bear_case", "N/A"),
                risks=_to_list(data.get("risks", [])),
                catalysts=_to_list(data.get("catalysts", [])),
                key_metrics={
                    str(k): str(v)
                    for k, v in (data.get("key_metrics") or {}).items()
                },
                sources=verified_sources,
                confidence=data.get("confidence", self._infer_confidence(docs, verified_sources)),
                signal=data.get("signal", "neutral"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to parse analyst JSON response (%s). Falling back to raw text.",
                type(exc).__name__,
            )
            return AnalystAnswer(
                answer=raw_response,
                bull_case="N/A",
                bear_case="N/A",
                risks=[],
                catalysts=[],
                key_metrics={},
                sources=verified_sources,
                confidence=self._infer_confidence(docs, verified_sources),
                signal="neutral",
            )

    def _contextualize_question(
        self,
        question: str,
        history: list[dict],
    ) -> str:
        """Rewrite a follow-up question as a standalone question using conversation history.

        Sends the last 6 history messages + current question to the LLM and asks it
        to produce a self-contained version suitable for independent retrieval.
        Falls back silently to the original question on error.

        Args:
            question: The user's current question (may be a follow-up).
            history: List of previous turns, each a dict with keys ``"role"``
                and ``"content"``.

        Returns:
            Standalone rewritten question, or the original if the call fails.
        """
        if not history:
            return question

        hist_lines = [
            f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content'][:400]}"
            for h in history[-6:]
        ]
        hist_text = "\n".join(hist_lines)
        prompt = STANDALONE_QUESTION_TEMPLATE.format(history=hist_text, question=question)
        try:
            rewritten = self._call_llm(
                system_prompt=(
                    "You are a conversation contextualizer for a financial AI. "
                    "Return ONLY the rewritten standalone question with no other text."
                ),
                user_message=prompt,
            )
            logger.debug(
                "_contextualize_question: '%s' → '%s'.",
                question[:60],
                (rewritten or question)[:60],
            )
            return rewritten or question
        except RuntimeError:
            return question

    def rewrite_query(self, question: str) -> str:
        """Rewrite a user question to improve vector-store retrieval quality.

        Sends the question to the LLM with a query-optimisation prompt that
        encourages the addition of relevant financial terminology and ticker
        symbols.  Falls back silently to the original question on error.

        Args:
            question: Original natural-language question.

        Returns:
            Rewritten query string, or the original *question* if the LLM
            call fails.
        """
        prompt = QUERY_REWRITE_TEMPLATE.format(question=question)
        try:
            rewritten: str = self._call_llm(
                system_prompt=(
                    "You are a financial search query optimizer. "
                    "Return ONLY the rewritten query with no additional text."
                ),
                user_message=prompt,
            )
            logger.debug(
                "rewrite_query: '%s' → '%s'.", question[:60], rewritten[:60]
            )
            return rewritten or question
        except RuntimeError as exc:
            logger.warning("rewrite_query failed (%s). Using original question.", type(exc).__name__)
            return question

    def answer_with_agent(
        self,
        question: str,
        mode: str = "simple",
        history: list[dict] | None = None,
    ) -> SimpleAnswer | AnalystAnswer:
        """Full agentic RAG pipeline.

        Uses FinancialAgent for ticker detection, collection routing,
        sub-query decomposition, reranking, and live enrichment, then
        generates a structured answer via Groq.

        Args:
            question: Natural-language question from the user.
            mode: Answer mode – ``"simple"`` (plain Q&A) or ``"analyst"``
                (structured bull/bear JSON analysis).
            history: Optional list of previous conversation turns, each a dict
                with keys ``"role"`` (``"user"`` or ``"assistant"``) and
                ``"content"`` (plain text).  When provided, the question is
                first rewritten as a standalone query before retrieval, and
                the history is injected into the generation prompt so the LLM
                can reference earlier answers.

        Returns:
            A :class:`SimpleAnswer` when *mode* is ``"simple"``, or an
            :class:`AnalystAnswer` when *mode* is ``"analyst"``.

        Raises:
            ValueError: If *mode* is not ``"simple"`` or ``"analyst"``.
            RuntimeError: If the Groq API call fails.
        """
        if mode not in {"simple", "analyst"}:
            raise ValueError(f"Unknown mode '{mode}'. Expected 'simple' or 'analyst'.")

        # 1. Lazy-init the FinancialAgent.
        if self._agent is None:
            self._agent = FinancialAgent()

        # 2. When history is present, rewrite the question as a standalone query
        #    so the retrieval layer has enough context to find the right documents.
        retrieval_query = (
            self._contextualize_question(question, history) if history else question
        )

        # 3. Run the full agentic pipeline: plan → retrieve → rerank → live enrich.
        docs, live_context, plan = self._agent.run(retrieval_query)

        logger.info(
            "answer_with_agent: plan=query_type='%s', collections=%s, tickers=%s, "
            "docs=%d, live_context=%d chars.",
            plan.query_type,
            plan.collections,
            plan.tickers,
            len(docs),
            len(live_context),
        )

        # 4. Build the context block from retrieved documents + optional live data.
        context: str = self._format_context(docs)
        if live_context:
            context = f"{context}\n\n## Live Market Data\n{live_context}"

        sources: list[SourceReference] = self._extract_sources(docs)

        # 5. Build a conversation-history block to prepend to the generation prompt.
        #    Truncate each turn to 500 chars to avoid token bloat.
        history_block = ""
        if history:
            hist_lines = [
                f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content'][:500]}"
                for h in history[-6:]
            ]
            history_block = "Conversation History:\n" + "\n".join(hist_lines) + "\n\n"

        # 6. Generate the answer using the requested mode.
        if mode == "simple":
            user_message = history_block + QA_USER_TEMPLATE.format(
                context=context, question=question
            )
            raw_response: str = self._call_llm(QA_SYSTEM_PROMPT, user_message)
            verified_sources = self._verify_citations(raw_response, sources)
            return SimpleAnswer(
                answer=raw_response,
                sources=verified_sources,
                confidence=self._infer_confidence(docs, verified_sources),
            )

        # mode == "analyst"
        user_message = history_block + ANALYST_USER_TEMPLATE.format(
            context=context, question=question
        )
        raw_response = self._call_llm(ANALYST_SYSTEM_PROMPT, user_message)
        return self._parse_analyst_response(raw_response, sources, docs, answer_text=raw_response)

    def answer_portfolio(
        self,
        question: str,
        user_id: str | None = None,
        mode: str = "analyst",
    ) -> SimpleAnswer | AnalystAnswer:
        """Agentic RAG analysis scoped to the user's portfolio positions.

        Fetches Supabase positions, extracts the tickers, then runs a
        targeted RAG analysis focused on those tickers.

        Args:
            question: Natural-language question from the user.
            user_id: Optional user identifier to filter portfolios by owner.
            mode: Answer mode – ``"simple"`` or ``"analyst"``.

        Returns:
            A :class:`SimpleAnswer` when *mode* is ``"simple"``, or an
            :class:`AnalystAnswer` when *mode* is ``"analyst"``.
        """
        # 1. Fetch portfolio positions from Supabase.
        try:
            positions: list[dict] = SupabaseClient().fetch_positions(user_id)
        except Exception as exc:
            logger.error("answer_portfolio: failed to fetch positions: %s", type(exc).__name__)
            return SimpleAnswer(
                answer="Failed to retrieve portfolio positions.",
                sources=[],
                confidence="low",
            )

        # 2. Extract unique tickers from the positions.
        tickers: list[str] = [
            p["ticker"] for p in positions if p.get("ticker")
        ]

        # 3. No positions found — return a graceful SimpleAnswer.
        if not tickers:
            logger.warning(
                "answer_portfolio: no positions found for user_id=%s.", user_id
            )
            return SimpleAnswer(
                answer="No portfolio found. Please add positions to your portfolio first.",
                sources=[],
                confidence="low",
            )

        # 4. Build a human-readable portfolio summary for additional context.
        portfolio_lines: list[str] = []
        for pos in positions:
            ticker = pos.get("ticker", "N/A")
            name = pos.get("name", "N/A")
            quantity = pos.get("quantity", "N/A")
            pru = pos.get("pru", "N/A")
            currency = pos.get("currency", "N/A")
            portfolio_lines.append(
                f"{ticker} ({name}): {quantity} shares @ {pru} {currency}"
            )
        portfolio_summary: str = "Portfolio positions:\n" + "\n".join(portfolio_lines)

        logger.info(
            "answer_portfolio: %d positions found, tickers=%s.", len(tickers), tickers
        )

        # 5. Prepend the tickers to the question so the agent detects them,
        #    and include the portfolio summary as a prefix.
        tickers_prefix: str = ", ".join(tickers)
        enriched_question: str = (
            f"{tickers_prefix}. {portfolio_summary}\n\nQuestion: {question}"
        )

        return self.answer_with_agent(enriched_question, mode=mode)


# ---------------------------------------------------------------------------
# Module-level integration helper
# ---------------------------------------------------------------------------


def get_financial_signal(ticker: str) -> dict:
    """Generate a structured investment signal for BrightVest integration.

    Args:
        ticker: Stock ticker symbol (e.g. "NVDA").

    Returns:
        dict with keys: ticker, signal, confidence (float), bull_case,
        bear_case, key_metrics, news_count, date_range, error.
    """
    _CONF_MAP = {"high": 0.9, "medium": 0.6, "low": 0.3}
    try:
        gen = RAGGenerator()
        answer = gen.answer_with_agent(
            f"Analyze {ticker} as an investment. Give bull and bear case.",
            mode="analyst",
        )
        if not isinstance(answer, AnalystAnswer):
            raise TypeError(f"Expected AnalystAnswer, got {type(answer).__name__}")
        # Count news docs from agent's last retrieval (best effort)
        news_count = 0
        if gen._agent is not None:
            try:
                docs, _, _ = gen._agent.run(f"{ticker} recent news", top_k=5)
                news_count = sum(1 for d in docs if d.metadata.get("doc_type") == "news")
            except Exception:
                pass
        # Compute date_range from key_metrics or leave N/A
        date_range = "N/A"
        return {
            "ticker": ticker,
            "signal": answer.signal,
            "confidence": _CONF_MAP.get(answer.confidence, 0.3),
            "bull_case": answer.bull_case,
            "bear_case": answer.bear_case,
            "key_metrics": answer.key_metrics,
            "news_count": news_count,
            "date_range": date_range,
            "error": None,
        }
    except Exception as exc:
        logger.error("get_financial_signal(%s) failed: %s", ticker, type(exc).__name__)
        return {
            "ticker": ticker,
            "signal": "neutral",
            "confidence": 0.0,
            "bull_case": "",
            "bear_case": "",
            "key_metrics": {},
            "news_count": 0,
            "date_range": "N/A",
            "error": type(exc).__name__,
        }
