"""FastAPI REST API for the BrightVest Financial RAG system.

Exposes the RAGGenerator as a JSON API for consumption by the Next.js frontend.

Run with:
    uvicorn src.api:app --reload --port 8000

Endpoints:
    POST /api/chat    — Generate a RAG answer (simple or analyst mode).
    GET  /api/stats   — ChromaDB collection statistics.
    GET  /api/health  — Health check.
"""

import logging
from contextlib import asynccontextmanager
from typing import Literal

import chromadb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import CHROMA_DB_PATH, LLM_MODEL
from src.generation.generator import RAGGenerator

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and teardown application-level resources.

    Loads RAGGenerator once at startup to avoid reloading ChromaDB and BM25
    on every request.  If ChromaDB collections do not yet exist (ingestion not
    run), startup completes without a generator and the chat endpoint returns
    503 with a clear message rather than crashing uvicorn.
    """
    try:
        app.state.generator = RAGGenerator()
        logger.info("RAGGenerator initialised successfully.")
    except Exception as exc:
        logger.error(
            "RAGGenerator failed to initialise (%s): %s. "
            "Run the ingestion pipeline first: python -m src.ingestion.pipeline",
            type(exc).__name__,
            exc,
        )
        app.state.generator = None
    yield


app = FastAPI(title="BrightVest RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Request body for the /api/chat endpoint."""

    question: str
    mode: Literal["simple", "analyst"] = "simple"
    ticker: str | None = None
    history: list[dict] | None = None  # [{role: "user"|"assistant", content: str}]


@app.post("/api/chat")
async def chat(request: Request, body: ChatRequest) -> JSONResponse:
    """Generate a RAG answer for a financial question.

    Args:
        request: FastAPI request object (used to access app state).
        body: Chat request containing the question, mode, and optional ticker.

    Returns:
        JSONResponse with a SimpleAnswer or AnalystAnswer serialized as JSON,
        plus a ``_mode`` field indicating which answer type was returned.

    Raises:
        HTTPException: 500 if the RAG pipeline raises an unexpected error.
    """
    generator: RAGGenerator | None = request.app.state.generator
    if generator is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not ready. Run the ingestion pipeline first: "
                   "python -m src.ingestion.pipeline",
        )
    try:
        # Prepend ticker to question so FinancialAgent can detect it via its own extraction.
        # Strip whitespace to avoid prepending "   . question" for whitespace-only inputs.
        ticker = body.ticker.strip() if body.ticker else None
        question = f"{ticker}. {body.question}" if ticker else body.question
        answer = generator.answer_with_agent(question, mode=body.mode, history=body.history or None)
        result = answer.model_dump()
        result["_mode"] = body.mode
        return JSONResponse(content=result)
    except Exception as exc:
        logger.error("chat endpoint error: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail=type(exc).__name__) from exc


@app.get("/api/stats")
async def stats() -> JSONResponse:
    """Return ChromaDB collection statistics.

    Returns:
        JSONResponse with counts per collection::

            {
                "news": {"collection": "news", "count": 4823},
                "fundamentals": {"collection": "fundamentals", "count": 739},
                "macro": {"collection": "macro", "count": 45204},
            }

    Raises:
        HTTPException: 500 if the ingestion pipeline raises an unexpected error.
    """
    try:
        # Use a lightweight ChromaDB-only path — no indexers or Ollama needed.
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        data: dict[str, dict] = {}
        for source_key, coll_name in [
            ("news", "news"),
            ("fundamentals", "earnings"),
            ("macro", "macro"),
        ]:
            try:
                coll = client.get_collection(coll_name)
                data[source_key] = {"collection": coll_name, "count": coll.count()}
            except Exception:
                data[source_key] = {"collection": coll_name, "count": 0}
        return JSONResponse(content=data)
    except Exception as exc:
        logger.error("stats endpoint error: %s", type(exc).__name__, exc_info=True)
        raise HTTPException(status_code=500, detail=type(exc).__name__) from exc


@app.get("/api/health")
async def health() -> JSONResponse:
    """Health check endpoint.

    Returns:
        JSONResponse with ``{"status": "ok", "model": <LLM_MODEL>}``.
    """
    return JSONResponse(content={"status": "ok", "model": LLM_MODEL})
