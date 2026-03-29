"""Centralized configuration for the BrightVest Financial RAG system.

Loads environment variables from a .env file at startup and exposes all
tunable constants used across ingestion, retrieval, generation and evaluation
modules.  Import individual names rather than the module object so that type
checkers can narrow the types correctly.

Example:
    from src.config import GROQ_API_KEY, LLM_MODEL, DENSE_TOP_K
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env (silently ignored when the file is absent)
# ---------------------------------------------------------------------------
load_dotenv()           # .env
load_dotenv(".env.local", override=True)  # .env.local overrides (Next.js convention)

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL: str = os.environ.get("NEXT_PUBLIC_SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")

# ---------------------------------------------------------------------------
# Groq LLM
# ---------------------------------------------------------------------------
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------
CHROMA_DB_PATH: str = str(Path(__file__).parent.parent / "chroma_db")

# ---------------------------------------------------------------------------
# Embeddings (Ollama — local, Apple-Silicon-friendly)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = "nomic-embed-text"
OLLAMA_BASE_URL: str = "http://localhost:11434"

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.1

# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------
GROQ_RPM_LIMIT: int = 30  # requests per minute on the free tier

# ---------------------------------------------------------------------------
# Retrieval hyper-parameters
# ---------------------------------------------------------------------------
DENSE_TOP_K: int = 15   # candidates returned by ChromaDB dense search
HYBRID_TOP_K: int = 30  # candidates after RRF fusion (dense + sparse)
RERANK_TOP_K: int = 5   # final passages kept after cross-encoder reranking

# ---------------------------------------------------------------------------
# Ingestion settings
# ---------------------------------------------------------------------------
NEWS_FETCH_LIMIT: int = 5000  # max articles fetched per ingestion run
NEWS_CHUNK_SIZE: int = 512    # max tokens per news chunk
