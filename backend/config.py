"""
backend/config.py

Single source of truth for every swappable choice in the system.

Design rule: callers import from here, never from os.environ directly.
"""

import os
from dotenv import load_dotenv
from backend.ingest.chunk import ChunkConfig

load_dotenv()  # reads .env from cwd (project root when run via scripts/)


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
COHERE_API_KEY: str = os.environ.get("COHERE_API_KEY", "")
AZURE_STORAGE_CONNECTION_STRING: str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER: str = os.environ.get("AZURE_STORAGE_CONTAINER", "")


# ---------------------------------------------------------------------------
# Embedding model — per-collection with a sensible default
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

COLLECTION_EMBEDDING_MODELS: dict[str, str] = {}


def get_embedding_model(collection: str) -> str:
    return COLLECTION_EMBEDDING_MODELS.get(collection, DEFAULT_EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Collection registry — used by the router for auto-classification
# ---------------------------------------------------------------------------

COLLECTION_REGISTRY: dict[str, str] = {
    "sec-filings": (
        "Official SEC 10-K annual reports filed by public companies (AAPL, MSFT, "
        "NVDA, GOOGL, AMZN, TSLA, META, JPM, WMT, JNJ, PFE, XOM, DIS, KO, V). "
        "Contains audited financials, income statements, revenue, net income, "
        "risk factors, MD&A, business descriptions, and forward-looking statements."
    ),
    "legal-docs": (
        "Legal contracts, material agreements, and corporate exhibits filed with "
        "the SEC — includes credit agreements, executive compensation agreements, "
        "clawback policies, indemnification exhibits, employment agreements, "
        "and merger agreements from public companies (TSLA, MSFT, JPM, META, WMT)."
    ),
}


def get_embedding_dimension(model: str | None = None) -> int:
    m = model or DEFAULT_EMBEDDING_MODEL
    if m not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Unknown embedding model {m!r}. Add it to EMBEDDING_DIMENSIONS in config.py."
        )
    return EMBEDDING_DIMENSIONS[m]


# ---------------------------------------------------------------------------
# Role-Based Access Control
#
# Maps collection namespace → list of roles that may query it.
# The access_check_node enforces this before retrieval begins.
# Roles: admin (all), finance (SEC), legal (contracts), general (SEC only)
# ---------------------------------------------------------------------------

COLLECTION_ROLES: dict[str, list[str]] = {
    "sec-filings": ["admin", "finance", "general"],
    "legal-docs":  ["admin", "legal"],
}

ALL_ROLES: list[str] = ["admin", "finance", "legal", "general"]

ROLE_DESCRIPTIONS: dict[str, str] = {
    "admin":   "Full access to all collections",
    "finance": "Access to SEC filings only",
    "legal":   "Access to legal documents only",
    "general": "Read-only access to SEC filings",
}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

DEFAULT_LLM_MODEL: str = "gpt-4o-mini"

# Pricing per 1M tokens (as of mid-2025)
GPT4O_MINI_INPUT_COST_PER_1M: float = 0.15    # $0.15 per 1M input tokens
GPT4O_MINI_OUTPUT_COST_PER_1M: float = 0.60   # $0.60 per 1M output tokens


def estimate_cost(input_tokens: int, output_tokens: int, model: str = DEFAULT_LLM_MODEL) -> float:
    """Estimate USD cost for a given token usage."""
    if model == "gpt-4o-mini":
        return (
            input_tokens * GPT4O_MINI_INPUT_COST_PER_1M / 1_000_000
            + output_tokens * GPT4O_MINI_OUTPUT_COST_PER_1M / 1_000_000
        )
    # Fallback: blended rate
    return (input_tokens + output_tokens) * 0.0002 / 1_000


# ---------------------------------------------------------------------------
# Parser backend selector
# ---------------------------------------------------------------------------

PARSER_BACKEND: str = os.environ.get("PARSER_BACKEND", "unstructured")


# ---------------------------------------------------------------------------
# Chunk config defaults
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_CONFIG = ChunkConfig(
    child_tokens=350,
    parent_tokens=1750,
    overlap_pct=0.12,
    semantic_threshold=0.5,
    encoding_name="cl100k_base",
    min_parent_chars=80,
)


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

NEO4J_URI: str = os.environ.get("NEO4J_URI", "")
NEO4J_USER: str = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", ""))
NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE: str = os.environ.get("NEO4J_DATABASE", "neo4j")


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------

PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX_NAME", "agentic-rag")
PINECONE_CLOUD: str = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.environ.get("PINECONE_REGION", "us-east-1")
PINECONE_METRIC: str = "cosine"
