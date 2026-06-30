"""
backend/config.py

Single source of truth for every swappable choice in the system.

Design rule: callers import from here, never from os.environ directly.
Changing a model, index name, or chunk size means changing one value here —
nothing else needs touching.

Per-collection embedding model override
----------------------------------------
The embedding model is a per-COLLECTION config, not per-document or per-query.
This is an architectural constraint: vectors in one Pinecone namespace must all
use the same embedding space — you can't mix text-embedding-3-small vectors
with voyage-finance-2 vectors in the same namespace and search them together.

So the rule is:
  - default_embedding_model  → used for every collection not in the override map
  - COLLECTION_EMBEDDING_MODELS → maps specific collection names to specialist models
  - get_embedding_model(collection) → the one function everything calls

Upgrade rule: only add a collection to COLLECTION_EMBEDDING_MODELS when RAGAS
evals prove the specialist model earns its higher cost for that collection.
"""

import os
from dotenv import load_dotenv
from backend.ingest.chunk import ChunkConfig

load_dotenv()  # reads .env from cwd (project root when run via scripts/)


# ---------------------------------------------------------------------------
# API keys — read from environment, never hard-coded
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
COHERE_API_KEY: str = os.environ.get("COHERE_API_KEY", "")
AZURE_STORAGE_CONNECTION_STRING: str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
AZURE_STORAGE_CONTAINER: str = os.environ.get("AZURE_STORAGE_CONTAINER", "")


# ---------------------------------------------------------------------------
# Embedding model — per-collection, with a sensible default
# ---------------------------------------------------------------------------

# Default: text-embedding-3-small — best cost/performance ratio for general use
# (~$0.02/M tokens, ~96% of large's quality). Every collection uses this unless
# explicitly overridden below.
DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

# Dimensions produced by each supported embedding model.
# Pinecone index creation needs to know the dimension upfront.
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Add specialist models here as collections adopt them, e.g.:
    # "voyage-finance-2": 1024,
}

# Per-collection overrides: only set a collection here after RAGAS evals prove
# the specialist model earns its cost for that specific collection.
COLLECTION_EMBEDDING_MODELS: dict[str, str] = {
    # "sec_filings": "voyage-finance-2",  # example — enable after eval justifies it
}


def get_embedding_model(collection: str) -> str:
    """Return the embedding model configured for this collection."""
    return COLLECTION_EMBEDDING_MODELS.get(collection, DEFAULT_EMBEDDING_MODEL)


# ---------------------------------------------------------------------------
# Collection registry — used by the router for auto-classification
#
# Each entry maps a collection namespace name to a short description that the
# classifier reads to decide which collection(s) a query belongs to.
# Add an entry here whenever a new collection is indexed.
# "legal" is a stub — no documents are indexed yet; the description lets the
# classifier route correctly as soon as documents are added.
# ---------------------------------------------------------------------------

COLLECTION_REGISTRY: dict[str, str] = {
    "demo": (
        "Personal resume, work experience, skills, certifications, projects, "
        "and academic background of a data science / ML candidate."
    ),
    "finance": (
        "Financial statements, SEC filings, earnings reports, revenue, ROI, "
        "profit, fiscal year data, and stock metrics for public companies."
    ),
    "legal": (
        "Legal contracts, compliance documents, regulatory filings, litigation "
        "records, court cases, settlements, and legal dispute cases. "
        "[stub — no documents indexed yet]"
    ),
}


def get_embedding_dimension(model: str | None = None) -> int:
    """Return the vector dimension for the given model (or the default)."""
    m = model or DEFAULT_EMBEDDING_MODEL
    if m not in EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Unknown embedding model {m!r}. Add it to EMBEDDING_DIMENSIONS in config.py."
        )
    return EMBEDDING_DIMENSIONS[m]


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

# Default for both planning/grading nodes AND generation.
# Upgrade a specific node to a stronger model only when RAGAS or manual eval
# shows the node is the quality bottleneck — not upfront.
DEFAULT_LLM_MODEL: str = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Parser backend selector
# ---------------------------------------------------------------------------

# "unstructured" → UnstructuredBackend (default, open-source)
# "azure_doc_intelligence" → AzureDocIntelligenceBackend (TODO, premium PDF tables)
# Switch only after RAGAS context-precision traces weak performance to table parsing.
PARSER_BACKEND: str = os.environ.get("PARSER_BACKEND", "unstructured")


# ---------------------------------------------------------------------------
# Chunk config defaults
# ---------------------------------------------------------------------------

# These become the ChunkConfig() defaults. Override per-collection if RAGAS
# shows a different collection benefits from different sizing.
DEFAULT_CHUNK_CONFIG = ChunkConfig(
    child_tokens=350,
    parent_tokens=1750,
    overlap_pct=0.12,
    semantic_threshold=0.5,
    encoding_name="cl100k_base",
    min_parent_chars=80,   # merge standalone headers/dates into their first content block
)


# ---------------------------------------------------------------------------
# Neo4j (graph layer — used by graph_rag/)
# ---------------------------------------------------------------------------

NEO4J_URI: str = os.environ.get("NEO4J_URI", "")
# AuraDB uses NEO4J_USERNAME; fall back to NEO4J_USER for self-managed installs
NEO4J_USER: str = os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", ""))
NEO4J_PASSWORD: str = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE: str = os.environ.get("NEO4J_DATABASE", "neo4j")


# ---------------------------------------------------------------------------
# Pinecone
# ---------------------------------------------------------------------------

# One index for the whole project — collections are Pinecone namespaces within it.
# Pinecone free tier: 5 indexes, 100 namespaces each — ample.
PINECONE_INDEX_NAME: str = os.environ.get("PINECONE_INDEX_NAME", "agentic-rag")

# Pinecone cloud + region for index creation (only matters when creating a new index).
PINECONE_CLOUD: str = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION: str = os.environ.get("PINECONE_REGION", "us-east-1")

# Metric used for similarity search. cosine is correct for OpenAI embeddings
# (they are L2-normalised, so cosine == dot product, both work).
PINECONE_METRIC: str = "cosine"
