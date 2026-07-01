"""
backend/agent/state.py

LangGraph agent state — the shared object passed between every node in the graph.

ARCHITECTURE CONSTRAINT:
  State MUST carry questions and retrieved source chunks for rewriting and grounding.
  State MUST NOT carry prior generated answers (hallucination laundering risk).

LANGGRAPH REDUCER SEMANTICS:
  - Annotated[list[X], operator.add]  → additive (append)
  - Annotated[dict, _merge_dicts]     → merge (union)
  - Annotated[int, operator.add]      → additive (sum)
  - All other fields                  → last-write-wins
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated

from langgraph.graph import MessagesState

from backend.retrieval.hybrid import ChunkResult


# ---------------------------------------------------------------------------
# Dict merge reducer — used for step_latencies so each node can add its key
# without reading-then-writing the full dict.
# ---------------------------------------------------------------------------

def _merge_dicts(a: dict, b: dict) -> dict:
    return {**a, **b}


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """
    One source reference attached to the generated answer.

    Enhanced with structured metadata so the UI can render human-readable
    labels like "Microsoft FY2025 Form 10-K" instead of raw filenames.
    """
    chunk_id: str
    doc_id: str
    filename: str
    collection: str
    source_text: str       # verbatim chunk text — shown in citation card
    # Enhanced metadata (populated by generate_node)
    company: str = ""
    filing_type: str = ""  # "10-K", "EX-10", "EX-10.7", etc.
    fiscal_year: str = ""  # "FY2025", "FY2024", etc.
    section: str = ""      # "Item 7", "Item 1A", "Exhibit 10.17", etc.
    display_name: str = "" # "Microsoft FY2025 Form 10-K → Item 7"


@dataclass
class CollectionQuery:
    """
    One collection-scoped sub-question produced by decompose_node.
    """
    collection: str
    sub_question: str


@dataclass
class ConversationTurn:
    """
    One complete Q&A turn stored in history for query rewriting.
    No answer field — see architecture constraint above.
    """
    question: str
    rewritten_query: str | None
    retrieved_chunks: list[ChunkResult]


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(MessagesState):
    """
    Shared state object threaded through every LangGraph node.

    Node responsibilities:
      rewrite_node        → writes rewritten_query
      classify_node       → writes query_classification
      router_node         → writes collection, route, active_collections
      access_check_node   → writes access_denied, access_denial_reason
      decompose_node      → writes collection_queries
      retrieve_*_node     → writes retrieved_chunks; appends reusable_chunks
      grade_node          → writes grade, facet_grades, answerability, conflict_detected
      validate_numbers_node → writes numeric_validation
      generate_node       → writes answer, citations
      (all nodes)         → update step_latencies, total_tokens
    """

    # ── Input ────────────────────────────────────────────────────────────────

    question: str
    """The user's current raw question."""

    collection: str
    """Pinecone namespace to search."""

    allowed_scopes: list[str]
    """Access identifiers for the requesting user/role."""

    user_role: str
    """
    Role of the requesting user: 'admin' | 'finance' | 'legal' | 'general'.
    Controls which collections the user may access (enforced by access_check_node).
    """

    # ── Rewrite node ─────────────────────────────────────────────────────────

    rewritten_query: str
    """Standalone, self-contained version of the question for retrieval."""

    # ── Query classification ─────────────────────────────────────────────────

    query_classification: dict
    """
    Structured output from classify_node. Keys:
      query_type: str  — one of:
        factual_lookup | comparison | trend_analysis | calculation |
        risk_analysis | multi_document_reasoning | summarization |
        out_of_scope | unclear
      requires_calculation: bool
      requires_multi_doc: bool
      requires_graph: bool
      expected_output_format: str  — short_answer_with_citation |
        comparison_table | trend_summary | calculated_result |
        risk_bullet_list | multi_part_answer | clarification_needed
      reason: str
    """

    # ── Access control ────────────────────────────────────────────────────────

    access_denied: bool
    """True if access_check_node blocked the query due to role restrictions."""

    access_denial_reason: str
    """Human-readable explanation of access denial."""

    # ── Collection routing ────────────────────────────────────────────────────

    active_collections: list[str]
    """Collection(s) to search this turn."""

    collection_queries: list[CollectionQuery]
    """Per-collection sub-questions from decompose_node."""

    route: str
    """Retrieval path: 'vector' | 'graph' | 'cag'"""

    # ── Retrieval ────────────────────────────────────────────────────────────

    retrieved_chunks: list[ChunkResult]
    """Chunks retrieved in the current turn."""

    reusable_chunks: Annotated[list[ChunkResult], operator.add]
    """Source chunks accumulated across all prior turns."""

    # ── Grade node ────────────────────────────────────────────────────────────

    grade: str
    """'sufficient' | 'insufficient'"""

    facet_grades: list[dict]
    """Per-collection grade results for multi-collection queries."""

    retry_count: int
    """How many grade→retrieve retry loops have run this turn."""

    answerability: str
    """
    'sufficient' | 'insufficient' | 'conflicting'
    More detailed than grade — surfaced in UI trace and done event.
    """

    answerability_reason: str
    """Why the context is insufficient or conflicting."""

    missing_info: list[str]
    """Specific information absent from retrieved context."""

    conflict_detected: bool
    """
    True when retrieved chunks contain conflicting values for the same
    metric/period (e.g., two different revenue figures for FY2024).
    """

    # ── Numeric validation ────────────────────────────────────────────────────

    numeric_validation: dict
    """
    Populated by validate_numbers_node for calculation queries. Structure:
    {
      "metric": str,
      "company": str,
      "periods": [{"year": int, "value": float, "unit": str, "source_id": str}],
      "calculation": {"absolute_change": float, "percentage_change": float, "formula": str}
    }
    Empty dict when validation was not triggered or not needed.
    """

    # ── Conversation history ─────────────────────────────────────────────────

    conversation_history: Annotated[list[ConversationTurn], operator.add]
    """Recent Q&A turns for follow-up rewriting."""

    # ── Generate node ─────────────────────────────────────────────────────────

    answer: str
    """The generated answer, grounded in retrieved source chunks."""

    citations: list[Citation]
    """Source references with structured metadata for the UI."""

    # ── Performance tracking ─────────────────────────────────────────────────

    total_tokens: Annotated[int, operator.add]
    """Cumulative LLM token count across all nodes (prompt + completion)."""

    input_tokens: Annotated[int, operator.add]
    """Cumulative prompt/input tokens across all nodes."""

    output_tokens: Annotated[int, operator.add]
    """Cumulative completion/output tokens across all nodes."""

    step_latencies: Annotated[dict, _merge_dicts]
    """
    Per-node latency in milliseconds.
    Each node returns {"step_latencies": {"node_name": ms}} and the
    _merge_dicts reducer accumulates them so no node overwrites another's time.
    Example: {"rewrite": 45, "classify": 120, "retrieve_vector": 830}
    """

    metrics: dict
    """
    Final aggregated metrics written by generate_node:
    {
      "total_latency_ms": int,
      "retrieve_latency_ms": int,
      "rerank_latency_ms": int,
      "generate_latency_ms": int,
      "graph_latency_ms": int,
      "model": str,
      "input_tokens": int,
      "output_tokens": int,
      "estimated_cost_usd": float,
      "chunk_count": int,
      "citation_count": int
    }
    """
