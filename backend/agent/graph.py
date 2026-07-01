"""
backend/agent/graph.py

Wires the agent nodes into a compiled LangGraph StateGraph.

Graph topology (upgraded):

  rewrite → classify → router → access_check
                                     │
              ┌──── denied ──────────┘
              │                     │ allowed
              │             ┌───────┴──────────┐
              │         decompose        (single-collection)
              │             │             ├── retrieve_vector
              │             └────────┐    ├── retrieve_cag
              │                     ↓    └── retrieve_graph
              │                   grade
              │                    │
              │            ┌───────┴────────────┐
              │         sufficient           insufficient
              │            │                    │
              │     [validate_numbers?]       (retry or cap)
              │            │
              └──────→ generate → END

New nodes vs original:
  classify        — query type classification (factual/comparison/risk/etc.)
  access_check    — RBAC: blocks queries to restricted collections
  validate_numbers — deterministic numeric extraction + Python calculation

New edges:
  access_check → generate       (when access_denied=True)
  grade → validate_numbers      (when requires_calculation=True and sufficient)
  validate_numbers → generate
"""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

import backend.config as cfg
from backend.agent.nodes import (
    access_check_node,
    classify_node,
    decompose_node,
    generate_node,
    grade_node,
    retrieve_cag_node,
    retrieve_graph_node,
    retrieve_vector_node,
    rewrite_node,
    router_node,
    validate_numbers_node,
)
from backend.agent.state import AgentState, ConversationTurn
from backend.retrieval.hybrid import ChunkResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Conditional edge decision functions
# ---------------------------------------------------------------------------

def _grade_decision(state: AgentState) -> str:
    """
    After grade_node: proceed to generate/validate or retry retrieval.

    'validate_numbers' — grade sufficient AND query requires calculation
    'generate'         — grade sufficient (no calculation), or retry cap hit
    'retrieve_vector'  — grade insufficient and retries remain
    """
    grade = state.get("grade", "insufficient")
    retry_count = state.get("retry_count", 0)
    qc: dict = state.get("query_classification") or {}
    requires_calc = qc.get("requires_calculation", False)

    if grade == "sufficient":
        if requires_calc:
            return "validate_numbers"
        return "generate"

    if retry_count >= _MAX_RETRIES:
        logger.warning(
            "grade: retry cap (%d) reached — forcing generate with current context",
            retry_count,
        )
        return "generate"

    return "retrieve_vector"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> Any:
    """
    Clean graph build — uses two separate conditional edges to handle the
    access_check → denied→generate / allowed→[route] branch cleanly.
    """
    builder = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("classify", classify_node)
    builder.add_node("router", router_node)
    builder.add_node("access_check", access_check_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve_vector", retrieve_vector_node)
    builder.add_node("retrieve_cag", retrieve_cag_node)
    builder.add_node("retrieve_graph", retrieve_graph_node)
    builder.add_node("grade", grade_node)
    builder.add_node("validate_numbers", validate_numbers_node)
    builder.add_node("generate", generate_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    builder.set_entry_point("rewrite")

    # ── Fixed edges ──────────────────────────────────────────────────────────
    builder.add_edge("rewrite", "classify")
    builder.add_edge("classify", "router")
    builder.add_edge("router", "access_check")
    builder.add_edge("decompose", "retrieve_vector")
    builder.add_edge("retrieve_vector", "grade")
    builder.add_edge("retrieve_cag", "grade")
    builder.add_edge("retrieve_graph", "grade")
    builder.add_edge("validate_numbers", "generate")
    builder.add_edge("generate", END)

    # ── Conditional: access_check → denied→generate OR route decision ────────
    def _access_then_route(state: AgentState) -> str:
        if state.get("access_denied"):
            logger.info("access_check: denied — short-circuit to generate")
            return "generate"
        # Inline route decision when access is allowed
        route = state.get("route", "vector")
        if route != "vector":
            return route
        active = state.get("active_collections") or []
        if len(active) >= 2:
            return "decompose"
        return "vector"

    builder.add_conditional_edges(
        "access_check",
        _access_then_route,
        {
            "generate": "generate",
            "decompose": "decompose",
            "vector": "retrieve_vector",
            "cag": "retrieve_cag",
            "graph": "retrieve_graph",
        },
    )

    # ── Conditional: grade → validate_numbers / generate / retry ─────────────
    builder.add_conditional_edges(
        "grade",
        _grade_decision,
        {
            "validate_numbers": "validate_numbers",
            "generate": "generate",
            "retrieve_vector": "retrieve_vector",
        },
    )

    return builder.compile()


# Module-level compiled graph — built once on import.
graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_query(
    question: str,
    collection: str,
    allowed_scopes: list[str] | None = None,
    conversation_history: list[ConversationTurn] | None = None,
    reusable_chunks: list[ChunkResult] | None = None,
    user_role: str = "general",
) -> dict:
    """
    Run one question through the full agent graph and return the final state.

    Parameters
    ----------
    question : str
        The user's raw question.
    collection : str
        Pinecone namespace to search. Use 'auto' for automatic routing.
    allowed_scopes : list[str] | None
        Access identifiers for permission filtering. Defaults to ['public'].
    conversation_history : list[ConversationTurn] | None
        Prior turns for follow-up rewriting. None for the first turn.
    reusable_chunks : list[ChunkResult] | None
        Accumulated source chunks from prior turns.
    user_role : str
        RBAC role: 'admin' | 'finance' | 'legal' | 'general'.
        Controls which collections access_check_node allows. Default: 'general'.

    Returns
    -------
    dict
        Final LangGraph state. Key fields:
          state['answer']                — generated answer string
          state['citations']             — list of Citation objects with display_name
          state['route']                 — retrieval path taken
          state['rewritten_query']       — standalone rewrite
          state['query_classification']  — {query_type, requires_calculation, ...}
          state['access_denied']         — True if RBAC blocked the query
          state['answerability']         — sufficient | insufficient | conflicting
          state['conflict_detected']     — True if conflicting values found
          state['numeric_validation']    — extraction + calculation dict
          state['metrics']               — latency, tokens, cost dict
          state['conversation_history']  — accumulated turns
          state['reusable_chunks']       — accumulated source chunks
    """
    initial: dict = {
        "question": question,
        "collection": collection,
        "allowed_scopes": allowed_scopes or ["public"],
        "user_role": user_role,
        # routing
        "active_collections": [collection] if collection != "auto" else [],
        "collection_queries": [],
        # retrieval
        "conversation_history": conversation_history or [],
        "reusable_chunks": reusable_chunks or [],
        "messages": [],
        "rewritten_query": "",
        "route": "vector",
        "retrieved_chunks": [],
        # classification + access
        "query_classification": {},
        "access_denied": False,
        "access_denial_reason": "",
        # grading
        "grade": "",
        "facet_grades": [],
        "retry_count": 0,
        "answerability": "",
        "answerability_reason": "",
        "missing_info": [],
        "conflict_detected": False,
        # numeric
        "numeric_validation": {},
        # output
        "answer": "",
        "citations": [],
        # performance
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "step_latencies": {},
        "metrics": {},
    }
    return graph.invoke(initial)
