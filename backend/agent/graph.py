"""
backend/agent/graph.py

Wires the agent nodes into a compiled LangGraph StateGraph.

Graph topology:
  rewrite
    │
  router ──────────────────────────────────────────────────────┐
    │                                                          │ (retry, under cap)
    ├── route='vector' → retrieve_vector ──┐                   │
    ├── route='cag'    → retrieve_cag    ──┤→ grade ───────────┘
    └── route='graph'  → retrieve_graph ──┘    │
                                               └── sufficient / cap hit → generate → END

Retry loop:
  grade → 'insufficient' AND retry_count < MAX_RETRIES → retrieve_vector
  grade → 'insufficient' AND retry_count >= MAX_RETRIES → generate
  (The retry goes to retrieve_vector regardless of original route; re-retrieving
  with the same hybrid approach is the safest general fallback.)
"""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

import backend.config as cfg
from backend.agent.nodes import (
    generate_node,
    grade_node,
    retrieve_cag_node,
    retrieve_graph_node,
    retrieve_vector_node,
    rewrite_node,
    router_node,
)
from backend.agent.state import AgentState, ConversationTurn
from backend.retrieval.hybrid import ChunkResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2


# ---------------------------------------------------------------------------
# Conditional edge decision functions
# ---------------------------------------------------------------------------

def _route_decision(state: AgentState) -> str:
    """After router_node: choose which retrieve node to enter."""
    return state.get("route", "vector")


def _grade_decision(state: AgentState) -> str:
    """
    After grade_node: proceed to generate or retry retrieval.

    'generate'        — grade is sufficient, or retry cap reached (force generate
                        with whatever context was retrieved rather than looping forever)
    'retrieve_vector' — grade is insufficient and retries remain
    """
    grade = state.get("grade", "insufficient")
    retry_count = state.get("retry_count", 0)

    if grade == "sufficient" or retry_count >= _MAX_RETRIES:
        if grade != "sufficient":
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
    builder = StateGraph(AgentState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("router", router_node)
    builder.add_node("retrieve_vector", retrieve_vector_node)
    builder.add_node("retrieve_cag", retrieve_cag_node)
    builder.add_node("retrieve_graph", retrieve_graph_node)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    builder.set_entry_point("rewrite")

    # ── Fixed edges ──────────────────────────────────────────────────────────
    builder.add_edge("rewrite", "router")
    builder.add_edge("retrieve_vector", "grade")
    builder.add_edge("retrieve_cag", "grade")
    builder.add_edge("retrieve_graph", "grade")
    builder.add_edge("generate", END)

    # ── Conditional: router → retrieve ───────────────────────────────────────
    builder.add_conditional_edges(
        "router",
        _route_decision,
        {
            "vector": "retrieve_vector",
            "cag": "retrieve_cag",
            "graph": "retrieve_graph",
        },
    )

    # ── Conditional: grade → generate or retry ───────────────────────────────
    builder.add_conditional_edges(
        "grade",
        _grade_decision,
        {
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
) -> dict:
    """
    Run one question through the full agent graph and return the final state.

    Parameters
    ----------
    question : str
        The user's raw question.
    collection : str
        Pinecone namespace to search (maps to a logical document collection).
    allowed_scopes : list[str] | None
        Access identifiers for permission filtering. Defaults to ['public'].
    conversation_history : list[ConversationTurn] | None
        Prior turns from this session. Pass the list from a previous run_query
        result to enable follow-up rewriting. None for the first turn.
    reusable_chunks : list[ChunkResult] | None
        Source chunks accumulated in prior turns. Passed in so the generate
        node can draw on earlier-retrieved context without re-fetching.

    Returns
    -------
    dict
        The final LangGraph state after all nodes have run. Key fields:
          state['answer']               — the generated answer string
          state['citations']            — list of Citation objects
          state['route']                — which retrieval path was taken
          state['rewritten_query']      — the standalone rewrite
          state['conversation_history'] — accumulated turns (pass to next call)
          state['reusable_chunks']      — accumulated source chunks
    """
    initial: dict = {
        "question": question,
        "collection": collection,
        "allowed_scopes": allowed_scopes or ["public"],
        "active_collections": [collection] if collection != "auto" else [],
        "conversation_history": conversation_history or [],
        "reusable_chunks": reusable_chunks or [],
        "messages": [],
        "rewritten_query": "",
        "route": "vector",
        "retrieved_chunks": [],
        "grade": "",
        "retry_count": 0,
        "answer": "",
        "citations": [],
    }
    return graph.invoke(initial)
