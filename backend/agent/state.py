"""
backend/agent/state.py

LangGraph agent state — the shared object passed between every node in the graph.

ARCHITECTURE CONSTRAINT (from ARCHITECTURE.md, section 5.9 and 5.10):
  State MUST carry:
    - The user's questions (current + recent history) — for query rewriting.
    - Retrieved source chunks (current + reusable from prior turns) — grounded facts.
  State MUST NOT carry:
    - Prior generated answers — using a past LLM generation as input to a new
      generation launders hallucinations forward. If the first answer was wrong,
      the second answer compounds the error. Source chunks are safe to reuse
      because they come from the document store, not from model generation.

LANGGRAPH REDUCER SEMANTICS:
  Fields annotated with `Annotated[list[X], operator.add]` are *additive*:
  when a node returns a value for that field, LangGraph appends it to the
  existing list rather than replacing it. This is used for:
    - conversation_history: accumulates turns across the session.
    - reusable_chunks: accumulates source chunks retrieved in prior turns so
      the generate node can draw on them without re-fetching.
  All other fields are last-write-wins: a node's returned value replaces the
  current state value for that field.
"""

import operator
from dataclasses import dataclass, field
from typing import Annotated

from langgraph.graph import MessagesState

from backend.retrieval.hybrid import ChunkResult


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """
    One source reference attached to the generated answer.

    The generate node populates citations so the UI can render
    "Source: filename, page/section" links next to each claim.
    source_text is stored verbatim so the UI can show the exact
    passage without a second retrieval call.
    """
    chunk_id: str
    doc_id: str
    filename: str
    collection: str
    source_text: str       # verbatim chunk text — shown in the citation card


@dataclass
class ConversationTurn:
    """
    One complete Q&A turn, stored in history for query rewriting.

    WHY NO `answer` FIELD:
      The rewrite node needs to know what the user asked before so it can
      resolve pronouns and follow-ups ("tell me more about that").  It reads
      prior *questions* to produce a standalone rewrite. It does NOT need
      prior generated answers — and storing answers here would risk the rewrite
      node treating a past hallucination as factual context.

    retrieved_chunks carries the source passages from this turn. These ARE
    safe to surface in later turns because they are direct document quotes,
    not model-generated text.
    """
    question: str                        # original user question this turn
    rewritten_query: str | None          # standalone rewrite (None for turn 1)
    retrieved_chunks: list[ChunkResult]  # source passages retrieved this turn
    # NOTE: no `answer` field — see module docstring


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class AgentState(MessagesState):
    """
    Shared state object threaded through every LangGraph node.

    MessagesState base adds a `messages` field (Annotated list with add_messages
    reducer) for the conversation message history. We extend it with all the
    fields the RAG agent needs.

    Node responsibilities:
      rewrite_node   → writes rewritten_query
      router_node    → writes collection, route
      retrieve_node  → writes retrieved_chunks; appends to reusable_chunks
      grade_node     → writes grade, retry_count
      generate_node  → writes answer, citations
      (all nodes)    → read question, collection, allowed_scopes
    """

    # ---- Input (set at conversation start, not modified by nodes) ----------

    question: str
    """The user's current raw question, as typed."""

    collection: str
    """
    Pinecone namespace to search. Set by the caller before invoking the graph,
    or overridden by the router node if it detects a collection mismatch.
    """

    allowed_scopes: list[str]
    """
    Access identifiers for the requesting user/role. Passed to hybrid_search()
    to filter out chunks the user is not permitted to see.
    E.g. ['public'] for unauthenticated access, ['user-123', 'team-a'] for
    a logged-in user.
    """

    # ---- Rewrite node output -----------------------------------------------

    rewritten_query: str
    """
    The standalone, self-contained version of `question` produced by the
    rewrite node. Follow-up questions like "What about their liabilities?"
    become "What are Company X's total liabilities in FY2023?" — fully
    resolvable without the prior conversation. This is what retrieval uses.
    """

    # ---- Collection routing ------------------------------------------------

    active_collections: list[str]
    """
    The collection(s) to search this turn. Set by the router when collection
    is 'auto' (cross-collection mode) or explicitly by the caller.
    Single entry → standard path. Two entries → parallel search + RRF merge.
    Always contains at least [collection] as a fallback.
    """

    # ---- Router node output ------------------------------------------------

    route: str
    """
    Retrieval path chosen by the router node.
    One of: 'vector' | 'graph' | 'cag'
      vector — standard hybrid search (default for factual lookups)
      graph  — Neo4j graph retrieval (relational/comparative questions)
      cag    — context-stuffing (entire document set fits in context window)
    """

    # ---- Retrieval ---------------------------------------------------------

    retrieved_chunks: list[ChunkResult]
    """
    Chunks retrieved in the CURRENT turn by the retrieve node.
    Replaced on every retrieve call (not additive). The grade node reads
    these to decide if retrieval was sufficient.
    """

    reusable_chunks: Annotated[list[ChunkResult], operator.add]
    """
    Source chunks accumulated across ALL prior turns in this session.

    WHY REUSE CHUNKS (not re-retrieve):
      If the user asks two questions about the same section of a document,
      re-retrieval costs latency + embedding API calls. The chunks already
      fetched are source-grounded and safe to carry forward.

    WHY NOT REUSE ANSWERS:
      If the generate node produced an answer and it was subtly wrong, carrying
      that answer into the next turn's context would compound the error. Chunks
      are direct document quotes — reusing them is safe.

    Annotated with operator.add so each retrieve call appends new chunks
    rather than replacing prior turns' chunks.
    """

    # ---- Grade node output -------------------------------------------------

    grade: str
    """
    Retrieval quality decision from the grade node.
    'sufficient'   → proceed to generate
    'insufficient' → retry with a reformulated query (the self-correction loop)
    """

    retry_count: int
    """
    How many grade→retrieve retry loops have run this turn. The graph's
    conditional edge caps this (typically at 2) to prevent infinite loops
    when the corpus genuinely doesn't contain the answer.
    """

    # ---- Conversation history ----------------------------------------------

    conversation_history: Annotated[list[ConversationTurn], operator.add]
    """
    Recent turns (question + retrieved chunks) accumulated across the session.
    The rewrite node reads the last N entries to resolve pronouns and
    follow-ups. Annotated with operator.add so each completed turn appends.

    Only questions and retrieved chunks are stored here — see ConversationTurn
    for the explicit design note on why generated answers are excluded.
    """

    # ---- Generate node output ----------------------------------------------

    answer: str
    """The generated answer, grounded in retrieved_chunks + reusable_chunks."""

    citations: list[Citation]
    """
    Source references for the answer. The generate node populates one Citation
    per source chunk it drew on. The API layer serialises these for the UI's
    SourceCitations panel and the LiveTrace transparency view.
    """
