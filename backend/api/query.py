"""
backend/api/query.py

POST /query — run the agent graph and stream node-level trace via SSE.

SSE event format (one JSON object per line after "data: "):
  {"event": "node_complete", "node": "<name>", <node-specific fields>}
  {"event": "done", "answer": "...", "citations": [...], "metrics": {...}, ...}

Live Trace pipeline (11 steps):
  Rewrite → Classify → Router → Access Check → Decompose? →
  Retrieve → Rerank → Grade → Validate Numbers? → Generate → Evaluate

New vs original:
  classify      → query_type, requires_calculation, expected_output_format
  access_check  → access_denied, role, allowed/denied collections
  validate_numbers → validated calculation with formula
  done event    → full metrics dict (latency, tokens, cost, step_latencies)
                   + answerability, conflict_detected, query_classification
"""

import asyncio
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agent.graph import graph
from backend.agent.state import Citation, CollectionQuery, ConversationTurn
from backend.retrieval.hybrid import ChunkResult

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class HistoryItem(BaseModel):
    """One prior turn from the client — only the question is needed for query rewriting."""
    question: str
    rewritten_query: str | None = None


class QueryRequest(BaseModel):
    question: str
    collection: str = "auto"
    allowed_scopes: list[str] = ["public"]
    conversation_history: list[HistoryItem] = []
    role: str = "general"  # admin | finance | legal | general


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_chunks(chunks: list[ChunkResult], max_items: int = 5) -> list[dict]:
    return [
        {
            "chunk_id": c.chunk_id,
            "filename": c.filename,
            "score": round(c.score, 4),
            "is_parent": c.is_parent,
            "preview": c.source_text[:120].replace("\n", " "),
        }
        for c in (chunks or [])[:max_items]
    ]


def _serialise_citation(c: Citation) -> dict:
    """Serialize citation with enriched metadata for the UI."""
    return {
        "chunk_id": c.chunk_id,
        "filename": c.filename,
        "company": c.company,
        "filing_type": c.filing_type,
        "fiscal_year": c.fiscal_year,
        "display_name": c.display_name or c.filename,
        "preview": c.source_text[:120].replace("\n", " "),
    }


def _collection_queries_from_state(state: dict) -> list[CollectionQuery]:
    raw = state.get("collection_queries") or []
    result: list[CollectionQuery] = []
    for item in raw:
        if isinstance(item, CollectionQuery):
            result.append(item)
        elif isinstance(item, dict) and item.get("collection"):
            result.append(CollectionQuery(
                collection=item["collection"],
                sub_question=item.get("sub_question", ""),
            ))
    return result


def _facet_retrievals(state: dict) -> list[dict]:
    """Build per-facet retrieve summaries from merged state."""
    facet_queries = _collection_queries_from_state(state)
    if len(facet_queries) < 2:
        return []
    chunks: list[ChunkResult] = state.get("retrieved_chunks") or []
    return [
        {
            "collection": cq.collection,
            "sub_question": cq.sub_question,
            "chunk_count": len([c for c in chunks if c.collection == cq.collection]),
            "chunks": _serialise_chunks([c for c in chunks if c.collection == cq.collection]),
        }
        for cq in facet_queries
    ]


# ---------------------------------------------------------------------------
# Node event builder
# ---------------------------------------------------------------------------

def _node_event(node_name: str, state: dict) -> dict:
    """Convert merged graph state into a serialisable SSE payload for the Live Trace UI."""
    base = {"event": "node_complete", "node": node_name}
    facet_queries = _collection_queries_from_state(state)
    has_facets = len(facet_queries) >= 2
    step_latencies = state.get("step_latencies") or {}

    if node_name == "rewrite":
        return {
            **base,
            "rewritten_query": state.get("rewritten_query", ""),
            "changed": state.get("rewritten_query", "") != state.get("question", ""),
            "latency_ms": step_latencies.get("rewrite", 0),
        }

    if node_name == "classify":
        qc = state.get("query_classification") or {}
        return {
            **base,
            "query_type": qc.get("query_type", "factual_lookup"),
            "requires_calculation": qc.get("requires_calculation", False),
            "requires_multi_doc": qc.get("requires_multi_doc", False),
            "requires_graph": qc.get("requires_graph", False),
            "expected_output_format": qc.get("expected_output_format", "short_answer_with_citation"),
            "reason": qc.get("reason", ""),
            "latency_ms": step_latencies.get("classify", 0),
        }

    if node_name == "router":
        route = state.get("route", "vector")
        reasons = {
            "vector": "hybrid search (BM25 + semantic + section-type filter)",
            "cag": "small collection — context stuffing",
            "graph": "relational/comparative query → Neo4j",
        }
        active = state.get("active_collections") or []
        return {
            **base,
            "route": route,
            "reason": reasons.get(route, route),
            "active_collections": active,
            "multi_collection": len(active) >= 2,
            "will_decompose": len(active) >= 2 and route == "vector",
            "latency_ms": step_latencies.get("router", 0),
        }

    if node_name == "access_check":
        denied = state.get("access_denied", False)
        return {
            **base,
            "access_denied": denied,
            "role": state.get("user_role", "general"),
            "active_collections": state.get("active_collections") or [],
            "denial_reason": state.get("access_denial_reason", "") if denied else "",
            "latency_ms": step_latencies.get("access_check", 0),
        }

    if node_name == "decompose":
        facets = [
            {"collection": cq.collection, "sub_question": cq.sub_question}
            for cq in facet_queries
        ]
        return {
            **base,
            "facets": facets,
            "facet_count": len(facets),
            "multi_collection": len(facets) >= 2,
            "latency_ms": step_latencies.get("decompose", 0),
        }

    if node_name in ("retrieve_vector", "retrieve_cag", "retrieve_graph"):
        chunks = state.get("retrieved_chunks", [])
        retrieve_key = node_name.replace("retrieve_", "retrieve_")
        payload = {
            **base,
            "chunk_count": len(chunks),
            "chunks": _serialise_chunks(chunks) if not has_facets else [],
            "multi_collection": has_facets,
            "latency_ms": step_latencies.get(retrieve_key, step_latencies.get("retrieve_vector", 0)),
        }
        if has_facets:
            payload["facets"] = _facet_retrievals(state)
        return payload

    if node_name == "grade":
        answerability = state.get("answerability", state.get("grade", ""))
        payload = {
            **base,
            "grade": state.get("grade", ""),
            "answerability": answerability,
            "answerability_reason": state.get("answerability_reason", ""),
            "conflict_detected": state.get("conflict_detected", False),
            "missing_info": state.get("missing_info", []),
            "retry_count": state.get("retry_count", 0),
            "multi_collection": has_facets,
            "latency_ms": step_latencies.get("grade", 0),
        }
        facet_grades = state.get("facet_grades") or []
        if facet_grades:
            payload["facets"] = facet_grades
        return payload

    if node_name == "validate_numbers":
        nv = state.get("numeric_validation") or {}
        calc = nv.get("calculation") or {}
        return {
            **base,
            "metric": nv.get("metric", ""),
            "company": nv.get("company", ""),
            "validated": nv.get("validated", False),
            "period_count": len(nv.get("periods", [])),
            "formula": calc.get("formula", ""),
            "result": (
                f"{calc.get('direction', '')} of {abs(calc.get('percentage_change', 0)):.2f}%"
                if calc else ""
            ),
            "latency_ms": step_latencies.get("validate_numbers", 0),
        }

    if node_name == "generate":
        citations = state.get("citations", [])
        return {
            **base,
            "answer": state.get("answer", ""),
            "citation_count": len(citations),
            "multi_collection": has_facets,
            "latency_ms": step_latencies.get("generate", 0),
            "citations": [_serialise_citation(c) for c in citations],
        }

    # Unknown node — pass through safe scalar fields
    safe = {k: v for k, v in state.items() if isinstance(v, (str, int, float, bool, type(None)))}
    return {**base, **safe}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# SSE generator
# ---------------------------------------------------------------------------

async def _stream_query(req: QueryRequest):
    history = [
        ConversationTurn(
            question=h.question,
            rewritten_query=h.rewritten_query,
            retrieved_chunks=[],
        )
        for h in req.conversation_history
    ]

    initial = {
        "question": req.question,
        "collection": req.collection,
        "allowed_scopes": req.allowed_scopes,
        "user_role": req.role,
        # routing
        "active_collections": [req.collection] if req.collection != "auto" else [],
        "collection_queries": [],
        # retrieval
        "conversation_history": history,
        "reusable_chunks": [],
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
        # numeric validation
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

    final_state: dict = {}

    try:
        async for chunk in graph.astream(initial):
            node_name = next(iter(chunk))
            delta = chunk[node_name]
            final_state.update(delta)
            event = _node_event(node_name, final_state)
            yield _sse(event)
            await asyncio.sleep(0)
    except Exception as exc:
        logger.exception("query stream error")
        yield _sse({"event": "error", "detail": str(exc)})
        return

    # ── Final done event ──────────────────────────────────────────────────────
    citations = final_state.get("citations", [])
    metrics = final_state.get("metrics") or {}
    qc = final_state.get("query_classification") or {}

    yield _sse({
        "event": "done",
        "answer": final_state.get("answer", ""),
        "route": final_state.get("route", ""),
        "active_collections": final_state.get("active_collections", []),
        "rewritten_query": final_state.get("rewritten_query", ""),
        # Classification
        "query_type": qc.get("query_type", "factual_lookup"),
        "requires_calculation": qc.get("requires_calculation", False),
        # Access control
        "access_denied": final_state.get("access_denied", False),
        "user_role": final_state.get("user_role", "general"),
        # Grade + answerability
        "grade": final_state.get("grade", ""),
        "answerability": final_state.get("answerability", ""),
        "conflict_detected": final_state.get("conflict_detected", False),
        "retry_count": final_state.get("retry_count", 0),
        # Citations (enriched)
        "citation_count": len(citations),
        "citations": [_serialise_citation(c) for c in citations],
        "contexts": [c.source_text[:1500].replace("\n", " ") for c in citations],
        # Full metrics
        "metrics": {
            "total_latency_ms": metrics.get("total_latency_ms", 0),
            "retrieve_latency_ms": metrics.get("retrieve_latency_ms", 0),
            "generate_latency_ms": metrics.get("generate_latency_ms", 0),
            "validate_latency_ms": metrics.get("validate_latency_ms", 0),
            "model": metrics.get("model", "gpt-4o-mini"),
            "input_tokens": metrics.get("input_tokens", 0),
            "output_tokens": metrics.get("output_tokens", 0),
            "estimated_cost_usd": metrics.get("estimated_cost_usd", 0.0),
            "chunk_count": metrics.get("chunk_count", 0),
            "citation_count": len(citations),
            "step_latencies": final_state.get("step_latencies") or {},
        },
        # Legacy field (kept for backwards compat with older frontend versions)
        "total_tokens": final_state.get("total_tokens", 0),
        "latency_ms": metrics.get("total_latency_ms", 0),
    })


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/query")
async def query(request: QueryRequest):
    """
    Run a question through the agent graph and stream the node-level trace.

    Returns a Server-Sent Events stream. Each event is a JSON object on a
    `data:` line. Clients should parse events until `event == "done"`.

    Example curl:
        curl -N -X POST http://localhost:8000/api/query \\
          -H 'Content-Type: application/json' \\
          -d '{"question": "What was Microsoft revenue FY2025?", "collection": "auto", "role": "finance"}'
    """
    return StreamingResponse(
        _stream_query(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
