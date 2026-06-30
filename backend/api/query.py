"""
backend/api/query.py

POST /query — run the agent graph and stream node-level trace via SSE.

SSE event format (one JSON object per line after "data: "):
  {"event": "node_complete", "node": "<name>", <node-specific fields>}
  {"event": "done", "answer": "...", "citations": [...], "route": "..."}

The live-trace UI receives these events as nodes complete and renders:
  rewrite  → rewritten query (or "unchanged")
  router   → route chosen + reason
  retrieve → chunk count + previews
  grade    → sufficient/insufficient + retry count
  generate → final answer + citations

Streaming uses LangGraph's astream() which yields one state-delta dict
per node as it completes — no polling, no full-graph re-serialisation.
"""

import asyncio
import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.agent.graph import graph
from backend.agent.state import ConversationTurn
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
    collection: str = "demo"
    allowed_scopes: list[str] = ["public"]
    conversation_history: list[HistoryItem] = []


# ---------------------------------------------------------------------------
# Node-output serialiser
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


def _node_event(node_name: str, delta: dict) -> dict:
    """Convert a LangGraph state-delta dict into a serialisable SSE payload."""
    base = {"event": "node_complete", "node": node_name}

    if node_name == "rewrite":
        return {**base,
                "rewritten_query": delta.get("rewritten_query", ""),
                "changed": delta.get("rewritten_query", "") != delta.get("question", "")}

    if node_name == "router":
        route = delta.get("route", "vector")
        reasons = {"vector": "hybrid search (default)", "cag": "small collection — context stuffing", "graph": "relational/comparative query"}
        return {**base, "route": route, "reason": reasons.get(route, route)}

    if node_name in ("retrieve_vector", "retrieve_cag", "retrieve_graph"):
        chunks = delta.get("retrieved_chunks", [])
        return {**base, "chunk_count": len(chunks), "chunks": _serialise_chunks(chunks)}

    if node_name == "grade":
        return {**base, "grade": delta.get("grade", ""), "retry_count": delta.get("retry_count", 0)}

    if node_name == "generate":
        citations = delta.get("citations", [])
        return {
            **base,
            "answer": delta.get("answer", ""),
            "citation_count": len(citations),
            "citations": [
                {"chunk_id": c.chunk_id, "filename": c.filename,
                 "preview": c.source_text[:120].replace("\n", " ")}
                for c in citations
            ],
        }

    # Any other node — emit raw serialisable fields only
    safe = {k: v for k, v in delta.items() if isinstance(v, (str, int, float, bool, type(None)))}
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
        "active_collections": [req.collection] if req.collection != "auto" else [],
        "conversation_history": history,
        "reusable_chunks": [],
        "messages": [],
        "rewritten_query": "",
        "route": "vector",
        "retrieved_chunks": [],
        "grade": "",
        "retry_count": 0,
        "answer": "",
        "citations": [],
    }

    final_state: dict = {}

    try:
        async for chunk in graph.astream(initial):
            node_name = next(iter(chunk))
            delta = chunk[node_name]
            final_state.update(delta)
            event = _node_event(node_name, delta)
            yield _sse(event)
            # Yield control so the event loop can flush to the client
            await asyncio.sleep(0)
    except Exception as exc:
        logger.exception("query stream error")
        yield _sse({"event": "error", "detail": str(exc)})
        return

    # Final "done" event with full answer
    citations = final_state.get("citations", [])
    yield _sse({
        "event": "done",
        "answer": final_state.get("answer", ""),
        "route": final_state.get("route", ""),
        "rewritten_query": final_state.get("rewritten_query", ""),
        "grade": final_state.get("grade", ""),
        "retry_count": final_state.get("retry_count", 0),
        "citation_count": len(citations),
        "citations": [
            {"chunk_id": c.chunk_id, "filename": c.filename,
             "preview": c.source_text[:120].replace("\n", " ")}
            for c in citations
        ],
        # Full source texts for inline RAGAS eval (truncated to 500 chars each)
        "contexts": [c.source_text[:500].replace("\n", " ") for c in citations],
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
          -d '{"question": "What AWS certifications?", "collection": "demo"}'
    """
    return StreamingResponse(
        _stream_query(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx proxy buffering
            "Connection": "keep-alive",
        },
    )
