"""
backend/main.py

FastAPI application entry point.

Run:
  uvicorn backend.main:app --reload --port 8000

API layout:
  POST   /api/query                   — SSE agent trace stream
  GET    /api/documents               — list indexed documents
  POST   /api/documents/upload        — upload + ingest a file
  DELETE /api/documents/{doc_id}      — remove document vectors
  POST   /api/ingest/trigger          — trigger batch re-index
  GET    /api/ingest/status/{job_id}  — poll ingest job
  GET    /api/ingest/jobs             — list all ingest jobs
  GET    /healthz                     — liveness probe
"""

import pathlib
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env before any backend imports so config.py reads the right values
load_dotenv(pathlib.Path(__file__).parent / "storage" / ".env")

from backend.api import query as query_router
from backend.api import documents as documents_router
from backend.api import ingest as ingest_router
from backend.api import eval as eval_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the compiled graph on startup (avoids cold-start on first request)
    from backend.agent.graph import graph  # noqa: F401
    yield


app = FastAPI(
    title="Agentic RAG Knowledge Platform",
    description="LangGraph-powered RAG with hybrid search, graph retrieval, and live trace streaming.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins in development; tighten for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(query_router.router,     prefix="/api")
app.include_router(documents_router.router, prefix="/api")
app.include_router(ingest_router.router,    prefix="/api")
app.include_router(eval_router.router,      prefix="/api")


# ---------------------------------------------------------------------------
# Liveness probe
# ---------------------------------------------------------------------------

@app.get("/healthz", tags=["ops"])
async def healthz():
    return {"status": "ok"}
