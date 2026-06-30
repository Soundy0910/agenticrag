"""
backend/api/ingest.py

Ingest pipeline trigger and status endpoints.

Endpoints:
  POST /ingest/trigger  — kick off a full re-index for a collection
  GET  /ingest/status/{job_id}  — poll job progress

Jobs run in a background thread (FastAPI's run_in_executor) so the trigger
endpoint returns immediately with a job_id. The status endpoint polls an
in-memory store. For production, replace _JOBS with Redis or a DB.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import backend.config as cfg
from backend.ingest.embed_index import embed_and_index
from backend.ingest.parse import parse
from backend.storage.base import DocumentMetadata, DocumentSource

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

# In-memory job store — keyed by job_id
_JOBS: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TriggerRequest(BaseModel):
    collection: str = "demo"
    source_type: Literal["local", "azure"] = "azure"
    # For local source: path to the folder to watch
    local_path: str | None = None
    # Optional: restrict to specific doc_ids (empty = all)
    doc_ids: list[str] = []


class TriggerResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["pending", "running", "done", "error"]
    collection: str
    started_at: str
    finished_at: str | None
    docs_processed: int
    vectors_upserted: int
    error: str | None


# ---------------------------------------------------------------------------
# Background ingest worker
# ---------------------------------------------------------------------------

def _run_ingest_job(job_id: str, request: TriggerRequest) -> None:
    """
    Synchronous ingest worker — runs in a thread pool executor.

    For each document from the source:
      1. Fetch content via the storage connector.
      2. Parse into text (Files 4).
      3. Chunk + embed + index into Pinecone (Files 5-6).
    """
    job = _JOBS[job_id]
    job["status"] = "running"

    try:
        source: DocumentSource = _build_source(request)
        doc_metas = source.list_documents(collection=request.collection)
        if request.doc_ids:
            doc_metas = [m for m in doc_metas if m.doc_id in request.doc_ids]

        docs_processed = 0
        vectors_total = 0

        for meta in doc_metas:
            try:
                content = source.fetch_document(meta.doc_id)
                parsed = parse(content, meta)
                if not parsed.ok:
                    logger.warning("ingest: skipping %s — parse produced empty text", meta.doc_id)
                    continue
                count = embed_and_index(parsed, meta)
                vectors_total += count
                docs_processed += 1
                job["docs_processed"] = docs_processed
                job["vectors_upserted"] = vectors_total
                logger.info("ingest: %s → %d vectors", meta.doc_id, count)
            except Exception as exc:
                logger.error("ingest: error on doc %s: %s", meta.doc_id, exc)

        job["status"] = "done"
        job["finished_at"] = _now()

    except Exception as exc:
        logger.exception("ingest job %s failed", job_id)
        job["status"] = "error"
        job["error"] = str(exc)
        job["finished_at"] = _now()


def _build_source(request: TriggerRequest) -> DocumentSource:
    if request.source_type == "local":
        from backend.storage.local import LocalFolderSource
        path = request.local_path or "data/"
        return LocalFolderSource(root=path)
    else:
        from backend.storage.azure_blob import AzureBlobSource
        return AzureBlobSource(
            connection_string=cfg.AZURE_STORAGE_CONNECTION_STRING,
            container=cfg.AZURE_STORAGE_CONTAINER,
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# POST /ingest/trigger
# ---------------------------------------------------------------------------

@router.post("/trigger", response_model=TriggerResponse)
async def trigger_ingest(request: TriggerRequest):
    """
    Start a background re-index job for a collection.

    Returns immediately with a job_id. Poll GET /ingest/status/{job_id}
    to track progress.
    """
    job_id = str(uuid.uuid4())[:8]
    _JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "collection": request.collection,
        "started_at": _now(),
        "finished_at": None,
        "docs_processed": 0,
        "vectors_upserted": 0,
        "error": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_ingest_job, job_id, request)

    return TriggerResponse(
        job_id=job_id,
        status="pending",
        message=f"Ingest job started for collection={request.collection!r}. Poll /ingest/status/{job_id}.",
    )


# ---------------------------------------------------------------------------
# GET /ingest/status/{job_id}
# ---------------------------------------------------------------------------

@router.get("/status/{job_id}", response_model=JobStatus)
async def ingest_status(job_id: str):
    """Poll the status of an ingest job."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    return JobStatus(**job)


# ---------------------------------------------------------------------------
# GET /ingest/jobs  — list all jobs (debug/admin)
# ---------------------------------------------------------------------------

@router.get("/jobs")
async def list_jobs():
    """List all ingest jobs (most recent first)."""
    return list(reversed(list(_JOBS.values())))
