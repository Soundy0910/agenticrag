"""
backend/api/documents.py

Document management endpoints — list, upload, delete.
All operations go through the storage layer (Files 1-3) and Pinecone.

Endpoints:
  GET    /documents                — list all documents in a collection
  POST   /documents/upload         — upload a file and run the full ingest pipeline
  DELETE /documents/{doc_id}       — delete from Pinecone (storage deletion is manual)
"""

import asyncio
import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

import backend.config as cfg
from backend.ingest.embed_index import _get_pinecone, embed_and_index
from backend.ingest.parse import parse
from backend.storage.base import DocumentMetadata

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    collection: str
    file_type: str
    vector_count: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    collection: str
    vectors_upserted: int


# ---------------------------------------------------------------------------
# GET /documents
# ---------------------------------------------------------------------------

@router.get("", response_model=list[DocumentInfo])
async def list_documents(collection: str = "demo"):
    """
    List all documents indexed in a Pinecone namespace (collection).

    Groups child vectors by doc_id to produce one entry per source document.
    Returns filename, collection, file_type, and the number of indexed vectors.
    """
    def _fetch():
        pc = _get_pinecone()
        idx = pc.Index(cfg.PINECONE_INDEX_NAME)
        all_ids = []
        for page in idx.list(namespace=collection):
            all_ids.extend(item.id for item in page.vectors)
        if not all_ids:
            return []

        docs: dict[str, dict] = {}
        batch = 100
        for start in range(0, len(all_ids), batch):
            resp = idx.fetch(ids=all_ids[start:start + batch], namespace=collection)
            for vid, vec in resp.vectors.items():
                meta = vec.metadata
                if meta.get("is_parent"):
                    continue  # count only children to avoid double-counting
                doc_id = meta.get("doc_id", vid)
                if doc_id not in docs:
                    docs[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get("filename", doc_id),
                        "collection": collection,
                        "file_type": meta.get("file_type", ""),
                        "vector_count": 0,
                    }
                docs[doc_id]["vector_count"] += 1
        return list(docs.values())

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, _fetch)
    return results


# ---------------------------------------------------------------------------
# POST /documents/upload
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form("demo"),
    access_scope: str = Form("public"),
):
    """
    Upload a document and run the full ingest pipeline (parse → chunk → embed → index).

    Accepts any file type supported by the parse layer (pdf, docx, pptx, txt, md, csv, xlsx).
    The file is saved to a temp directory, parsed, chunked, embedded, and indexed
    into Pinecone under the specified collection namespace.

    Returns the doc_id, filename, and number of vectors upserted.
    """
    from backend.storage._utils import is_supported, file_type_from_name

    filename = file.filename or "upload"
    if not is_supported(filename):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {Path(filename).suffix}. "
                   f"Supported: pdf, docx, pptx, txt, md, csv, xlsx",
        )

    # Save to temp file (embed_and_index needs a filesystem path for unstructured)
    suffix = Path(filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    metadata = DocumentMetadata(
        doc_id=filename,
        filename=filename,
        collection=collection,
        source_type="upload",
        file_type=file_type_from_name(filename),
        embedding_model=cfg.get_embedding_model(collection),
        access_scope=[s.strip() for s in access_scope.split(",")],
    )

    def _run_ingest():
        from backend.ingest.parse import ParsedDocument
        parsed = parse(tmp_path, metadata)
        if not parsed.ok:
            raise ValueError(f"Parse failed for {filename}: empty text")
        count = embed_and_index(parsed, metadata)
        tmp_path.unlink(missing_ok=True)
        return count

    loop = asyncio.get_event_loop()
    try:
        vectors_upserted = await loop.run_in_executor(None, _run_ingest)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        logger.exception("upload ingest failed for %s", filename)
        raise HTTPException(status_code=500, detail=str(exc))

    return UploadResponse(
        doc_id=filename,
        filename=filename,
        collection=collection,
        vectors_upserted=vectors_upserted,
    )


# ---------------------------------------------------------------------------
# DELETE /documents/{doc_id}
# ---------------------------------------------------------------------------

@router.delete("/{doc_id}")
async def delete_document(doc_id: str, collection: str = "demo"):
    """
    Delete all Pinecone vectors for a document from the given collection.

    Finds all vectors whose metadata.doc_id matches, then deletes them.
    Does not delete the source file from Azure Blob / local storage.
    """
    def _delete():
        pc = _get_pinecone()
        idx = pc.Index(cfg.PINECONE_INDEX_NAME)
        all_ids = []
        for page in idx.list(namespace=collection):
            all_ids.extend(item.id for item in page.vectors)
        if not all_ids:
            return 0

        to_delete = []
        batch = 100
        for start in range(0, len(all_ids), batch):
            resp = idx.fetch(ids=all_ids[start:start + batch], namespace=collection)
            for vid, vec in resp.vectors.items():
                if vec.metadata.get("doc_id") == doc_id:
                    to_delete.append(vid)

        if to_delete:
            idx.delete(ids=to_delete, namespace=collection)
        return len(to_delete)

    loop = asyncio.get_event_loop()
    deleted = await loop.run_in_executor(None, _delete)

    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No vectors found for doc_id={doc_id!r} in collection={collection!r}")

    return {"deleted": deleted, "doc_id": doc_id, "collection": collection}
