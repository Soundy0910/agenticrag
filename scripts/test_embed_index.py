"""
scripts/test_embed_index.py

End-to-end pipeline test: Azure → parse → chunk (semantic) → embed → Pinecone.
Then queries Pinecone to confirm vectors are searchable.

Run from the project root:  python3 scripts/test_embed_index.py

Required env vars in backend/storage/.env:
  OPENAI_API_KEY=sk-...
  PINECONE_API_KEY=pcsk_...
  PINECONE_INDEX_NAME=agentic-rag          # optional, this is the default
  AZURE_STORAGE_CONNECTION_STRING=...
  AZURE_STORAGE_CONTAINER=...
"""

import logging
import pathlib
import sys

logging.basicConfig(level=logging.WARNING)  # suppress library noise; keep our logs
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

import backend.config as cfg
from backend.storage.azure_blob import AzureBlobSource
from backend.ingest.parse import parse
from backend.ingest.embed_index import embed_and_index, embed_texts, _get_pinecone

# ── 1. Fetch, parse, chunk, embed, upsert ───────────────────────────────────
print("=== Full pipeline: Azure → parse → chunk → embed → Pinecone ===\n")

src = AzureBlobSource(collection="demo")
docs = list(src.list_documents())

for meta in docs:
    print(f"Processing: {meta.doc_id}  ({meta.file_type})")
    content = src.fetch_document(meta.doc_id)
    parsed = parse(content, meta)

    if not parsed.ok:
        print(f"  PARSE FAILED: {parsed.error}")
        continue

    count = embed_and_index(parsed, meta)
    print(f"  Upserted: {count} vectors into namespace={meta.collection!r}")

# ── 2. Query Pinecone to confirm vectors are searchable ──────────────────────
print("\n=== Query test: 'data science education' ===\n")

query = "data science education"
model = cfg.get_embedding_model("demo")
query_vector = embed_texts([query], model)[0]

pc = _get_pinecone()
index = pc.Index(cfg.PINECONE_INDEX_NAME)
results = index.query(
    vector=query_vector,
    top_k=3,
    namespace="demo",
    include_metadata=True,
)

for i, match in enumerate(results.matches):
    m = match.metadata
    print(f"[{i+1}] score={match.score:.4f}  file={m.get('filename')}  is_parent={m.get('is_parent')}")
    print(f"     chunk_id={match.id}")
    preview = m.get("source_text", "")[:150].replace("\n", " ")
    print(f"     text: {preview!r}")
    print()
