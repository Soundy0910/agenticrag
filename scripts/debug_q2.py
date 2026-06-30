"""
scripts/debug_q2.py

Diagnose why Q2 ("What is the candidate's work experience?") returned
"I don't know" despite retrieving EXPERIENCE chunks.

Steps:
  1. Re-run hybrid_search + rerank for Q2 and print full source_text of each chunk.
  2. Dump every chunk in Pinecone whose text contains "experience" (case-insensitive).
  3. Cross-reference to identify: chunking gap vs retrieval miss.

Run from project root:  python3 scripts/debug_q2.py
"""

import pathlib, sys, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.rerank import rerank
from backend.ingest.embed_index import _get_pinecone
import backend.config as cfg

COLLECTION = "demo"
SCOPES     = ["public"]
QUERY      = "What is the candidate's work experience?"
SEP        = "─" * 72


# ── Part 1: What actually fed generate_node ──────────────────────────────────

print(f"\n{'='*72}")
print("  PART 1 — Chunks that fed generate_node (hybrid_search → rerank)")
print(f"{'='*72}")
print(f"  Query: {QUERY!r}\n")

candidates = hybrid_search(QUERY, COLLECTION, SCOPES, top_k=20)
reranked   = rerank(QUERY, candidates, top_n=6)

print(f"  hybrid_search returned {len(candidates)} candidates")
print(f"  rerank kept {len(reranked)} chunks\n")

for i, c in enumerate(reranked, 1):
    print(f"{SEP}")
    print(f"  [{i}] chunk_id={c.chunk_id}")
    print(f"       doc_id={c.doc_id}  filename={c.filename}")
    print(f"       is_parent={c.is_parent}  score={c.score:.4f}")
    print(f"       vector_rank={c.vector_rank}  bm25_rank={c.bm25_rank}")
    print(f"  full source_text ({len(c.source_text)} chars):")
    # Print full text, indented
    for line in c.source_text.splitlines():
        print(f"    {line}")
    print()


# ── Part 2: All Pinecone chunks mentioning "experience" ──────────────────────

print(f"\n{'='*72}")
print("  PART 2 — All Pinecone chunks containing 'experience' (case-insensitive)")
print(f"{'='*72}\n")

pc = _get_pinecone()
idx = pc.Index(cfg.PINECONE_INDEX_NAME)

# Fetch all IDs in the namespace
all_ids = []
for page in idx.list(namespace=COLLECTION):
    all_ids.extend(item.id for item in page.vectors)

print(f"  Total vectors in '{COLLECTION}' namespace: {len(all_ids)}")

# Fetch in batches of 100
BATCH = 100
experience_chunks = []
for start in range(0, len(all_ids), BATCH):
    batch_ids = all_ids[start:start + BATCH]
    resp = idx.fetch(ids=batch_ids, namespace=COLLECTION)
    for vid, vec in resp.vectors.items():
        text = vec.metadata.get("source_text", "")
        if "experience" in text.lower():
            experience_chunks.append((vid, vec.metadata))

print(f"  Chunks containing 'experience': {len(experience_chunks)}\n")

for vid, meta in experience_chunks:
    text = meta.get("source_text", "")
    print(f"{SEP}")
    print(f"  chunk_id : {vid}")
    print(f"  filename : {meta.get('filename', '?')}")
    print(f"  is_parent: {meta.get('is_parent', '?')}")
    print(f"  doc_id   : {meta.get('doc_id', '?')}")
    print(f"  text len : {len(text)} chars")
    print(f"  full text:")
    for line in text.splitlines():
        print(f"    {line}")
    print()


# ── Part 3: Cross-reference — which were missed ───────────────────────────────

print(f"\n{'='*72}")
print("  PART 3 — Cross-reference: retrieved vs all experience chunks")
print(f"{'='*72}\n")

retrieved_ids = {c.chunk_id for c in reranked}
all_exp_ids   = {vid for vid, _ in experience_chunks}

hit  = retrieved_ids & all_exp_ids
miss = all_exp_ids - retrieved_ids

print(f"  Experience chunks in Pinecone : {len(all_exp_ids)}")
print(f"  Retrieved by rerank           : {len(hit)}")
print(f"  Missed by retrieval           : {len(miss)}\n")

if miss:
    print("  MISSED chunks:")
    for vid, meta in experience_chunks:
        if vid in miss:
            text = meta.get("source_text", "")
            print(f"    {vid}  is_parent={meta.get('is_parent')}  len={len(text)}")
            snippet = text[:200].replace("\n", " ")
            print(f"    snippet: {snippet!r}")
            print()
else:
    print("  All experience chunks were retrieved — this is a GENERATION problem, not retrieval.")
