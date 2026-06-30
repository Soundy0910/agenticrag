"""
scripts/test_rerank.py

Shows before/after ordering: hybrid_search → rerank.
Run from the project root:  python3 scripts/test_rerank.py
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.retrieval.hybrid import hybrid_search
from backend.retrieval.rerank import rerank

COLLECTION = "demo"
SCOPES     = ["public"]
QUERY      = "AWS certifications and cloud skills"


def show(label: str, results: list, *, show_score_label: str = "score") -> None:
    print(f"\n{'─'*62}")
    print(f"  {label}")
    print(f"{'─'*62}")
    for i, r in enumerate(results, 1):
        v = f"v={r.vector_rank}" if r.vector_rank else "    "
        b = f"b={r.bm25_rank}"   if r.bm25_rank  else "    "
        print(f"  #{i}  [{v} {b}]  {show_score_label}={r.score:.4f}  {'PARENT' if r.is_parent else 'child ':6}")
        print(f"       {r.source_text[:100].replace(chr(10),' ')!r}")


print(f"\nQuery: {QUERY!r}\n")

# 1. Hybrid — fetch top 20 candidates
candidates = hybrid_search(QUERY, COLLECTION, SCOPES, top_k=20)
show("Hybrid (before rerank) — top 10", candidates[:10], show_score_label="rrf")

# 2. Rerank top 20 → return best 5
reranked = rerank(QUERY, candidates, top_n=5)
show("After Cohere Rerank — top 5", reranked, show_score_label="cohere")

print("\n--- What changed ---")
hybrid_ids  = [r.chunk_id for r in candidates[:5]]
rerank_ids  = [r.chunk_id for r in reranked]
for i, r in enumerate(reranked, 1):
    prev = hybrid_ids.index(r.chunk_id) + 1 if r.chunk_id in hybrid_ids else ">10"
    arrow = f"#{prev} → #{i}"
    print(f"  {arrow:10}  {r.source_text[:80].replace(chr(10),' ')!r}")
