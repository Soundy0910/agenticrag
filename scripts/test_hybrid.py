"""
scripts/test_hybrid.py

Compare vector-only vs BM25-only vs hybrid for two query types:
  1. Semantic query   — "educational background"
  2. Exact-term query — "1943" (specific year in the CSV data)

Run from the project root:  python3 scripts/test_hybrid.py
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.retrieval.hybrid import (
    hybrid_search, _vector_search, _bm25_search, _rrf_fusion
)

COLLECTION = "demo"
SCOPES = ["public"]
TOP_K = 4


def show(label: str, results: list) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    if not results:
        print("  (no results)")
        return
    for r in results[:TOP_K]:
        v = f"v={r.vector_rank}" if r.vector_rank else "     "
        b = f"b={r.bm25_rank}"   if r.bm25_rank  else "     "
        print(f"  [{v} {b}]  score={r.score:.4f}  {'PARENT' if r.is_parent else 'child ':6}  {r.filename}")
        print(f"           {r.source_text[:110].replace(chr(10),' ')!r}")


def compare(query: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  QUERY: {query!r}")
    print(f"{'═'*60}")

    candidate_k = max(TOP_K * 3, 20)
    vec   = _vector_search(query, COLLECTION, SCOPES, candidate_k)
    bm25  = _bm25_search(query, COLLECTION, candidate_k)
    combo = _rrf_fusion(vec, bm25, TOP_K)

    show("Vector-only", vec[:TOP_K])
    show("BM25-only",   bm25[:TOP_K])
    show("Hybrid (RRF)", combo)


compare("educational background")
compare("1943")
compare("AWS certified")
