"""
backend/retrieval/hybrid.py

Hybrid retrieval: vector search + BM25 keyword search, fused with RRF.

WHY HYBRID BEATS PURE VECTOR:
  Vector (semantic) search embeds query and chunks into the same latent space
  and retrieves by cosine similarity. This is excellent at catching meaning
  ("tell me about revenue growth" → finds "top-line expansion" chunks) but
  it *drifts* on exact terms. Embeddings compress meaning; a ticker like "NVDA",
  a line-item like "goodwill impairment", or a specific number like "8192" are
  just tokens — the model may not map them to a region that's close to the
  query's region, especially for rare terms never seen together in training.

  BM25 (Best Match 25) is a classic TF-IDF-style bag-of-words scorer. It has
  zero understanding of meaning but scores exact token overlap perfectly. A
  query "1943" will find every chunk containing "1943" even if the semantic
  distance is high. BM25 is cheap, deterministic, and requires no embeddings.

  Hybrid = vector catches semantics, BM25 catches exact terms. The fusion step
  combines both ranked lists so a chunk that appears in both gets boosted —
  the two signals are complementary.

FUSION METHOD — Reciprocal Rank Fusion (RRF):
  RRF score(chunk) = Σ_i  1 / (k + rank_i(chunk))

  where:
    k = 60  (constant from the original RRF paper; dampens high-rank advantage)
    rank_i = position of this chunk in result list i (1 = top)

  Why RRF over weighted score merge:
    - Vector cosine scores (0–1) and BM25 scores (unbounded float) live on
      completely different scales. Normalising and weighting them requires
      tuning another hyperparameter.
    - RRF uses only rank, not raw scores, so scale differences are irrelevant.
    - RRF is shown to match or beat weighted fusion empirically while requiring
      zero tuning. (Cormack, Clarke & Buettcher, SIGIR 2009.)
    - Chunks that appear in *both* lists get two RRF contributions; chunks that
      appear in only one get one. Agreement between signals = boost.

BM25 SCALABILITY NOTE (production path):
  The current implementation builds a BM25 index by fetching all chunk texts
  from Pinecone into memory and indexing them with rank_bm25. This works fine
  at demo scale (hundreds of chunks, rebuilt per process start and cached).

  At production scale (millions of chunks) the right move is Pinecone's native
  hybrid/sparse-dense search:
    - Encode queries and documents as sparse vectors (BM25 term weights) in
      addition to dense vectors.
    - Upsert sparse+dense together; query with alpha parameter controlling the
      vector/keyword blend.
    - This runs entirely inside Pinecone — no separate BM25 index, no memory
      overhead, no round-trip to fetch the full corpus.
  The alpha blend in Pinecone hybrid is equivalent to our RRF weight parameter.
  Migrating is a config change in embed_index.py (add sparse encoding at upsert)
  and here (switch to sparse+dense query instead of two separate queries).
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from pinecone import Pinecone
from rank_bm25 import BM25Okapi

import backend.config as cfg
from backend.ingest.embed_index import embed_texts, _get_pinecone

logger = logging.getLogger(__name__)

# RRF constant — 60 is the value from the original paper. Lower k amplifies
# the advantage of top-ranked results; higher k flattens the curve.
_RRF_K = 60


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ChunkResult:
    """
    One retrieved chunk, ready for the reranker or the agent's generate node.

    Fields used by the agent:
      source_text  — the chunk's verbatim text (citations + reranking input)
      parent_id    — retriever fetches the parent by this ID for LLM context
      score        — final RRF score (higher = more relevant)
      vector_rank  — position in the vector result list (1-based; None if absent)
      bm25_rank    — position in the BM25 result list (1-based; None if absent)

    Having both ranks visible lets the caller see which signal drove the result:
      vector_rank=1, bm25_rank=None → pure semantic match
      vector_rank=None, bm25_rank=1 → exact-term match, would be missed by vector-only
      vector_rank=2, bm25_rank=3    → strong agreement from both signals
    """
    chunk_id: str
    parent_id: str | None
    doc_id: str
    filename: str
    collection: str
    source_text: str
    is_parent: bool
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# BM25 corpus cache — built lazily, one per collection per process
# ---------------------------------------------------------------------------

@dataclass
class _CorpusEntry:
    chunk_id: str
    tokens: list[str]
    metadata: dict[str, Any]


# module-level cache: collection name → (BM25Okapi, list[_CorpusEntry])
_bm25_cache: dict[str, tuple[BM25Okapi, list[_CorpusEntry]]] = {}


def _tokenize(text: str) -> list[str]:
    """
    Lowercase word tokeniser for BM25. Strips punctuation so "AWS," and "AWS"
    match, and numbers stay intact so "1943" matches "1943".
    """
    return re.findall(r"\b\w+\b", text.lower())


def _build_bm25_index(collection: str) -> tuple[BM25Okapi, list[_CorpusEntry]]:
    """
    Fetch all chunk texts from Pinecone for this collection and build a BM25 index.

    Strategy: use index.list() to get all IDs in the namespace, then batch-fetch
    their metadata. This loads the full corpus into memory — acceptable at demo
    scale (hundreds of chunks). See module docstring for the production path.

    The result is cached in _bm25_cache so subsequent queries hit memory only.
    """
    if collection in _bm25_cache:
        return _bm25_cache[collection]

    pc = _get_pinecone()
    index = pc.Index(cfg.PINECONE_INDEX_NAME)

    logger.info("Building BM25 index for collection=%r ...", collection)

    # list() returns a generator of ID pages; chain them into a flat list.
    all_ids: list[str] = []
    for page in index.list(namespace=collection):
        # list() yields ListResponse objects; extract string IDs from .vectors
        all_ids.extend(item.id for item in page.vectors)

    if not all_ids:
        logger.warning("No vectors found in namespace=%r", collection)
        _bm25_cache[collection] = (None, [])   # _bm25_search guards on empty corpus
        return _bm25_cache[collection]

    # Fetch in batches (Pinecone fetch limit: 1000 IDs per request).
    corpus: list[_CorpusEntry] = []
    batch_size = 200
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i : i + batch_size]
        response = index.fetch(ids=batch_ids, namespace=collection)
        for vid, vec in response.vectors.items():
            meta = vec.metadata or {}
            text = meta.get("source_text", "")
            corpus.append(_CorpusEntry(
                chunk_id=vid,
                tokens=_tokenize(text),
                metadata=meta,
            ))

    bm25 = BM25Okapi([entry.tokens for entry in corpus])
    _bm25_cache[collection] = (bm25, corpus)
    logger.info("BM25 index built: %d documents.", len(corpus))
    return bm25, corpus


def invalidate_bm25_cache(collection: str | None = None) -> None:
    """
    Drop the cached BM25 index so the next query rebuilds from fresh Pinecone data.
    Pass collection=None to invalidate all collections.
    Called by the ingestion pipeline after upserting new documents.
    """
    if collection is None:
        _bm25_cache.clear()
    else:
        _bm25_cache.pop(collection, None)


# ---------------------------------------------------------------------------
# Individual search methods
# ---------------------------------------------------------------------------

def _vector_search(
    query: str,
    collection: str,
    allowed_scopes: list[str],
    top_k: int,
) -> list[ChunkResult]:
    """
    Query Pinecone for the top-k most semantically similar chunks.

    Access scope filtering is applied at the Pinecone query level — only chunks
    whose access_scope list overlaps with allowed_scopes are returned. This is
    the permission-aware retrieval mechanism: the vector DB enforces it inline,
    no separate ACL lookup needed.
    """
    model = cfg.get_embedding_model(collection)
    query_vector = embed_texts([query], model)[0]

    pc = _get_pinecone()
    index = pc.Index(cfg.PINECONE_INDEX_NAME)

    pinecone_filter: dict = {}
    if allowed_scopes:
        pinecone_filter["access_scope"] = {"$in": allowed_scopes}

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=collection,
        include_metadata=True,
        filter=pinecone_filter or None,
    )

    results: list[ChunkResult] = []
    for rank, match in enumerate(response.matches, start=1):
        m = match.metadata or {}
        results.append(ChunkResult(
            chunk_id=match.id,
            parent_id=m.get("parent_id") or None,
            doc_id=m.get("doc_id", ""),
            filename=m.get("filename", ""),
            collection=collection,
            source_text=m.get("source_text", ""),
            is_parent=bool(m.get("is_parent", False)),
            score=float(match.score),
            vector_rank=rank,
            metadata=m,
        ))
    return results


def _bm25_search(
    query: str,
    collection: str,
    top_k: int,
) -> list[ChunkResult]:
    """
    Score all chunks in the collection with BM25 and return the top-k.

    BM25 operates on the in-memory corpus (built lazily from Pinecone on first
    call). It does NOT apply access_scope filtering — that filter runs at the
    vector search level and in the final fusion step if needed. For the demo
    all content is public; a production system would pre-filter the corpus by
    scope before building the BM25 index, or filter results post-scoring.
    """
    bm25, corpus = _build_bm25_index(collection)
    if not corpus:
        return []

    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Pair (score, corpus_index), sort descending, take top_k.
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    results: list[ChunkResult] = []
    for rank, (idx, score) in enumerate(ranked, start=1):
        if score == 0.0:
            break  # no keyword overlap — remaining results are useless
        entry = corpus[idx]
        m = entry.metadata
        results.append(ChunkResult(
            chunk_id=entry.chunk_id,
            parent_id=m.get("parent_id") or None,
            doc_id=m.get("doc_id", ""),
            filename=m.get("filename", ""),
            collection=collection,
            source_text=m.get("source_text", ""),
            is_parent=bool(m.get("is_parent", False)),
            score=score,
            bm25_rank=rank,
            metadata=m,
        ))
    return results


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fusion(
    vector_results: list[ChunkResult],
    bm25_results: list[ChunkResult],
    top_k: int,
) -> list[ChunkResult]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.

    For each unique chunk_id seen across both lists:
      rrf_score = 1/(k + vector_rank) + 1/(k + bm25_rank)
    where missing rank contributes 0 (the chunk wasn't in that list at all).

    Chunks appearing in both lists are effectively double-counted — agreement
    between signals is a strong relevance signal that boosts them to the top.
    """
    # Index results by chunk_id for fast lookup.
    by_id: dict[str, ChunkResult] = {}

    for result in vector_results:
        by_id[result.chunk_id] = result  # carries vector_rank

    for result in bm25_results:
        if result.chunk_id in by_id:
            by_id[result.chunk_id].bm25_rank = result.bm25_rank
        else:
            by_id[result.chunk_id] = result  # bm25-only result

    # Compute RRF scores and sort.
    fused: list[ChunkResult] = []
    for chunk in by_id.values():
        rrf = 0.0
        if chunk.vector_rank is not None:
            rrf += 1.0 / (_RRF_K + chunk.vector_rank)
        if chunk.bm25_rank is not None:
            rrf += 1.0 / (_RRF_K + chunk.bm25_rank)
        chunk.score = rrf
        fused.append(chunk)

    fused.sort(key=lambda c: c.score, reverse=True)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    collection: str,
    allowed_scopes: list[str],
    top_k: int = 10,
) -> list[ChunkResult]:
    """
    Hybrid retrieval: vector + BM25 fused with RRF.

    Parameters
    ----------
    query : str
        The user's query (or the rewritten standalone query from the agent's
        query-rewrite node — callers should pass the rewritten form).
    collection : str
        Pinecone namespace to search. Scopes retrieval so a legal query never
        pulls finance chunks.
    allowed_scopes : list[str]
        Access identifiers for the requesting user/role. Only chunks whose
        access_scope overlaps with this list are returned. Pass ["public"] for
        unauthenticated access.
    top_k : int
        Number of results to return after fusion.

    Returns
    -------
    list[ChunkResult]
        Ranked results, highest RRF score first. Each carries source_text for
        citations, parent_id for context fetching, and both rank fields so the
        caller can see which signal drove the result.
    """
    # Fetch more candidates than top_k from each source so fusion has enough
    # material to work with. Standard practice: 3–5× the desired final count.
    candidate_k = max(top_k * 3, 20)

    vector_results = _vector_search(query, collection, allowed_scopes, candidate_k)
    bm25_results = _bm25_search(query, collection, candidate_k)

    return _rrf_fusion(vector_results, bm25_results, top_k)
