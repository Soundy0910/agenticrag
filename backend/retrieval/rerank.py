"""
backend/retrieval/rerank.py

Reranks hybrid retrieval candidates with Cohere Rerank.

WHY RERANKING:
  Hybrid search (vector + BM25) is good at *recall* — it surfaces the right
  chunks in its top-20. But the ranking within those 20 is noisy: cosine
  similarity and BM25 scores don't model fine-grained query-document relevance.
  Cohere Rerank is a cross-encoder: it sees the full (query, document) pair
  together and produces a relevance score far more precise than a bi-encoder
  vector match. Running it only on the top-20 candidates (not the full corpus)
  keeps the cost negligible while getting near-oracle ordering for the LLM.

  Empirically, reranking with a cross-encoder is one of the highest ROI moves
  in a RAG pipeline — typically 5–10% improvement in faithfulness and answer
  relevance on RAGAS benchmarks, at ~$0.001 per query on Cohere's free tier.

SELF-HOSTED ALTERNATIVE (scale):
  At high query volume, Cohere Rerank costs add up. The at-scale alternative
  is bge-reranker-base or bge-reranker-large (BAAI/bge-reranker-* on HuggingFace):
    - Run locally or on a cheap GPU instance
    - Same cross-encoder architecture, similar quality to Cohere Rerank v2
    - Zero per-query cost
    - Latency: ~50–200ms on CPU for 20 candidates (acceptable for async API)
  To swap: replace the cohere.Client call below with a HuggingFace pipeline
  call using the cross-encoder/ms-marco-* or BAAI/bge-reranker-* checkpoint.
  The function signature and return type are identical — callers don't change.
"""

import logging
import os
import time

import backend.config as cfg
from backend.retrieval.hybrid import ChunkResult

logger = logging.getLogger(__name__)

# Cohere rerank model — rerank-english-v3.0 is their strongest general model.
# rerank-multilingual-v3.0 if the corpus contains non-English content.
_COHERE_MODEL = "rerank-english-v3.0"

_cohere_client = None


def _get_cohere():
    global _cohere_client
    if _cohere_client is None:
        try:
            import cohere
        except ImportError as exc:
            raise ImportError(
                "cohere is not installed. Run: pip install cohere"
            ) from exc
        key = cfg.COHERE_API_KEY or os.environ.get("COHERE_API_KEY", "")
        if not key:
            raise EnvironmentError(
                "COHERE_API_KEY is not set. Add it to your .env file."
            )
        _cohere_client = cohere.ClientV2(api_key=key)
    return _cohere_client


def rerank(
    query: str,
    candidates: list[ChunkResult],
    top_n: int = 5,
) -> list[ChunkResult]:
    """
    Re-score retrieval candidates with Cohere Rerank and return the best top_n.

    Cohere's cross-encoder reads the full (query, document) pair and produces
    a relevance score that is strictly more accurate than the RRF scores from
    hybrid_search — it understands query-document interaction, not just
    similarity in embedding space.

    Parameters
    ----------
    query : str
        The user's query (same string passed to hybrid_search).
    candidates : list[ChunkResult]
        Output of hybrid_search() — typically top 10–20 candidates.
    top_n : int
        Number of results to return after reranking. Architecture target: 3–5.

    Returns
    -------
    list[ChunkResult]
        Reranked subset of candidates, highest relevance first. Each result's
        .score field is replaced with the Cohere relevance score (0–1 range).
        The original .vector_rank and .bm25_rank fields are preserved so the
        caller can see how reranking changed the order.
    """
    if not candidates:
        return []

    top_n = min(top_n, len(candidates))
    client = _get_cohere()
    docs = [c.source_text for c in candidates]

    # Retry with exponential backoff for Cohere trial-key rate limits (429).
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            response = client.rerank(
                model=_COHERE_MODEL,
                query=query,
                documents=docs,
                top_n=top_n,
            )
            break
        except Exception as exc:
            is_rate_limit = "429" in str(exc) or "TooManyRequests" in type(exc).__name__
            if is_rate_limit and attempt < max_attempts - 1:
                wait = 6 * (2 ** attempt)   # 6s, 12s, 24s
                logger.warning("Cohere rate limit hit — retrying in %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
            else:
                # Non-rate-limit error or out of retries — return candidates ranked by RRF score
                logger.warning("rerank: Cohere failed (%s), falling back to hybrid ranking", exc)
                return sorted(candidates, key=lambda c: c.score, reverse=True)[:top_n]

    reranked: list[ChunkResult] = []
    for result in response.results:
        chunk = candidates[result.index]
        chunk.score = result.relevance_score
        reranked.append(chunk)

    return reranked
