"""
backend/agent/nodes.py

LangGraph node functions — each takes the full AgentState and returns a dict
of fields to update in the state. LangGraph merges the returned dict into the
current state (additive fields via their reducers; others are replaced).

Node execution order (wired in graph.py):
  rewrite → router → retrieve (vector|cag|graph) → grade → [retry|generate]

Each node is a plain function — no classes, no side-effects outside state.
"""

import logging
import re
from typing import Any

from openai import OpenAI

import backend.config as cfg
from backend.agent.state import AgentState, Citation, ConversationTurn
from backend.retrieval.hybrid import ChunkResult, hybrid_search
from backend.retrieval.rerank import rerank

logger = logging.getLogger(__name__)

# CAG: if estimated total tokens in the collection is below this threshold,
# stuff all documents into context instead of retrieving.
# Based on gpt-4o-mini's 128k context; leaving headroom for system prompt and answer.
_CAG_TOKEN_THRESHOLD = 80_000

# Grade/retry cap — prevent infinite loops when corpus lacks the answer.
_MAX_RETRIES = 2

# Comparison question keywords (checked before deciding to decompose).
_COMPARISON_PATTERNS = re.compile(
    r"\b(compar|vs\.?|versus|differ|between|both|which is (better|higher|lower|more|less))\b",
    re.IGNORECASE,
)

# ── OpenAI client singleton ──────────────────────────────────────────────────

_openai: OpenAI | None = None


def _llm() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=cfg.OPENAI_API_KEY)
    return _openai


def _chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
    """Thin wrapper around OpenAI chat completions."""
    resp = _llm().chat.completions.create(
        model=cfg.DEFAULT_LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content.strip()


# ── Helper: context string from chunks ──────────────────────────────────────

_PARENT_IDX_RE = re.compile(r"__p(\d+)")


def _chunk_doc_order(chunk: "ChunkResult") -> int:
    """
    Extract the parent_index from a chunk_id so we can sort chunks in the
    order they originally appeared in the document.

    Chunk IDs follow the pattern  {doc_hash}__p{parent_idx}[__c{child_idx}].
    Sorting by parent_idx restores document order, which is critical for
    structured documents (resumes, reports) where section headers immediately
    precede their content.  Random Pinecone fetch order breaks this.
    """
    m = _PARENT_IDX_RE.search(chunk.chunk_id)
    return int(m.group(1)) if m else 0


def _chunks_to_context(chunks: list[ChunkResult], max_chunks: int | None = None, group_by_doc: bool = False) -> str:
    """Format chunks as a numbered context block for prompts.

    max_chunks=None means use all chunks (correct for CAG, which already
    verified the full collection fits in context). Vector/graph paths pass
    an explicit cap so the prompt stays within a predictable size.

    group_by_doc=True groups chunks under their source filename AND sorts
    them by their original document order (parent_index encoded in chunk_id).
    This keeps company headers adjacent to their bullet points, and ensures
    the LLM reads each document from top to bottom — critical for CAG where
    Pinecone returns chunks in random fetch order.
    """
    items = chunks if max_chunks is None else chunks[:max_chunks]

    if group_by_doc:
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for c in items:
            groups[c.filename].append(c)
        parts = []
        n = 1
        for filename, doc_chunks in groups.items():
            # Restore document order within each file
            doc_chunks_sorted = sorted(doc_chunks, key=_chunk_doc_order)
            parts.append(f"── Document: {filename} ──")
            for c in doc_chunks_sorted:
                parts.append(f"[{n}]\n{c.source_text}")
                n += 1
        return "\n\n".join(parts)

    return "\n\n".join(
        f"[{i+1}] (from {c.filename})\n{c.source_text}"
        for i, c in enumerate(items)
    )


def _is_comparison(query: str) -> bool:
    return bool(_COMPARISON_PATTERNS.search(query))


def _classify_collections(query: str) -> list[str]:
    """
    Return the collection(s) most relevant to this query, best match first.

    Two-stage approach:
      1. Keyword fast-path: count hits from a per-collection keyword table.
         If one collection scores ≥2× the next, return it alone.
      2. LLM fallback: when keyword scores tie or all are zero, ask the LLM to
         pick from the registry descriptions. Returns top-2 when still ambiguous
         so the retrieve step can search both namespaces and merge via RRF.

    Using 'auto' as the collection in QueryRequest triggers this function.
    """
    from backend.config import COLLECTION_REGISTRY

    registry = COLLECTION_REGISTRY
    if not registry:
        return ["demo"]

    # Keyword table — add terms as new collections are introduced
    _KEYWORDS: dict[str, list[str]] = {
        "finance": [
            "revenue", "roi", "earnings", "fiscal", "financial", "profit", "loss",
            "stock", "sec", "filing", "balance sheet", "cash flow", "ebitda",
            "quarter", "annual report", "dividend", "market cap",
        ],
        "legal": [
            "legal", "lawsuit", "dispute", "litigation", "contract", "compliance",
            "regulation", "court", "settlement", "attorney", "plaintiff", "defendant",
            "jurisdiction", "statute", "tort", "breach",
        ],
        "demo": [
            "resume", "candidate", "skills", "certification", "experience",
            "gpa", "degree", "project", "internship", "work history",
        ],
    }

    q_lower = query.lower()
    scores: dict[str, int] = {name: 0 for name in registry}
    for collection, kws in _KEYWORDS.items():
        if collection in registry:
            for kw in kws:
                if kw in q_lower:
                    scores[collection] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    # Clear keyword winner
    if top_score > 0 and top_score >= 2 * max(second_score, 1):
        logger.info("classify_collections: keyword winner=%r (score=%d)", ranked[0][0], top_score)
        return [ranked[0][0]]

    # Ambiguous or no keyword hits — ask the LLM
    descriptions = "\n".join(f"  {name}: {desc}" for name, desc in registry.items())
    prompt = (
        f"You are a document router. Given the query below, decide which 1 or 2 "
        f"document collections are most relevant. Return ONLY the collection name(s), "
        f"comma-separated, from this list:\n{descriptions}\n\n"
        f"Query: {query}\n\n"
        f"Answer (collection name(s) only):"
    )
    raw = _chat(
        [{"role": "user", "content": prompt}],
        max_tokens=30,
    ).strip().lower()

    chosen = [c.strip() for c in raw.split(",") if c.strip() in registry]
    if not chosen:
        # LLM returned something unexpected — fall back to keyword winner or demo
        chosen = [ranked[0][0]] if top_score > 0 else ["demo"]

    logger.info("classify_collections: llm chose=%r for query=%r", chosen, query[:60])
    return chosen[:2]  # cap at 2


def _has_named_entity(query: str) -> bool:
    """
    Return True if the query references a specific named entity mid-sentence
    (e.g. "AmplifAI", "Pinnacle Weaving Mills", "AWS").

    WHY THIS MATTERS FOR ROUTING:
      CAG (context-stuffing) dumps the entire collection into context. This is
      correct for aggregate/broad questions ("list all work experiences") where
      the LLM needs all content to build a comprehensive answer.

      For entity-specific questions ("explain AmplifAI experience", "what did he
      do at Pinnacle"), CAG can cause misattribution: the LLM sees 30+ chunks and
      conflates adjacent resume sections (Work Experience + Projects) because they
      happen to sit near each other in the document.

      Vector + BM25 is strictly better for entity queries: BM25 finds chunks that
      explicitly mention the entity name, so retrieved chunks are already scoped
      to that company/project — no cross-section confusion possible.

    HEURISTIC:
      A word appearing mid-sentence (not the first word) that starts with a capital
      letter and is longer than 2 characters is very likely a proper noun (company,
      person, product, certification). This catches "AmplifAI", "Pinnacle", "AWS",
      "Neo4j", "LangGraph" etc. while not triggering on sentence-start words.
    """
    words = query.split()
    for word in words[1:]:          # skip the first word (always capitalised)
        clean = word.strip('.,?!;:()')
        if clean and clean[0].isupper() and len(clean) > 2:
            return True
    return False


# ── Node 1: rewrite ──────────────────────────────────────────────────────────

def rewrite_node(state: AgentState) -> dict[str, Any]:
    """
    Rewrite the user's question into a self-contained standalone query.

    WHY:
      Follow-up questions like "What about their revenue?" are ambiguous without
      history. Converting them to "What was Company X's revenue in FY2023?"
      makes retrieval precise — the embedding of a vague pronoun-heavy question
      will drift in the embedding space and miss the right chunks.

    HOW:
      Shows the LLM the last 3 questions from conversation_history (not answers)
      plus the current question. Instructs it to resolve pronouns and references
      and return a fully standalone question. Cost: ~200 tokens per turn, flat.

    Turn 1 shortcut:
      No history → return question unchanged (no LLM call needed).
    """
    question: str = state["question"]
    history: list[ConversationTurn] = state.get("conversation_history", [])

    if not history:
        return {"rewritten_query": question}

    recent_qs = "\n".join(f"- {t.question}" for t in history[-3:])
    prompt = (
        f"Rewrite the follow-up question as a fully standalone question.\n\n"
        f"Recent questions:\n{recent_qs}\n\n"
        f"Follow-up: {question}\n\n"
        f"Standalone question (return only the rewritten question, no explanation):"
    )
    standalone = _chat(
        [
            {"role": "system", "content": "You rewrite follow-up questions into standalone questions."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
    )
    logger.info("rewrite: %r → %r", question, standalone)
    return {"rewritten_query": standalone}


# ── Node 2: router ───────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict[str, Any]:
    """
    Choose the retrieval path for this query.

    Routes:
      'cag'    — Context-stuffing: all docs fit in the LLM context window.
                 Gated on estimated total tokens < _CAG_TOKEN_THRESHOLD.
                 Token estimate = total vectors in namespace × avg child tokens.
                 More precise than file-count gating (ARCHITECTURE.md §5.3).

      'graph'  — Neo4j GraphRAG: relational or comparative questions on a
                 finance/graph-enabled collection. Detected by keyword heuristic
                 + collection name check. Stub-safe: falls through to 'vector'
                 if the graph layer isn't built yet.

      'vector' — Default. Hybrid search + reranker (Files 7–8).

    Note on comparison questions:
      Comparison routing goes to 'vector'; the generate_node handles decomposition
      into sub-queries and fresh per-fact retrieval. The router only picks the
      *retrieval mechanism*, not the generation strategy.
    """
    query: str = state.get("rewritten_query") or state["question"]
    collection: str = state["collection"]

    # ── Cross-collection classification ──────────────────────────────────────
    # When the caller passes collection='auto', classify which namespace(s) fit.
    # Otherwise respect the caller's explicit collection choice.
    if collection == "auto":
        active = _classify_collections(query)
        collection = active[0]   # primary namespace drives CAG/graph checks below
    else:
        active = [collection]

    # ── CAG check ────────────────────────────────────────────────────────────
    # CAG is only used for BROAD/AGGREGATE questions on small collections.
    # Entity-specific queries (mentioning a company, project, or product by name)
    # are routed to vector even when the collection is small, because:
    #   - BM25 finds chunks that explicitly mention the named entity
    #   - CAG dumps all chunks and risks cross-section misattribution in structured
    #     documents (e.g. resume Work Experience + Projects sections mixed up)
    try:
        from backend.ingest.embed_index import _get_pinecone
        import backend.config as _cfg
        pc = _get_pinecone()
        stats = pc.Index(_cfg.PINECONE_INDEX_NAME).describe_index_stats()
        ns_stats = stats.namespaces or {}
        vector_count = ns_stats.get(collection, {}).get("vector_count", 0) if isinstance(ns_stats.get(collection), dict) else getattr(ns_stats.get(collection), "vector_count", 0)
        # Only child chunks get embedded for semantic search (roughly half the total).
        estimated_tokens = (vector_count // 2) * cfg.DEFAULT_CHUNK_CONFIG.child_tokens
        if 0 < estimated_tokens < _CAG_TOKEN_THRESHOLD:
            if _has_named_entity(query):
                logger.info("router: vector (entity-specific query — bypassing CAG to prevent cross-section misattribution)")
            else:
                logger.info("router: cag (est. %d tokens, broad query)", estimated_tokens)
                return {"route": "cag", "collection": collection, "active_collections": active}
    except Exception as exc:
        logger.warning("router: CAG check failed (%s), falling through", exc)

    # ── Graph check ───────────────────────────────────────────────────────────
    # Finance collection + relational/comparative question → graph path.
    finance_collections = {"sec_filings", "finance", "filings"}
    if collection in finance_collections and _is_comparison(query):
        logger.info("router: graph (relational query on finance collection)")
        return {"route": "graph", "collection": collection, "active_collections": active}

    # ── Default: vector ───────────────────────────────────────────────────────
    logger.info("router: vector (active_collections=%r)", active)
    return {"route": "vector", "collection": collection, "active_collections": active}


# ── Node 3a: retrieve (vector path) ─────────────────────────────────────────

def retrieve_vector_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve relevant chunks via hybrid search + Cohere reranking.

    Single-collection path (default):
      1. hybrid_search() — vector + BM25 fused with RRF (File 7).
         Returns top 20 candidates from the collection namespace.
      2. rerank() — Cohere cross-encoder scores candidates and returns top 8.

    Multi-collection path (active_collections has 2 entries):
      Searches both namespaces in parallel (each top_k=15), then re-applies
      RRF fusion across both result sets before reranking the merged top-20.
      This means a query like "Apple ROI" can pull from both 'finance' and
      'legal' without the caller knowing which collection to pick upfront.
    """
    query: str = state.get("rewritten_query") or state["question"]
    collection: str = state["collection"]
    scopes: list[str] = state.get("allowed_scopes", ["public"])
    active: list[str] = state.get("active_collections") or [collection]

    if len(active) >= 2:
        # Parallel search across both collections, then fuse + rerank
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(hybrid_search, query, col, scopes, 15): col for col in active[:2]}
            all_candidates: list[ChunkResult] = []
            for fut in concurrent.futures.as_completed(futures):
                col = futures[fut]
                try:
                    all_candidates.extend(fut.result())
                except Exception as exc:
                    logger.warning("retrieve_vector_node: search on %r failed: %s", col, exc)

        # Deduplicate by chunk_id (same chunk could theoretically appear in both)
        seen: set[str] = set()
        deduped = [c for c in all_candidates if not (c.chunk_id in seen or seen.add(c.chunk_id))]  # type: ignore[func-returns-value]
        logger.info("retrieve_vector_node: multi-collection merge %r → %d candidates", active, len(deduped))
        reranked = rerank(query, deduped, top_n=8)
    else:
        candidates = hybrid_search(query, collection, scopes, top_k=20)
        reranked = rerank(query, candidates, top_n=8)

    return {
        "retrieved_chunks": reranked,
        "reusable_chunks": reranked,
    }


# ── Node 3b: retrieve (CAG path) ────────────────────────────────────────────

def retrieve_cag_node(state: AgentState) -> dict[str, Any]:
    """
    Context-stuffing retrieval: fetch ALL content in the collection and return
    it as context, bypassing vector search entirely.

    WHY:
      When the entire document set fits the LLM's context window (the router
      already verified estimated_tokens < _CAG_TOKEN_THRESHOLD), ranked retrieval
      introduces noise — wrong chunks silently omit facts. Stuffing the full
      collection lets the LLM locate every relevant passage itself, making it
      impossible to miss entries like a resume work experience section that ranks
      below the top-N cutoff in vector search.

    HOW:
      1. List every vector ID in the Pinecone namespace.
      2. Batch-fetch metadata; keep only parent chunks (is_parent=True).
         Parents carry the full section text; using them avoids duplicate content
         since each parent already contains its children's text.
      3. If no parent chunks exist (e.g. collection indexed without parent-doc
         chunking), fall back to all child chunks.
      4. Apply access_scope filter so permission-aware retrieval still holds.

    Returns all chunks with score=1.0 (all equally relevant — LLM decides).
    Falls back to vector retrieval on any Pinecone error.
    """
    from backend.ingest.embed_index import _get_pinecone
    import backend.config as _cfg

    collection: str = state["collection"]
    scopes: list[str] = state.get("allowed_scopes", ["public"])

    try:
        pc = _get_pinecone()
        index = pc.Index(_cfg.PINECONE_INDEX_NAME)

        # Gather every vector ID in this namespace
        all_ids: list[str] = []
        for page in index.list(namespace=collection):
            all_ids.extend(item.id for item in page.vectors)

        if not all_ids:
            logger.warning("retrieve_cag_node: empty namespace=%r, falling back", collection)
            return retrieve_vector_node(state)

        # Batch-fetch all metadata; collect parent chunks
        parent_chunks: list[ChunkResult] = []
        child_chunks: list[ChunkResult] = []
        batch_size = 200

        for start in range(0, len(all_ids), batch_size):
            resp = index.fetch(ids=all_ids[start : start + batch_size], namespace=collection)
            for vid, vec in resp.vectors.items():
                meta = vec.metadata or {}
                # Access scope filter
                chunk_scopes = meta.get("access_scope", ["public"])
                if not any(s in chunk_scopes for s in scopes):
                    continue
                cr = ChunkResult(
                    chunk_id=vid,
                    parent_id=meta.get("parent_id") or None,
                    doc_id=meta.get("doc_id", ""),
                    filename=meta.get("filename", ""),
                    collection=collection,
                    source_text=meta.get("source_text", ""),
                    is_parent=bool(meta.get("is_parent", False)),
                    score=1.0,
                    metadata=meta,
                )
                if cr.is_parent:
                    parent_chunks.append(cr)
                else:
                    child_chunks.append(cr)

        # Prefer parents (full context); fall back to children if collection
        # was indexed without parent-document chunking
        chunks = parent_chunks if parent_chunks else child_chunks

        if not chunks:
            logger.warning("retrieve_cag_node: no accessible chunks, falling back to vector")
            return retrieve_vector_node(state)

        logger.info("retrieve_cag_node: loaded %d chunks for CAG (%d parents, %d children)",
                    len(chunks), len(parent_chunks), len(child_chunks))
        return {
            "retrieved_chunks": chunks,
            "reusable_chunks": chunks,
        }

    except Exception as exc:
        logger.warning("retrieve_cag_node: failed (%s), falling back to vector", exc)
        return retrieve_vector_node(state)


# ── Node 3c: retrieve (graph path) ──────────────────────────────────────────

def retrieve_graph_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieve from the Neo4j knowledge graph for relational/comparative questions.

    Calls graph_rag.query.graph_query() which extracts structured params from
    the question (company names, metric, year) and runs the appropriate Cypher
    template. Results come back as ChunkResult objects — identical interface to
    vector retrieval, so grade_node and generate_node need no special handling.

    Falls back to vector retrieval if Neo4j returns no results (e.g. the
    entities haven't been ingested into the graph yet).
    """
    from backend.graph_rag.query import graph_query
    from backend.graph_rag.schema import get_schema

    query: str = state.get("rewritten_query") or state["question"]
    collection: str = state["collection"]
    schema = get_schema(collection)

    results = graph_query(query, schema, collection)

    if results:
        return {
            "retrieved_chunks": results,
            "reusable_chunks": results,
        }

    # Graph returned nothing — fall back to hybrid vector search
    logger.warning("retrieve_graph_node: no graph results, falling back to vector")
    return retrieve_vector_node(state)


# ── Node 4: grade ────────────────────────────────────────────────────────────

def grade_node(state: AgentState) -> dict[str, Any]:
    """
    Assess whether the retrieved chunks are sufficient to answer the question.

    WHY:
      Retrieval can fail: the corpus might not contain the answer, the query
      might have been ambiguous, or BM25 + reranking might have pulled
      tangentially relevant chunks. Grading closes the agentic loop: instead
      of generating a hallucinated answer from bad context, the agent retries
      with a reformulated query.

    HOW:
      Prompt gpt-4o-mini with the question and chunk texts. Ask for a one-word
      verdict: 'sufficient' or 'insufficient'. Temperature=0 for determinism.

    Retry cap:
      retry_count is incremented here. The graph's conditional edge (graph.py)
      routes to generate if grade=='sufficient' OR retry_count>=_MAX_RETRIES,
      preventing infinite loops when the corpus genuinely lacks the answer.
    """
    query: str = state.get("rewritten_query") or state["question"]
    chunks: list[ChunkResult] = state.get("retrieved_chunks", [])
    retry_count: int = state.get("retry_count", 0)

    if not chunks:
        return {"grade": "insufficient", "retry_count": retry_count + 1}

    context = _chunks_to_context(chunks, max_chunks=8)
    verdict = _chat(
        [
            {
                "role": "system",
                "content": (
                    "You are a retrieval grader. Given a question and retrieved passages, "
                    "decide if the passages contain enough information to answer the question. "
                    "Reply with exactly one word: 'sufficient' or 'insufficient'."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nPassages:\n{context}",
            },
        ],
        max_tokens=10,
    ).lower()

    grade = "sufficient" if "sufficient" in verdict else "insufficient"
    logger.info("grade: %s (retry %d)", grade, retry_count)
    return {"grade": grade, "retry_count": retry_count + 1}


# ── Node 5: generate ─────────────────────────────────────────────────────────

def generate_node(state: AgentState) -> dict[str, Any]:
    """
    Generate the final answer, grounded in retrieved source chunks.

    Two generation paths:

    COMPARISON PATH (detected by keyword heuristic on rewritten_query):
      Per ARCHITECTURE.md §5.9: "the agent decomposes comparison questions into
      sub-queries, retrieves each fact fresh from source, compares on grounded
      facts — never reuses a prior generated answer."

      Steps:
        1. LLM decomposes the comparison question into 2 sub-queries.
        2. Each sub-query runs through hybrid_search + rerank independently.
        3. Both fresh result sets are combined as context.
        4. Final answer is generated from the grounded combined context.

      WHY FRESH RETRIEVAL:
        If we retrieved "Company A revenue: $5B" in one turn and now ask
        "compare A vs B", reusing the cached $5B answer risks propagating
        a number that may have been hallucinated or retrieved from the wrong
        document. Fresh retrieval from the source document is always safer.

    STANDARD PATH:
      Combines retrieved_chunks (current turn) with reusable_chunks from prior
      turns, deduplicates by chunk_id, and generates from the merged context.
      Source chunks are safe to reuse — they are direct document quotes, not
      model-generated text.

    Returns:
      answer             — the generated response string.
      citations          — one Citation per chunk used in the answer.
      conversation_history — [new ConversationTurn] appended via additive reducer.
    """
    query: str = state.get("rewritten_query") or state["question"]
    collection: str = state["collection"]
    scopes: list[str] = state.get("allowed_scopes", ["public"])
    current_chunks: list[ChunkResult] = state.get("retrieved_chunks", [])
    prior_chunks: list[ChunkResult] = state.get("reusable_chunks", [])

    route: str = state.get("route", "vector")

    # ── Comparison path: decompose → retrieve fresh ───────────────────────────
    if _is_comparison(query):
        context_chunks = _comparison_retrieval(query, collection, scopes)
    else:
        # Merge current + prior, deduplicate by chunk_id.
        seen: set[str] = set()
        merged: list[ChunkResult] = []
        for c in current_chunks + prior_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                merged.append(c)
        # CAG already fetched ALL content — don't cap it; vector uses top 8.
        context_chunks = merged if route == "cag" else merged[:8]

    if not context_chunks:
        return {
            "answer": "I don't have enough information in the provided documents to answer this question.",
            "citations": [],
            "conversation_history": [
                ConversationTurn(
                    question=state["question"],
                    rewritten_query=state.get("rewritten_query"),
                    retrieved_chunks=[],
                )
            ],
        }

    # CAG: pass all chunks grouped by document so company headers and their
    # adjacent bullet-point chunks appear contiguously in the context.
    # Vector/graph: numbered list is fine since chunks are already the top-ranked.
    context = _chunks_to_context(context_chunks, group_by_doc=(route == "cag"))

    # CAG gets a synthesis-oriented prompt because content is spread across many
    # passages and the LLM needs to aggregate across chunks (e.g. company name in
    # one chunk, bullet points in an adjacent chunk). Vector/graph use a stricter
    # prompt because they receive only the top-scored chunks.
    if route == "cag":
        system_prompt = (
            "You are a precise, helpful assistant with access to a document collection. "
            "The context below contains ALL passages from ALL documents in the collection, "
            "presented in their original document order. "
            "Rules you MUST follow:\n"
            "1. Read EVERY passage from EVERY document before composing your answer. "
            "Do not stop after the first document — the collection may contain multiple "
            "files and important information may appear in later documents.\n"
            "2. COMPLETENESS: For questions asking to list or summarise all items "
            "(e.g. work experiences, skills, projects), include EVERY distinct item "
            "found across ALL documents. Missing even one item is an error.\n"
            "3. DEDUPLICATION: A document may describe the same employer in multiple "
            "sections (e.g. 'Work Experience', 'Applied Data Science Experience'). "
            "When the same company name appears more than once, MERGE all mentions into "
            "ONE entry. If the company had multiple distinct roles, list each as a "
            "sub-item — never list the same employer twice as separate top-level entries.\n"
            "4. ATTRIBUTION: Only attribute projects or achievements to a company if they "
            "are EXPLICITLY stated under that company's section. Never pull details from "
            "a standalone Projects section and attribute them to a Work Experience entry.\n"
            "5. HONESTY: If an entry has limited detail, report exactly what is stated; "
            "do not invent or infer responsibilities.\n"
            "6. Never speculate. Only report what the text explicitly says."
        )
    else:
        system_prompt = (
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "Attribute information only to the source it explicitly appears under. "
            "If the context does not contain sufficient information, say so clearly. "
            "Do not fabricate or infer information."
        )

    answer = _chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=2000,
        temperature=0.1,
    )

    citations = [
        Citation(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            filename=c.filename,
            collection=c.collection,
            source_text=c.source_text,
        )
        for c in context_chunks
    ]

    new_turn = ConversationTurn(
        question=state["question"],
        rewritten_query=state.get("rewritten_query"),
        retrieved_chunks=context_chunks,
    )

    return {
        "answer": answer,
        "citations": citations,
        "conversation_history": [new_turn],  # additive — appended by reducer
    }


def _comparison_retrieval(
    query: str, collection: str, scopes: list[str]
) -> list[ChunkResult]:
    """
    Decompose a comparison query into sub-queries and retrieve each fresh.

    Returns combined, deduplicated chunks from both sub-retrievals.
    """
    decompose_prompt = (
        f"Decompose this comparison question into exactly 2 simple sub-questions, "
        f"one for each subject being compared. Return only the 2 questions, one per line.\n\n"
        f"Question: {query}"
    )
    raw = _chat(
        [{"role": "user", "content": decompose_prompt}],
        max_tokens=150,
    )
    sub_queries = [q.strip().lstrip("12.-) ") for q in raw.strip().splitlines() if q.strip()][:2]

    if len(sub_queries) < 2:
        # Decomposition failed — fall back to single retrieval
        sub_queries = [query]

    all_chunks: list[ChunkResult] = []
    seen: set[str] = set()
    for sq in sub_queries:
        candidates = hybrid_search(sq, collection, scopes, top_k=10)
        fresh = rerank(sq, candidates, top_n=3)
        for c in fresh:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                all_chunks.append(c)

    return all_chunks
