"""
backend/agent/nodes.py

LangGraph node functions — each takes AgentState and returns a state delta.

Node execution order (wired in graph.py):
  rewrite → classify → router → access_check → retrieve → grade
      → [validate_numbers] → generate → END

New nodes vs original:
  classify_node       — query type classification (factual/comparison/risk/etc.)
  access_check_node   — RBAC: blocks queries to restricted collections
  validate_numbers_node — deterministic numeric extraction + calculation

All nodes record their latency in step_latencies for the Live Trace UI.
"""

import json
import logging
import re
import time
from typing import Any

from openai import OpenAI

import backend.config as cfg
from backend.agent.state import AgentState, Citation, CollectionQuery, ConversationTurn
from backend.retrieval.hybrid import ChunkResult, hybrid_search
from backend.retrieval.rerank import rerank

logger = logging.getLogger(__name__)

_CAG_TOKEN_THRESHOLD = 80_000
_FACET_TOP_K = 6
_MAX_RETRIES = 2

_COMPARISON_PATTERNS = re.compile(
    r"\b(compar\w*|vs\.?|versus|differ\w*|between|both|which is (better|higher|lower|more|less))\b",
    re.IGNORECASE,
)

# ── OpenAI client ────────────────────────────────────────────────────────────

_openai: OpenAI | None = None


def _llm() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=cfg.OPENAI_API_KEY)
    return _openai


def _chat_with_usage(
    messages: list[dict], max_tokens: int = 512, temperature: float = 0.0
) -> tuple[str, int, int]:
    """Chat completion returning (content, prompt_tokens, completion_tokens)."""
    resp = _llm().chat.completions.create(
        model=cfg.DEFAULT_LLM_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = resp.choices[0].message.content.strip()
    usage = resp.usage
    prompt_tok = usage.prompt_tokens if usage else 0
    completion_tok = usage.completion_tokens if usage else 0
    return content, prompt_tok, completion_tok


def _chat(messages: list[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
    content, _, _ = _chat_with_usage(messages, max_tokens, temperature)
    return content


def _tok(prompt: int, completion: int) -> dict:
    """Return token accounting fields for state update."""
    return {
        "total_tokens": prompt + completion,
        "input_tokens": prompt,
        "output_tokens": completion,
    }


# ── Context helpers ──────────────────────────────────────────────────────────

_PARENT_IDX_RE = re.compile(r"__p(\d+)")


def _chunk_doc_order(chunk: "ChunkResult") -> int:
    m = _PARENT_IDX_RE.search(chunk.chunk_id)
    return int(m.group(1)) if m else 0


def _chunks_to_context(
    chunks: list[ChunkResult], max_chunks: int | None = None, group_by_doc: bool = False
) -> str:
    items = chunks if max_chunks is None else chunks[:max_chunks]
    if group_by_doc:
        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for c in items:
            groups[c.filename].append(c)
        parts = []
        n = 1
        for filename, doc_chunks in groups.items():
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


def _indexed_collections() -> set[str]:
    from backend.config import COLLECTION_REGISTRY
    try:
        from backend.ingest.embed_index import _get_pinecone
        import backend.config as _cfg
        pc = _get_pinecone()
        stats = pc.Index(_cfg.PINECONE_INDEX_NAME).describe_index_stats()
        ns = stats.get("namespaces") or {}
        indexed = set()
        for name, info in ns.items():
            vc = info.get("vector_count", 0) if isinstance(info, dict) else getattr(info, "vector_count", 0)
            if vc > 0 and name in COLLECTION_REGISTRY:
                indexed.add(name)
        return indexed or set(COLLECTION_REGISTRY.keys())
    except Exception as exc:
        logger.warning("_indexed_collections: %s", exc)
        return set(COLLECTION_REGISTRY.keys())


def _classify_collections(query: str, allowed: set[str] | None = None) -> list[str]:
    """Return the collection(s) most relevant to this query, best match first.

    allowed: optional set of collection names to restrict the search space (for RBAC).
    """
    from backend.config import COLLECTION_REGISTRY
    registry = COLLECTION_REGISTRY
    indexed = _indexed_collections()
    if allowed is not None:
        indexed = indexed & allowed
    if not registry:
        return ["sec-filings"]

    _KEYWORDS: dict[str, list[str]] = {
        "sec-filings": [
            "10-k", "annual report", "revenue", "net income", "fiscal year",
            "earnings", "operating income", "eps", "diluted", "shares outstanding",
            "risk factors", "management discussion", "md&a", "audited financials",
            "aapl", "msft", "nvda", "googl", "amzn", "tsla", "meta",
            "jpm", "jnj", "wmt", "xom", "pfe", "dis", "ko", "v",
            "apple", "microsoft", "nvidia", "google", "amazon", "tesla",
            "jpmorgan", "johnson", "walmart", "exxon", "pfizer", "disney",
            "coca-cola", "visa", "gross margin", "r&d spending", "net sales",
        ],
        "legal-docs": [
            "contract", "agreement", "clause", "indemnification", "liability",
            "exhibit", "credit facility", "license agreement", "supply agreement",
            "employment agreement", "nda", "non-disclosure", "merger agreement",
            "material agreement", "governing law", "termination", "intellectual property",
            "warranty", "arbitration", "representations", "covenants",
            "clawback", "recoupment", "recoup", "compensation agreement",
            "award agreement", "equity award", "incentive plan", "executive compensation",
            "restricted stock", "rsu", "severance", "deferred compensation",
        ],
    }

    _COMPANY_KWS = [
        "aapl", "msft", "nvda", "googl", "amzn", "tsla", "meta", "jpm", "jnj",
        "apple", "microsoft", "nvidia", "google", "amazon", "tesla", "jpmorgan",
        "johnson", "walmart", "exxon", "pfizer", "disney",
    ]
    _FIN_KWS = [
        "revenue", "net income", "fiscal year", "earnings", "annual report",
        "10-k", "operating income", "profit", "roi", "ebitda",
    ]
    _LEGAL_KWS = _KEYWORDS["legal-docs"]

    q_lower = query.lower()

    def _count_hits(kws: list[str]) -> int:
        return sum(1 for kw in kws if kw in q_lower)

    has_company = _count_hits(_COMPANY_KWS) > 0
    has_financial = _count_hits(_FIN_KWS) > 0
    has_legal = _count_hits(_LEGAL_KWS) > 0

    cross_domain = (
        has_company and has_legal and (
            has_financial or "annual report" in q_lower or "10-k" in q_lower
        )
    )
    if cross_domain and "sec-filings" in indexed and "legal-docs" in indexed:
        logger.info("classify_collections: cross-domain")
        return ["sec-filings", "legal-docs"]

    scores: dict[str, int] = {name: 0 for name in registry if name in indexed}
    for collection, kws in _KEYWORDS.items():
        if collection in scores:
            for kw in kws:
                if kw in q_lower:
                    scores[collection] += 1

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return ["sec-filings"] if "sec-filings" in indexed else [next(iter(indexed))]

    top_score = ranked[0][1]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score > 0 and top_score >= 2 * max(second_score, 1):
        return [ranked[0][0]]

    descriptions = "\n".join(
        f"  {name}: {desc}" for name, desc in registry.items() if name in indexed
    )
    prompt = (
        f"You are a document router. Given the query below, decide which 1 or 2 "
        f"document collections are most relevant. Return ONLY the collection name(s), "
        f"comma-separated, from this list:\n{descriptions}\n\nQuery: {query}\n\nAnswer:"
    )
    raw = _chat([{"role": "user", "content": prompt}], max_tokens=30).strip().lower()
    chosen = [c.strip() for c in raw.split(",") if c.strip() in registry and c.strip() in indexed]
    if not chosen:
        if top_score > 0:
            winner = ranked[0][0]
        elif has_financial and "sec-filings" in indexed:
            winner = "sec-filings"
        elif has_legal and "legal-docs" in indexed:
            winner = "legal-docs"
        else:
            winner = "sec-filings" if "sec-filings" in indexed else next(iter(indexed))
        chosen = [winner]
    return chosen[:2]


def _has_named_entity(query: str) -> bool:
    words = query.split()
    for word in words[1:]:
        clean = word.strip('.,?!;:()')
        if clean and clean[0].isupper() and len(clean) > 2:
            return True
    return False


_RELATIVE_YEAR_RE = re.compile(
    r"\b(?:the\s+)?(?:previous|prior|preceding)\s+year\b|"
    r"\bthe\s+year\s+before(?:\s+that)?\b|"
    r"\blast\s+year\b",
    re.IGNORECASE,
)


def _anchor_year_from_history(history: list[ConversationTurn]) -> int | None:
    years: list[int] = []
    for turn in history[-3:]:
        src = turn.rewritten_query or turn.question
        for match in re.finditer(r"\b(20\d{2}|19\d{2})\b", src):
            years.append(int(match.group(1)))
    return years[-1] if years else None


def _apply_relative_year_resolution(text: str, anchor_year: int | None) -> str:
    if anchor_year is None:
        return text
    target = str(anchor_year - 1)
    return _RELATIVE_YEAR_RE.sub(lambda _: target, text)


# ── Citation helpers ─────────────────────────────────────────────────────────

# Company name → (ticker, display name) for citation enrichment
_COMPANY_NAME_MAP: dict[str, tuple[str, str]] = {
    "microsoft": ("MSFT", "Microsoft"),
    "msft": ("MSFT", "Microsoft"),
    "apple": ("AAPL", "Apple"),
    "aapl": ("AAPL", "Apple"),
    "nvidia": ("NVDA", "NVIDIA"),
    "nvda": ("NVDA", "NVIDIA"),
    "alphabet": ("GOOGL", "Alphabet/Google"),
    "google": ("GOOGL", "Alphabet/Google"),
    "googl": ("GOOGL", "Alphabet/Google"),
    "amazon": ("AMZN", "Amazon"),
    "amzn": ("AMZN", "Amazon"),
    "meta": ("META", "Meta"),
    "tesla": ("TSLA", "Tesla"),
    "tsla": ("TSLA", "Tesla"),
    "jpmorgan": ("JPM", "JPMorgan Chase"),
    "jpm": ("JPM", "JPMorgan Chase"),
    "walmart": ("WMT", "Walmart"),
    "wmt": ("WMT", "Walmart"),
    "johnson": ("JNJ", "Johnson & Johnson"),
    "jnj": ("JNJ", "J&J"),
    "pfizer": ("PFE", "Pfizer"),
    "pfe": ("PFE", "Pfizer"),
    "exxon": ("XOM", "ExxonMobil"),
    "xom": ("XOM", "ExxonMobil"),
    "disney": ("DIS", "Disney"),
    "dis": ("DIS", "Disney"),
    "coca-cola": ("KO", "Coca-Cola"),
    "ko": ("KO", "Coca-Cola"),
    "visa": ("V", "Visa"),
}


def _detect_company_display(filename: str, doc_id: str) -> str:
    """Return human-readable company name from filename/doc_id."""
    combined = (filename + " " + doc_id).lower()
    for key, (_, display) in _COMPANY_NAME_MAP.items():
        if key in combined:
            return display
    return ""


def _detect_filing_type(filename: str) -> str:
    """Return filing type from filename."""
    fn = filename.lower()
    if "10k" in fn or "10-k" in fn:
        # Check for exhibit markers
        m = re.search(r"exhibit(\d+)", fn)
        if m:
            return f"EX-{m.group(1)}"
        return "10-K"
    if "ex10" in fn or "exhibit10" in fn or "ex-10" in fn:
        m = re.search(r"ex(?:hibit)?[\s_-]?10[_.]?(\d+)", fn)
        if m:
            return f"EX-10.{m.group(1)}"
        return "EX-10"
    if "exhibit" in fn:
        m = re.search(r"exhibit(\d+)", fn)
        if m:
            return f"EX-{m.group(1)}"
    return "Filing"


def _detect_fiscal_year(filename: str) -> str:
    """Extract fiscal year label from filename date (e.g. 10k_2025-07-30.htm → FY2025)."""
    m = re.search(r"(\d{4})-\d{2}-\d{2}", filename)
    if m:
        return f"FY{m.group(1)}"
    m = re.search(r"(\d{4})", filename)
    if m:
        return f"FY{m.group(1)}"
    return ""


def _build_citation(chunk: ChunkResult) -> Citation:
    """Build an enriched Citation from a retrieved chunk."""
    company = _detect_company_display(chunk.filename, chunk.doc_id)
    filing_type = _detect_filing_type(chunk.filename)
    fiscal_year = _detect_fiscal_year(chunk.filename)

    # Build human-readable display name
    parts = []
    if company:
        parts.append(company)
    if fiscal_year and filing_type == "10-K":
        parts.append(fiscal_year)
    if filing_type:
        parts.append(f"Form {filing_type}")
    display_name = " ".join(parts) if parts else chunk.filename

    return Citation(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        filename=chunk.filename,
        collection=chunk.collection,
        source_text=chunk.source_text,
        company=company,
        filing_type=filing_type,
        fiscal_year=fiscal_year,
        display_name=display_name,
    )


# ── Company markers for facet retrieval ─────────────────────────────────────

_COMPANY_MARKERS: dict[str, list[str]] = {
    "microsoft": ["microsoft", "msft", "2025-07-30"],
    "apple": ["apple", "aapl", "2025-10-31"],
    "nvidia": ["nvidia", "nvda", "2026-02-25"],
    "amazon": ["amazon", "amzn", "2026-02-06"],
    "google": ["google", "alphabet", "googl", "2026-02-05"],
    "meta": ["meta platforms", "meta", "fb", "2026-01-29"],
    "tesla": ["tesla", "tsla"],
    "jpmorgan": ["jpmorgan", "jpm", "jpmorgan chase", "2026-02-13"],
    "walmart": ["walmart", "wmt", "2026-03-13"],
    "johnson": ["johnson", "jnj", "2026-02-11"],
    "pfizer": ["pfizer", "pfe", "2026-02-26"],
    "exxon": ["exxon", "xom", "2026-02-18"],
    "disney": ["disney", "dis", "2026-03-16"],
    "coca-cola": ["coca-cola", "coca cola", "ko", "2026-02-20"],
    "visa": ["visa", "2025-11-06"],
}


def _company_markers_for_query(sub_question: str) -> list[str]:
    q_lower = sub_question.lower()
    for key, markers in _COMPANY_MARKERS.items():
        if key in q_lower:
            return markers
    return []


def _filter_by_company(sub_question: str, candidates: list[ChunkResult]) -> list[ChunkResult]:
    markers = _company_markers_for_query(sub_question)
    if not markers:
        return candidates
    matched = [
        c for c in candidates
        if any(m in c.filename.lower() or m in c.doc_id.lower() or m in c.source_text[:400].lower() for m in markers)
    ]
    if matched:
        rest = [c for c in candidates if c not in matched]
        return matched + rest
    return candidates


def _boost_facet_candidates(
    sub_question: str, collection: str, candidates: list[ChunkResult],
) -> list[ChunkResult]:
    q_lower = sub_question.lower()
    signals: list[str] = []
    if collection == "sec-filings":
        if any(w in q_lower for w in ("revenue", "sales", "income", "fiscal", "profit", "earnings")):
            signals = [
                "summary results of operations",
                "consolidated statements of operations",
                "consolidated net sales", "net revenues", "revenue $",
                "net sales", "total revenue", "total net sales",
            ]
        if any(w in q_lower for w in ("risk factor", "risk factors", "risks")):
            risk_signals = [
                "item 1a", "risk factor", "risk factors", "our business is subject",
                "competition", "cybersecurity", "cyber", "supply chain", "regulatory",
                "macroeconomic", "labor", "data privacy", "climate", "liquidity",
                "material adverse", "could materially",
            ]
            signals = risk_signals + signals
    elif collection == "legal-docs":
        if any(w in q_lower for w in ("clawback", "recoupment", "recoup")):
            signals = ["clawback", "recoupment", "recoup", "erroneously awarded", "recovery of"]
        if any(w in q_lower for w in ("terminat", "expiration", "default")):
            signals = ["termination", "terminat", "event of default", "duration, termination"] + signals
        if any(w in q_lower for w in ("indemnif", "liability")):
            signals = ["indemnif", "liability"] + signals
    if not signals:
        return candidates
    boosted = [c for c in candidates if any(s in c.source_text.lower() for s in signals)]
    rest = [c for c in candidates if c not in boosted]
    if boosted:
        logger.info("facet boost %r: promoted %d/%d", collection, len(boosted), len(candidates))
    return boosted + rest


def _ensure_signal_in_top(
    ranked: list[ChunkResult], pool: list[ChunkResult],
    signals: list[str], top_n: int, company_markers: list[str] | None = None,
) -> list[ChunkResult]:
    if any(any(s in c.source_text.lower() for s in signals) for c in ranked):
        return ranked
    for sig in signals:
        for c in pool:
            if sig in c.source_text.lower() and c not in ranked:
                if company_markers and not any(
                    m in c.filename.lower() or m in c.doc_id.lower() or m in c.source_text[:400].lower()
                    for m in company_markers
                ):
                    continue
                ranked = [c] + ranked[:top_n - 1]
                return ranked
    return ranked


def _infer_section_filter(query: str, collection: str) -> str | None:
    """
    Map query intent to a section_type filter for section-aware retrieval.

    Returns None for broad queries or when no single section type dominates.
    The filter is passed to hybrid_search which applies it at both the
    Pinecone metadata level (vector search) and corpus level (BM25).
    """
    q = query.lower()

    if "legal" in collection.lower():
        return None  # legal-docs are all legal_provision, filter adds no value

    if any(w in q for w in (
        "revenue", "net sales", "net income", "earnings", "total income",
        "operating income", "fiscal year", "profit", "gross margin",
        "income statement", "eps", "diluted", "consolidated net",
    )):
        return "income_statement"

    if any(w in q for w in (
        "risk factor", "risk factors", "identified risk", "key risk",
        "main risk", "what risk", "cybersecurity risk", "supply chain risk",
    )):
        return "risk_factors"

    if any(w in q for w in (
        "management discussion", "management's discussion", "md&a",
        "business outlook", "results of operations",
    )):
        return "mda"

    # Comparison with revenue signal → income_statement
    if _is_comparison(query) and any(w in q for w in ("revenue", "sales", "income")):
        return "income_statement"

    return None


def _retrieve_for_facet(
    sub_question: str, collection: str, scopes: list[str], top_n: int = _FACET_TOP_K,
    section_filter: str | None = None,
) -> list[ChunkResult]:
    if section_filter is None:
        section_filter = _infer_section_filter(sub_question, collection)
    candidates = hybrid_search(sub_question, collection, scopes, top_k=40, section_filter=section_filter)
    # Fallback: if section filter yields nothing (vectors not yet backfilled), retry without
    if not candidates and section_filter:
        logger.info("_retrieve_for_facet: section_filter=%r empty, retrying unfiltered", section_filter)
        candidates = hybrid_search(sub_question, collection, scopes, top_k=40)
    markers = _company_markers_for_query(sub_question)
    if markers:
        def _matches(c: ChunkResult, ms: list[str]) -> bool:
            fl = c.filename.lower(); dl = c.doc_id.lower(); tl = c.source_text[:400].lower()
            return any(m in fl or m in dl or m in tl for m in ms)
        scoped = [c for c in candidates if _matches(c, markers)]
        if not scoped:
            company_key = markers[0]
            extra = hybrid_search(f"{company_key} {sub_question}", collection, scopes, top_k=50)
            scoped = [c for c in extra if _matches(c, markers)]
        search_pool = scoped if scoped else candidates
    else:
        search_pool = candidates

    search_pool = _boost_facet_candidates(sub_question, collection, search_pool)
    ranked = rerank(sub_question, search_pool, top_n=top_n)
    q_lower = sub_question.lower()

    if collection == "legal-docs" and any(w in q_lower for w in ("terminat", "expiration", "default")):
        ranked = _ensure_signal_in_top(ranked, search_pool, ["terminat", "termination"], top_n, markers or None)
    if collection == "legal-docs" and any(w in q_lower for w in ("clawback", "recoupment", "recoup")):
        ranked = _ensure_signal_in_top(ranked, search_pool, ["clawback", "recoupment", "recoup"], top_n, markers or None)
    if collection == "sec-filings" and any(w in q_lower for w in ("risk factor", "risk factors", "risks")):
        ranked = _ensure_signal_in_top(
            ranked, search_pool,
            ["risk factor", "item 1a", "competition", "cybersecurity", "supply chain", "regulatory", "macroeconomic"],
            top_n, markers or None,
        )
    if collection == "sec-filings" and any(w in q_lower for w in ("revenue", "sales", "income")):
        ranked = _ensure_signal_in_top(
            ranked, search_pool, ["summary results of operations"], top_n, markers or None,
        )
        _fin_signals = (
            "summary results of operations", "consolidated net sales",
            "consolidated statements of operations", "net revenues", "total net sales", "revenue $",
        )
        has_total_revenue = any(any(sig in c.source_text.lower() for sig in _fin_signals) for c in ranked)
        if not has_total_revenue and markers:
            company = next((k for k in _COMPANY_MARKERS if k in q_lower), "company")
            fallback_q = (
                f"{company} SUMMARY RESULTS OF OPERATIONS consolidated net sales "
                f"total revenue net sales millions income statement fiscal year"
            )
            fb_candidates = hybrid_search(fallback_q, collection, scopes, top_k=50)
            fb_pool = [
                c for c in fb_candidates
                if any(m in c.filename.lower() or m in c.doc_id.lower() or m in c.source_text[:400].lower() for m in markers)
            ]
            fb_pool = _boost_facet_candidates(fallback_q, collection, fb_pool)
            fb_ranked = rerank(fallback_q, fb_pool, top_n=8)
            seen = {c.chunk_id for c in ranked}
            for c in fb_ranked:
                if any(sig in c.source_text.lower() for sig in _fin_signals) and c.chunk_id not in seen:
                    ranked = [c] + [x for x in ranked if x.chunk_id != c.chunk_id][:top_n - 1]
                    break
    return ranked


def _collection_queries_from_state(state: AgentState) -> list[CollectionQuery]:
    raw = state.get("collection_queries") or []
    result: list[CollectionQuery] = []
    for item in raw:
        if isinstance(item, CollectionQuery):
            result.append(item)
        elif isinstance(item, dict) and item.get("collection"):
            result.append(CollectionQuery(
                collection=item["collection"],
                sub_question=item.get("sub_question", ""),
            ))
    return result


# ════════════════════════════════════════════════════════════════════════════
# Node 1: rewrite
# ════════════════════════════════════════════════════════════════════════════

def rewrite_node(state: AgentState) -> dict[str, Any]:
    """Rewrite the question into a self-contained standalone query."""
    t0 = time.time()
    question: str = state["question"]
    history: list[ConversationTurn] = state.get("conversation_history", [])

    if not history:
        return {
            "rewritten_query": question,
            "step_latencies": {"rewrite": round((time.time() - t0) * 1000)},
            **_tok(0, 0),
        }

    recent_turns = "\n".join(
        f"- Q: {t.question}" + (f"\n  → resolved: {t.rewritten_query}" if t.rewritten_query else "")
        for t in history[-3:]
    )
    anchor_year = _anchor_year_from_history(history)
    resolved_question = _apply_relative_year_resolution(question, anchor_year)
    year_hint = (
        f"\nTemporal context: prior turn resolved to year {anchor_year}. "
        f"'Previous year', 'last year' mean {anchor_year - 1}."
        if anchor_year else ""
    )

    prompt = (
        f"Rewrite the follow-up question as a fully standalone question.\n\n"
        f"Prior conversation turns:\n{recent_turns}{year_hint}\n\n"
        f"Follow-up: {resolved_question}\n\n"
        f"Rules:\n"
        f"1. Replace pronouns with explicit entity names from prior queries.\n"
        f"2. Resolve relative time references to actual years.\n"
        f"3. Return only the rewritten question."
    )
    standalone, pt, ct = _chat_with_usage(
        [
            {"role": "system", "content": "You rewrite follow-up questions into standalone questions."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
    )
    standalone = _apply_relative_year_resolution(standalone, anchor_year)
    logger.info("rewrite: %r → %r", question, standalone)
    return {
        "rewritten_query": standalone,
        "step_latencies": {"rewrite": round((time.time() - t0) * 1000)},
        **_tok(pt, ct),
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 1b: classify
# ════════════════════════════════════════════════════════════════════════════

# Valid query types for validation
_VALID_QUERY_TYPES = {
    "factual_lookup", "comparison", "trend_analysis", "calculation",
    "risk_analysis", "multi_document_reasoning", "summarization",
    "out_of_scope", "unclear",
}

_DEFAULT_CLASSIFICATION = {
    "query_type": "factual_lookup",
    "requires_calculation": False,
    "requires_multi_doc": False,
    "requires_graph": False,
    "expected_output_format": "short_answer_with_citation",
    "reason": "Default classification — LLM parse failed.",
}


def classify_node(state: AgentState) -> dict[str, Any]:
    """
    Classify the query into a structured type that downstream nodes use
    to choose retrieval strategy, prompt template, and answer format.

    Classification influences:
      - factual_lookup    → brief answer with citation
      - comparison        → table format, fresh per-entity retrieval
      - trend_analysis    → temporal summary with direction and drivers
      - calculation       → numeric extraction + deterministic calculation
      - risk_analysis     → bullet summary with categories and evidence
      - multi_document_reasoning → multi-facet retrieval
      - out_of_scope      → polite refusal with scope explanation
      - unclear           → clarification request
    """
    t0 = time.time()
    query: str = state.get("rewritten_query") or state["question"]

    # Fast pre-check: if query has zero domain signals, mark out_of_scope without LLM call
    _DOMAIN_SIGNALS = [
        "revenue", "income", "profit", "earnings", "fiscal", "annual", "10-k", "filing",
        "risk", "agreement", "contract", "clause", "indemnif", "terminat", "exhibit",
        "company", "corporation", "shares", "stock", "dividend", "margin", "ebitda",
        "apple", "microsoft", "nvidia", "google", "amazon", "tesla", "jpmorgan",
        "walmart", "disney", "pfizer", "exxon", "coca", "visa", "johnson",
        "aapl", "msft", "nvda", "googl", "amzn", "tsla", "meta", "jpm",
    ]
    q_lower = query.lower()
    has_domain_signal = any(sig in q_lower for sig in _DOMAIN_SIGNALS)
    if not has_domain_signal:
        oos = {
            **_DEFAULT_CLASSIFICATION,
            "query_type": "out_of_scope",
            "expected_output_format": "refusal",
            "reason": "Query contains no financial or legal document signals.",
        }
        return {
            "query_classification": oos,
            "step_latencies": {"classify": round((time.time() - t0) * 1000)},
        }

    prompt = f"""Classify this document query. Return valid JSON only — no markdown, no explanation.

Query: {query}

Return exactly this JSON structure:
{{
  "query_type": "<one of: factual_lookup | comparison | trend_analysis | calculation | risk_analysis | multi_document_reasoning | summarization | out_of_scope | unclear>",
  "requires_calculation": <true|false>,
  "requires_multi_doc": <true|false>,
  "requires_graph": <true|false>,
  "expected_output_format": "<one of: short_answer_with_citation | comparison_table | trend_summary | calculated_result | risk_bullet_list | multi_part_answer | clarification_needed | refusal>",
  "reason": "<one sentence explanation>"
}}

Guidelines:
- comparison: asks to compare two or more entities/periods
- trend_analysis: asks about change over time or year-over-year direction
- calculation: asks for a computed result (growth rate, difference, percentage)
- risk_analysis: asks about risk factors, threats, or vulnerabilities
- out_of_scope: asks about something not in SEC filings or legal docs
- unclear: ambiguous, missing subject or time period"""

    raw, pt, ct = _chat_with_usage(
        [
            {"role": "system", "content": "You are a query classifier. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        temperature=0.0,
    )

    classification = _DEFAULT_CLASSIFICATION.copy()
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:]
        parsed = json.loads(cleaned)
        if parsed.get("query_type") in _VALID_QUERY_TYPES:
            classification = parsed
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning("classify_node: JSON parse failed (%s), using default", exc)

    logger.info("classify: type=%r requires_calc=%s", classification.get("query_type"), classification.get("requires_calculation"))
    return {
        "query_classification": classification,
        "step_latencies": {"classify": round((time.time() - t0) * 1000)},
        **_tok(pt, ct),
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 2: router
# ════════════════════════════════════════════════════════════════════════════

def router_node(state: AgentState) -> dict[str, Any]:
    """Choose the retrieval path (vector | cag | graph) and active collections."""
    t0 = time.time()
    query: str = state.get("rewritten_query") or state["question"]
    collection: str = state["collection"]

    if collection == "auto":
        role = state.get("user_role", "general")
        allowed_set = {c for c, roles in cfg.COLLECTION_ROLES.items() if role in roles}
        # Pass role's allowed collections as a filter so the classifier never picks a
        # namespace the user can't access. Admin is unrestricted (sees all collections).
        classify_allowed = allowed_set if allowed_set and role != "admin" else None
        active = _classify_collections(query, allowed=classify_allowed) or (
            sorted(allowed_set)[:1] if allowed_set else ["sec-filings"]
        )
        collection = active[0]
    else:
        active = [collection]

    # CAG check
    try:
        from backend.ingest.embed_index import _get_pinecone
        import backend.config as _cfg
        pc = _get_pinecone()
        stats = pc.Index(_cfg.PINECONE_INDEX_NAME).describe_index_stats()
        ns_stats = stats.namespaces or {}
        ns_entry = ns_stats.get(collection, {})
        vector_count = (
            ns_entry.get("vector_count", 0) if isinstance(ns_entry, dict)
            else getattr(ns_entry, "vector_count", 0)
        )
        estimated_tokens = (vector_count // 2) * cfg.DEFAULT_CHUNK_CONFIG.child_tokens
        if 0 < estimated_tokens < _CAG_TOKEN_THRESHOLD:
            if _has_named_entity(query):
                logger.info("router: vector (entity-specific — bypassing CAG)")
            else:
                logger.info("router: cag (est. %d tokens)", estimated_tokens)
                return {
                    "route": "cag",
                    "collection": collection,
                    "active_collections": active,
                    "step_latencies": {"router": round((time.time() - t0) * 1000)},
                }
    except Exception as exc:
        logger.warning("router: CAG check failed (%s)", exc)

    # Graph check — trigger for structural/relational queries only.
    # Broad content questions ("summarize risk factors") stay on vector;
    # topology/relationship questions go to graph.
    import re as _re
    finance_collections = {"sec_filings", "sec-filings", "finance", "filings"}
    neo4j_configured = bool(cfg.NEO4J_URI)
    q_lower_router = query.lower()

    _GRAPH_TOPOLOGY_RE = _re.compile(
        r'\bwhich\s+companies?\b'       # which companies have X
        r'|\bwhat\s+companies?\b'       # what companies ...
        r'|\bwhich\s+topics?\b'        # which topics are ...
        r'|\bwhat\s+topics?\b'         # what topics ...
        r'|\bconnected\s+to\b'         # connected to
        r'|\blinked\s+to\b'            # linked to
        r'|\bhow\s+are\b'              # how are X connected
        r'|\bgraph\s+path\b'           # graph path explanation
        r'|\brelationship\s+between\b' # relationship between
        r'|\btopic.*connected\b'       # topic connected to
        r'|\bsegment.*breakdown\b'     # segment breakdown
        r'|\bshow.*segment\b'          # show segments
        r'|\blist.*segment\b'          # list segments
    )
    is_graph_query = (
        _is_comparison(query)
        or bool(_GRAPH_TOPOLOGY_RE.search(q_lower_router))
    )
    if collection in finance_collections and is_graph_query and neo4j_configured:
        logger.info("router: graph")
        return {
            "route": "graph",
            "collection": collection,
            "active_collections": active,
            "step_latencies": {"router": round((time.time() - t0) * 1000)},
        }

    logger.info("router: vector (active=%r)", active)
    return {
        "route": "vector",
        "collection": collection,
        "active_collections": active,
        "step_latencies": {"router": round((time.time() - t0) * 1000)},
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 2b: access_check
# ════════════════════════════════════════════════════════════════════════════

def access_check_node(state: AgentState) -> dict[str, Any]:
    """
    RBAC gate: check that the requesting role is allowed to query each
    active collection. If denied, set access_denied=True so the conditional
    edge in graph.py short-circuits directly to generate_node, which returns
    a structured denial message without touching Pinecone.

    Roles and their allowed collections (from config.COLLECTION_ROLES):
      admin   → all collections
      finance → sec-filings only
      legal   → legal-docs only
      general → sec-filings only
    """
    t0 = time.time()
    role: str = state.get("user_role", "general")
    active: list[str] = state.get("active_collections") or []

    denied: list[str] = []
    for col in active:
        allowed_roles = cfg.COLLECTION_ROLES.get(col, list(cfg.ALL_ROLES))
        if role not in allowed_roles:
            denied.append(col)

    latency = {"access_check": round((time.time() - t0) * 1000)}

    if denied:
        denied_str = ", ".join(denied)
        allowed_for_role = [
            c for c, roles in cfg.COLLECTION_ROLES.items() if role in roles
        ]
        reason = (
            f"Role '{role}' does not have permission to access: {denied_str}. "
            f"Required role(s) for {denied[0]}: {cfg.COLLECTION_ROLES.get(denied[0], [])}. "
            f"Your role ('{role}') has access to: {allowed_for_role or ['none']}."
        )
        logger.warning("access_check: denied role=%r collections=%r", role, denied)
        return {
            "access_denied": True,
            "access_denial_reason": reason,
            "step_latencies": latency,
        }

    logger.info("access_check: role=%r allowed for %r", role, active)
    return {
        "access_denied": False,
        "access_denial_reason": "",
        "step_latencies": latency,
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 2c: decompose (multi-collection)
# ════════════════════════════════════════════════════════════════════════════

def decompose_node(state: AgentState) -> dict[str, Any]:
    """Split a multi-collection question into one focused sub-question per namespace."""
    t0 = time.time()
    query: str = state.get("rewritten_query") or state["question"]
    active: list[str] = (state.get("active_collections") or [])[:2]

    if len(active) < 2:
        return {
            "collection_queries": [],
            "step_latencies": {"decompose": round((time.time() - t0) * 1000)},
        }

    from backend.config import COLLECTION_REGISTRY
    descriptions = "\n".join(
        f'  "{name}": {COLLECTION_REGISTRY[name]}'
        for name in active if name in COLLECTION_REGISTRY
    )
    collections_json = json.dumps(active)
    prompt = (
        f"Split the user's question into exactly {len(active)} focused sub-questions — "
        f"one per collection below. Each sub-question must only ask for information "
        f"that collection contains. Preserve company names and fiscal years.\n\n"
        f"Retrieval hints:\n"
        f'  sec-filings: target "SUMMARY RESULTS OF OPERATIONS", income statement, '
        f'"total Revenue" or "Net sales" in millions for the fiscal year.\n'
        f'  legal-docs: target the relevant legal provision keywords — '
        f'"termination", "event of default" for credit agreements; '
        f'"recoupment", "clawback" for compensation agreements.\n\n'
        f"Collections:\n{descriptions}\n\n"
        f"Original question: {query}\n\n"
        f"Return ONLY a JSON array with {len(active)} objects:\n"
        f'[{{"collection": "<exact name from {collections_json}>", "sub_question": "<focused standalone question>"}}]'
    )
    raw, pt, ct = _chat_with_usage(
        [
            {"role": "system", "content": "You decompose questions into collection-specific sub-questions. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
    )

    collection_queries: list[CollectionQuery] = []
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:]
        items = json.loads(cleaned)
        if isinstance(items, list):
            for item in items:
                col = (item.get("collection") or "").strip()
                sub_q = (item.get("sub_question") or "").strip()
                if col in active and sub_q:
                    collection_queries.append(CollectionQuery(collection=col, sub_question=sub_q))
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning("decompose: JSON parse failed (%s)", exc)

    covered = {cq.collection for cq in collection_queries}
    for col in active:
        if col not in covered:
            collection_queries.append(CollectionQuery(collection=col, sub_question=query))

    return {
        "collection_queries": collection_queries,
        "step_latencies": {"decompose": round((time.time() - t0) * 1000)},
        **_tok(pt, ct),
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 3a: retrieve_vector
# ════════════════════════════════════════════════════════════════════════════

def retrieve_vector_node(state: AgentState) -> dict[str, Any]:
    """Retrieve via hybrid search + Cohere reranking."""
    t0 = time.time()
    query: str = state.get("rewritten_query") or state["question"]
    collection: str = state["collection"]
    scopes: list[str] = state.get("allowed_scopes", ["public"])
    active: list[str] = state.get("active_collections") or [collection]
    facet_queries = _collection_queries_from_state(state)

    if len(facet_queries) >= 2:
        import concurrent.futures
        all_chunks: list[ChunkResult] = []
        seen: set[str] = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(facet_queries)) as pool:
            futures = {
                pool.submit(_retrieve_for_facet, cq.sub_question, cq.collection, scopes): cq
                for cq in facet_queries
            }
            for fut in concurrent.futures.as_completed(futures):
                cq = futures[fut]
                try:
                    ranked = fut.result()
                except Exception as exc:
                    logger.warning("facet %r failed: %s", cq.collection, exc)
                    ranked = []
                for c in ranked:
                    if c.chunk_id not in seen:
                        seen.add(c.chunk_id)
                        all_chunks.append(c)
        return {
            "retrieved_chunks": all_chunks,
            "reusable_chunks": all_chunks,
            "step_latencies": {"retrieve_vector": round((time.time() - t0) * 1000)},
        }

    if len(active) >= 2:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(hybrid_search, query, col, scopes, 15): col for col in active[:2]}
            all_candidates: list[ChunkResult] = []
            for fut in concurrent.futures.as_completed(futures):
                try:
                    all_candidates.extend(fut.result())
                except Exception as exc:
                    logger.warning("search on %r failed: %s", futures[fut], exc)
        seen: set[str] = set()
        deduped = [c for c in all_candidates if not (c.chunk_id in seen or seen.add(c.chunk_id))]  # type: ignore
        reranked = rerank(query, deduped, top_n=8)
    else:
        q_lower = query.lower()
        needs_risk = any(w in q_lower for w in ("risk factor", "risk factors"))
        needs_fin = any(w in q_lower for w in ("revenue", "sales", "income", "profit", "earnings"))
        if needs_risk and needs_fin:
            company_hint = next((k for k in _COMPANY_MARKERS if k in q_lower), query.split()[0] if query.split() else "company")
            fin_sub = f"{company_hint} consolidated net sales total revenue income statement fiscal year summary results of operations"
            risk_sub = f"{company_hint} risk factors Item 1A competition cybersecurity supply chain regulatory material adverse"
            fin_chunks = _retrieve_for_facet(fin_sub, collection, scopes, top_n=6, section_filter="income_statement")
            risk_chunks = _retrieve_for_facet(risk_sub, collection, scopes, top_n=6, section_filter="risk_factors")
            seen: set[str] = set()
            reranked = []
            for c in fin_chunks + risk_chunks:
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    reranked.append(c)
        else:
            reranked = _retrieve_for_facet(query, collection, scopes, top_n=10)

    return {
        "retrieved_chunks": reranked,
        "reusable_chunks": reranked,
        "step_latencies": {"retrieve_vector": round((time.time() - t0) * 1000)},
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 3b: retrieve_cag
# ════════════════════════════════════════════════════════════════════════════

def retrieve_cag_node(state: AgentState) -> dict[str, Any]:
    """Context-stuffing: fetch ALL content in the collection."""
    t0 = time.time()
    from backend.ingest.embed_index import _get_pinecone
    import backend.config as _cfg

    collection: str = state["collection"]
    scopes: list[str] = state.get("allowed_scopes", ["public"])

    try:
        pc = _get_pinecone()
        index = pc.Index(_cfg.PINECONE_INDEX_NAME)
        all_ids: list[str] = []
        for page in index.list(namespace=collection):
            all_ids.extend(item.id for item in page.vectors)

        if not all_ids:
            logger.warning("retrieve_cag_node: empty namespace=%r", collection)
            result = retrieve_vector_node(state)
            result["step_latencies"] = {"retrieve_cag": round((time.time() - t0) * 1000)}
            return result

        parent_chunks: list[ChunkResult] = []
        child_chunks: list[ChunkResult] = []
        batch_size = 200
        for start in range(0, len(all_ids), batch_size):
            resp = index.fetch(ids=all_ids[start:start + batch_size], namespace=collection)
            for vid, vec in resp.vectors.items():
                meta = vec.metadata or {}
                chunk_scopes = meta.get("access_scope", ["public"])
                if not any(s in chunk_scopes for s in scopes):
                    continue
                cr = ChunkResult(
                    chunk_id=vid, parent_id=meta.get("parent_id") or None,
                    doc_id=meta.get("doc_id", ""), filename=meta.get("filename", ""),
                    collection=collection, source_text=meta.get("source_text", ""),
                    is_parent=bool(meta.get("is_parent", False)), score=1.0,
                    metadata={**meta, "_vec": list(vec.values) if vec.values else []},
                )
                (parent_chunks if cr.is_parent else child_chunks).append(cr)

        chunks = parent_chunks if parent_chunks else child_chunks
        if not chunks:
            result = retrieve_vector_node(state)
            result["step_latencies"] = {"retrieve_cag": round((time.time() - t0) * 1000)}
            return result

        # Rank by cosine similarity
        query: str = state.get("rewritten_query") or state["question"]
        try:
            import numpy as np
            model = cfg.get_embedding_model(collection)
            q_emb = _llm().embeddings.create(input=[query], model=model).data[0].embedding
            q_vec = np.array(q_emb, dtype=np.float32)
            q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
            scored = []
            for c in chunks:
                cached = c.metadata.get("_vec") if c.metadata else None
                if cached:
                    v = np.array(cached, dtype=np.float32)
                else:
                    v = np.array(_llm().embeddings.create(input=[c.source_text[:512]], model=model).data[0].embedding, dtype=np.float32)
                v_norm = v / (np.linalg.norm(v) + 1e-9)
                scored.append((float(np.dot(q_norm, v_norm)), c))
            chunks = [c for _, c in sorted(scored, key=lambda x: x[0], reverse=True)]
        except Exception as exc:
            logger.warning("retrieve_cag_node: ranking failed (%s)", exc)

        return {
            "retrieved_chunks": chunks,
            "reusable_chunks": chunks,
            "step_latencies": {"retrieve_cag": round((time.time() - t0) * 1000)},
        }
    except Exception as exc:
        logger.warning("retrieve_cag_node: failed (%s)", exc)
        result = retrieve_vector_node(state)
        result["step_latencies"] = {"retrieve_cag": round((time.time() - t0) * 1000)}
        return result


# ════════════════════════════════════════════════════════════════════════════
# Node 3c: retrieve_graph
# ════════════════════════════════════════════════════════════════════════════

def retrieve_graph_node(state: AgentState) -> dict[str, Any]:
    """
    Hybrid retrieval: graph + vector in parallel, merged.

    Graph contributes exact structured facts (metrics, risk factors, relationships).
    Vector contributes supporting narrative evidence and source-grounded citations.
    Graph results go first so the LLM sees precise facts before narrative context.
    """
    import concurrent.futures
    t0 = time.time()
    from backend.graph_rag.query import graph_query
    from backend.graph_rag.schema import get_schema

    query: str = state.get("rewritten_query") or state["question"]
    collection: str = state["collection"]
    scopes: list[str] = state.get("allowed_scopes", ["public"])
    schema = get_schema(collection)

    from backend.graph_rag.query import GraphQueryResult
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        graph_future = pool.submit(graph_query, query, schema, collection)
        vector_future = pool.submit(_retrieve_for_facet, query, collection, scopes, 6)
        try:
            graph_result: GraphQueryResult = graph_future.result()
            graph_chunks = graph_result.chunks if graph_result else []
        except Exception as exc:
            logger.warning("retrieve_graph_node: graph query failed: %s", exc)
            graph_result = None
            graph_chunks = []
        try:
            vector_results = vector_future.result() or []
        except Exception as exc:
            logger.warning("retrieve_graph_node: vector query failed: %s", exc)
            vector_results = []

    # Merge: graph facts first (precise structure), then vector evidence (source grounding)
    seen: set[str] = set()
    merged: list[ChunkResult] = []
    for c in graph_chunks + vector_results:
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            merged.append(c)

    query_type = graph_result.query_type if graph_result else "lookup"
    unsupported_count = graph_result.unsupported_count if graph_result else 0

    logger.info(
        "retrieve_graph_node: type=%s graph=%d vector=%d merged=%d unsupported=%d",
        query_type, len(graph_chunks), len(vector_results), len(merged), unsupported_count,
    )

    if not merged:
        logger.warning("retrieve_graph_node: both graph and vector returned nothing")

    return {
        "retrieved_chunks": merged,
        "reusable_chunks": merged,
        "graph_chunk_count": len(graph_chunks),
        "graph_query_type": query_type,
        "graph_unsupported_count": unsupported_count,
        "step_latencies": {"retrieve_graph": round((time.time() - t0) * 1000)},
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 4: grade
# ════════════════════════════════════════════════════════════════════════════

def _detect_conflicts(chunks: list[ChunkResult], query: str) -> tuple[bool, str]:
    """
    Detect conflicting numeric values for the same metric across chunks.
    Returns (conflict_detected, reason_string).
    """
    # Extract (value, year) pairs for financial metrics
    # Pattern: dollar amount followed by "million" or "billion" + optional year
    amount_pattern = re.compile(
        r"\$?([\d,]+(?:\.\d+)?)\s*(million|billion|M|B)\b",
        re.IGNORECASE,
    )
    year_pattern = re.compile(r"\b(20\d{2}|19\d{2})\b")

    # Normalize to millions
    def normalize(value: float, unit: str) -> float:
        u = unit.lower()
        if u in ("billion", "b"):
            return value * 1000
        return value

    def _company_key(c: "ChunkResult") -> str:
        fn = c.filename.lower()
        for key in _COMPANY_MARKERS:
            if key in fn or key in c.doc_id.lower():
                return key
        return c.doc_id or c.filename

    chunk_values: list[tuple[float, str, str]] = []
    for c in chunks[:6]:
        amounts = amount_pattern.findall(c.source_text)
        years = year_pattern.findall(c.source_text)
        year = years[0] if years else "unknown"
        company = _company_key(c)
        for val_str, unit in amounts:
            try:
                val = normalize(float(val_str.replace(",", "")), unit)
                if val > 1000:  # only flag values that look like revenue-scale numbers
                    chunk_values.append((val, year, company))
            except ValueError:
                pass

    if len(chunk_values) < 2:
        return False, ""

    # Only flag conflict when the same company has diverging values for the same year.
    from collections import defaultdict
    by_company_year: dict[tuple[str, str], list[float]] = defaultdict(list)
    for val, year, company in chunk_values:
        by_company_year[(company, year)].append(val)

    for (company, year), vals in by_company_year.items():
        if year == "unknown":
            continue
        unique_vals = list(set(round(v, -2) for v in vals))  # round to nearest 100M
        if len(unique_vals) >= 2:
            min_v, max_v = min(unique_vals), max(unique_vals)
            if max_v > 0 and (max_v - min_v) / max_v > 0.15:  # >15% difference
                return True, (
                    f"Multiple different values found for {company} in {year}: "
                    f"${min_v:,.0f}M and ${max_v:,.0f}M. "
                    f"These may refer to different periods, segments, or amended filings."
                )

    return False, ""


def grade_node(state: AgentState) -> dict[str, Any]:
    """
    Grade retrieved context quality with conflict detection and answerability assessment.

    Returns:
      grade           — 'sufficient' | 'insufficient'
      answerability   — 'sufficient' | 'insufficient' | 'conflicting'
      conflict_detected — bool
      missing_info    — list of what's absent
    """
    t0 = time.time()
    query: str = state.get("rewritten_query") or state["question"]
    chunks: list[ChunkResult] = state.get("retrieved_chunks", [])
    retry_count: int = state.get("retry_count", 0)
    facet_queries = _collection_queries_from_state(state)

    if not chunks:
        return {
            "grade": "insufficient",
            "answerability": "insufficient",
            "answerability_reason": "No context was retrieved from the document store.",
            "missing_info": ["No relevant documents found"],
            "conflict_detected": False,
            "retry_count": retry_count + 1,
            "facet_grades": [],
            "step_latencies": {"grade": round((time.time() - t0) * 1000)},
        }

    # Conflict detection
    conflict_detected, conflict_reason = _detect_conflicts(chunks, query)

    if len(facet_queries) >= 2:
        facet_grades: list[dict[str, Any]] = []
        total_pt = total_ct = 0
        all_sufficient = True

        for cq in facet_queries:
            facet_chunks = [c for c in chunks if c.collection == cq.collection]
            if not facet_chunks:
                facet_grades.append({
                    "collection": cq.collection, "sub_question": cq.sub_question,
                    "grade": "insufficient", "chunk_count": 0,
                })
                all_sufficient = False
                continue

            context = _chunks_to_context(facet_chunks, max_chunks=_FACET_TOP_K)
            verdict, pt, ct = _chat_with_usage(
                [
                    {"role": "system", "content": "Retrieval grader. Reply with exactly: 'sufficient' or 'insufficient'."},
                    {"role": "user", "content": f"Question: {cq.sub_question}\n\nPassages:\n{context}"},
                ],
                max_tokens=10,
            )
            total_pt += pt; total_ct += ct
            facet_grade = "sufficient" if "sufficient" in verdict.lower() else "insufficient"
            if facet_grade != "sufficient":
                all_sufficient = False
            facet_grades.append({
                "collection": cq.collection, "sub_question": cq.sub_question,
                "grade": facet_grade, "chunk_count": len(facet_chunks),
            })

        grade = "sufficient" if all_sufficient else "insufficient"
        answerability = "conflicting" if conflict_detected else grade
        return {
            "grade": grade,
            "answerability": answerability,
            "answerability_reason": conflict_reason if conflict_detected else "",
            "conflict_detected": conflict_detected,
            "missing_info": [
                f"{fg['collection']}: insufficient context" for fg in facet_grades if fg["grade"] == "insufficient"
            ],
            "retry_count": retry_count + 1,
            "facet_grades": facet_grades,
            "step_latencies": {"grade": round((time.time() - t0) * 1000)},
            **_tok(total_pt, total_ct),
        }

    # Single-collection grading
    context = _chunks_to_context(chunks, max_chunks=8)
    verdict, pt, ct = _chat_with_usage(
        [
            {"role": "system", "content": (
                "Retrieval grader. Given a question and passages, decide if the passages "
                "contain enough information to answer it. "
                "Reply with exactly one word: 'sufficient' or 'insufficient'."
            )},
            {"role": "user", "content": f"Question: {query}\n\nPassages:\n{context}"},
        ],
        max_tokens=10,
    )
    grade = "sufficient" if "sufficient" in verdict.lower() else "insufficient"
    answerability = "conflicting" if conflict_detected else grade

    # Determine missing info for insufficient grade
    missing_info: list[str] = []
    if grade == "insufficient":
        years = re.findall(r"\b(20\d{2})\b", query)
        if years:
            missing_info.append(f"Data for fiscal year {', '.join(years)} not found in indexed documents")
        else:
            missing_info.append("Relevant passages not found in the indexed document corpus")

    logger.info("grade: %s answerability=%s conflict=%s retry=%d", grade, answerability, conflict_detected, retry_count)
    return {
        "grade": grade,
        "answerability": answerability,
        "answerability_reason": conflict_reason if conflict_detected else "",
        "conflict_detected": conflict_detected,
        "missing_info": missing_info,
        "retry_count": retry_count + 1,
        "facet_grades": [],
        "step_latencies": {"grade": round((time.time() - t0) * 1000)},
        **_tok(pt, ct),
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 4b: validate_numbers
# ════════════════════════════════════════════════════════════════════════════

def validate_numbers_node(state: AgentState) -> dict[str, Any]:
    """
    Deterministic numeric validation for financial calculation queries.

    Workflow:
      1. Ask LLM to extract numeric values from retrieved chunks (structured JSON).
      2. Perform all arithmetic in Python — never let the LLM do math.
      3. Return structured validation result that generate_node uses to
         produce a precise, citable answer with explicit formula shown.

    This prevents the LLM from silently computing wrong percentages or
    rounding figures incorrectly.
    """
    t0 = time.time()
    query: str = state.get("rewritten_query") or state["question"]
    chunks: list[ChunkResult] = state.get("retrieved_chunks", [])
    context = _chunks_to_context(chunks, max_chunks=8)

    extraction_prompt = f"""Extract ALL numeric financial values from the context below.
Return valid JSON only.

Context:
{context}

Question: {query}

Return this JSON structure (multiple periods/companies if present):
{{
  "metric": "<revenue | net_income | operating_income | eps | other>",
  "company": "<company name>",
  "periods": [
    {{
      "label": "<e.g. FY2024 or Q3 2025>",
      "year": <integer year>,
      "value": <numeric value in millions USD>,
      "unit": "millions USD",
      "source_chunk_idx": <0-based index of source chunk>
    }}
  ]
}}

Rules:
- Convert billions to millions (multiply by 1000)
- Extract from comparison columns in tables (prior year data)
- If multiple metrics present, extract the one most relevant to the question
- If no numeric values found, return {{"metric": "unknown", "company": "", "periods": []}}"""

    raw, pt, ct = _chat_with_usage(
        [
            {"role": "system", "content": "You extract financial numbers from text. Return valid JSON only."},
            {"role": "user", "content": extraction_prompt},
        ],
        max_tokens=400,
        temperature=0.0,
    )

    extraction: dict = {"metric": "unknown", "company": "", "periods": []}
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```", 2)[1]
            if cleaned.lstrip().startswith("json"):
                cleaned = cleaned.lstrip()[4:]
        extraction = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError, AttributeError) as exc:
        logger.warning("validate_numbers_node: extraction parse failed (%s)", exc)

    # Deterministic calculation in Python
    periods = extraction.get("periods", [])
    calculation: dict = {}

    if len(periods) >= 2:
        # Sort by year to get chronological order
        sorted_periods = sorted(periods, key=lambda p: p.get("year", 0))
        earlier = sorted_periods[0]
        later = sorted_periods[-1]

        v_earlier = float(earlier.get("value", 0))
        v_later = float(later.get("value", 0))

        if v_earlier > 0:
            abs_change = v_later - v_earlier
            pct_change = (abs_change / v_earlier) * 100
            calculation = {
                "from_period": earlier.get("label", ""),
                "to_period": later.get("label", ""),
                "from_value": v_earlier,
                "to_value": v_later,
                "absolute_change": round(abs_change, 2),
                "percentage_change": round(pct_change, 2),
                "formula": (
                    f"({v_later:,.0f} - {v_earlier:,.0f}) / {v_earlier:,.0f} × 100 "
                    f"= {pct_change:+.2f}%"
                ),
                "direction": "increase" if abs_change > 0 else "decrease",
            }
            logger.info(
                "validate_numbers: %s %s→%s Δ%.1f%%",
                extraction.get("metric"), earlier.get("label"), later.get("label"), pct_change,
            )

    numeric_validation = {
        "metric": extraction.get("metric", "unknown"),
        "company": extraction.get("company", ""),
        "periods": periods,
        "calculation": calculation,
        "validated": bool(calculation),
    }

    return {
        "numeric_validation": numeric_validation,
        "step_latencies": {"validate_numbers": round((time.time() - t0) * 1000)},
        **_tok(pt, ct),
    }


# ════════════════════════════════════════════════════════════════════════════
# Node 5: generate
# ════════════════════════════════════════════════════════════════════════════

# Query-type → system prompt template
_QT_SYSTEM_PROMPTS: dict[str, str] = {
    "factual_lookup": (
        "You are a precise financial and legal document analyst. "
        "Answer the question DIRECTLY with a single clear answer including: "
        "the company or entity name, the metric or topic being reported, the exact value, its unit, "
        "the fiscal period, and the source. "
        "Be concise. Cite the specific document section."
    ),
    "comparison": (
        "You are a financial analyst producing a structured comparison. "
        "Present results in a MARKDOWN TABLE with columns for the entities/periods compared. "
        "After the table, add 2-3 sentences interpreting the comparison. "
        "Cite the source document for each figure."
    ),
    "trend_analysis": (
        "You are a financial analyst. Answer the question DIRECTLY in 1-2 sentences — "
        "state the key figure, direction of change, and prior year value from the context. "
        "Only include drivers or explanations if they are explicitly stated in the retrieved text. "
        "Do NOT add context from your training data. Cite the source document."
    ),
    "calculation": (
        "You are a financial analyst presenting a calculated result. "
        "Lead with the DIRECT ANSWER: state the result and its unit in the first sentence. "
        "Then show the values used and the formula — sourced only from the retrieved context. "
        "If a pre-computed calculation is provided in the context, use it exactly. "
        "Never perform arithmetic mentally. Never add explanations not in the context."
    ),
    "risk_analysis": (
        "You are a risk analyst. Present risks as a BULLET LIST grouped by category. "
        "For each risk: state the category, describe the risk using only language from "
        "the retrieved text, and cite the source section. "
        "Do not add risk factors from your training knowledge."
    ),
    "summarization": (
        "You are a document summarizer. Produce a structured summary covering "
        "key facts, financial highlights, and notable points — all drawn strictly from "
        "the retrieved context. Use headers. Cite sources for specific claims. "
        "Do not add background knowledge."
    ),
    "multi_document_reasoning": (
        "You are a cross-document analyst. Answer the question by synthesizing "
        "information strictly from the provided sources. Clearly indicate which document "
        "each piece of information comes from. Use headers when sources are distinct. "
        "Do not add information from your training data."
    ),
}

_EXTRACTIVE_RULES = (
    "\n\nEXTRACTION RULES (always apply):\n"
    "1. LEGAL PROVISIONS: Extract ALL relevant clauses verbatim or in close paraphrase — "
    "even references to external policies by name are valid answers.\n"
    "2. FINANCIAL DATA: In 10-K filings, prior-year columns contain historical data. "
    "Extract figures for the requested year from the correct column. "
    "Always prefer CONSOLIDATED totals over segment figures unless asked.\n"
    "3. RISK FACTORS: Report all risk categories mentioned, even high-level ones.\n"
    "4. NEVER say 'no information' if ANY relevant text exists. Report what IS there.\n"
    "5. Never fabricate numbers, names, or terms not in the context.\n"
    "6. GROUNDING: Every factual claim in your answer MUST be traceable to the provided "
    "context. Do NOT add explanations, drivers, or background knowledge from your training "
    "data. If the context does not explain WHY something happened, do not speculate — "
    "omit the explanation entirely or say 'the filing does not state the reason'."
)


def generate_node(state: AgentState) -> dict[str, Any]:
    """
    Generate the final answer, using query classification to choose the right
    prompt style and numeric_validation results for calculation queries.

    Handles four special cases before normal generation:
      1. access_denied   → structured denial message
      2. out_of_scope    → scope explanation
      3. unclear         → clarification request
      4. conflicting     → surfaces the conflict with explanation
    """
    t0 = time.time()
    query: str = state.get("rewritten_query") or state["question"]
    collection: str = state["collection"]
    scopes: list[str] = state.get("allowed_scopes", ["public"])
    current_chunks: list[ChunkResult] = state.get("retrieved_chunks", [])
    prior_chunks: list[ChunkResult] = state.get("reusable_chunks", [])
    route: str = state.get("route", "vector")
    facet_queries = _collection_queries_from_state(state)
    qc: dict = state.get("query_classification") or {}
    query_type: str = qc.get("query_type", "factual_lookup")
    step_latencies: dict = state.get("step_latencies") or {}

    # ── Case 1: access denied ────────────────────────────────────────────────
    if state.get("access_denied"):
        reason = state.get("access_denial_reason", "Access denied.")
        answer = (
            f"**Access Denied**\n\n{reason}\n\n"
            f"Please contact your administrator if you need access to additional collections."
        )
        return _generate_result(answer, [], state, t0, step_latencies, 0, 0)

    # ── Case 2: out_of_scope ─────────────────────────────────────────────────
    if query_type == "out_of_scope":
        answer = (
            "**Out of scope**\n\n"
            "This question falls outside the indexed document collections. "
            "The available collections cover:\n"
            "- **sec-filings**: SEC 10-K annual reports (AAPL, MSFT, NVDA, GOOGL, AMZN, TSLA, META, JPM, WMT, JNJ, PFE, XOM, DIS, KO, V)\n"
            "- **legal-docs**: Material contracts and EX-10 exhibits (TSLA, MSFT, JPM, META, WMT)\n\n"
            f"Your question: *{query}*\n\n"
            "Please rephrase to focus on financials, risk factors, MD&A, or legal agreements for the companies above."
        )
        return _generate_result(answer, [], state, t0, step_latencies, 0, 0)

    # ── Case 3: unclear ──────────────────────────────────────────────────────
    if query_type == "unclear":
        answer = (
            "**Clarification needed**\n\n"
            f"Your question *\"{query}\"* is missing some details. Please specify:\n"
            "- Which company (e.g. Microsoft, Apple, JPMorgan)?\n"
            "- Which fiscal year or period?\n"
            "- What specific metric or topic (revenue, risk factors, clawback conditions)?\n\n"
            "Example: *\"What was Microsoft's total revenue in fiscal year 2025?\"*"
        )
        return _generate_result(answer, [], state, t0, step_latencies, 0, 0)

    # ── Gather context ───────────────────────────────────────────────────────
    original_question: str = state["question"]
    if (_is_comparison(query) or _is_comparison(original_question)) and len(facet_queries) < 2:
        context_chunks = _comparison_retrieval(query, collection, scopes)
    else:
        seen: set[str] = set()
        merged: list[ChunkResult] = []
        for c in current_chunks + prior_chunks:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                merged.append(c)
        context_chunks = merged if route == "cag" else merged[:8]

    if not context_chunks:
        missing = state.get("missing_info") or []
        missing_str = "; ".join(missing) if missing else "the indexed documents"
        answer = (
            f"**Insufficient context**\n\n"
            f"The available documents do not contain enough evidence to answer this confidently.\n\n"
            f"Missing: {missing_str}\n\n"
            f"*Original question: {query}*"
        )
        return _generate_result(answer, [], state, t0, step_latencies, 0, 0)

    # ── Case 4: conflicting context ──────────────────────────────────────────
    conflict_detected = state.get("conflict_detected", False)
    conflict_reason = state.get("answerability_reason", "")

    # ── Numeric validation injection ─────────────────────────────────────────
    numeric_validation: dict = state.get("numeric_validation") or {}
    numeric_context_block = ""
    if numeric_validation.get("validated") and numeric_validation.get("calculation"):
        calc = numeric_validation["calculation"]
        periods = numeric_validation.get("periods", [])
        periods_str = "\n".join(
            f"  - {p.get('label', '')}: ${p.get('value', 0):,.0f}M ({p.get('unit', '')})"
            for p in periods
        )
        numeric_context_block = (
            f"\n\n[VALIDATED CALCULATION — use these exact figures]\n"
            f"Metric: {numeric_validation.get('metric', '')}\n"
            f"Company: {numeric_validation.get('company', '')}\n"
            f"Extracted values:\n{periods_str}\n"
            f"Formula: {calc.get('formula', '')}\n"
            f"Result: {calc.get('direction', '')} of {abs(calc.get('percentage_change', 0)):.2f}% "
            f"(${abs(calc.get('absolute_change', 0)):,.0f}M absolute change)\n"
        )

    # ── Build prompt ─────────────────────────────────────────────────────────
    if len(facet_queries) >= 2:
        sections: list[str] = []
        for i, cq in enumerate(facet_queries, 1):
            facet_chunks = [c for c in context_chunks if c.collection == cq.collection]
            section_ctx = _chunks_to_context(facet_chunks) if facet_chunks else "(no passages retrieved)"
            sections.append(
                f"### Part {i}: {cq.collection}\n"
                f"Sub-question: {cq.sub_question}\n\n"
                f"Context:\n{section_ctx}"
            )
        context = "\n\n".join(sections) + numeric_context_block
        system_prompt = (
            "You are a precise financial and legal document analyst answering a multi-part question. "
            "Answer EACH part separately using ONLY its own context section. "
            "Use markdown headers matching the part labels."
            + _EXTRACTIVE_RULES
        )
        if conflict_detected:
            system_prompt += (
                f"\n\nWARNING — CONFLICTING VALUES DETECTED: {conflict_reason} "
                f"Present both values to the user and note the discrepancy. Do NOT merge them."
            )
        user_content = f"{context}\n\nOriginal question: {query}\n\nProvide a structured answer addressing every part."

    elif route == "graph":
        # Split graph facts from vector evidence so citations only reference source passages
        graph_chunks = [c for c in context_chunks if c.chunk_id.startswith("graph:")]
        vector_chunks = [c for c in context_chunks if not c.chunk_id.startswith("graph:")]
        graph_ctx = _chunks_to_context(graph_chunks) if graph_chunks else "(no graph facts retrieved)"
        vector_ctx = _chunks_to_context(vector_chunks) if vector_chunks else "(no source passages retrieved)"
        context = (
            f"GRAPH_FACTS (from Neo4j knowledge graph — use for structure and relationships):\n{graph_ctx}"
            f"\n\nSOURCE_EVIDENCE (verbatim document passages — cite only from this section):\n{vector_ctx}"
        ) + numeric_context_block
        system_prompt = (
            "You are a precise financial analyst answering questions using a hybrid knowledge graph + document retrieval system. "
            "Two context sections are provided:\n"
            "1. GRAPH_FACTS — structured relational data from Neo4j (metrics, segments, risk topics, entity paths). Use this for relationship structure, entity connections, and precise numeric facts.\n"
            "2. SOURCE_EVIDENCE — verbatim passages from source documents. Cite ONLY from SOURCE_EVIDENCE.\n\n"
            "Rules:\n"
            "- GRAPH_FACTS may contain [UNSUPPORTED — no source chunk] markers; treat those facts as unverified and do not cite them.\n"
            "- Always ground your answer in SOURCE_EVIDENCE. If SOURCE_EVIDENCE contradicts GRAPH_FACTS, surface both values.\n"
            "- Never fabricate. If context is insufficient, say so."
            + _EXTRACTIVE_RULES
        )
        if conflict_detected:
            system_prompt += (
                f"\n\nWARNING — CONFLICTING VALUES DETECTED: {conflict_reason} "
                f"Present both values and note the discrepancy."
            )
        user_content = f"{context}\n\nQuestion: {query}"
        # Citations come from vector (source-grounded) chunks only — not raw graph facts
        context_chunks = vector_chunks if vector_chunks else context_chunks

    else:
        context = _chunks_to_context(context_chunks, group_by_doc=(route == "cag")) + numeric_context_block
        # Legal-docs with multiple chunks → synthesize ALL clauses, not just the top one
        effective_query_type = (
            "multi_document_reasoning"
            if collection == "legal-docs" and len(context_chunks) > 1
            else query_type
        )
        base_prompt = _QT_SYSTEM_PROMPTS.get(effective_query_type, _QT_SYSTEM_PROMPTS["factual_lookup"])
        if route == "cag":
            system_prompt = (
                "You are a precise, helpful assistant with access to a document collection. "
                "Read EVERY passage before composing your answer. "
                "Deduplicate: same employer mentioned in multiple sections → merge into ONE entry. "
                "Never fabricate."
            )
        else:
            system_prompt = base_prompt + _EXTRACTIVE_RULES
            if conflict_detected:
                system_prompt += (
                    f"\n\nWARNING — CONFLICTING VALUES DETECTED: {conflict_reason} "
                    f"Present both values and note the discrepancy instead of picking one."
                )
        user_content = f"Context:\n{context}\n\nQuestion: {query}"

    answer, gen_pt, gen_ct = _chat_with_usage(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=2000,
        temperature=0.1,
    )

    # Prepend conflict notice to answer if needed
    if conflict_detected and "CONFLICTING" not in answer[:200]:
        answer = (
            f"> **Note:** Conflicting values detected in retrieved context. "
            f"{conflict_reason}\n\n"
        ) + answer

    citations = [_build_citation(c) for c in context_chunks]
    return _generate_result(answer, citations, state, t0, step_latencies, gen_pt, gen_ct)


def _generate_result(
    answer: str,
    citations: list[Citation],
    state: AgentState,
    t0: float,
    step_latencies: dict,
    gen_pt: int,
    gen_ct: int,
) -> dict[str, Any]:
    """Assemble the generate node return dict including metrics."""
    gen_latency_ms = round((time.time() - t0) * 1000)
    all_latencies = {**step_latencies, "generate": gen_latency_ms}

    # Aggregate token counts from state + this call
    prior_input = state.get("input_tokens", 0)
    prior_output = state.get("output_tokens", 0)
    total_input = prior_input + gen_pt
    total_output = prior_output + gen_ct

    retrieve_ms = (
        all_latencies.get("retrieve_vector")
        or all_latencies.get("retrieve_cag")
        or all_latencies.get("retrieve_graph")
        or 0
    )

    metrics = {
        "total_latency_ms": sum(v for v in all_latencies.values()),
        "retrieve_latency_ms": retrieve_ms,
        "rerank_latency_ms": 0,  # included in retrieve
        "graph_latency_ms": all_latencies.get("retrieve_graph", 0),
        "generate_latency_ms": gen_latency_ms,
        "validate_latency_ms": all_latencies.get("validate_numbers", 0),
        "model": cfg.DEFAULT_LLM_MODEL,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "estimated_cost_usd": round(cfg.estimate_cost(total_input, total_output), 6),
        "chunk_count": len(state.get("retrieved_chunks", [])),
        "citation_count": len(citations),
        "step_latencies": all_latencies,
    }

    new_turn = ConversationTurn(
        question=state["question"],
        rewritten_query=state.get("rewritten_query"),
        retrieved_chunks=state.get("retrieved_chunks", []),
    )

    return {
        "answer": answer,
        "citations": citations,
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "conversation_history": [new_turn],
        "metrics": metrics,
        "step_latencies": {"generate": gen_latency_ms},
        **_tok(gen_pt, gen_ct),
    }


def _comparison_retrieval(
    query: str, collection: str, scopes: list[str]
) -> list[ChunkResult]:
    """Decompose comparison query → retrieve each subject fresh."""
    raw = _chat(
        [{"role": "user", "content": (
            f"Decompose this comparison question into exactly 2 simple sub-questions, "
            f"one for each subject. Return only 2 questions, one per line.\n\nQuestion: {query}"
        )}],
        max_tokens=150,
    )
    sub_queries = [q.strip().lstrip("12.-) ") for q in raw.strip().splitlines() if q.strip()][:2]
    if len(sub_queries) < 2:
        sub_queries = [query]

    all_chunks: list[ChunkResult] = []
    seen: set[str] = set()
    for sq in sub_queries:
        fresh = _retrieve_for_facet(sq, collection, scopes, top_n=5)
        for c in fresh:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                all_chunks.append(c)
    return all_chunks
