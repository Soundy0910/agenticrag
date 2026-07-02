"""
backend/graph_rag/extract.py

LLM-based entity extraction from SEC filing text + Neo4j upsert.

Flow:
  filing text → LLM → structured JSON → normalize IDs → upsert to Neo4j

Schema (v2):
  Company -[FILED]-> Filing -[HAS_SEGMENT]-> BusinessSegment
  BusinessSegment -[REPORTED_METRIC]-> Metric -[SUPPORTED_BY]-> Chunk
  Filing -[HAS_RISK_FACTOR]-> RiskFactor -[RELATED_TO_TOPIC]-> Topic
  RiskFactor -[SUPPORTED_BY]-> Chunk
  Filing -[HAS_CHUNK]-> Chunk

Every Metric and RiskFactor links back to a Chunk so answers can be
grounded in source text (citations).
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

from openai import OpenAI

import backend.config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction result types
# ---------------------------------------------------------------------------

@dataclass
class ExtractionResult:
    doc_id: str
    collection: str
    ticker: str
    company_name: str
    company_id: str
    filing_id: str
    fiscal_year: int | None
    filing_type: str
    filing_date: str
    segments: list[dict] = field(default_factory=list)     # [{segment_id, name}]
    metrics: list[dict] = field(default_factory=list)      # [{metric_id, name, value, unit, ...}]
    risk_factors: list[dict] = field(default_factory=list) # [{risk_id, title, summary, topics, ...}]
    topics: list[dict] = field(default_factory=list)       # [{topic_id, name}]
    chunks: list[dict] = field(default_factory=list)       # [{chunk_id, section, text_preview}]

    @property
    def entity_count(self) -> int:
        return (1 + 1 + len(self.segments) + len(self.metrics)
                + len(self.risk_factors) + len(self.topics) + len(self.chunks))

    @property
    def relationship_count(self) -> int:
        return (1  # Company->Filing
                + len(self.segments)          # Filing->BusinessSegment
                + len(self.metrics)           # BusinessSegment->Metric + Metric->Chunk
                + len(self.risk_factors)      # Filing->RiskFactor + RiskFactor->Chunk
                + sum(len(r["topic_ids"]) for r in self.risk_factors)  # RiskFactor->Topic
                + len(self.chunks))           # Filing->Chunk


# ---------------------------------------------------------------------------
# ID normalization — deterministic, no LLM involvement
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    """Lowercase, strip special chars, collapse spaces to underscores."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _short_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def company_id(ticker: str) -> str:
    return f"company:{ticker.lower()}"


def filing_id(ticker: str, filing_type: str, fiscal_year: int | str) -> str:
    return f"filing:{ticker.lower()}:{_slug(filing_type)}:{fiscal_year}"


def segment_id(ticker: str, segment_name: str) -> str:
    return f"segment:{ticker.lower()}:{_slug(segment_name)}"


def metric_id(ticker: str, metric_name: str, fiscal_year: int | str, segment: str = "") -> str:
    seg_part = f":{_slug(segment)}" if segment else ""
    return f"metric:{ticker.lower()}:{_slug(metric_name)}:{fiscal_year}{seg_part}"


def risk_id(ticker: str, fiscal_year: int | str, title: str, source_chunk_id: str) -> str:
    return f"risk:{ticker.lower()}:{fiscal_year}:{_short_hash(title + source_chunk_id)}"


def topic_id(topic_name: str) -> str:
    return f"topic:{_slug(topic_name)}"


def chunk_id(doc_id: str, section: str) -> str:
    return f"chunk:{_short_hash(doc_id + section)}"


# ---------------------------------------------------------------------------
# LLM extraction
# ---------------------------------------------------------------------------

_openai: OpenAI | None = None


def _llm() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=cfg.OPENAI_API_KEY)
    return _openai


_SYSTEM = """\
You are a structured data extractor for SEC 10-K filings.
Extract ONLY what is explicitly stated in the text — do not infer or hallucinate.
Return ONLY valid JSON matching the schema below. No explanation, no markdown.

JSON structure:
{
  "company": {"name": "<string>", "ticker": "<string>"},
  "filing": {"filing_type": "<10-K|10-Q|...>", "fiscal_year": <int>, "filing_date": "<YYYY-MM-DD or null>"},
  "business_segments": [{"name": "<string>"}],
  "metrics": [
    {
      "name": "<Revenue|Net Income|Operating Income|...>",
      "value": <float>,
      "unit": "<millions USD|billions USD|...>",
      "fiscal_year": <int>,
      "period": "<FY2024|Q4 2024|...>",
      "segment": "<segment name or null>"
    }
  ],
  "risk_factors": [
    {
      "title": "<short title>",
      "summary": "<1-2 sentence summary>",
      "topics": ["<AI|Cloud|Cybersecurity|Competition|Regulation|Data Privacy|Supply Chain|...>"],
      "affected_segments": ["<segment name or null>"]
    }
  ],
  "topics": ["<topic name>"]
}

Rules:
- Metric values must be numeric (float). Strip currency symbols.
- Do not invent metrics, years, or risk factors not in the text.
- If unsure, return empty arrays.
- financial values must preserve unit exactly as written.
"""


def _extract_llm(text: str, doc_id: str) -> dict:
    """Call LLM and return parsed JSON. Returns empty dict on failure."""
    try:
        resp = _llm().chat.completions.create(
            model=cfg.DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"Extract from this SEC filing text:\n\"\"\"\n{text[:6000]}\n\"\"\""},
            ],
            max_tokens=2000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as exc:
        logger.error("extract_llm: LLM/parse error for doc %r: %s", doc_id, exc)
        return {}


# ---------------------------------------------------------------------------
# Public extraction entry point
# ---------------------------------------------------------------------------

def extract_graph(
    text: str,
    doc_id: str,
    collection: str,
    ticker: str = "",
    filing_date: str = "",
) -> ExtractionResult:
    """
    Extract entities from a filing text section and return a normalized ExtractionResult.

    Parameters
    ----------
    text : str
        Text to extract from (header + income statement + risk factors sections).
    doc_id : str
        Stable doc identifier (e.g. "sec-filings/MSFT/10k_2025.htm").
    collection : str
        Pinecone collection name.
    ticker : str
        Ticker override — used if LLM fails to extract it.
    filing_date : str
        Filing date override in YYYY-MM-DD format.
    """
    raw = _extract_llm(text, doc_id)

    # ── Company ──────────────────────────────────────────────────────────────
    company_raw = raw.get("company") or {}
    c_name = company_raw.get("name") or ticker
    c_ticker = (company_raw.get("ticker") or ticker).upper()
    c_id = company_id(c_ticker or c_name)

    # ── Filing ───────────────────────────────────────────────────────────────
    filing_raw = raw.get("filing") or {}
    f_type = filing_raw.get("filing_type") or "10-K"
    f_year = filing_raw.get("fiscal_year")
    f_date = filing_raw.get("filing_date") or filing_date
    f_id = filing_id(c_ticker or c_name, f_type, f_year or "unknown")

    # ── Segments ─────────────────────────────────────────────────────────────
    segments = []
    seg_name_to_id: dict[str, str] = {}
    for s in raw.get("business_segments") or []:
        name = s.get("name", "").strip()
        if not name:
            continue
        s_id = segment_id(c_ticker, name)
        seg_name_to_id[name.lower()] = s_id
        segments.append({"segment_id": s_id, "name": name})

    # ── Chunks (one synthetic chunk per major section extracted) ─────────────
    # These represent the source sections — linked to Metric and RiskFactor nodes
    # for citation grounding. chunk_id matches Pinecone chunk IDs when available.
    income_chunk_id = chunk_id(doc_id, "income_statement")
    risk_chunk_id   = chunk_id(doc_id, "risk_factors")
    chunks = [
        {
            "chunk_id": income_chunk_id,
            "source_file": doc_id,
            "collection": collection,
            "section": "income_statement",
            "text_preview": text[:200],
        },
        {
            "chunk_id": risk_chunk_id,
            "source_file": doc_id,
            "collection": collection,
            "section": "risk_factors",
            "text_preview": text[:200],
        },
    ]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = []
    for m in raw.get("metrics") or []:
        name = m.get("name", "").strip()
        value = m.get("value")
        if not name or value is None:
            continue
        seg_name = m.get("segment") or ""
        m_seg_id = seg_name_to_id.get(seg_name.lower(), "") if seg_name else ""
        m_year = m.get("fiscal_year") or f_year
        m_id = metric_id(c_ticker, name, m_year or "unknown", seg_name)
        metrics.append({
            "metric_id": m_id,
            "name": name,
            "value": float(value),
            "unit": m.get("unit") or "",
            "fiscal_year": m_year,
            "period": m.get("period") or f"FY{m_year}",
            "segment_id": m_seg_id,
            "segment_name": seg_name,
            "chunk_id": income_chunk_id,
        })

    # ── Topics ────────────────────────────────────────────────────────────────
    all_topic_names: set[str] = set(raw.get("topics") or [])
    for rf in raw.get("risk_factors") or []:
        all_topic_names.update(rf.get("topics") or [])

    topics = []
    topic_name_to_id: dict[str, str] = {}
    for t_name in sorted(all_topic_names):
        t_name = t_name.strip()
        if not t_name:
            continue
        t_id = topic_id(t_name)
        topic_name_to_id[t_name.lower()] = t_id
        topics.append({"topic_id": t_id, "name": t_name})

    # ── Risk factors ──────────────────────────────────────────────────────────
    risk_factors = []
    for rf in raw.get("risk_factors") or []:
        title = rf.get("title", "").strip()
        if not title:
            continue
        r_id = risk_id(c_ticker, f_year or "unknown", title, risk_chunk_id)
        rf_topics = [
            topic_name_to_id[t.strip().lower()]
            for t in (rf.get("topics") or [])
            if t.strip().lower() in topic_name_to_id
        ]
        risk_factors.append({
            "risk_id": r_id,
            "title": title,
            "summary": rf.get("summary") or "",
            "fiscal_year": f_year,
            "topic_ids": rf_topics,
            "chunk_id": risk_chunk_id,
        })

    return ExtractionResult(
        doc_id=doc_id,
        collection=collection,
        ticker=c_ticker,
        company_name=c_name,
        company_id=c_id,
        filing_id=f_id,
        fiscal_year=f_year,
        filing_type=f_type,
        filing_date=f_date,
        segments=segments,
        metrics=metrics,
        risk_factors=risk_factors,
        topics=topics,
        chunks=chunks,
    )


# ---------------------------------------------------------------------------
# Neo4j upsert
# ---------------------------------------------------------------------------

def upsert_to_neo4j(result: ExtractionResult, driver) -> None:
    """
    Write an ExtractionResult into Neo4j using MERGE for idempotency.
    Failures on individual entities are logged but do not abort the whole upsert.
    """
    with driver.session(database=cfg.NEO4J_DATABASE) as session:
        _upsert_company(session, result)
        _upsert_filing(session, result)
        _upsert_chunks(session, result)
        _upsert_segments(session, result)
        _upsert_metrics(session, result)
        _upsert_topics(session, result)
        _upsert_risk_factors(session, result)


def _run(session, cypher: str, **params) -> None:
    try:
        session.run(cypher, **params)
    except Exception as exc:
        logger.error("Neo4j write error: %s\n  Cypher: %s", exc, cypher[:120])


def _upsert_company(session, r: ExtractionResult) -> None:
    _run(
        session,
        "MERGE (c:Company {company_id: $company_id}) "
        "SET c.name = $name, c.ticker = $ticker",
        company_id=r.company_id, name=r.company_name, ticker=r.ticker,
    )


def _upsert_filing(session, r: ExtractionResult) -> None:
    _run(
        session,
        "MERGE (f:Filing {filing_id: $filing_id}) "
        "SET f.filing_type = $filing_type, f.fiscal_year = $fiscal_year, "
        "    f.filing_date = $filing_date, f.source_file = $source_file, "
        "    f.collection = $collection",
        filing_id=r.filing_id, filing_type=r.filing_type,
        fiscal_year=r.fiscal_year, filing_date=r.filing_date,
        source_file=r.doc_id, collection=r.collection,
    )
    _run(
        session,
        "MATCH (c:Company {company_id: $company_id}) "
        "MATCH (f:Filing {filing_id: $filing_id}) "
        "MERGE (c)-[:FILED]->(f)",
        company_id=r.company_id, filing_id=r.filing_id,
    )


def _upsert_chunks(session, r: ExtractionResult) -> None:
    for ch in r.chunks:
        _run(
            session,
            "MERGE (ch:Chunk {chunk_id: $chunk_id}) "
            "SET ch.source_file = $source_file, ch.collection = $collection, "
            "    ch.section = $section, ch.text_preview = $text_preview",
            **ch,
        )
        _run(
            session,
            "MATCH (f:Filing {filing_id: $filing_id}) "
            "MATCH (ch:Chunk {chunk_id: $chunk_id}) "
            "MERGE (f)-[:HAS_CHUNK]->(ch)",
            filing_id=r.filing_id, chunk_id=ch["chunk_id"],
        )


def _upsert_segments(session, r: ExtractionResult) -> None:
    for s in r.segments:
        _run(
            session,
            "MERGE (s:BusinessSegment {segment_id: $segment_id}) SET s.name = $name",
            segment_id=s["segment_id"], name=s["name"],
        )
        _run(
            session,
            "MATCH (f:Filing {filing_id: $filing_id}) "
            "MATCH (s:BusinessSegment {segment_id: $segment_id}) "
            "MERGE (f)-[:HAS_SEGMENT]->(s)",
            filing_id=r.filing_id, segment_id=s["segment_id"],
        )


def _upsert_metrics(session, r: ExtractionResult) -> None:
    for m in r.metrics:
        _run(
            session,
            "MERGE (m:Metric {metric_id: $metric_id}) "
            "SET m.name = $name, m.value = $value, m.unit = $unit, "
            "    m.fiscal_year = $fiscal_year, m.period = $period",
            metric_id=m["metric_id"], name=m["name"], value=m["value"],
            unit=m["unit"], fiscal_year=m["fiscal_year"], period=m["period"],
        )
        # Link Metric -> Chunk (citation grounding)
        _run(
            session,
            "MATCH (m:Metric {metric_id: $metric_id}) "
            "MATCH (ch:Chunk {chunk_id: $chunk_id}) "
            "MERGE (m)-[:SUPPORTED_BY]->(ch)",
            metric_id=m["metric_id"], chunk_id=m["chunk_id"],
        )
        # Link to segment if present, else to company via filing
        if m["segment_id"]:
            _run(
                session,
                "MATCH (s:BusinessSegment {segment_id: $segment_id}) "
                "MATCH (m:Metric {metric_id: $metric_id}) "
                "MERGE (s)-[:REPORTED_METRIC]->(m)",
                segment_id=m["segment_id"], metric_id=m["metric_id"],
            )
        else:
            # Company-level metric: link via Filing
            _run(
                session,
                "MATCH (f:Filing {filing_id: $filing_id}) "
                "MATCH (m:Metric {metric_id: $metric_id}) "
                "MERGE (f)-[:REPORTED_METRIC]->(m)",
                filing_id=r.filing_id, metric_id=m["metric_id"],
            )


def _upsert_topics(session, r: ExtractionResult) -> None:
    for t in r.topics:
        _run(
            session,
            "MERGE (t:Topic {topic_id: $topic_id}) SET t.name = $name",
            topic_id=t["topic_id"], name=t["name"],
        )


def _upsert_risk_factors(session, r: ExtractionResult) -> None:
    for rf in r.risk_factors:
        _run(
            session,
            "MERGE (rf:RiskFactor {risk_id: $risk_id}) "
            "SET rf.title = $title, rf.summary = $summary, rf.fiscal_year = $fiscal_year",
            risk_id=rf["risk_id"], title=rf["title"],
            summary=rf["summary"], fiscal_year=rf["fiscal_year"],
        )
        # Filing -> RiskFactor
        _run(
            session,
            "MATCH (f:Filing {filing_id: $filing_id}) "
            "MATCH (rf:RiskFactor {risk_id: $risk_id}) "
            "MERGE (f)-[:HAS_RISK_FACTOR]->(rf)",
            filing_id=r.filing_id, risk_id=rf["risk_id"],
        )
        # RiskFactor -> Chunk (citation)
        _run(
            session,
            "MATCH (rf:RiskFactor {risk_id: $risk_id}) "
            "MATCH (ch:Chunk {chunk_id: $chunk_id}) "
            "MERGE (rf)-[:SUPPORTED_BY]->(ch)",
            risk_id=rf["risk_id"], chunk_id=rf["chunk_id"],
        )
        # RiskFactor -> Topic
        for t_id in rf["topic_ids"]:
            _run(
                session,
                "MATCH (rf:RiskFactor {risk_id: $risk_id}) "
                "MATCH (t:Topic {topic_id: $topic_id}) "
                "MERGE (rf)-[:RELATED_TO_TOPIC]->(t)",
                risk_id=rf["risk_id"], topic_id=t_id,
            )
