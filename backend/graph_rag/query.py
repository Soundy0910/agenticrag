"""
backend/graph_rag/query.py

Query Neo4j for relational / comparative financial questions.

Entry point: graph_query(question, schema, collection) -> GraphQueryResult

Schema v2 Cypher templates (parameterized, no user input interpolation):
  lookup                  — one company's metrics via Company → Filing → Metric
  comparison              — two companies, same metric, same fiscal year
  segment                 — company's segment-level metric breakdown
  risk                    — company's risk factors + related topics
  topic_to_companies      — which companies have risk factors related to a topic
  company_risk_topics     — which topics are connected to a company's filing
  company_segment_metrics — company's segments + all reported metrics
  graph_path_explanation  — how entities (Company→Filing→Segment→Metric) are connected

Returns GraphQueryResult with:
  chunks          — ChunkResult objects (graph facts first, citation-grounded)
  query_type      — which template was dispatched
  unsupported_count — graph facts with no SUPPORTED_BY Chunk
"""

import json
import logging
from dataclasses import dataclass, field

from neo4j import GraphDatabase
from openai import OpenAI

import backend.config as cfg
from backend.graph_rag.schema import GraphSchema
from backend.retrieval.hybrid import ChunkResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class GraphQueryResult:
    chunks: list[ChunkResult] = field(default_factory=list)
    query_type: str = "lookup"
    unsupported_count: int = 0
    total_rows: int = 0


# ---------------------------------------------------------------------------
# Neo4j driver singleton
# ---------------------------------------------------------------------------

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            cfg.NEO4J_URI,
            auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
        )
    return _driver


def _session(driver):
    return driver.session(database=cfg.NEO4J_DATABASE)


# ---------------------------------------------------------------------------
# OpenAI singleton
# ---------------------------------------------------------------------------

_openai: OpenAI | None = None


def _llm() -> OpenAI:
    global _openai
    if _openai is None:
        _openai = OpenAI(api_key=cfg.OPENAI_API_KEY)
    return _openai


# ---------------------------------------------------------------------------
# Step 1 — extract query params
# ---------------------------------------------------------------------------

_PARAM_SYSTEM = """\
Extract structured query parameters from a financial question.
Return ONLY valid JSON — no explanation, no markdown.

JSON structure:
{
  "companies": ["<name or ticker>"],   // 1 or 2 companies mentioned
  "metric": "<metric name or null>",   // e.g. "Revenue", "Net Income"
  "year": "<year string or null>",     // e.g. "2024", "FY2025"
  "topics": ["<topic>"],               // e.g. ["AI", "Cybersecurity"] for risk/topic questions
  "query_type": "<type>"
}

query_type values:
  lookup                  — one company's financial metrics (revenue, net income, etc.)
  comparison              — same metric for two companies side-by-side
  segment                 — company's segment breakdown with metrics
  risk                    — company's risk factors and their topic labels
  topic_to_companies      — which companies have risk factors connected to a topic (e.g. "regulation")
  company_risk_topics     — which risk topics are linked to a specific company's filing
  company_segment_metrics — show a company's segments and all their reported metrics
  graph_path_explanation  — how entities (Company → Filing → Segment → Metric) are connected
"""


def _extract_query_params(question: str) -> dict:
    try:
        resp = _llm().chat.completions.create(
            model=cfg.DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": _PARAM_SYSTEM},
                {"role": "user", "content": f"Question: {question}"},
            ],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as exc:
        logger.warning("_extract_query_params failed: %s", exc)
        return {"companies": [], "metric": None, "year": None, "topics": [], "query_type": "lookup"}


def _norm_year(year: str | None) -> str | None:
    if not year:
        return year
    return year.lstrip("FYfy").split("Q")[0].strip() or year


def _company_match(alias: str) -> str:
    """Cypher WHERE fragment matching company by name or ticker."""
    return f"toLower(c.name) CONTAINS toLower(${alias}) OR toLower(c.ticker) CONTAINS toLower(${alias})"


# ---------------------------------------------------------------------------
# Cypher templates — schema v2
# ---------------------------------------------------------------------------

def _run_lookup(driver, companies: list[str], metric: str | None, year: str | None) -> list[dict]:
    company = companies[0] if companies else ""
    params: dict = {"company": company}

    cypher_company = f"""
MATCH (c:Company)-[:FILED]->(f:Filing)-[:REPORTED_METRIC]->(m:Metric)
WHERE {_company_match('company')}
"""
    if metric:
        cypher_company += "AND toLower(m.name) CONTAINS toLower($metric)\n"
        params["metric"] = metric
    if year:
        cypher_company += "AND toString(m.fiscal_year) = $year\n"
        params["year"] = str(year)
    cypher_company += "OPTIONAL MATCH (m)-[:SUPPORTED_BY]->(ch:Chunk)\nRETURN c.name AS company, null AS segment, m.name AS metric, m.value AS value, m.unit AS unit, m.fiscal_year AS year, ch.source_file AS source_file, ch.chunk_id AS source_chunk_id, ch.section AS section, substring(ch.text, 0, 200) AS text_preview\nORDER BY m.fiscal_year DESC LIMIT 8"

    cypher_segment = f"""
MATCH (c:Company)-[:FILED]->(f:Filing)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:REPORTED_METRIC]->(m:Metric)
WHERE {_company_match('company')}
"""
    if metric:
        cypher_segment += "AND toLower(m.name) CONTAINS toLower($metric)\n"
    if year:
        cypher_segment += "AND toString(m.fiscal_year) = $year\n"
    cypher_segment += "OPTIONAL MATCH (m)-[:SUPPORTED_BY]->(ch:Chunk)\nRETURN c.name AS company, s.name AS segment, m.name AS metric, m.value AS value, m.unit AS unit, m.fiscal_year AS year, ch.source_file AS source_file, ch.chunk_id AS source_chunk_id, ch.section AS section, substring(ch.text, 0, 200) AS text_preview\nORDER BY m.fiscal_year DESC LIMIT 8"

    rows = []
    with _session(driver) as session:
        for cypher in (cypher_company, cypher_segment):
            try:
                rows.extend([dict(r) for r in session.run(cypher, **params)])
            except Exception as exc:
                logger.error("_run_lookup cypher error: %s", exc)
    return rows


def _run_comparison(driver, companies: list[str], metric: str | None, year: str | None) -> list[dict]:
    if len(companies) < 2:
        return _run_lookup(driver, companies, metric, year)

    params: dict = {"company1": companies[0], "company2": companies[1]}
    cypher = f"""
MATCH (c1:Company)-[:FILED]->(f1:Filing)-[:REPORTED_METRIC]->(m1:Metric)
WHERE {_company_match('company1').replace('c.', 'c1.')}
MATCH (c2:Company)-[:FILED]->(f2:Filing)-[:REPORTED_METRIC]->(m2:Metric)
WHERE {_company_match('company2').replace('c.', 'c2.')}
  AND toLower(m1.name) = toLower(m2.name)
  AND m1.fiscal_year = m2.fiscal_year
"""
    if metric:
        cypher += "  AND toLower(m1.name) CONTAINS toLower($metric)\n"
        params["metric"] = metric
    if year:
        cypher += "  AND toString(m1.fiscal_year) = $year\n"
        params["year"] = str(year)

    cypher += """OPTIONAL MATCH (m1)-[:SUPPORTED_BY]->(ch1:Chunk)
OPTIONAL MATCH (m2)-[:SUPPORTED_BY]->(ch2:Chunk)
RETURN c1.name AS company1, m1.value AS value1, c2.name AS company2, m2.value AS value2,
       m1.name AS metric, m1.unit AS unit, m1.fiscal_year AS year,
       ch1.source_file AS source_file1, ch2.source_file AS source_file2
ORDER BY m1.fiscal_year DESC LIMIT 5"""

    with _session(driver) as session:
        try:
            return [dict(r) for r in session.run(cypher, **params)]
        except Exception as exc:
            logger.error("_run_comparison error: %s", exc)
            return []


def _run_segment(driver, companies: list[str], year: str | None) -> list[dict]:
    company = companies[0] if companies else ""
    params: dict = {"company": company}
    cypher = f"""
MATCH (c:Company)-[:FILED]->(f:Filing)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:REPORTED_METRIC]->(m:Metric)
WHERE {_company_match('company')}
"""
    if year:
        cypher += "  AND toString(m.fiscal_year) = $year\n"
        params["year"] = str(year)
    cypher += """OPTIONAL MATCH (m)-[:SUPPORTED_BY]->(ch:Chunk)
RETURN c.name AS company, s.name AS segment, m.name AS metric, m.value AS value,
       m.unit AS unit, m.fiscal_year AS year, ch.source_file AS source_file,
       ch.chunk_id AS source_chunk_id, ch.section AS section
ORDER BY m.fiscal_year DESC, s.name LIMIT 20"""

    with _session(driver) as session:
        try:
            return [dict(r) for r in session.run(cypher, **params)]
        except Exception as exc:
            logger.error("_run_segment error: %s", exc)
            return []


def _run_risk(driver, companies: list[str], topics: list[str], year: str | None) -> list[dict]:
    company = companies[0] if companies else ""
    params: dict = {"company": company}
    cypher = f"""
MATCH (c:Company)-[:FILED]->(f:Filing)-[:HAS_RISK_FACTOR]->(rf:RiskFactor)
WHERE {_company_match('company')}
"""
    if year:
        cypher += "  AND toString(rf.fiscal_year) = $year\n"
        params["year"] = str(year)
    if topics:
        topic_conditions = " OR ".join(
            f"toLower(t{i}.name) CONTAINS toLower($topic{i})"
            for i in range(len(topics))
        )
        cypher += "MATCH (rf)-[:RELATED_TO_TOPIC]->(t:Topic)\nWHERE " + topic_conditions + "\n"
        for i, t in enumerate(topics):
            params[f"topic{i}"] = t
    else:
        cypher += "OPTIONAL MATCH (rf)-[:RELATED_TO_TOPIC]->(t:Topic)\n"

    cypher += """OPTIONAL MATCH (rf)-[:SUPPORTED_BY]->(ch:Chunk)
RETURN c.name AS company, rf.title AS risk_title, rf.summary AS summary,
       collect(DISTINCT t.name) AS topics, rf.fiscal_year AS year,
       ch.source_file AS source_file, ch.chunk_id AS source_chunk_id, ch.section AS section
ORDER BY rf.fiscal_year DESC LIMIT 10"""

    with _session(driver) as session:
        try:
            return [dict(r) for r in session.run(cypher, **params)]
        except Exception as exc:
            logger.error("_run_risk error: %s", exc)
            return []


def _run_topic_to_companies(driver, topics: list[str]) -> list[dict]:
    """Which companies have risk factors related to a given topic."""
    if not topics:
        return []
    topic_conditions = " OR ".join(
        f"toLower(t.name) CONTAINS toLower($topic{i})"
        for i in range(len(topics))
    )
    params: dict = {f"topic{i}": t for i, t in enumerate(topics)}
    cypher = f"""
MATCH (c:Company)-[:FILED]->(f:Filing)-[:HAS_RISK_FACTOR]->(rf:RiskFactor)-[:RELATED_TO_TOPIC]->(t:Topic)
WHERE {topic_conditions}
OPTIONAL MATCH (rf)-[:SUPPORTED_BY]->(ch:Chunk)
RETURN c.name AS company, c.ticker AS ticker,
       collect(DISTINCT t.name) AS topics,
       collect(DISTINCT rf.title) AS risks,
       ch.source_file AS source_file
ORDER BY c.name LIMIT 15"""

    with _session(driver) as session:
        try:
            return [dict(r) for r in session.run(cypher, **params)]
        except Exception as exc:
            logger.error("_run_topic_to_companies error: %s", exc)
            return []


def _run_company_risk_topics(driver, companies: list[str], year: str | None) -> list[dict]:
    """Which risk topics are connected to a company's filing."""
    company = companies[0] if companies else ""
    params: dict = {"company": company}
    cypher = f"""
MATCH (c:Company)-[:FILED]->(f:Filing)-[:HAS_RISK_FACTOR]->(rf:RiskFactor)-[:RELATED_TO_TOPIC]->(t:Topic)
WHERE {_company_match('company')}
"""
    if year:
        cypher += "  AND toString(rf.fiscal_year) = $year\n"
        params["year"] = str(year)
    cypher += """OPTIONAL MATCH (rf)-[:SUPPORTED_BY]->(ch:Chunk)
RETURN t.name AS topic, count(DISTINCT rf) AS risk_count,
       collect(DISTINCT rf.title)[..3] AS sample_risks,
       ch.source_file AS source_file
ORDER BY risk_count DESC LIMIT 15"""

    with _session(driver) as session:
        try:
            return [dict(r) for r in session.run(cypher, **params)]
        except Exception as exc:
            logger.error("_run_company_risk_topics error: %s", exc)
            return []


def _run_company_segment_metrics(driver, companies: list[str], year: str | None) -> list[dict]:
    """Show a company's segments and all their reported metrics."""
    return _run_segment(driver, companies, year)


def _run_graph_path_explanation(driver, companies: list[str], year: str | None) -> list[dict]:
    """How Company → Filing → Segment → Metric entities are connected."""
    company = companies[0] if companies else ""
    params: dict = {"company": company}
    cypher = f"""
MATCH (c:Company)-[:FILED]->(f:Filing)
WHERE {_company_match('company')}
OPTIONAL MATCH (f)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:REPORTED_METRIC]->(m:Metric)
OPTIONAL MATCH (f)-[:HAS_RISK_FACTOR]->(rf:RiskFactor)
OPTIONAL MATCH (m)-[:SUPPORTED_BY]->(ch:Chunk)
RETURN c.name AS company, c.ticker AS ticker,
       f.filing_type AS filing_type, f.fiscal_year AS fiscal_year,
       collect(DISTINCT s.name)[..5] AS segments,
       collect(DISTINCT {{metric: m.name, value: m.value, unit: m.unit}})[..5] AS metrics,
       count(DISTINCT rf) AS risk_factor_count,
       ch.source_file AS source_file
ORDER BY f.fiscal_year DESC LIMIT 3"""

    with _session(driver) as session:
        try:
            return [dict(r) for r in session.run(cypher, **params)]
        except Exception as exc:
            logger.error("_run_graph_path_explanation error: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Format rows as ChunkResult objects — with citation grounding
# ---------------------------------------------------------------------------

def _rows_to_chunk_results(
    rows: list[dict],
    query_type: str,
    collection: str,
) -> tuple[list[ChunkResult], int]:
    """
    Format Neo4j rows as a single ChunkResult containing the formatted text.
    Returns (chunks, unsupported_count) where unsupported_count is the number
    of rows with no SUPPORTED_BY Chunk (source_file is None).

    Every row that has a source_file is cited; rows without are clearly
    marked [UNSUPPORTED — no source chunk] so the LLM knows not to cite them.
    """
    if not rows:
        return [], 0

    unsupported_count = 0
    lines: list[str] = []
    source_files: set[str] = set()

    # ── comparison ───────────────────────────────────────────────────────────
    if query_type == "comparison" and rows and "company1" in rows[0]:
        lines.append("Comparison results from knowledge graph:\n")
        for r in rows:
            has_src = bool(r.get("source_file1") or r.get("source_file2"))
            citation = ""
            if r.get("source_file1"):
                source_files.add(r["source_file1"])
                citation += f" [source: {r['source_file1']}]"
            if r.get("source_file2"):
                source_files.add(r["source_file2"])
                citation += f" [source: {r['source_file2']}]"
            if not has_src:
                unsupported_count += 1
                citation = " [UNSUPPORTED — no source chunk]"
            lines.append(
                f"  {r.get('company1')} {r.get('metric', '')}: {r.get('value1')} {r.get('unit', '')} (FY{r.get('year', '')}){citation}"
            )
            lines.append(
                f"  {r.get('company2')} {r.get('metric', '')}: {r.get('value2')} {r.get('unit', '')} (FY{r.get('year', '')})"
            )

    # ── risk ─────────────────────────────────────────────────────────────────
    elif query_type == "risk" and rows:
        lines.append("Risk factor analysis from knowledge graph:\n")
        for r in rows:
            has_src = bool(r.get("source_file"))
            citation = f" [source: {r['source_file']}]" if has_src else " [UNSUPPORTED — no source chunk]"
            if has_src:
                source_files.add(r["source_file"])
            else:
                unsupported_count += 1
            topics = r.get("topics") or []
            topics_str = ", ".join(topics) if topics else "—"
            lines.append(f"  [{r.get('company')} FY{r.get('year', '')}]")
            lines.append(f"  Risk: {r.get('risk_title', '')}{citation}")
            lines.append(f"  Summary: {r.get('summary', '')}")
            lines.append(f"  Topics: {topics_str}\n")

    # ── topic_to_companies ───────────────────────────────────────────────────
    elif query_type == "topic_to_companies" and rows:
        lines.append("Companies with matching risk topics (from knowledge graph):\n")
        for r in rows:
            has_src = bool(r.get("source_file"))
            citation = f" [source: {r['source_file']}]" if has_src else " [UNSUPPORTED]"
            if has_src:
                source_files.add(r["source_file"])
            else:
                unsupported_count += 1
            topics = ", ".join(r.get("topics") or [])
            risks = "; ".join((r.get("risks") or [])[:3])
            lines.append(
                f"  {r.get('company')} ({r.get('ticker', '')}): topics=[{topics}] risks=[{risks}]{citation}"
            )

    # ── company_risk_topics ──────────────────────────────────────────────────
    elif query_type == "company_risk_topics" and rows:
        lines.append("Risk topics connected to this company's filing (from knowledge graph):\n")
        for r in rows:
            has_src = bool(r.get("source_file"))
            citation = f" [source: {r['source_file']}]" if has_src else " [UNSUPPORTED]"
            if has_src:
                source_files.add(r["source_file"])
            else:
                unsupported_count += 1
            samples = "; ".join((r.get("sample_risks") or [])[:3])
            lines.append(
                f"  Topic: {r.get('topic', '')} ({r.get('risk_count', 0)} risks) — e.g. {samples}{citation}"
            )

    # ── graph_path_explanation ───────────────────────────────────────────────
    elif query_type == "graph_path_explanation" and rows:
        lines.append("Knowledge graph path explanation:\n")
        for r in rows:
            has_src = bool(r.get("source_file"))
            if has_src:
                source_files.add(r["source_file"])
            else:
                unsupported_count += 1
            segments = ", ".join(r.get("segments") or [])
            metrics_raw = r.get("metrics") or []
            metrics_str = "; ".join(
                f"{m.get('metric')}: {m.get('value')} {m.get('unit', '')}"
                for m in metrics_raw if isinstance(m, dict) and m.get("metric")
            )
            lines.append(
                f"  {r.get('company')} ({r.get('ticker')}) — {r.get('filing_type')} FY{r.get('fiscal_year')}"
            )
            if segments:
                lines.append(f"    Segments: {segments}")
            if metrics_str:
                lines.append(f"    Metrics: {metrics_str}")
            lines.append(f"    Risk factors: {r.get('risk_factor_count', 0)}")

    # ── segment / company_segment_metrics ────────────────────────────────────
    elif rows and "segment" in rows[0]:
        lines.append("Segment breakdown from knowledge graph:\n")
        for r in rows:
            has_src = bool(r.get("source_file"))
            citation = f" [source: {r['source_file']}]" if has_src else " [UNSUPPORTED]"
            if has_src:
                source_files.add(r["source_file"])
            else:
                unsupported_count += 1
            lines.append(
                f"  {r.get('company')} / {r.get('segment')}: "
                f"{r.get('metric')} = {r.get('value')} {r.get('unit', '')} (FY{r.get('year', '')}){citation}"
            )

    # ── lookup (default) ─────────────────────────────────────────────────────
    else:
        lines.append("Financial data from knowledge graph:\n")
        for r in rows:
            has_src = bool(r.get("source_file"))
            citation = f" [source: {r['source_file']}]" if has_src else " [UNSUPPORTED]"
            if has_src:
                source_files.add(r["source_file"])
            else:
                unsupported_count += 1
            seg = f" / {r['segment']}" if r.get("segment") else ""
            lines.append(
                f"  {r.get('company')}{seg}: {r.get('metric')} = {r.get('value')} {r.get('unit', '')} (FY{r.get('year', '')}){citation}"
            )

    if source_files:
        lines.append(f"\n  Sources: {', '.join(sorted(source_files))}")

    source_text = "\n".join(lines)
    is_unsupported = unsupported_count == len(rows)

    chunk = ChunkResult(
        chunk_id=f"graph:{query_type}:{hash(source_text) & 0xFFFFFF:06x}",
        parent_id=None,
        doc_id="neo4j_graph",
        source_text=source_text,
        filename="neo4j_graph",
        collection=collection,
        is_parent=True,
        score=1.0,
        vector_rank=None,
        bm25_rank=None,
        metadata={
            "unsupported_graph_fact": is_unsupported,
            "supported_count": len(rows) - unsupported_count,
            "total_count": len(rows),
            "query_type": query_type,
        },
    )
    return [chunk], unsupported_count


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def graph_query(
    question: str,
    schema: GraphSchema,
    collection: str,
) -> GraphQueryResult:
    """
    Answer a relational/comparative/risk question by querying Neo4j.
    Returns GraphQueryResult (chunks, query_type, unsupported_count, total_rows).
    Returns empty result on connection error or no results.
    """
    try:
        driver = get_driver()
    except Exception as exc:
        logger.error("graph_query: cannot connect to Neo4j: %s", exc)
        return GraphQueryResult()

    params = _extract_query_params(question)
    query_type = params.get("query_type", "lookup")
    companies = params.get("companies") or []
    metric = params.get("metric")
    year = _norm_year(params.get("year"))
    topics = params.get("topics") or []

    logger.info(
        "graph_query: type=%s companies=%s metric=%r year=%r topics=%s",
        query_type, companies, metric, year, topics,
    )

    try:
        if query_type == "comparison":
            rows = _run_comparison(driver, companies, metric, year)
        elif query_type == "segment":
            rows = _run_segment(driver, companies, year)
        elif query_type == "risk":
            rows = _run_risk(driver, companies, topics, year)
        elif query_type == "topic_to_companies":
            rows = _run_topic_to_companies(driver, topics)
        elif query_type == "company_risk_topics":
            rows = _run_company_risk_topics(driver, companies, year)
        elif query_type == "company_segment_metrics":
            rows = _run_company_segment_metrics(driver, companies, year)
        elif query_type == "graph_path_explanation":
            rows = _run_graph_path_explanation(driver, companies, year)
        else:
            rows = _run_lookup(driver, companies, metric, year)
    except Exception as exc:
        logger.error("graph_query: query error: %s", exc)
        return GraphQueryResult(query_type=query_type)

    chunks, unsupported_count = _rows_to_chunk_results(rows, query_type, collection)
    return GraphQueryResult(
        chunks=chunks,
        query_type=query_type,
        unsupported_count=unsupported_count,
        total_rows=len(rows),
    )
