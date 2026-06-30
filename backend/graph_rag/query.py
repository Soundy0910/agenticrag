"""
backend/graph_rag/query.py

Query Neo4j for relational / comparative financial questions.

Entry point: graph_query(question, schema, collection) -> list[ChunkResult]

This is what retrieve_graph_node in nodes.py calls. Results are returned as
ChunkResult objects so the generate_node handles them identically to vector
or CAG results — no special-casing needed downstream.

QUERY STRATEGY:
  1. LLM extracts structured query params from the natural-language question:
       { companies: [...], metric: "...", year: "...", query_type: "..." }
  2. query_type drives the Cypher template:
       "comparison" — two companies, same metric, same period
       "lookup"     — one company's metrics (optionally filtered by metric/year)
       "segment"    — company's segment-level breakdown
  3. Neo4j results are formatted as readable text and wrapped in ChunkResult.

CYPHER TEMPLATES:
  Parameterised — company names use CONTAINS for partial matching so
  "Apple" matches "Apple Inc." in the graph without exact string equality.
"""

import json
import logging
from functools import lru_cache

from neo4j import GraphDatabase

from openai import OpenAI

import backend.config as cfg
from backend.graph_rag.schema import GraphSchema
from backend.retrieval.hybrid import ChunkResult

logger = logging.getLogger(__name__)


def _norm_year(year: str | None) -> str | None:
    """Strip 'FY'/'Q' prefixes so 'FY2024' and '2024' both match year='2024' in the graph."""
    if not year:
        return year
    return year.lstrip("FYfy").split("Q")[0].strip() or year


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
    """Open a session on the configured database (AuraDB uses instance ID, not 'neo4j')."""
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
# Step 1 — extract query params from the natural-language question
# ---------------------------------------------------------------------------

_PARAM_SYSTEM = """\
Extract structured query parameters from a financial question.
Return ONLY valid JSON — no explanation, no markdown.

JSON structure:
{
  "companies": ["<name>"],          // 1 or 2 company names mentioned
  "metric": "<metric name or null>", // e.g. "Revenue", "EBITDA", "Net Income"
  "year": "<year string or null>",   // e.g. "2024", "FY2023"
  "quarter": "<quarter or null>",    // e.g. "Q4", null for full year
  "query_type": "<comparison|lookup|segment>"
                                     // comparison: two companies same metric
                                     // lookup: one company, any metrics
                                     // segment: segment breakdown for a company
}
"""


def _extract_query_params(question: str) -> dict:
    try:
        resp = _llm().chat.completions.create(
            model=cfg.DEFAULT_LLM_MODEL,
            messages=[
                {"role": "system", "content": _PARAM_SYSTEM},
                {"role": "user",   "content": f"Question: {question}"},
            ],
            max_tokens=200,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as exc:
        logger.warning("_extract_query_params failed: %s", exc)
        return {"companies": [], "metric": None, "year": None, "quarter": None, "query_type": "lookup"}


# ---------------------------------------------------------------------------
# Step 2 — Cypher templates
# ---------------------------------------------------------------------------

def _run_lookup(driver, companies: list[str], metric: str | None, year: str | None) -> list[dict]:
    """Fetch metrics for one company, optionally filtered by metric name and year."""
    company = companies[0] if companies else ""
    cypher_parts = [
        "MATCH (c:Company)-[:REPORTED]->(m:Metric)-[:IN_PERIOD]->(f:FiscalYear)",
        "WHERE toLower(c.name) CONTAINS toLower($company)",
    ]
    params: dict = {"company": company}

    if metric:
        cypher_parts.append("AND toLower(m.name) CONTAINS toLower($metric)")
        params["metric"] = metric
    if year:
        cypher_parts.append("AND f.year = $year")
        params["year"] = str(year)

    cypher_parts.append("RETURN c.name AS company, m.name AS metric, m.value AS value, m.unit AS unit, f.year AS year")
    cypher_parts.append("ORDER BY f.year DESC LIMIT 10")

    with _session(driver) as session:
        result = session.run("\n".join(cypher_parts), **params)
        return [dict(r) for r in result]


def _run_comparison(driver, companies: list[str], metric: str | None, year: str | None) -> list[dict]:
    """Fetch the same metric for two companies in the same period."""
    if len(companies) < 2:
        return _run_lookup(driver, companies, metric, year)

    cypher = """
MATCH (c1:Company)-[:REPORTED]->(m1:Metric)-[:IN_PERIOD]->(f:FiscalYear)
WHERE toLower(c1.name) CONTAINS toLower($company1)
MATCH (c2:Company)-[:REPORTED]->(m2:Metric)-[:IN_PERIOD]->(f)
WHERE toLower(c2.name) CONTAINS toLower($company2)
  AND toLower(m1.name) = toLower(m2.name)
"""
    params: dict = {"company1": companies[0], "company2": companies[1]}

    if metric:
        cypher += "  AND toLower(m1.name) CONTAINS toLower($metric)\n"
        params["metric"] = metric
    if year:
        cypher += "  AND f.year = $year\n"
        params["year"] = str(year)

    cypher += "RETURN c1.name AS company1, m1.value AS value1, c2.name AS company2, m2.value AS value2, m1.name AS metric, m1.unit AS unit, f.year AS year\nORDER BY f.year DESC LIMIT 5"

    with _session(driver) as session:
        result = session.run(cypher, **params)
        return [dict(r) for r in result]


def _run_segment(driver, companies: list[str], year: str | None) -> list[dict]:
    """Fetch segment-level metric breakdown for a company."""
    company = companies[0] if companies else ""
    cypher = """
MATCH (c:Company)-[:HAS_SEGMENT]->(s:Segment)-[:SEGMENT_REPORTED]->(m:Metric)-[:IN_PERIOD]->(f:FiscalYear)
WHERE toLower(c.name) CONTAINS toLower($company)
"""
    params: dict = {"company": company}
    if year:
        cypher += "  AND f.year = $year\n"
        params["year"] = str(year)
    cypher += "RETURN c.name AS company, s.name AS segment, m.name AS metric, m.value AS value, m.unit AS unit, f.year AS year\nORDER BY f.year DESC, s.name LIMIT 20"

    with _session(driver) as session:
        result = session.run(cypher, **params)
        return [dict(r) for r in result]


# ---------------------------------------------------------------------------
# Step 3 — format Neo4j rows as ChunkResult objects
# ---------------------------------------------------------------------------

def _rows_to_chunk_results(rows: list[dict], query_type: str, collection: str) -> list[ChunkResult]:
    """Convert Neo4j result rows into ChunkResult objects for the agent."""
    if not rows:
        return []

    if query_type == "comparison" and rows and "company1" in rows[0]:
        lines = ["Comparison results from knowledge graph:\n"]
        for r in rows:
            lines.append(
                f"  {r.get('company1')} {r.get('metric', '')}: {r.get('value1')} {r.get('unit', '')}"
                f" (FY{r.get('year', '')})"
            )
            lines.append(
                f"  {r.get('company2')} {r.get('metric', '')}: {r.get('value2')} {r.get('unit', '')}"
                f" (FY{r.get('year', '')})"
            )
        source_text = "\n".join(lines)
    elif query_type == "segment" and rows and "segment" in rows[0]:
        lines = [f"Segment breakdown from knowledge graph:\n"]
        for r in rows:
            lines.append(
                f"  {r.get('company')} / {r.get('segment')}: "
                f"{r.get('metric')} = {r.get('value')} {r.get('unit', '')} (FY{r.get('year', '')})"
            )
        source_text = "\n".join(lines)
    else:
        lines = ["Financial data from knowledge graph:\n"]
        for r in rows:
            lines.append(
                f"  {r.get('company')}: {r.get('metric')} = {r.get('value')} {r.get('unit', '')} (FY{r.get('year', '')})"
            )
        source_text = "\n".join(lines)

    return [
        ChunkResult(
            chunk_id=f"graph:{query_type}:{hash(source_text) & 0xFFFFFF:06x}",
            parent_id=None,
            doc_id="neo4j_graph",
            source_text=source_text,
            filename="neo4j_graph",
            collection=collection,
            is_parent=True,
            score=1.0,          # exact graph match — no similarity score
            vector_rank=None,
            bm25_rank=None,
        )
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def graph_query(
    question: str,
    schema: GraphSchema,
    collection: str,
) -> list[ChunkResult]:
    """
    Answer a relational/comparative question by querying Neo4j.

    Called by retrieve_graph_node in nodes.py. Returns ChunkResult objects
    so the generate_node receives them identically to vector retrieval results.

    Returns an empty list if:
      - Neo4j has no relevant data (question not covered by ingested docs)
      - Connection or query error (logged; caller should fall back to vector)
    """
    try:
        driver = get_driver()
    except Exception as exc:
        logger.error("graph_query: cannot connect to Neo4j: %s", exc)
        return []

    params = _extract_query_params(question)
    query_type = params.get("query_type", "lookup")
    companies = params.get("companies") or []
    metric = params.get("metric")
    year = _norm_year(params.get("year"))

    logger.info(
        "graph_query: type=%s companies=%s metric=%r year=%r",
        query_type, companies, metric, year,
    )

    try:
        if query_type == "comparison":
            rows = _run_comparison(driver, companies, metric, year)
        elif query_type == "segment":
            rows = _run_segment(driver, companies, year)
        else:
            rows = _run_lookup(driver, companies, metric, year)
    except Exception as exc:
        logger.error("graph_query: Cypher error: %s", exc)
        return []

    return _rows_to_chunk_results(rows, query_type, collection)
