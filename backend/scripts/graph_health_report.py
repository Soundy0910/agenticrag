"""
backend/scripts/graph_health_report.py

Diagnostic report on Neo4j graph health.

Reports:
  - Node counts by label
  - Relationship counts by type
  - Metric nodes without SUPPORTED_BY Chunk (ungrounded facts)
  - RiskFactor nodes without SUPPORTED_BY Chunk
  - Orphan Chunk nodes (no incoming edges)
  - Top topics by connected risk factor count
  - Companies with no BusinessSegment nodes
  - Filings with no RiskFactor nodes

Usage:
    cd /path/to/AgenticRAG
    python -m backend.scripts.graph_health_report
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

for _env in [ROOT / ".env", ROOT / "backend" / ".env", ROOT / "backend" / "storage" / ".env"]:
    if _env.exists():
        from dotenv import load_dotenv
        load_dotenv(_env)
        break

import backend.config as cfg
from backend.graph_rag.query import get_driver


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _run(session, cypher: str, **params) -> list[dict]:
    try:
        return [dict(r) for r in session.run(cypher, **params)]
    except Exception as exc:
        print(f"    ERROR: {exc}")
        return []


def _section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _rows(rows: list[dict], empty_msg: str = "  (none)") -> None:
    if not rows:
        print(f"  {empty_msg}")
        return
    for r in rows:
        parts = "  " + "  |  ".join(f"{k}: {v}" for k, v in r.items() if v is not None)
        print(parts)


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def report_node_counts(session) -> None:
    _section("Node counts by label")
    rows = _run(session, """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY count DESC
    """)
    total = sum(r["count"] for r in rows)
    for r in rows:
        pct = r["count"] / total * 100 if total else 0
        print(f"  {r['label']:<20} {r['count']:>6}  ({pct:.1f}%)")
    print(f"  {'TOTAL':<20} {total:>6}")


def report_relationship_counts(session) -> None:
    _section("Relationship counts by type")
    rows = _run(session, """
        MATCH ()-[r]->()
        RETURN type(r) AS rel_type, count(r) AS count
        ORDER BY count DESC
    """)
    for r in rows:
        print(f"  {r['rel_type']:<30} {r['count']:>6}")


def report_ungrounded_metrics(session) -> None:
    _section("Metric nodes without SUPPORTED_BY Chunk  ← unverifiable facts")
    rows = _run(session, """
        MATCH (m:Metric)
        WHERE NOT (m)-[:SUPPORTED_BY]->(:Chunk)
        MATCH (c:Company)-[:FILED]->(:Filing)-[:REPORTED_METRIC]->(m)
        RETURN c.ticker AS ticker, m.name AS metric, m.value AS value,
               m.fiscal_year AS year
        ORDER BY c.ticker, m.fiscal_year DESC
        LIMIT 20
    """)
    if not rows:
        print("  ✅  All Metric nodes have source chunks.")
        return
    print(f"  ⚠️   {len(rows)} ungrounded metrics (showing up to 20):")
    _rows(rows)


def report_ungrounded_risk_factors(session) -> None:
    _section("RiskFactor nodes without SUPPORTED_BY Chunk  ← unverifiable facts")
    rows = _run(session, """
        MATCH (rf:RiskFactor)
        WHERE NOT (rf)-[:SUPPORTED_BY]->(:Chunk)
        MATCH (c:Company)-[:FILED]->(:Filing)-[:HAS_RISK_FACTOR]->(rf)
        RETURN c.ticker AS ticker, rf.title AS risk_title, rf.fiscal_year AS year
        ORDER BY c.ticker, rf.fiscal_year DESC
        LIMIT 20
    """)
    if not rows:
        print("  ✅  All RiskFactor nodes have source chunks.")
        return
    print(f"  ⚠️   {len(rows)} ungrounded risk factors (showing up to 20):")
    _rows(rows)


def report_orphan_chunks(session) -> None:
    _section("Orphan Chunk nodes  ← not linked from any Metric or RiskFactor")
    rows = _run(session, """
        MATCH (ch:Chunk)
        WHERE NOT ()-[:SUPPORTED_BY]->(ch)
          AND NOT ()-[:HAS_CHUNK]->(ch)
        RETURN ch.chunk_id AS chunk_id, ch.source_file AS source_file,
               ch.section AS section
        LIMIT 10
    """)
    count_row = _run(session, """
        MATCH (ch:Chunk)
        WHERE NOT ()-[:SUPPORTED_BY]->(ch) AND NOT ()-[:HAS_CHUNK]->(ch)
        RETURN count(ch) AS total
    """)
    total = count_row[0]["total"] if count_row else 0
    if total == 0:
        print("  ✅  No orphan Chunk nodes.")
        return
    print(f"  ⚠️   {total} orphan chunks (showing up to 10):")
    _rows(rows)


def report_top_topics(session) -> None:
    _section("Top topics by connected RiskFactor count")
    rows = _run(session, """
        MATCH (t:Topic)<-[:RELATED_TO_TOPIC]-(rf:RiskFactor)
        RETURN t.name AS topic, count(rf) AS risk_count
        ORDER BY risk_count DESC
        LIMIT 15
    """)
    if not rows:
        print("  (no Topic nodes found)")
        return
    for r in rows:
        bar = "█" * min(r["risk_count"], 30)
        print(f"  {r['topic']:<25} {r['risk_count']:>4}  {bar}")


def report_companies_missing_segments(session) -> None:
    _section("Companies with no BusinessSegment nodes")
    rows = _run(session, """
        MATCH (c:Company)-[:FILED]->(f:Filing)
        WHERE NOT (f)-[:HAS_SEGMENT]->(:BusinessSegment)
        RETURN c.ticker AS ticker, c.name AS company,
               f.filing_type AS filing_type, f.fiscal_year AS year
        ORDER BY c.ticker
    """)
    if not rows:
        print("  ✅  All filings have BusinessSegment nodes.")
        return
    print(f"  ⚠️   {len(rows)} filings without segments:")
    _rows(rows)


def report_filings_missing_risk_factors(session) -> None:
    _section("Filings with no RiskFactor nodes")
    rows = _run(session, """
        MATCH (c:Company)-[:FILED]->(f:Filing)
        WHERE NOT (f)-[:HAS_RISK_FACTOR]->(:RiskFactor)
        RETURN c.ticker AS ticker, c.name AS company,
               f.filing_type AS filing_type, f.fiscal_year AS year
        ORDER BY c.ticker
    """)
    if not rows:
        print("  ✅  All filings have RiskFactor nodes.")
        return
    print(f"  ⚠️   {len(rows)} filings without risk factors:")
    _rows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 60)
    print("  AgenticRAG — Neo4j Graph Health Report")
    print("=" * 60)

    try:
        driver = get_driver()
        driver.verify_connectivity()
        print("  Neo4j connected ✓")
    except Exception as exc:
        print(f"  ERROR: cannot connect to Neo4j: {exc}")
        return

    with driver.session(database=cfg.NEO4J_DATABASE) as session:
        report_node_counts(session)
        report_relationship_counts(session)
        report_ungrounded_metrics(session)
        report_ungrounded_risk_factors(session)
        report_orphan_chunks(session)
        report_top_topics(session)
        report_companies_missing_segments(session)
        report_filings_missing_risk_factors(session)

    print("\n" + "=" * 60)
    print("  Report complete.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
