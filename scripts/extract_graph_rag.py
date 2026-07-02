#!/usr/bin/env python3
"""
scripts/extract_graph_rag.py

Read each 10-K HTML file from data/filings/, extract entities using the new
v2 schema (Company, Filing, BusinessSegment, Metric, RiskFactor, Topic, Chunk)
and upsert them to Neo4j.

Run AFTER reset_neo4j.py to start from a clean graph.

Usage:
    cd /Users/soundariyanvenkatachalam/Desktop/AgenticRAG
    python scripts/extract_graph_rag.py
    python scripts/extract_graph_rag.py --ticker MSFT AAPL   # subset
"""

import argparse
import re
import sys
import time
import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _env in [ROOT / ".env", ROOT / "backend" / ".env", ROOT / "backend" / "storage" / ".env"]:
    if _env.exists():
        from dotenv import load_dotenv
        load_dotenv(_env)
        break

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpcore", "openai", "neo4j"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import backend.config as cfg
from backend.graph_rag.extract import extract_graph, upsert_to_neo4j
from backend.graph_rag.query import get_driver

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FILINGS_DIR = ROOT / "data" / "filings"
COLLECTION  = "sec-filings"

ALL_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "JPM", "TSLA", "META", "JNJ", "V",
    "WMT", "XOM", "PFE", "KO", "DIS",
]

SECTION_CHARS = 6000


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _read_filing(path: Path) -> str:
    import html as html_lib
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", " ", raw)
    text = html_lib.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_section(text: str, keywords: list[str], chars: int = SECTION_CHARS) -> str:
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw.lower())
        if idx != -1:
            start = max(0, idx - 200)
            return text[start : start + chars]
    return text[:chars]


def _build_extract_text(text: str) -> str:
    """
    Combine 4 strategic sections into ~6000 chars for LLM extraction:
      1. Header (company identity, ticker, fiscal year)
      2. Income statement (revenue, net income, operating income)
      3. Business segments (segment breakdown)
      4. Risk factors (risk titles and summaries)
    """
    header = text[:800]

    income = _extract_section(text, [
        "summary results of operations",
        "consolidated statements of operations",
        "results of operations",
        "net revenues",
        "total net sales",
        "total revenue",
    ], chars=1400)

    segment = _extract_section(text, [
        "segment information",
        "business segments",
        "reportable segments",
        "operating segments",
    ], chars=1000)

    risk = _extract_section(text, [
        "item 1a",
        "risk factors",
        "risks related to",
        "the following risks",
    ], chars=2600)

    return f"{header}\n\n{income}\n\n{segment}\n\n{risk}"


# ---------------------------------------------------------------------------
# Validation queries (new schema)
# ---------------------------------------------------------------------------

def _test_queries(driver) -> None:
    print("\n" + "=" * 64)
    print("  Neo4j Validation Queries")
    print("=" * 64)

    queries = [
        (
            "Node counts",
            """MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count
               ORDER BY count DESC""",
        ),
        (
            "Company → Filing sample",
            """MATCH (c:Company)-[:FILED]->(f:Filing)
               RETURN c.ticker, f.filing_type, f.fiscal_year
               ORDER BY c.ticker LIMIT 5""",
        ),
        (
            "Segment metrics — Microsoft",
            """MATCH (f:Filing)<-[:FILED]-(c:Company {ticker: 'MSFT'})
               MATCH (f)-[:HAS_SEGMENT]->(s:BusinessSegment)-[:REPORTED_METRIC]->(m:Metric)
               RETURN s.name, m.name, m.value, m.unit, m.fiscal_year
               ORDER BY m.fiscal_year DESC LIMIT 6""",
        ),
        (
            "Risk factors with topics — sample",
            """MATCH (f:Filing)-[:HAS_RISK_FACTOR]->(rf:RiskFactor)-[:RELATED_TO_TOPIC]->(t:Topic)
               MATCH (c:Company)-[:FILED]->(f)
               RETURN c.ticker, rf.title, t.name
               LIMIT 6""",
        ),
        (
            "Chunk citation check",
            """MATCH (m:Metric)-[:SUPPORTED_BY]->(ch:Chunk)
               RETURN m.name, ch.section, ch.source_file
               LIMIT 3""",
        ),
    ]

    with driver.session(database=cfg.NEO4J_DATABASE) as session:
        for label, cypher in queries:
            print(f"\n  [{label}]")
            try:
                rows = [dict(r) for r in session.run(cypher)]
                if not rows:
                    print("    (no results)")
                else:
                    for r in rows[:6]:
                        print(f"    {r}")
            except Exception as exc:
                print(f"    ERROR: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(tickers: list[str]) -> None:
    print("\n" + "=" * 64)
    print("  GraphRAG Extraction → Neo4j  (schema v2)")
    print(f"  Collection : {COLLECTION}")
    print(f"  Tickers    : {', '.join(tickers)}")
    print("=" * 64 + "\n")

    try:
        driver = get_driver()
        driver.verify_connectivity()
        print("  Neo4j connected ✓\n")
    except Exception as exc:
        print(f"  ERROR: cannot connect to Neo4j: {exc}")
        return

    results: dict[str, dict] = {}

    for ticker in tickers:
        ticker_dir = FILINGS_DIR / ticker
        filings = sorted(ticker_dir.glob("10k_*.htm")) if ticker_dir.exists() else []

        if not filings:
            print(f"  [{ticker}] no 10-K file found in {ticker_dir} — skipping")
            results[ticker] = {"error": "no local file"}
            continue

        local_path = filings[-1]
        doc_id = f"sec-filings/{ticker}/{local_path.name}"

        print(f"  [{ticker}] {local_path.name}", end=" … ", flush=True)
        t0 = time.time()

        try:
            text = _read_filing(local_path)
            extract_text = _build_extract_text(text)

            result = extract_graph(
                text=extract_text,
                doc_id=doc_id,
                collection=COLLECTION,
                ticker=ticker,
            )

            upsert_to_neo4j(result, driver)
            elapsed = time.time() - t0

            print(
                f"entities={result.entity_count}  "
                f"rels={result.relationship_count}  "
                f"metrics={len(result.metrics)}  "
                f"risks={len(result.risk_factors)}  "
                f"({elapsed:.1f}s)"
            )
            results[ticker] = {
                "entities": result.entity_count,
                "relationships": result.relationship_count,
                "metrics": len(result.metrics),
                "risks": len(result.risk_factors),
            }

        except Exception as exc:
            print(f"FAILED: {exc}")
            results[ticker] = {"error": str(exc)}

        time.sleep(1.2)  # respect OpenAI rate limits

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 64)
    print("  Extraction Summary")
    print("=" * 64)
    print(f"  {'Ticker':<8} {'Entities':>9} {'Rels':>7} {'Metrics':>8} {'Risks':>6}  Status")
    print(f"  {'─'*6:<8} {'─'*7:>9} {'─'*5:>7} {'─'*6:>8} {'─'*4:>6}  {'─'*8}")
    for ticker, r in results.items():
        if r.get("error"):
            print(f"  {ticker:<8} {'—':>9} {'—':>7} {'—':>8} {'—':>6}  {r['error'][:30]}")
        else:
            print(f"  {ticker:<8} {r['entities']:>9} {r['relationships']:>7} {r['metrics']:>8} {r['risks']:>6}  OK")

    ok = sum(1 for r in results.values() if not r.get("error"))
    print(f"\n  Processed: {ok}/{len(tickers)} tickers successfully\n")

    _test_queries(driver)

    print("\n" + "=" * 64)
    print("  Done. Neo4j graph ready.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract graph entities from 10-K filings into Neo4j.")
    parser.add_argument("--ticker", nargs="+", help="Subset of tickers to process (default: all)")
    args = parser.parse_args()
    tickers = [t.upper() for t in args.ticker] if args.ticker else ALL_TICKERS
    main(tickers)
