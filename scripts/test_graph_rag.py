"""
scripts/test_graph_rag.py

Tests for backend/graph_rag/: schema, extract, query.

Steps:
  1. Print the finance schema prompt block (no API calls).
  2. Extract entities from a sample finance sentence (LLM call).
  3. Upsert extracted entities + relationships into AuraDB.
  4. Run a lookup query and a comparison query.
  5. Print ChunkResult output — what retrieve_graph_node would return.

Run from project root:  python3 scripts/test_graph_rag.py
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.graph_rag.schema import FINANCE_SCHEMA
from backend.graph_rag.extract import extract_graph, upsert_to_neo4j
from backend.graph_rag.query import get_driver, graph_query

SEP = "=" * 68

# ── 1. Schema ────────────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  1. Finance schema prompt block")
print(SEP)
print(FINANCE_SCHEMA.to_prompt_block())

# ── 2. Extraction ────────────────────────────────────────────────────────────

SAMPLE_TEXT = """
Apple Inc. reported total revenue of $394.3 billion for fiscal year 2024,
representing a 2% increase year over year. The Services segment contributed
$96.2 billion in revenue, while the Products segment contributed $298.1 billion.
Net income for FY2024 was $93.7 billion.

Microsoft Corporation reported revenue of $245.1 billion for fiscal year 2024,
with the Intelligent Cloud segment contributing $105.4 billion.
Net income was $88.1 billion for FY2024.
"""

print(f"\n{SEP}")
print("  2. Entity extraction from sample finance text")
print(SEP)
print(f"  Text ({len(SAMPLE_TEXT)} chars): Apple FY2024 revenue + Microsoft FY2024 revenue\n")

result = extract_graph(SAMPLE_TEXT, FINANCE_SCHEMA, doc_id="sample_finance_text", collection="finance")

print(f"  Entities extracted: {len(result.entities)}")
for e in result.entities:
    print(f"    [{e.type:12s}] {e.id:25s} {e.properties}")

print(f"\n  Relationships extracted: {len(result.relationships)}")
for r in result.relationships:
    print(f"    {r.from_id:25s} --[{r.type}]--> {r.to_id}")

# ── 3. Upsert to Neo4j ───────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  3. Upsert to AuraDB Neo4j")
print(SEP)

driver = get_driver()
upsert_to_neo4j(result, driver)
print("  Upsert complete.")

# Verify: count nodes
with driver.session() as session:
    for label in ["Company", "Metric", "FiscalYear", "Segment"]:
        count = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        print(f"    {label:12s}: {count} nodes")

# ── 4. Lookup query ──────────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  4. Lookup query — 'What was Apple's revenue in FY2024?'")
print(SEP)

chunks = graph_query(
    "What was Apple's revenue in FY2024?",
    FINANCE_SCHEMA,
    collection="finance",
)
print(f"  ChunkResults returned: {len(chunks)}")
for c in chunks:
    print(f"\n  chunk_id : {c.chunk_id}")
    print(f"  score    : {c.score}")
    print(f"  source_text:\n")
    for line in c.source_text.splitlines():
        print(f"    {line}")

# ── 5. Comparison query ──────────────────────────────────────────────────────

print(f"\n{SEP}")
print("  5. Comparison query — 'Compare Apple and Microsoft revenue for 2024'")
print(SEP)

chunks2 = graph_query(
    "Compare Apple and Microsoft revenue for fiscal year 2024",
    FINANCE_SCHEMA,
    collection="finance",
)
print(f"  ChunkResults returned: {len(chunks2)}")
for c in chunks2:
    print(f"\n  chunk_id : {c.chunk_id}")
    print(f"  source_text:\n")
    for line in c.source_text.splitlines():
        print(f"    {line}")

driver.close()
print(f"\n{SEP}")
print("  File 12 complete.")
print(SEP)
