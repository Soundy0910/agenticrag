"""
scripts/test_nodes.py

Exercises each node function individually with a hand-built state dict.
Run from the project root:  python3 scripts/test_nodes.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.agent.nodes import (
    rewrite_node, router_node, retrieve_vector_node, grade_node, generate_node,
)

BASE = {
    "question": "What AWS certifications does the candidate have?",
    "collection": "demo",
    "allowed_scopes": ["public"],
    "conversation_history": [],
    "reusable_chunks": [],
    "retry_count": 0,
    "messages": [],
}

print("=" * 60)
print("1. rewrite_node  (no history → pass-through)")
r = rewrite_node(BASE)
print(f"   rewritten_query: {r['rewritten_query']!r}")

state = {**BASE, **r}

print("\n2. router_node")
r = router_node(state)
print(f"   route: {r['route']!r}")

state = {**state, **r}

print("\n3. retrieve_vector_node")
r = retrieve_vector_node(state)
chunks = r["retrieved_chunks"]
print(f"   retrieved_chunks: {len(chunks)} chunks")
for i, c in enumerate(chunks, 1):
    print(f"   [{i}] score={c.score:.4f}  {c.source_text[:80].replace(chr(10),' ')!r}")

state = {**state, **r, "reusable_chunks": r["reusable_chunks"]}

print("\n4. grade_node")
r = grade_node(state)
print(f"   grade: {r['grade']!r}  retry_count: {r['retry_count']}")

state = {**state, **r}

print("\n5. generate_node")
r = generate_node(state)
print(f"   answer ({len(r['answer'])} chars):")
print(f"   {r['answer'][:300].replace(chr(10),' ')!r}")
print(f"   citations: {len(r['citations'])}")

print("\n6. rewrite_node  (with history — follow-up)")
follow_up_state = {
    **BASE,
    "question": "What about their work experience?",
    "conversation_history": r["conversation_history"],
    "reusable_chunks": r["reusable_chunks"] if "reusable_chunks" in r else state.get("reusable_chunks", []),
}
r2 = rewrite_node(follow_up_state)
print(f"   original:  {follow_up_state['question']!r}")
print(f"   rewritten: {r2['rewritten_query']!r}")
