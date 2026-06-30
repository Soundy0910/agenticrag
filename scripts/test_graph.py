"""
scripts/test_graph.py

End-to-end graph test: runs 3 questions through the compiled LangGraph,
including a follow-up to exercise the rewrite+history loop.

Run from the project root:  python3 scripts/test_graph.py
"""

import pathlib, sys, textwrap
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.agent.graph import run_query

COLLECTION = "demo"
SCOPES     = ["public"]
SEP        = "=" * 68


def print_result(label: str, state: dict) -> None:
    print(f"\n{SEP}")
    print(f"  {label}")
    print(SEP)
    print(f"  Question        : {state['question']!r}")
    print(f"  Rewritten query : {state['rewritten_query']!r}")
    print(f"  Route taken     : {state['route']!r}")
    print(f"  Grade           : {state['grade']!r}  (retry_count={state['retry_count']})")
    print(f"  Citations       : {len(state['citations'])} source(s)")
    for i, c in enumerate(state["citations"], 1):
        snippet = c.source_text[:80].replace("\n", " ")
        print(f"    [{i}] {c.filename}  …{snippet!r}")
    print()
    answer_wrapped = textwrap.fill(state["answer"], width=66, initial_indent="  ", subsequent_indent="  ")
    print(f"  Answer:\n{answer_wrapped}")
    print()


# ── Q1: standalone factual question ─────────────────────────────────────────

print("\nRunning Q1 …")
state1 = run_query(
    question="What AWS certifications does the candidate have?",
    collection=COLLECTION,
    allowed_scopes=SCOPES,
)
print_result("Q1 — Standalone factual", state1)

# ── Q2: follow-up — exercises rewrite node with history ─────────────────────

print("Running Q2 (follow-up) …")
state2 = run_query(
    question="What about their work experience?",
    collection=COLLECTION,
    allowed_scopes=SCOPES,
    conversation_history=state1["conversation_history"],
    reusable_chunks=state1["reusable_chunks"],
)
print_result("Q2 — Follow-up (should rewrite with history context)", state2)

# ── Q3: second follow-up — passes both prior turns ──────────────────────────

print("Running Q3 (second follow-up) …")
state3 = run_query(
    question="Can you summarise the skills mentioned so far?",
    collection=COLLECTION,
    allowed_scopes=SCOPES,
    conversation_history=state2["conversation_history"],
    reusable_chunks=state2["reusable_chunks"],
)
print_result("Q3 — Second follow-up (rewrite resolves 'so far')", state3)

# ── Summary table ────────────────────────────────────────────────────────────

print(SEP)
print("  Summary")
print(SEP)
for label, s in [("Q1", state1), ("Q2", state2), ("Q3", state3)]:
    q_short   = s["question"][:45]
    rw_short  = (s["rewritten_query"] or "")[:45]
    changed   = "✓ rewritten" if rw_short != q_short else "— unchanged"
    print(f"  {label}  route={s['route']:6}  grade={s['grade']:12}  rewrite={changed}")
print()
