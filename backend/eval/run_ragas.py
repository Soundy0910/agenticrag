"""
backend/eval/run_ragas.py

Run the agent on every test case and score results with RAGAS.

Metrics (RAGAS 0.4.x):
  faithfulness       — is the answer entailed by the retrieved contexts? (anti-hallucination)
  answer_relevancy   — does the answer address the question?
  context_precision  — are the retrieved contexts ranked by relevance to the ground truth?
  context_recall     — do the retrieved contexts cover the ground truth answer?

Output:
  Per-case table with all four scores.
  Aggregate averages by case type (factual / comparison / not_in_docs).
  Routing verification table (expected vs actual route, pass/fail).

Run from project root:
  python3 backend/eval/run_ragas.py
"""

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")          # suppress RAGAS deprecation noise
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "storage" / ".env")

# ---------------------------------------------------------------------------
# Imports (after env is loaded)
# ---------------------------------------------------------------------------

import textwrap
from dataclasses import dataclass

import backend.config as cfg
from backend.agent.graph import run_query
from backend.eval.testset import EVAL_CASES, ROUTING_CASES, EvalCase, RoutingCase

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
# Old-style singletons — these are ragas.Metric subclasses and work with evaluate()
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# ---------------------------------------------------------------------------
# RAGAS LLM / embeddings — passed to evaluate() which injects them into metrics
# ---------------------------------------------------------------------------

_ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=cfg.OPENAI_API_KEY))
_ragas_emb = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=cfg.OPENAI_API_KEY)
)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    case_id: str
    case_type: str
    question: str
    ground_truth: str
    answer: str
    route: str
    contexts: list[str]
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


def _run_case(case: EvalCase) -> RunResult:
    """Run one eval case through the agent and return a RunResult."""
    print(f"  [{case.id}] {case.type:12s}  {case.question[:60]!r}...")
    state = run_query(
        question=case.question,
        collection=case.collection,
        allowed_scopes=case.allowed_scopes,
    )
    contexts = [c.source_text for c in state.get("retrieved_chunks", [])]
    # Also include reusable_chunks (prior-turn context available during generation)
    seen = {c for c in contexts}
    for c in state.get("reusable_chunks", []):
        if c.source_text not in seen:
            contexts.append(c.source_text)
            seen.add(c.source_text)

    return RunResult(
        case_id=case.id,
        case_type=case.type,
        question=case.question,
        ground_truth=case.ground_truth,
        answer=state.get("answer", ""),
        route=state.get("route", ""),
        contexts=contexts or ["(no context retrieved)"],
    )


def _run_routing_case(case: RoutingCase) -> dict:
    """Run a routing case and return the actual route taken."""
    print(f"  [{case.id}] collection={case.collection!r}  {case.question[:55]!r}...")
    state = run_query(
        question=case.question,
        collection=case.collection,
        allowed_scopes=["public"],
    )
    actual = state.get("route", "?")
    return {
        "id": case.id,
        "expected": case.expected_route,
        "actual": actual,
        "pass": actual == case.expected_route,
        "notes": case.notes,
    }


def _score(results: list[RunResult]) -> list[RunResult]:
    """Run RAGAS evaluation on all results and populate score fields."""
    samples = [
        SingleTurnSample(
            user_input=r.question,
            retrieved_contexts=r.contexts,
            response=r.answer,
            reference=r.ground_truth,
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)

    print("\nRunning RAGAS evaluation (LLM-as-judge calls)...")
    scores = evaluate(
        dataset,
        metrics=METRICS,
        llm=_ragas_llm,
        embeddings=_ragas_emb,
        show_progress=False,
        raise_exceptions=False,
    )
    df = scores.to_pandas()

    metric_cols = {
        "faithfulness": "faithfulness",
        "answer_relevancy": "answer_relevancy",
        "context_precision": "context_precision",
        "context_recall": "context_recall",
    }

    for i, r in enumerate(results):
        row = df.iloc[i]
        for attr, col in metric_cols.items():
            try:
                v = float(row[col]) if col in row and row[col] == row[col] else None
            except (ValueError, TypeError):
                v = None
            setattr(r, attr, v)

    return results


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else " N/A "


def _bar(v: float | None, width: int = 8) -> str:
    """ASCII bar chart for a 0–1 score."""
    if v is None:
        return "─" * width
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


SEP = "─" * 100


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 100)
    print("  Agentic RAG — RAGAS Evaluation")
    print("=" * 100)

    # ── Phase 1: Run all eval cases ──────────────────────────────────────────
    print(f"\n{'─'*100}")
    print("  Phase 1 — Running agent on eval cases")
    print(f"{'─'*100}")
    results: list[RunResult] = []
    for case in EVAL_CASES:
        results.append(_run_case(case))

    # ── Phase 2: RAGAS scoring ───────────────────────────────────────────────
    results = _score(results)

    # ── Phase 3: Run routing cases ───────────────────────────────────────────
    print(f"\n{'─'*100}")
    print("  Phase 2 — Routing verification")
    print(f"{'─'*100}")
    routing_results = [_run_routing_case(rc) for rc in ROUTING_CASES]

    # ── Print per-case scores ────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  RAGAS Scores — Per Case")
    print(f"{'='*100}")
    print(f"  {'ID':4}  {'Type':12}  {'Faith':>6}  {'AnswRel':>6}  {'CtxPre':>6}  {'CtxRec':>6}  Route  Question")
    print(SEP)

    type_buckets: dict[str, list[RunResult]] = {}
    for r in results:
        type_buckets.setdefault(r.case_type, []).append(r)
        q_short = r.question[:45]
        print(
            f"  {r.case_id:4}  {r.case_type:12}  "
            f"{_fmt(r.faithfulness):>6}  {_fmt(r.answer_relevancy):>6}  "
            f"{_fmt(r.context_precision):>6}  {_fmt(r.context_recall):>6}  "
            f"{r.route:6}  {q_short!r}"
        )

    # ── Aggregate by type ────────────────────────────────────────────────────
    def _avg(items, attr):
        vals = [getattr(i, attr) for i in items if getattr(i, attr) is not None]
        return sum(vals) / len(vals) if vals else None

    print(f"\n{'='*100}")
    print("  Aggregate Scores by Case Type")
    print(f"{'='*100}")
    print(f"  {'Type':12}  {'N':>3}  {'Faith':>6}  {'AnswRel':>6}  {'CtxPre':>6}  {'CtxRec':>6}  Bars (Faith | AnswRel | CtxPre | CtxRec)")
    print(SEP)

    all_types = list(type_buckets.keys()) + ["OVERALL"]
    for t in all_types:
        items = type_buckets.get(t, results) if t != "OVERALL" else results
        f  = _avg(items, "faithfulness")
        ar = _avg(items, "answer_relevancy")
        cp = _avg(items, "context_precision")
        cr = _avg(items, "context_recall")
        bars = f"{_bar(f)} {_bar(ar)} {_bar(cp)} {_bar(cr)}"
        label = t.upper() if t == "OVERALL" else t
        print(
            f"  {label:12}  {len(items):>3}  "
            f"{_fmt(f):>6}  {_fmt(ar):>6}  {_fmt(cp):>6}  {_fmt(cr):>6}  {bars}"
        )

    # ── Routing verification table ───────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  Routing Verification")
    print(f"{'='*100}")
    print(f"  {'ID':4}  {'Expected':8}  {'Actual':8}  {'Pass':5}  Question")
    print(SEP)
    passes = 0
    for rr in routing_results:
        icon = "  ✓  " if rr["pass"] else "  ✗  "
        passes += int(rr["pass"])
        q = ROUTING_CASES[[r.id for r in ROUTING_CASES].index(rr["id"])].question[:55]
        print(f"  {rr['id']:4}  {rr['expected']:8}  {rr['actual']:8}  {icon}  {q!r}")
    print(SEP)
    print(f"  Routing pass rate: {passes}/{len(routing_results)}")

    # ── Interpretation notes ─────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("  Interpretation Notes")
    print(f"{'='*100}")
    print(textwrap.dedent("""
      faithfulness     — measures hallucination. Score < 0.8 means the answer contains
                         claims not supported by retrieved context. Target: > 0.85.

      answer_relevancy — measures whether the answer addresses the question. Low scores
                         indicate the agent answered something adjacent, not the question asked.
                         Target: > 0.80.

      context_precision — measures whether the top-ranked contexts are the most relevant.
                          Lower scores indicate retrieval is returning off-topic chunks first.
                          Target: > 0.70.

      context_recall   — measures whether the retrieved contexts cover the ground truth.
                         EXPECTED TO BE LOW for not_in_docs cases (correct behavior:
                         corpus genuinely lacks the answer). Target for factual: > 0.70.

      not_in_docs faithfulness should be HIGHEST (agent correctly grounds "I don't know"
      in the actual absence of information rather than fabricating an answer).
    """).rstrip())
    print()


if __name__ == "__main__":
    main()
