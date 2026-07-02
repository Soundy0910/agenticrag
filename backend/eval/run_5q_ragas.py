"""
backend/eval/run_5q_ragas.py

Run the 5 chat-UI test questions through the agent and score with RAGAS.
Questions target sec-filings and legal-docs collections via collection='auto'.

Run from project root:
  python3 backend/eval/run_5q_ragas.py
"""

import pathlib, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "storage" / ".env")

import time

import backend.config as cfg
from backend.agent.graph import run_query

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# ---------------------------------------------------------------------------
# The 5 test questions (collection='auto' so the router classifies them)
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        "id": "T01",
        "label": "MSFT revenue+income FY2025",
        "question": "What was Microsoft's total revenue and net income in FY2025?",
        "ground_truth": (
            "Microsoft's total revenue in fiscal year 2025 was $281,724 million ($281.7 billion), "
            "a 15% increase from FY2024 revenue of $245,122 million. "
            "Net income was $101,832 million ($101.8 billion)."
        ),
        "collection": "auto",
    },
    {
        "id": "T02",
        "label": "JPM revenue + recoupment conditions (cross-collection)",
        "question": (
            "What was JPMorgan's total revenue in 2025, and what are the key recoupment "
            "conditions in their executive compensation agreements?"
        ),
        "ground_truth": (
            "JPMorgan Chase's total net revenue for fiscal year 2025 was approximately $175–182 billion. "
            "Per JPMorgan's RSU Award Agreement (Exhibit 10.17), employees are subject to the "
            "JPMorganChase Bonus Recoupment Policy, which applies both to cash incentive compensation "
            "awarded for performance year 2025 and to the RSU award itself. "
            "The Bonus Recoupment Policy governs recovery of incentive compensation."
        ),
        "collection": "auto",
    },
    {
        "id": "T03",
        "label": "Apple vs Microsoft revenue FY2024 (comparison/graph)",
        "question": "Compare Apple and Microsoft's revenue for fiscal year 2024",
        "ground_truth": (
            "Apple's total net sales for fiscal year 2024 were approximately $391 billion. "
            "Microsoft's total revenue for fiscal year 2024 was approximately $245 billion. "
            "Apple had higher revenue than Microsoft in FY2024."
        ),
        "collection": "auto",
    },
    {
        "id": "T04",
        "label": "Walmart net sales + risk factors",
        "question": "What were Walmart's net sales and what are the main risk factors they identified?",
        "ground_truth": (
            "Walmart's consolidated net sales for fiscal year 2026 (ended January 31, 2026) were "
            "$706,413 million ($706 billion). Walmart U.S. segment net sales were approximately "
            "$483 billion, Walmart International approximately $130 billion, and Sam's Club "
            "approximately $90 billion. "
            "Key risk factors identified in their 10-K include competition from retailers and "
            "e-commerce companies, macroeconomic conditions, supply chain disruptions, "
            "cybersecurity and data privacy threats, labor pressures, tax, legal, regulatory, "
            "and compliance risks."
        ),
        "collection": "auto",
    },
    {
        "id": "T05",
        "label": "Tesla executive clawback (legal-docs)",
        "question": "What are the compensation clawback conditions in Tesla's executive agreements?",
        "ground_truth": (
            "Tesla's executive equity award agreements (Exhibit 10.9 Stock Option Award Agreement "
            "and Exhibit 10.10 RSU Award Agreement) include clawback provisions stating that "
            "compensation is subject to recoupment and clawback under applicable law and the "
            "Company's clawback policies. The Company may require participants to repay shares, "
            "options, or proceeds from the sale of shares."
        ),
        "collection": "auto",
    },
]

# ---------------------------------------------------------------------------
# RAGAS setup
# ---------------------------------------------------------------------------

_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=cfg.OPENAI_API_KEY))
_emb = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=cfg.OPENAI_API_KEY)
)
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

SEP = "─" * 110

def _fmt(v):
    return f"{v:.3f}" if v is not None else " N/A "

def _bar(v, width=8):
    if v is None:
        return "─" * width
    filled = round(v * width)
    return "█" * filled + "░" * (width - filled)


def main():
    print("=" * 110)
    print("  AgenticRAG — 5 Chat-UI Question RAGAS Eval  (collection=auto)")
    print("=" * 110)

    results = []
    for idx, tc in enumerate(TEST_CASES):
        print(f"\n[{tc['id']}] {tc['label']}")
        print(f"  Q: {tc['question'][:90]}")
        state = run_query(
            question=tc["question"],
            collection=tc["collection"],
            allowed_scopes=["public"],
        )
        chunks = [c.source_text for c in state.get("retrieved_chunks", [])]
        # deduplicate with reusable chunks
        seen = set(chunks)
        for c in state.get("reusable_chunks", []):
            if c.source_text not in seen:
                chunks.append(c.source_text)
                seen.add(c.source_text)

        answer = state.get("answer", "")
        route = state.get("route", "?")
        active = state.get("active_collections", [tc["collection"]])
        print(f"  route={route}  active_collections={active}  chunks={len(chunks)}")
        print(f"  answer[:200]: {answer[:200]!r}")

        results.append({
            **tc,
            "answer": answer,
            "route": route,
            "active_collections": active,
            "contexts": chunks or ["(no context retrieved)"],
        })
        if idx < len(TEST_CASES) - 1:
            time.sleep(15)  # stay within Cohere trial 10 calls/min limit (15s gap between queries)

    # ── RAGAS scoring ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Running RAGAS evaluation...")
    print(SEP)

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["ground_truth"],
        )
        for r in results
    ]
    dataset = EvaluationDataset(samples=samples)
    scores = evaluate(
        dataset,
        metrics=METRICS,
        llm=_llm,
        embeddings=_emb,
        show_progress=False,
        raise_exceptions=False,
    )
    df = scores.to_pandas()

    # Attach scores
    for i, r in enumerate(results):
        row = df.iloc[i]
        for col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            try:
                r[col] = float(row[col]) if col in row and row[col] == row[col] else None
            except (ValueError, TypeError):
                r[col] = None

    # ── Per-question table ────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("  RAGAS Scores — Per Question")
    print(f"{'='*110}")
    print(f"  {'ID':4}  {'Route':6}  {'Faith':>6}  {'AnswRel':>7}  {'CtxPre':>7}  {'CtxRec':>7}  {'Nchunks':>7}  Label")
    print(SEP)
    for r in results:
        print(
            f"  {r['id']:4}  {r['route']:6}  "
            f"{_fmt(r.get('faithfulness')):>6}  {_fmt(r.get('answer_relevancy')):>7}  "
            f"{_fmt(r.get('context_precision')):>7}  {_fmt(r.get('context_recall')):>7}  "
            f"{len(r['contexts']):>7}  {r['label']}"
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def avg(key):
        vals = [r.get(key) for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    f  = avg("faithfulness")
    ar = avg("answer_relevancy")
    cp = avg("context_precision")
    cr = avg("context_recall")

    print(SEP)
    print(
        f"  {'OVERALL':4}  {'':6}  {_fmt(f):>6}  {_fmt(ar):>7}  {_fmt(cp):>7}  {_fmt(cr):>7}  "
        f"  bars: {_bar(f)} {_bar(ar)} {_bar(cp)} {_bar(cr)}"
    )

    # ── Per-question answers for inspection ──────────────────────────────────
    print(f"\n{'='*110}")
    print("  Answers & Contexts (for manual inspection)")
    print(f"{'='*110}")
    for r in results:
        print(f"\n[{r['id']}] {r['label']}")
        print(f"  Q : {r['question']}")
        print(f"  A : {r['answer'][:400]}")
        print(f"  Route={r['route']}  active_collections={r['active_collections']}")
        print(f"  Contexts retrieved: {len(r['contexts'])}")
        for j, ctx in enumerate(r["contexts"][:3], 1):
            print(f"    [{j}] {ctx[:200]!r}")
        if len(r["contexts"]) > 3:
            print(f"    ... and {len(r['contexts'])-3} more")

    print(f"\n{'='*110}")
    print("  Done.")


if __name__ == "__main__":
    main()
