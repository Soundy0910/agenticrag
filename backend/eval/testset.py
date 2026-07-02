"""
backend/eval/testset.py

Production test cases for RAGAS evaluation — sec-filings and legal-docs collections.

Three categories:
  EVAL_CASES    — Q+A pairs for RAGAS scoring (faithfulness, relevance, precision, recall)
  ROUTING_CASES — questions where we verify the router's path decision, not the answer quality

EVAL_CASES types:
  factual       — single-hop factual lookup; answer is in the docs
  multi_facet   — question spans both sec-filings and legal-docs
  comparison    — multi-entity comparison; generate_node decomposes into sub-queries
  not_in_docs   — answer is genuinely absent; agent should say "I don't know"

Ground-truth notes:
  Derived from indexed 10-K filings and legal exhibits (data/filings/ and data/legal/).
  Figures match the actual filed documents — verify against source if re-indexing.
  "not_in_docs" ground truths state absence explicitly — context_recall near 0 is correct.
"""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    type: str          # factual | multi_facet | comparison | not_in_docs
    question: str
    ground_truth: str
    collection: str = "auto"
    allowed_scopes: list[str] = field(default_factory=lambda: ["public"])
    notes: str = ""


@dataclass
class RoutingCase:
    id: str
    question: str
    collection: str
    expected_route: str   # vector | graph
    notes: str = ""


# ---------------------------------------------------------------------------
# Eval cases — used for RAGAS scoring
# ---------------------------------------------------------------------------

EVAL_CASES: list[EvalCase] = [

    # ── Factual: sec-filings ─────────────────────────────────────────────────

    EvalCase(
        id="E01",
        type="factual",
        question="What was Microsoft's total revenue and net income in FY2025?",
        ground_truth=(
            "Microsoft's total revenue in fiscal year 2025 was $281,724 million ($281.7 billion), "
            "a 15% increase from FY2024 revenue of $245,122 million. "
            "Net income was $101,832 million ($101.8 billion)."
        ),
        collection="auto",
        notes=(
            "Tests income statement retrieval for MSFT. "
            "Failure mode: agent may confuse Microsoft Cloud revenue ($168.9B) "
            "with total revenue — the SUMMARY RESULTS OF OPERATIONS chunk must be retrieved."
        ),
    ),

    EvalCase(
        id="E02",
        type="factual",
        question="What were Walmart's net sales and what are the main risk factors they identified?",
        ground_truth=(
            "Walmart's consolidated net sales for fiscal year 2026 were approximately $674 billion "
            "(Walmart U.S. $483B + Walmart International $130B + Sam's Club $90B). "
            "Key risk factors include macroeconomic conditions affecting consumer spending, "
            "intense retail competition, supply chain disruptions, cybersecurity threats, "
            "labor and wage pressures, and regulatory compliance across global markets."
        ),
        collection="auto",
        notes=(
            "Multi-facet question: net sales and risk factors are in separate document sections. "
            "Both facets must be retrieved — the 8-chunk cap often misses risk factors "
            "when net sales chunks dominate the top results."
        ),
    ),

    EvalCase(
        id="E03",
        type="factual",
        question="What was Apple's total net sales in fiscal year 2024?",
        ground_truth=(
            "Apple's total net sales for fiscal year 2024 (ended September 28, 2024) "
            "were $391.0 billion, compared to $383.3 billion in fiscal year 2023."
        ),
        collection="auto",
        notes=(
            "Single-company factual lookup from AAPL 10-K. "
            "Tests that income statement chunk is retrieved over MD&A narrative."
        ),
    ),

    EvalCase(
        id="E04",
        type="factual",
        question="What was JPMorgan Chase's total net revenue for fiscal year 2025?",
        ground_truth=(
            "JPMorgan Chase's total net revenue for fiscal year 2025 was approximately "
            "$175 billion, reflecting strong performance across Consumer & Community Banking, "
            "Commercial Banking, and Corporate & Investment Bank segments."
        ),
        collection="auto",
        notes=(
            "Tests retrieval of consolidated total revenue vs segment-level figures. "
            "Failure mode: agent returns international segment revenue ($24B) instead of total."
        ),
    ),

    # ── Multi-facet: cross-collection ────────────────────────────────────────

    EvalCase(
        id="E05",
        type="multi_facet",
        question=(
            "What was JPMorgan's total revenue in 2025, and what are the key recoupment "
            "conditions in their executive compensation agreements?"
        ),
        ground_truth=(
            "JPMorgan Chase's total net revenue for fiscal year 2025 was approximately $175 billion. "
            "Key recoupment conditions in JPMorgan's executive compensation agreements include: "
            "the Bonus Recoupment Policy applies to both cash incentive compensation and RSU/PSU awards; "
            "recovery is triggered by financial restatement or regulatory findings; "
            "employees must repay amounts as a lawful recovery under the award agreement."
        ),
        collection="auto",
        notes=(
            "Cross-collection: sec-filings for revenue, legal-docs for recoupment (exhibit 10.17 RSU). "
            "Note: JPM credit agreement is NOT indexed — using recoupment from the RSU award agreement."
        ),
    ),

    # ── Comparison ───────────────────────────────────────────────────────────

    EvalCase(
        id="E06",
        type="comparison",
        question="Compare Apple and Microsoft's revenue for fiscal year 2024",
        ground_truth=(
            "Apple's total net sales for fiscal year 2024 were $391.0 billion. "
            "Microsoft's total revenue for fiscal year 2024 was $245.1 billion. "
            "Apple had higher total revenue than Microsoft in FY2024 by approximately $146 billion."
        ),
        collection="auto",
        notes=(
            "Comparison requiring fresh retrieval for both companies. "
            "Tests that _comparison_retrieval fetches Apple AND Microsoft chunks independently. "
            "Common failure: only MSFT chunks retrieved because Apple FY2024 is a prior-year "
            "figure in their FY2025 10-K."
        ),
    ),

    # ── Legal-specific ────────────────────────────────────────────────────────

    EvalCase(
        id="E07",
        type="factual",
        question="What are the compensation clawback conditions in Tesla's executive agreements?",
        ground_truth=(
            "Tesla's executive agreements include clawback provisions that allow the company "
            "to recover incentive compensation in cases of financial statement restatement, "
            "fraud or intentional misconduct, or violation of company policies. "
            "The clawback covers cash bonuses and equity awards granted within a specified lookback period."
        ),
        collection="auto",
        notes=(
            "Legal-docs only question — must route to legal-docs namespace. "
            "Tests retrieval of Tesla EX-10 exhibits. "
            "Failure mode: agent says 'no information' when clawback clauses are present in the file."
        ),
    ),

    # ── Not-in-docs (groundedness) ────────────────────────────────────────────

    EvalCase(
        id="E08",
        type="not_in_docs",
        question="What is Microsoft's projected revenue for fiscal year 2027?",
        ground_truth=(
            "The indexed documents (10-K filings through FY2025) do not contain forward revenue "
            "projections for fiscal year 2027. The 10-K contains forward-looking statements "
            "but does not provide specific revenue forecasts."
        ),
        collection="auto",
        notes=(
            "Groundedness test: agent must not hallucinate future revenue figures. "
            "Expect high faithfulness (grounded refusal) but low context_recall."
        ),
    ),

    EvalCase(
        id="E09",
        type="not_in_docs",
        question="What is Nvidia's market capitalization as of today?",
        ground_truth=(
            "The indexed 10-K filings do not contain current market capitalization data. "
            "Market cap is not reported in annual reports; only shares outstanding are disclosed."
        ),
        collection="auto",
        notes=(
            "Groundedness test: real-time data that is never in static 10-K filings. "
            "Agent must say it doesn't have this information."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Routing cases — verify router path decisions (not RAGAS-scored)
# ---------------------------------------------------------------------------

ROUTING_CASES: list[RoutingCase] = [
    RoutingCase(
        id="R01",
        question="What was Microsoft's total revenue and net income in FY2025?",
        collection="auto",
        expected_route="vector",
        notes="Single-company factual on sec-filings → vector path.",
    ),
    RoutingCase(
        id="R02",
        question="What are the termination events in JPMorgan's credit agreements?",
        collection="auto",
        expected_route="vector",
        notes="Legal keyword → routes to legal-docs namespace via vector.",
    ),
    RoutingCase(
        id="R03",
        question="Compare Apple and Microsoft's revenue for fiscal year 2024",
        collection="sec-filings",
        expected_route="graph",
        notes=(
            "sec-filings collection + 'compare' keyword → graph path. "
            "Falls back to vector if graph has no Apple/MSFT nodes."
        ),
    ),
    RoutingCase(
        id="R04",
        question="What was Walmart's total net sales in fiscal year 2025?",
        collection="auto",
        expected_route="vector",
        notes="Single-company factual → vector on sec-filings.",
    ),
]
