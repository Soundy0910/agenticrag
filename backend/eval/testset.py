"""
backend/eval/testset.py

Static test cases for RAGAS evaluation of the demo corpus.

Three categories:
  EVAL_CASES    — Q+A pairs for RAGAS scoring (faithfulness, relevance, precision, recall)
  ROUTING_CASES — questions where we verify the router's path decision, not the answer quality

EVAL_CASES types:
  factual       — single-hop factual lookup; answer is in the docs
  comparison    — multi-entity comparison; generate_node decomposes into sub-queries
  not_in_docs   — answer is genuinely absent; agent should say "I don't know" (groundedness test)

Routing decisions to verify:
  cag   — demo collection is small enough to fit in context window
  vector — larger collection or non-comparison question on finance
  graph  — finance/sec_filings collection + comparison keyword → Neo4j path

Ground-truth notes:
  Factual answers are derived from the indexed demo documents (PDF resume + DOCX resume + TOTALSL.csv).
  "not_in_docs" ground truths state absence explicitly — RAGAS context_recall will be near 0
  for these, which is correct (the corpus genuinely lacks this information).
"""

from dataclasses import dataclass, field


@dataclass
class EvalCase:
    id: str
    type: str          # factual | comparison | not_in_docs
    question: str
    ground_truth: str  # reference answer for RAGAS context_recall / answer_relevance
    collection: str = "demo"
    allowed_scopes: list[str] = field(default_factory=lambda: ["public"])
    notes: str = ""    # human-readable rationale for this case


@dataclass
class RoutingCase:
    id: str
    question: str
    collection: str
    expected_route: str   # cag | vector | graph
    notes: str = ""


# ---------------------------------------------------------------------------
# Eval cases — used for RAGAS scoring
# ---------------------------------------------------------------------------

EVAL_CASES: list[EvalCase] = [
    # ── Factual ──────────────────────────────────────────────────────────────
    EvalCase(
        id="E01",
        type="factual",
        question="What AWS certifications does the candidate have?",
        ground_truth=(
            "The candidate holds three certifications: AWS Certified AI Practitioner, "
            "AWS Certified Machine Learning - Specialty, and SnowPro Associate."
        ),
        notes="Directly stated in the Certifications section of both resume files.",
    ),
    EvalCase(
        id="E02",
        type="factual",
        question="What is the candidate's current job title and employer?",
        ground_truth=(
            "The candidate is currently a Product Development Intern on the Data Science Team "
            "at AmplifAI Solutions Inc., a role that started in January 2026."
        ),
        notes="Most recent EXPERIENCE entry in the PDF resume.",
    ),
    EvalCase(
        id="E03",
        type="factual",
        question="What GPA did the candidate achieve in their Master's degree?",
        ground_truth=(
            "The candidate achieved a GPA of 3.85 in their Master of Science in Business Analytics "
            "and AI at the University of Texas at Dallas."
        ),
        notes="EDUCATION section, MS entry.",
    ),
    EvalCase(
        id="E04",
        type="factual",
        question="What programming languages does the candidate know?",
        ground_truth=(
            "The candidate is proficient in Python, R, and SQL."
        ),
        notes="Technical Skills section lists Python, R, SQL as core languages.",
    ),
    EvalCase(
        id="E05",
        type="factual",
        question="What cloud platforms and ML tools does the candidate have experience with?",
        ground_truth=(
            "The candidate has experience with AWS (SageMaker, Glue, Athena), Azure, and Databricks "
            "as cloud platforms, and uses Scikit-learn, XGBoost, LangChain, RAG, Google ADK, and Neo4j "
            "as ML/AI tools."
        ),
        notes="Technical Skills section — cloud + AI/engineering tools subsections.",
    ),
    EvalCase(
        id="E06",
        type="factual",
        question="Describe the candidate's AWS purchase intent prediction project.",
        ground_truth=(
            "The candidate architected an ETL pipeline using AWS Athena to aggregate 400M+ events into "
            "90M session-level records, and deployed a real-time XGBoost endpoint on AWS SageMaker "
            "achieving 96.5% AUC for sub-second live inference."
        ),
        notes="PROJECTS section — Scalable Real Time Purchase Intent Prediction on AWS.",
    ),
    # ── Comparison ───────────────────────────────────────────────────────────
    EvalCase(
        id="E07",
        type="comparison",
        question=(
            "Compare the candidate's responsibilities at AmplifAI Solutions vs WellKnown Textile Mills. "
            "What were the key differences in their work?"
        ),
        ground_truth=(
            "At AmplifAI Solutions Inc. (2026–present) the candidate works as a Product Development Intern "
            "building Python automation tools for Azure cloud operations and operating an AI-powered quality "
            "assurance product. At WellKnown Textile Mills (2022–2024) they worked as an Applied Data Scientist "
            "orchestrating production predictive pipelines for scheduling and engineering multivariate "
            "time-series forecasting systems. The key difference is that AmplifAI is an AI/cloud software role "
            "while WellKnown was a manufacturing analytics role."
        ),
        notes="Comparison between two EXPERIENCE entries. Tests decomposition in generate_node.",
    ),
    # ── Not in docs (groundedness) ───────────────────────────────────────────
    EvalCase(
        id="E08",
        type="not_in_docs",
        question="What is the candidate's expected salary or compensation?",
        ground_truth=(
            "The documents do not contain any information about the candidate's salary expectation or compensation."
        ),
        notes=(
            "Groundedness test: agent must say 'I don't know' rather than hallucinate a number. "
            "Expect high faithfulness (grounded in context = nothing) but low context_recall."
        ),
    ),
    EvalCase(
        id="E09",
        type="not_in_docs",
        question="Does the candidate have a LinkedIn Premium subscription?",
        ground_truth=(
            "The documents do not mention whether the candidate has a LinkedIn Premium subscription."
        ),
        notes="Groundedness test: obscure personal detail guaranteed not in any resume.",
    ),
    EvalCase(
        id="E10",
        type="not_in_docs",
        question="What is the candidate's nationality or visa status?",
        ground_truth=(
            "The documents do not contain information about the candidate's nationality or visa status."
        ),
        notes=(
            "Groundedness test: sensitive personal info typically omitted from resumes. "
            "Agent must not guess or hallucinate."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Routing cases — used to verify router path decisions (not RAGAS-scored)
# ---------------------------------------------------------------------------

ROUTING_CASES: list[RoutingCase] = [
    RoutingCase(
        id="R01",
        question="What AWS certifications does the candidate have?",
        collection="demo",
        expected_route="vector",
        notes=(
            "demo namespace is small, but 'AWS' is detected as a named entity by _has_named_entity "
            "(capital mid-sentence word, len>2). Entity-specific queries bypass CAG → vector path. "
            "This prevents cross-section misattribution in the resume document."
        ),
    ),
    RoutingCase(
        id="R02",
        question="What is the candidate's GPA?",
        collection="demo",
        expected_route="vector",
        notes=(
            "'GPA' is detected as a named entity (all-caps, len>2, mid-sentence). "
            "Entity-specific queries bypass CAG → vector path even on small collections."
        ),
    ),
    RoutingCase(
        id="R03",
        question="Compare Apple and Microsoft revenue for fiscal year 2024",
        collection="finance",
        expected_route="graph",
        notes=(
            "finance collection + 'compare' keyword → router should pick graph path. "
            "Neo4j has the Apple/Microsoft data from the test_graph_rag run."
        ),
    ),
    RoutingCase(
        id="R04",
        question="What was Apple's total revenue in 2024?",
        collection="finance",
        expected_route="vector",
        notes=(
            "finance collection, single-company lookup, no comparison keyword → vector path. "
            "Pinecone has no finance vectors so retrieve will return empty, but the routing "
            "decision itself (vector) is what's being verified."
        ),
    ),
]
