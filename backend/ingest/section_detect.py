"""
backend/ingest/section_detect.py

Section type detection for 10-K SEC filings and legal exhibit chunks.

WHY THIS EXISTS:
  Pure semantic retrieval can't distinguish "income statement chunks" from
  "risk factor chunks" — they're all about the same company. Without
  section-type metadata, a revenue query retrieves whichever chunks are most
  semantically similar to the query, which may be segment breakdowns rather
  than the consolidated income statement, or section headers rather than the
  actual risk descriptions.

  By tagging each chunk with section_type at ingest time and filtering at
  query time, retrieval becomes: "give me the income_statement chunks most
  similar to this query" rather than "give me any chunk similar to this query."

SECTION TYPE TAXONOMY:
  income_statement  — Consolidated Statements of Operations / Income, revenue
                      tables, net sales, net income, EPS tables
  risk_factors      — Item 1A content: detailed risk descriptions (not headers)
  mda               — Item 7 Management's Discussion and Analysis narrative
  business_overview — Item 1 business description, company overview
  financial_notes   — Notes to Consolidated Financial Statements
  legal_provision   — EX-10 exhibit contract text, legal agreement clauses
  general           — Everything else (balance sheet, SCF, cover pages, etc.)

DETECTION APPROACH:
  Each chunk's first 600 characters (the "header area") carry the strongest
  signal — section titles like "ITEM 1A. RISK FACTORS" or
  "CONSOLIDATED STATEMENTS OF OPERATIONS" appear at the top of the section
  and therefore near the top of the first chunk in that section.

  Two detection passes:
    Pass 1: header area (first 600 chars) — high-confidence signal
    Pass 2: full chunk text — catches body-only content that lacks a header
              in this specific chunk (e.g., a continuation paragraph)

  Rules are ordered from most specific to least specific to avoid false
  classification. For example, "net sales $" is more specific than "revenue"
  alone, which could appear in many section types.
"""

import re

# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------
# Each entry: (section_type, [regex_patterns])
# Any matching pattern triggers that section type.
# Rules are evaluated in order — first match wins.

_SEC_FILING_RULES: list[tuple[str, list[str]]] = [
    (
        "income_statement",
        [
            # Exact section header patterns
            r"consolidated\s+statements?\s+of\s+(?:operations|income|earnings)",
            r"summary\s+results?\s+of\s+operations",
            # Revenue / income line items with dollar amounts (high confidence)
            r"(?:total\s+)?net\s+(?:sales|revenues?)\s+\$?\s*\d",
            r"net\s+revenues?\s+\$?\s*\d",
            r"total\s+revenues?\s+\$?\s*\d",
            r"total\s+net\s+revenues?\s+\$?\s*\d",
            r"total\s+net\s+sales\s+\$?\s*\d",
            # Table column header patterns (in millions, comparing years)
            r"(?:in\s+millions?|dollars\s+in\s+millions?).*(?:change|increase|decrease)",
            r"fiscal\s+\d{4}.*fiscal\s+\d{4}.*(?:revenue|net\s+sales)",
            # Net income/EPS lines
            r"net\s+income\s+(?:attributable\s+to\s+)?[\w\s]+\$?\s*\d{2,3},",
            r"diluted\s+(?:earnings|net\s+income)\s+per\s+(?:share|common\s+share)",
        ],
    ),
    (
        "risk_factors",
        [
            # Item 1A header
            r"\bitem\s+1a[\.\s]",
            r"item\s+1a\s*[\-—:]\s*risk\s+factors?",
            # Body text patterns (actual risk content, not headers)
            r"(?:the\s+following|these)\s+(?:are\s+)?(?:the\s+)?(?:material\s+)?risk\s+factors?",
            r"risk\s+factors?\s+that\s+(?:could|may|might|would)\s+(?:affect|impact|harm|adversely)",
            r"our\s+business\s+(?:is|are)\s+subject\s+to\s+(?:various|numerous|significant|many)",
            r"we\s+(?:face|are\s+exposed\s+to|may\s+experience)\s+(?:significant|various|several)\s+risk",
            r"material\s+adverse\s+(?:effect|impact|change)",
            r"could\s+materially\s+(?:and\s+adversely\s+)?(?:affect|impact|harm)",
            # Risk category signals that appear IN risk factor body text
            r"competition\s+(?:from|in\s+the|we\s+face)",
            r"cybersecurity\s+(?:incidents?|risks?|threats?|breaches?)",
            r"supply\s+chain\s+(?:disruptions?|risks?|challenges?)",
            r"macroeconomic\s+(?:conditions?|risks?|factors?|uncertainties?)",
            r"data\s+privacy\s+(?:and\s+)?(?:security\s+)?(?:risks?|regulations?|laws?)",
            r"regulatory\s+(?:changes?|requirements?|risks?|compliance)",
            r"climate\s+change\s+(?:risks?|may\s+impact|could\s+affect)",
            r"labor\s+(?:shortages?|disruptions?|costs?|relations?)",
        ],
    ),
    (
        "mda",
        [
            r"\bitem\s+7[\.\s]",
            r"management'?s?\s+discussion\s+and\s+analysis",
            r"overview\s+of\s+(?:our\s+)?(?:business\s+)?(?:results|financial\s+(?:performance|condition))",
            r"(?:fiscal|the\s+following)\s+(?:year|quarter)\s+(?:ended|overview)",
        ],
    ),
    (
        "business_overview",
        [
            # Item 1 but NOT Item 1A
            r"\bitem\s+1[\.\s](?!a\b)",
            r"general\s+(?:development\s+of\s+)?(?:our\s+)?business",
            r"we\s+(?:design|develop|manufacture|sell|produce|operate|provide)\s+(?:and\s+(?:sell|market|distribute))?",
            r"our\s+(?:company|business)\s+(?:was\s+)?(?:founded|incorporated|formed|established)",
        ],
    ),
    (
        "financial_notes",
        [
            r"notes?\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements?",
            r"note\s+\d+[\s\-—:]+\w",
            r"summary\s+of\s+significant\s+accounting\s+(?:policies|estimates|judgments)",
            r"basis\s+of\s+(?:presentation|consolidation|preparation)",
        ],
    ),
]

_LEGAL_RULES: list[tuple[str, list[str]]] = [
    (
        "legal_provision",
        [
            # Agreement / exhibit openers
            r"(?:this|the)\s+(?:agreement|contract|exhibit|award|plan|policy)",
            r"whereas\b|recitals?\b|witnesseth\b",
            # Legal clause markers
            r"(?:terms?\s+and\s+conditions?|governing\s+law|applicable\s+law)",
            r"indemnif(?:y|ication|ied|ies)",
            r"liability\s+(?:of|for|arising)",
            r"clawback|recoupment|erroneously\s+awarded",
            r"termination|event\s+of\s+default",
            r"compensation|bonus|award(?:\s+agreement)?|equity\s+(?:award|grant|incentive)",
            r"representations?\s+and\s+warranties?",
            r"arbitration|dispute\s+resolution",
            r"intellectual\s+property|proprietary\s+(?:rights?|information)",
            r"confidential(?:ity)?|non-?disclosure",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Compile to regex patterns
# ---------------------------------------------------------------------------

def _compile(
    rules: list[tuple[str, list[str]]],
) -> list[tuple[str, list[re.Pattern]]]:
    return [
        (
            section_type,
            [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns],
        )
        for section_type, patterns in rules
    ]


_COMPILED_SEC: list[tuple[str, list[re.Pattern]]] = _compile(_SEC_FILING_RULES)
_COMPILED_LEGAL: list[tuple[str, list[re.Pattern]]] = _compile(_LEGAL_RULES)

# Character range for the "header area" — first ~600 chars contain section titles
_HEADER_CHARS = 600


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_section_type(
    text: str,
    collection: str = "",
    filename: str = "",
) -> str:
    """
    Detect the section type of a chunk.

    Parameters
    ----------
    text : str
        The chunk's source text (parent text, not child text).
    collection : str
        Pinecone namespace — 'legal-docs' forces legal rules.
    filename : str
        Source filename — EX-10 / exhibit filenames force legal rules.

    Returns
    -------
    str
        One of: income_statement | risk_factors | mda | business_overview |
                financial_notes | legal_provision | general
    """
    if not text:
        return "general"

    is_legal = (
        "legal" in collection.lower()
        or re.search(r"\bex(?:hibit)?\s*10\b", filename, re.IGNORECASE) is not None
        or "legal" in filename.lower()
    )

    if is_legal:
        # Legal docs: try rules first, then default to legal_provision
        header = text[:_HEADER_CHARS]
        for section_type, patterns in _COMPILED_LEGAL:
            for p in patterns:
                if p.search(header) or p.search(text):
                    return section_type
        return "legal_provision"

    # SEC filings: two-pass detection
    header = text[:_HEADER_CHARS]

    # Pass 1: header area (strongest signal — section titles appear here)
    for section_type, patterns in _COMPILED_SEC:
        for p in patterns:
            if p.search(header):
                return section_type

    # Pass 2: full text (catches continuation paragraphs without their own header)
    for section_type, patterns in _COMPILED_SEC:
        for p in patterns:
            if p.search(text):
                return section_type

    return "general"


def section_type_label(section_type: str) -> str:
    """Human-readable label for display in citations and trace UI."""
    return {
        "income_statement": "Income Statement",
        "risk_factors": "Risk Factors (Item 1A)",
        "mda": "MD&A (Item 7)",
        "business_overview": "Business (Item 1)",
        "financial_notes": "Financial Notes",
        "legal_provision": "Legal Provision",
        "general": "General",
    }.get(section_type, section_type.replace("_", " ").title())
