# AgenticRAG — Validation Report

**Date:** 2026-07-01  
**Iterations:** 10  
**Result:** 8/10 passed (80%)  
**Backend:** FastAPI + LangGraph on `localhost:8000`  
**Model:** gpt-4o-mini | Pinecone (25,265 vectors) | Neo4j GraphRAG  

---

## Test Results Summary

| ID | Label | Route | Collection(s) | Type | Grade | Citations | Latency | Pass |
|----|-------|-------|---------------|------|-------|-----------|---------|------|
| T01 | Microsoft revenue FY2025 | vector | sec-filings | factual_lookup | sufficient | 4 | 7.3s | ✅ |
| T02 | JPMorgan revenue + recoupment | vector | sec-filings, legal-docs | multi_document_reasoning | sufficient | 8 | 12.3s | ✅ |
| T03 | Apple vs Microsoft revenue FY2024 | graph | sec-filings | comparison | sufficient | 8 | 8.2s | ✅ |
| T04 | Walmart net sales + risk factors | vector | sec-filings | risk_analysis | sufficient | 8 | 10.9s | ✅ |
| T05 | Tesla clawback (legal role) | vector | **sec-filings** | factual_lookup | — | 0 | 1.6s | ❌ |
| T06 | NVIDIA risk factors | vector | sec-filings | risk_analysis | sufficient | 3 | 8.5s | ✅ |
| T07 | Amazon net income FY2025 | vector | sec-filings | factual_lookup | sufficient | 5 | **28.9s** | ❌ |
| T08 | Out-of-scope: weather question | vector | sec-filings | out_of_scope | sufficient | 0 | 2.1s | ✅ |
| T09 | Google vs Meta revenue FY2025 | graph | sec-filings | comparison | sufficient | 10 | 14.7s | ✅ |
| T10 | RBAC: legal role → sec-filings | vector | sec-filings | factual_lookup | — | 0 | 2.3s | ✅ |

---

## Iteration Details

### T01 — Microsoft revenue FY2025 ✅
**Q:** What was Microsoft's total revenue in fiscal year 2025?  
**A:** `$281,724 million for the fiscal year ended June 30, 2025. Source: [2] Total revenue section.`  
**Notes:** Perfect. Correct value, citation, collection routing. Fast (7.3s).

---

### T02 — JPMorgan revenue + recoupment (multi-collection) ✅
**Q:** What was JPMorgan's revenue in 2025 and what are the recoupment conditions in their executive compensation agreements?  
**A:** Multi-part answer — Part 1: $182,447M revenue from sec-filings; Part 2: recoupment terms from legal-docs.  
**Notes:** Multi-collection routing and decomposition worked correctly. 8 citations, grounded in both namespaces.

---

### T03 — Apple vs Microsoft revenue FY2024 (comparison) ✅
**Q:** Compare Apple and Microsoft total revenue for fiscal year 2024.  
**Route:** graph  
**A:** `Apple Inc.: $391,035M vs Microsoft: $245,122M for FY2024.`  
**Notes:** Correctly routed to Neo4j graph path for comparison. GraphRAG + vector hybrid retrieved correct figures. 8 citations.

---

### T04 — Walmart net sales + risk factors (multi-signal) ✅
**Q:** What were Walmart's net sales and what are their main risk factors?  
**A:** Net sales $706,413M for FY ended Jan 31 2026, plus categorised risk factors (cyber, operational, regulatory).  
**Notes:** Previously T04 failed (Faith=0.000, risk text missing). Now working — risk factors and financials both retrieved. 8 citations.

---

### T05 — Tesla clawback (legal role) ❌
**Q:** What are Tesla's executive clawback conditions?  
**Role:** legal  
**A:** `Access Denied — Role 'legal' does not have permission to access: sec-filings.`  
**Root cause:** `_classify_collections` scored sec-filings=1 (`tesla` keyword hit) and legal-docs=1 (`clawback` keyword hit). Scores tied, LLM fallback chose `sec-filings`. RBAC correctly blocked the legal role from sec-filings — but there is no retry/fallback to route to the user's allowed collection (`legal-docs`). The correct answer exists in legal-docs but was never queried.  
**Bug severity:** Medium — valid question from an authorized user returns an access denial instead of answering from the right namespace.

---

### T06 — NVIDIA risk factors ✅
**Q:** What are NVIDIA's main risk factors from their 10-K filing?  
**A:** Regulatory, competition, supply chain, AI demand risk factors from NVDA 10-K.  
**Notes:** Correct risk_analysis classification, vector retrieval, structured bullet output. 3 citations (slightly low — could retrieve more risk sections).

---

### T07 — Amazon net income FY2025 ❌
**Q:** What was Amazon's net income for fiscal year 2025?  
**A:** `Net income $ 77,670 million, fiscal year 2025, source: [1] Section: "Consolidated Statements of Operations".`  
**Issues:**  
1. **Company name missing from answer.** The value ($77,670M) is correct and grounded, but the answer omits "Amazon" from the response text. The `factual_lookup` system prompt doesn't explicitly require the subject company name in the output.  
2. **Unusually slow: 28.9s** — 3–4× typical latency (7–14s for other queries). Likely Amazon's 10-K is large and retrieval/reranking hit a slow path or had a retry cycle. Grade was `sufficient` so no retry loop, but retrieval still took much longer.  
**Bug severity:** Low (answer content correct, company name dropped) / Medium (latency spike needs investigation).

---

### T08 — Out-of-scope: weather question ✅
**Q:** What is the weather in San Francisco today?  
**A:** `Out of scope — This question falls outside the indexed document collections. [lists available collections]`  
**Notes:** Correctly detected with zero LLM retrieval cost (classify node fast-path, no Pinecone call). Note: `grade=sufficient` appears in result even for OOS — this is a stale state value carried over; does not affect behavior since generate_node short-circuits before grading.

---

### T09 — Google vs Meta revenue FY2025 ✅
**Q:** Compare Google and Meta's total revenue for fiscal year 2025.  
**Route:** graph  
**A:** Correct revenues with conflict notice: `Multiple different values found for 2024: $5,000M and $16,200M.`  
**Notes:** Answer is correct ($350,018M Google, $164,501M Meta). The conflict warning is a **false positive** — `_detect_conflicts` fired on segment-level figures ($5B, $16.2B) that exist in the retrieved chunks alongside total revenue, not on conflicting total revenue figures. 10 citations.

---

### T10 — RBAC: legal role accessing sec-filings ✅
**Q:** What was Apple's total revenue in fiscal year 2025?  
**Role:** legal  
**A:** `Access Denied — Role 'legal' does not have permission to access: sec-filings.`  
**Notes:** RBAC is working correctly when the collection is explicitly set. Denial message includes which roles are required and what the user's role has access to.

---

## Bug Report

### BUG-01 — Role-aware collection fallback missing (T05) 🔴 Medium
**Location:** `backend/agent/nodes.py` — `_classify_collections()` → `router_node()` → `access_check_node()`  
**Description:** When the collection classifier's LLM fallback picks a collection the user's role cannot access, the pipeline returns an access denial with no retry. There is no mechanism to re-route to a collection the user _is_ authorized for, even when the query clearly targets that collection (e.g., `legal` role asking about `clawback` gets routed to `sec-filings` due to Tesla company keywords winning the tie-break, then blocked).  
**Fix direction:** In `router_node` or `access_check_node`, after a denial, check if any of the user's allowed collections are plausible for the query (non-zero keyword score or LLM re-classification restricted to allowed collections) and retry routing to that collection.

---

### BUG-02 — Company name dropped from factual_lookup answer (T07) 🟡 Low
**Location:** `backend/agent/nodes.py` — `generate_node()` → `_QT_SYSTEM_PROMPTS["factual_lookup"]`  
**Description:** The factual_lookup system prompt instructs the model to include "the exact value, its unit, the fiscal period, and the source" but does not explicitly require the company/subject name. When the retrieved context doesn't repeat the company name prominently, the model returns a bare number with no subject attribution, making the answer hard to verify out of context.  
**Fix direction:** Add "the company or entity name" to the factual_lookup system prompt checklist.

---

### BUG-03 — False positive conflict detection on segment figures (T09) 🟡 Low
**Location:** `backend/agent/nodes.py` — `_detect_conflicts()` (line 1220)  
**Description:** The numeric conflict detector flags dollar amounts >$1,000M that differ by >15% within the same year. For multi-company comparisons (Google vs Meta), segment-level figures ($5B search ads, $16.2B cloud) appear alongside total revenues and trigger the warning even when total revenues are internally consistent. The conflict note is shown to users unnecessarily.  
**Fix direction:** Scope conflict detection to same-company chunks only. When `active_collections` contains a single collection and multiple companies are being compared, skip the conflict check or group by company before diffing values.

---

### OBSERVATION — OOS grade field carries stale state (T08) ℹ️ Cosmetic
**Location:** `backend/api/query.py` — `_stream_query()` done event  
**Description:** `grade=sufficient` appears in the done event for out-of-scope queries. The grade node never runs for OOS queries (generate_node short-circuits), but the initial state value (`""`) is never explicitly reset to `"not_applicable"` or similar, so whatever was in state previously can bleed through. No user-facing impact since the OOS answer is correctly returned.

---

### OBSERVATION — Amazon query 28.9s latency spike (T07) 🟡 Investigate
**Description:** T07 took 28.9s vs 7–15s for comparable factual lookups. Amazon's 10-K (AMZN) may have a larger indexed chunk count, triggering heavier BM25 scoring. No retry occurred (grade=sufficient), so the slowdown is purely in retrieval+reranking. Should profile `hybrid_search` timing for AMZN to determine if Cohere rerank is the bottleneck.

---

## What's Working Well

- **Multi-collection routing** (T02): keyword scoring + LLM decomposition correctly splits JPMorgan query across sec-filings + legal-docs.
- **GraphRAG hybrid path** (T03, T09): comparison queries correctly route to Neo4j; hybrid graph+vector retrieval returns precise metrics with source grounding.
- **Out-of-scope fast-path** (T08): OOS detected in classify node before any Pinecone call; clean refusal message with collection listing.
- **RBAC gate** (T10): access_check_node correctly blocks unauthorized collection access with clear denial message.
- **Risk factor retrieval** (T04, T06): Previously a known failure (RAGAS iter 7 T04 Faith=0.000). Now returning structured, grounded risk sections.
- **Conflict detection** (T09): System surfaces conflicting context rather than silently picking one value — correct behavior, but needs tighter scoping (BUG-03).
- **RAGAS regressions resolved**: T01, T02, T04, T05 (from prior eval) are all passing or improved.

---

## Priority Fixes

| Priority | Bug | File | Lines | Effort |
|----------|-----|------|-------|--------|
| 1 | BUG-01: Role-aware collection fallback | `nodes.py` `router_node` / `access_check_node` | ~774–910 | Medium |
| 2 | BUG-03: False-positive conflict on segment figures | `nodes.py` `_detect_conflicts` | ~1220–1276 | Small |
| 3 | BUG-02: Company name in factual answer | `nodes.py` `_QT_SYSTEM_PROMPTS` | ~1522–1530 | Trivial |
| 4 | OBS: Amazon latency spike | `retrieval/hybrid.py` | profile | Investigate |
