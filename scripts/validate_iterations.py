"""
Validation script: 10 test iterations against the live AgenticRAG API.
Parses SSE stream, captures answer + metadata, prints pass/fail verdict.

Usage:
    python scripts/validate_iterations.py
"""

import json
import sys
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000/api/query"

# ── Test cases ────────────────────────────────────────────────────────────────

TESTS = [
    {
        "id": "T01",
        "label": "Microsoft revenue FY2025 (factual_lookup, sec-filings)",
        "payload": {"question": "What was Microsoft's total revenue in fiscal year 2025?", "collection": "auto", "role": "finance"},
        "expect_route": "vector",
        "expect_collection": "sec-filings",
        "expect_keywords": ["281", "revenue"],
        "expect_not_oos": True,
    },
    {
        "id": "T02",
        "label": "JPMorgan revenue + recoupment (multi-collection)",
        "payload": {"question": "What was JPMorgan's revenue in 2025 and what are the recoupment conditions in their executive compensation agreements?", "collection": "auto", "role": "admin"},
        "expect_route": "vector",
        "expect_keywords": ["recoupment", "jpmorgan"],
        "expect_not_oos": True,
    },
    {
        "id": "T03",
        "label": "Apple vs Microsoft revenue FY2024 (comparison)",
        "payload": {"question": "Compare Apple and Microsoft total revenue for fiscal year 2024.", "collection": "auto", "role": "finance"},
        "expect_route": None,
        "expect_keywords": ["apple", "microsoft"],
        "expect_not_oos": True,
    },
    {
        "id": "T04",
        "label": "Walmart net sales + risk factors (multi-signal)",
        "payload": {"question": "What were Walmart's net sales and what are their main risk factors?", "collection": "auto", "role": "finance"},
        "expect_route": "vector",
        "expect_collection": "sec-filings",
        "expect_keywords": ["walmart", "risk"],
        "expect_not_oos": True,
    },
    {
        "id": "T05",
        "label": "Tesla clawback conditions (legal-docs)",
        "payload": {"question": "What are Tesla's executive clawback conditions?", "collection": "auto", "role": "legal"},
        "expect_route": "vector",
        "expect_collection": "legal-docs",
        "expect_keywords": ["clawback", "tesla"],
        "expect_not_oos": True,
    },
    {
        "id": "T06",
        "label": "NVIDIA risk factors (risk_analysis)",
        "payload": {"question": "What are NVIDIA's main risk factors from their 10-K filing?", "collection": "auto", "role": "finance"},
        "expect_route": "vector",
        "expect_collection": "sec-filings",
        "expect_keywords": ["nvidia", "risk"],
        "expect_not_oos": True,
    },
    {
        "id": "T07",
        "label": "Amazon net income FY2025 (factual_lookup)",
        "payload": {"question": "What was Amazon's net income for fiscal year 2025?", "collection": "auto", "role": "finance"},
        "expect_route": "vector",
        "expect_collection": "sec-filings",
        "expect_keywords": ["amazon", "net income"],
        "expect_not_oos": True,
    },
    {
        "id": "T08",
        "label": "Out-of-scope: weather question (OOS handling)",
        "payload": {"question": "What is the weather in San Francisco today?", "collection": "auto", "role": "general"},
        "expect_route": None,
        "expect_keywords": ["out of scope", "scope"],
        "expect_not_oos": False,
        "expect_oos": True,
    },
    {
        "id": "T09",
        "label": "Google vs Meta revenue comparison",
        "payload": {"question": "Compare Google and Meta's total revenue for fiscal year 2025.", "collection": "auto", "role": "finance"},
        "expect_route": None,
        "expect_keywords": ["google", "meta"],
        "expect_not_oos": True,
    },
    {
        "id": "T10",
        "label": "RBAC: legal role accessing sec-filings (access denied)",
        "payload": {"question": "What was Apple's total revenue in fiscal year 2025?", "collection": "sec-filings", "role": "legal"},
        "expect_route": None,
        "expect_keywords": ["access denied", "denied"],
        "expect_not_oos": False,
        "expect_access_denied": True,
    },
]


# ── SSE parser ────────────────────────────────────────────────────────────────

def call_query_sse(payload: dict, timeout: int = 60) -> dict:
    """POST to /api/query and parse the SSE stream. Returns the 'done' event dict."""
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    done_event = {}
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            buffer = b""
            while True:
                chunk = resp.read(1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n\n" in buffer:
                    line, buffer = buffer.split(b"\n\n", 1)
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        try:
                            ev = json.loads(line[6:])
                            events.append(ev)
                            if ev.get("event") == "done":
                                done_event = ev
                        except json.JSONDecodeError:
                            pass
    except urllib.error.URLError as e:
        return {"error": str(e), "events": events}
    except Exception as e:
        return {"error": str(e), "events": events}

    done_event["_all_events"] = events
    return done_event


# ── Verdict checker ───────────────────────────────────────────────────────────

def evaluate(test: dict, result: dict) -> tuple[bool, list[str]]:
    issues = []
    answer = (result.get("answer") or "").lower()

    if "error" in result:
        issues.append(f"API error: {result['error']}")
        return False, issues

    if not answer:
        issues.append("Empty answer")

    # OOS check
    if test.get("expect_oos"):
        if "out of scope" not in answer and "scope" not in answer and "outside" not in answer:
            issues.append(f"Expected out-of-scope response but got: {answer[:100]}")
    elif test.get("expect_not_oos"):
        if "out of scope" in answer or result.get("query_type") == "out_of_scope":
            issues.append("Incorrectly classified as out-of-scope")

    # Access denied check
    if test.get("expect_access_denied"):
        if not result.get("access_denied") and "access denied" not in answer:
            issues.append("Expected access denied but was not triggered")

    # Route check
    if test.get("expect_route") and result.get("route") != test["expect_route"]:
        issues.append(f"Route: expected {test['expect_route']!r} got {result.get('route')!r}")

    # Collection check
    active_cols = result.get("active_collections") or []
    if test.get("expect_collection"):
        if test["expect_collection"] not in active_cols:
            issues.append(f"Collection: expected {test['expect_collection']!r} in {active_cols}")

    # Keyword check
    for kw in test.get("expect_keywords", []):
        if kw.lower() not in answer:
            issues.append(f"Missing keyword: {kw!r}")

    # Citation check (should have citations for domain questions)
    if test.get("expect_not_oos") and not test.get("expect_access_denied"):
        citations = result.get("citations") or []
        if not citations:
            issues.append("No citations returned")

    return len(issues) == 0, issues


# ── Main runner ───────────────────────────────────────────────────────────────

def main():
    results = []
    print(f"\n{'='*70}")
    print(f"  AgenticRAG — Validation Run ({len(TESTS)} iterations)")
    print(f"{'='*70}\n")

    for test in TESTS:
        print(f"[{test['id']}] {test['label']}")
        print(f"  Q: {test['payload']['question']}")
        t0 = time.time()
        result = call_query_sse(test["payload"])
        elapsed = round(time.time() - t0, 1)

        passed, issues = evaluate(test, result)

        answer_preview = (result.get("answer") or "")[:200].replace("\n", " ")
        route = result.get("route", "?")
        cols = result.get("active_collections", [])
        query_type = result.get("query_type", "?")
        grade = result.get("grade", "?")
        citations = result.get("citations") or []
        metrics = result.get("metrics") or {}

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  [{elapsed}s | route={route} | cols={cols} | type={query_type} | grade={grade} | citations={len(citations)}]")
        print(f"  Answer: {answer_preview!r}")
        if issues:
            for iss in issues:
                print(f"  ⚠  {iss}")
        print()

        results.append({
            "id": test["id"],
            "label": test["label"],
            "question": test["payload"]["question"],
            "passed": passed,
            "issues": issues,
            "answer_preview": answer_preview,
            "route": route,
            "active_collections": cols,
            "query_type": query_type,
            "grade": grade,
            "citation_count": len(citations),
            "elapsed_s": elapsed,
            "metrics": metrics,
            "access_denied": result.get("access_denied", False),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    passed_count = sum(1 for r in results if r["passed"])
    failed = [r for r in results if not r["passed"]]

    print(f"\n{'='*70}")
    print(f"  SUMMARY: {passed_count}/{len(results)} passed")
    print(f"{'='*70}")

    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  [{r['id']}] {r['label']}")
            for iss in r["issues"]:
                print(f"        ⚠  {iss}")

    # Write JSON results for the report
    with open("/Users/soundariyanvenkatachalam/Desktop/AgenticRAG/VALIDATION_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to VALIDATION_RESULTS.json")
    return results


if __name__ == "__main__":
    main()
