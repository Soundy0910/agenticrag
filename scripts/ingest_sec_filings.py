#!/usr/bin/env python3
"""
scripts/ingest_sec_filings.py

Download the most recent 10-K filing from SEC EDGAR for a set of companies,
upload to Azure Blob Storage under the sec-filings/ prefix, run through the
existing parse → chunk → embed → Pinecone pipeline (collection=sec-filings),
then validate retrieval with comparison queries through the full agent graph.

Companies: AAPL, MSFT, NVDA, GOOGL, AMZN, JPM, TSLA, META, JNJ, V,
           WMT, XOM, PFE, KO, DIS  (15 tickers across 8 sectors)

Already-indexed documents are detected via Pinecone metadata filter and
skipped — safe to re-run incrementally as new tickers are added.

Usage:
    cd /Users/soundariyanvenkatachalam/Desktop/AgenticRAG
    python scripts/ingest_sec_filings.py

SEC EDGAR policy: requests must include a User-Agent header identifying
the requester. Without it the SEC blocks the request.
"""

import os
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# ---------------------------------------------------------------------------
# Bootstrap: project root on sys.path, .env loaded
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Try both common .env locations
for _env in [ROOT / ".env", ROOT / "backend" / "storage" / ".env", ROOT / "backend" / ".env"]:
    if _env.exists():
        load_dotenv(_env)
        break

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

# Silence noisy libraries
for _noisy in ("httpx", "httpcore", "openai", "pinecone", "unstructured", "azure"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Now import backend modules (after sys.path and .env are set)
# ---------------------------------------------------------------------------

import backend.config as cfg
from backend.ingest.parse import parse
from backend.ingest.embed_index import embed_and_index, _get_pinecone
from backend.storage.base import DocumentMetadata
from backend.storage._utils import file_type_from_name
from backend.agent.graph import run_query

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

COLLECTION = "sec-filings"
BLOB_PREFIX = "sec-filings"          # blobs land at  sec-filings/{TICKER}/{filename}
LOCAL_DIR = ROOT / "data" / "filings"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# SEC EDGAR User-Agent — required by SEC policy
SEC_USER_AGENT = "SoundariyanVenkatachalam vgsowindarian@gmail.com"
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Companies: ticker → CIK (10-digit zero-padded)
COMPANIES = {
    # Original 5 — tech
    "AAPL":  "0000320193",
    "MSFT":  "0000789019",
    "NVDA":  "0001045810",
    "GOOGL": "0001652044",
    "AMZN":  "0001018724",
    # 10 new — mixed sectors
    "JPM":   "0000019617",   # financials
    "TSLA":  "0001318605",   # automotive / EV
    "META":  "0001326801",   # social media
    "JNJ":   "0000200406",   # healthcare
    "V":     "0001403161",   # payments
    "WMT":   "0000104169",   # retail
    "XOM":   "0000034088",   # energy
    "PFE":   "0000078003",   # pharma
    "KO":    "0000021344",   # consumer staples
    "DIS":   "0000357301",   # media / entertainment
}

# Post-ingestion validation queries
BASIC_QUERIES = [
    "What was the total net revenue or net sales in the most recent fiscal year?",
    "What are the main risk factors the company identifies?",
]

COMPARISON_QUERIES = [
    (
        "Compare Tesla and Toyota's R&D spending. If you don't have Toyota's "
        "filing, say so and share what Tesla reported.",
        "Cross-sector R&D (graceful missing-data test)",
    ),
    (
        "Compare JPMorgan Chase and Apple's total revenue. What does each "
        "company say about its most important revenue driver?",
        "Cross-sector revenue comparison",
    ),
]

# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def _get_latest_10k(cik: str) -> tuple[str, str, str]:
    """
    Return (accession_nodash, primary_document_name, filing_date) for the
    most recent 10-K in EDGAR's submissions JSON.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()

    recent = data["filings"]["recent"]
    forms   = recent["form"]
    accnos  = recent["accessionNumber"]
    primdoc = recent["primaryDocument"]
    dates   = recent["filingDate"]

    for i, form in enumerate(forms):
        if form == "10-K":
            return accnos[i].replace("-", ""), primdoc[i], dates[i]

    raise ValueError(f"No 10-K found for CIK {cik}")


def download_10k(ticker: str, cik: str) -> Path:
    """
    Download the primary 10-K HTML document for `ticker` and save it locally.
    Returns the local file path.
    """
    local_ticker_dir = LOCAL_DIR / ticker
    local_ticker_dir.mkdir(parents=True, exist_ok=True)

    print(f"  [{ticker}] Fetching filing index from EDGAR …")
    accession, primary_doc, filing_date = _get_latest_10k(cik)
    time.sleep(0.12)  # SEC rate limit: ≤10 req/s

    # Determine extension; fall back to .htm
    ext = Path(primary_doc).suffix.lower() or ".htm"
    local_path = local_ticker_dir / f"10k_{filing_date}{ext}"

    if local_path.exists() and local_path.stat().st_size > 10_000:
        print(f"  [{ticker}] Already downloaded ({local_path.stat().st_size // 1024:,} KB) — skipping download")
        return local_path

    doc_url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession}/{primary_doc}"
    )
    print(f"  [{ticker}] Downloading {filing_date} 10-K → {local_path.name} …")
    r = requests.get(doc_url, headers=SEC_HEADERS, timeout=120, stream=True)
    r.raise_for_status()

    size = 0
    with open(local_path, "wb") as fh:
        for chunk in r.iter_content(chunk_size=65_536):
            fh.write(chunk)
            size += len(chunk)

    time.sleep(0.12)
    print(f"  [{ticker}] Downloaded {size // 1024:,} KB")
    return local_path


# ---------------------------------------------------------------------------
# Azure Blob upload
# ---------------------------------------------------------------------------

def upload_to_blob(local_path: Path, ticker: str) -> str:
    """
    Upload a local file to Azure Blob Storage under sec-filings/{ticker}/.
    Returns the blob name.
    """
    conn_str = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container = os.environ["AZURE_STORAGE_CONTAINER"]
    blob_name = f"{BLOB_PREFIX}/{ticker}/{local_path.name}"

    client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = client.get_blob_client(container=container, blob=blob_name)

    ext = local_path.suffix.lower().lstrip(".")
    content_type_map = {
        "htm": "text/html", "html": "text/html",
        "pdf": "application/pdf", "txt": "text/plain",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    from azure.storage.blob import ContentSettings
    with open(local_path, "rb") as fh:
        blob_client.upload_blob(
            fh,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
        )
    return blob_name


# ---------------------------------------------------------------------------
# Ingestion: parse → chunk → embed → Pinecone
# ---------------------------------------------------------------------------

def ingest_blob(blob_name: str, ticker: str, local_path: Path) -> dict:
    """
    Build DocumentMetadata from the blob name and run the full ingest pipeline.
    Returns a summary dict with char_count, parent_chunks, child_chunks, vectors.
    """
    filename = local_path.name
    file_type = file_type_from_name(filename)
    model = cfg.get_embedding_model(COLLECTION)

    meta = DocumentMetadata(
        doc_id=blob_name,
        source_type="azure_blob",
        collection=COLLECTION,
        filename=filename,
        file_type=file_type,
        last_modified=datetime.now(timezone.utc),
        access_scope=["public"],
        embedding_model=model,
        extra={"ticker": ticker, "container": os.environ.get("AZURE_STORAGE_CONTAINER", "")},
    )

    # Fetch bytes from Azure (real production path through AzureBlobSource)
    from backend.storage.azure_blob import AzureBlobSource
    source = AzureBlobSource(collection=COLLECTION, blob_prefix=f"{BLOB_PREFIX}/")
    content = source.fetch_document(blob_name)

    parsed = parse(content, meta)
    if not parsed.ok:
        return {"error": parsed.error or "empty parse", "vectors": 0}

    char_count = len(parsed.text)
    vectors = embed_and_index(parsed, meta)

    return {
        "char_count": char_count,
        "vectors": vectors,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Pinecone helpers
# ---------------------------------------------------------------------------

def _is_already_indexed(blob_name: str) -> bool:
    """
    Return True if at least one vector with doc_id == blob_name already exists
    in the sec-filings namespace. Uses a dummy zero-vector query with a
    metadata filter — cheap (1 match checked, no embedding call needed).
    """
    try:
        pc = _get_pinecone()
        index = pc.Index(cfg.PINECONE_INDEX_NAME)
        dim = cfg.get_embedding_dimension(cfg.get_embedding_model(COLLECTION))
        results = index.query(
            vector=[0.0] * dim,
            filter={"doc_id": {"$eq": blob_name}},
            top_k=1,
            namespace=COLLECTION,
            include_metadata=False,
        )
        return bool(results.get("matches"))
    except Exception:
        return False


def get_namespace_vector_count() -> int:
    """Return the total vector count in the sec-filings Pinecone namespace."""
    try:
        pc = _get_pinecone()
        index = pc.Index(cfg.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        ns = stats.get("namespaces", {}).get(COLLECTION, {})
        return ns.get("vector_count", 0)
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_query_block(label: str, question: str, idx: int) -> None:
    print(f"\n  Q{idx}: {label}")
    print(f"  {question}")
    print(f"  {'─' * 60}")
    try:
        t0 = time.time()
        state = run_query(question, collection=COLLECTION)
        elapsed = time.time() - t0
        answer = state.get("answer", "").strip()
        route  = state.get("route", "?")
        cites  = state.get("citations", [])
        print(f"  Route: {route}  |  Latency: {elapsed:.1f}s  |  Citations: {len(cites)}")
        print()
        for line in answer.split("\n")[:25]:
            print(f"    {line}")
        if len(answer.split("\n")) > 25:
            print("    … (truncated)")
    except Exception as exc:
        print(f"  ERROR: {exc}")


def main():
    print("\n" + "=" * 64)
    print("  SEC 10-K Ingestion Pipeline")
    print(f"  Collection : {COLLECTION}")
    print(f"  Companies  : {', '.join(COMPANIES)}")
    print("=" * 64 + "\n")

    results = {}

    # ── Step 1: Download + Upload + Ingest ──────────────────────────────────
    for ticker, cik in COMPANIES.items():
        print(f"\n{'─' * 50}")
        print(f"  {ticker}  (CIK {cik})")
        print(f"{'─' * 50}")

        try:
            local_path = download_10k(ticker, cik)

            # Derive blob name early so we can check if already indexed
            ext = local_path.suffix.lower()
            blob_name = f"{BLOB_PREFIX}/{ticker}/{local_path.name}"

            if _is_already_indexed(blob_name):
                print(f"  [{ticker}] Already indexed in Pinecone — skipping embed step")
                results[ticker] = {"skipped": True, "vectors": 0}
                continue

            print(f"  [{ticker}] Uploading to Azure Blob …")
            blob_name = upload_to_blob(local_path, ticker)
            print(f"  [{ticker}] Blob: {blob_name}")

            print(f"  [{ticker}] Running parse → chunk → embed → Pinecone …")
            t0 = time.time()
            summary = ingest_blob(blob_name, ticker, local_path)
            elapsed = time.time() - t0

            if summary.get("error"):
                print(f"  [{ticker}] ERROR: {summary['error']}")
                results[ticker] = summary
                continue

            print(f"  [{ticker}] ✓  {summary['char_count']:,} chars  |  {summary['vectors']} vectors  |  {elapsed:.1f}s")
            results[ticker] = summary

        except Exception as exc:
            print(f"  [{ticker}] FAILED: {exc}")
            results[ticker] = {"error": str(exc), "vectors": 0}

    # ── Step 2: Summary table ────────────────────────────────────────────────
    print("\n\n" + "=" * 64)
    print("  Ingestion Summary")
    print("=" * 64)
    print(f"  {'Ticker':<8} {'Chars':>12} {'Vectors':>10}  Status")
    print(f"  {'─'*6:<8} {'─'*10:>12} {'─'*8:>10}  {'─'*14}")
    vectors_this_run = 0
    for ticker, r in results.items():
        if r.get("skipped"):
            print(f"  {ticker:<8} {'—':>12} {'—':>10}  already indexed")
        elif r.get("error"):
            print(f"  {ticker:<8} {'—':>12} {'—':>10}  ERROR: {r['error'][:35]}")
        else:
            vectors_this_run += r.get("vectors", 0)
            print(f"  {ticker:<8} {r['char_count']:>12,} {r['vectors']:>10}  OK")

    ns_count = get_namespace_vector_count()
    # Storage estimate: 1536 dims × 4 bytes/float + ~400 bytes metadata per vector
    bytes_per_vec = 1536 * 4 + 400
    est_mb = (ns_count * bytes_per_vec) / (1024 ** 2) if ns_count > 0 else 0

    print(f"\n  Vectors upserted this run  : {vectors_this_run:,}")
    print(f"  Pinecone namespace total   : {ns_count:,} vectors")
    print(f"  Estimated storage used     : {est_mb:.1f} MB  (cap ~2,048 MB free tier)")
    print(f"  Headroom remaining         : ~{max(0, 2048 - est_mb):.0f} MB\n")

    # ── Step 3: Basic validation queries ────────────────────────────────────
    any_new = any(not r.get("skipped") and not r.get("error") for r in results.values())
    if not any(not r.get("error") for r in results.values()):
        print("No documents available — skipping queries.")
        return

    print("=" * 64)
    print("  Validation Queries  (collection=sec-filings)")
    print("=" * 64)

    q_idx = 1
    for question in BASIC_QUERIES:
        _run_query_block("Basic", question, q_idx)
        q_idx += 1

    # ── Step 4: Cross-sector comparison queries ──────────────────────────────
    print("\n" + "=" * 64)
    print("  Comparison Queries  (cross-sector)")
    print("=" * 64)

    for question, label in COMPARISON_QUERIES:
        _run_query_block(label, question, q_idx)
        q_idx += 1

    print("\n" + "=" * 64)
    print("  Done.")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    main()
