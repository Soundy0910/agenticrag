#!/usr/bin/env python3
"""
scripts/ingest_legal_docs.py

1. Clean up the demo collection:
   - Delete the 'demo' Pinecone namespace (all vectors)
   - Delete demo-era blobs from Azure (anything not under a known prefix)

2. Download 3-5 real material contracts (EX-10.*) from SEC EDGAR exhibit
   pages of companies already in our sec-filings index. These are public
   domain government filings — no licensing restrictions.

3. Upload to Azure Blob under 'legal-docs/' prefix → ingest through the
   existing parse → chunk → embed → Pinecone pipeline, collection=legal-docs.

4. Cross-collection query test:
   - Financial question  → sec-filings  (must NOT bleed into legal-docs)
   - Legal question      → legal-docs   (must NOT bleed into sec-filings)

5. Print final Pinecone state: collections, vector counts, storage estimate.

Usage:
    cd /path/to/AgenticRAG
    python scripts/ingest_legal_docs.py
"""

import os
import re
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient, ContentSettings

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _env in [ROOT / ".env", ROOT / "backend" / "storage" / ".env", ROOT / "backend" / ".env"]:
    if _env.exists():
        load_dotenv(_env)
        break

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for _noisy in ("httpx", "httpcore", "openai", "pinecone", "unstructured", "azure", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

import backend.config as cfg
from backend.ingest.parse import parse
from backend.ingest.embed_index import embed_and_index, _get_pinecone
from backend.storage.base import DocumentMetadata
from backend.storage._utils import file_type_from_name
from backend.storage.azure_blob import AzureBlobSource
from backend.agent.graph import run_query

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLLECTION      = "legal-docs"
BLOB_PREFIX     = "legal-docs"
LOCAL_DIR       = ROOT / "data" / "legal"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

SEC_HEADERS = {
    "User-Agent": "SoundariyanVenkatachalam vgsowindarian@gmail.com",
    "Accept-Encoding": "gzip, deflate",
}

# Known prefixes we want to KEEP in Azure Blob — everything else is demo
KEEPER_PREFIXES = ("sec-filings/", "legal-docs/")

# Target companies for exhibit discovery: (ticker, CIK)
# Broad mix: financials, tech, healthcare, payments, retail, energy, pharma
EXHIBIT_TARGETS = [
    ("JPM",   "0000019617"),   # financials — credit agreements
    ("MSFT",  "0000789019"),   # tech — indemnification, license
    ("AMZN",  "0001018724"),   # tech — supply, employment
    ("JNJ",   "0000200406"),   # healthcare — regulatory, IP licensing
    ("V",     "0001403161"),   # payments — card network agreements
    ("GOOGL", "0001652044"),   # tech — cloud service agreements
    ("TSLA",  "0001318605"),   # EV — manufacturing, supply agreements
    ("META",  "0001326801"),   # social media — data use, employment
    ("WMT",   "0000104169"),   # retail — supplier agreements
    ("PFE",   "0000078003"),   # pharma — licensing, milestone payments
]

MAX_EXHIBITS      = 10     # get up to 10 exhibits for broader coverage
MAX_EXHIBIT_BYTES = 4 * 1024 * 1024  # skip individual exhibits > 4 MB


# ---------------------------------------------------------------------------
# EDGAR exhibit text extractor
# ---------------------------------------------------------------------------

def _edgar_to_text(raw: bytes) -> str:
    """
    Convert an EDGAR exhibit file to clean plain text.

    EDGAR wraps all exhibit content in an SGML <DOCUMENT> envelope:
        <DOCUMENT>
        <TYPE>EX-10.1
        ...
        <TEXT>
        <html>...</html>
        </TEXT>
    unstructured's "fast" strategy can't parse this wrapper, so we strip it
    ourselves: extract the inner HTML/text after <TEXT>, then strip HTML tags
    and unescape HTML entities to get readable prose.
    """
    import html as html_lib

    text = raw.decode("utf-8", errors="replace")

    # Strip EDGAR SGML envelope: grab everything between <TEXT> and </TEXT>
    # (or end of file if </TEXT> is absent)
    text_block_m = re.search(r"<TEXT>(.*?)(?:</TEXT>|\Z)", text, re.DOTALL | re.IGNORECASE)
    if text_block_m:
        text = text_block_m.group(1)

    # Strip all HTML/XML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode HTML entities (&amp; → &, &#x2019; → ', etc.)
    text = html_lib.unescape(text)

    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------------------------------------------------------------------------
# Phase 1: Cleanup demo collection
# ---------------------------------------------------------------------------

def delete_demo_pinecone() -> int:
    """Delete all vectors in the 'demo' Pinecone namespace. Returns 0 or -1."""
    print("\n[cleanup] Deleting Pinecone namespace 'demo' …")
    try:
        pc = _get_pinecone()
        index = pc.Index(cfg.PINECONE_INDEX_NAME)
        stats_before = index.describe_index_stats()
        ns = stats_before.get("namespaces", {})
        if "demo" not in ns:
            print("         namespace 'demo' not found — already clean")
            return 0
        count = ns["demo"].get("vector_count", 0)
        index.delete(delete_all=True, namespace="demo")
        print(f"         deleted {count:,} vectors from namespace 'demo'")
        return count
    except Exception as exc:
        print(f"         WARNING: could not delete demo namespace: {exc}")
        return -1


def delete_demo_blobs() -> list[str]:
    """
    Delete blobs in the Azure container that are NOT under a known prefix.
    Returns list of deleted blob names.
    """
    print("\n[cleanup] Scanning Azure Blob container for demo files …")
    conn_str  = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container = os.environ["AZURE_STORAGE_CONTAINER"]
    client    = BlobServiceClient.from_connection_string(conn_str)
    cc        = client.get_container_client(container)

    to_delete = []
    for blob in cc.list_blobs():
        name = blob["name"]
        if not any(name.startswith(p) for p in KEEPER_PREFIXES):
            to_delete.append(name)

    if not to_delete:
        print("         no demo blobs found")
        return []

    for name in to_delete:
        cc.delete_blob(name)
        print(f"         deleted blob: {name}")

    return to_delete


# ---------------------------------------------------------------------------
# Phase 2: Discover and download EDGAR EX-10.* exhibits
# ---------------------------------------------------------------------------

def _get_latest_10k_accession(cik: str) -> tuple[str, str]:
    """Return (accession_nodash, accession_with_dashes) for the latest 10-K."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    r   = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data   = r.json()
    recent = data["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            acc_dashes = recent["accessionNumber"][i]          # "0000019617-26-000042"
            acc_nodash = acc_dashes.replace("-", "")           # "000001961726000042"
            return acc_nodash, acc_dashes
    raise ValueError(f"No 10-K found for CIK {cik}")


def _find_exhibits(cik: str, acc_nodash: str, acc_dashes: str) -> list[dict]:
    """
    Fetch the EDGAR filing index page and extract EX-10.* exhibit URLs.
    Returns list of dicts with keys: type, filename, url, description.
    """
    cik_int   = int(cik)
    index_url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}/"
        f"{acc_nodash}/{acc_dashes}-index.htm"
    )
    r = requests.get(index_url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(0.12)

    html     = r.text
    exhibits = []

    # EDGAR index tables have rows like:
    #   <td>1</td><td>Description</td>
    #   <td><a href="/Archives/...">filename.htm</a></td>
    #   <td>EX-10.1</td><td>12345</td>
    row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
    href_pattern = re.compile(
        r'href="(/Archives/edgar/data/[^"]+\.(htm|html|txt))"',
        re.IGNORECASE
    )
    type_pattern = re.compile(r'>(EX-10[^<]{0,20})<', re.IGNORECASE)
    desc_pattern = re.compile(r'<td[^>]*>([^<]{5,120})</td>', re.IGNORECASE)

    for row_match in row_pattern.finditer(html):
        row = row_match.group(1)
        type_m = type_pattern.search(row)
        if not type_m:
            continue
        href_m = href_pattern.search(row)
        if not href_m:
            continue
        descs = desc_pattern.findall(row)
        desc  = descs[0].strip() if descs else ""
        exhibits.append({
            "type":        type_m.group(1).strip(),
            "url":         "https://www.sec.gov" + href_m.group(1),
            "filename":    href_m.group(1).split("/")[-1],
            "description": desc,
        })

    return exhibits


def download_exhibits() -> list[Path]:
    """
    Discover and download up to MAX_EXHIBITS EX-10.* contracts from EDGAR.
    Returns list of local file paths.
    """
    downloaded: list[Path] = []

    for ticker, cik in EXHIBIT_TARGETS:
        if len(downloaded) >= MAX_EXHIBITS:
            break

        print(f"\n  [{ticker}] Fetching filing index …")
        try:
            acc_nodash, acc_dashes = _get_latest_10k_accession(cik)
            exhibits = _find_exhibits(cik, acc_nodash, acc_dashes)
        except Exception as exc:
            print(f"  [{ticker}] index fetch failed: {exc}")
            continue

        if not exhibits:
            print(f"  [{ticker}] no EX-10.* exhibits found in filing index")
            continue

        print(f"  [{ticker}] found {len(exhibits)} EX-10.* exhibit(s)")

        for ex in exhibits:
            if len(downloaded) >= MAX_EXHIBITS:
                break

            safe_name  = re.sub(r'[^a-zA-Z0-9._-]', '_', ex["filename"])
            local_path = LOCAL_DIR / f"{ticker}_{safe_name}"
            txt_cached = local_path.with_suffix(".txt")

            if txt_cached.exists() and txt_cached.stat().st_size > 1_000:
                print(f"  [{ticker}] {ex['type']} already cached ({txt_cached.name})")
                downloaded.append(txt_cached)
                continue

            try:
                r = requests.get(ex["url"], headers=SEC_HEADERS, timeout=60, stream=True)
                r.raise_for_status()
                time.sleep(0.12)

                size = 0
                chunks = []
                for chunk in r.iter_content(65_536):
                    size += len(chunk)
                    if size > MAX_EXHIBIT_BYTES:
                        break
                    chunks.append(chunk)

                if size > MAX_EXHIBIT_BYTES:
                    print(f"  [{ticker}] {ex['type']} too large (>{MAX_EXHIBIT_BYTES//1024//1024}MB) — skipping")
                    continue

                raw_bytes = b"".join(chunks)

                # Convert EDGAR wrapper + HTML to clean plain text so that
                # parse.py's txt path handles it (unstructured can't parse
                # the EDGAR SGML <DOCUMENT> envelope format directly).
                clean_text = _edgar_to_text(raw_bytes)
                if len(clean_text) < 200:
                    print(f"  [{ticker}] {ex['type']} yielded too little text after stripping — skipping")
                    continue

                # Save as .txt so the pipeline uses the direct decode path
                txt_path = local_path.with_suffix(".txt")
                txt_path.write_text(clean_text, encoding="utf-8")
                print(f"  [{ticker}] {ex['type']} → {txt_path.name}  ({len(clean_text)//1024:,} KB text)")
                downloaded.append(txt_path)

            except Exception as exc:
                print(f"  [{ticker}] {ex['type']} download failed: {exc}")
                continue

    return downloaded


# ---------------------------------------------------------------------------
# Phase 3 & 4: Upload + Ingest
# ---------------------------------------------------------------------------

def upload_to_blob(local_path: Path) -> str:
    """Upload a file under the legal-docs/ prefix. Returns blob name."""
    conn_str  = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
    container = os.environ["AZURE_STORAGE_CONTAINER"]
    blob_name = f"{BLOB_PREFIX}/{local_path.name}"

    client      = BlobServiceClient.from_connection_string(conn_str)
    blob_client = client.get_blob_client(container=container, blob=blob_name)

    ext = local_path.suffix.lower().lstrip(".")
    ct  = {"htm": "text/html", "html": "text/html", "txt": "text/plain"}.get(ext, "application/octet-stream")

    with open(local_path, "rb") as fh:
        blob_client.upload_blob(fh, overwrite=True, content_settings=ContentSettings(content_type=ct))

    return blob_name


def ingest_blob(blob_name: str, local_path: Path) -> dict:
    """Fetch from Azure Blob → parse → chunk → embed → Pinecone."""
    model     = cfg.get_embedding_model(COLLECTION)
    file_type = file_type_from_name(local_path.name)

    meta = DocumentMetadata(
        doc_id       = blob_name,
        source_type  = "azure_blob",
        collection   = COLLECTION,
        filename     = local_path.name,
        file_type    = file_type,
        last_modified= datetime.now(timezone.utc),
        access_scope = ["public"],
        embedding_model = model,
        extra        = {"container": os.environ.get("AZURE_STORAGE_CONTAINER", "")},
    )

    source  = AzureBlobSource(collection=COLLECTION, blob_prefix=f"{BLOB_PREFIX}/")
    content = source.fetch_document(blob_name)

    parsed  = parse(content, meta)
    if not parsed.ok:
        return {"error": parsed.error or "empty parse", "vectors": 0}

    vectors = embed_and_index(parsed, meta)
    return {"char_count": len(parsed.text), "vectors": vectors, "error": None}


# ---------------------------------------------------------------------------
# Phase 5: Cross-collection routing test
# ---------------------------------------------------------------------------

def run_cross_collection_test():
    TESTS = [
        {
            "label":      "Financial → sec-filings",
            "question":   "What was Apple's total net revenue or net sales in the most recent fiscal year?",
            "collection": "sec-filings",
            "expect":     "Apple revenue from their 10-K (should NOT mention contracts/clauses)",
        },
        {
            "label":      "Legal → legal-docs",
            "question":   "What are the key termination clauses or events of default described in these agreements?",
            "collection": "legal-docs",
            "expect":     "Contract termination terms (should NOT mention Apple's revenue)",
        },
    ]

    print("\n" + "=" * 64)
    print("  Cross-Collection Isolation Test")
    print("=" * 64)

    for i, t in enumerate(TESTS, 1):
        print(f"\n  Q{i} [{t['label']}]")
        print(f"  {t['question']}")
        print(f"  {'─' * 60}")
        try:
            t0     = time.time()
            state  = run_query(t["question"], collection=t["collection"])
            elapsed= time.time() - t0
            answer = (state.get("answer") or "").strip()
            route  = state.get("route", "?")
            cites  = state.get("citations", [])
            print(f"  Collection : {t['collection']}")
            print(f"  Route      : {route}  |  Latency: {elapsed:.1f}s  |  Citations: {len(cites)}")
            print(f"  Expect     : {t['expect']}")
            print()
            for line in answer.split("\n")[:20]:
                print(f"    {line}")
            if len(answer.split("\n")) > 20:
                print("    … (truncated)")
        except Exception as exc:
            print(f"  ERROR: {exc}")


# ---------------------------------------------------------------------------
# Phase 6: Final state report
# ---------------------------------------------------------------------------

def print_final_state():
    print("\n" + "=" * 64)
    print("  Final Pinecone State")
    print("=" * 64)
    try:
        pc    = _get_pinecone()
        index = pc.Index(cfg.PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()
        ns    = stats.get("namespaces", {})

        bytes_per_vec = 1536 * 4 + 400  # 1536 dims × 4 bytes + ~400B metadata
        total_vecs    = sum(v.get("vector_count", 0) for v in ns.values())
        total_mb      = (total_vecs * bytes_per_vec) / (1024 ** 2)

        print(f"\n  {'Collection':<18} {'Vectors':>10}")
        print(f"  {'─'*16:<18} {'─'*8:>10}")
        for name, info in sorted(ns.items()):
            vc = info.get("vector_count", 0)
            print(f"  {name:<18} {vc:>10,}")

        print(f"\n  Total vectors    : {total_vecs:,}")
        print(f"  Est. storage     : {total_mb:.1f} MB")
        print(f"  Free-tier cap    : 2,048 MB")
        print(f"  Headroom         : ~{max(0, 2048 - total_mb):.0f} MB\n")
    except Exception as exc:
        print(f"  ERROR reading Pinecone stats: {exc}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 64)
    print("  Legal Docs Ingestion + Demo Cleanup")
    print(f"  Collection : {COLLECTION}")
    print("=" * 64)

    # ── Phase 1: Cleanup ────────────────────────────────────────────────────
    delete_demo_pinecone()
    deleted_blobs = delete_demo_blobs()
    print(f"\n  Cleanup done — {len(deleted_blobs)} blob(s) removed")

    # ── Phase 2: Download exhibits ──────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  Downloading EDGAR EX-10.* material contracts …")
    print("─" * 50)
    local_files = download_exhibits()

    if not local_files:
        print("\n  ERROR: no exhibit files downloaded — aborting ingestion")
        print_final_state()
        return

    print(f"\n  {len(local_files)} file(s) downloaded")

    # ── Phase 3 & 4: Upload + Ingest ────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  Uploading to Azure Blob + ingesting …")
    print("─" * 50)

    ingest_results = {}
    for local_path in local_files:
        print(f"\n  {local_path.name}")
        try:
            blob_name = upload_to_blob(local_path)
            print(f"    blob  : {blob_name}")
            t0 = time.time()
            r  = ingest_blob(blob_name, local_path)
            elapsed = time.time() - t0
            if r.get("error"):
                print(f"    ERROR : {r['error']}")
            else:
                print(f"    ✓  {r['char_count']:,} chars  |  {r['vectors']} vectors  |  {elapsed:.1f}s")
            ingest_results[local_path.name] = r
        except Exception as exc:
            print(f"    FAILED: {exc}")
            ingest_results[local_path.name] = {"error": str(exc), "vectors": 0}

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 64)
    print("  Ingestion Summary — legal-docs")
    print("=" * 64)
    print(f"  {'File':<45} {'Vectors':>8}  Status")
    print(f"  {'─'*43:<45} {'─'*6:>8}  {'─'*8}")
    total = 0
    for fname, r in ingest_results.items():
        short = fname[:43]
        if r.get("error"):
            print(f"  {short:<45} {'—':>8}  ERROR")
        else:
            total += r.get("vectors", 0)
            print(f"  {short:<45} {r['vectors']:>8}  OK")
    print(f"\n  Vectors upserted (legal-docs) : {total:,}")

    # ── Phase 5: Cross-collection test ──────────────────────────────────────
    run_cross_collection_test()

    # ── Phase 6: Final state ─────────────────────────────────────────────────
    print_final_state()


if __name__ == "__main__":
    main()
