"""
scripts/backfill_section_types.py

Backfill section_type metadata onto existing Pinecone vectors WITHOUT re-embedding.

WHY THIS IS NEEDED:
  The 25,265 vectors already in Pinecone were upserted before section_type was added
  to the ingest pipeline. New ingestions (embed_index.py) now write section_type
  automatically, but existing vectors have no section_type field in their metadata.

  hybrid_search() section_filter relies on the Pinecone metadata field
  {"section_type": {"$eq": "income_statement"}} — if the field is absent, the filter
  returns zero results and retrieval falls back to unfiltered search.

HOW IT WORKS:
  1. Fetch all vector IDs in each namespace (list()).
  2. Batch-fetch vectors WITH their stored values (include_values=True).
  3. Detect section_type from each vector's source_text metadata field.
  4. Re-upsert the same vector (same ID, same values) with updated metadata.

  This is an in-place metadata update — no embedding API calls, no download/upload
  of document text, no changes to vector geometry. Only metadata is updated.

COST:
  ~25k Pinecone reads + ~25k Pinecone writes. At Pinecone free tier rates this
  is negligible. Runtime: ~3-5 minutes depending on network latency.

Run from project root:
  python3 scripts/backfill_section_types.py [--dry-run] [--collection sec-filings]

Options:
  --dry-run        Print what would be updated without writing to Pinecone
  --collection X   Only process this namespace (default: all namespaces)
  --batch-size N   Pinecone upsert batch size (default: 100)
"""

import argparse
import logging
import pathlib
import sys
import time
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

import backend.config as cfg
from backend.ingest.embed_index import _get_pinecone
from backend.ingest.section_detect import detect_section_type, section_type_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# How many IDs to fetch/upsert per Pinecone API call
_FETCH_BATCH = 200
_UPSERT_BATCH = 100


def _list_all_ids(index, namespace: str) -> list[str]:
    """List every vector ID in a namespace."""
    all_ids: list[str] = []
    for page in index.list(namespace=namespace):
        all_ids.extend(item.id for item in page.vectors)
    return all_ids


def _process_namespace(
    index,
    namespace: str,
    dry_run: bool,
    batch_size: int,
) -> dict[str, int]:
    """
    Fetch, classify, and re-upsert all vectors in one Pinecone namespace.

    Returns a dict of section_type → count for reporting.
    """
    logger.info("namespace=%r: listing IDs ...", namespace)
    all_ids = _list_all_ids(index, namespace)
    logger.info("namespace=%r: %d total vectors", namespace, len(all_ids))

    if not all_ids:
        return {}

    counts: Counter = Counter()
    upserted = 0
    skipped = 0

    for batch_start in range(0, len(all_ids), _FETCH_BATCH):
        batch_ids = all_ids[batch_start : batch_start + _FETCH_BATCH]

        # Fetch WITH vector values so we can re-upsert the same vectors
        response = index.fetch(ids=batch_ids, namespace=namespace)

        upsert_batch: list[dict] = []
        for vid, vec in response.vectors.items():
            meta = dict(vec.metadata or {})

            # Already has section_type — skip to avoid unnecessary writes
            if meta.get("section_type") and meta["section_type"] != "general":
                # Re-check in case detection improved, but only update if different
                source_text = meta.get("source_text", "")
                new_type = detect_section_type(
                    source_text, namespace, meta.get("filename", "")
                )
                if new_type == meta["section_type"]:
                    skipped += 1
                    counts[meta["section_type"]] += 1
                    continue

            source_text = meta.get("source_text", "")
            filename = meta.get("filename", "")
            section_type = detect_section_type(source_text, namespace, filename)

            counts[section_type] += 1
            meta["section_type"] = section_type

            upsert_batch.append({
                "id": vid,
                "values": list(vec.values) if vec.values else [],
                "metadata": meta,
            })

        if upsert_batch and not dry_run:
            for i in range(0, len(upsert_batch), batch_size):
                index.upsert(vectors=upsert_batch[i : i + batch_size], namespace=namespace)
                upserted += len(upsert_batch[i : i + batch_size])
        elif upsert_batch and dry_run:
            upserted += len(upsert_batch)

        done = min(batch_start + _FETCH_BATCH, len(all_ids))
        logger.info(
            "namespace=%r: %d/%d processed  (upserted=%d skipped=%d)",
            namespace, done, len(all_ids), upserted, skipped,
        )

    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill section_type on Pinecone vectors")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing")
    parser.add_argument("--collection", default=None, help="Only process this namespace")
    parser.add_argument("--batch-size", type=int, default=_UPSERT_BATCH)
    args = parser.parse_args()

    if args.dry_run:
        logger.info("DRY RUN — no writes will be made to Pinecone")

    pc = _get_pinecone()
    index = pc.Index(cfg.PINECONE_INDEX_NAME)

    # Determine which namespaces to process
    stats = index.describe_index_stats()
    ns_stats = stats.namespaces or {}
    available = list(ns_stats.keys())
    logger.info("Available namespaces: %s", available)

    if args.collection:
        if args.collection not in available:
            logger.error("Namespace %r not found in index. Available: %s", args.collection, available)
            sys.exit(1)
        namespaces = [args.collection]
    else:
        namespaces = [ns for ns in available if ns in cfg.COLLECTION_REGISTRY]

    sep = "─" * 70
    print(f"\n{'='*70}")
    print(f"  Backfill section_type — {cfg.PINECONE_INDEX_NAME}")
    print(f"{'='*70}")
    print(f"  Namespaces: {namespaces}")
    print(f"  Dry run:    {args.dry_run}")
    print(f"  Batch size: {args.batch_size}")
    print(sep)

    t_start = time.time()
    grand_total: Counter = Counter()

    for ns in namespaces:
        print(f"\n  Processing namespace: {ns}")
        ns_counts = _process_namespace(index, ns, args.dry_run, args.batch_size)
        grand_total.update(ns_counts)

        print(f"\n  {ns} section_type distribution:")
        for stype, count in sorted(ns_counts.items(), key=lambda x: -x[1]):
            label = section_type_label(stype)
            bar = "█" * min(30, count // max(1, sum(ns_counts.values()) // 30))
            print(f"    {label:30s} {count:6d}  {bar}")

    elapsed = time.time() - t_start
    print(f"\n{sep}")
    print(f"  Grand total ({elapsed:.1f}s):")
    for stype, count in sorted(grand_total.items(), key=lambda x: -x[1]):
        label = section_type_label(stype)
        print(f"    {label:30s} {count:6d}")
    print(f"  Total vectors processed: {sum(grand_total.values())}")

    if args.dry_run:
        print("\n  DRY RUN complete — no changes written to Pinecone.")
        print("  Re-run without --dry-run to apply.")
    else:
        print("\n  Backfill complete.")
        print("  BM25 cache will auto-rebuild on next query (section_type now in metadata).")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
