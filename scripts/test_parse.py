"""
scripts/test_parse.py

Smoke test: fetch the 3 real files from Azure Blob and run each through the
parser. Prints the first 200 chars of extracted text per file.

Run from the project root:  python3 scripts/test_parse.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.storage.azure_blob import AzureBlobSource
from backend.ingest.parse import parse

src = AzureBlobSource(collection="demo")
docs = list(src.list_documents())

print(f"Found {len(docs)} document(s) in container\n")
print("=" * 60)

for meta in docs:
    print(f"\nFile    : {meta.doc_id}")
    print(f"Type    : {meta.file_type}")

    content = src.fetch_document(meta.doc_id)
    result = parse(content, meta)

    if not result.ok:
        print(f"FAILED  : {result.error}")
    else:
        preview = result.text[:200].replace("\n", " ")
        print(f"Chars   : {len(result.text):,}")
        print(f"Elements: {len(result.structure)}")
        print(f"Preview : {preview!r}")
    print("-" * 60)
