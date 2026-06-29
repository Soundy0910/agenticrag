"""
scripts/test_azure_blob.py

Manual smoke test for AzureBlobSource.
Run from the project root:  python3 scripts/test_azure_blob.py
"""

import pathlib
import sys

# Ensure project root is on the path so backend.* imports resolve
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Load .env from its actual location before importing AzureBlobSource
from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.storage.azure_blob import AzureBlobSource

src = AzureBlobSource(collection="demo")

print("=== list_documents() ===")
docs = list(src.list_documents())
for d in docs:
    print(f"  {d.doc_id}")
    print(f"    file_type={d.file_type}  modified={d.last_modified}  source={d.source_type}")

print(f"\n{len(docs)} supported document(s) found in container\n")

if docs:
    first = docs[0]
    print(f"=== fetch_document('{first.doc_id}') ===")
    content = src.fetch_document(first.doc_id)
    print(f"  {len(content):,} bytes  starts with {content[:24]!r}")
else:
    print("No supported documents found — upload a .pdf/.txt/.docx to the container and re-run.")
