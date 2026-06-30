"""
scripts/test_chunk.py

Smoke test for chunk.py — fetches the PDF resume from Azure, parses it,
chunks it, and prints the parent/child structure.

Run from the project root:  python3 scripts/test_chunk.py
"""

import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(pathlib.Path(__file__).parent.parent / "backend" / "storage" / ".env")

from backend.storage.azure_blob import AzureBlobSource
from backend.ingest.parse import parse
from backend.ingest.chunk import chunk_document, ChunkConfig

src = AzureBlobSource(collection="demo")
docs = {m.doc_id: m for m in src.list_documents()}

meta = docs["Soundariyan_Venkatachalam_Resume.pdf"]
content = src.fetch_document(meta.doc_id)
parsed = parse(content, meta)

print(f"Parsed text: {len(parsed.text):,} chars\n")

cfg = ChunkConfig(child_tokens=350, parent_tokens=1750, overlap_pct=0.12)
chunks = chunk_document(parsed, meta, config=cfg)

parents  = [c for c in chunks if c.is_parent]
children = [c for c in chunks if not c.is_parent]

print(f"Parents : {len(parents)}")
print(f"Children: {len(children)}")
print()

print("=== PARENTS (text preview, 120 chars each) ===")
for p in parents:
    print(f"  [{p.chunk_id}]  {len(p.source_text)} chars")
    print(f"    {p.source_text[:120].replace(chr(10), ' ')!r}")
print()

print("=== FIRST CHILD ===")
child = children[0]
print(f"  chunk_id : {child.chunk_id}")
print(f"  parent_id: {child.parent_id}")
print(f"  chars    : {len(child.source_text)}")
print(f"  text     : {child.source_text[:300].replace(chr(10), ' ')!r}")
print()

print("=== SECOND CHILD (overlap check) ===")
if len(children) > 1:
    child2 = children[1]
    print(f"  chunk_id : {child2.chunk_id}")
    print(f"  parent_id: {child2.parent_id}")
    overlap_end   = child.source_text[-80:]
    overlap_start = child2.source_text[:80]
    print(f"  tail of child 0 : {overlap_end.replace(chr(10),' ')!r}")
    print(f"  head of child 1 : {overlap_start.replace(chr(10),' ')!r}")
