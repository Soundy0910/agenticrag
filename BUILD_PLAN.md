# BUILD_PLAN.md — Operational Build Checklist

> **Purpose:** The file-by-file build sequence Claude Code executes against. Read `ARCHITECTURE.md` for the *why* behind any decision. This doc is the *what* and *in what order*. Core principle: **build in slices that RUN, hit a deployable milestone early, layer ambition on top of a working system.** Never be one unfinished feature away from having nothing to demo.

---

## Prerequisites (set up before/while building)

**Local env (before File 1):**
- Python 3.11+ (`python --version`), Node.js 18+ (`node --version`)
- Project folder, `git init`, Python venv/conda
- `pip install --break-system-packages` not needed locally — use the venv

**Git identity (BEFORE first commit — critical):**
- Repo under **personal GitHub account**, not work account
- Per-repo config (NOT global): `git config user.email "personal@email.com"` + `git config user.name "..."`
- Verify with `git config user.email` before commit #1 (work-email commits are permanent in history)
- Ensure Cursor/Claude Code pushes with personal GitHub credentials, not cached work token

**Accounts (set up just-in-time per the file that needs them):**
| Service | Needed by | Notes |
|---------|-----------|-------|
| OpenAI | embed/LLM files | add payment method; ~$5 covers dev |
| Pinecone | indexing file | free Starter tier sufficient |
| Cohere | rerank file | free trial tier |
| Azure | Blob connector | **personal** account, separate from AmplifAI; label clearly in Storage Explorer to avoid touching company storage |
| Neo4j | GraphRAG | free local Docker or AuraDB free |
| LangSmith | tracing (optional) | free tier |

---

## Repo structure (target)

```
agentic-rag/
├── README.md                  # arch diagram + setup + eval results
├── ARCHITECTURE.md            # design + rationale (companion doc)
├── BUILD_PLAN.md              # this file
├── docker-compose.yml
├── backend/
│   ├── requirements.txt
│   ├── .env.example           # keys (never commit real .env)
│   ├── config.py              # swappable models/keys per collection
│   ├── main.py                # FastAPI entry
│   ├── storage/
│   │   ├── base.py            # DocumentSource interface + DocumentMetadata
│   │   ├── local.py           # local folder source (test first)
│   │   ├── azure_blob.py      # real connector
│   │   └── stubs.py           # Slack/Monday interface stubs
│   ├── ingest/
│   │   ├── parse.py           # multi-format, swappable (unstructured default)
│   │   ├── chunk.py           # semantic + parent-document
│   │   ├── embed_index.py     # embed + upsert with metadata
│   │   ├── pipeline.py        # event-driven orchestration
│   │   └── reindex.py         # re-embed migration (deprecation insurance)
│   ├── retrieval/
│   │   ├── hybrid.py          # vector + BM25
│   │   └── rerank.py          # Cohere
│   ├── agent/
│   │   ├── state.py           # LangGraph state (turns + retrieved chunks)
│   │   ├── nodes.py           # rewrite, route, retrieve, grade, generate
│   │   └── graph.py           # wires nodes into the state graph
│   ├── graph_rag/
│   │   ├── schema.py          # pluggable entity schema (finance built)
│   │   ├── extract.py         # schema-driven entity/relation extraction
│   │   └── query.py           # graph retrieval
│   ├── api/
│   │   ├── query.py           # POST /query (streaming, emits node trace)
│   │   ├── documents.py       # upload/list/delete
│   │   └── ingest.py          # trigger/status
│   └── eval/
│       ├── testset.py         # Q+A pairs + routing test cases
│       └── run_ragas.py
└── frontend/
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── App.jsx
        ├── api/client.js
        └── components/
            ├── DocumentLibrary.jsx
            ├── UploadDropzone.jsx
            ├── ChatPanel.jsx
            ├── SourceCitations.jsx
            └── LiveTrace.jsx   # the signature transparency panel
```

---

## Build order (milestone-driven)

### Week 1 — Ingestion core (runs via scripts, no UI yet)
1. **`storage/base.py`** — `DocumentMetadata` dataclass (fields: `doc_id`, `source_type`, `collection`, `filename`, `file_type`, `last_modified`, `access_scope`, `embedding_model`, `extra`) + abstract `DocumentSource` (`list_documents`, `fetch_document`, `watch_for_changes`). Pure Python, no deps.
2. **`storage/local.py`** — local folder implementation. Test the interface end-to-end with zero cloud cost.
3. **`storage/azure_blob.py`** — real Azure Blob connector (personal account). Set up Azure Storage + Event Grid here.
4. **`ingest/parse.py`** — multi-format parser, `unstructured` default, swappable, routes by file type. Retain source text.
5. **`ingest/chunk.py`** — semantic + parent-document chunking (child ~300 / parent ~1800 tok, ~12% overlap).
6. **`config.py`** + **`ingest/embed_index.py`** — embedding (`text-embedding-3-small` default, per-collection config) + Pinecone upsert with metadata + namespace.
   - **MILESTONE: drop a PDF in Blob → it's chunked, embedded, searchable in Pinecone.** Verify by querying Pinecone directly.

### Week 2 — Agent + API (backend fully demoable)
7. **`retrieval/hybrid.py`** — vector + BM25, combine results.
8. **`retrieval/rerank.py`** — Cohere rerank top candidates.
   - **MILESTONE: retrieval visibly better than vector-only.**
9. **`agent/state.py`** — LangGraph state (recent turns + retrieved chunks; never answers).
10. **`agent/nodes.py`** — rewrite, router, retrieve, grade, generate nodes.
11. **`agent/graph.py`** — wire the state graph with conditional edges (route + grade/retry loop).
12. **`graph_rag/`** — finance schema, schema-driven extraction, Neo4j graph query; agent routes relational Qs here.
13. **`api/query.py`, `documents.py`, `ingest.py`** — FastAPI endpoints; `/query` streams node-level trace events (SSE/WebSocket).
    - **MILESTONE: curl `/query` → cited answer. Backend complete and demoable even before React.**

### Week 3 — UI + evals + deploy
14. **`eval/testset.py` + `run_ragas.py`** — RAGAS scoring + routing eval. Record numbers for README.
15. **`frontend/`** — React: DocumentLibrary, UploadDropzone, ChatPanel, SourceCitations, **LiveTrace** (the signature panel; consumes the streamed node trace; clean/debug toggle; raw-prompt expanders).
16. **Deploy** — backend containerized; frontend built; deploy (free tier: Railway/Render/HF Spaces, or Azure). Azure Blob can stay personal-account.
    - **MILESTONE: live NotebookLM-style UI + eval numbers in README.**

### Post-MVP (only after deployed & applying)
- Slack/Monday real connectors (interface already stubbed)
- OKF curated-concept bundle for finance definitions
- Streaming token output, cost dashboard, prompt-injection guardrails

---

## Build discipline (keep the understanding)
- Build one slice, **run it, verify the milestone, then move on.** Don't batch unverified code.
- After each meaningful file, **read it and confirm you can explain WHY it works** — the goal is interview-ready understanding, not just a working repo.
- Max ~2 logical changes per iteration; verify before the next.
- Keep `source_text` on every chunk (deprecation insurance + citations).
- Every model/parser/embedding choice stays a **config value behind an interface** — never hard-coded.
- Commit at each milestone with the personal git identity.

---

## Definition of done (for the demo)
- [ ] Drop any supported file into a collection → auto-ingested with metadata
- [ ] Multi-collection: legal question doesn't pull finance chunks
- [ ] Agent routes: factual→vector, relational→graph, small-set→CAG
- [ ] Comparison question decomposes + retrieves both facts fresh + cites both
- [ ] Live trace shows input→transform→output per stage, with raw-prompt expanders
- [ ] RAGAS numbers (faithfulness etc.) in README
- [ ] Deployed, shareable URL
- [ ] You can explain every component and every "why this not the cheaper alt" in an interview

---

## Progress Log (last updated: 2026-06-30)

> **For a fresh session with no memory of prior work:** read this section first. It tells you exactly what is done, what gaps exist, and where to start next.

### Files built and status

| # | File / Module | Status | Notes |
|---|---|---|---|
| 1 | `storage/base.py` | ✅ done | DocumentMetadata dataclass + DocumentSource abstract interface |
| 2 | `storage/local.py` | ✅ done | Local folder connector |
| 3 | `storage/azure_blob.py` | ✅ done | Real Azure Blob connector (personal account) |
| 4 | `ingest/parse.py` | ✅ done | Multi-format parser (unstructured) |
| 5 | `ingest/chunk.py` | ✅ done | Semantic + parent-document chunking; child ~300 tok, parent ~1800 tok |
| 6 | `config.py` + `ingest/embed_index.py` | ✅ done | text-embedding-3-small, Pinecone upsert with metadata; COLLECTION_REGISTRY added |
| 7 | `retrieval/hybrid.py` | ✅ done | Vector + BM25 hybrid search; BM25 None-sentinel fix for empty namespaces |
| 8 | `retrieval/rerank.py` | ✅ done | Cohere rerank |
| 9 | `agent/state.py` | ✅ done | LangGraph AgentState; active_collections field added |
| 10 | `agent/nodes.py` | ✅ done | All nodes: rewrite, router (_classify_collections + collection='auto'), retrieve_vector (parallel multi-collection), retrieve_cag, retrieve_graph, grade, generate |
| 11 | `agent/graph.py` | ✅ done | StateGraph compiled; active_collections in initial state |
| 12 | `graph_rag/` | ✅ done | Finance entity schema, extraction, Neo4j query |
| 13 | `api/query.py` | ✅ done | POST /query SSE stream; 'done' event includes contexts[:500] for inline eval |
| 13b | `api/documents.py` + `api/ingest.py` | ✅ done | Upload/list/delete + ingest job endpoints |
| 13c | `api/eval.py` | ✅ done | POST /api/eval — referenceless faithfulness + answer_relevancy; lazy RAGAS init; runs in thread pool |
| 14 | `eval/testset.py` + `eval/run_ragas.py` | ✅ done | RAGAS 0.4.3 with old-style metric singletons + LangchainLLMWrapper/EmbeddingsWrapper |
| 15 | `frontend/` | ✅ done | React 3-panel UI: DocumentLibrary, ChatPanel, LiveTrace, SourceCitations, UploadDropzone |

**Extra work done (not in original plan):**
- Cross-collection routing (`collection='auto'`, `_classify_collections`, `active_collections`, parallel hybrid search) — fully wired end-to-end
- Frontend visual polish: `text-2xs`/`text-3xs` Tailwind tokens, `surface-825`, `letter-spacing: -0.011em`, `font-mono` on all data values in LiveTrace + SourceCitations
- Inline RAGAS eval button per chat message (FlaskConical button → ScoreChip badges)
- ARCHITECTURE.md §8.1 compliance-crawler extension documented

### Known issues / quality gaps

**CAG context_precision = 0.291 (low — expected but fixable)**
CAG (`retrieve_cag_node`) fetches ALL chunks from the collection and stuffs them into context without filtering to the most relevant ones. RAGAS context_precision penalises this heavily because the top-ranked chunks are not ranked by relevance to the question — they're ordered by document structure. Fix: add a relevance-ranking step inside `retrieve_cag_node` (embed the query, cosine-rank the CAG chunks, pass only top-K to generate). This is not a correctness bug — the answer is still grounded — but it tanks the precision metric.

**R03 routing test miss (expected)**
R03 tests finance routing but the `finance` namespace was empty during eval. The agent fell back to `demo` collection. This is correct behaviour for an empty collection, not a bug. Will pass once finance documents are indexed.

**Legal collection is a stub**
`legal` collection has no indexed documents. The compliance-crawler extension (ARCHITECTURE.md §8.1) documents how to fill it.

**Inline eval context quality**
The `/api/eval` endpoint receives citation `source_text[:500]` as contexts. These are truncated to 500 chars. Faithfulness scoring works but longer contexts would be more accurate. The batch eval (`run_ragas.py`) uses full source text and is more reliable for regression testing.

**langgraph 1.2.7 + Python 3.13 — package conflicts**
`langchain` 0.3.26 and `langchain-community` 0.3.27 show pip resolver warnings (require langchain-core<0.4). Neither is directly imported in this codebase. Safe to ignore until LangChain publishes a 3.x-compatible release.

### RAGAS baseline scores (2026-06 eval run, demo collection)
| Metric | Score | Target |
|---|---|---|
| faithfulness | 0.800 | >0.85 |
| answer_relevancy | 0.718 | >0.80 |
| context_precision | 0.291 | >0.70 — see CAG gap above |
| context_recall | 0.733 | >0.70 ✅ |

### What the next session should start with

**Priority 1 — CAG context_precision fix:**
In `backend/agent/nodes.py`, `retrieve_cag_node()` fetches all chunk IDs and metadata, then passes them to `_chunks_to_context` unranked. Add an embedding-based relevance sort: embed `rewritten_query`, cosine-rank the CAG chunks, pass only the top 12 to the generate node. This should push context_precision from 0.291 → ~0.60+.

**Priority 2 — Deploy:**
Backend is containerisable (FastAPI + uvicorn). Frontend is Vite-built. Target: Railway or Render for backend, Vercel for frontend. Backend needs `PORT` env var and `CORS` origin set to the Vercel URL.

**Priority 3 — Finance documents:**
Index real SEC 10-K filings into the `finance` collection to make the cross-collection routing demo work end-to-end. The ingest pipeline is ready — just need documents.

**Priority 4 — README with eval numbers and architecture diagram:**
Add RAGAS scores table and a system diagram (can be ASCII or Mermaid) to README.md.
