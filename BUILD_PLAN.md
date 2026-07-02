# BUILD_PLAN.md — Setup & Development Guide

> Read `ARCHITECTURE.md` for design decisions and rationale. This document covers environment setup, how to run the system locally, how to re-ingest the corpus, and what's left to do.

---

## Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11+ | 3.13 tested |
| Node.js | 18+ | for the React frontend |
| Pinecone account | free Starter tier | 1 index, 100 namespaces, 2 GB |
| OpenAI account | any tier | `text-embedding-3-small` + `gpt-4o-mini` |
| Cohere account | free trial | reranking only |
| Neo4j AuraDB | free tier | or local via Docker |
| Azure Storage | optional | local folder connector works without it |

---

## Environment setup

```bash
# Clone
git clone https://github.com/<you>/agentic-rag.git
cd agentic-rag

# Python virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..

# Environment variables
cp .env.example backend/storage/.env
# Edit backend/storage/.env and fill in your keys
```

---

## Running the app

```bash
# Backend — from the project root
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend — separate terminal
cd frontend && npm run dev
```

- API: `http://localhost:8000` (Swagger docs at `/docs`)
- UI: `http://localhost:5173`

---

## Corpus ingestion (first time or re-ingest)

The `data/` directory is gitignored — place raw HTM files there before ingesting.

```bash
# Ingest SEC 10-K filings into Pinecone (sec-filings namespace)
python scripts/ingest_sec_filings.py

# Ingest EX-10 legal exhibits into Pinecone (legal-docs namespace)
python scripts/ingest_legal_docs.py

# (Re)build Neo4j knowledge graph from the same filings
python scripts/extract_graph_rag.py

# Reset the graph and rebuild from scratch
python backend/scripts/reset_neo4j.py
python scripts/extract_graph_rag.py

# Check graph health
python backend/scripts/graph_health_report.py
```

---

## Evaluation

```bash
# 5-question RAGAS eval (faithfulness, answer_relevancy, context_precision, context_recall)
python -m backend.eval.run_5q_ragas

# Full 10-question end-to-end API validation (requires running server)
python scripts/validate_iterations.py
```

See `VALIDATION_REPORT.md` for the latest run results.

---

## Project status

### What's built

| Component | Status | Notes |
|---|---|---|
| Storage connectors | ✅ | Local folder + Azure Blob behind `DocumentSource` interface |
| Ingestion pipeline | ✅ | parse → chunk (semantic + parent-doc) → embed → Pinecone |
| Section detection | ✅ | 10-K section type classifier for targeted retrieval |
| Hybrid retrieval | ✅ | BM25 + vector, merged and Cohere-reranked |
| LangGraph agent | ✅ | 7 nodes: rewrite, classify, router, access_check, retrieve, grade, generate |
| Multi-collection routing | ✅ | Keyword + LLM classifier, role-aware RBAC filter |
| GraphRAG | ✅ | Neo4j — Company/Filing/Metric/RiskFactor schema; 4 Cypher query types |
| Decompose node | ✅ | Splits multi-collection questions into per-namespace sub-questions |
| Numeric validation | ✅ | Deterministic Python arithmetic for calculation queries |
| FastAPI backend | ✅ | SSE streaming, RBAC, document upload/delete, inline eval endpoint |
| React UI | ✅ | DocumentLibrary, ChatPanel, LiveTrace, SourceCitations, UploadDropzone |
| RAGAS eval harness | ✅ | Batch eval + 5-question focused suite |

### Corpus state

| Collection | Vectors | Documents |
|---|---|---|
| `sec-filings` | 23,694 | 15 × 10-K HTMs — AAPL MSFT NVDA GOOGL AMZN TSLA META JPM JNJ V WMT XOM PFE KO DIS |
| `legal-docs` | 1,571 | 10 × EX-10 exhibits — JPM MSFT TSLA META WMT |

### Neo4j graph state

- Schema v2: Company, Filing, BusinessSegment, Metric, RiskFactor, Topic, Chunk
- All 15 tickers ingested; every Metric and RiskFactor links `SUPPORTED_BY` a source Chunk
- Query types: `lookup`, `comparison`, `segment`, `risk`

---

## What's next

| Priority | Item |
|---|---|
| 1 | **Deploy** — Backend → Azure Container Apps or Railway; Frontend → Vercel/Azure Static Web Apps |
| 2 | **README with live URL** — add RAGAS scores table and architecture diagram once deployed |
| 3 | **NVDA + KO graph fix** — scan multiple extraction windows per large filing; merge results |
| 4 | **PM app connectors** — Slack/Monday stubs exist in `storage/`; wire real webhook listeners |

---

## Known issues

| Issue | Impact | Workaround |
|---|---|---|
| NVDA + KO graph extraction weak | Falls back to vector search; answer still correct | Re-run extraction with wider window scan |
| Pinecone cold-start latency | First request after server start can take 30–60 s | Server warms up after the first successful query |
| `langchain-community` pip resolver warning | None — not directly imported | Safe to ignore |
