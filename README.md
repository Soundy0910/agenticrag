# Agentic RAG Knowledge Platform

A production-style **agentic retrieval-augmented generation** system built on LangGraph, FastAPI, and React. SEC 10-K filings and legal exhibits are the demo corpus — chosen because financial documents are a hard, table-heavy, regulated stress test — but the ingestion pipeline, retrieval layer, and agent are domain-agnostic.

---

## What it demonstrates

| Capability | Implementation |
|---|---|
| Agentic orchestration | LangGraph state machine — 7 nodes, conditional edges, self-correction loop |
| Hybrid retrieval | BM25 + vector search merged, Cohere reranked |
| GraphRAG | Neo4j knowledge graph (Company → Filing → Metric/RiskFactor → Chunk) |
| Multi-collection routing | Keyword + LLM classifier routes to sec-filings / legal-docs / both |
| RBAC | Role-scoped collection access checked before any retrieval |
| Streaming UI | FastAPI SSE → React live trace — shows every node's input/output in real time |
| RAGAS evaluation | Faithfulness, answer relevancy, context precision/recall — scored and tracked |

---

## Architecture

```
React UI  (DocumentLibrary · ChatPanel · LiveTrace · UploadDropzone)
    │  SSE stream (node-level trace events)
    ▼
FastAPI  /api/query · /api/documents · /api/ingest · /api/eval
    │
LangGraph Agent
    rewrite → classify → router → access_check → [decompose]
        → retrieve (vector | cag | graph) → grade → [validate_numbers] → generate
    │
┌───────────────────────────────────────────────────────┐
│  Retrieval layer                                       │
│  Pinecone (hybrid: BM25 + text-embedding-3-small)     │
│  Cohere Rerank                                        │
│  Neo4j  (GraphRAG — Company/Metric/RiskFactor graph)  │
└───────────────────────────────────────────────────────┘
    │
Ingestion pipeline
    parse (unstructured) → chunk (semantic + parent-doc) → embed → Pinecone
    Storage connectors: Azure Blob · Local folder
```

---

## Corpus (demo dataset)

| Collection | Vectors | Documents |
|---|---|---|
| `sec-filings` | 23,694 | 15 × 10-K HTMs — AAPL MSFT NVDA GOOGL AMZN TSLA META JPM JNJ V WMT XOM PFE KO DIS |
| `legal-docs` | 1,571 | 10 × EX-10 exhibits — JPM MSFT TSLA META WMT |

---

## RAGAS Evaluation (iter 7)

| Question type | Faithfulness | Answer Rel. | Ctx Precision | Ctx Recall |
|---|---|---|---|---|
| MSFT revenue FY2025 | 1.000 | 1.000 | 0.667 | 1.000 |
| JPM revenue + recoupment (multi-collection) | 1.000 | 0.998 | 0.367 | 0.667 |
| Apple vs MSFT comparison (GraphRAG) | — | — | — | — |
| Walmart net sales + risk factors | — | — | — | — |
| Tesla clawback (legal-docs) | 0.800 | 0.989 | 0.261 | 1.000 |
| **Aggregate** | **0.627** | **0.781** | **0.342** | **0.800** |

---

## Tech stack

- **Agent:** LangGraph 0.5, LangChain 0.3
- **LLM / Embeddings:** OpenAI gpt-4o-mini / text-embedding-3-small
- **Vector DB:** Pinecone (namespaced collections)
- **Reranker:** Cohere Rerank v3
- **Graph DB:** Neo4j AuraDB
- **Backend:** FastAPI + uvicorn (async, stateless)
- **Frontend:** React 18 + Vite + Tailwind CSS
- **Eval:** RAGAS 0.4

---

## Project structure

```
├── backend/
│   ├── main.py                  # FastAPI app entry point
│   ├── config.py                # model config, collection registry, RBAC roles
│   ├── requirements.txt
│   ├── agent/
│   │   ├── graph.py             # LangGraph StateGraph wiring
│   │   ├── nodes.py             # all 7 pipeline nodes
│   │   └── state.py             # AgentState schema
│   ├── api/
│   │   ├── query.py             # POST /api/query — SSE streaming
│   │   ├── documents.py         # upload / list / delete
│   │   ├── ingest.py            # trigger / status
│   │   └── eval.py              # POST /api/eval — inline RAGAS scoring
│   ├── ingest/
│   │   ├── parse.py             # multi-format parser (unstructured)
│   │   ├── chunk.py             # semantic + parent-document chunking
│   │   ├── embed_index.py       # embed + Pinecone upsert with metadata
│   │   └── section_detect.py    # 10-K section type classifier
│   ├── retrieval/
│   │   ├── hybrid.py            # BM25 + vector search
│   │   └── rerank.py            # Cohere reranking
│   ├── graph_rag/
│   │   ├── schema.py            # pluggable entity schema (finance built)
│   │   ├── extract.py           # LLM-driven entity/relation extraction → Neo4j
│   │   └── query.py             # 4 Cypher query types (lookup/comparison/segment/risk)
│   ├── eval/
│   │   ├── testset.py           # Q+A test pairs
│   │   ├── run_ragas.py         # RAGAS batch eval runner
│   │   └── run_5q_ragas.py      # 5-question focused eval
│   ├── scripts/
│   │   ├── reset_neo4j.py       # wipe and recreate graph constraints
│   │   └── graph_health_report.py
│   └── storage/
│       ├── base.py              # DocumentSource interface
│       ├── local.py             # local folder connector
│       └── azure_blob.py        # Azure Blob connector
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── api/client.js
│       └── components/
│           ├── ChatPanel.jsx
│           ├── DocumentLibrary.jsx
│           ├── LiveTrace.jsx        # streaming node-level trace UI
│           ├── SourceCitations.jsx
│           └── UploadDropzone.jsx
├── scripts/                     # operational one-off scripts
│   ├── extract_graph_rag.py     # ingest 10-K HTMs into Neo4j
│   ├── ingest_sec_filings.py    # ingest filings into Pinecone
│   ├── ingest_legal_docs.py     # ingest legal exhibits into Pinecone
│   └── validate_iterations.py  # 10-question end-to-end validation harness
├── ARCHITECTURE.md              # design decisions and rationale
├── VALIDATION_REPORT.md         # latest 10-iteration quality report
└── .env.example
```

---

## Local setup

### 1. Clone and install

```bash
git clone https://github.com/<you>/agentic-rag.git
cd agentic-rag

# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install
```

### 2. Configure environment

```bash
cp .env.example backend/storage/.env
# fill in OPENAI_API_KEY, PINECONE_API_KEY, COHERE_API_KEY, NEO4J_* credentials
```

### 3. Ingest corpus (first time only)

```bash
# Embed SEC filings → Pinecone
python scripts/ingest_sec_filings.py

# Embed legal exhibits → Pinecone
python scripts/ingest_legal_docs.py

# Build Neo4j knowledge graph
python scripts/extract_graph_rag.py
```

### 4. Run

```bash
# Backend (from project root)
uvicorn backend.main:app --reload

# Frontend (separate terminal)
cd frontend && npm run dev
```

Open `http://localhost:5173`.

---

## Agent pipeline

```
User question
    │
    ▼ rewrite       — resolve follow-ups and relative time references
    │
    ▼ classify      — factual_lookup / comparison / risk_analysis / calculation / out_of_scope
    │
    ▼ router        — pick vector | cag | graph path; classify active collections
    │
    ▼ access_check  — RBAC: deny if role can't access the routed collection
    │
    ▼ decompose*    — multi-collection: split question into per-namespace sub-questions
    │
    ▼ retrieve      — hybrid search (BM25 + semantic) + Cohere rerank
    │                 or context-stuffing (CAG) for small collections
    │                 or Neo4j graph + vector hybrid
    │
    ▼ grade         — is retrieved context sufficient? retry with reformulated query if not
    │
    ▼ validate_numbers*  — deterministic arithmetic for calculation queries
    │
    ▼ generate      — grounded answer with citations, format driven by query type
```

`*` conditional nodes — only run when needed.

---

## Eval

```bash
# Run the 5-question RAGAS eval
python -m backend.eval.run_5q_ragas

# Run the 10-question end-to-end API validation
python scripts/validate_iterations.py
```
