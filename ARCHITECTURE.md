# ARCHITECTURE.md — Agentic RAG Knowledge Platform

> **Purpose of this document:** This is the design + reasoning reference for the project. It explains *what* we're building and *why* each decision was made. Read this to understand the system and to explain it in interviews. The companion `BUILD_PLAN.md` is the operational, file-by-file build checklist.

---

## 1. What this project is

A **domain-agnostic, dynamic, agentic RAG platform**. SEC filings (10-Ks) are the **demo dataset**, but the system is built to ingest documents on *any* topic. Documents arrive dynamically (new files are ingested automatically), are stored with rich metadata, and are queried by an agent that decides *how* to retrieve per question.

**One-line positioning:** "A general-purpose agentic RAG platform, demonstrated on SEC filings because financial documents are a hard, regulated, table-heavy stress test. Ingestion, retrieval, and the agent layer are domain-agnostic; domain specialization lives in swappable config."

---

## 2. Core design principles (the thread through every decision)

1. **Start cheap, make it swappable, let evals justify upgrades.** Every model/component (embeddings, LLM, parser, reranker) is a config value behind an interface. Default to the cheap/general option; only upgrade when RAGAS evals prove it earns its cost.
2. **Design cloud-shaped, build locally first.** Abstract storage, metadata-first, event-driven ingestion. Local folder and Azure Blob are two implementations of one interface — moving between them is a config change, not a rewrite.
3. **Metadata is first-class.** Every chunk carries source, collection, permissions, embedding model, and retains its source text. This enables filtering, citations, permission-aware retrieval, and embedding-deprecation recovery.
4. **API-first.** All RAG logic lives behind a FastAPI service. The UI (React) is just a client. Streamlit, React, CLI — all interchangeable front-ends. This also makes the live-trace transparency feature possible.
5. **The agent never treats its own prior answers as facts.** It remembers *questions* and *retrieved source chunks*, never laundering a past generation into an input.

---

## 3. Target role context (why this project exists)

Built to demonstrate **Applied AI / LLM Engineer** capability — the builder side (agents, RAG, LLM apps), not classical ML or MLOps. The project proves: production-style RAG, agentic orchestration (LangGraph), retrieval engineering (hybrid search, reranking), evaluation rigor (RAGAS), and full-stack delivery (FastAPI + React + Azure).

---

## 4. System architecture (the layers)

```
React UI (NotebookLM-style: library, upload, chat, live trace, download)
        │  REST / streaming (SSE or WebSocket)
        ▼
FastAPI backend (async, stateless, containerized)
   ├── Query service  → agentic RAG (LangGraph)
   └── Ingestion service → event-driven
        │
Connectors (one interface: DocumentSource)
   ├── Azure Blob  ── built for real
   ├── Local folder ── for testing
   └── Slack / Monday ── clean stubs + interface
        │
Ingestion pipeline
   parse (multi-format) → chunk (semantic + parent) → attach metadata → embed → Pinecone
        │
Pinecone (vectors + metadata, multiple namespaces = collections)
   + Neo4j (bounded financial knowledge graph for GraphRAG path)
        │
RAGAS eval harness (faithfulness, answer relevance, context precision/recall)
```

---

## 5. Component decisions and rationale

### 5.1 Embedding model
- **Default:** `text-embedding-3-small` — best cost/performance (~$0.02/M tokens, ~96% of large's quality), general-purpose.
- **Dynamic per collection (NOT per query/doc):** You cannot mix embedding spaces in one searchable index — a query must be embedded with the same model as the docs it searches. So the unit of dynamism is the **collection**. Each collection records its embedding model in metadata; queries to it use that model. A finance collection could use `voyage-finance-2`; a general one uses `3-small`.
- **Upgrade rule:** only assign a specialist model to a collection when RAGAS on that collection proves it helps.
- **Deprecation resilience:** always retain chunk **source text**, record each collection's embedding model. Migration = re-embed from retained text into a new namespace, switch reads over. A `reindex.py` script handles this. (Without retained text, deprecated-model vectors are stranded — fatal.)

### 5.2 LLM (agent reasoning + generation)
- **Default:** `gpt-4o-mini` for both planning/grading and generation.
- **Principle:** In RAG, retrieval quality matters more than generation horsepower — a bigger LLM can't fix bad retrieval, it just confidently summarizes the wrong chunks. Spend the quality budget on hybrid search + reranking, use a cheap generation model.
- **Node-level upgrade:** if the grading/planning node proves unreliable with mini, swap *just that node* to a stronger model. (This is "model routing.")
- **Hosting:** Azure OpenAI (one cloud/bill/security boundary with Blob). Note: Azure doesn't offer OpenAI direct's 50% Batch discount — a defensible hybrid is Azure for live LLM, OpenAI direct batch for bulk embedding.

### 5.3 Prompt caching / CAG path
- **CAG (context-stuffing) as a size-gated path:** if the selected document set fits within a token threshold, skip retrieval and load all docs into context. Gate on **token count**, not file count.
- **KV-cache reality on hosted APIs:** OpenAI direct *does* offer prompt caching (automatic, prefix-based, exact-match, short-lived ~5–10 min, 50% off cached tokens, kicks in ≥1024 tokens). To exploit it: structure prompts **stable content first (system, docs), variable query last**, and use a session-scoped `prompt_cache_key` (e.g. `userid-sessionid`). Frame accurately: "context-stuffing for small sets, with prompt-cache reuse where the serving layer supports it" — not "manual KV-cache reuse" unless self-hosting.

### 5.4 Parser (multi-format, swappable)
- **The system is NOT PDF-only.** Parser sits between connector and chunker, routes by file type: PDF → `unstructured`/Azure Document Intelligence; DOCX → `unstructured`/python-docx; TXT/MD → direct; HTML → unstructured/BeautifulSoup; CSV/XLSX → tabular handling; PPTX → unstructured. Downstream pipeline never knows the original type — parser normalizes to clean text + metadata.
- **Default:** `unstructured` (open-source, layout-aware, multi-format).
- **Upgrade rule:** if RAGAS context-precision is weak AND traced to mangled tables, upgrade to **Azure Document Intelligence** (same-ecosystem premium) before third-party. Parser is a swappable module.

### 5.5 Chunking
- **Semantic chunking** decides *where* to split (at meaning shifts, via sentence-embedding similarity drops) — not fixed-size, which would slice tables/ideas.
- **Parent-document retrieval** decides *what's returned*: small child chunks for precise matching, linked to larger parent sections returned to the LLM for context. Resolves the small-vs-large tension.
- **Starting params (eval-tunable):** child ~256–400 tokens, parent ~1,500–2,000 tokens, ~10–15% child overlap.
- **Cost note:** semantic chunking costs more at ingest (embeds sentences to find splits). Fine at demo scale; for huge corpora, benchmark vs. recursive splitting.

### 5.6 Retrieval
- **Hybrid search:** vector (semantic) + BM25 (keyword). Vector catches meaning; BM25 catches exact terms (tickers, line items) embeddings drift on. Critical for financial docs.
- **Reranking:** Cohere Rerank re-scores the top ~20 candidates, keeps best 3–5. Cheap quality boost, considered mandatory in production. Self-hosted `bge-reranker` is the at-scale cheaper alternative.

### 5.7 Collections (multi-topic support)
- Multiple topics coexist via **Pinecone namespaces + metadata filtering**. Each collection = one namespace, embedded with one model. Retrieval is scoped to the relevant collection so a legal question never pulls finance chunks. Maps to NotebookLM-style "notebooks." Pinecone free tier allows 100 namespaces — ample.

### 5.8 GraphRAG (bounded, finance demo)
- **Scope:** schema-pluggable design, **built and proven on the finance/SEC entity schema only** for the demo. Entities: Company, Metric, FiscalYear, Segment (+ relationships). Neo4j.
- **Why bounded:** generic cross-domain graph extraction is its own multi-week project. Build the extractor schema-driven (schema is an input param), validate on finance, document how a new domain supplies a new schema. Same architectural story, finishable.
- **Routing:** the agent routes relational/comparative questions to the graph path, factual lookups to vector, small-set to CAG. Non-finance collections (no graph) route to vector/CAG.

### 5.9 The agent (LangGraph — the orchestration brain)
LangGraph is where ALL routing lives, as an explicit state machine. Nodes + conditional edges:
- **Query-rewrite node** (front): rewrites follow-ups into standalone questions using recent-turn context. Precision-first (clean query = accurate retrieval); cost-flat per turn vs. unbounded history-stuffing growth. The rewrite is *visible* in the trace.
- **Router node:** inspects question + selected docs → picks CAG / vector / graph path.
- **Retrieve nodes:** the three paths.
- **Grade node:** is retrieved context sufficient? If yes → generate; if no → retry with reformulated query (the agentic self-correction loop).
- **Generate node:** synthesizes grounded answer + citations.
- **Comparison handling:** the agent *decomposes* comparison questions into sub-queries, retrieves each fact fresh from source, compares on grounded facts — never reuses a prior generated answer.
- **State carries:** recent turns (for rewrite) + recently-retrieved chunks (reusable source context). Never generated answers as facts.
- **Observability:** LangSmith tracing shows which path each query took.

### 5.10 Conversation memory
- **Query-rewriting memory** (not full-history-stuffing, not none). An LLM node rewrites the follow-up into standalone form. Better retrieval precision + flat per-turn cost. Reuse prior **retrieved chunks** (source-grounded, safe) across turns, never prior **answers**.

### 5.11 Transparency UI (the signature feature)
Live, streaming, node-level execution trace — shows **input → transformation → output** per stage: query rewrite, router decision (+why), hybrid retrieval candidates, rerank scores, grade decision, generation with per-claim grounding, plus tokens/cost/latency per stage and last-eval faithfulness. "Show raw" expanders reveal actual prompts + raw completions (the "LLM verbose").
- **Real, not fake:** streams the agent's actual LangGraph node state via FastAPI SSE/WebSocket → React appends each stage live.
- **Two modes via toggle:** clean answer by default; "Show trace" for the full debug view.

### 5.12 Evaluation
- **RAGAS** for retrieval/answer quality: faithfulness, answer relevance, context precision, context recall. Test set of Q+A pairs, scored, catches regressions. The eval harness is the judge for every "should I upgrade X" decision.
- **Routing eval:** small set of routing test cases (did it pick graph vs vector vs CAG correctly?).
- **Groundedness/guardrail:** when answer isn't in any doc, say "I don't know" rather than invent — matters for regulated content.

---

## 6. Vector DB choice
- **Pinecone** (you know it; clean default). Free Starter tier covers the whole demo: 1 index needed (5 allowed), 100 namespaces (collections), 2GB storage (curated corpus fits with room), generous RU/WU. Account-level RBAC is NOT your permission system — document `access_scope` is implemented in your own metadata + query filtering.
- Alternatives to name in interview: pgvector (simplest if already on Postgres), Qdrant/Weaviate (open-source, self-hosted).

---

## 7. Domain-agnostic guarantees (where domain could leak in)
1. **Embeddings:** general default, specialist per-collection via config — not finance-locked.
2. **GraphRAG:** finance schema for demo, schema pluggable; non-finance routes to vector/CAG.
3. **Parser:** routes by file *type*, not topic — already general.
4. **Collections + metadata:** the mechanism that makes multi-topic real (scoped retrieval).

---

## 8. Designed-but-deferred (talking points / post-MVP)
- **Slack / Monday connectors:** interface + stubs now; one connector (Blob) built real. PM-app specifics: assemble streams into units, rich metadata, **permission scope** (agent constrained to user's data permissions), webhooks for freshness.
- **OKF (Google Open Knowledge Format, v0.1, June 2026):** markdown+YAML curated-concept format for the *curated-knowledge* layer (complements raw-doc RAG, doesn't replace it). Conceptually close to the GraphRAG concept layer. **Post-MVP optional:** express curated financial concept definitions as a small OKF bundle the agent reads as authoritative definitions. Strong "I'm current" interview point. Not core (v0.1, unstable).
- **Document updates/deletes:** metadata has `doc_id` to support update/delete of stale chunks.
- **Cost/latency observability, prompt-injection-via-documents awareness, streaming token output, multi-user auth:** designed-for or named, not core-built.

---

## 9. Scalability note (the Python question)
RAG is **I/O-bound** (waiting on API calls), not CPU-bound — so Python's GIL isn't the bottleneck. Production-scalable via: async FastAPI, stateless service behind a load balancer (horizontal scaling), containerized (Docker), background workers (Celery/Azure Functions) for heavy ingestion separate from the query path. The heavy compute lives inside the model providers, not your process. (The AQA Python→.NET migration was organizational — team standardizing on a known stack — not Python failing to scale.)
