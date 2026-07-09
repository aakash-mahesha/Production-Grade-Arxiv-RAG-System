# CLAUDE.md — Production Grade RAG System

> **Context:** This file was written during a Cowork session on 2026-07-08. It captures the full project status, architecture decisions, and the exact development roadmap. Pick up from **"Next: Immediate Tasks"** below.

---

## Project Overview

**Project name:** arXiv Paper Curator — Production RAG System
**Goal:** A production-grade RAG system over arXiv CS.AI papers. FastAPI + PostgreSQL + OpenSearch (hybrid BM25 + kNN) + Ollama (Llama 3.2) + Jina Embeddings + Airflow + Langfuse + Redis.

**Course reference:** This project follows [jamwithai/production-agentic-rag-course](https://github.com/jamwithai/production-agentic-rag-course). The course README on `main` branch has the full week-by-week breakdown and architecture diagrams. Code from weeks 5, 6, 7 can be copied/adapted directly.

---

## Current Status: Week 4 Complete, Week 5 Missing

### What's Built and Working ✅

| Component | File(s) | Status |
|-----------|---------|--------|
| FastAPI app shell | `src/main.py`, `src/dependencies.py`, `src/middleware.py` | ✅ Done |
| PostgreSQL layer | `src/models/paper.py`, `src/repositories/paper.py`, `src/db/` | ✅ Done |
| OpenSearch hybrid search | `src/services/opensearch/` — `search_unified()` method | ✅ Done |
| Jina Embeddings client | `src/services/embeddings/jina_client.py` — `embed_query()` + `embed_passages()` | ✅ Done |
| Text chunker | `src/services/indexing/text_chunker.py` | ✅ Done |
| HybridIndexingService | `src/services/indexing/hybrid_indexer.py` | ✅ Done |
| arXiv ingestion pipeline | `src/services/metadata_fetcher.py` | ✅ Done |
| Airflow DAG | `airflow/dags/arxiv_paper_ingestion.py` | ✅ Done |
| Hybrid search router | `src/routers/hybrid_search.py` | ✅ Done |
| Docker Compose stack | `compose.yaml` — PostgreSQL, OpenSearch, Ollama, Airflow | ✅ Done |

### What's Missing ❌ (Phase 5 = the LLM answer layer)

| Gap | File | What's needed |
|-----|------|---------------|
| OllamaClient only has `health_check()` | `src/services/ollama/__init__.py` | Add `generate_rag_answer()` + `generate_rag_answer_stream()` |
| `/ask` router returns hardcoded mock | `src/routers/ask.py` | Wire to real RAG pipeline |
| No RAGPromptBuilder | (doesn't exist) | Optimized prompt template for academic papers |
| No Langfuse tracing | (doesn't exist) | `src/services/langfuse/` — RAGTracer |
| No Redis cache | (doesn't exist) | `src/services/cache/` — exact-match caching |
| No agentic RAG | (doesn't exist) | `src/services/agents/` — LangGraph workflow |
| `app.state.llm_service = None` | `src/main.py` line 40 | Wire OllamaClient into DI |

---

## Development Roadmap

### Phase 5 (Week 5) — Complete RAG Pipeline (~2 hours)

**Goal:** Get a real answer out of `POST /api/v1/ask`.

**Step 1 — OllamaClient.generate()** (~30 min)

Create `src/services/ollama/client.py` with:

```python
class OllamaClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_host
        self.model = settings.ollama_model  # "llama3.2"
        self.timeout = settings.ollama_timeout

    async def health_check(self) -> dict: ...  # already exists

    async def generate_rag_answer(self, query: str, chunks: list[dict], model: str = None) -> dict:
        """Call Ollama /api/chat with context chunks, return {"answer": str}"""
        # Build messages: system prompt + user message with chunks as context
        # POST to {base_url}/api/chat with stream=False
        # Return {"answer": response["message"]["content"]}

    async def generate_rag_answer_stream(self, query: str, chunks: list[dict], model: str = None):
        """Async generator yielding Ollama stream chunks"""
        # POST to /api/chat with stream=True
        # yield each {"response": token} chunk
```

The course repo has this fully implemented at:
`https://raw.githubusercontent.com/jamwithai/production-agentic-rag-course/main/src/services/ollama/client.py`

**Step 2 — RAGPromptBuilder** (~20 min)

Create `src/services/ollama/prompts/` with a `RAGPromptBuilder` class:
- `create_rag_prompt(query, chunks)` — simple concatenation fallback
- `create_structured_prompt(query, chunks)` — optimized (80% token reduction vs naive)

Course reference: `src/services/ollama/prompts/` in the course repo.

**Step 3 — Real /ask router** (~30 min)

Update `src/routers/ask.py` to:
1. Accept question via `AskRequest`
2. Generate embedding if `use_hybrid=True` via `EmbeddingsServiceDep`
3. Call `opensearch_client.search_unified()` to retrieve top-k chunks
4. Build prompt via `RAGPromptBuilder`
5. Call `ollama_client.generate_rag_answer()` 
6. Return `AskResponse(answer, sources, chunks_used, search_mode)`

Also add `/stream` endpoint using `StreamingResponse` + `generate_rag_answer_stream()`.

**Step 4 — Wire into main.py** (~10 min)

Replace `app.state.llm_service = None` with:
```python
from src.services.ollama.client import OllamaClient
app.state.llm_service = OllamaClient(settings)
```

Add `OllamaDep` to `src/dependencies.py`.

**Step 5 — Gradio UI** (~20 min)

Copy `gradio_app.py` and `gradio_launcher.py` from course repo. Runs on port 7861.
Add to `compose.yaml` if you want it containerized, or just run locally with `uv run python gradio_launcher.py`.

---

### Phase 6 (Week 6) — LLMOps: Langfuse + Redis (~2 hours)

**Goal:** Trace every RAG request end-to-end + cache repeated queries.

**Langfuse tracing** — `src/services/langfuse/`

Copy `src/services/langfuse/` from course repo. Key class: `RAGTracer` with context managers:
- `trace_request()` — wraps the whole request
- `trace_embedding()` — tracks Jina API latency
- `trace_search()` — tracks OpenSearch query + hit count
- `trace_prompt_construction()` — tracks prompt size
- `trace_generation()` — tracks Ollama latency + tokens

**IMPORTANT — Langfuse gives you RAGAS data for free:**
Every traced request logs: question + retrieved chunks + generated answer.
That's 3 of RAGAS's 4 required fields. You only need to manually write `ground_truth` for 10-20 test cases. Much better than using TestsetGenerator.

**Redis cache** — `src/services/cache/`

Copy `src/services/cache/` from course repo. Key methods:
- `find_cached_response(request)` — exact match on query string
- `store_response(request, response)` — TTL-based storage

**Docker additions** — add to `compose.yaml`:
```yaml
redis:
  image: redis:7-alpine
  ports: ["6379:6379"]

langfuse-server:
  image: langfuse/langfuse:latest
  ports: ["3000:3000"]
  # + postgres connection config
```

Env vars needed: `REDIS_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

---

### Phase 7 (Week 7) — Agentic RAG with LangGraph (~2 hours)

**Goal:** Replace naive retrieve→answer with a stateful decision-making agent.

**The LangGraph workflow** — `src/services/agents/`

Copy `src/services/agents/` from course repo. Structure:

```
src/services/agents/
├── agentic_rag.py      # AgenticRAGService — builds + runs the StateGraph
├── state.py            # AgentState (TypedDict with all state fields)
├── context.py          # Context (dependency injection for nodes)
├── config.py           # GraphConfig (model, top_k, max_attempts, guardrail_threshold)
├── tools.py            # create_retriever_tool() — wraps search_unified as LangChain tool
└── nodes/
    ├── guardrail.py    # ainvoke_guardrail_step — scores query 0-100 for domain relevance
    ├── retrieve.py     # ainvoke_retrieve_step — creates tool call for ToolNode
    ├── grade.py        # ainvoke_grade_documents_step — semantic relevance per chunk
    ├── rewrite.py      # ainvoke_rewrite_query_step — LLM rewrites query on grade failure
    ├── generate.py     # ainvoke_generate_answer_step — answer from relevant chunks only
    └── out_of_scope.py # ainvoke_out_of_scope_step — handles off-domain queries
```

**Graph edges:**
```
START → guardrail
guardrail → [continue → retrieve | out_of_scope → END]
retrieve → tool_retrieve (via tools_condition)
tool_retrieve → grade_documents
grade_documents → [generate_answer | rewrite_query]
rewrite_query → retrieve  (loops until max_retrieval_attempts)
generate_answer → END
```

**New endpoint** — `src/routers/agentic_ask.py`:
```
POST /api/v1/agentic-ask
```
Returns: `answer`, `sources`, `reasoning_steps`, `retrieval_attempts`, `rewritten_query`, `guardrail_score`.

**Why this matters for Aakash's resume:**
This is the same `StateGraph` + conditional edge pattern already in Anomaly Intelligence. The document grading + query rewriting is what distinguishes "production RAG" from "demo RAG" in technical interviews at Anthropic, OpenAI, Google.

**Telegram bot** (optional): `src/services/telegram/` — skip unless there's time.

---

### RAGAS Evaluation Suite (~2 hours, run after Phase 6)

**Goal:** Quantify RAG pipeline quality with 4 metrics: faithfulness, answer relevancy, context precision, context recall.

**Approach:** Use Langfuse traces (from Phase 6) as the ground truth dataset.

Create `scripts/eval_ragas.py`:

```python
# 1. Pull 20 recent traces from Langfuse (each trace has question, contexts, answer)
# 2. Manually write ground_truth for each (or use LLM to generate them)
# 3. Build ragas.Dataset from (question, answer, contexts, ground_truth)
# 4. Run evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
# 5. Print results table + save to scripts/ragas_results.json

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
```

**Dependencies to add to pyproject.toml:**
```toml
"ragas>=0.2.0",
"langchain-openai>=0.1.0",  # RAGAS uses OpenAI as judge LLM
```

**Required env var:** `OPENAI_API_KEY` (for RAGAS judge LLM — doesn't need to be expensive, gpt-4o-mini works)

**Resume bullet this produces:**
> Evaluated RAG pipeline using RAGAS (faithfulness, answer relevancy, context precision/recall) — identifying retrieval bottlenecks and improving context precision by X% through chunk-size tuning

---

## Key Architecture Decisions (don't change these)

- **Embeddings:** Jina AI `jina-embeddings-v3`, 1024 dimensions, retrieval task type
- **Search:** OpenSearch `search_unified()` — RRF fusion of BM25 + kNN, `use_hybrid=True` by default
- **LLM:** Ollama Llama 3.2 (local, no API cost) — model string: `"llama3.2"`
- **Chunking:** Section-aware with overlap, min/max size configured in `TextChunker`
- **DB:** PostgreSQL with SQLAlchemy 2.0 + Repository pattern — `PaperRepository.upsert()`
- **DI pattern:** FastAPI `Annotated[X, Depends(get_x)]` — see `src/dependencies.py` for existing pattern
- **Async:** Everything in services is `async def` — use `httpx.AsyncClient` for HTTP calls

## Environment Variables

Current `.env` / config (see `src/config.py` — `Settings` class with Pydantic Settings):

```
POSTGRES_DATABASE_URL=postgresql+psycopg2://...
OPENSEARCH_HOST=http://opensearch:9200
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
JINA_API_KEY=<required for embeddings>
ENVIRONMENT=development
```

Add for Phase 6:
```
REDIS_URL=redis://localhost:6379
LANGFUSE_PUBLIC_KEY=<from Langfuse dashboard>
LANGFUSE_SECRET_KEY=<from Langfuse dashboard>
LANGFUSE_HOST=http://localhost:3000
```

Add for RAGAS:
```
OPENAI_API_KEY=<for RAGAS judge LLM>
```

---

## Service URLs (when stack is running)

| Service | URL |
|---------|-----|
| FastAPI API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Gradio Chat UI | http://localhost:7861 |
| OpenSearch | http://localhost:9200 |
| OpenSearch Dashboards | http://localhost:5601 |
| Airflow | http://localhost:8080 (admin/admin) |
| Ollama | http://localhost:11434 |
| Langfuse Dashboard | http://localhost:3000 (Phase 6+) |
| Redis | localhost:6379 (Phase 6+) |

---

## Running the Stack

```bash
# Start all services
make start
# or
docker compose up --build -d

# Check health
make health
curl http://localhost:8000/api/v1/health

# Run the app locally (hot reload)
uv run uvicorn src.main:app --reload --port 8000

# Test hybrid search (should work now)
curl -X POST http://localhost:8000/api/v1/hybrid-search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "size": 5, "use_hybrid": true}'

# Test ask (currently returns mock — this is what we're fixing)
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'

# Ingest papers manually
uv run python scripts/manual_ingest.py

# Run tests
make test
```

---

## Dependency Management

Uses **UV** (not pip directly):
```bash
uv sync                          # Install all deps
uv add <package>                 # Add new dependency
uv run python <script>           # Run script in venv
uv run pytest                    # Run tests
```

---

## Code Quality

```bash
make format    # ruff format
make lint      # ruff check + mypy
make test      # pytest
make test-cov  # pytest with coverage
```

---

## Context from Career Session

This project is being built as part of a portfolio targeting MANGOS companies (Meta, Anthropic, Nvidia, Google, OpenAI, SpaceX). Key signals this project demonstrates:

- **LangGraph stateful agents** — LinkedIn's #1 trending AI engineer skill (Week 7)
- **RAG + retrieval evaluation (RAGAS)** — Anthropic/OpenAI explicitly look for this
- **LLMOps with Langfuse** — on LinkedIn's skills-on-the-rise 2026 list
- **FastAPI for model serving** — explicitly on LinkedIn's 2026 skills list
- **Hybrid search (BM25 + vector)** — production pattern, not demo pattern

**Target resume bullets to produce:**
1. (Week 5) Built production RAG pipeline with Ollama Llama 3.2, streaming SSE, and optimized prompt templates achieving 80% token reduction
2. (Week 6) Added LLMOps layer: Langfuse end-to-end tracing + Redis exact-match caching (150-400× speedup on repeated queries)
3. (Week 7) Extended to Agentic RAG via LangGraph — guardrail node, semantic document grading, adaptive query rewriting
4. (RAGAS) Evaluated RAG pipeline with RAGAS across 4 metrics, identifying and fixing context precision bottleneck

---

## Immediate Next Steps (start here)

1. `cd /Users/akm/Documents/moai/rag-system/prod-grade-rag-system`
2. `make start` — ensure Docker stack is running
3. Fetch the Week 5 Ollama client from course repo:
   `curl -s https://raw.githubusercontent.com/jamwithai/production-agentic-rag-course/main/src/services/ollama/client.py`
4. Create `src/services/ollama/client.py` with `generate_rag_answer()` + `generate_rag_answer_stream()`
5. Create `src/services/ollama/prompts/` with `RAGPromptBuilder`
6. Update `src/routers/ask.py` to use real RAG pipeline
7. Wire `OllamaClient` into `src/dependencies.py` + `src/main.py`
8. Test: `curl -X POST http://localhost:8000/api/v1/ask -d '{"question":"What is transformer attention?"}'`
9. Add Gradio UI: `uv run python gradio_launcher.py`
10. Proceed to Phase 6 (Langfuse + Redis)
