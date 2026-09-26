# Architecture

> **Status:** design. The package layout under `src/rag_eval_platform/` is in place; the modules are placeholders and will be implemented layer by layer.

## Goals

- Answer questions from a private document set, grounded in retrieved context and with citations back to sources.
- Treat evaluation as part of the system: every change is scored against a golden dataset before it ships.
- Keep each layer independently testable and replaceable (vector store, embedding model, LLM provider).

## System overview

```
                         OFFLINE (ingestion)
┌───────────┐   ┌──────────┐   ┌────────────┐   ┌──────────────────┐
│ data/raw  │──▶│ Loaders  │──▶│ Chunking   │──▶│ Embedding        │──┐
└───────────┘   └──────────┘   └────────────┘   └──────────────────┘  │
                                                                     ▼
                                                         ┌──────────────────────┐
                                                         │ Vector store         │
                                                         │ (Chroma / Qdrant)    │
                                                         └──────────┬───────────┘
                         ONLINE (query)                             │
┌───────────┐   ┌──────────────┐   ┌──────────────────┐   ┌─────────▼────────┐   ┌──────────────┐
│ Client    │──▶│ FastAPI      │──▶│ Query embedding  │──▶│ Retrieval top-k  │──▶│ Re-ranker    │
└───────────┘   └──────┬───────┘   └──────────────────┘   └──────────────────┘   │ (optional)   │
      ▲                │                                                         └──────┬───────┘
      │                │           ┌──────────────────────────────┐                     │
      └────────────────┴───────────│ Generation: prompt + LLM,    │◀────────────────────┘
          answer + citations       │ answer with [n] citations    │
                                   └──────────────┬───────────────┘
                                                  ▼
                                   ┌──────────────────────────────┐
                                   │ Observability: query log,    │
                                   │ latency, cost, eval scores   │
                                   └──────────────────────────────┘
```

## Components

| Layer | Module | Responsibility |
|---|---|---|
| Ingestion | `ingestion/loaders.py` | Read raw documents (Markdown, text, PDF) into `Document` objects with a stable id (path relative to the corpus). |
| | `ingestion/chunking.py` | Split documents into chunks: fixed-size with overlap, recursive (paragraph → line → word), semantic (planned). |
| | `ingestion/embedding.py` | Turn chunks and queries into vectors (Sentence Transformers or OpenAI). |
| Retrieval | `retrieval/vector_store.py` | Store and search vectors behind one interface; Chroma for local dev, Qdrant for production. |
| | `retrieval/retriever.py` | Embed the query, fetch top-k candidates, optionally re-rank. |
| | `retrieval/reranker.py` | Cross-encoder re-ranking of candidates for more precise ordering. |
| Generation | `generation/prompt_templates.py` | System prompt and context formatting; numbered sources for citation. |
| | `generation/generator.py` | Call the LLM and map `[n]` markers in the answer back to source chunks. |
| Evaluation | `evaluation/golden_dataset.py` | Load and validate `data/golden_dataset/qa_pairs.json`. |
| | `evaluation/metrics.py` | Precision@k, Recall@k, MRR, NDCG. |
| | `evaluation/evaluator.py` | Run the golden dataset through the pipeline and compare scores to thresholds. |
| Observability | `observability/logger.py` | Log queries, retrieved chunks, latency and evaluation runs. |
| | `observability/dashboard.py` | Streamlit dashboard: scores over time, recent queries, latency. |
| Config | `config/settings.py` | Typed settings loaded from environment variables / `.env`. |
| | `config/logging_config.py` | Process-wide logging setup. |
| API | `api/main.py`, `api/routes.py` | FastAPI app: `GET /health`, `POST /query`. |

## Request flow

1. **Query**: the client sends a question to `POST /query`.
2. **Embed**: the question is embedded with the same model used at ingestion.
3. **Retrieve**: the vector store returns the top-k most similar chunks; the optional re-ranker reorders a larger candidate set.
4. **Generate**: the chunks are numbered and inserted into the prompt; the LLM answers only from that context and cites sources as `[n]`.
5. **Log and evaluate**: the question, retrieved chunk ids, answer and latency are logged; sampled traffic is evaluated in the background.

## Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Package layout | `src/` layout | Keeps the installable package separate from tests, scripts and notebooks. |
| Pluggable backends | Interfaces (protocols) for embedder, vector store, re-ranker and LLM client | Swap Chroma ↔ Qdrant or one LLM provider for another without touching business logic. |
| Default chunking | Recursive | Preserves paragraph and sentence boundaries better than fixed-size, at low cost. |
| Citations | Numbered context blocks, `[n]` markers | Traceability back to source documents is an enterprise requirement. |
| Evaluation data | Golden dataset versioned in the repo | Evaluation results are reproducible and reviewed like code. |
| Quality gate | Separate `evaluation_gate.yml` workflow | Quality regressions block merges, independent of unit tests. |

## Configuration

All settings come from environment variables (see `.env.example`); nothing is hard-coded. Main groups:

- **Ingestion**: data directory, chunk strategy, chunk size and overlap.
- **Embedding**: provider and model name.
- **Vector store**: backend, collection name, connection URL or path.
- **Retrieval**: top-k, re-ranking on/off, re-ranker model.
- **Generation**: LLM provider, model name, API key.
- **Evaluation**: golden dataset path and metric thresholds.

## Tooling by phase

Phases are listed in build order. **Status**: *in use* = already installed or configured; *chosen* = decided in these docs, not yet added; *proposed* = recommended to fill a gap, confirm before adding. Dependencies are added with `uv add` only when the phase that needs them starts.

| Phase | Tool | Purpose | Status |
|---|---|---|---|
| **0. Project setup** | Python 3.12 | Runtime (pinned in `.python-version`) | in use |
| | uv | Dependency management, virtualenv, lockfile | in use |
| | hatchling | Build backend for the `src/` package | in use |
| | Pytest | Test runner for unit, integration and evaluation tests | in use |
| | Ruff | Lint and format | in use |
| | mypy | Static type checking of the protocol interfaces | in use |
| | pydantic-settings | Typed settings from env / `.env` in `config/settings.py` | in use |
| **1. Ingestion** | `langchain-text-splitters` | Recursive chunking (fixed-size is our own sliding window) | in use |
| | pypdf | Extract text and page offsets from PDFs; Markdown/text are read directly | in use |
| | Sentence Transformers | Local, free embedding model (default); optional extra `local-embeddings` | in use |
| | OpenAI Embeddings | Hosted embedding alternative; optional extra `openai` | in use |
| | tiktoken | Token counts for chunk sizing and cost estimates | proposed |
| **2. Retrieval** | Chroma (server in Docker, `chromadb-client`) | Dev vector store over HTTP; also a CI service container | in use |
| | Qdrant | Production vector store with metadata filtering | chosen |
| | Sentence Transformers `CrossEncoder` | Optional re-ranker (`RAG_RERANK=true`, MS MARCO MiniLM) | in use |
| | NumPy | Vector math, similarity checks in tests | chosen |
| **3. Generation** | Anthropic / OpenAI SDKs (via LangChain chat models) | LLM answer generation with `[n]` citations | chosen |
| | LangGraph | Control flow beyond a linear chain (retry, confidence branching); add only when needed | chosen, deferred |
| **4. Evaluation** | RAGAS | Primary generation metrics: faithfulness, answer relevance, context precision/recall | chosen |
| | DeepEval | Pytest-style LLM evaluation tests | chosen |
| | Custom `evaluation/metrics.py` | Deterministic retrieval metrics: Precision@k, Recall@k, MRR, NDCG@k | chosen |
| | Pandas | Golden dataset and score reports, per-`query_type` breakdowns | chosen |
| **5. CI/CD gate** | GitHub Actions | `ci.yml` (lint, types, unit tests, secret scan) in use; `evaluation_gate.yml` (block merge on regression) chosen | in use |
| | `astral-sh/setup-uv` action | Install uv and cache dependencies in CI | in use |
| | pytest-cov | Coverage report in CI | in use |
| | gitleaks (GitHub Action) | Fail CI if a secret is committed | in use |
| **6. API and deployment** | FastAPI | `GET /health`, `POST /query` | chosen |
| | Uvicorn | ASGI server for the API | proposed |
| | httpx | FastAPI `TestClient` for integration tests | proposed |
| | Docker / Docker Compose | Local infrastructure (Chroma now); API image later | in use |
| | chromadb-admin (community image, `ui` profile) | Browse Chroma collections and chunks in a web UI | in use |
| | JupyterLab + pandas (`notebook` group) | `notebooks/exploration.ipynb`: inspect chunks, ask questions, run experiments | in use |
| **7. Observability** | Python `logging` (JSON lines) | Query, latency, cost and eval-run logs | chosen |
| | Streamlit | Dashboard: scores over time, recent queries, latency | chosen |
| | TruLens | Continuous quality tracking with feedback functions | chosen, optional |
| **8. Security and governance** | Microsoft Presidio | Detect and mask PII at ingestion, before embedding | proposed |
| | Vector store metadata filters (Chroma / Qdrant) | Retrieval-time access control | chosen |

## Deployment

- `docker/docker-compose.yml` runs the local infrastructure: Chroma on `localhost:8001`, with data in a named volume. `docker/Dockerfile` (planned) will build the API image, and the API will be added to the same compose file.
- The API is stateless; all state lives in the vector store and the observability log.

## Security and governance

- **Secrets** come from the environment only; `.env` is git-ignored.
- **PII**: detect and mask or redact personal data at ingestion, before it is embedded.
- **Access control**: filter by user permissions inside the vector search (metadata filters), not after retrieval.
- **Audit logging**: record which chunks were shown to which user and when.

## Repository layout

```
src/rag_eval_platform/   application package (layers above)
data/raw/                source documents
data/processed/          chunked output
data/golden_dataset/     evaluation ground truth (qa_pairs.json)
tests/                   unit, integration and evaluation tests
scripts/                 ingestion, evaluation and seeding entry points
.github/workflows/       ci.yml and evaluation_gate.yml
docker/                  container build and local stack
docs/                    architecture, evaluation methodology, background guide
```

See [evaluation_methodology.md](evaluation_methodology.md) for how quality is measured and gated.
