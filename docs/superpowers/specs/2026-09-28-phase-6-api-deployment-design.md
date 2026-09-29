# Phase 6 — API and deployment: design

**Date:** 2026-09-28 · **Branch:** `feat/phase-6-api-deployment` · **Status:** approved in chat, awaiting spec review

## 1. Goal

Make the RAG pipeline usable outside the playground: a small, safe HTTP API that answers questions with citations, and a container that runs it next to Chroma. As with every phase, it ships with docs and a playground view that shows it working.

**Success criteria**

- `GET /health` reports whether Chroma and the LLM are reachable (200 or 503).
- `POST /query` returns a cited answer as JSON; `POST /query/stream` streams it as server-sent events.
- `docker compose -f docker/docker-compose.yml --profile api up -d` starts the API next to Chroma, and its health check passes.
- CI builds the image and checks `/health`.
- The playground's **⑦ API** tab sends real requests and shows the curl command, the JSON response, the status code and the latency.

## 2. Decisions

| Question | Decision |
|---|---|
| Dependencies | `fastapi` and `uvicorn` (runtime), `httpx` (dev group, for FastAPI's `TestClient`). Approved. |
| Streaming | Server-sent events with FastAPI's `StreamingResponse`; no extra dependency. |
| Auth | Optional API key (`RAG_API_KEY`, `SecretStr`), off by default, checked in constant time. |
| Rate limit | In-process sliding window per client (`RAG_API_RATE_LIMIT`, default 30 per minute); no extra dependency. |
| Ollama in Docker | Stays native on the host (GPU); the container reaches it at `host.docker.internal:11434`. |
| Compose | A new `api` service under an `api` profile, so the default `up -d` still starts only Chroma. |

## 3. API (`src/rag_eval_platform/api/`)

### 3.1 Schemas (`api/schemas.py`, pydantic)

- `QueryRequest`: `question: str` (stripped, 1–2,000 characters), `answer_style: Literal["concise", "detailed"] = "concise"`; unknown fields are rejected (`extra="forbid"`).
- `CitationOut`: `number`, `doc_id`, `page`, `chunk_id`.
- `SourceOut`: `number`, `doc_id`, `page`, `chunk_id`, `score`, `text`.
- `QueryResponse`: `question`, `answer`, `refusal`, `citations`, `invalid_citations`, `sources`, `model`, `prompt_version`, `input_tokens`, `output_tokens`, `timings` (`retrieve_ms`, `generate_ms`, `total_ms`).
- `HealthResponse`: `status: Literal["ok", "degraded"]`, `checks: dict[str, CheckOut]`, where `CheckOut` is `ok: bool, detail: str`.
- `ErrorResponse`: `error: str`, `hint: str | None`.
- `to_response(answer, retrieve_ms, total_ms) -> QueryResponse`: a pure mapping from the pipeline's `Answer`.

### 3.2 Endpoints (`api/routes.py`)

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Runs the injected health checks (Chroma count, LLM `/v1/models` lists the configured model). 200 when all pass, 503 otherwise; never raises. |
| `POST /query` | Retrieves and generates through `RagPipeline`, so the "Question answered" log line is still written. 200 with `QueryResponse`. |
| `POST /query/stream` | `text/event-stream`: `event: status` (`retrieving`, `generating`), `event: token` (each text piece), `event: done` (the full `QueryResponse` as JSON), or `event: error`. |
| `GET /docs`, `/openapi.json` | FastAPI's built-in documentation. |

Errors: `GenerationError` → 502, `VectorStoreError` → 503, `OptionalDependencyError` → 503, validation → 422, auth → 401, rate limit → 429 with `Retry-After`. Every error body is `ErrorResponse`; no stack traces reach the client, and details are logged server-side with `extra={...}`.

### 3.3 App (`api/main.py`)

- `create_app(services: ApiServices) -> FastAPI`, where `ApiServices` holds a `pipeline_for(style) -> RagPipeline` factory, the health checks, `api_key: SecretStr | None` and a `RateLimiter`. Tests pass fakes; `create_app_from_settings(settings)` builds the real ones (models load once, at startup).
- Middleware order: rate limit, then API key (except `/health`, `/docs`, `/openapi.json`), then the route.
- `api/security.py`: `check_api_key(given, expected) -> bool` (`hmac.compare_digest`) and `RateLimiter(limit, window_s, clock)` with `allow(client) -> tuple[bool, float]` (allowed, retry-after seconds), both pure and unit-tested.
- `scripts/serve_api.py`: `uvicorn` on `RAG_API_HOST`:`RAG_API_PORT` (defaults `127.0.0.1:8000`).

### 3.4 Settings

`api_host = "127.0.0.1"`, `api_port = 8000`, `api_key: SecretStr | None = None` (env `RAG_API_KEY`), `api_rate_limit = 30` (per minute), plus commented entries in `.env.example` (empty key).

## 4. Deployment (`docker/`)

- **`docker/Dockerfile`**, multi-stage:
  - The builder uses the official uv image and runs `uv sync --locked --no-dev --extra local-embeddings` (CPU torch on Linux, as configured).
  - The runtime stage is `python:3.12-slim` with the virtualenv copied in and a non-root user.
  - The embedding model is downloaded at build time (`HF_HOME` inside the image), so the container needs no internet.
  - `HEALTHCHECK` calls `/health`; `CMD` runs uvicorn on `0.0.0.0:8000`.
- **`.dockerignore`**: excludes `.venv`, `data/playground`, `reports`, `.env`, notebooks, caches and `.git`.
- **Compose** service `api` (profile `api`):
  - builds from the repository root and depends on a healthy `chroma`;
  - sets `RAG_CHROMA_HOST=chroma`, `RAG_CHROMA_PORT=8000` and `RAG_OLLAMA_BASE_URL=http://host.docker.internal:11434/v1`, with `extra_hosts: host.docker.internal:host-gateway` for Linux;
  - publishes `127.0.0.1:8000`.
- **CI** (`ci.yml`), a new job, "API image":
  - builds the image with Docker Buildx and a GitHub Actions cache;
  - runs it with a Chroma service;
  - checks that `/health` returns JSON. It may be 503 because CI has no Ollama; the body must name the LLM check.
  - The image is not pushed.

## 5. Playground: the ⑦ API tab

Pure logic in `playground/api_view.py` (unit-tested); `tabs/api.py` only draws.

- **Status:** a pill for "API up at http://localhost:8000" or "not running", with the command to start it (`scripts/serve_api.py`, or the compose profile).
- **Request flow:** a Graphviz diagram of client → FastAPI (auth, rate limit) → retrieve → generate → JSON, with each stage's milliseconds after a request.
- **Request builder:**
  - question plus style;
  - the exact **curl** command (`curl_command(url, request, api_key)`, shell-quoted, with the key masked in the display);
  - **Send** makes a real HTTP request with `urllib` (no new dependency) and shows the status code, latency and formatted JSON;
  - a **Stream** toggle shows the `/query/stream` events arriving live (status, a token count and the text, then `done`).
- **Links:** to `/docs`.
- Errors (API down, 401, 429, 502) show as a clear message with the fix.

## 6. Docs

- `docs/learning/phase-6.md`: what an API is and why, the request flow (sequence diagram), the endpoints with examples, streaming events, the container setup (a diagram of the host, Docker, Chroma and Ollama), security basics, and what to try in ⑦ API.
- `readme.md`, `CLAUDE.md`, `docs/architecture.md` (Phase 6 rows become *in use*) and `docs/learning/playground.md` (the new tab).

## 7. Testing

- **Unit:** schemas and validation, `to_response`, `check_api_key`, `RateLimiter` (window, retry-after), and each endpoint through `TestClient` with fake pipelines (200, 422, 401, 429, 502, 503, the SSE event order). Also `api_view` (the curl command with quoting and a masked key, the SSE line parser, flow-diagram timings).
- **Integration:** `create_app_from_settings` against real Chroma (skips when it's unavailable) and `/health` over HTTP; the playground smoke test lists the ⑦ API tab.
- **CI:** the image-build job.

## 8. Out of scope

- Multi-user auth, tenants, and access control inside retrieval (Phase 8).
- Persisting logs for dashboards (Phase 7).
- Publishing the image to a registry, and cloud deployment.
