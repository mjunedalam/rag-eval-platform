# Phase 6 API and Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the RAG pipeline over a small, safe HTTP API (`/health`, `/query`, `/query/stream`), package it as a Docker image next to Chroma, and show it working in a new ⑦ API playground tab.

**Architecture:** Pure pieces are unit-tested on their own: request and response schemas, the API-key check, the rate limiter, SSE formatting and health checks. They are assembled by `create_app(services)`, which takes injected pipelines, so tests use fakes through FastAPI's `TestClient`. `create_app_from_settings` builds the real services once. The container is a multi-stage uv build (CPU torch, non-root user, model baked in) run as a compose profile. The playground tab talks to the running API over plain `urllib`.

**Tech Stack:** FastAPI 0.141, Uvicorn 0.54, httpx (TestClient), pydantic 2, Docker Buildx, GitHub Actions, Streamlit.

**Spec:** `docs/superpowers/specs/2026-09-28-phase-6-api-deployment-design.md`

## Global Constraints

- New dependencies only: `fastapi>=0.141`, `uvicorn>=0.54` (runtime), `httpx>=0.28` (dev group). Commit `uv.lock` with `pyproject.toml`.
- Settings only through `get_settings()`: `api_host="127.0.0.1"`, `api_port=8000`, `api_key: SecretStr | None` (env `RAG_API_KEY`; empty means off), `api_rate_limit=30` (per minute).
- Question: stripped, 1–2,000 characters. `answer_style`: `"concise"` (default) or `"detailed"`. Unknown fields are rejected.
- Status codes: 401 bad key, 422 invalid request, 429 rate limited (with `Retry-After`), 502 LLM failure, 503 Chroma or missing extra. Every error body is `{"error": ..., "hint": ...}`, and no stack traces reach the client.
- `/health`, `/docs` and `/openapi.json` never need the key and are never rate limited.
- Compose: the `api` service lives under the `api` profile; plain `up -d` still starts only Chroma. Ollama stays on the host (`host.docker.internal:11434`).
- mypy strict; Ruff line length 100; no `print` in library code (logging only; CLI scripts may print); secrets never logged.
- Every phase updates the docs and adds a playground view. **Commit only when the user says so.**

## Review Focus

1. **An empty `RAG_API_KEY=` in `.env`** must mean auth is off, not "the key is the empty string". Tested in Task 1.
2. **A question of only spaces** must be a 422, not a model call. Tested in Task 3.
3. **A stream that fails midway** must end with `event: error`, not a broken connection or a traceback. Tested in Task 5.
4. **The rate limit window slides, and clients are independent:** after 60 s the same client is allowed again, and client B isn't blocked by client A. Tested in Task 3.
5. **The API key must never be shown in full** in the playground's curl command. It is masked. Tested in Task 7.

---

### Task 1: Dependencies and API settings

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (via `uv add`)
- Modify: `src/rag_eval_platform/config/settings.py`, `.env.example`
- Test: `tests/unit/test_settings.py`

**Interfaces:**
- Produces: `Settings.api_host: str`, `Settings.api_port: int`, `Settings.api_key: SecretStr | None`, `Settings.api_rate_limit: int`.

- [ ] **Step 1: Add the dependencies**

Run: `uv add "fastapi>=0.141" "uvicorn>=0.54" && uv add --dev "httpx>=0.28"`
Expected: `pyproject.toml` and `uv.lock` updated; `uv lock --check` passes.

- [ ] **Step 2: Write the failing tests** (add to `tests/unit/test_settings.py`)

```python
def test_api_defaults() -> None:
    settings = Settings(_env_file=None)

    assert (settings.api_host, settings.api_port) == ("127.0.0.1", 8000)
    assert settings.api_key is None
    assert settings.api_rate_limit == 30


def test_an_empty_api_key_means_auth_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "")

    assert Settings(_env_file=None).api_key is None


def test_an_api_key_is_kept_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "s3cret-value")

    settings = Settings(_env_file=None)

    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "s3cret-value"
    assert "s3cret-value" not in repr(settings)
```

(Use the file's existing imports; add `import pytest` if missing.)

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/unit/test_settings.py -q`
Expected: FAIL, `'Settings' object has no attribute 'api_host'`.

- [ ] **Step 4: Implement** (in `Settings`, after the evaluation thresholds block)

```python
    # API (Phase 6)
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, gt=0, lt=65536)
    api_key: SecretStr | None = None  # RAG_API_KEY; empty means auth is off
    api_rate_limit: int = Field(default=30, gt=0)  # requests per minute per client
```

and a validator next to the existing ones:

```python
    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_api_key_is_none(cls, value: object) -> object:
        return None if value in ("", None) else value
```

(import `field_validator` from pydantic). In `.env.example`, add:

```
# RAG_API_HOST=127.0.0.1
# RAG_API_PORT=8000
# RAG_API_KEY=                       # set to require the X-API-Key header; empty = off
# RAG_API_RATE_LIMIT=30              # requests per minute per client
```

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/test_settings.py -q`
Expected: all pass.

- [ ] **Step 6: Commit (when the user asks)**

```bash
git add pyproject.toml uv.lock src/rag_eval_platform/config/settings.py .env.example tests/unit/test_settings.py
git diff --cached
git commit -m "build: add FastAPI and Uvicorn, and API settings"
```

---

### Task 2: Timed answers and a style-aware generator

**Files:**
- Modify: `src/rag_eval_platform/pipeline.py`, `src/rag_eval_platform/generation/generator.py` (`create_generator`)
- Test: `tests/unit/test_pipeline.py`, `tests/unit/test_generator.py`

**Interfaces:**
- Produces: `TimedAnswer(answer: Answer, retrieve_ms: float, total_ms: float)` (frozen dataclass in `pipeline.py`); `RagPipeline.ask_timed(question) -> TimedAnswer`; `create_generator(settings, style: AnswerStyle = "concise") -> Generator` (the detailed style uses `llm_detailed_max_tokens`).

- [ ] **Step 1: Write the failing tests**

In `tests/unit/test_pipeline.py`:

```python
def test_ask_timed_reports_retrieval_and_total_time() -> None:
    timed = make_pipeline().ask_timed("What is MRR?")

    assert timed.answer.text == "MRR is the mean reciprocal rank [1]."
    assert 0 <= timed.retrieve_ms <= timed.total_ms
```

In `tests/unit/test_generator.py`:

```python
def test_create_generator_uses_the_style_and_its_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Settings] = []

    def fake_from_settings(settings: Settings) -> FakeLlm:
        seen.append(settings)
        return FakeLlm("x")

    monkeypatch.setattr(OpenAICompatibleClient, "from_settings", staticmethod(fake_from_settings))
    settings = Settings(_env_file=None)

    concise = create_generator(settings)
    detailed = create_generator(settings, "detailed")

    assert (concise.style, detailed.style) == ("concise", "detailed")
    assert seen[0].llm_max_tokens == settings.llm_max_tokens
    assert seen[1].llm_max_tokens == settings.llm_detailed_max_tokens
```

(Import `Settings`, `OpenAICompatibleClient` and `create_generator` if the file doesn't already.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_pipeline.py tests/unit/test_generator.py -q`
Expected: FAIL, `'RagPipeline' object has no attribute 'ask_timed'` and `create_generator() takes 1 positional argument`.

- [ ] **Step 3: Implement**

`generator.py`:

```python
def create_generator(settings: Settings, style: AnswerStyle = "concise") -> Generator:
    """The generator for a style; the detailed style gets its larger token limit."""
    if style == "detailed":
        settings = settings.model_copy(update={"llm_max_tokens": settings.llm_detailed_max_tokens})
    return Generator(OpenAICompatibleClient.from_settings(settings), style)
```

`pipeline.py`: add `from dataclasses import dataclass`, then:

```python
@dataclass(frozen=True)
class TimedAnswer:
    answer: Answer
    retrieve_ms: float
    total_ms: float
```

and in `RagPipeline`:

```python
def ask(self, question: str) -> Answer:
    return self.ask_timed(question).answer


def ask_timed(self, question: str) -> TimedAnswer:
    """Answer and report how long retrieval and the whole request took."""
    started = time.perf_counter()
    results = self.retriever.retrieve(question)
    retrieved = time.perf_counter()
    answer = self.generator.generate(question, results)
    total_ms = (time.perf_counter() - started) * 1000
    _log_answer(answer, total_ms=total_ms)
    return TimedAnswer(answer, (retrieved - started) * 1000, total_ms)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_pipeline.py tests/unit/test_generator.py -q`
Expected: all pass, including the existing `ask` tests.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/pipeline.py src/rag_eval_platform/generation/generator.py tests/unit/test_pipeline.py tests/unit/test_generator.py
git commit -m "feat: timed answers and a style-aware generator for the API"
```

---

### Task 3: API schemas and security

**Files:**
- Create: `src/rag_eval_platform/api/schemas.py`, `src/rag_eval_platform/api/security.py`
- Test: `tests/unit/test_api_schemas.py`, `tests/unit/test_api_security.py`

**Interfaces:**
- Consumes: `TimedAnswer` (Task 2), `Answer`, `SearchResult`.
- Produces: `QueryRequest(question, answer_style)`, `CitationOut`, `SourceOut`, `Timings`, `QueryResponse`, `CheckOut(ok, detail)`, `HealthResponse(status, checks)`, `ErrorResponse(error, hint)`; `to_response(timed: TimedAnswer) -> QueryResponse`; `check_api_key(given: str | None, expected: str | None) -> bool`; `RateLimiter(limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic)` with `.allow(client: str) -> tuple[bool, float]`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_api_schemas.py`:

```python
"""Tests for api.schemas: what the API accepts and returns."""

import pytest
from pydantic import ValidationError

from rag_eval_platform.api.schemas import QueryRequest, to_response
from rag_eval_platform.generation.generator import Answer, Citation
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.pipeline import TimedAnswer
from rag_eval_platform.retrieval.vector_store import SearchResult

SOURCE = SearchResult(Chunk(id="book.pdf#4", doc_id="book.pdf", index=4, text="F=ma",
                            start_index=0, page=3), 0.41)  # fmt: skip


def test_request_strips_the_question_and_defaults_to_concise() -> None:
    request = QueryRequest(question="  What is MRR?  ")

    assert (request.question, request.answer_style) == ("What is MRR?", "concise")


@pytest.mark.parametrize("body", [
    {"question": "   "}, {"question": "x" * 2001}, {"question": "q", "answer_style": "long"},
    {"question": "q", "extra": 1},
])  # fmt: skip
def test_bad_requests_are_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(body)


def test_to_response_maps_the_answer_with_citations_sources_and_timings() -> None:
    answer = Answer(
        question="q?", text="F equals ma [1].", citations=(Citation(1, SOURCE),),
        invalid_citations=(7,), sources=(SOURCE,), model="qwen3:8b", prompt_version="v1",
        input_tokens=100, output_tokens=12, latency_ms=900.0,
    )  # fmt: skip

    response = to_response(TimedAnswer(answer, retrieve_ms=40.0, total_ms=950.0))

    assert response.answer == "F equals ma [1]."
    assert response.citations[0].model_dump() == {"number": 1, "doc_id": "book.pdf", "page": 3,
                                                  "chunk_id": "book.pdf#4"}  # fmt: skip
    assert response.sources[0].score == 0.41
    assert response.invalid_citations == [7]
    assert response.refusal is False
    assert response.timings.model_dump() == {"retrieve_ms": 40.0, "generate_ms": 900.0,
                                             "total_ms": 950.0}  # fmt: skip
```

`tests/unit/test_api_security.py`:

```python
"""Tests for api.security: the API key check and the rate limiter."""

from rag_eval_platform.api.security import RateLimiter, check_api_key


def test_no_expected_key_means_everyone_is_allowed() -> None:
    assert check_api_key(None, None)
    assert check_api_key("anything", None)


def test_the_key_must_match_exactly() -> None:
    assert check_api_key("s3cret", "s3cret")
    assert not check_api_key("s3cre", "s3cret")
    assert not check_api_key(None, "s3cret")


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_rate_limit_blocks_after_the_limit_then_slides() -> None:
    clock = Clock()
    limiter = RateLimiter(limit=2, window_s=60, clock=clock)

    assert limiter.allow("a") == (True, 0.0)
    clock.now += 10
    assert limiter.allow("a") == (True, 0.0)
    allowed, retry = limiter.allow("a")
    assert not allowed
    assert retry == 50.0  # the first request leaves the window in 50 s

    clock.now += 50.001
    assert limiter.allow("a")[0]  # the window has slid


def test_clients_are_limited_separately() -> None:
    limiter = RateLimiter(limit=1, window_s=60, clock=Clock())

    assert limiter.allow("a")[0]
    assert not limiter.allow("a")[0]
    assert limiter.allow("b")[0]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_api_schemas.py tests/unit/test_api_security.py -q`
Expected: collection errors, `No module named 'rag_eval_platform.api.schemas'`.

- [ ] **Step 3: Implement**

`src/rag_eval_platform/api/schemas.py`:

```python
"""What the API accepts and returns (pydantic models), and the mapping from an Answer."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from rag_eval_platform.generation.prompt_templates import AnswerStyle
from rag_eval_platform.pipeline import TimedAnswer

MAX_QUESTION_CHARS = 2000
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                            max_length=MAX_QUESTION_CHARS)]  # fmt: skip


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    answer_style: AnswerStyle = "concise"


class CitationOut(BaseModel):
    number: int
    doc_id: str
    page: int | None
    chunk_id: str


class SourceOut(BaseModel):
    number: int
    doc_id: str
    page: int | None
    chunk_id: str
    score: float
    text: str


class Timings(BaseModel):
    retrieve_ms: float
    generate_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    refusal: bool
    citations: list[CitationOut]
    invalid_citations: list[int]
    sources: list[SourceOut]
    model: str
    prompt_version: str
    input_tokens: int | None
    output_tokens: int | None
    timings: Timings


class CheckOut(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, CheckOut]


class ErrorResponse(BaseModel):
    error: str
    hint: str | None = None


def to_response(timed: TimedAnswer) -> QueryResponse:
    answer = timed.answer
    return QueryResponse(
        question=answer.question,
        answer=answer.text,
        refusal=answer.is_refusal,
        citations=[CitationOut(number=c.number, doc_id=c.source.chunk.doc_id,
                               page=c.source.chunk.page, chunk_id=c.source.chunk.id)
                   for c in answer.citations],
        invalid_citations=list(answer.invalid_citations),
        sources=[SourceOut(number=n, doc_id=s.chunk.doc_id, page=s.chunk.page,
                           chunk_id=s.chunk.id, score=s.score, text=s.chunk.text)
                 for n, s in enumerate(answer.sources, start=1)],
        model=answer.model,
        prompt_version=answer.prompt_version,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        timings=Timings(retrieve_ms=round(timed.retrieve_ms, 1),
                        generate_ms=round(answer.latency_ms, 1),
                        total_ms=round(timed.total_ms, 1)),
    )  # fmt: skip
```

`src/rag_eval_platform/api/security.py`:

```python
"""API key check and a per-client sliding-window rate limiter (no extra dependency)."""

import hmac
import time
from collections import deque
from collections.abc import Callable


def check_api_key(given: str | None, expected: str | None) -> bool:
    """True when no key is configured, or the given one matches (constant-time compare)."""
    if expected is None:
        return True
    return given is not None and hmac.compare_digest(given.encode(), expected.encode())


class RateLimiter:
    """At most ``limit`` requests per client in any ``window_s`` seconds."""

    def __init__(
        self, limit: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self._limit, self._window, self._clock = limit, window_s, clock
        self._seen: dict[str, deque[float]] = {}

    def allow(self, client: str) -> tuple[bool, float]:
        """(allowed, seconds to wait before the next request is allowed)."""
        now = self._clock()
        seen = self._seen.setdefault(client, deque())
        while seen and now - seen[0] >= self._window:
            seen.popleft()
        if len(seen) >= self._limit:
            return False, round(seen[0] + self._window - now, 3)
        seen.append(now)
        return True, 0.0
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_api_schemas.py tests/unit/test_api_security.py -q`
Expected: all pass.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/api/schemas.py src/rag_eval_platform/api/security.py tests/unit/test_api_schemas.py tests/unit/test_api_security.py
git commit -m "feat: API schemas, API key check and rate limiter"
```

---

### Task 4: Health checks and SSE formatting

**Files:**
- Create: `src/rag_eval_platform/api/health.py`, `src/rag_eval_platform/api/sse.py`
- Test: `tests/unit/test_api_health.py`, `tests/unit/test_api_sse.py`

**Interfaces:**
- Consumes: `CheckOut` (Task 3), `VectorStore` protocol.
- Produces: `HealthCheck = Callable[[], CheckOut]`; `chroma_check(store: VectorStore) -> HealthCheck`; `llm_check(base_url: str, model: str, fetch: Callable[[str], bytes] = _fetch) -> HealthCheck`; `sse_event(event: str, data: str) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_api_health.py`:

```python
"""Tests for api.health: checks that never raise."""

import json

from rag_eval_platform.api.health import chroma_check, llm_check
from rag_eval_platform.retrieval.vector_store import VectorStoreError


class Store:
    def __init__(self, count: int | None) -> None:
        self._count = count

    def count(self) -> int:
        if self._count is None:
            raise VectorStoreError("connection refused")
        return self._count


def test_chroma_check_reports_the_chunk_count_or_the_error() -> None:
    assert chroma_check(Store(40))().model_dump() == {"ok": True, "detail": "40 chunks"}
    down = chroma_check(Store(None))()
    assert not down.ok
    assert "connection refused" in down.detail


def models(*names: str) -> bytes:
    return json.dumps({"data": [{"id": n} for n in names]}).encode()


def test_llm_check_needs_the_configured_model() -> None:
    ok = llm_check("http://x/v1", "qwen3:8b", fetch=lambda url: models("qwen3:8b"))()
    missing = llm_check("http://x/v1", "qwen3:8b", fetch=lambda url: models("other"))()

    assert ok.ok
    assert not missing.ok
    assert "ollama pull qwen3:8b" in missing.detail


def test_llm_check_reports_an_unreachable_server() -> None:
    def refuse(url: str) -> bytes:
        raise OSError("connection refused")

    check = llm_check("http://x/v1", "qwen3:8b", fetch=refuse)()

    assert not check.ok
    assert "unreachable" in check.detail
```

`tests/unit/test_api_sse.py`:

```python
"""Tests for api.sse: server-sent event formatting."""

from rag_eval_platform.api.sse import sse_event


def test_one_event_per_block_and_one_data_line_per_text_line() -> None:
    assert sse_event("token", "Hello") == "event: token\ndata: Hello\n\n"
    assert sse_event("token", "a\nb") == "event: token\ndata: a\ndata: b\n\n"
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_api_health.py tests/unit/test_api_sse.py -q`
Expected: `No module named 'rag_eval_platform.api.health'`.

- [ ] **Step 3: Implement**

`src/rag_eval_platform/api/health.py`:

```python
"""Health checks for /health: each returns a CheckOut and never raises."""

import json
import urllib.request
from collections.abc import Callable

from rag_eval_platform.api.schemas import CheckOut
from rag_eval_platform.retrieval.vector_store import VectorStore, VectorStoreError

HealthCheck = Callable[[], CheckOut]


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310 - configured URL
        data: bytes = response.read()
        return data


def chroma_check(store: VectorStore) -> HealthCheck:
    def check() -> CheckOut:
        try:
            return CheckOut(ok=True, detail=f"{store.count():,} chunks")
        except VectorStoreError as exc:
            return CheckOut(ok=False, detail=f"Chroma unavailable: {exc}")

    return check


def llm_check(base_url: str, model: str, fetch: Callable[[str], bytes] = _fetch) -> HealthCheck:
    """The OpenAI-compatible server lists the configured model (Ollama or OpenAI)."""

    def check() -> CheckOut:
        try:
            listed = {m["id"] for m in json.loads(fetch(base_url.rstrip("/") + "/models"))["data"]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return CheckOut(ok=False, detail=f"LLM server unreachable at {base_url}: {exc}")
        if model not in listed:
            return CheckOut(ok=False, detail=f"model {model} not found; run: ollama pull {model}")
        return CheckOut(ok=True, detail=model)

    return check
```

`src/rag_eval_platform/api/sse.py`:

```python
"""Server-sent events: the text/event-stream format used by /query/stream."""


def sse_event(event: str, data: str) -> str:
    lines = "".join(f"data: {line}\n" for line in data.split("\n"))
    return f"event: {event}\n{lines}\n"
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_api_health.py tests/unit/test_api_sse.py -q`
Expected: all pass.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/api/health.py src/rag_eval_platform/api/sse.py tests/unit/test_api_health.py tests/unit/test_api_sse.py
git commit -m "feat: API health checks and server-sent event formatting"
```

---

### Task 5: The app, routes and server entry point

**Files:**
- Modify: `src/rag_eval_platform/api/main.py`, `src/rag_eval_platform/api/routes.py`
- Create: `src/rag_eval_platform/api/services.py`, `src/rag_eval_platform/api/serve.py`, `scripts/serve_api.py`
- Test: `tests/unit/test_api_app.py`, `tests/integration/test_api_server.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces: `ApiServices(pipelines: Mapping[AnswerStyle, QueryPipeline], checks: Mapping[str, HealthCheck], api_key: str | None, limiter: RateLimiter)`; `QueryPipeline` protocol (`ask_timed(question) -> TimedAnswer`, `ask_stream(question) -> AnswerStream`); `create_app(services) -> FastAPI`; `create_app_from_settings(settings) -> FastAPI`; `serve.main() -> None`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_api_app.py`:

```python
"""Tests for the API app through FastAPI's TestClient, with fake pipelines."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from tests.unit.test_api_schemas import SOURCE

from rag_eval_platform.api.main import ApiServices, create_app
from rag_eval_platform.api.schemas import CheckOut
from rag_eval_platform.api.security import RateLimiter
from rag_eval_platform.generation.generator import Answer, Citation, GenerationError
from rag_eval_platform.pipeline import TimedAnswer
from rag_eval_platform.retrieval.vector_store import VectorStoreError


def answer(text: str = "F equals ma [1].") -> Answer:
    return Answer(question="q?", text=text, citations=(Citation(1, SOURCE),),
                  invalid_citations=(), sources=(SOURCE,), model="fake", prompt_version="v1",
                  input_tokens=10, output_tokens=4, latency_ms=20.0)  # fmt: skip


class FakeStream:
    def __init__(self, pieces: list[str], fail: bool = False) -> None:
        self.pieces, self.fail = pieces, fail
        self.answer = answer("".join(pieces))

    def __iter__(self) -> Iterator[str]:
        yield from self.pieces
        if self.fail:
            raise GenerationError("model crashed")


class FakePipeline:
    def __init__(self, error: Exception | None = None, fail_stream: bool = False) -> None:
        self.error, self.fail_stream = error, fail_stream
        self.asked: list[str] = []

    def ask_timed(self, question: str) -> TimedAnswer:
        self.asked.append(question)
        if self.error:
            raise self.error
        return TimedAnswer(answer(), 5.0, 30.0)

    def ask_stream(self, question: str) -> FakeStream:
        if self.error:
            raise self.error
        return FakeStream(["F equals ", "ma [1]."], fail=self.fail_stream)


def client(pipeline: FakePipeline | None = None, *, key: str | None = None, limit: int = 30,
           healthy: bool = True) -> TestClient:  # fmt: skip
    pipe = pipeline or FakePipeline()
    services = ApiServices(
        pipelines={"concise": pipe, "detailed": pipe},
        checks={"chroma": lambda: CheckOut(ok=True, detail="40 chunks"),
                "llm": lambda: CheckOut(ok=healthy, detail="qwen3:8b" if healthy else "down")},
        api_key=key, limiter=RateLimiter(limit),
    )  # fmt: skip
    return TestClient(create_app(services))


def test_health_is_ok_or_503_and_needs_no_key() -> None:
    ok = client(key="k").get("/health")
    degraded = client(healthy=False).get("/health")

    assert ok.status_code == 200
    assert ok.json()["status"] == "ok"
    assert degraded.status_code == 503
    assert degraded.json()["checks"]["llm"] == {"ok": False, "detail": "down"}


def test_query_returns_the_cited_answer() -> None:
    pipeline = FakePipeline()

    response = client(pipeline).post("/query", json={"question": "  What is F?  "})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "F equals ma [1]."
    assert body["citations"][0]["doc_id"] == "book.pdf"
    assert body["timings"] == {"retrieve_ms": 5.0, "generate_ms": 20.0, "total_ms": 30.0}
    assert pipeline.asked == ["What is F?"]


@pytest.mark.parametrize("body", [{"question": "   "}, {"question": "q", "extra": 1}, {}])
def test_invalid_requests_are_422_without_calling_the_model(body: dict[str, object]) -> None:
    pipeline = FakePipeline()

    response = client(pipeline).post("/query", json=body)

    assert response.status_code == 422
    assert pipeline.asked == []


def test_api_key_is_required_when_configured() -> None:
    api = client(key="s3cret")

    assert api.post("/query", json={"question": "q"}).status_code == 401
    assert (
        api.post("/query", json={"question": "q"}, headers={"X-API-Key": "nope"}).status_code == 401
    )
    assert (
        api.post("/query", json={"question": "q"}, headers={"X-API-Key": "s3cret"}).status_code
        == 200
    )


def test_rate_limit_answers_429_with_retry_after() -> None:
    api = client(limit=1)

    assert api.post("/query", json={"question": "q"}).status_code == 200
    limited = api.post("/query", json={"question": "q"})
    assert limited.status_code == 429
    assert int(limited.headers["Retry-After"]) >= 1


@pytest.mark.parametrize(("error", "status"), [
    (GenerationError("LLM down. Hint: ollama serve"), 502),
    (VectorStoreError("Chroma down"), 503),
])  # fmt: skip
def test_backend_failures_become_clean_json_errors(error: Exception, status: int) -> None:
    response = client(FakePipeline(error=error)).post("/query", json={"question": "q"})

    assert response.status_code == status
    assert set(response.json()) == {"error", "hint"}
    assert "Traceback" not in response.text


def events(text: str) -> list[tuple[str, str]]:
    found = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        name = lines[0].removeprefix("event: ")
        data = "\n".join(line.removeprefix("data: ") for line in lines[1:])
        found.append((name, data))
    return found


def test_stream_sends_status_tokens_then_done() -> None:
    response = client().post("/query/stream", json={"question": "q"})

    assert response.headers["content-type"].startswith("text/event-stream")
    got = events(response.text)
    assert [name for name, _ in got] == ["status", "status", "token", "token", "done"]
    assert got[2][1] == "F equals "
    assert json.loads(got[-1][1])["answer"] == "F equals ma [1]."


def test_a_stream_that_fails_midway_ends_with_an_error_event() -> None:
    response = client(FakePipeline(fail_stream=True)).post("/query/stream", json={"question": "q"})

    got = events(response.text)
    assert got[-1][0] == "error"
    assert "model crashed" in json.loads(got[-1][1])["error"]
    assert "Traceback" not in response.text
```

(Wrap the three long lines of `test_api_key_is_required_when_configured` to stay under 100 characters.)

`tests/integration/test_api_server.py`:

```python
"""The real app from settings, against Chroma (skips without Chroma or the extra)."""

import chromadb
import pytest
from fastapi.testclient import TestClient

from rag_eval_platform.api.main import create_app_from_settings
from rag_eval_platform.config.settings import Settings

pytestmark = pytest.mark.integration


def test_health_reports_each_service() -> None:
    settings = Settings()
    try:
        chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port).heartbeat()
    except Exception:
        pytest.skip("no Chroma server")
    pytest.importorskip("sentence_transformers")

    response = TestClient(create_app_from_settings(settings)).get("/health")

    assert response.status_code in (200, 503)
    assert set(response.json()["checks"]) == {"chroma", "llm"}
    assert response.json()["checks"]["chroma"]["ok"] is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_api_app.py -q`
Expected: `cannot import name 'ApiServices' from 'rag_eval_platform.api.main'`.

- [ ] **Step 3: Implement**

`src/rag_eval_platform/api/services.py`:

```python
"""What the API needs at run time, injected so tests can use fakes."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from fastapi import HTTPException, Request

from rag_eval_platform.api.health import HealthCheck
from rag_eval_platform.api.security import RateLimiter, check_api_key
from rag_eval_platform.generation.generator import Answer
from rag_eval_platform.generation.prompt_templates import AnswerStyle
from rag_eval_platform.pipeline import TimedAnswer


class _AnswerStream(Iterable[str], Protocol):
    @property
    def answer(self) -> Answer: ...


class QueryPipeline(Protocol):
    def ask_timed(self, question: str) -> TimedAnswer: ...

    def ask_stream(self, question: str) -> _AnswerStream: ...


@dataclass(frozen=True)
class ApiServices:
    pipelines: Mapping[AnswerStyle, QueryPipeline]
    checks: Mapping[str, HealthCheck]
    api_key: str | None
    limiter: RateLimiter


def services_of(request: Request) -> ApiServices:
    services: ApiServices = request.app.state.services
    return services


def guard(request: Request) -> None:
    """Rate limit first, then the API key (both skipped for /health and the docs)."""
    services = services_of(request)
    client = request.client.host if request.client else "unknown"
    allowed, retry_after = services.limiter.allow(client)
    if not allowed:
        raise HTTPException(429, detail="Too many requests; slow down",
                            headers={"Retry-After": str(max(1, round(retry_after)))})  # fmt: skip
    if not check_api_key(request.headers.get("X-API-Key"), services.api_key):
        raise HTTPException(401, detail="Missing or wrong X-API-Key header")
```

`src/rag_eval_platform/api/routes.py`:

```python
"""API routes: /health, /query and /query/stream."""

import json
import logging
import time
from collections.abc import Iterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from rag_eval_platform.api.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    to_response,
)
from rag_eval_platform.api.services import QueryPipeline, guard, services_of
from rag_eval_platform.api.sse import sse_event
from rag_eval_platform.pipeline import TimedAnswer

logger = logging.getLogger(__name__)
router = APIRouter()
_ERRORS = {code: {"model": ErrorResponse} for code in (401, 422, 429, 502, 503)}


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(request: Request) -> JSONResponse:
    checks = {name: check() for name, check in services_of(request).checks.items()}
    ok = all(c.ok for c in checks.values())
    body = HealthResponse(status="ok" if ok else "degraded", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)


@router.post("/query", response_model=QueryResponse, responses=_ERRORS,
             dependencies=[Depends(guard)])  # fmt: skip
def query(body: QueryRequest, request: Request) -> QueryResponse:
    pipeline = services_of(request).pipelines[body.answer_style]
    return to_response(pipeline.ask_timed(body.question))


@router.post("/query/stream", responses={200: {"content": {"text/event-stream": {}}}, **_ERRORS},
             dependencies=[Depends(guard)])  # fmt: skip
def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    pipeline = services_of(request).pipelines[body.answer_style]
    return StreamingResponse(_events(pipeline, body.question), media_type="text/event-stream")


def _events(pipeline: QueryPipeline, question: str) -> Iterator[str]:
    started = time.perf_counter()
    try:
        yield sse_event("status", "retrieving")
        stream = pipeline.ask_stream(question)
        retrieve_ms = (time.perf_counter() - started) * 1000
        yield sse_event("status", "generating")
        for piece in stream:
            yield sse_event("token", piece)
        timed = TimedAnswer(stream.answer, retrieve_ms, (time.perf_counter() - started) * 1000)
        yield sse_event("done", to_response(timed).model_dump_json())
    except Exception as exc:  # the response has started: report the failure as an event
        logger.warning("Stream failed", extra={"error": str(exc)})
        yield sse_event("error", json.dumps(ErrorResponse(error=str(exc)).model_dump()))
```

`src/rag_eval_platform/api/main.py`:

```python
"""FastAPI application: create_app wires injected services; create_app_from_settings the real ones."""

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.api.health import chroma_check, llm_check
from rag_eval_platform.api.routes import router
from rag_eval_platform.api.schemas import ErrorResponse
from rag_eval_platform.api.security import RateLimiter
from rag_eval_platform.api.services import ApiServices
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import GenerationError, create_generator
from rag_eval_platform.pipeline import RagPipeline
from rag_eval_platform.retrieval.retriever import create_retriever
from rag_eval_platform.retrieval.vector_store import VectorStoreError, create_vector_store

logger = logging.getLogger(__name__)

_FAILURES: dict[type[Exception], tuple[int, str]] = {
    GenerationError: (502, "Is Ollama running with the model pulled? See GET /health."),
    VectorStoreError: (503, "Start Chroma: docker compose -f docker/docker-compose.yml up -d"),
    OptionalDependencyError: (503, "Install the extras: uv sync --all-extras"),
}


def _error(status: int, error: str, hint: str | None = None,
           headers: dict[str, str] | None = None) -> JSONResponse:  # fmt: skip
    body = ErrorResponse(error=error, hint=hint).model_dump()
    return JSONResponse(body, status_code=status, headers=headers)


def create_app(services: ApiServices) -> FastAPI:
    app = FastAPI(title="RAG Eval Platform API", version="1.0.0",
                  description="Cited answers from your documents: retrieve, then generate.")  # fmt: skip
    app.state.services = services
    app.include_router(router)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error(422, f"Invalid request: {where} {first.get('msg', '')}".strip(),
                      hint="Send {\"question\": \"...\", \"answer_style\": \"concise\"}")  # fmt: skip

    for failure, (status, hint) in _FAILURES.items():

        def handler(request: Request, exc: Exception, status: int = status,
                    hint: str = hint) -> JSONResponse:  # fmt: skip
            logger.warning("Query failed", extra={"error": str(exc), "status": status})
            return _error(status, str(exc), hint)

        app.add_exception_handler(failure, handler)
    return app


def create_app_from_settings(settings: Settings) -> FastAPI:
    """The real services: one retriever (models load once) shared by both answer styles."""
    retriever = create_retriever(settings)
    pipelines = {style: RagPipeline(retriever, create_generator(settings, style))
                 for style in ("concise", "detailed")}  # fmt: skip
    key = settings.api_key.get_secret_value() if settings.api_key else None
    services = ApiServices(
        pipelines=pipelines,
        checks={"chroma": chroma_check(create_vector_store(settings)), "llm": _llm_check(settings)},
        api_key=key, limiter=RateLimiter(settings.api_rate_limit),
    )  # fmt: skip
    return create_app(services)
```

and, in the same module:

```python
def _llm_check(settings: Settings) -> HealthCheck:
    """Ollama must list the model; a hosted provider is not probed (it needs a paid call)."""
    if settings.llm_provider == "ollama":
        return llm_check(settings.ollama_base_url, settings.llm_model)
    return lambda: CheckOut(ok=True, detail=f"{settings.llm_provider} (not probed)")
```

(importing `HealthCheck` from `api.health` and `CheckOut` from `api.schemas`).

`src/rag_eval_platform/api/serve.py`:

```python
"""Run the API with Uvicorn: ``uv run python scripts/serve_api.py``."""

import uvicorn

from rag_eval_platform.api.main import create_app_from_settings
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import get_settings


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(create_app_from_settings(settings), host=settings.api_host,
                port=settings.api_port, log_config=None)  # fmt: skip
```

`scripts/serve_api.py`:

```python
"""Serve the RAG API on RAG_API_HOST:RAG_API_PORT (default 127.0.0.1:8000); docs at /docs.

Usage: uv run python scripts/serve_api.py   (Chroma running; --all-extras for the models)
"""

from rag_eval_platform.api.serve import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_api_app.py -q && uv run pytest tests/integration/test_api_server.py -m integration -q`
Expected: unit tests pass. The integration test passes with Chroma running, or skips without it.

- [ ] **Step 5: Try it for real**

Run: `uv run python scripts/serve_api.py` (in the background), then `curl -s localhost:8000/health` and `curl -s -X POST localhost:8000/query -H 'content-type: application/json' -d '{"question":"What does MRR measure?"}'`.
Expected: `/health` shows `{"status":"ok",...}` with both checks, and `/query` returns a cited answer.

- [ ] **Step 6: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/api scripts/serve_api.py tests/unit/test_api_app.py tests/integration/test_api_server.py
git commit -m "feat: FastAPI app with /health, /query and /query/stream"
```

---

### Task 6: Docker image, compose service and CI job

**Files:**
- Modify: `docker/Dockerfile`, `docker/docker-compose.yml`, `.github/workflows/ci.yml`
- Create: `.dockerignore`

- [ ] **Step 1: Write `docker/Dockerfile`**

```dockerfile
# syntax=docker/dockerfile:1
# The RAG API: multi-stage uv build, CPU-only torch (see pyproject), non-root, model baked in.
# Build from the repo root: docker build -f docker/Dockerfile -t rag-eval-api .

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 HF_HOME=/app/.hf
WORKDIR /app
# Dependencies first, so code changes don't reinstall torch.
COPY pyproject.toml uv.lock readme.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra local-embeddings
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra local-embeddings
# Download the embedding model now, so the container needs no internet.
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.hf HF_HUB_OFFLINE=1 \
    RAG_API_HOST=0.0.0.0 RAG_API_PORT=8000 PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "-m", "rag_eval_platform.api.serve"]
```

For `python -m` to work, add to `api/serve.py`:

```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `.dockerignore`**

```
.git
.venv
.env
.env.*
!.env.example
data/playground
data/processed
reports
notebooks
.superpowers
**/__pycache__
.pytest_cache
.mypy_cache
.ruff_cache
.coverage
```

- [ ] **Step 3: Add the compose service** (in `docker/docker-compose.yml`, under `services:`)

```yaml
  api:
    # The RAG API (Phase 6). Start with:
    #   docker compose -f docker/docker-compose.yml --profile api up -d --build
    # Ollama runs natively on the host (GPU); the container reaches it through
    # host.docker.internal. Seed Chroma first (scripts/seed_vector_store.py).
    build:
      context: ..
      dockerfile: docker/Dockerfile
    profiles: ["api"]
    ports:
      - "127.0.0.1:${API_HOST_PORT:-8000}:8000"
    environment:
      RAG_CHROMA_HOST: chroma
      RAG_CHROMA_PORT: "8000"
      RAG_OLLAMA_BASE_URL: http://host.docker.internal:11434/v1
      RAG_API_KEY: ${RAG_API_KEY:-}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      chroma:
        condition: service_healthy
    restart: unless-stopped
```

- [ ] **Step 4: Validate locally**

Run: `docker compose -f docker/docker-compose.yml --profile api config >/dev/null && echo ok`
Expected: `ok`.

Run: `docker build -f docker/Dockerfile -t rag-eval-api . && docker compose -f docker/docker-compose.yml --profile api up -d && sleep 30 && curl -s localhost:8000/health`
Expected: the build succeeds (a few minutes the first time), and `/health` returns JSON with `chroma` ok and `llm` ok (Ollama is on the host). Stop the API with `docker compose -f docker/docker-compose.yml --profile api stop api` afterwards, leaving Chroma running. If the machine is short of memory, report it and skip the run, keeping the build.

- [ ] **Step 5: Add the CI job** (in `.github/workflows/ci.yml`, a new job)

```yaml
  api-image:
    name: API image
    runs-on: ubuntu-latest
    services:
      chroma:
        image: chromadb/chroma:1.5.9
        ports:
          - 8001:8000
    steps:
      - uses: actions/checkout@v7

      - uses: docker/setup-buildx-action@v4

      - name: Build the image
        uses: docker/build-push-action@v7
        with:
          context: .
          file: docker/Dockerfile
          tags: rag-eval-api:ci
          load: true
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Run it and check /health
        # No Ollama in CI, so /health is expected to be 503 with the llm check failing.
        run: |
          docker run -d --name api --network host \
            -e RAG_CHROMA_HOST=localhost -e RAG_CHROMA_PORT=8001 rag-eval-api:ci
          for i in $(seq 1 45); do
            code=$(curl -s -o health.json -w '%{http_code}' localhost:8000/health || true)
            if [ "$code" = 200 ] || [ "$code" = 503 ]; then break; fi
            sleep 2
          done
          cat health.json
          python3 -c "import json; b=json.load(open('health.json')); assert b['checks']['chroma']['ok'], b; assert 'llm' in b['checks'], b"

      - name: Container logs
        if: failure()
        run: docker logs api
```

- [ ] **Step 6: Commit (when the user asks)**

```bash
git add docker/Dockerfile .dockerignore docker/docker-compose.yml .github/workflows/ci.yml src/rag_eval_platform/api/serve.py
git commit -m "build: API Docker image, compose api profile and CI image check"
```

---

### Task 7: Playground API view (pure)

**Files:**
- Create: `src/rag_eval_platform/playground/api_view.py`
- Test: `tests/unit/test_api_view.py`

**Interfaces:**
- Produces: `API_URL = "http://127.0.0.1:8000"`; `curl_command(base_url: str, body: Mapping[str, object], *, api_key: str | None, stream: bool = False, reveal_key: bool = False) -> str`; `SseEvent(event: str, data: str)`; `parse_sse(lines: Iterable[str]) -> Iterator[SseEvent]`; `HttpResult(status: int, body: str, ms: float)`; `post_json(url, body, api_key, timeout=120, opener=urllib.request.urlopen) -> HttpResult`; `get_json(url, timeout=3, opener=...) -> HttpResult`; `stream_events(url, body, api_key, opener=...) -> Iterator[SseEvent]`; `api_flow_dot(timings: Mapping[str, float] | None) -> str`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_api_view.py`:

```python
"""Tests for playground.api_view: the ⑦ API tab's pure pieces."""

import io
import json
import urllib.error
from typing import Any

from rag_eval_platform.playground.api_view import (
    SseEvent,
    api_flow_dot,
    curl_command,
    get_json,
    parse_sse,
    post_json,
    stream_events,
)


def test_curl_command_quotes_the_body_and_masks_the_key() -> None:
    body = {"question": "What's MRR?", "answer_style": "concise"}

    shown = curl_command("http://127.0.0.1:8000", body, api_key="s3cret-long-key")
    real = curl_command("http://127.0.0.1:8000", body, api_key="s3cret-long-key", reveal_key=True)

    assert shown.startswith("curl -s -X POST http://127.0.0.1:8000/query")
    assert "s3cret-long-key" not in shown
    assert "X-API-Key: s3cr…" in shown
    assert "X-API-Key: s3cret-long-key" in real
    assert "'\"'\"'" in shown  # the apostrophe in What's is shell-quoted
    assert "/query/stream" in curl_command("http://x", body, api_key=None, stream=True)
    assert "X-API-Key" not in curl_command("http://x", body, api_key=None)


def test_parse_sse_joins_data_lines_per_event() -> None:
    lines = ["event: status\n", "data: retrieving\n", "\n", "event: token\n", "data: a\n",
             "data: b\n", "\n"]  # fmt: skip

    assert list(parse_sse(lines)) == [SseEvent("status", "retrieving"), SseEvent("token", "a\nb")]


class Response(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200) -> None:
        super().__init__(body)
        self.status = status

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_post_json_returns_status_body_and_time() -> None:
    sent: list[Any] = []

    def opener(request: Any, timeout: float) -> Response:
        sent.append(request)
        return Response(b'{"answer": "x"}')

    result = post_json("http://x/query", {"question": "q"}, "k", opener=opener)

    assert (result.status, json.loads(result.body)) == (200, {"answer": "x"})
    assert result.ms >= 0
    assert sent[0].get_header("X-api-key") == "k"


def test_http_errors_keep_their_status_and_body() -> None:
    def opener(request: Any, timeout: float) -> Response:
        raise urllib.error.HTTPError(
            "http://x",
            429,
            "Too Many",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error": "slow down"}'),
        )

    result = post_json("http://x/query", {"question": "q"}, None, opener=opener)

    assert result.status == 429
    assert "slow down" in result.body


def test_an_unreachable_api_is_status_0() -> None:
    def opener(request: Any, timeout: float) -> Response:
        raise urllib.error.URLError("connection refused")

    assert get_json("http://x/health", opener=opener).status == 0


def test_stream_events_reads_the_event_stream() -> None:
    def opener(request: Any, timeout: float) -> Response:
        return Response(b"event: token\ndata: Hi\n\nevent: done\ndata: {}\n\n")

    events = list(stream_events("http://x/query/stream", {"question": "q"}, None, opener=opener))

    assert events == [SseEvent("token", "Hi"), SseEvent("done", "{}")]


def test_flow_diagram_shows_each_stage_time_after_a_request() -> None:
    assert "ms" not in api_flow_dot(None)
    dot = api_flow_dot({"retrieve_ms": 38.0, "generate_ms": 4200.0, "total_ms": 4300.0})
    assert "38 ms" in dot
    assert "4.2 s" in dot
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_api_view.py -q`
Expected: `No module named 'rag_eval_platform.playground.api_view'`.

- [ ] **Step 3: Implement** `src/rag_eval_platform/playground/api_view.py`:

```python
"""Pure pieces of the ⑦ API tab: the curl command, HTTP calls and the flow diagram.

Plain urllib (no new dependency); ``opener`` is injectable so tests need no server.
"""

import json
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

API_URL = "http://127.0.0.1:8000"
Opener = Callable[..., Any]


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: str


@dataclass(frozen=True)
class HttpResult:
    status: int  # 0 when the API could not be reached
    body: str
    ms: float


def curl_command(
    base_url: str, body: Mapping[str, object], *, api_key: str | None, stream: bool = False,
    reveal_key: bool = False,
) -> str:  # fmt: skip
    """The same request as a curl command; the key is masked unless ``reveal_key``."""
    path = "/query/stream" if stream else "/query"
    parts = ["curl", "-s", "-N"] if stream else ["curl", "-s"]
    parts += ["-X", "POST", f"{base_url}{path}", "-H", "Content-Type: application/json"]
    if api_key:
        shown = api_key if reveal_key else api_key[:4] + "…"
        parts += ["-H", f"X-API-Key: {shown}"]
    parts += ["-d", json.dumps(dict(body))]
    return " ".join(shlex.quote(p) if i else p for i, p in enumerate(parts))


def parse_sse(lines: Iterable[str]) -> Iterator[SseEvent]:
    event, data = "message", []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            if data:
                yield SseEvent(event, "\n".join(data))
            event, data = "message", []
        elif line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data.append(line.removeprefix("data: "))
    if data:
        yield SseEvent(event, "\n".join(data))


def _request(url: str, body: Mapping[str, object] | None, api_key: str | None) -> Any:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    data = json.dumps(dict(body)).encode() if body is not None else None
    return urllib.request.Request(url, data=data, headers=headers,
                                  method="POST" if body is not None else "GET")  # fmt: skip


def _call(request: Any, timeout: float, opener: Opener) -> HttpResult:
    started = time.perf_counter()
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read().decode()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        body, status = exc.read().decode(), exc.code
    except (urllib.error.URLError, OSError) as exc:
        body, status = str(exc), 0
    return HttpResult(status, body, (time.perf_counter() - started) * 1000)


def post_json(url: str, body: Mapping[str, object], api_key: str | None, timeout: float = 120,
              opener: Opener = urllib.request.urlopen) -> HttpResult:  # fmt: skip
    return _call(_request(url, body, api_key), timeout, opener)


def get_json(url: str, timeout: float = 3, opener: Opener = urllib.request.urlopen) -> HttpResult:
    return _call(_request(url, None, None), timeout, opener)


def stream_events(url: str, body: Mapping[str, object], api_key: str | None,
                  opener: Opener = urllib.request.urlopen) -> Iterator[SseEvent]:  # fmt: skip
    """The events of /query/stream as they arrive (the response is read line by line)."""
    with opener(_request(url, body, api_key), timeout=300) as response:
        yield from parse_sse(line.decode() for line in response)


def api_flow_dot(timings: Mapping[str, float] | None) -> str:
    """Client → FastAPI (auth, rate limit) → retrieve → generate → JSON, with times if known."""

    def took(ms: float) -> str:
        return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"

    retrieve = f"\\n{took(timings['retrieve_ms'])}" if timings else ""
    generate = f"\\n{took(timings['generate_ms'])}" if timings else ""
    total = f"total {took(timings['total_ms'])}" if timings else ""
    return "\n".join([
        "digraph api {", '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 '
        'fillcolor="#eef2ff" color="#6366f1"];',
        '  edge [color="#9ca3af" fontname="Helvetica" fontsize=9 fontcolor="#6b7280"];',
        '  client [label="Client\\ncurl · app"];',
        '  api [label="FastAPI\\nkey · rate limit"];',
        f'  retrieve [label="Retrieve{retrieve}"];',
        f'  generate [label="Generate{generate}"];',
        '  json [label="JSON\\ncited answer" fillcolor="#dcfce7" color="#16a34a"];',
        f'  client -> api [label="POST /query"]; api -> retrieve; retrieve -> generate;'
        f' generate -> json [label="{total}"];',
        "}",
    ])  # fmt: skip
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_api_view.py -q`
Expected: all pass.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/playground/api_view.py tests/unit/test_api_view.py
git commit -m "feat: playground API view logic: curl command, HTTP calls, SSE parser, flow"
```

---

### Task 8: The ⑦ API tab

**Files:**
- Create: `src/rag_eval_platform/playground/tabs/api.py`
- Modify: `src/rag_eval_platform/playground/shared.py` (`TAB_LABELS`), `src/rag_eval_platform/playground/app.py` (`RENDERERS`), `tests/integration/test_playground.py`

- [ ] **Step 1: Update the smoke test first (red):** append `"⑦ API"` to the expected tab labels in `tests/integration/test_playground.py`.

Run: `uv run pytest tests/integration/test_playground.py::test_app_renders_without_errors -q`
Expected: FAIL, the labels list is missing `"⑦ API"`.

- [ ] **Step 2: Wire the tab.** Add `"⑦ API"` to `TAB_LABELS`, then `api` to the tabs import in `app.py` and `"⑦ API": api.render` to `RENDERERS`.

`src/rag_eval_platform/playground/tabs/api.py`:

```python
"""⑦ API: the same pipeline over HTTP — status, a request builder, curl, and live streaming."""

import json

import streamlit as st

from rag_eval_platform.playground.api_view import (
    API_URL,
    api_flow_dot,
    curl_command,
    get_json,
    post_json,
    stream_events,
)
from rag_eval_platform.playground.header import Pill, header_html
from rag_eval_platform.playground.shared import TabContext, settings

API_LAST = "api_last"  # (status, body, ms) of the last request, kept across reruns


def render(ctx: TabContext) -> None:
    health = get_json(f"{API_URL}/health")
    up = health.status in (200, 503)
    label = f"API up at {API_URL}" if up else "API not running"
    st.markdown(header_html((Pill(label, ok=health.status == 200),), ()), unsafe_allow_html=True)
    if not up:
        st.info("Start it with `uv run python scripts/serve_api.py`, or in Docker: "
                "`docker compose -f docker/docker-compose.yml --profile api up -d --build`.")  # fmt: skip
    last = st.session_state.get(API_LAST)
    timings = json.loads(last[1]).get("timings") if last and last[0] == 200 else None
    st.graphviz_chart(api_flow_dot(timings), width="stretch")
    left, right = st.columns([1, 1], gap="large")
    with left:
        _builder(up)
    with right:
        _response()


def _builder(up: bool) -> None:
    question = st.text_area("Question", "What does MRR measure?", key="api-question")
    style = st.segmented_control("Answer style", ["concise", "detailed"], default="concise",
                                 key="api-style", required=True)  # fmt: skip
    stream = st.toggle("Stream (/query/stream)", key="api-stream")
    key = settings.api_key.get_secret_value() if settings.api_key else None
    body = {"question": question, "answer_style": style}
    st.code(curl_command(API_URL, body, api_key=key, stream=stream), language="bash")
    st.caption(f"Interactive docs: {API_URL}/docs")
    if st.button("Send", type="primary", disabled=not up, key="api-send"):
        if stream:
            _stream(body, key)
        else:
            result = post_json(f"{API_URL}/query", body, key)
            st.session_state[API_LAST] = (result.status, result.body, result.ms)
            st.rerun()


def _stream(body: dict[str, object], key: str | None) -> None:
    status, text, events = st.empty(), st.empty(), 0
    written = ""
    for event in stream_events(f"{API_URL}/query/stream", body, key):
        events += 1
        if event.event == "status":
            status.caption(f"event: status · {event.data}")
        elif event.event == "token":
            written += event.data
            text.markdown(written)
            status.caption(f"event: token · {events} events so far")
        elif event.event in ("done", "error"):
            # Saved before the next st.* call, so a click cannot lose it.
            st.session_state[API_LAST] = (200 if event.event == "done" else 500, event.data, 0.0)
            status.caption(f"event: {event.event}")
    st.rerun()


def _response() -> None:
    last = st.session_state.get(API_LAST)
    if last is None:
        st.info("Send a request to see the JSON response.")
        return
    status, body, ms = last
    colour = "green" if status == 200 else "red"
    timing = f" · {ms:.0f} ms" if ms else ""
    st.markdown(f"**Response** :{colour}[{status}]{timing}")
    try:
        st.json(json.loads(body), expanded=2)
    except ValueError:
        st.code(body)
```

- [ ] **Step 3: Run the checks**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit tests/integration/test_playground.py -q`
Expected: all clean, and the smoke test passes with `"⑦ API"`.

- [ ] **Step 4: Check it in the browser** (restart the app; start the API with `scripts/serve_api.py`). In ⑦ API: the pill shows "API up", **Send** returns 200 with JSON, and the flow shows the retrieve and generate times. Turn on **Stream** and watch the text arrive. Stop the API and check that the tab says "API not running" with the start command. Screenshot each state at 1440×900.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/playground/tabs/api.py src/rag_eval_platform/playground/shared.py src/rag_eval_platform/playground/app.py tests/integration/test_playground.py
git commit -m "feat: playground ⑦ API tab: status, request builder, curl, live stream"
```

---

### Task 9: Docs

- [ ] **Step 1: Write `docs/learning/phase-6.md`** in the style of the other learning pages. Mermaid must parse on GitHub: no `&quot;`, quote labels that contain punctuation. Sections:
  1. What an API is and why (the playground is for people; an API is for other programs).
  2. The request flow (`sequenceDiagram`: client → FastAPI → guard → retriever → generator → JSON).
  3. The endpoints, with a `curl` example and a trimmed JSON response for each.
  4. Streaming: a timeline of the `status`, `token` and `done` events.
  5. The container setup (`flowchart`: host Mac with Ollama, the Docker network with chroma and api, and ports 8000 and 8001).
  6. Security basics: validation, API key, rate limit, clean errors.
  7. What to try in ⑦ API.
- [ ] **Step 2: Update the other docs.**
  - `readme.md`: status table (Phase 6 ✅, Phase 7 🔜), the commands (`serve_api.py`, the compose profile), the ⑦ API row in the playground table, the tech table, the docs list.
  - `CLAUDE.md`: project status (API and Dockerfile no longer placeholders), the commands, an API bullet in Architecture.
  - `docs/architecture.md`: Phase 6 rows become *in use* (FastAPI, Uvicorn, httpx), and the Deployment section describes the image and profile.
  - `docs/learning/playground.md`: the ⑦ API row and a short section.
- [ ] **Step 3: Verify.** Run: `grep -rn "placeholder" CLAUDE.md | grep -i "api\|dockerfile"`. Expected: no match.
- [ ] **Step 4: Commit (when the user asks)**

```bash
git add docs/learning/phase-6.md readme.md CLAUDE.md docs/architecture.md docs/learning/playground.md docs/superpowers/specs/2026-09-28-phase-6-api-deployment-design.md docs/superpowers/plans/2026-09-28-phase-6-api-deployment.md
git commit -m "docs: phase 6 API and deployment guide, readme, spec and plan"
```

---

### Task 10: Final verification

- [ ] **Step 1:** Run: `uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit --cov -q && uv run pytest -m integration -q`
Expected: all green; coverage stays at or above 95%.
- [ ] **Step 2:** Start the API and run `curl -s -X POST localhost:8000/query -d '{"question":"   "}' -H 'content-type: application/json'`.
Expected: 422 with `{"error": ..., "hint": ...}`.
- [ ] **Step 3:** Send 31 requests in a row to `/query` with a bad key set (`RAG_API_KEY=x`).
Expected: 401 for the requests within the limit, then 429 with `Retry-After`. This confirms the rate limit runs before the key check.
- [ ] **Step 4:** Report to the user. Commit and push only when they ask.
