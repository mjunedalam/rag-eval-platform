# Phase 7a — Observability core (SQLite) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store every answered question, every 👍/👎 and every evaluation run in one local SQLite file, and show traffic, latency, cost, quality over time, drift warnings and a review queue in a new ⑧ Observe playground tab.

**Architecture:** `observability/store.py` owns the SQLite schema and rows; `observability/logger.py` turns an `Answer` into a row and never raises; `RagPipeline` (API, `ask.py`) and the playground record through it. Pure modules (`cost`, `review`, `drift`, `dashboard`) compute everything the API and the tab show, so they are unit-tested without Streamlit.

**Tech Stack:** Python 3.12 standard library `sqlite3`, pydantic-settings, FastAPI, Streamlit, Plotly. No new dependency.

**Spec:** `docs/superpowers/specs/2026-09-29-phase-7a-observability-core-design.md`

## Global Constraints

- No new dependency. `sqlite3` from the standard library only.
- Settings (all with defaults, validated): `RAG_OBSERVE_ENABLED=true`, `RAG_OBSERVE_DB=data/observability/rag.db`, `RAG_OBSERVE_STORE_TEXT=true`, `RAG_PRICE_INPUT_PER_1M=0`, `RAG_PRICE_OUTPUT_PER_1M=0`, `RAG_DRIFT_WINDOW=20`, `RAG_DRIFT_LATENCY_RISE=0.5`, `RAG_DRIFT_ERROR_RISE=0.10`, `RAG_DRIFT_REFUSAL_RISE=0.15`, `RAG_DRIFT_CITATION_RISE=0.10`, `RAG_DRIFT_DOWNVOTE_RISE=0.20`, `RAG_REVIEW_SLOW_MS=30000`. Window > 0, rises > 0, prices ≥ 0.
- Recording, drift for `/health`, and eval-run writes never break answering, evaluation or `/health`.
- SQLite: `journal_mode=DELETE`, `busy_timeout` 5 s (connect `timeout=5.0`), one short connection per operation, schema version in `PRAGMA user_version` (= 1).
- Golden-set evaluation never writes `queries`; each evaluation script writes one `eval_runs` row.
- Every API error body is `{error, hint}`. `/feedback` → 201, 404 unknown query, 503 disabled or store failure.
- `/health` warnings never change the 200/503 status.
- `data/observability/` is git-ignored and docker-ignored (it holds question text).
- CI checks: `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src tests` (strict), `uv run pytest tests/unit`, coverage ≥ 80%. Ruff line length 100.
- Never commit or push unless the user explicitly says so. Stage by name; never commit `.env` or `data/`.

## Review Focus

1. **Two writers at once** (the API container and the playground on the same file): no row is lost and neither side errors; `busy_timeout` absorbs the lock. Test in Task 2: two `ObserveStore` objects on one path, 2 threads × 25 inserts → 50 rows.
2. **Naive or non-UTC timestamps** from a caller: stored as UTC, sorted correctly. Test in Task 2.
3. **Long and non-ASCII text** (a 2,000-character question, emoji, Arabic, quotes) round-trips unchanged. Test in Task 2.
4. **Rating an answer that was never logged** (logging off or the write failed, so `query_id is None`): the playground click does nothing harmful and returns False. Test in Task 9.
5. **Windows full of errors** (no `total_ms` latency, no ok answers): drift reports no latency or refusal warning instead of crashing on empty lists. Test in Task 5.

---

## File structure

| File | Responsibility |
|---|---|
| Create `src/rag_eval_platform/observability/store.py` | Schema, row dataclasses, `ObserveStore` reads and writes, errors |
| Create `src/rag_eval_platform/observability/cost.py` | `Prices`, `cost_usd` |
| Replace `src/rag_eval_platform/observability/logger.py` | `QueryLogger`, `create_store`, `create_query_logger`, `record_feedback`, `record_eval_run` |
| Create `src/rag_eval_platform/observability/review.py` | `latest_feedback`, `review_queue` |
| Create `src/rag_eval_platform/observability/drift.py` | `DriftThresholds`, `percentile`, `drift_report` |
| Replace `src/rag_eval_platform/observability/dashboard.py` | Summary tiles, buckets, eval history, table rows |
| Modify `config/settings.py`, `.env.example`, `.gitignore`, `.dockerignore`, `tests/conftest.py` | Settings and isolation |
| Modify `pipeline.py`, `ask.py` | Recorder, `RecordedStream`, `TimedAnswer.query_id`, source `cli` |
| Modify `api/schemas.py`, `api/services.py`, `api/routes.py`, `api/main.py` | `query_id`, `/feedback`, health warnings |
| Modify `evaluation/fingerprint.py`, `evaluation/cli.py`, `evaluation/generation_cli.py`, `evaluation/gate_cli.py` | One `eval_runs` row per run |
| Modify `playground/core.py`, `playground/chat.py`, `playground/charts.py`, `playground/shared.py`, `playground/app.py`, `playground/tabs/overview.py`, `playground/tabs/chat_panel.py`; create `playground/tabs/observe.py` | Recording, persistent feedback, ⑧ tab |
| Modify `docker/Dockerfile`, `docker/docker-compose.yml` | Writable folder, bind mount |
| Docs | `docs/learning/phase-7.md` (new), readme, CLAUDE.md, architecture, playground |

Test helper shared by several tasks: `tests/unit/observe_records.py` (Task 2).

---

### Task 1: Settings and test isolation

**Files:**
- Modify: `src/rag_eval_platform/config/settings.py` (the `# Observability` block)
- Modify: `.env.example`, `.gitignore`, `.dockerignore`, `tests/conftest.py`
- Test: `tests/unit/test_settings.py`

**Interfaces:**
- Produces: `Settings.observe_enabled: bool`, `observe_db: Path`, `observe_store_text: bool`, `price_input_per_1m: float`, `price_output_per_1m: float`, `drift_window: int`, `drift_latency_rise`, `drift_error_rise`, `drift_refusal_rise`, `drift_citation_rise`, `drift_downvote_rise: float`, `review_slow_ms: float`.

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_settings.py`)

```python
def test_observability_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_OBSERVE_DB", raising=False)

    settings = Settings(_env_file=None)

    assert settings.observe_enabled is True
    assert settings.observe_db == Path("data/observability/rag.db")
    assert settings.observe_store_text is True
    assert (settings.price_input_per_1m, settings.price_output_per_1m) == (0.0, 0.0)
    assert settings.drift_window == 20
    assert settings.drift_latency_rise == 0.5
    assert settings.drift_error_rise == 0.10
    assert settings.drift_refusal_rise == 0.15
    assert settings.drift_citation_rise == 0.10
    assert settings.drift_downvote_rise == 0.20
    assert settings.review_slow_ms == 30000


def test_observability_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_OBSERVE_ENABLED", "false")
    monkeypatch.setenv("RAG_PRICE_INPUT_PER_1M", "0.15")
    monkeypatch.setenv("RAG_DRIFT_WINDOW", "5")

    settings = Settings(_env_file=None)

    assert settings.observe_enabled is False
    assert settings.price_input_per_1m == 0.15
    assert settings.drift_window == 5


@pytest.mark.parametrize(
    ("name", "value"),
    [("RAG_DRIFT_WINDOW", "0"), ("RAG_PRICE_OUTPUT_PER_1M", "-1"),
     ("RAG_DRIFT_ERROR_RISE", "0"), ("RAG_REVIEW_SLOW_MS", "0")],
)  # fmt: skip
def test_observability_settings_are_validated(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)
```

(Import `Path` and `ValidationError` at the top of the file if they are not already imported.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_settings.py -q -k observability`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'observe_enabled'`

- [ ] **Step 3: Implement** — replace the `# Observability` block in `settings.py`:

```python
    # Observability (Phase 7)
    log_level: str = "INFO"
    observe_enabled: bool = True
    observe_db: Path = Path("data/observability/rag.db")
    observe_store_text: bool = True  # false: store a SHA-256 of the question, no answer text
    price_input_per_1m: float = Field(default=0.0, ge=0.0)  # USD per 1M tokens (0 = local)
    price_output_per_1m: float = Field(default=0.0, ge=0.0)
    drift_window: int = Field(default=20, gt=0)  # answers per window: last N vs previous N
    drift_latency_rise: float = Field(default=0.5, gt=0)  # p95 may rise by 50%
    drift_error_rise: float = Field(default=0.10, gt=0)  # rates: absolute rise, 0.10 = 10 points
    drift_refusal_rise: float = Field(default=0.15, gt=0)
    drift_citation_rise: float = Field(default=0.10, gt=0)
    drift_downvote_rise: float = Field(default=0.20, gt=0)
    review_slow_ms: float = Field(default=30000.0, gt=0)
```

Append to `.env.example` (after `# RAG_API_RATE_LIMIT`):

```
# RAG_OBSERVE_ENABLED=true            # store every answer, rating and evaluation run
# RAG_OBSERVE_DB=data/observability/rag.db
# RAG_OBSERVE_STORE_TEXT=true         # false: keep only a hash of the question
# RAG_PRICE_INPUT_PER_1M=0            # USD per 1M input tokens (0 for local Ollama)
# RAG_PRICE_OUTPUT_PER_1M=0
# RAG_DRIFT_WINDOW=20                 # drift: last N answers vs the N before
# RAG_DRIFT_LATENCY_RISE=0.5
# RAG_DRIFT_ERROR_RISE=0.10
# RAG_DRIFT_REFUSAL_RISE=0.15
# RAG_DRIFT_CITATION_RISE=0.10
# RAG_DRIFT_DOWNVOTE_RISE=0.20
# RAG_REVIEW_SLOW_MS=30000
```

Append to `.gitignore`:

```
# Observability log: holds question text, never committed
data/observability/
```

Append `data/observability` to `.dockerignore` (after `data/processed`).

Append to `tests/conftest.py` (so no test ever writes the real log):

```python
from collections.abc import Iterator
from pathlib import Path

from rag_eval_platform.config.settings import get_settings


@pytest.fixture(autouse=True)
def _isolated_observe_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test logs to its own temporary database, never data/observability/."""
    monkeypatch.setenv("RAG_OBSERVE_DB", str(tmp_path / "observe.db"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
```

(Move the new imports to the top of `conftest.py`, next to `import pytest`.)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_settings.py -q`
Expected: PASS (all settings tests)

- [ ] **Step 5: Run the whole unit suite** (the autouse fixture touches every test)

Run: `uv run pytest tests/unit -q`
Expected: PASS, 509 + the new tests

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_platform/config/settings.py .env.example .gitignore .dockerignore tests/conftest.py tests/unit/test_settings.py
git commit -m "build: observability settings (store, prices, drift, review) and an isolated test log"
```

---

### Task 2: The SQLite store

**Files:**
- Create: `src/rag_eval_platform/observability/store.py`
- Create: `tests/unit/observe_records.py` (record factories reused by later tests)
- Test: `tests/unit/test_observe_store.py`

**Interfaces:**
- Produces:
  - `Source = Literal["api", "playground", "cli"]`, `EvalKind = Literal["retrieval", "generation", "gate"]`, `SCHEMA_VERSION = 1`
  - `class ObserveStoreError(Exception)`, `class UnknownQueryError(LookupError)`
  - frozen `QueryRecord(id, ts, source, collection, question, answer, style, prompt_version, model, refusal, retrieved_chunk_ids, cited_doc_ids, invalid_citations, input_tokens, output_tokens, retrieve_ms, generate_ms, total_ms, cost_usd, status, error=None, reviewed_at=None)`
  - frozen `FeedbackRecord(id, query_id, ts, rating, comment, source)`, frozen `EvalRunRecord(id, ts, kind, passed, metrics: Mapping[str, float], fingerprint)`
  - `ObserveStore(path: Path)` with `add_query(QueryRecord) -> None`, `get_query(str) -> QueryRecord | None`, `queries(since: datetime | None = None) -> list[QueryRecord]` (oldest first), `mark_reviewed(query_id: str, at: datetime) -> bool`, `add_feedback(FeedbackRecord) -> None` (raises `UnknownQueryError`), `feedback(since: datetime | None = None) -> list[FeedbackRecord]`, `add_eval_run(EvalRunRecord) -> None`, `eval_runs(kind: str | None = None) -> list[EvalRunRecord]`
  - `utc(ts: datetime) -> datetime` (naive → UTC, aware → converted to UTC)

- [ ] **Step 1: Write the record factories** — `tests/unit/observe_records.py`:

```python
"""Record factories for observability tests."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from rag_eval_platform.observability.store import EvalRunRecord, FeedbackRecord, QueryRecord

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


def make_query(n: int = 0, **changes: object) -> QueryRecord:
    """Query number ``n``, one minute after query ``n - 1``; override any field."""
    fields: dict[str, object] = {
        "id": f"q{n}", "ts": T0 + timedelta(minutes=n), "source": "api",
        "collection": "rag_documents", "question": f"Question {n}?", "answer": "Answer [1].",
        "style": "concise", "prompt_version": "v1", "model": "qwen3:8b", "refusal": False,
        "retrieved_chunk_ids": ("a.md#0", "b.md#1"), "cited_doc_ids": ("a.md",),
        "invalid_citations": (), "input_tokens": 300, "output_tokens": 40,
        "retrieve_ms": 200.0, "generate_ms": 1800.0, "total_ms": 2000.0, "cost_usd": 0.0,
        "status": "ok",
    }  # fmt: skip
    fields.update(changes)
    return QueryRecord(**fields)  # type: ignore[arg-type]


def make_feedback(query_id: str, rating: int, minutes: int = 0,
                  comment: str | None = None) -> FeedbackRecord:  # fmt: skip
    return FeedbackRecord(id=f"f-{query_id}-{minutes}", query_id=query_id,
                          ts=T0 + timedelta(minutes=minutes), rating=rating, comment=comment,
                          source="playground")  # fmt: skip


def make_run(
    kind: str, passed: bool, metrics: Mapping[str, float], minutes: int = 0
) -> EvalRunRecord:
    return EvalRunRecord(id=f"r-{kind}-{minutes}", ts=T0 + timedelta(minutes=minutes), kind=kind,
                         passed=passed, metrics=dict(metrics), fingerprint="abc123")  # fmt: skip
```

- [ ] **Step 2: Write the failing tests** — `tests/unit/test_observe_store.py`:

```python
"""Tests for observability.store: a real SQLite file in a temporary folder."""

import sqlite3
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from tests.unit.observe_records import T0, make_feedback, make_query, make_run

from rag_eval_platform.observability.store import (
    SCHEMA_VERSION,
    ObserveStore,
    ObserveStoreError,
    UnknownQueryError,
)


@pytest.fixture
def store(tmp_path: Path) -> ObserveStore:
    return ObserveStore(tmp_path / "nested" / "rag.db")


def test_a_query_round_trips_every_field(store: ObserveStore) -> None:
    record = make_query(1, invalid_citations=(7,), error=None)

    store.add_query(record)

    assert store.get_query("q1") == record
    assert store.get_query("missing") is None


def test_creates_the_folder_and_stamps_the_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "rag.db"

    ObserveStore(path).add_query(make_query())

    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_queries_are_oldest_first_and_filter_by_time(store: ObserveStore) -> None:
    for n in (3, 1, 2):
        store.add_query(make_query(n))

    assert [q.id for q in store.queries()] == ["q1", "q2", "q3"]
    assert [q.id for q in store.queries(since=T0 + timedelta(minutes=2))] == ["q2", "q3"]


def test_timestamps_are_stored_as_utc(store: ObserveStore) -> None:
    plus_two = timezone(timedelta(hours=2))
    store.add_query(make_query(1, ts=datetime(2026, 9, 29, 14, 30, tzinfo=plus_two)))  # 12:30 UTC
    store.add_query(make_query(2, ts=datetime(2026, 9, 29, 12, 15)))  # naive: taken as UTC

    rows = store.queries()

    assert [q.id for q in rows] == ["q2", "q1"]
    assert rows[1].ts == datetime(2026, 9, 29, 12, 30, tzinfo=UTC)


def test_long_and_non_ascii_text_round_trips(store: ObserveStore) -> None:
    question = "Why? 🤔 ما هو «RAG»? 'quotes' \"double\" " + "x" * 2000
    store.add_query(make_query(1, question=question, answer="Réponse ✅ [1]."))

    saved = store.get_query("q1")

    assert saved is not None
    assert (saved.question, saved.answer) == (question, "Réponse ✅ [1].")


def test_mark_reviewed(store: ObserveStore) -> None:
    store.add_query(make_query(1))

    assert store.mark_reviewed("q1", T0) is True
    assert store.mark_reviewed("nope", T0) is False
    saved = store.get_query("q1")
    assert saved is not None and saved.reviewed_at == T0


def test_feedback_needs_a_known_query(store: ObserveStore) -> None:
    store.add_query(make_query(1))
    store.add_feedback(make_feedback("q1", 0, comment="wrong source"))

    with pytest.raises(UnknownQueryError):
        store.add_feedback(make_feedback("ghost", 1))
    assert store.feedback() == [make_feedback("q1", 0, comment="wrong source")]


def test_eval_runs_round_trip_and_filter_by_kind(store: ObserveStore) -> None:
    store.add_eval_run(make_run("retrieval", True, {"recall": 0.93}, minutes=1))
    store.add_eval_run(make_run("gate", False, {"recall": 0.7}, minutes=2))

    assert [r.kind for r in store.eval_runs()] == ["retrieval", "gate"]
    assert store.eval_runs("gate") == [make_run("gate", False, {"recall": 0.7}, minutes=2)]


def test_a_corrupt_file_raises_with_a_hint(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    path.write_bytes(b"this is not a database" * 100)

    with pytest.raises(ObserveStoreError, match="aside"):
        ObserveStore(path).queries()


def test_a_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(ObserveStoreError, match="newer"):
        ObserveStore(path).queries()


def test_two_writers_on_one_file_lose_nothing(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    ObserveStore(path).queries()  # create the schema first

    def write(offset: int) -> None:
        writer = ObserveStore(path)
        for n in range(offset, offset + 25):
            writer.add_query(make_query(n))

    threads = [threading.Thread(target=write, args=(start,)) for start in (0, 100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ObserveStore(path).queries()) == 50
```

- [ ] **Step 3: Run to verify they fail**

Run: `uv run pytest tests/unit/test_observe_store.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_eval_platform.observability.store'`

- [ ] **Step 4: Implement** — `src/rag_eval_platform/observability/store.py`:

```python
"""SQLite store for the query log, feedback and evaluation runs (Phase 7).

One short connection per operation, the classic rollback journal and a 5 s busy timeout:
safe across a Docker bind mount and Streamlit's threads. The schema version is SQLite's
``user_version``. Timestamps are stored as UTC ISO-8601 text, so they sort as text.
"""

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Source = Literal["api", "playground", "cli"]
EvalKind = Literal["retrieval", "generation", "gate"]
SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, source TEXT NOT NULL, collection TEXT NOT NULL,
    question TEXT NOT NULL, answer TEXT, style TEXT NOT NULL, prompt_version TEXT,
    model TEXT, refusal INTEGER NOT NULL, retrieved_chunk_ids TEXT NOT NULL,
    cited_doc_ids TEXT NOT NULL, invalid_citations TEXT NOT NULL, input_tokens INTEGER,
    output_tokens INTEGER, retrieve_ms REAL, generate_ms REAL, total_ms REAL,
    cost_usd REAL NOT NULL, status TEXT NOT NULL, error TEXT, reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY, query_id TEXT NOT NULL REFERENCES queries(id), ts TEXT NOT NULL,
    rating INTEGER NOT NULL, comment TEXT, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, passed INTEGER NOT NULL,
    metrics TEXT NOT NULL, fingerprint TEXT
);
CREATE INDEX IF NOT EXISTS queries_ts ON queries(ts);
CREATE INDEX IF NOT EXISTS queries_source_ts ON queries(source, ts);
CREATE INDEX IF NOT EXISTS feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS eval_runs_kind_ts ON eval_runs(kind, ts);
"""


class ObserveStoreError(Exception):
    """The observability database cannot be opened or used."""


class UnknownQueryError(LookupError):
    """Feedback for a query id that is not in the log."""


@dataclass(frozen=True)
class QueryRecord:
    id: str
    ts: datetime
    source: str
    collection: str
    question: str  # or "sha256:<hex>" when text storage is off
    answer: str | None
    style: str
    prompt_version: str | None
    model: str | None
    refusal: bool
    retrieved_chunk_ids: tuple[str, ...]
    cited_doc_ids: tuple[str, ...]
    invalid_citations: tuple[int, ...]
    input_tokens: int | None
    output_tokens: int | None
    retrieve_ms: float | None
    generate_ms: float | None
    total_ms: float | None
    cost_usd: float
    status: str  # "ok" or "error"
    error: str | None = None
    reviewed_at: datetime | None = None


@dataclass(frozen=True)
class FeedbackRecord:
    id: str
    query_id: str
    ts: datetime
    rating: int  # 1 = 👍, 0 = 👎
    comment: str | None
    source: str


@dataclass(frozen=True)
class EvalRunRecord:
    id: str
    ts: datetime
    kind: str
    passed: bool
    metrics: Mapping[str, float]
    fingerprint: str | None


def utc(ts: datetime) -> datetime:
    """Naive timestamps are taken as UTC; aware ones are converted to UTC."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _text(ts: datetime | None) -> str | None:
    return None if ts is None else utc(ts).isoformat()


def _time(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


class ObserveStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        hint = f"Move {self.path} aside to start a fresh log"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.path, timeout=5.0)
        except (OSError, sqlite3.Error) as exc:
            raise ObserveStoreError(f"Cannot open {self.path}: {exc}. {hint}") from exc
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA foreign_keys=ON")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ObserveStoreError(
                    f"{self.path} has schema v{version}, newer than this code (v"
                    f"{SCHEMA_VERSION}). {hint}"
                )
            if version < SCHEMA_VERSION:
                conn.executescript(_SCHEMA)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            with conn:  # one transaction: commit on success, roll back on error
                yield conn
        except sqlite3.Error as exc:
            raise ObserveStoreError(f"{self.path}: {exc}. {hint}") from exc
        finally:
            conn.close()

    def add_query(self, record: QueryRecord) -> None:
        values = astuple(record)
        row = (
            *values[:1], _text(record.ts), *values[2:10],
            json.dumps(record.retrieved_chunk_ids), json.dumps(record.cited_doc_ids),
            json.dumps(record.invalid_citations), *values[13:21], _text(record.reviewed_at),
        )  # fmt: skip
        with self._connect() as conn:
            conn.execute(f"INSERT INTO queries VALUES ({', '.join('?' * len(row))})", row)  # noqa: S608

    def get_query(self, query_id: str) -> QueryRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
        return None if row is None else _query(row)

    def queries(self, since: datetime | None = None) -> list[QueryRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM queries WHERE ts >= ? ORDER BY ts, id",
                                (_text(since) or "",)).fetchall()  # fmt: skip
        return [_query(row) for row in rows]

    def mark_reviewed(self, query_id: str, at: datetime) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("UPDATE queries SET reviewed_at = ? WHERE id = ?",
                                  (_text(at), query_id))  # fmt: skip
        return cursor.rowcount == 1

    def add_feedback(self, record: FeedbackRecord) -> None:
        with self._connect() as conn:
            if (
                conn.execute("SELECT 1 FROM queries WHERE id = ?", (record.query_id,)).fetchone()
                is None
            ):
                raise UnknownQueryError(record.query_id)
            conn.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                         (record.id, record.query_id, _text(record.ts), record.rating,
                          record.comment, record.source))  # fmt: skip

    def feedback(self, since: datetime | None = None) -> list[FeedbackRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM feedback WHERE ts >= ? ORDER BY ts, id",
                                (_text(since) or "",)).fetchall()  # fmt: skip
        return [FeedbackRecord(row["id"], row["query_id"], _utc_time(row["ts"]), row["rating"],
                               row["comment"], row["source"]) for row in rows]  # fmt: skip

    def add_eval_run(self, record: EvalRunRecord) -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO eval_runs VALUES (?, ?, ?, ?, ?, ?)",
                         (record.id, _text(record.ts), record.kind, int(record.passed),
                          json.dumps(dict(record.metrics)), record.fingerprint))  # fmt: skip

    def eval_runs(self, kind: str | None = None) -> list[EvalRunRecord]:
        sql = "SELECT * FROM eval_runs" + (" WHERE kind = ?" if kind else "") + " ORDER BY ts, id"
        with self._connect() as conn:
            rows = conn.execute(sql, (kind,) if kind else ()).fetchall()
        return [EvalRunRecord(row["id"], _utc_time(row["ts"]), row["kind"], bool(row["passed"]),
                              json.loads(row["metrics"]), row["fingerprint"])
                for row in rows]  # fmt: skip


def _utc_time(text: str) -> datetime:
    parsed = _time(text)
    assert parsed is not None  # noqa: S101 - NOT NULL column
    return parsed


def _query(row: Mapping[str, Any]) -> QueryRecord:
    return QueryRecord(
        id=row["id"], ts=_utc_time(row["ts"]), source=row["source"],
        collection=row["collection"], question=row["question"], answer=row["answer"],
        style=row["style"], prompt_version=row["prompt_version"], model=row["model"],
        refusal=bool(row["refusal"]),
        retrieved_chunk_ids=tuple(json.loads(row["retrieved_chunk_ids"])),
        cited_doc_ids=tuple(json.loads(row["cited_doc_ids"])),
        invalid_citations=tuple(json.loads(row["invalid_citations"])),
        input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
        retrieve_ms=row["retrieve_ms"], generate_ms=row["generate_ms"],
        total_ms=row["total_ms"], cost_usd=row["cost_usd"], status=row["status"],
        error=row["error"], reviewed_at=_time(row["reviewed_at"]),
    )  # fmt: skip
```

Note: `astuple` copies the tuple fields as tuples; the slices map to the column order of the `CREATE TABLE`: `id`, `ts`, `source … refusal` (indexes 2–9), three JSON columns (10–12), `input_tokens … error` (13–20), `reviewed_at` (21). If mypy rejects `sqlite3.Row` as `Mapping[str, Any]`, type `_query`'s parameter as `sqlite3.Row`.

- [ ] **Step 5: Run to verify they pass**

Run: `uv run pytest tests/unit/test_observe_store.py -q`
Expected: PASS (11 tests)

- [ ] **Step 6: Lint and types**

Run: `uv run ruff check src/rag_eval_platform/observability tests/unit && uv run ruff format src/rag_eval_platform/observability tests/unit && uv run mypy src tests`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add src/rag_eval_platform/observability/store.py tests/unit/observe_records.py tests/unit/test_observe_store.py
git commit -m "feat: SQLite observability store for queries, feedback and evaluation runs"
```

---

### Task 3: Cost and the query logger

**Files:**
- Create: `src/rag_eval_platform/observability/cost.py`
- Replace: `src/rag_eval_platform/observability/logger.py`
- Test: `tests/unit/test_observe_cost.py`, `tests/unit/test_observe_logger.py`

**Interfaces:**
- Consumes: Task 2's store, records, `Source`, `ObserveStoreError`, `UnknownQueryError`.
- Produces:
  - `Prices(input_per_1m: float = 0.0, output_per_1m: float = 0.0)` with `.is_free` and `Prices.from_settings(settings)`; `cost_usd(input_tokens: int | None, output_tokens: int | None, prices: Prices) -> float`
  - `Clock = Callable[[], datetime]`; `new_id() -> str`
  - `QueryLogger(store, *, source: Source, collection: str, style: str, prices: Prices = Prices(), store_text: bool = True, clock: Clock = utc_now)` with `record(answer: Answer, *, retrieve_ms: float, total_ms: float) -> str | None` and `record_error(question: str, error: BaseException, *, total_ms: float) -> str | None`
  - `create_store(settings) -> ObserveStore | None` (None when disabled)
  - `create_query_logger(settings, source: Source, *, collection: str | None = None, style: str = "concise") -> QueryLogger | None`
  - `record_feedback(store, query_id: str, rating: int, comment: str | None, source: Source, clock: Clock = utc_now) -> str` (raises `UnknownQueryError`, `ObserveStoreError`)
  - `record_eval_run(store: ObserveStore | None, kind: EvalKind, passed: bool, metrics: Mapping[str, float | None], fingerprint: Mapping[str, Any] | None, clock: Clock = utc_now) -> str | None` (never raises)

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_observe_cost.py`:

```python
"""Tests for observability.cost."""

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.cost import Prices, cost_usd


def test_cost_is_tokens_times_price_per_million() -> None:
    prices = Prices(input_per_1m=0.15, output_per_1m=0.60)

    assert cost_usd(1_000_000, 500_000, prices) == 0.45
    assert cost_usd(None, None, prices) == 0.0


def test_local_models_are_free_by_default() -> None:
    prices = Prices.from_settings(Settings(_env_file=None))

    assert prices.is_free
    assert cost_usd(300, 40, prices) == 0.0
    assert not Prices(output_per_1m=1.0).is_free
```

`tests/unit/test_observe_logger.py`:

```python
"""Tests for observability.logger: answers become rows, and failures never escape."""

import hashlib
from pathlib import Path

import pytest
from tests.unit.observe_records import T0
from tests.unit.test_api_app import answer

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import GenerationError
from rag_eval_platform.observability.cost import Prices
from rag_eval_platform.observability.logger import (
    QueryLogger,
    create_query_logger,
    create_store,
    record_eval_run,
    record_feedback,
)
from rag_eval_platform.observability.store import ObserveStore, UnknownQueryError


def logger_for(store: ObserveStore, **options: object) -> QueryLogger:
    return QueryLogger(store, source="api", collection="rag_documents", style="concise",
                       clock=lambda: T0, **options)  # type: ignore[arg-type]  # fmt: skip


def test_record_stores_the_answer_and_returns_its_id(tmp_path: Path) -> None:
    store = ObserveStore(tmp_path / "rag.db")

    query_id = logger_for(store, prices=Prices(1.0, 2.0)).record(
        answer(), retrieve_ms=5.04, total_ms=30.06
    )

    saved = store.get_query(query_id or "")
    assert saved is not None
    assert (saved.source, saved.collection, saved.style, saved.status) == (
        "api", "rag_documents", "concise", "ok")  # fmt: skip
    assert (saved.question, saved.answer) == ("q?", "F equals ma [1].")
    assert saved.cited_doc_ids == (answer().sources[0].chunk.doc_id,)
    assert (saved.retrieve_ms, saved.generate_ms, saved.total_ms) == (5.0, 20.0, 30.1)
    assert saved.cost_usd == pytest.approx((10 * 1.0 + 4 * 2.0) / 1_000_000)
    assert saved.ts == T0


def test_text_storage_off_keeps_only_a_hash(tmp_path: Path) -> None:
    store = ObserveStore(tmp_path / "rag.db")

    query_id = logger_for(store, store_text=False).record(answer(), retrieve_ms=1, total_ms=2)

    saved = store.get_query(query_id or "")
    assert saved is not None
    assert saved.question == "sha256:" + hashlib.sha256(b"q?").hexdigest()
    assert saved.answer is None


def test_record_error_stores_an_error_row(tmp_path: Path) -> None:
    store = ObserveStore(tmp_path / "rag.db")

    query_id = logger_for(store).record_error("q?", GenerationError("down"), total_ms=12.0)

    saved = store.get_query(query_id or "")
    assert saved is not None
    assert (saved.status, saved.error, saved.answer) == ("error", "GenerationError: down", None)


def test_a_broken_store_returns_none_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "rag.db").write_bytes(b"not a database" * 100)
    broken = ObserveStore(tmp_path / "rag.db")

    assert logger_for(broken).record(answer(), retrieve_ms=1, total_ms=2) is None
    assert logger_for(broken).record_error("q", ValueError("x"), total_ms=1) is None
    assert "Could not record" in caplog.text


def test_factories_follow_the_settings(tmp_path: Path) -> None:
    on = Settings(_env_file=None, observe_db=tmp_path / "rag.db")
    off = Settings(_env_file=None, observe_enabled=False)

    logger = create_query_logger(on, "cli", style="detailed")

    assert create_store(off) is None
    assert create_query_logger(off, "cli") is None
    assert logger is not None
    query_id = logger.record(answer(), retrieve_ms=1, total_ms=2)
    saved = ObserveStore(tmp_path / "rag.db").get_query(query_id or "")
    assert saved is not None
    assert (saved.source, saved.collection, saved.style) == ("cli", "rag_documents", "detailed")


def test_record_feedback_needs_a_known_query(tmp_path: Path) -> None:
    store = ObserveStore(tmp_path / "rag.db")
    query_id = logger_for(store).record(answer(), retrieve_ms=1, total_ms=2) or ""

    feedback_id = record_feedback(store, query_id, 0, "wrong", "api", clock=lambda: T0)

    assert [f.id for f in store.feedback()] == [feedback_id]
    with pytest.raises(UnknownQueryError):
        record_feedback(store, "ghost", 1, None, "api")


def test_record_eval_run_drops_unscored_metrics_and_hashes_the_fingerprint(tmp_path: Path) -> None:
    store = ObserveStore(tmp_path / "rag.db")

    run_id = record_eval_run(store, "generation", True,
                             {"faithfulness": 0.93, "context_recall": None},
                             {"llm_model": "qwen3:8b"}, clock=lambda: T0)  # fmt: skip

    [run] = store.eval_runs()
    assert run.id == run_id
    assert dict(run.metrics) == {"faithfulness": 0.93}
    assert run.fingerprint is not None and len(run.fingerprint) == 16
    assert record_eval_run(None, "gate", True, {}, None) is None


def test_record_eval_run_never_raises(tmp_path: Path) -> None:
    (tmp_path / "rag.db").write_bytes(b"not a database" * 100)

    assert record_eval_run(ObserveStore(tmp_path / "rag.db"), "gate", True, {}, None) is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_observe_cost.py tests/unit/test_observe_logger.py -q`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: cannot import name 'QueryLogger'`

- [ ] **Step 3: Implement**

`src/rag_eval_platform/observability/cost.py`:

```python
"""What an answer cost: tokens times the configured price (0 for local models)."""

from dataclasses import dataclass
from typing import Self

from rag_eval_platform.config.settings import Settings


@dataclass(frozen=True)
class Prices:
    input_per_1m: float = 0.0  # USD per million input tokens
    output_per_1m: float = 0.0

    @property
    def is_free(self) -> bool:
        return self.input_per_1m == 0 and self.output_per_1m == 0

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.price_input_per_1m, settings.price_output_per_1m)


def cost_usd(input_tokens: int | None, output_tokens: int | None, prices: Prices) -> float:
    spent = (input_tokens or 0) * prices.input_per_1m + (output_tokens or 0) * prices.output_per_1m
    return spent / 1_000_000
```

`src/rag_eval_platform/observability/logger.py`:

```python
"""Record answered questions, feedback and evaluation runs in the observability store.

Recording never breaks the caller: ``QueryLogger`` and ``record_eval_run`` log a warning
and return None when the store fails. ``record_feedback`` raises, because the API must
tell the client that the query id was unknown.
"""

import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Answer
from rag_eval_platform.observability.cost import Prices, cost_usd
from rag_eval_platform.observability.store import (
    EvalKind,
    EvalRunRecord,
    FeedbackRecord,
    ObserveStore,
    QueryRecord,
    Source,
)

logger = logging.getLogger(__name__)
Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class QueryLogger:
    """Turns an answer (or a failure) into one ``queries`` row."""

    def __init__(self, store: ObserveStore, *, source: Source, collection: str, style: str,
                 prices: Prices = Prices(), store_text: bool = True,
                 clock: Clock = utc_now) -> None:  # fmt: skip
        self._store, self._source, self._collection, self._style = store, source, collection, style
        self._prices, self._store_text, self._clock = prices, store_text, clock

    def record(self, answer: Answer, *, retrieve_ms: float, total_ms: float) -> str | None:
        return self._save(QueryRecord(
            id=new_id(), ts=self._clock(), source=self._source, collection=self._collection,
            question=self._question(answer.question),
            answer=answer.text if self._store_text else None, style=self._style,
            prompt_version=answer.prompt_version, model=answer.model, refusal=answer.is_refusal,
            retrieved_chunk_ids=tuple(r.chunk.id for r in answer.sources),
            cited_doc_ids=tuple(dict.fromkeys(c.source.chunk.doc_id for c in answer.citations)),
            invalid_citations=tuple(answer.invalid_citations),
            input_tokens=answer.input_tokens, output_tokens=answer.output_tokens,
            retrieve_ms=round(retrieve_ms, 1), generate_ms=round(answer.latency_ms, 1),
            total_ms=round(total_ms, 1),
            cost_usd=cost_usd(answer.input_tokens, answer.output_tokens, self._prices),
            status="ok",
        ))  # fmt: skip

    def record_error(self, question: str, error: BaseException, *, total_ms: float) -> str | None:
        return self._save(QueryRecord(
            id=new_id(), ts=self._clock(), source=self._source, collection=self._collection,
            question=self._question(question), answer=None, style=self._style,
            prompt_version=None, model=None, refusal=False, retrieved_chunk_ids=(),
            cited_doc_ids=(), invalid_citations=(), input_tokens=None, output_tokens=None,
            retrieve_ms=None, generate_ms=None, total_ms=round(total_ms, 1), cost_usd=0.0,
            status="error", error=f"{type(error).__name__}: {error}"[:500],
        ))  # fmt: skip

    def _question(self, text: str) -> str:
        if self._store_text:
            return text
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save(self, record: QueryRecord) -> str | None:
        try:
            self._store.add_query(record)
        except Exception as exc:  # recording must never break answering
            logger.warning("Could not record the query", extra={"error": str(exc)})
            return None
        return record.id


def create_store(settings: Settings) -> ObserveStore | None:
    return ObserveStore(settings.observe_db) if settings.observe_enabled else None


def create_query_logger(settings: Settings, source: Source, *, collection: str | None = None,
                        style: str = "concise") -> QueryLogger | None:  # fmt: skip
    store = create_store(settings)
    if store is None:
        return None
    return QueryLogger(store, source=source, collection=collection or settings.collection_name,
                       style=style, prices=Prices.from_settings(settings),
                       store_text=settings.observe_store_text)  # fmt: skip


def record_feedback(store: ObserveStore, query_id: str, rating: int, comment: str | None,
                    source: Source, clock: Clock = utc_now) -> str:  # fmt: skip
    record = FeedbackRecord(new_id(), query_id, clock(), rating, comment, source)
    store.add_feedback(record)
    return record.id


def record_eval_run(store: ObserveStore | None, kind: EvalKind, passed: bool,
                    metrics: Mapping[str, float | None], fingerprint: Mapping[str, Any] | None,
                    clock: Clock = utc_now) -> str | None:  # fmt: skip
    """One row per evaluation run; unscored metrics are dropped. Never raises."""
    if store is None:
        return None
    digest = (hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode())
              .hexdigest()[:16] if fingerprint else None)  # fmt: skip
    record = EvalRunRecord(new_id(), clock(), kind, passed,
                           {k: float(v) for k, v in metrics.items() if v is not None}, digest)  # fmt: skip
    try:
        store.add_eval_run(record)
    except Exception as exc:  # evaluation results matter more than their history
        logger.warning("Could not record the evaluation run", extra={"error": str(exc)})
        return None
    return record.id
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_observe_cost.py tests/unit/test_observe_logger.py -q`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint and types**

Run: `uv run ruff check . && uv run ruff format . && uv run mypy src tests`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_platform/observability/cost.py src/rag_eval_platform/observability/logger.py tests/unit/test_observe_cost.py tests/unit/test_observe_logger.py
git commit -m "feat: query logger, feedback and evaluation-run recording, token cost"
```

---

### Task 4: Pipeline recording and the `cli` source

**Files:**
- Modify: `src/rag_eval_platform/pipeline.py`, `src/rag_eval_platform/ask.py`
- Test: `tests/unit/test_pipeline.py` (new tests; update the `create_pipeline` fakes to accept keywords)

**Interfaces:**
- Consumes: `create_query_logger(settings, source, ...)`, `Source`.
- Produces:
  - `QueryRecorder` protocol: `record(answer, *, retrieve_ms, total_ms) -> str | None`, `record_error(question, error, *, total_ms) -> str | None`
  - `TimedAnswer(answer, retrieve_ms, total_ms, query_id: str | None = None)`
  - `RecordedStream` (iterable of `str`, `.answer: Answer`, `.query_id: str | None` set after the stream finishes)
  - `RagPipeline(retriever, generator, recorder: QueryRecorder | None = None)`; `ask_stream` returns `RecordedStream`
  - `create_pipeline(settings, source: Source | None = None) -> RagPipeline` (recorder only when `source` is given)

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/test_pipeline.py`)

```python
class FakeRecorder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.rows: list[tuple[str, object]] = []

    def record(self, answer: Answer, *, retrieve_ms: float, total_ms: float) -> str | None:
        if self.fail:
            raise RuntimeError("store down")
        self.rows.append(("ok", answer.text))
        return f"id-{len(self.rows)}"

    def record_error(self, question: str, error: BaseException, *, total_ms: float) -> str | None:
        self.rows.append(("error", type(error).__name__))
        return f"id-{len(self.rows)}"


class DownRetriever:
    def retrieve(self, query: str) -> list[SearchResult]:
        raise VectorStoreError("Chroma is down")


def test_ask_timed_records_the_answer_and_returns_its_id() -> None:
    recorder = FakeRecorder()
    pipeline = RagPipeline(FakeRetriever(), Generator(FakeLlm("Answer [1].")), recorder)

    timed = pipeline.ask_timed("q?")

    assert timed.query_id == "id-1"
    assert recorder.rows == [("ok", "Answer [1].")]


def test_a_failed_question_is_recorded_then_re_raised() -> None:
    recorder = FakeRecorder()
    pipeline = RagPipeline(DownRetriever(), Generator(FakeLlm("x")), recorder)

    with pytest.raises(VectorStoreError):
        pipeline.ask_timed("q?")
    with pytest.raises(VectorStoreError):
        pipeline.ask_stream("q?")
    assert recorder.rows == [("error", "VectorStoreError"), ("error", "VectorStoreError")]


def test_a_stream_is_recorded_once_it_finishes() -> None:
    recorder = FakeRecorder()
    pipeline = RagPipeline(FakeRetriever(), Generator(FakeLlm("Streamed [1].")), recorder)

    stream = pipeline.ask_stream("q?")
    assert stream.query_id is None
    text = "".join(stream)

    assert text == "Streamed [1]."
    assert stream.query_id == "id-1"
    assert stream.answer.text == "Streamed [1]."


def test_a_stream_that_fails_is_recorded_as_an_error() -> None:
    class BrokenLlm(FakeLlm):
        def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
            yield Completion(text="Half", input_tokens=None, output_tokens=None)
            raise GenerationError("model crashed")

    recorder = FakeRecorder()
    pipeline = RagPipeline(FakeRetriever(), Generator(BrokenLlm("x")), recorder)

    with pytest.raises(GenerationError):
        list(pipeline.ask_stream("q?"))
    assert recorder.rows == [("error", "GenerationError")]


def test_a_failing_recorder_never_breaks_an_answer() -> None:
    pipeline = RagPipeline(
        FakeRetriever(), Generator(FakeLlm("Fine [1].")), FakeRecorder(fail=True)
    )

    timed = pipeline.ask_timed("q?")
    stream = pipeline.ask_stream("q?")

    assert (timed.answer.text, timed.query_id) == ("Fine [1].", None)
    assert "".join(stream) == "Fine [1]."
    assert stream.query_id is None


def test_create_pipeline_records_only_with_a_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "create_retriever", lambda settings: FakeRetriever())
    monkeypatch.setattr(pipeline_module, "create_generator",
                        lambda settings: Generator(FakeLlm("A [1].")))  # fmt: skip

    assert create_pipeline(Settings()).recorder is None
    recorded = create_pipeline(Settings(), source="cli")
    assert recorded.recorder is not None
    assert recorded.ask_timed("q?").query_id is not None
```

Imports to add at the top: `from rag_eval_platform.generation.generator import Answer` and `from rag_eval_platform.retrieval.vector_store import VectorStoreError` (next to the existing imports). Update the existing ask tests' fakes from `lambda s: ...` to `lambda s, **kw: ...` and `down(settings)` to `down(settings, **kw)`, because `ask.py` now calls `create_pipeline(settings, source="cli")`. Also make `test_create_pipeline_builds_retriever_and_generator`'s monkeypatched factories accept the same arguments they do now (unchanged).

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_pipeline.py -q`
Expected: FAIL (`RagPipeline() takes 3 positional arguments`, `create_pipeline() got an unexpected keyword argument 'source'`)

- [ ] **Step 3: Implement** — in `pipeline.py`:

Replace the docstring's second paragraph with: "Every answered question is logged as one structured line, and, when a recorder is given, stored in the observability log (see ``observability/logger.py``)."

Add after `_Generator`:

```python
class QueryRecorder(Protocol):
    def record(self, answer: Answer, *, retrieve_ms: float, total_ms: float) -> str | None: ...

    def record_error(
        self, question: str, error: BaseException, *, total_ms: float
    ) -> str | None: ...
```

Change `TimedAnswer` and add `RecordedStream`, the new `RagPipeline`, helpers and factory:

```python
@dataclass(frozen=True)
class TimedAnswer:
    answer: Answer
    retrieve_ms: float
    total_ms: float
    query_id: str | None = None  # the observability log's id, when recorded


class RecordedStream:
    """Iterate for the answer text; once it finishes, ``query_id`` holds the log's id."""

    def __init__(self, stream: AnswerStream, question: str, recorder: QueryRecorder | None,
                 started: float, retrieve_ms: float) -> None:  # fmt: skip
        self._stream, self._question, self._recorder = stream, question, recorder
        self._started, self._retrieve_ms = started, retrieve_ms
        self.query_id: str | None = None

    @property
    def answer(self) -> Answer:
        return self._stream.answer

    def __iter__(self) -> Iterator[str]:
        try:
            yield from self._stream
        except Exception as exc:
            _record_error(self._recorder, self._question, exc, self._started)
            raise
        if self._recorder is not None:
            recorder, answer = self._recorder, self._stream.answer
            self.query_id = _safely(lambda: recorder.record(
                answer, retrieve_ms=self._retrieve_ms, total_ms=_ms_since(self._started)))  # fmt: skip


class RagPipeline:
    def __init__(self, retriever: _Retriever, generator: _Generator,
                 recorder: QueryRecorder | None = None) -> None:  # fmt: skip
        self.retriever = retriever
        self.generator = generator
        self.recorder = recorder

    def ask(self, question: str) -> Answer:
        return self.ask_timed(question).answer

    def ask_timed(self, question: str) -> TimedAnswer:
        """Answer, report how long retrieval and the whole request took, and record it."""
        started = time.perf_counter()
        try:
            results = self.retriever.retrieve(question)
            retrieve_ms = _ms_since(started)
            answer = self.generator.generate(question, results)
        except Exception as exc:
            _record_error(self.recorder, question, exc, started)
            raise
        total_ms = _ms_since(started)
        _log_answer(answer, total_ms=total_ms)
        recorder = self.recorder
        query_id = (_safely(lambda: recorder.record(answer, retrieve_ms=retrieve_ms,
                                                     total_ms=total_ms))
                    if recorder is not None else None)  # fmt: skip
        return TimedAnswer(answer, retrieve_ms, total_ms, query_id)

    def ask_stream(self, question: str) -> RecordedStream:
        """Retrieve now, then stream the answer; it is logged and recorded once it finishes."""
        started = time.perf_counter()
        try:
            results = self.retriever.retrieve(question)
        except Exception as exc:
            _record_error(self.recorder, question, exc, started)
            raise
        stream = self.generator.stream(
            question, results,
            on_complete=lambda answer: _log_answer(answer, total_ms=_ms_since(started)),
        )  # fmt: skip
        return RecordedStream(stream, question, self.recorder, started, _ms_since(started))


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _safely(call: Callable[[], str | None]) -> str | None:
    """A recorder that fails must never break an answer."""
    try:
        return call()
    except Exception as exc:
        logger.warning("Could not record the query", extra={"error": str(exc)})
        return None


def _record_error(recorder: QueryRecorder | None, question: str, error: Exception,
                  started: float) -> None:  # fmt: skip
    if recorder is not None:
        _safely(lambda: recorder.record_error(question, error, total_ms=_ms_since(started)))
```

Add `Iterator` to the `collections.abc` import. Replace `create_pipeline`:

```python
def create_pipeline(settings: Settings, source: Source | None = None) -> RagPipeline:
    """The pipeline for ``settings``; with a ``source``, answers go to the observability log."""
    recorder = create_query_logger(settings, source) if source is not None else None
    return RagPipeline(create_retriever(settings), create_generator(settings), recorder)
```

with `from rag_eval_platform.observability.logger import create_query_logger` and `from rag_eval_platform.observability.store import Source`.

In `ask.py`, change `create_pipeline(settings)` to `create_pipeline(settings, source="cli")`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_pipeline.py -q`
Expected: PASS

- [ ] **Step 5: Whole suite, lint, types** (the API tests use `TimedAnswer` and fake streams)

Run: `uv run pytest tests/unit -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`
Expected: PASS, clean

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_platform/pipeline.py src/rag_eval_platform/ask.py tests/unit/test_pipeline.py
git commit -m "feat: the pipeline records every answer and failure; ask.py logs as cli"
```

---

### Task 5: Review queue and drift (pure)

**Files:**
- Create: `src/rag_eval_platform/observability/review.py`, `src/rag_eval_platform/observability/drift.py`
- Test: `tests/unit/test_observe_review.py`, `tests/unit/test_observe_drift.py`

**Interfaces:**
- Consumes: `QueryRecord`, `FeedbackRecord`, `EvalRunRecord`, `Settings`.
- Produces:
  - `Reason = Literal["thumbs_down", "error", "refused", "invalid_citation", "slow"]`, `REASON_LABELS: dict[str, str]`
  - `ReviewItem(query: QueryRecord, reasons: tuple[Reason, ...], comment: str | None)`
  - `latest_feedback(feedback: Sequence[FeedbackRecord]) -> dict[str, FeedbackRecord]`
  - `review_queue(queries, feedback, *, slow_ms: float, include_reviewed: bool = False) -> list[ReviewItem]`
  - `DriftThresholds(window=20, latency_rise=0.5, error_rise=0.10, refusal_rise=0.15, citation_rise=0.10, downvote_rise=0.20, max_drop=0.02)` with `from_settings`; `MIN_RATED = 5`
  - `DriftReport(warnings: tuple[str, ...], have: int, need: int)` with `.enough_data`
  - `percentile(values: Sequence[float], q: float) -> float | None` (nearest rank)
  - `drift_report(queries, feedback, runs, thresholds) -> DriftReport` (inputs oldest first)

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_observe_review.py`:

```python
"""Tests for observability.review."""

from tests.unit.observe_records import T0, make_feedback, make_query

from rag_eval_platform.observability.review import latest_feedback, review_queue


def test_the_latest_rating_wins() -> None:
    history = [make_feedback("q1", 1, minutes=1), make_feedback("q1", 0, minutes=5)]

    assert latest_feedback(history)["q1"].rating == 0


def test_each_reason_puts_an_answer_in_the_queue() -> None:
    queries = [
        make_query(1),  # fine: not queued
        make_query(2, refusal=True),
        make_query(3, invalid_citations=(9,)),
        make_query(4, status="error", error="GenerationError: down", total_ms=10.0),
        make_query(5, total_ms=45000.0),
        make_query(6),
    ]

    items = review_queue(queries, [make_feedback("q6", 0, comment="wrong")], slow_ms=30000)

    reasons = {item.query.id: item.reasons for item in items}
    assert reasons == {"q2": ("refused",), "q3": ("invalid_citation",), "q4": ("error",),
                       "q5": ("slow",), "q6": ("thumbs_down",)}  # fmt: skip
    assert next(i for i in items if i.query.id == "q6").comment == "wrong"


def test_thumbs_down_first_then_errors_then_newest() -> None:
    queries = [make_query(1, refusal=True), make_query(2, status="error", total_ms=None),
               make_query(3, refusal=True), make_query(4)]  # fmt: skip

    items = review_queue(queries, [make_feedback("q4", 0)], slow_ms=30000)

    assert [i.query.id for i in items] == ["q4", "q2", "q3", "q1"]


def test_reviewed_answers_leave_the_queue_unless_asked_for() -> None:
    queries = [make_query(1, refusal=True, reviewed_at=T0)]

    assert review_queue(queries, [], slow_ms=30000) == []
    assert len(review_queue(queries, [], slow_ms=30000, include_reviewed=True)) == 1


def test_a_thumbs_up_after_a_thumbs_down_clears_it() -> None:
    history = [make_feedback("q1", 0, minutes=1), make_feedback("q1", 1, minutes=2)]

    assert review_queue([make_query(1)], history, slow_ms=30000) == []
```

`tests/unit/test_observe_drift.py`:

```python
"""Tests for observability.drift: last N answers against the N before."""

from tests.unit.observe_records import make_feedback, make_query, make_run

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.drift import (
    DriftThresholds,
    drift_report,
    percentile,
)
from rag_eval_platform.observability.store import QueryRecord

T = DriftThresholds(window=4)


def windows(previous: list[QueryRecord], recent: list[QueryRecord]) -> list[QueryRecord]:
    """Renumber so the previous window is older than the recent one."""
    rows = [*previous, *recent]
    return [make_query(n, **{k: getattr(q, k) for k in ("status", "total_ms", "refusal",
                                                          "invalid_citations")})
            for n, q in enumerate(rows)]  # fmt: skip


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 0.95) is None
    assert percentile([5.0], 0.5) == 5.0
    assert percentile([float(n) for n in range(1, 21)], 0.95) == 19.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0


def test_thresholds_come_from_settings() -> None:
    t = DriftThresholds.from_settings(Settings(_env_file=None, drift_window=7, gate_max_drop=0.05))

    assert (t.window, t.max_drop, t.latency_rise) == (7, 0.05, 0.5)


def test_not_enough_data_gives_no_traffic_warnings() -> None:
    report = drift_report([make_query(n, total_ms=99999.0) for n in range(7)], [], [], T)

    assert not report.enough_data
    assert (report.have, report.need) == (7, 8)
    assert report.warnings == ()


def test_steady_traffic_gives_no_warnings() -> None:
    report = drift_report([make_query(n) for n in range(8)], [], [], T)

    assert report.enough_data
    assert report.warnings == ()


def test_latency_rise_warns_with_both_values() -> None:
    rows = windows([make_query(total_ms=2000.0)] * 4, [make_query(total_ms=3100.0)] * 4)

    [warning] = drift_report(rows, [], [], T).warnings

    assert warning == "p95 latency rose from 2.0 s to 3.1 s (last 4 vs previous 4 answers)."


def test_latency_at_exactly_the_limit_does_not_warn() -> None:
    rows = windows([make_query(total_ms=2000.0)] * 4, [make_query(total_ms=3000.0)] * 4)

    assert drift_report(rows, [], [], T).warnings == ()


def test_error_rate_rise_warns_at_the_threshold() -> None:
    ok, bad = make_query(), make_query(status="error", total_ms=None)
    t = DriftThresholds(window=10, error_rise=0.10)
    rows = windows([ok] * 10, [bad] + [ok] * 9)

    assert drift_report(rows, [], [], t).warnings == (
        "Error rate rose from 0% to 10% (last 10 vs previous 10 answers).",
    )


def test_refusal_and_invalid_citation_rises_warn() -> None:
    ok = make_query()
    rows = windows([ok] * 4, [make_query(refusal=True), make_query(invalid_citations=(9,)), ok, ok])

    warnings = drift_report(rows, [], [], T).warnings

    assert "Refusal rate rose from 0% to 25% (last 4 vs previous 4 answers)." in warnings
    assert "Invalid-citation rate rose from 0% to 25% (last 4 vs previous 4 answers)." in warnings


def test_windows_full_of_errors_do_not_crash() -> None:
    bad = make_query(status="error", total_ms=None)

    report = drift_report(windows([bad] * 4, [bad] * 4), [], [], T)

    assert report.enough_data
    assert report.warnings == ()  # 100% -> 100% is no rise; no latency, no refusal data


def test_downvote_rise_needs_enough_ratings() -> None:
    rows = [make_query(n) for n in range(10)]
    t = DriftThresholds(window=5)
    up_before = [make_feedback(f"q{n}", 1) for n in range(5)]
    down_now = [make_feedback(f"q{n}", 0) for n in range(5, 10)]

    assert drift_report(rows, [*up_before, *down_now], [], t).warnings == (
        "👎 rate rose from 0% to 100% (last 5 vs previous 5 answers).",
    )
    assert drift_report(rows, [*up_before, *down_now[:4]], [], t).warnings == ()


def test_evaluation_failures_and_drops_warn_even_without_traffic() -> None:
    runs = [
        make_run("retrieval", True, {"recall": 0.93, "mrr": 0.88}, minutes=1),
        make_run("retrieval", True, {"recall": 0.90, "mrr": 0.88}, minutes=2),
        make_run("gate", False, {"recall": 0.9}, minutes=3),
    ]

    warnings = drift_report([], [], runs, T).warnings

    assert warnings == (
        "Retrieval Recall@k fell from 0.93 to 0.90 since the previous run.",
        "The latest gate run failed (2026-09-29 12:03 UTC).",
    )
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_observe_review.py tests/unit/test_observe_drift.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`src/rag_eval_platform/observability/review.py`:

```python
"""The review queue: answers that deserve a human look, with the reasons why."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag_eval_platform.observability.store import FeedbackRecord, QueryRecord

Reason = Literal["thumbs_down", "error", "refused", "invalid_citation", "slow"]
REASON_LABELS: dict[str, str] = {
    "thumbs_down": "👎 thumbs down", "error": "error", "refused": "refused",
    "invalid_citation": "invalid citation", "slow": "slow",
}  # fmt: skip


@dataclass(frozen=True)
class ReviewItem:
    query: QueryRecord
    reasons: tuple[Reason, ...]
    comment: str | None  # the 👎 comment, if any


def latest_feedback(feedback: Sequence[FeedbackRecord]) -> dict[str, FeedbackRecord]:
    """The newest rating per query (a re-rating replaces the shown one; history is kept)."""
    return {f.query_id: f for f in sorted(feedback, key=lambda f: (f.ts, f.id))}


def _reasons(
    query: QueryRecord, rating: FeedbackRecord | None, slow_ms: float
) -> tuple[Reason, ...]:
    checks: tuple[tuple[Reason, bool], ...] = (
        ("thumbs_down", rating is not None and rating.rating == 0),
        ("error", query.status == "error"),
        ("refused", query.refusal),
        ("invalid_citation", bool(query.invalid_citations)),
        ("slow", query.total_ms is not None and query.total_ms > slow_ms),
    )
    return tuple(reason for reason, hit in checks if hit)


def review_queue(queries: Sequence[QueryRecord], feedback: Sequence[FeedbackRecord], *,
                 slow_ms: float, include_reviewed: bool = False) -> list[ReviewItem]:  # fmt: skip
    """👎 first, then errors, then the rest; newest first within each group."""
    latest = latest_feedback(feedback)
    items = [
        ReviewItem(q, reasons, latest[q.id].comment if q.id in latest else None)
        for q in queries
        if (include_reviewed or q.reviewed_at is None)
        and (reasons := _reasons(q, latest.get(q.id), slow_ms))
    ]

    def group(item: ReviewItem) -> int:
        return 0 if "thumbs_down" in item.reasons else 1 if "error" in item.reasons else 2

    return sorted(items, key=lambda i: (group(i), -i.query.ts.timestamp()))
```

`src/rag_eval_platform/observability/drift.py`:

```python
"""Drift: is recent traffic worse than the traffic just before it?

Windows count answers, not hours (local traffic is light): the last ``window`` answers
against the ``window`` before them. Evaluation runs are checked on their own.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.review import latest_feedback
from rag_eval_platform.observability.store import EvalRunRecord, FeedbackRecord, QueryRecord

MIN_RATED = 5  # ratings needed in each window before the 👎 rate is compared
_RETRIEVAL_METRICS = {"recall": "Recall@k", "mrr": "MRR", "ndcg": "NDCG@k"}


@dataclass(frozen=True)
class DriftThresholds:
    window: int = 20
    latency_rise: float = 0.5  # relative: p95 may grow by 50%
    error_rise: float = 0.10  # absolute: 10 points
    refusal_rise: float = 0.15
    citation_rise: float = 0.10
    downvote_rise: float = 0.20
    max_drop: float = 0.02  # a retrieval metric's allowed fall between evaluation runs

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.drift_window, settings.drift_latency_rise, settings.drift_error_rise,
                   settings.drift_refusal_rise, settings.drift_citation_rise,
                   settings.drift_downvote_rise, settings.gate_max_drop)  # fmt: skip


@dataclass(frozen=True)
class DriftReport:
    warnings: tuple[str, ...]
    have: int  # answers logged
    need: int  # answers needed for traffic checks (2 x window)

    @property
    def enough_data(self) -> bool:
        return self.have >= self.need


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def _rate(rows: Sequence[QueryRecord], hit: Callable[[QueryRecord], bool]) -> float | None:
    return sum(1 for r in rows if hit(r)) / len(rows) if rows else None


def _rose(before: float | None, now: float | None, rise: float) -> bool:
    return before is not None and now is not None and round(now - before, 9) >= rise


def _traffic(previous: list[QueryRecord], recent: list[QueryRecord],
             feedback: Sequence[FeedbackRecord], t: DriftThresholds) -> list[str]:  # fmt: skip
    span = f"(last {t.window} vs previous {t.window} answers)."
    warnings = []

    def p95(rows: list[QueryRecord]) -> float | None:
        return percentile(
            [r.total_ms for r in rows if r.status == "ok" and r.total_ms is not None], 0.95
        )

    before, now = p95(previous), p95(recent)
    if before is not None and now is not None and now > before * (1 + t.latency_rise):
        warnings.append(f"p95 latency rose from {before / 1000:.1f} s to {now / 1000:.1f} s {span}")

    ratings = latest_feedback(feedback)

    def ok(rows: list[QueryRecord]) -> list[QueryRecord]:
        return [r for r in rows if r.status == "ok"]

    def downvote_rate(rows: list[QueryRecord]) -> float | None:
        rated = [ratings[r.id].rating for r in rows if r.id in ratings]
        return rated.count(0) / len(rated) if len(rated) >= MIN_RATED else None

    rates = (
        ("Error rate", _rate(previous, lambda r: r.status == "error"),
         _rate(recent, lambda r: r.status == "error"), t.error_rise),
        ("Refusal rate", _rate(ok(previous), lambda r: r.refusal),
         _rate(ok(recent), lambda r: r.refusal), t.refusal_rise),
        ("Invalid-citation rate", _rate(ok(previous), lambda r: bool(r.invalid_citations)),
         _rate(ok(recent), lambda r: bool(r.invalid_citations)), t.citation_rise),
        ("👎 rate", downvote_rate(previous), downvote_rate(recent), t.downvote_rise),
    )  # fmt: skip
    for label, before_rate, now_rate, rise in rates:
        if _rose(before_rate, now_rate, rise):
            warnings.append(f"{label} rose from {before_rate:.0%} to {now_rate:.0%} {span}")
    return warnings


def _evaluation(runs: Sequence[EvalRunRecord], max_drop: float) -> list[str]:
    warnings = []
    for kind in ("retrieval", "generation", "gate"):
        of_kind = [r for r in runs if r.kind == kind]
        if not of_kind:
            continue
        latest = of_kind[-1]
        if len(of_kind) >= 2 and kind != "generation":
            before = of_kind[-2].metrics
            for key, label in _RETRIEVAL_METRICS.items():
                if key in before and key in latest.metrics and round(
                        before[key] - latest.metrics[key], 9) > max_drop:  # fmt: skip
                    warnings.append(f"{kind.capitalize()} {label} fell from {before[key]:.2f} "
                                    f"to {latest.metrics[key]:.2f} since the previous run.")  # fmt: skip
        if not latest.passed:
            warnings.append(f"The latest {kind} run failed ({latest.ts:%Y-%m-%d %H:%M} UTC).")
    return warnings


def drift_report(queries: Sequence[QueryRecord], feedback: Sequence[FeedbackRecord],
                 runs: Sequence[EvalRunRecord], t: DriftThresholds) -> DriftReport:  # fmt: skip
    """Warnings for recent traffic and evaluation runs (all inputs oldest first)."""
    need = 2 * t.window
    warnings: list[str] = []
    if len(queries) >= need:
        rows = list(queries[-need:])
        warnings += _traffic(rows[: t.window], rows[t.window :], feedback, t)
    warnings += _evaluation(runs, t.max_drop)
    return DriftReport(tuple(warnings), len(queries), need)
```

Note: `_rose` narrows the `float | None` rates for mypy only inside the call; if mypy objects to formatting `before_rate` in the f-string, assert both are not None inside the `if`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_observe_review.py tests/unit/test_observe_drift.py -q`
Expected: PASS (16 tests)

- [ ] **Step 5: Lint and types**

Run: `uv run ruff check . && uv run ruff format . && uv run mypy src tests`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_platform/observability/review.py src/rag_eval_platform/observability/drift.py tests/unit/test_observe_review.py tests/unit/test_observe_drift.py
git commit -m "feat: review queue and drift warnings over the query log"
```

---

### Task 6: API — `query_id`, `POST /feedback`, health warnings

**Files:**
- Modify: `src/rag_eval_platform/api/schemas.py`, `api/services.py`, `api/routes.py`, `api/main.py`
- Test: `tests/unit/test_api_app.py`, `tests/unit/test_api_schemas.py`

**Interfaces:**
- Consumes: `TimedAnswer.query_id`, `RecordedStream.query_id`, `ObserveStore`, `record_feedback`, `UnknownQueryError`, `ObserveStoreError`, `DriftThresholds`, `drift_report`, `create_store`, `create_query_logger`.
- Produces: `QueryResponse.query_id: str | None`; `HealthResponse.warnings: list[str]`; `FeedbackRequest`, `FeedbackResponse`; `ApiServices(..., store: ObserveStore | None = None, drift: DriftThresholds = DriftThresholds())`; route `POST /feedback`.

- [ ] **Step 1: Write the failing tests** (in `tests/unit/test_api_app.py`)

Change the fakes: `FakeStream.__init__` sets `self.query_id: str | None = None` and `__iter__` sets `self.query_id = "stream-id"` after yielding every piece (only when not failing); `FakePipeline.ask_timed` returns `TimedAnswer(answer(), 5.0, 30.0, "query-id")`. Extend `client()` with `store: ObserveStore | None = None` and pass `store=store` to `ApiServices`. Then add:

```python
from tests.unit.observe_records import make_feedback, make_query, make_run

from rag_eval_platform.observability.drift import DriftThresholds
from rag_eval_platform.observability.store import ObserveStore


@pytest.fixture
def store(tmp_path: Path) -> ObserveStore:
    return ObserveStore(tmp_path / "rag.db")


def test_query_and_stream_return_the_log_id() -> None:
    api = client()

    body = api.post("/query", json={"question": "q"}).json()
    events = api.post("/query/stream", json={"question": "q"}).text

    assert body["query_id"] == "query-id"
    done = events.split("event: done\ndata: ")[1].split("\n")[0]
    assert json.loads(done)["query_id"] == "stream-id"


def test_feedback_is_stored_for_a_logged_query(store: ObserveStore) -> None:
    store.add_query(make_query(1))

    response = client(store=store).post(
        "/feedback", json={"query_id": "q1", "rating": "down", "comment": "  wrong source  "}
    )

    assert response.status_code == 201
    [saved] = store.feedback()
    assert (saved.id, saved.rating, saved.comment, saved.source) == (
        response.json()["id"], 0, "wrong source", "api")  # fmt: skip


def test_feedback_for_an_unknown_query_is_404(store: ObserveStore) -> None:
    response = client(store=store).post("/feedback", json={"query_id": "ghost", "rating": "up"})

    assert response.status_code == 404
    assert set(response.json()) == {"error", "hint"}


def test_feedback_is_503_when_observability_is_off() -> None:
    response = client(store=None).post("/feedback", json={"query_id": "q1", "rating": "up"})

    assert response.status_code == 503
    assert "RAG_OBSERVE_ENABLED" in response.json()["hint"]


@pytest.mark.parametrize("body", [{"query_id": "q1", "rating": "meh"},
                                  {"query_id": "q1", "rating": "up", "comment": "x" * 501},
                                  {"query_id": "", "rating": "up"},
                                  {"query_id": "q1", "rating": "up", "extra": 1}])  # fmt: skip
def test_invalid_feedback_is_422(store: ObserveStore, body: dict[str, object]) -> None:
    assert client(store=store).post("/feedback", json=body).status_code == 422


def test_feedback_needs_the_key_and_is_rate_limited(store: ObserveStore) -> None:
    store.add_query(make_query(1))
    api = client(store=store, key="fake-key", limit=1)

    assert api.post("/feedback", json={"query_id": "q1", "rating": "up"}).status_code == 401
    assert api.post("/feedback", json={"query_id": "q1", "rating": "up"}).status_code == 429


def test_health_lists_drift_warnings_without_changing_the_status(store: ObserveStore) -> None:
    store.add_eval_run(make_run("gate", False, {"recall": 0.7}))

    body = client(store=store).get("/health")

    assert body.status_code == 200
    assert body.json()["warnings"] == ["The latest gate run failed (2026-09-29 12:00 UTC)."]


def test_a_broken_log_never_breaks_health(tmp_path: Path) -> None:
    (tmp_path / "rag.db").write_bytes(b"not a database" * 100)

    body = client(store=ObserveStore(tmp_path / "rag.db")).get("/health")

    assert body.status_code == 200
    assert body.json()["warnings"] == []
```

(Import `Path` from `pathlib`. `make_feedback` and `DriftThresholds` are used by later assertions only if needed; drop unused imports so Ruff stays clean.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_api_app.py -q`
Expected: FAIL (`query_id` missing, `/feedback` 404, `ApiServices` has no `store`)

- [ ] **Step 3: Implement**

`schemas.py` — add `query_id: str | None = None` as the last field of `QueryResponse`, set `query_id=timed.query_id` in `to_response`, add `warnings: list[str] = Field(default_factory=list)` to `HealthResponse` (import `Field` from pydantic), and add:

```python
MAX_COMMENT_CHARS = 500


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]
    rating: Literal["up", "down"]
    comment: Annotated[str, StringConstraints(strip_whitespace=True,
                                              max_length=MAX_COMMENT_CHARS)] | None = None  # fmt: skip


class FeedbackResponse(BaseModel):
    id: str
```

`services.py` — add to the `_AnswerStream` protocol:

```python
    query_id: str | None
```

and to `ApiServices` (after `limiter`):

```python
    store: ObserveStore | None = None  # None: observability is off
    drift: DriftThresholds = DriftThresholds()
```

(Import `ObserveStore` and `DriftThresholds`.)

`routes.py`:

```python
@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(request: Request) -> JSONResponse:
    services = services_of(request)
    checks = {name: check() for name, check in services.checks.items()}
    ok = all(c.ok for c in checks.values())
    body = HealthResponse(status="ok" if ok else "degraded", checks=checks,
                          warnings=_drift_warnings(services))  # fmt: skip
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)


def _drift_warnings(services: ApiServices) -> list[str]:
    """Drift is information, never an outage: any failure yields no warnings."""
    store = services.store
    if store is None:
        return []
    try:
        report = drift_report(store.queries(), store.feedback(), store.eval_runs(), services.drift)
    except Exception as exc:
        logger.warning("Could not compute drift warnings", extra={"error": str(exc)})
        return []
    return list(report.warnings)


@router.post("/feedback", status_code=201, response_model=FeedbackResponse,
             responses={404: {"model": ErrorResponse}, **_ERRORS},
             dependencies=[Depends(guard)])  # fmt: skip
def feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    store = services_of(request).store
    if store is None:
        raise ObserveStoreError("Observability is turned off")
    rating = 1 if body.rating == "up" else 0
    return FeedbackResponse(id=record_feedback(store, body.query_id, rating, body.comment or None,
                                               "api"))  # fmt: skip
```

In `_events`, build `TimedAnswer(stream.answer, retrieve_ms, total_ms, stream.query_id)`. Import `ApiServices` from `services`, `FeedbackRequest`, `FeedbackResponse`, `drift_report`, `record_feedback`, `ObserveStoreError`.

`main.py` — add to `_FAILURES`:

```python
    UnknownQueryError: (404, "Use the query_id from a /query response"),
    ObserveStoreError: (503, "Check RAG_OBSERVE_ENABLED and RAG_OBSERVE_DB"),
```

and in `create_app_from_settings`:

```python
    pipelines: dict[AnswerStyle, QueryPipeline] = {
        style: RagPipeline(retriever, create_generator(settings, style),
                           create_query_logger(settings, "api", style=style))
        for style in get_args(AnswerStyle)
    }  # fmt: skip
    ...
    services = ApiServices(
        pipelines=pipelines,
        checks={"chroma": chroma_check(create_vector_store(settings)), "llm": _llm_check(settings)},
        api_key=key, limiter=RateLimiter(settings.api_rate_limit),
        store=create_store(settings), drift=DriftThresholds.from_settings(settings),
    )  # fmt: skip
```

Note: `UnknownQueryError` is a `LookupError`, so the handler's generic "Query failed" warning also logs it; that is acceptable.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_api_app.py tests/unit/test_api_schemas.py -q`
Expected: PASS

- [ ] **Step 5: Whole suite, lint, types**

Run: `uv run pytest tests/unit -q && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`
Expected: PASS, clean

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_platform/api/schemas.py src/rag_eval_platform/api/services.py src/rag_eval_platform/api/routes.py src/rag_eval_platform/api/main.py tests/unit/test_api_app.py tests/unit/test_api_schemas.py
git commit -m "feat: API returns query ids, accepts feedback and reports drift warnings in /health"
```

---

### Task 7: Evaluation history

**Files:**
- Modify: `src/rag_eval_platform/evaluation/fingerprint.py`, `evaluation/cli.py`, `evaluation/generation_cli.py`, `evaluation/gate_cli.py`
- Test: `tests/unit/test_evaluation_cli.py`, `tests/unit/test_generation_cli.py`, `tests/unit/test_gate_cli.py`

**Interfaces:**
- Consumes: `record_eval_run(store, kind, passed, metrics, fingerprint)`, `create_store(settings)`.
- Produces: `fingerprint_or_none(settings) -> dict[str, Any] | None` in `fingerprint.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_evaluation_cli.py`:

```python
def test_each_run_adds_one_evaluation_history_row(
    golden: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_retriever(monkeypatch, "forces.md")

    cli.main(["--golden", str(golden), "--report", str(tmp_path / "r.json")])

    [run] = ObserveStore(get_settings().observe_db).eval_runs()
    assert (run.kind, run.passed) == ("retrieval", True)
    assert run.metrics["recall"] == 1.0
```

Append to `tests/unit/test_gate_cli.py`:

```python
def test_each_gate_run_adds_one_evaluation_history_row(workspace: Path) -> None:
    fresh_generation_baseline(workspace)

    run(workspace)

    [row] = ObserveStore(get_settings().observe_db).eval_runs("gate")
    assert row.passed is True
    assert row.metrics["recall"] == 1.0
    assert row.fingerprint is not None
```

Append to `tests/unit/test_generation_cli.py` (use the file's `fakes` fixture and its existing run helper/argument style — mirror `test_passing_run_writes_report_prints_table_and_returns_0`):

```python
def test_each_run_adds_one_evaluation_history_row(fakes: Fakes, tmp_path: Path) -> None:
    cli.main(["--golden", str(fakes.golden), "--report", str(tmp_path / "g.json")])

    [row] = ObserveStore(get_settings().observe_db).eval_runs("generation")
    assert "faithfulness" in row.metrics
    assert "context_recall" not in row.metrics  # unscored metrics are dropped
```

(Import `ObserveStore` from `rag_eval_platform.observability.store` and `get_settings` where missing. If the `fakes` fixture names its golden path differently, use that name.)

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation_cli.py tests/unit/test_generation_cli.py tests/unit/test_gate_cli.py -q -k history`
Expected: FAIL with `ValueError: not enough values to unpack (expected 1, got 0)`

- [ ] **Step 3: Implement**

`fingerprint.py`:

```python
def fingerprint_or_none(settings: Settings) -> dict[str, Any] | None:
    """The fingerprint as a dict, or None when it cannot be computed (no corpus folder)."""
    try:
        return current_fingerprint(settings).to_dict()
    except OSError:
        return None
```

`cli.py` — before `return EXIT_PASSED if report.passed ...`:

```python
    record_eval_run(create_store(settings), "retrieval", report.passed, asdict(report.overall),
                    fingerprint_or_none(settings.model_copy(
                        update={"golden_dataset_path": args.golden})))  # fmt: skip
```

`generation_cli.py` — before the final return:

```python
    record_eval_run(create_store(settings), "generation", report.passed, asdict(report.overall),
                    report_data.get("fingerprint"))  # fmt: skip
```

`gate_cli.py` — before the final return:

```python
    record_eval_run(create_store(settings), "gate", result.passed, asdict(retrieval.overall),
                    current.to_dict())  # fmt: skip
```

(Import `asdict` from `dataclasses`, `create_store` and `record_eval_run` from `observability.logger`, `fingerprint_or_none` where used.)

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_evaluation_cli.py tests/unit/test_generation_cli.py tests/unit/test_gate_cli.py -q`
Expected: PASS (all, including the existing tests)

- [ ] **Step 5: Lint and types**

Run: `uv run ruff check . && uv run ruff format . && uv run mypy src tests`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_platform/evaluation/fingerprint.py src/rag_eval_platform/evaluation/cli.py src/rag_eval_platform/evaluation/generation_cli.py src/rag_eval_platform/evaluation/gate_cli.py tests/unit/test_evaluation_cli.py tests/unit/test_generation_cli.py tests/unit/test_gate_cli.py
git commit -m "feat: every evaluation and gate run is added to the evaluation history"
```

---

### Task 8: Dashboard data and charts

**Files:**
- Replace: `src/rag_eval_platform/observability/dashboard.py`
- Modify: `src/rag_eval_platform/playground/charts.py`
- Test: `tests/unit/test_observe_dashboard.py`, `tests/unit/test_charts.py`

**Interfaces:**
- Consumes: records, `latest_feedback`, `percentile`, `Prices`, `Settings`.
- Produces:
  - `RANGES: dict[str, timedelta | None]` (keys `"24 h"`, `"7 days"`, `"30 days"`, `"All"`); `since_for(label: str, now: datetime) -> datetime | None`; `bucket_size(label: str) -> timedelta`
  - `Summary(queries, p50_ms, p95_ms, error_rate, refusal_rate, thumbs_up_rate, rated, tokens, cost_usd)`; `summarize(queries, feedback) -> Summary`; `cost_label(summary: Summary, prices: Prices) -> str`
  - `Bucket(start, counts: Mapping[str, int], p50_ms, p95_ms, tokens, cost_usd)`; `buckets(queries, size: timedelta) -> list[Bucket]`
  - `EVAL_METRICS: dict[str, str]`; `EvalPoint(ts, kind, value)`; `eval_history(runs) -> dict[str, list[EvalPoint]]`; `baseline_values(baselines_dir: Path) -> dict[str, float]`; `metric_thresholds(settings) -> dict[str, float]`
  - `query_rows(queries, feedback, source: str | None = None) -> list[dict[str, object]]` (newest first)
  - charts: `observe_latency_chart(buckets)`, `observe_volume_chart(buckets)`, `observe_tokens_chart(buckets)`, `eval_history_chart(label, points, baseline, threshold)`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_observe_dashboard.py`:

```python
"""Tests for observability.dashboard: what the ⑧ Observe tab shows."""

import json
from datetime import timedelta
from pathlib import Path

from tests.unit.observe_records import T0, make_feedback, make_query, make_run

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.cost import Prices
from rag_eval_platform.observability.dashboard import (
    baseline_values,
    bucket_size,
    buckets,
    cost_label,
    eval_history,
    metric_thresholds,
    query_rows,
    since_for,
    summarize,
)


def test_time_ranges() -> None:
    assert since_for("24 h", T0) == T0 - timedelta(hours=24)
    assert since_for("All", T0) is None
    assert bucket_size("24 h") == timedelta(hours=1)
    assert bucket_size("7 days") == timedelta(days=1)


def test_summary_of_an_empty_log() -> None:
    summary = summarize([], [])

    assert summary.queries == 0
    assert (summary.p50_ms, summary.p95_ms, summary.error_rate) == (None, None, None)
    assert summary.thumbs_up_rate is None


def test_summary_tiles() -> None:
    queries = [make_query(1, total_ms=1000.0), make_query(2, total_ms=3000.0, refusal=True),
               make_query(3, status="error", total_ms=None, input_tokens=None,
                          output_tokens=None),
               make_query(4, total_ms=2000.0, cost_usd=0.5)]  # fmt: skip
    feedback = [make_feedback("q1", 1), make_feedback("q2", 0), make_feedback("q4", 1)]

    s = summarize(queries, feedback)

    assert (s.queries, s.p50_ms, s.p95_ms) == (4, 2000.0, 3000.0)
    assert (s.error_rate, s.refusal_rate) == (0.25, 1 / 3)
    assert (s.thumbs_up_rate, s.rated) == (2 / 3, 3)
    assert (s.tokens, s.cost_usd) == (3 * 340, 0.5)


def test_cost_label() -> None:
    s = summarize([make_query(1, cost_usd=0.0123)], [])

    assert cost_label(s, Prices()) == "local · $0"
    assert cost_label(s, Prices(0.15, 0.6)) == "$0.0123"


def test_buckets_group_by_hour_and_source() -> None:
    queries = [make_query(1), make_query(2, source="playground"), make_query(70)]

    rows = buckets(queries, timedelta(hours=1))

    assert [b.start for b in rows] == [T0, T0 + timedelta(hours=1)]
    assert dict(rows[0].counts) == {"api": 1, "playground": 1}
    assert rows[0].p95_ms == 2000.0
    assert rows[1].tokens == 340


def test_eval_history_by_metric_and_the_baselines(tmp_path: Path) -> None:
    runs = [make_run("retrieval", True, {"recall": 0.93, "mrr": 0.88}, minutes=1),
            make_run("gate", True, {"recall": 0.9}, minutes=2),
            make_run("generation", True, {"faithfulness": 0.95}, minutes=3)]  # fmt: skip
    (tmp_path / "retrieval.json").write_text(
        json.dumps(
            {"report": {"overall": {"recall": 0.93, "mrr": 0.88, "ndcg": 0.87, "precision": 0.3}}}
        )
    )
    (tmp_path / "generation.json").write_text("not json")

    history = eval_history(runs)

    assert [(p.kind, p.value) for p in history["recall"]] == [("retrieval", 0.93), ("gate", 0.9)]
    assert [p.value for p in history["faithfulness"]] == [0.95]
    assert baseline_values(tmp_path) == {"recall": 0.93, "mrr": 0.88, "ndcg": 0.87}
    assert baseline_values(tmp_path / "missing") == {}


def test_metric_thresholds_come_from_settings() -> None:
    assert metric_thresholds(Settings(_env_file=None)) == {
        "recall": 0.80, "mrr": 0.70, "ndcg": 0.70, "faithfulness": 0.85, "answer_relevance": 0.80,
    }  # fmt: skip


def test_query_rows_are_newest_first_and_filter_by_source() -> None:
    queries = [make_query(1, question="x" * 200), make_query(2, source="cli")]

    rows = query_rows(queries, [make_feedback("q1", 0)])

    assert [r["id"] for r in rows] == ["q2", "q1"]
    assert rows[1]["rating"] == "👎"
    assert len(str(rows[1]["question"])) <= 80
    assert [r["id"] for r in query_rows(queries, [], source="cli")] == ["q2"]
```

Append to `tests/unit/test_charts.py`:

```python
def test_observe_charts() -> None:
    from datetime import timedelta

    from tests.unit.observe_records import make_query

    from rag_eval_platform.observability.dashboard import EvalPoint, buckets

    rows = buckets([make_query(1), make_query(2, source="cli")], timedelta(hours=1))

    assert len(charts.observe_latency_chart(rows).data) == 2  # p50, p95
    assert {t.name for t in charts.observe_volume_chart(rows).data} == {"api", "cli"}
    assert len(charts.observe_tokens_chart(rows).data) == 1
    points = [EvalPoint(make_query(1).ts, "retrieval", 0.93)]
    figure = charts.eval_history_chart("Recall@k", points, baseline=0.9, threshold=0.8)
    assert len(figure.data) == 1
    assert len(figure.layout.shapes) == 2  # baseline and threshold lines
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_observe_dashboard.py tests/unit/test_charts.py -q`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

`src/rag_eval_platform/observability/dashboard.py`:

```python
"""Data for the ⑧ Observe tab: tiles, time buckets, evaluation history and table rows.

Pure functions over store records, so they are tested without Streamlit.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.cost import Prices
from rag_eval_platform.observability.drift import percentile
from rag_eval_platform.observability.review import latest_feedback
from rag_eval_platform.observability.store import EvalRunRecord, FeedbackRecord, QueryRecord

RANGES: dict[str, timedelta | None] = {
    "24 h": timedelta(hours=24), "7 days": timedelta(days=7),
    "30 days": timedelta(days=30), "All": None,
}  # fmt: skip
EVAL_METRICS: dict[str, str] = {
    "recall": "Recall@k", "mrr": "MRR", "ndcg": "NDCG@k",
    "faithfulness": "Faithfulness", "answer_relevance": "Answer relevance",
}  # fmt: skip
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def since_for(label: str, now: datetime) -> datetime | None:
    span = RANGES[label]
    return None if span is None else now - span


def bucket_size(label: str) -> timedelta:
    return timedelta(hours=1) if label == "24 h" else timedelta(days=1)


@dataclass(frozen=True)
class Summary:
    queries: int
    p50_ms: float | None
    p95_ms: float | None
    error_rate: float | None
    refusal_rate: float | None  # among answered (ok) queries
    thumbs_up_rate: float | None  # among rated queries
    rated: int
    tokens: int
    cost_usd: float


def _latencies(queries: Sequence[QueryRecord]) -> list[float]:
    return [q.total_ms for q in queries if q.status == "ok" and q.total_ms is not None]


def _tokens(q: QueryRecord) -> int:
    return (q.input_tokens or 0) + (q.output_tokens or 0)


def summarize(queries: Sequence[QueryRecord], feedback: Sequence[FeedbackRecord]) -> Summary:
    ok = [q for q in queries if q.status == "ok"]
    ids = {q.id for q in queries}
    rated = [f.rating for f in latest_feedback(feedback).values() if f.query_id in ids]
    latencies = _latencies(queries)
    return Summary(
        queries=len(queries), p50_ms=percentile(latencies, 0.5), p95_ms=percentile(latencies, 0.95),
        error_rate=(len(queries) - len(ok)) / len(queries) if queries else None,
        refusal_rate=sum(q.refusal for q in ok) / len(ok) if ok else None,
        thumbs_up_rate=rated.count(1) / len(rated) if rated else None, rated=len(rated),
        tokens=sum(_tokens(q) for q in queries), cost_usd=sum(q.cost_usd for q in queries),
    )  # fmt: skip


def cost_label(summary: Summary, prices: Prices) -> str:
    return "local · $0" if prices.is_free else f"${summary.cost_usd:.4f}"


@dataclass(frozen=True)
class Bucket:
    start: datetime
    counts: Mapping[str, int]  # queries per source
    p50_ms: float | None
    p95_ms: float | None
    tokens: int
    cost_usd: float


def buckets(queries: Sequence[QueryRecord], size: timedelta) -> list[Bucket]:
    groups: dict[datetime, list[QueryRecord]] = {}
    for q in queries:
        start = _EPOCH + ((q.ts - _EPOCH) // size) * size
        groups.setdefault(start, []).append(q)
    return [
        Bucket(start, dict(Counter(q.source for q in rows)),
               percentile(_latencies(rows), 0.5), percentile(_latencies(rows), 0.95),
               sum(_tokens(q) for q in rows), sum(q.cost_usd for q in rows))
        for start, rows in sorted(groups.items())
    ]  # fmt: skip


@dataclass(frozen=True)
class EvalPoint:
    ts: datetime
    kind: str
    value: float


def eval_history(runs: Sequence[EvalRunRecord]) -> dict[str, list[EvalPoint]]:
    """Metric -> its values over time (retrieval metrics from retrieval and gate runs)."""
    return {
        metric: [EvalPoint(r.ts, r.kind, r.metrics[metric]) for r in runs if metric in r.metrics]
        for metric in EVAL_METRICS
    }


def baseline_values(baselines_dir: Path) -> dict[str, float]:
    """The committed baselines' overall scores, as reference lines (unreadable files skipped)."""
    values: dict[str, float] = {}
    for name in ("retrieval.json", "generation.json"):
        try:
            overall = json.loads((baselines_dir / name).read_text(encoding="utf-8"))["report"][
                "overall"
            ]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        values |= {k: float(v) for k, v in overall.items() if k in EVAL_METRICS and v is not None}
    return values


def metric_thresholds(settings: Settings) -> dict[str, float]:
    return {"recall": settings.min_recall_at_k, "mrr": settings.min_mrr,
            "ndcg": settings.min_ndcg_at_k, "faithfulness": settings.min_faithfulness,
            "answer_relevance": settings.min_answer_relevance}  # fmt: skip


def query_rows(queries: Sequence[QueryRecord], feedback: Sequence[FeedbackRecord],
               source: str | None = None) -> list[dict[str, object]]:  # fmt: skip
    """Rows for the recent-queries table, newest first."""
    ratings = latest_feedback(feedback)
    rows = [q for q in queries if source is None or q.source == source]
    return [
        {"id": q.id, "time": f"{q.ts:%Y-%m-%d %H:%M}", "source": q.source,
         "question": q.question if len(q.question) <= 80 else q.question[:79] + "…",
         "status": "refused" if q.refusal else q.status,
         "seconds": None if q.total_ms is None else round(q.total_ms / 1000, 1),
         "tokens": _tokens(q),
         "rating": {1: "👍", 0: "👎"}.get(ratings[q.id].rating, "") if q.id in ratings else ""}
        for q in sorted(rows, key=lambda q: q.ts, reverse=True)
    ]  # fmt: skip
```

Append to `src/rag_eval_platform/playground/charts.py` (import `Bucket` and `EvalPoint` from `rag_eval_platform.observability.dashboard`):

```python
SOURCE_COLORS = {"api": ACCENT, "playground": NEUTRAL, "cli": "#16a34a"}


def observe_latency_chart(rows: Sequence[Bucket], height: int = 220) -> go.Figure:
    """p50 and p95 answer time per bucket, in seconds."""
    x = [b.start for b in rows]
    fig = go.Figure()
    for name, values, color in (("p50", [b.p50_ms for b in rows], NEUTRAL),
                                ("p95", [b.p95_ms for b in rows], ACCENT)):  # fmt: skip
        fig.add_scatter(x=x, y=[None if v is None else v / 1000 for v in values], name=name,
                        mode="lines+markers", line={"color": color, "width": 2})  # fmt: skip
    fig.update_yaxes(title_text="seconds")
    fig.update_layout(legend={"orientation": "h", "y": 1.15})
    return _layout(fig, height)


def observe_volume_chart(rows: Sequence[Bucket], height: int = 220) -> go.Figure:
    """Queries per bucket, stacked by source."""
    sources = sorted({s for b in rows for s in b.counts})
    fig = go.Figure([go.Bar(x=[b.start for b in rows], y=[b.counts.get(s, 0) for b in rows],
                            name=s, marker_color=SOURCE_COLORS.get(s, MUTED))
                     for s in sources])  # fmt: skip
    fig.update_layout(barmode="stack", legend={"orientation": "h", "y": 1.15})
    fig.update_yaxes(title_text="queries")
    return _layout(fig, height)


def observe_tokens_chart(rows: Sequence[Bucket], height: int = 220) -> go.Figure:
    """Tokens per bucket; the cost is in the hover text."""
    fig = go.Figure(go.Bar(x=[b.start for b in rows], y=[b.tokens for b in rows],
                           marker_color=NEUTRAL, name="tokens",
                           hovertext=[f"${b.cost_usd:.4f}" for b in rows]))  # fmt: skip
    fig.update_yaxes(title_text="tokens")
    return _layout(fig, height)


def eval_history_chart(label: str, points: Sequence[EvalPoint], baseline: float | None,
                       threshold: float | None, height: int = 200) -> go.Figure:  # fmt: skip
    """One metric over evaluation runs, with the committed baseline and the threshold."""
    fig = go.Figure(go.Scatter(x=[p.ts for p in points], y=[p.value for p in points],
                               mode="lines+markers", name=label,
                               hovertext=[p.kind for p in points],
                               line={"color": NEUTRAL, "width": 2}))  # fmt: skip
    if baseline is not None:
        fig.add_hline(y=baseline, line_dash="dot", line_color=MUTED,
                      annotation_text="baseline")  # fmt: skip
    if threshold is not None:
        fig.add_hline(y=threshold, line_dash="dash", line_color="#dc2626",
                      annotation_text="threshold")  # fmt: skip
    fig.update_yaxes(range=[0, 1.05])
    return _layout(fig, height, label)
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_observe_dashboard.py tests/unit/test_charts.py -q`
Expected: PASS

- [ ] **Step 5: Lint and types**

Run: `uv run ruff check . && uv run ruff format . && uv run mypy src tests`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/rag_eval_platform/observability/dashboard.py src/rag_eval_platform/playground/charts.py tests/unit/test_observe_dashboard.py tests/unit/test_charts.py
git commit -m "feat: observability dashboard data and charts (latency, volume, tokens, evaluation history)"
```

---

### Task 9: Playground — record answers, keep ratings, ⑧ Observe tab

**Files:**
- Modify: `src/rag_eval_platform/playground/core.py`, `playground/chat.py`, `playground/tabs/overview.py`, `playground/tabs/chat_panel.py`, `playground/shared.py`, `playground/app.py`
- Create: `src/rag_eval_platform/playground/tabs/observe.py`
- Test: `tests/unit/test_playground_observe.py`, `tests/integration/test_playground.py`

**Interfaces:**
- Consumes: `create_query_logger`, `create_store`, `record_feedback`, `ObserveStoreError`, `UnknownQueryError`, dashboard, review, drift, charts.
- Produces: `record_answer(settings, answer, *, retrieve_ms, total_ms, style) -> str | None`, `record_failure(settings, question, error, *, total_ms, style) -> str | None`, `save_rating(settings, query_id: str | None, value: int) -> bool` (all in `core.py`); `Turn.query_id: str | None = None`; `TAB_LABELS` ends with `"⑧ Observe"`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/test_playground_observe.py`:

```python
"""Tests for the playground's observability helpers (core.py)."""

from pathlib import Path

from tests.unit.test_api_app import answer

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.store import ObserveStore
from rag_eval_platform.playground.chat import Turn
from rag_eval_platform.playground.core import (
    PLAYGROUND_COLLECTION,
    record_answer,
    record_failure,
    save_rating,
)


def on(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, observe_db=tmp_path / "rag.db")


def test_answers_are_logged_as_playground_queries(tmp_path: Path) -> None:
    settings = on(tmp_path)

    query_id = record_answer(settings, answer(), retrieve_ms=5, total_ms=30, style="detailed")

    saved = ObserveStore(settings.observe_db).get_query(query_id or "")
    assert saved is not None
    assert (saved.source, saved.collection, saved.style) == (
        "playground", PLAYGROUND_COLLECTION, "detailed")  # fmt: skip


def test_failures_are_logged(tmp_path: Path) -> None:
    settings = on(tmp_path)

    query_id = record_failure(settings, "q?", ValueError("bad"), total_ms=3, style="concise")

    saved = ObserveStore(settings.observe_db).get_query(query_id or "")
    assert saved is not None and saved.status == "error"


def test_a_rating_is_saved_for_a_logged_answer(tmp_path: Path) -> None:
    settings = on(tmp_path)
    query_id = record_answer(settings, answer(), retrieve_ms=5, total_ms=30, style="concise")

    assert save_rating(settings, query_id, 0) is True
    [saved] = ObserveStore(settings.observe_db).feedback()
    assert (saved.query_id, saved.rating, saved.source) == (query_id, 0, "playground")


def test_rating_an_unlogged_answer_does_nothing(tmp_path: Path) -> None:
    settings = on(tmp_path)

    assert save_rating(settings, None, 1) is False
    assert save_rating(settings, "never-logged", 1) is False
    assert save_rating(Settings(_env_file=None, observe_enabled=False), "x", 1) is False


def test_logging_off_records_nothing(tmp_path: Path) -> None:
    off = Settings(_env_file=None, observe_enabled=False, observe_db=tmp_path / "rag.db")

    assert record_answer(off, answer(), retrieve_ms=1, total_ms=2, style="concise") is None
    assert not (tmp_path / "rag.db").exists()


def test_a_turn_keeps_its_query_id() -> None:
    assert Turn.__dataclass_fields__["query_id"].default is None
```

In `tests/integration/test_playground.py`, add `"⑧ Observe"` as the last expected tab label.

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_playground_observe.py -q`
Expected: FAIL with `ImportError: cannot import name 'record_answer'`

- [ ] **Step 3: Implement the helpers** — in `playground/core.py` (with the imports `create_query_logger`, `create_store`, `record_feedback` from `observability.logger`; `ObserveStoreError`, `UnknownQueryError` from `observability.store`; `logging`, `Answer` if not present):

```python
def record_answer(settings: Settings, answer: Answer, *, retrieve_ms: float, total_ms: float,
                  style: str) -> str | None:  # fmt: skip
    """Log a playground answer (collection ``playground``); None when logging is off or failed."""
    recorder = create_query_logger(settings, "playground", collection=PLAYGROUND_COLLECTION,
                                   style=style)  # fmt: skip
    return recorder.record(answer, retrieve_ms=retrieve_ms, total_ms=total_ms) if recorder else None


def record_failure(settings: Settings, question: str, error: BaseException, *, total_ms: float,
                   style: str) -> str | None:  # fmt: skip
    recorder = create_query_logger(settings, "playground", collection=PLAYGROUND_COLLECTION,
                                   style=style)  # fmt: skip
    return recorder.record_error(question, error, total_ms=total_ms) if recorder else None


def save_rating(settings: Settings, query_id: str | None, value: int) -> bool:
    """Store a 👍 (1) or 👎 (0); False when the answer was never logged or the store fails."""
    store = create_store(settings)
    if store is None or query_id is None:
        return False
    try:
        record_feedback(store, query_id, value, None, "playground")
    except (ObserveStoreError, UnknownQueryError) as exc:
        logger.warning("Could not save the rating", extra={"error": str(exc)})
        return False
    return True
```

In `playground/chat.py`, add `query_id: str | None = None  # the observability log's id` as the last field of `Turn`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_playground_observe.py -q`
Expected: PASS (6 tests)

- [ ] **Step 5: Wire the UI** (drawing code: not unit-tested, smoke-tested with `AppTest` and in the browser)

`tabs/overview.py` — in the generate step, time the whole question from the existing `started` (a `time.perf_counter()` value) and:

```python
    except (GenerationError, OptionalDependencyError, ValueError) as exc:
        record_failure(settings, question, exc, style=options.answer_style,
                       total_ms=(time.perf_counter() - started) * 1000)  # fmt: skip
        _fail(status, bubble, str(exc))
        return ctx

    answer = stream.answer
    query_id = record_answer(settings, answer, retrieve_ms=step.retrieval_ms,
                             total_ms=(time.perf_counter() - started) * 1000,
                             style=options.answer_style)  # fmt: skip
    trace = QueryTrace(...)  # unchanged
    turn = Turn(trace, first_token_s, query_id=query_id)
```

(If `started` is not in scope inside `_generate`, pass it in from the caller that defines it at line ~126, the same way `step` is passed. Record before any further `st.*` call, as the file's rule says.)

`tabs/chat_panel.py` — after the rating is stored in history:

```python
    if value is not None and value != turn.feedback:
        history: tuple[Turn, ...] = st.session_state.get(HISTORY, ())
        if index < len(history):
            st.session_state[HISTORY] = with_feedback(history, index, value)
            save_rating(settings, turn.query_id, value)
```

`shared.py` — append `"⑧ Observe"` to `TAB_LABELS`. `app.py` — import `observe` from `playground.tabs` and add `"⑧ Observe": observe.render` to `RENDERERS`.

Create `src/rag_eval_platform/playground/tabs/observe.py`:

```python
"""⑧ Observe: traffic, latency, cost, quality over time, drift warnings and a review queue."""

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from rag_eval_platform.observability.cost import Prices
from rag_eval_platform.observability.dashboard import (
    EVAL_METRICS,
    RANGES,
    baseline_values,
    bucket_size,
    buckets,
    cost_label,
    eval_history,
    metric_thresholds,
    query_rows,
    since_for,
    summarize,
)
from rag_eval_platform.observability.drift import DriftThresholds, drift_report
from rag_eval_platform.observability.logger import create_store
from rag_eval_platform.observability.review import REASON_LABELS, review_queue
from rag_eval_platform.observability.store import ObserveStore, ObserveStoreError
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.shared import TabContext, settings

BASELINES = Path("baselines")
EMPTY = "No queries yet — ask one in the chat or the API."


def render(ctx: TabContext) -> None:
    store = create_store(settings)
    if store is None:
        st.info("Observability is off (RAG_OBSERVE_ENABLED=false).")
        return
    try:
        _render(store)
    except ObserveStoreError as exc:
        st.error(str(exc))


def _render(store: ObserveStore) -> None:
    all_queries, all_feedback, runs = store.queries(), store.feedback(), store.eval_runs()
    report = drift_report(all_queries, all_feedback, runs, DriftThresholds.from_settings(settings))
    for warning in report.warnings:
        st.warning(warning, icon="⚠️")
    if not report.enough_data:
        st.caption(f"Drift checks need {report.need} answers; {report.have} logged so far.")

    label = st.segmented_control("Time range", list(RANGES), default="7 days",
                                 key="obs-range", required=True) or "7 days"  # fmt: skip
    since = since_for(label, datetime.now(UTC))
    queries = [q for q in all_queries if since is None or q.ts >= since]
    summary = summarize(queries, all_feedback)
    tiles = st.columns(8)
    values = (
        ("Queries", f"{summary.queries:,}"),
        ("p50", _seconds(summary.p50_ms)), ("p95", _seconds(summary.p95_ms)),
        ("Errors", _percent(summary.error_rate)), ("Refusals", _percent(summary.refusal_rate)),
        ("👍 rate", _percent(summary.thumbs_up_rate)), ("Tokens", f"{summary.tokens:,}"),
        ("Cost", cost_label(summary, Prices.from_settings(settings))),
    )  # fmt: skip
    for tile, (name, value) in zip(tiles, values, strict=True):
        tile.metric(name, value)

    if queries:
        rows = buckets(queries, bucket_size(label))
        left, middle, right = st.columns(3)
        left.plotly_chart(charts.observe_latency_chart(rows), key="obs-latency",
                          config=charts.CHART_CONFIG)  # fmt: skip
        middle.plotly_chart(charts.observe_volume_chart(rows), key="obs-volume",
                            config=charts.CHART_CONFIG)  # fmt: skip
        right.plotly_chart(charts.observe_tokens_chart(rows), key="obs-tokens",
                           config=charts.CHART_CONFIG)  # fmt: skip
    else:
        st.info(EMPTY)
    _evaluation(runs)
    _review(store, queries, all_feedback)
    _recent(store, queries, all_feedback)


def _evaluation(runs: list) -> None:  # type: ignore[type-arg]
    st.subheader("Evaluation history")
    history, baselines = eval_history(runs), baseline_values(BASELINES)
    thresholds = metric_thresholds(settings)
    if not runs:
        st.caption("No evaluation runs logged yet; the lines show the committed baselines. "
                   "Run scripts/run_evaluation.py or scripts/run_gate.py to add points.")  # fmt: skip
    columns = st.columns(len(EVAL_METRICS))
    for column, (metric, name) in zip(columns, EVAL_METRICS.items(), strict=True):
        column.plotly_chart(
            charts.eval_history_chart(name, history[metric], baselines.get(metric),
                                      thresholds.get(metric)),
            key=f"obs-eval-{metric}", config=charts.CHART_CONFIG,
        )  # fmt: skip


def _review(store: ObserveStore, queries: list, feedback: list) -> None:  # type: ignore[type-arg]
    st.subheader("Review queue")
    show_reviewed = st.toggle("Show reviewed", key="obs-show-reviewed")
    items = review_queue(queries, feedback, slow_ms=settings.review_slow_ms,
                         include_reviewed=show_reviewed)  # fmt: skip
    if not items:
        st.caption("Nothing to review: no 👎, refusals, invalid citations, errors or slow answers.")
    for item in items[:20]:
        with st.container(border=True):
            badges = " · ".join(REASON_LABELS[r] for r in item.reasons)
            st.markdown(f"**{badges}** — {item.query.ts:%Y-%m-%d %H:%M} · {item.query.source}")
            st.text(item.query.question)
            if item.comment:
                st.caption(f"Comment: {item.comment}")
            if item.query.answer:
                with st.expander("Answer"):
                    st.text(item.query.answer)
            if item.query.error:
                st.caption(item.query.error)
            if item.query.reviewed_at is None and st.button(
                    "Mark reviewed", key=f"obs-reviewed-{item.query.id}"):  # fmt: skip
                store.mark_reviewed(item.query.id, datetime.now(UTC))
                st.rerun()


def _recent(store: ObserveStore, queries: list, feedback: list) -> None:  # type: ignore[type-arg]
    st.subheader("Recent queries")
    source = st.segmented_control("Source", ["all", "api", "playground", "cli"], default="all",
                                  key="obs-source", required=True) or "all"  # fmt: skip
    rows = query_rows(queries, feedback, None if source == "all" else source)
    if not rows:
        st.caption(EMPTY)
        return
    picked = st.dataframe(rows, hide_index=True, key="obs-recent", on_select="rerun",
                          selection_mode="single-row", column_config={"id": None})  # fmt: skip
    selected = picked.selection.rows if picked else []
    if selected:
        query = store.get_query(str(rows[selected[0]]["id"]))
        if query is not None:
            with st.container(border=True):
                st.markdown(f"**{query.question}**")
                st.markdown(query.answer or query.error or "")
                st.caption(f"Cited: {', '.join(query.cited_doc_ids) or 'none'} · "
                           f"model {query.model} · prompt {query.prompt_version} · "
                           f"retrieve {_seconds(query.retrieve_ms)} · "
                           f"generate {_seconds(query.generate_ms)} · "
                           f"total {_seconds(query.total_ms)}")  # fmt: skip


def _seconds(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f} s"


def _percent(rate: float | None) -> str:
    return "—" if rate is None else f"{rate:.0%}"
```

(Replace the bare `list` annotations with `list[EvalRunRecord]`, `list[QueryRecord]`, `list[FeedbackRecord]` imported from the store module, and drop the `type: ignore` comments.)

- [ ] **Step 6: Run the tests, lint and types**

Run: `uv run pytest tests/unit -q && uv run pytest tests/integration/test_playground.py -q -k renders && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`
Expected: PASS (the AppTest sees eight tabs plus ⑧ Observe), clean

- [ ] **Step 7: Check it in the browser**

Run: `uv run streamlit run src/rag_eval_platform/playground/app.py` and open http://localhost:8501. Ask a question in the chat, rate it 👎, then open ⑧ Observe.
Expected: one playground query in the tiles and the recent table; it is in the review queue with the 👎 badge; *Mark reviewed* removes it; the evaluation history shows the baseline and threshold lines; the time range and source filters work; no exception.

- [ ] **Step 8: Commit**

```bash
git add src/rag_eval_platform/playground/core.py src/rag_eval_platform/playground/chat.py src/rag_eval_platform/playground/tabs/overview.py src/rag_eval_platform/playground/tabs/chat_panel.py src/rag_eval_platform/playground/tabs/observe.py src/rag_eval_platform/playground/shared.py src/rag_eval_platform/playground/app.py tests/unit/test_playground_observe.py tests/integration/test_playground.py
git commit -m "feat: playground logs answers and ratings; new ⑧ Observe tab"
```

---

### Task 10: Docker — writable log folder and bind mount

**Files:**
- Modify: `docker/Dockerfile`, `docker/docker-compose.yml`
- Test: `tests/unit/test_dockerfile.py`

- [ ] **Step 1: Write the failing test** (append to `tests/unit/test_dockerfile.py`)

```python
COMPOSE = DOCKERFILE.parent / "docker-compose.yml"


def test_the_api_container_can_write_the_observability_log() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    compose = COMPOSE.read_text(encoding="utf-8")

    assert "mkdir -p /app/data/observability" in dockerfile
    assert "chown app:app /app/data/observability" in dockerfile
    assert "../data/observability:/app/data/observability" in compose
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_dockerfile.py -q`
Expected: FAIL on the first `assert`

- [ ] **Step 3: Implement**

`docker/Dockerfile` — after `COPY --from=builder --chown=app:app /app /app` and before `USER app`:

```dockerfile
# The observability log (Phase 7); compose bind-mounts the host's data/observability here.
RUN mkdir -p /app/data/observability && chown app:app /app/data/observability
```

`docker/docker-compose.yml` — in the `api` service, after `extra_hosts`:

```yaml
    volumes:
      # The observability log: the playground on the host reads the same rag.db.
      - ../data/observability:/app/data/observability
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_dockerfile.py -q`
Expected: PASS

- [ ] **Step 5: Verify with the real container**

Run:
```bash
mkdir -p data/observability
docker compose -f docker/docker-compose.yml --profile api up -d --build
curl -s localhost:8000/health
ID=$(curl -s -X POST localhost:8000/query -H 'Content-Type: application/json' -d '{"question":"What does MRR measure?"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['query_id'])")
curl -s -X POST localhost:8000/feedback -H 'Content-Type: application/json' -d "{\"query_id\":\"$ID\",\"rating\":\"up\"}"
python3 -c "import sqlite3;c=sqlite3.connect('data/observability/rag.db');print(c.execute('select source,status from queries').fetchall(), c.execute('select rating from feedback').fetchall())"
```
Expected: `/health` 200 with `"warnings": []`; a non-null `query_id`; feedback `{"id": ...}` (201); the host sees `[('api', 'ok')] [(1,)]`; the ⑧ Observe tab on the host shows the API query. If the container cannot write (permission denied in its log), record a ledger ruling and fix ownership (e.g. `user: "${UID:-10001}"`), then re-verify.

- [ ] **Step 6: Commit**

```bash
git add docker/Dockerfile docker/docker-compose.yml tests/unit/test_dockerfile.py
git commit -m "build: API container writes the observability log through a bind mount"
```

---

### Task 11: Docs

**Files:**
- Create: `docs/learning/phase-7.md`
- Modify: `readme.md`, `CLAUDE.md`, `docs/architecture.md`, `docs/learning/playground.md`

- [ ] **Step 1: Write `docs/learning/phase-7.md`** in the style of `docs/learning/phase-6.md` (Part A ideas, Part B code, Try it), with these sections and diagrams:
  - Intro: "Phase 7 gives the system a memory": every answer, rating and evaluation run is kept, so you can see how it behaves over time. Analogy: a car's dashboard and service log.
  - Mermaid mindmap: Log (queries, feedback, eval runs) · See (⑧ Observe tiles, charts) · Act (review queue, mark reviewed) · Warn (drift, `/health` warnings).
  - "Where the data comes from": the flowchart from the spec's Architecture section.
  - "What is stored": the three tables in plain words; the `RAG_OBSERVE_STORE_TEXT` privacy switch; the file is git-ignored.
  - "Drift, explained": last 20 vs previous 20 answers; the table of checks and defaults; why count-based windows; an example warning.
  - "The review loop": sequence diagram — user rates 👎 → queue → you read it → fix (prompt, chunking, corpus) → evaluation run → history chart shows the effect.
  - "Cost": tokens × price; local Ollama = $0; how to set OpenAI prices.
  - Part B: table of `store.py`, `logger.py`, `cost.py`, `review.py`, `drift.py`, `dashboard.py`, the API changes (`query_id`, `POST /feedback` with a curl example, `/health` warnings), and `tabs/observe.py`.
  - Try it: ask three questions in the chat, rate one 👎, open ⑧ Observe, mark it reviewed, run `scripts/run_evaluation.py` and watch a history point appear, send feedback with curl.
- [ ] **Step 2: Update the other docs**
  - `readme.md`: Phase 7 row → "7a ✅ core (SQLite log, ⑧ Observe, feedback, drift) · 7b 🔜 TruLens + Langfuse"; add a ⑧ Observe row to the tabs table; add `POST /feedback` to the API section; add `phase-7.md` to the docs list; mention `data/observability/rag.db`.
  - `CLAUDE.md`: status (Phase 7a done; observability no longer a placeholder; 7b next); a new architecture bullet "**Observability**" (store, logger never raises, sources, drift windows, review reasons, `/feedback`, `/health` warnings, eval history rows, `RAG_OBSERVE_*`, bind mount); replace the "Every answered question is logged…" bullet's last clause to say it is also stored when the pipeline has a recorder; tabs now "nine" with ⑧ Observe; the conftest isolation fixture.
  - `docs/architecture.md`: tooling row "SQLite (`sqlite3`, standard library) — query log, feedback, evaluation history — in use"; the Streamlit dashboard row → in use (⑧ Observe); TruLens → "chosen, Phase 7b (without trulens-dashboard: it pins plotly<7)"; add Langfuse "chosen, Phase 7b"; Deployment section: the `data/observability` bind mount.
  - `docs/learning/playground.md`: a ⑧ Observe row, mindmap entry and section.
- [ ] **Step 3: Check the Mermaid blocks render** (GitHub's parser rejects some labels): keep labels in quotes and avoid parentheses outside quotes, as `phase-6.md` does.
- [ ] **Step 4: Commit**

```bash
git add docs/learning/phase-7.md readme.md CLAUDE.md docs/architecture.md docs/learning/playground.md
git commit -m "docs: phase 7a observability learning page, readme, architecture and CLAUDE.md"
```

---

## Final verification (after Task 11)

Run: `uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit --cov -q && uv run pytest -m integration -q`
Expected: all pass; coverage ≥ 80%; integration tests pass or skip only for missing services.
