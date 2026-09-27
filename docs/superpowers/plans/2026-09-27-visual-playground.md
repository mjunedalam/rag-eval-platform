# Visual Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Streamlit playground into six tabs (Overview + one per RAG phase) that show every Phase 1–4 concept live, with animated Plotly charts and Graphviz flow diagrams.

**Architecture:** Pure, unit-tested data functions (`playground/visuals.py`, `playground/query_visuals.py`, `playground/animation.py`, new pieces of `playground/core.py`) feed thin Plotly builders (`playground/charts.py`) that thin Streamlit tab modules (`playground/tabs/*.py`) draw. One `QueryTrace` in `st.session_state` drives every tab; tabs are lazy (`st.tabs(..., on_change="rerun")`), so only the open tab runs.

**Tech Stack:** Python 3.12, Streamlit 1.64 (lazy tabs, `st.graphviz_chart`), Plotly 7.1 (`ui` group), numpy (PCA), existing Chroma / Ollama / RAGAS pieces.

**Spec:** `docs/superpowers/specs/2026-09-27-visual-playground-design.md`

## Global Constraints

- Tooling: `uv` only; Python 3.12; Ruff (line length 100), `mypy src tests` strict, pytest with `--strict-markers`.
- **No git commits or pushes until the user explicitly asks.** Each task ends with a checks checkpoint instead of a commit.
- Plotly lives only in the `ui` dependency group; CI never installs it. Chart tests use `pytest.importorskip("plotly")`; mypy skips `plotly` like `streamlit`.
- `app.py`, `shared.py`, `tabs/*` and `charts.py` are excluded from coverage; everything else keeps ≥ 80% coverage.
- Every `st.plotly_chart` call passes a unique `key`; animation frames use `f"{key}-{i}"` and the static draw `f"{key}-final"` (verified: repeating a key raises `StreamlitDuplicateElementKey`, identical keyless charts raise `StreamlitDuplicateElementId`).
- Streamlit rule: store a long operation's result in `st.session_state` before any further `st.*` call.
- Uploads stay in git-ignored `data/playground/` and the `playground` Chroma collection.
- Text from documents or the LLM is HTML-escaped before `unsafe_allow_html`, and quote-escaped inside DOT labels.

## Review Focus

- App restarted after indexing (session state lost) → Ingest/Embed still draw from Chroma via `export()`; Retrieve/Generate/Evaluate ask for a question. Pinned by `test_index_snapshot_*` (Task 2).
- Answer is a refusal or cites nothing → Generate shows plain text, Evaluate refuses to judge with a message. Pinned by `test_judge_answer_rejects_refusals` (Task 2) and `test_answer_spans_without_citations` (Task 5).
- Re-ranking toggled on/off → cut-off line only without re-ranking; re-rank chart only with it. Pinned by `test_similarity_rows_*` and `test_was_reranked` (Task 5), `test_similarity_chart_cutoff_*` (Task 7).
- Index larger than the meaning-map cap → sampled, retrieved chunks always kept. Pinned by `test_meaning_map_samples_but_keeps_retrieved` (Task 4).
- Document or answer text containing `<`, `&` or `"` → escaped in HTML chips and DOT labels. Pinned by `test_spans_html_escapes_text` (Task 5) and `test_pipeline_dot_escapes_quotes`/`test_verdict_dot_escapes_quotes` (Tasks 3, 5).

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `src/rag_eval_platform/retrieval/vector_store.py` | modify | `ChromaVectorStore.export()`; shared `_to_chunk` |
| `src/rag_eval_platform/generation/generator.py` | modify | expose `CITATION_PATTERN` |
| `src/rag_eval_platform/playground/core.py` | modify | `query_embedding` on traces, `IndexSnapshot`, `index_snapshot`, `trace_key`, `JudgeResult`, `judge_answer`, `Reports`, `load_reports` |
| `src/rag_eval_platform/playground/animation.py` | create | `ease_steps`, `fingerprint`, `should_animate` |
| `src/rag_eval_platform/playground/visuals.py` | create | pipeline DOT, index counts, ingest flow DOT, chunk spans, size bins, meaning map, `vector_for`, `angle_degrees` |
| `src/rag_eval_platform/playground/query_visuals.py` | create | similarity rows, re-rank moves, funnel, prompt blocks, answer spans + HTML, timing rows, verdict DOT, report join |
| `src/rag_eval_platform/playground/charts.py` | create | Plotly figure builders (all take `progress` and `height`) |
| `src/rag_eval_platform/playground/shared.py` | create | settings, `Options`, cached resources, sidebar, status, pipeline header, `animated_chart`, `TabContext` |
| `src/rag_eval_platform/playground/tabs/__init__.py` | create | package marker |
| `src/rag_eval_platform/playground/tabs/{ingest,embed,retrieve,generate,evaluate,overview}.py` | create | one `render(ctx)` each |
| `src/rag_eval_platform/playground/app.py` | rewrite | page shell, question box, streaming, lazy tabs |
| `tests/unit/test_vector_store.py` | modify | `export()` tests |
| `tests/unit/test_playground_core.py` | modify | new core tests |
| `tests/unit/test_animation.py`, `test_visuals.py`, `test_query_visuals.py`, `test_charts.py` | create | unit tests |
| `tests/integration/test_playground.py` | modify | new tab labels |
| `pyproject.toml` / `uv.lock` | modify | `plotly` in `ui`, `numpy` explicit, mypy + coverage settings |
| docs: `readme.md`, `CLAUDE.md`, `docs/architecture.md`, `docs/learning/phase-4.md` | modify | describe the visual playground |

---

### Task 1: `ChromaVectorStore.export()`

**Files:**
- Modify: `src/rag_eval_platform/retrieval/vector_store.py`
- Test: `tests/unit/test_vector_store.py`

**Interfaces:**
- Produces: `ChromaVectorStore.export(self) -> tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]` — every stored chunk and its vector, sorted by `(doc_id, index)`; `((), ())` when the collection is missing. Raises `VectorStoreError` on malformed responses.

- [ ] **Step 1: Write the failing tests**

Add a `get` method to the existing `FakeCollection` in `tests/unit/test_vector_store.py` (insertion order = Chroma's order):

```python
    def get(self, include: list[str], limit: int, offset: int) -> dict[str, Any]:
        items = list(self.rows.items())[offset : offset + limit]
        return {
            "ids": [id_ for id_, _ in items],
            "embeddings": [row[0] for _, row in items],
            "documents": [row[1] for _, row in items],
            "metadatas": [row[2] for _, row in items],
        }
```

Append the tests:

```python
class TestExport:
    def test_returns_every_chunk_and_vector_in_document_order(self) -> None:
        client = FakeChromaClient(max_batch_size=2)
        store = ChromaVectorStore(client, "c")  # type: ignore[arg-type]
        chunks = [
            Chunk(id="b.md#0", doc_id="b.md", index=0, text="bee", start_index=0),
            Chunk(id="a.md#1", doc_id="a.md", index=1, text="second", start_index=5, page=2),
            Chunk(id="a.md#0", doc_id="a.md", index=0, text="first", start_index=0, page=1),
        ]
        store.replace_all(chunks, [[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]])

        exported, vectors = store.export()

        assert [c.id for c in exported] == ["a.md#0", "a.md#1", "b.md#0"]
        assert exported[1] == chunks[1]
        assert vectors == ((0.6, 0.8), (1.0, 0.0), (0.0, 1.0))

    def test_missing_collection_exports_nothing(self) -> None:
        store = ChromaVectorStore(FakeChromaClient(), "absent")  # type: ignore[arg-type]

        assert store.export() == ((), ())

    def test_malformed_response_is_a_vector_store_error(self) -> None:
        client = FakeChromaClient()
        store = ChromaVectorStore(client, "c")  # type: ignore[arg-type]
        store.replace_all([Chunk(id="a#0", doc_id="a", index=0, text="t", start_index=0)], [[1.0]])
        client.collections["c"].rows["a#0"] = ([1.0], "t", {})  # metadata lost

        with pytest.raises(VectorStoreError, match="Unexpected response"):
            store.export()
```

(If the existing tests construct `ChromaVectorStore` without a `type: ignore`, match their style.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_vector_store.py -k Export -v`
Expected: FAIL with `AttributeError: 'ChromaVectorStore' object has no attribute 'export'`

- [ ] **Step 3: Implement**

In `vector_store.py`, add after `count()`:

```python
    def export(self) -> tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]:
        """Every stored chunk and its vector, sorted by document and position.

        For inspection tools (the playground's charts), not for retrieval.
        """
        collection = self._collection()
        if collection is None:
            return (), ()
        batch_size = self._client.get_max_batch_size()
        rows: list[tuple[Chunk, tuple[float, ...]]] = []
        for offset in range(0, collection.count(), batch_size):
            page = collection.get(
                include=["documents", "metadatas", "embeddings"], limit=batch_size, offset=offset
            )
            try:
                for chunk_id, document, metadata, embedding in zip(
                    page["ids"], page["documents"], page["metadatas"], page["embeddings"],
                    strict=True,
                ):  # fmt: skip
                    vector = tuple(float(x) for x in embedding)
                    rows.append((_to_chunk(chunk_id, document, metadata), vector))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise VectorStoreError(f"Unexpected response from Chroma: {exc}") from exc
        rows.sort(key=lambda row: (row[0].doc_id, row[0].index))
        return tuple(r[0] for r in rows), tuple(r[1] for r in rows)
```

Extract the chunk-building code shared with `_to_results`:

```python
def _to_chunk(chunk_id: str, document: str, metadata: Any) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id=str(metadata["doc_id"]),
        index=int(metadata["index"]),
        text=document,
        start_index=int(metadata["start_index"]),
        page=int(metadata["page"]) if "page" in metadata else None,
    )
```

and change `_to_results` to use it:

```python
        return [
            SearchResult(chunk=_to_chunk(chunk_id, document, metadata), score=1.0 - float(distance))
            for chunk_id, document, metadata, distance in rows
        ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_vector_store.py -v`
Expected: all PASS (old and new).

- [ ] **Step 5: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`
Expected: clean.

---

### Task 2: Core additions — query embedding, snapshot, judging, reports

**Files:**
- Modify: `src/rag_eval_platform/playground/core.py`
- Test: `tests/unit/test_playground_core.py`

**Interfaces:**
- Consumes: `ChromaVectorStore.export()` (Task 1); `Judge`, `JudgeSample`, `JudgeScores`, `JudgeError` from `evaluation.judge`; `CitationValidity` from `evaluation.citation_validity`.
- Produces:
  - `RetrievalStep.query_embedding: tuple[float, ...]` (new last field)
  - `QueryTrace(vector_results, final_results, answer, retrieval_ms, query_embedding: tuple[float, ...] = ())`
  - `IndexSnapshot(chunks: tuple[Chunk, ...], vectors: tuple[tuple[float, ...], ...])` with `.doc_ids -> tuple[str, ...]`
  - `index_snapshot(store: _Exportable) -> IndexSnapshot`
  - `trace_key(trace: QueryTrace) -> str`
  - `JudgeResult(scores: JudgeScores, citations: CitationValidity, judge_ms: float, note: str = "")`
  - `judge_answer(trace, judge: Judge, checker: _CitationChecker) -> JudgeResult` (raises `ValueError` for refusals)
  - `Reports(retrieval: dict[str, Any] | None, generation: dict[str, Any] | None)`, `REPORTS_DIR = Path("reports")`, `load_reports(directory: Path) -> Reports`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_playground_core.py` (reuse its existing fakes for embedder/store where they exist; the ones below are self-contained):

```python
import json as _json
from dataclasses import dataclass as _dataclass

from rag_eval_platform.evaluation.citation_validity import CitationValidity, CitationVerdict
from rag_eval_platform.evaluation.judge import JudgeError, JudgeSample, JudgeScores
from rag_eval_platform.generation.generator import Answer
from rag_eval_platform.generation.prompt_templates import NO_ANSWER
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import (
    IndexSnapshot,
    JudgeResult,
    QueryTrace,
    Reports,
    index_snapshot,
    judge_answer,
    load_reports,
    retrieve_step,
    trace_key,
)
from rag_eval_platform.retrieval.vector_store import SearchResult


def _hit(chunk_id: str, text: str = "text", score: float = 0.5) -> SearchResult:
    doc_id = chunk_id.split("#")[0]
    return SearchResult(Chunk(id=chunk_id, doc_id=doc_id, index=0, text=text, start_index=0), score)


def _answer(text: str, sources: tuple[SearchResult, ...], question: str = "q?") -> Answer:
    return Answer(
        question=question, text=text, citations=(), invalid_citations=(), sources=sources,
        model="m", prompt_version="v1", input_tokens=1, output_tokens=1, latency_ms=10.0,
    )  # fmt: skip


def _trace(text: str = "A [1].", question: str = "q?") -> QueryTrace:
    sources = (_hit("a.md#0", "alpha"), _hit("b.md#0", "beta"))
    return QueryTrace(sources, sources, _answer(text, sources, question), 5.0, (0.1, 0.2))


class _VectorEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.6, 0.8]


class _OneHitStore:
    def search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        return [_hit("a.md#0")]


def test_retrieve_step_keeps_the_query_embedding() -> None:
    step = retrieve_step("q?", embedder=_VectorEmbedder(), store=_OneHitStore(), top_k=1)  # type: ignore[arg-type]

    assert step.query_embedding == (0.6, 0.8)


def test_query_trace_embedding_defaults_to_empty() -> None:
    sources = (_hit("a.md#0"),)
    trace = QueryTrace(sources, sources, _answer("A [1].", sources), 1.0)

    assert trace.query_embedding == ()


class _ExportingStore:
    def export(self) -> tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]:
        chunks = (
            Chunk(id="a.md#0", doc_id="a.md", index=0, text="x", start_index=0),
            Chunk(id="b.md#0", doc_id="b.md", index=0, text="y", start_index=0),
            Chunk(id="a.md#1", doc_id="a.md", index=1, text="z", start_index=1),
        )
        return chunks, ((1.0,), (2.0,), (3.0,))


def test_index_snapshot_reads_the_store() -> None:
    snapshot = index_snapshot(_ExportingStore())

    assert len(snapshot.chunks) == 3
    assert snapshot.vectors[1] == (2.0,)
    assert snapshot.doc_ids == ("a.md", "b.md")


def test_index_snapshot_of_an_empty_store_is_empty() -> None:
    class Empty:
        def export(self) -> tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]:
            return (), ()

    assert index_snapshot(Empty()) == IndexSnapshot((), ())


def test_trace_key_changes_with_question_and_answer() -> None:
    assert trace_key(_trace()) == trace_key(_trace())
    assert trace_key(_trace(question="other?")) != trace_key(_trace())
    assert trace_key(_trace(text="B [2].")) != trace_key(_trace())


@_dataclass
class _Judge:
    model: str = "judge"
    samples: list[JudgeSample] | None = None

    def score(self, sample: JudgeSample, *, full: bool = False) -> JudgeScores:
        self.samples = [sample]
        return JudgeScores(faithfulness=0.5, answer_relevance=0.9)


class _Checker:
    def __init__(self, error: bool = False) -> None:
        self.error = error

    def check(self, answer: Answer) -> CitationValidity:
        if self.error:
            raise JudgeError("unreadable")
        return CitationValidity((CitationVerdict("A.", 1, "a.md", True),))


def test_judge_answer_scores_the_answer_against_its_chunks() -> None:
    judge = _Judge()

    result = judge_answer(_trace(), judge, _Checker())

    assert result.scores == JudgeScores(faithfulness=0.5, answer_relevance=0.9)
    assert result.citations.score == 1.0
    assert result.judge_ms >= 0
    assert result.note == ""
    assert judge.samples is not None
    assert judge.samples[0].contexts == ("alpha", "beta")


def test_judge_answer_keeps_scores_when_citation_verdicts_fail() -> None:
    result = judge_answer(_trace(), _Judge(), _Checker(error=True))

    assert result.citations.score is None
    assert "unreadable" in result.note


def test_judge_answer_rejects_refusals() -> None:
    with pytest.raises(ValueError, match="refusal"):
        judge_answer(_trace(text=NO_ANSWER), _Judge(), _Checker())


def test_load_reports_reads_both_files(tmp_path: Path) -> None:
    (tmp_path / "retrieval_report.json").write_text(_json.dumps({"k": 5}), encoding="utf-8")
    (tmp_path / "generation_report.json").write_text(
        _json.dumps({"passed": True}), encoding="utf-8"
    )

    assert load_reports(tmp_path) == Reports({"k": 5}, {"passed": True})


def test_load_reports_tolerates_missing_and_broken_files(tmp_path: Path) -> None:
    (tmp_path / "generation_report.json").write_text("{not json", encoding="utf-8")

    assert load_reports(tmp_path) == Reports(None, None)
```

(`pytest` and `Path` are already imported at the top of this test file; add them if not.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_playground_core.py -v`
Expected: FAIL at import (`cannot import name 'IndexSnapshot'`).

- [ ] **Step 3: Implement**

In `core.py`:

1. Imports — extend the header:

```python
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Protocol

from rag_eval_platform.config.settings import ChunkStrategy
from rag_eval_platform.evaluation.citation_validity import CitationValidity
from rag_eval_platform.evaluation.judge import Judge, JudgeError, JudgeSample, JudgeScores
from rag_eval_platform.generation.generator import Answer, Generator
```

and `REPORTS_DIR = Path("reports")` next to `UPLOAD_DIR`.

2. `QueryTrace` — add the field last with a default:

```python
    retrieval_ms: float
    query_embedding: tuple[float, ...] = ()  # the question's vector, for the meaning map
```

3. `RetrievalStep` — add `query_embedding: tuple[float, ...]` as the last field, and in `retrieve_step` compute it once:

```python
    started = time.perf_counter()
    query_embedding = tuple(embedder.embed_query(question))
    candidates = top_k if reranker is None else max(rerank_candidates, top_k)
    vector_results = tuple(store.search(list(query_embedding), k=candidates))
    ...
    return RetrievalStep(
        vector_results, final_results, (time.perf_counter() - started) * 1000, query_embedding
    )
```

4. `run_query` — pass it on:

```python
    return QueryTrace(
        step.vector_results, step.final_results, answer, step.retrieval_ms, step.query_embedding
    )
```

5. New code at the end of the file:

```python
class _Exportable(Protocol):
    def export(self) -> tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]: ...


class _CitationChecker(Protocol):
    def check(self, answer: Answer) -> CitationValidity: ...


@dataclass(frozen=True)
class IndexSnapshot:
    """Everything stored in the playground collection: chunks and their vectors."""

    chunks: tuple[Chunk, ...]
    vectors: tuple[tuple[float, ...], ...]

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(c.doc_id for c in self.chunks))


def index_snapshot(store: _Exportable) -> IndexSnapshot:
    chunks, vectors = store.export()
    return IndexSnapshot(chunks, vectors)


def trace_key(trace: QueryTrace) -> str:
    """Identifies one answered question, so results tied to it (the judge) can be matched."""
    parts = (trace.answer.question, trace.answer.text, *(r.chunk.id for r in trace.final_results))
    return hashlib.sha1(repr(parts).encode(), usedforsecurity=False).hexdigest()[:16]


@dataclass(frozen=True)
class JudgeResult:
    scores: JudgeScores
    citations: CitationValidity
    judge_ms: float
    note: str = ""  # why citations were not scored, if they were not


def judge_answer(trace: QueryTrace, judge: Judge, checker: _CitationChecker) -> JudgeResult:
    """Grade one playground answer: faithfulness, relevance and citation validity."""
    answer = trace.answer
    if answer.is_refusal:
        raise ValueError("a refusal makes no claims, so there is nothing to judge")
    started = time.perf_counter()
    sample = JudgeSample(
        question=answer.question,
        answer=answer.text,
        contexts=tuple(r.chunk.text for r in trace.final_results),
        reference="",  # your own documents have no golden answer; not needed by these metrics
    )
    scores = judge.score(sample)
    note = ""
    try:
        citations = checker.check(answer)
    except JudgeError as exc:
        citations, note = CitationValidity(()), f"Citation verdicts unreadable: {exc}"
    return JudgeResult(scores, citations, (time.perf_counter() - started) * 1000, note)


@dataclass(frozen=True)
class Reports:
    retrieval: dict[str, Any] | None
    generation: dict[str, Any] | None


def load_reports(directory: Path) -> Reports:
    """The saved golden-set reports, or None for each one that is missing or unreadable."""
    return Reports(
        _read_json(directory / "retrieval_report.json"),
        _read_json(directory / "generation_report.json"),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_playground_core.py tests/integration/test_playground.py -v`
Expected: unit tests PASS; the integration test still passes or skips.

- [ ] **Step 5: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit -q`
Expected: clean, all pass.

---

### Task 3: Animation helpers and the pipeline diagram

**Files:**
- Create: `src/rag_eval_platform/playground/animation.py`, `src/rag_eval_platform/playground/visuals.py`
- Test: `tests/unit/test_animation.py`, `tests/unit/test_visuals.py`

**Interfaces:**
- Produces:
  - `ease_steps(steps: int) -> tuple[float, ...]` — ease-out progress in (0, 1], last exactly 1.0
  - `fingerprint(*parts: object) -> str`
  - `should_animate(state: MutableMapping[str, object], key: str, data_fingerprint: str) -> bool` — True once per new fingerprint
  - `PHASES = ("Ingest", "Embed", "Retrieve", "Generate", "Evaluate")`
  - `PipelineStats(documents=0, chunks=0, candidates=0, top_k=0, cited=0, faithfulness=None)`
  - `phases_with_data(stats: PipelineStats) -> frozenset[str]`
  - `pipeline_dot(active: str | None, stats: PipelineStats) -> str`
  - `dot_text(text: str, limit: int = 60) -> str`

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_animation.py`:

```python
"""Tests for playground.animation."""

import pytest

from rag_eval_platform.playground.animation import ease_steps, fingerprint, should_animate


def test_ease_steps_rise_to_exactly_one() -> None:
    steps = ease_steps(8)

    assert len(steps) == 8
    assert steps[-1] == 1.0
    assert all(0 < a < b for a, b in zip(steps, steps[1:], strict=False))


def test_ease_steps_are_front_loaded() -> None:
    first, *_ = ease_steps(4)

    assert first > 0.25  # ease-out: fast start, gentle finish


def test_ease_steps_needs_at_least_one_step() -> None:
    with pytest.raises(ValueError, match="steps"):
        ease_steps(0)


def test_fingerprint_is_stable_and_sensitive() -> None:
    assert fingerprint("a", 1, (0.5,)) == fingerprint("a", 1, (0.5,))
    assert fingerprint("a", 1) != fingerprint("a", 2)


def test_should_animate_once_per_new_data() -> None:
    state: dict[str, object] = {}

    assert should_animate(state, "chart", "v1") is True
    assert should_animate(state, "chart", "v1") is False  # same data: no replay
    assert should_animate(state, "chart", "v2") is True  # new data: animate again
    assert should_animate(state, "other", "v2") is True  # each chart tracks its own data
```

`tests/unit/test_visuals.py` (pipeline part):

```python
"""Tests for playground.visuals (pure data behind the index-level charts)."""

import pytest

from rag_eval_platform.playground.visuals import (
    PipelineStats,
    dot_text,
    phases_with_data,
    pipeline_dot,
)

FULL = PipelineStats(documents=2, chunks=40, candidates=20, top_k=5, cited=2, faithfulness=0.93)


def _node_line(dot: str, phase: str) -> str:
    return next(line for line in dot.splitlines() if line.strip().startswith(f"{phase} ["))


def test_phases_with_data_follow_the_pipeline() -> None:
    assert phases_with_data(PipelineStats()) == frozenset()
    assert phases_with_data(PipelineStats(documents=1, chunks=3)) == {"Ingest", "Embed"}
    assert phases_with_data(FULL) == {"Ingest", "Embed", "Retrieve", "Generate", "Evaluate"}


def test_pipeline_dot_highlights_the_active_phase_with_live_numbers() -> None:
    dot = pipeline_dot("Retrieve", FULL)

    assert dot.startswith("digraph pipeline {")
    assert "#d97706" in _node_line(dot, "Retrieve")  # active colour
    assert "top-5 of 20" in _node_line(dot, "Retrieve")
    assert "cited 2/5" in _node_line(dot, "Generate")
    assert "faithfulness 0.93" in _node_line(dot, "Evaluate")
    assert "#16a34a" in _node_line(dot, "Ingest")  # has data, not active
    assert 'label="40 chunks"' in dot  # Ingest -> Embed edge


def test_pipeline_dot_greys_out_phases_without_data() -> None:
    dot = pipeline_dot(None, PipelineStats(documents=1, chunks=3))

    assert "#9ca3af" in _node_line(dot, "Retrieve")
    assert "not judged" in _node_line(dot, "Evaluate")


def test_pipeline_dot_rejects_unknown_phases() -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        pipeline_dot("Deploy", FULL)


def test_pipeline_dot_escapes_quotes() -> None:
    assert dot_text('say "hi" \\ bye') == 'say \\"hi\\" \\\\ bye'
    assert dot_text("x" * 100, limit=10) == "xxxxxxxxx…"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_animation.py tests/unit/test_visuals.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement**

`src/rag_eval_platform/playground/animation.py`:

```python
"""Play-once animation helpers for the playground (pure, no Streamlit).

A chart animates when it receives data it has not shown before, and draws its final frame
directly on every later rerun (Streamlit reruns the page on every click).
"""

import hashlib
from collections.abc import MutableMapping

_SLOT = "_animated:"


def ease_steps(steps: int) -> tuple[float, ...]:
    """Progress values in (0, 1] with a cubic ease-out, ending exactly at 1.0."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    return tuple(1 - (1 - i / steps) ** 3 for i in range(1, steps + 1))


def fingerprint(*parts: object) -> str:
    """A short stable id for a chart's data."""
    return hashlib.sha1(repr(parts).encode(), usedforsecurity=False).hexdigest()[:16]


def should_animate(state: MutableMapping[str, object], key: str, data_fingerprint: str) -> bool:
    """True the first time ``key`` sees ``data_fingerprint``; records it in ``state``."""
    slot = _SLOT + key
    if state.get(slot) == data_fingerprint:
        return False
    state[slot] = data_fingerprint
    return True
```

`src/rag_eval_platform/playground/visuals.py` (first part; Task 4 appends more):

```python
"""Pure data behind the playground's index-level visuals: the pipeline diagram, chunks and
their overlaps, and the meaning map. No Streamlit or Plotly here, so it is unit-tested."""

from dataclasses import dataclass

PHASES = ("Ingest", "Embed", "Retrieve", "Generate", "Evaluate")

_ACTIVE = 'fillcolor="#fef3c7" color="#d97706" penwidth=2 fontcolor="#78350f"'
_DONE = 'fillcolor="#dcfce7" color="#16a34a" fontcolor="#14532d"'
_EMPTY = 'fillcolor="#f3f4f6" color="#d1d5db" fontcolor="#9ca3af"'


@dataclass(frozen=True)
class PipelineStats:
    documents: int = 0
    chunks: int = 0
    candidates: int = 0
    top_k: int = 0
    cited: int = 0
    faithfulness: float | None = None


def phases_with_data(stats: PipelineStats) -> frozenset[str]:
    ready: set[str] = set()
    if stats.chunks:
        ready |= {"Ingest", "Embed"}
    if stats.top_k:
        ready |= {"Retrieve", "Generate"}
    if stats.faithfulness is not None:
        ready.add("Evaluate")
    return frozenset(ready)


def dot_text(text: str, limit: int = 60) -> str:
    """Text safe inside a double-quoted DOT label, cut to ``limit`` characters."""
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("\\", "\\\\").replace('"', '\\"')


def pipeline_dot(active: str | None, stats: PipelineStats) -> str:
    """Graphviz source: the five phases, ``active`` highlighted, live numbers as captions."""
    if active is not None and active not in PHASES:
        raise ValueError(f"unknown phase: {active}")
    ready = phases_with_data(stats)
    captions = {
        "Ingest": f"{stats.documents} docs" if stats.chunks else "no documents",
        "Embed": f"{stats.chunks} vectors" if stats.chunks else "",
        "Retrieve": f"top-{stats.top_k} of {stats.candidates}" if stats.top_k else "",
        "Generate": f"cited {stats.cited}/{stats.top_k}" if stats.top_k else "",
        "Evaluate": (
            f"faithfulness {stats.faithfulness:.2f}"
            if stats.faithfulness is not None
            else "not judged"
        ),
    }
    lines = [
        "digraph pipeline {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.3;',
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11];',
        '  edge [color="#9ca3af" fontname="Helvetica" fontsize=9 fontcolor="#6b7280"];',
    ]
    for number, phase in enumerate(PHASES, start=1):
        style = _ACTIVE if phase == active else _DONE if phase in ready else _EMPTY
        caption = captions[phase]
        label = f"{number} · {phase}\\n{dot_text(caption)}" if caption else f"{number} · {phase}"
        lines.append(f'  {phase} [label="{label}" {style}];')
    edge_labels = [
        f"{stats.chunks} chunks" if stats.chunks else "",
        "cosine search" if stats.chunks else "",
        f"{stats.top_k} chunks" if stats.top_k else "",
        "answer" if stats.top_k else "",
    ]
    for (source, target), label in zip(zip(PHASES, PHASES[1:]), edge_labels, strict=True):
        lines.append(f'  {source} -> {target} [label="{label}"];')
    lines.append("}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_animation.py tests/unit/test_visuals.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`

---

### Task 4: Index-level visuals — counts, ingest flow, chunk spans, size bins, meaning map

**Files:**
- Modify: `src/rag_eval_platform/playground/visuals.py`, `pyproject.toml` (explicit `numpy`)
- Test: `tests/unit/test_visuals.py`

**Interfaces:**
- Consumes: `IndexSnapshot` (Task 2), `Chunk`.
- Produces:
  - `IndexCounts(documents, pages, characters, chunks, mean_chars)`; `index_counts(chunks: Sequence[Chunk]) -> IndexCounts`
  - `ingest_flow_dot(counts: IndexCounts) -> str`
  - `ChunkSpan(doc_id, chunk_id, index, start, end, overlap)`; `chunk_spans(chunks) -> tuple[ChunkSpan, ...]`
  - `SizeBin(start, end, count)`; `size_bins(sizes: Sequence[int], bins: int = 12) -> tuple[SizeBin, ...]`
  - `MapPoint(chunk_id, kind: Literal["chunk","retrieved","question"], x, y, z, text)`; `meaning_map(snapshot, query, retrieved_ids, *, dims=2, max_points=3000, seed=0) -> tuple[MapPoint, ...]`
  - `vector_for(snapshot, chunk_id) -> tuple[float, ...] | None`; `angle_degrees(a, b) -> float`

- [ ] **Step 1: Add numpy as an explicit dependency**

Run: `uv add "numpy>=2.0"` (it is already installed through `chromadb-client`; the tooling table lists NumPy as chosen for vector math). Expected: `pyproject.toml` gains `"numpy>=2.0"` in `dependencies`, lockfile updates.

- [ ] **Step 2: Write the failing tests**

Append to `tests/unit/test_visuals.py`:

```python
import math

from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import IndexSnapshot
from rag_eval_platform.playground.visuals import (
    ChunkSpan,
    IndexCounts,
    angle_degrees,
    chunk_spans,
    index_counts,
    ingest_flow_dot,
    meaning_map,
    size_bins,
    vector_for,
)


def _chunk(doc: str, index: int, start: int, text: str, page: int | None = None) -> Chunk:
    return Chunk(id=f"{doc}#{index}", doc_id=doc, index=index, text=text, start_index=start,
                 page=page)  # fmt: skip


CHUNKS = (
    _chunk("a.pdf", 0, 0, "abcdef", page=1),
    _chunk("a.pdf", 1, 4, "efghij", page=1),
    _chunk("a.pdf", 2, 10, "klmn", page=2),
    _chunk("b.md", 0, 0, "xyz"),
)


def test_index_counts() -> None:
    assert index_counts(CHUNKS) == IndexCounts(
        documents=2, pages=2, characters=17, chunks=4, mean_chars=19 / 4
    )
    assert index_counts(()) == IndexCounts(0, 0, 0, 0, 0.0)


def test_ingest_flow_dot_shows_each_stage_count() -> None:
    dot = ingest_flow_dot(index_counts(CHUNKS))

    for text in ("2 files", "2 PDF pages", "17 characters", "4 chunks"):
        assert text in dot


def test_chunk_spans_measure_overlap_with_the_previous_chunk() -> None:
    spans = chunk_spans(tuple(reversed(CHUNKS)))  # input order does not matter

    assert spans == (
        ChunkSpan("a.pdf", "a.pdf#0", 0, 0, 6, 0),
        ChunkSpan("a.pdf", "a.pdf#1", 1, 4, 10, 2),  # shares chars 4-5 with #0
        ChunkSpan("a.pdf", "a.pdf#2", 2, 10, 14, 0),
        ChunkSpan("b.md", "b.md#0", 0, 0, 3, 0),  # a new document starts fresh
    )


def test_size_bins_cover_every_size() -> None:
    bins = size_bins([100, 120, 480, 500], bins=2)

    assert [(b.start, b.end, b.count) for b in bins] == [(100, 300, 2), (300, 500, 2)]
    assert size_bins([]) == ()


def _snapshot(vectors: list[tuple[float, ...]]) -> IndexSnapshot:
    chunks = tuple(_chunk("d.md", i, i, f"text {i}") for i in range(len(vectors)))
    return IndexSnapshot(chunks, tuple(vectors))


def test_meaning_map_places_similar_chunks_close_and_projects_the_question() -> None:
    snapshot = _snapshot([(1.0, 0.0, 0.0), (0.9, 0.1, 0.0), (0.0, 0.0, 1.0)])

    points = meaning_map(snapshot, (1.0, 0.05, 0.0), {"d.md#1"})

    by_id = {p.chunk_id: p for p in points}
    assert [p.kind for p in points] == ["chunk", "retrieved", "chunk", "question"]
    near = math.dist(
        (by_id["question"].x, by_id["question"].y), (by_id["d.md#0"].x, by_id["d.md#0"].y)
    )
    far = math.dist(
        (by_id["question"].x, by_id["question"].y), (by_id["d.md#2"].x, by_id["d.md#2"].y)
    )
    assert near < far
    assert all(p.z == 0.0 for p in points)  # 2-D map


def test_meaning_map_in_3d_has_depth() -> None:
    snapshot = _snapshot([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)])

    points = meaning_map(snapshot, None, set(), dims=3)

    assert len(points) == 4
    assert any(p.z != 0.0 for p in points)


def test_meaning_map_samples_but_keeps_retrieved() -> None:
    snapshot = _snapshot([(float(i), float(i % 7), 1.0) for i in range(50)])

    points = meaning_map(snapshot, None, {"d.md#49", "d.md#3"}, max_points=10)

    ids = {p.chunk_id for p in points}
    assert len(points) == 10
    assert {"d.md#49", "d.md#3"} <= ids


def test_meaning_map_edge_cases() -> None:
    assert meaning_map(IndexSnapshot((), ()), (1.0,), set()) == ()
    single = meaning_map(_snapshot([(1.0, 2.0)]), None, set())
    assert len(single) == 1
    with pytest.raises(ValueError, match="dims"):
        meaning_map(_snapshot([(1.0, 2.0)]), None, set(), dims=4)


def test_vector_for_and_angle() -> None:
    snapshot = _snapshot([(1.0, 0.0), (0.0, 1.0)])

    assert vector_for(snapshot, "d.md#1") == (0.0, 1.0)
    assert vector_for(snapshot, "missing") is None
    assert angle_degrees((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)
    assert angle_degrees((1.0, 1.0), (2.0, 2.0)) == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="zero"):
        angle_degrees((0.0, 0.0), (1.0, 0.0))
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_visuals.py -v`
Expected: FAIL (`cannot import name 'ChunkSpan'`).

- [ ] **Step 4: Implement**

Append to `visuals.py` (and extend its imports to the block below):

```python
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

import numpy as np
import numpy.typing as npt

from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import IndexSnapshot


@dataclass(frozen=True)
class IndexCounts:
    documents: int
    pages: int
    characters: int
    chunks: int
    mean_chars: float


def index_counts(chunks: Sequence[Chunk]) -> IndexCounts:
    """Counts for the ingest flow: files, PDF pages, text length and chunks."""
    pages: dict[str, set[int]] = {}
    ends: dict[str, int] = {}
    for c in chunks:
        if c.page is not None:
            pages.setdefault(c.doc_id, set()).add(c.page)
        ends[c.doc_id] = max(ends.get(c.doc_id, 0), c.start_index + len(c.text))
    return IndexCounts(
        documents=len(ends),
        pages=sum(len(p) for p in pages.values()),
        characters=sum(ends.values()),
        chunks=len(chunks),
        mean_chars=fmean(len(c.text) for c in chunks) if chunks else 0.0,
    )


def ingest_flow_dot(counts: IndexCounts) -> str:
    """Graphviz source: files -> pages -> text -> chunks, with counts."""
    stages = [
        ("files", f"📄 {counts.documents} files"),
        ("pages", f"{counts.pages} PDF pages" if counts.pages else "no PDF pages"),
        ("text", f"{counts.characters:,} characters"),
        ("chunks", f"✂️ {counts.chunks} chunks\\n~{counts.mean_chars:.0f} chars each"),
    ]
    lines = [
        "digraph ingest {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box style="rounded,filled" fillcolor="#eef2ff" color="#6366f1" '
        'fontname="Helvetica" fontsize=11];',
        '  edge [color="#6366f1"];',
    ]
    lines += [f'  {name} [label="{label}"];' for name, label in stages]
    lines.append("  files -> pages -> text -> chunks;")
    lines.append("}")
    return "\n".join(lines)


@dataclass(frozen=True)
class ChunkSpan:
    doc_id: str
    chunk_id: str
    index: int
    start: int
    end: int
    overlap: int  # characters shared with the previous chunk of the same document


def chunk_spans(chunks: Sequence[Chunk]) -> tuple[ChunkSpan, ...]:
    """Where each chunk sits in its document, and how much it overlaps the previous one."""
    spans: list[ChunkSpan] = []
    previous: Chunk | None = None
    for c in sorted(chunks, key=lambda c: (c.doc_id, c.index)):
        end = c.start_index + len(c.text)
        overlap = 0
        if previous is not None and previous.doc_id == c.doc_id:
            previous_end = previous.start_index + len(previous.text)
            overlap = max(0, previous_end - c.start_index)
        spans.append(ChunkSpan(c.doc_id, c.id, c.index, c.start_index, end, overlap))
        previous = c
    return tuple(spans)


@dataclass(frozen=True)
class SizeBin:
    start: int
    end: int
    count: int


def size_bins(sizes: Sequence[int], bins: int = 12) -> tuple[SizeBin, ...]:
    """A histogram of chunk sizes (characters)."""
    if not sizes:
        return ()
    counts, edges = np.histogram(np.asarray(sizes), bins=bins)
    return tuple(
        SizeBin(round(float(edges[i])), round(float(edges[i + 1])), int(count))
        for i, count in enumerate(counts)
    )


@dataclass(frozen=True)
class MapPoint:
    chunk_id: str  # "question" for the question itself
    kind: Literal["chunk", "retrieved", "question"]
    x: float
    y: float
    z: float
    text: str


def meaning_map(
    snapshot: IndexSnapshot,
    query: Sequence[float] | None,
    retrieved_ids: Collection[str],
    *,
    dims: int = 2,
    max_points: int = 3000,
    seed: int = 0,
) -> tuple[MapPoint, ...]:
    """Chunks (and the question) squeezed from embedding space to 2 or 3 dimensions with PCA.

    Big indexes are sampled down to ``max_points``, always keeping the retrieved chunks.
    """
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}")
    if not snapshot.chunks:
        return ()
    keep = _sample(snapshot, retrieved_ids, max_points, seed)
    matrix = np.asarray([snapshot.vectors[i] for i in keep], dtype=np.float64)
    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    components = vt[:dims]

    def coords(vector: Sequence[float]) -> tuple[float, float, float]:
        values: npt.NDArray[np.float64] = (
            np.asarray(vector, dtype=np.float64) - mean
        ) @ components.T
        padded = [float(v) for v in values] + [0.0, 0.0, 0.0]
        return padded[0], padded[1], padded[2]

    points = []
    for i in keep:
        chunk = snapshot.chunks[i]
        kind: Literal["chunk", "retrieved"] = "retrieved" if chunk.id in retrieved_ids else "chunk"
        points.append(MapPoint(chunk.id, kind, *coords(snapshot.vectors[i]), text=chunk.text[:160]))
    if query is not None:
        points.append(MapPoint("question", "question", *coords(query), text="your question"))
    return tuple(points)


def _sample(
    snapshot: IndexSnapshot, retrieved_ids: Collection[str], max_points: int, seed: int
) -> list[int]:
    total = len(snapshot.chunks)
    if total <= max_points:
        return list(range(total))
    must = [i for i, c in enumerate(snapshot.chunks) if c.id in retrieved_ids]
    others = [i for i in range(total) if i not in set(must)]
    rng = np.random.default_rng(seed)
    picked = rng.choice(others, size=max(0, max_points - len(must)), replace=False)
    return sorted([*must, *(int(i) for i in picked)])


def vector_for(snapshot: IndexSnapshot, chunk_id: str) -> tuple[float, ...] | None:
    for chunk, vector in zip(snapshot.chunks, snapshot.vectors, strict=True):
        if chunk.id == chunk_id:
            return vector
    return None


def angle_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    """The angle between two vectors: 0° = same direction, 90° = unrelated."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norms = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norms == 0.0:
        raise ValueError("cannot measure the angle of a zero vector")
    cosine = float(np.clip(va @ vb / norms, -1.0, 1.0))
    return math.degrees(math.acos(cosine))
```

Move the first block's `from dataclasses import dataclass` into this merged import block so the module has one import section.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_visuals.py -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint (no commit)**

Run: `uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`

---

### Task 5: Query-level visuals — similarity, re-rank, funnel, prompt, answer spans, timing, verdicts, reports

**Files:**
- Modify: `src/rag_eval_platform/generation/generator.py` (expose `CITATION_PATTERN`)
- Create: `src/rag_eval_platform/playground/query_visuals.py`
- Test: `tests/unit/test_query_visuals.py`

**Interfaces:**
- Consumes: `QueryTrace`, `JudgeResult` (Task 2), `dot_text` (Task 3), `SYSTEM_PROMPT`, `Answer`.
- Produces:
  - `SOURCE_COLORS`, `INVALID_COLOR`, `source_color(number: int) -> str`, `origin(result: SearchResult) -> str`
  - `SimilarityRow(rank, chunk_id, score, kept, cited)`; `similarity_rows(trace) -> tuple[SimilarityRow, ...]`; `was_reranked(trace) -> bool`
  - `RerankMove(chunk_id, before: int, after: int | None)`; `rerank_moves(trace) -> tuple[RerankMove, ...]`
  - `funnel_stages(total_chunks: int, trace) -> tuple[tuple[str, int], ...]`
  - `PromptBlock(kind: Literal["rules","source","question"], title, text)`; `prompt_blocks(trace) -> tuple[PromptBlock, ...]`
  - `AnswerSpan(text, number: int | None, valid: bool)`; `answer_spans(answer) -> tuple[AnswerSpan, ...]`; `spans_html(spans) -> str`
  - `timing_rows(trace, judged: JudgeResult | None) -> tuple[tuple[str, float], ...]` (milliseconds)
  - `verdict_dot(result: JudgeResult) -> str`
  - `hit_vs_faithfulness(retrieval: dict, generation: dict) -> tuple[tuple[str, float, int], ...]`

- [ ] **Step 1: Expose the citation pattern**

In `generator.py`, rename `_CITATION_PATTERN` to `CITATION_PATTERN` (definition and its one use in `parse_citation_numbers`). Run `uv run pytest tests/unit/test_generator.py -q` — expected PASS.

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_query_visuals.py`:

```python
"""Tests for playground.query_visuals (pure data behind the per-question charts)."""

import pytest

from rag_eval_platform.evaluation.citation_validity import CitationValidity, CitationVerdict
from rag_eval_platform.evaluation.judge import JudgeScores
from rag_eval_platform.generation.generator import Answer, Citation
from rag_eval_platform.generation.prompt_templates import SYSTEM_PROMPT
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import JudgeResult, QueryTrace
from rag_eval_platform.playground.query_visuals import (
    INVALID_COLOR,
    AnswerSpan,
    RerankMove,
    SimilarityRow,
    answer_spans,
    funnel_stages,
    hit_vs_faithfulness,
    origin,
    prompt_blocks,
    rerank_moves,
    similarity_rows,
    source_color,
    spans_html,
    timing_rows,
    verdict_dot,
    was_reranked,
)
from rag_eval_platform.retrieval.vector_store import SearchResult


def hit(chunk_id: str, score: float, page: int | None = None) -> SearchResult:
    doc = chunk_id.split("#")[0]
    return SearchResult(Chunk(id=chunk_id, doc_id=doc, index=0, text=f"text of {chunk_id}",
                              start_index=0, page=page), score)  # fmt: skip


def answer(text: str, sources: tuple[SearchResult, ...], cited: tuple[int, ...] = ()) -> Answer:
    return Answer(
        question="What is MRR?", text=text,
        citations=tuple(Citation(n, sources[n - 1]) for n in cited),
        invalid_citations=tuple(n for n in (9,) if f"[{n}]" in text), sources=sources,
        model="qwen3:8b", prompt_version="v1", input_tokens=10, output_tokens=5, latency_ms=4000.0,
    )  # fmt: skip


VECTOR = (hit("a#0", 0.9), hit("b#0", 0.7), hit("c#0", 0.5))


def test_similarity_rows_without_reranking() -> None:
    final = VECTOR[:2]
    trace = QueryTrace(VECTOR, final, answer("A [1].", final, cited=(1,)), 20.0)

    assert similarity_rows(trace) == (
        SimilarityRow(1, "a#0", 0.9, kept=True, cited=True),
        SimilarityRow(2, "b#0", 0.7, kept=True, cited=False),
        SimilarityRow(3, "c#0", 0.5, kept=False, cited=False),
    )
    assert was_reranked(trace) is False


def test_similarity_rows_and_moves_with_reranking() -> None:
    final = (hit("c#0", 5.1), hit("a#0", 2.0))  # cross-encoder promoted c
    trace = QueryTrace(VECTOR, final, answer("C [1].", final, cited=(1,)), 20.0)

    rows = similarity_rows(trace)
    assert [(r.chunk_id, r.kept, r.cited) for r in rows] == [
        ("a#0", True, False),
        ("b#0", False, False),
        ("c#0", True, True),
    ]
    assert was_reranked(trace) is True
    assert rerank_moves(trace) == (
        RerankMove("a#0", 1, 2),
        RerankMove("b#0", 2, None),
        RerankMove("c#0", 3, 1),
    )


def test_funnel_stages() -> None:
    final = VECTOR[:2]
    trace = QueryTrace(VECTOR, final, answer("A [1][2].", final, cited=(1, 2)), 20.0)

    assert funnel_stages(40, trace) == (
        ("all chunks", 40),
        ("candidates", 3),
        ("sent to the model", 2),
        ("cited", 2),
    )


def test_prompt_blocks_follow_the_prompt_order() -> None:
    final = (hit("a#0", 0.9, page=3), hit("b#0", 0.7))
    trace = QueryTrace(final, final, answer("A [1].", final), 1.0)

    blocks = prompt_blocks(trace)

    assert [b.kind for b in blocks] == ["rules", "source", "source", "question"]
    assert blocks[0].text == SYSTEM_PROMPT
    assert blocks[1].title == "[1] a, page 3"
    assert blocks[-1].text == "What is MRR?"
    assert origin(final[1]) == "b"


def test_answer_spans_split_text_and_citations() -> None:
    sources = VECTOR[:2]

    spans = answer_spans(answer("MRR ranks [1, 2]. Also [9].", sources))

    assert spans == (
        AnswerSpan("MRR ranks ", None, True),
        AnswerSpan("[1]", 1, True),
        AnswerSpan("[2]", 2, True),
        AnswerSpan(". Also ", None, True),
        AnswerSpan("[9]", 9, False),
        AnswerSpan(".", None, True),
    )


def test_answer_spans_without_citations() -> None:
    assert answer_spans(answer("No citations.", VECTOR)) == (
        AnswerSpan("No citations.", None, True),
    )


def test_spans_html_colours_citations_and_escapes_text() -> None:
    html = spans_html((
        AnswerSpan("a < b & c\n", None, True),
        AnswerSpan("[1]", 1, True),
        AnswerSpan("[9]", 9, False),
    ))  # fmt: skip

    assert "a &lt; b &amp; c<br>" in html
    assert source_color(1) in html
    assert INVALID_COLOR in html
    assert "<script" not in spans_html((AnswerSpan("<script>x</script>", None, True),))


def test_spans_html_escapes_text() -> None:
    assert "&lt;b&gt;" in spans_html((AnswerSpan("<b>", None, True),))


def test_source_colors_cycle() -> None:
    assert source_color(1) != source_color(2)
    assert source_color(11) == source_color(1)


def _judged(verdicts: tuple[CitationVerdict, ...]) -> JudgeResult:
    return JudgeResult(JudgeScores(faithfulness=1.0, answer_relevance=0.9),
                       CitationValidity(verdicts), 60000.0)  # fmt: skip


def test_timing_rows() -> None:
    trace = QueryTrace(VECTOR, VECTOR, answer("A.", VECTOR), 250.0)

    assert timing_rows(trace, None) == (("retrieve", 250.0), ("generate", 4000.0))
    assert timing_rows(trace, _judged(()))[-1] == ("judge", 60000.0)


def test_verdict_dot_links_sentences_to_sources() -> None:
    dot = verdict_dot(_judged((
        CitationVerdict('He said "hi".', 1, "a.md", True),
        CitationVerdict("Other claim.", 2, "b.md", False),
    )))  # fmt: skip

    assert 's0 -> src1 [color="#16a34a"' in dot
    assert 's1 -> src2 [color="#dc2626"' in dot
    assert 'He said \\"hi\\".' in dot


def test_verdict_dot_escapes_quotes() -> None:
    dot = verdict_dot(_judged((CitationVerdict('"q"', 1, 'doc"x', True),)))

    assert '\\"q\\"' in dot
    assert 'doc\\"x' in dot


def test_verdict_dot_without_citations() -> None:
    assert "No citations to check" in verdict_dot(_judged(()))


def test_hit_vs_faithfulness_joins_the_two_reports() -> None:
    retrieval = {"examples": [
        {"example_id": "q1", "scores": {"recall": 1.0}},
        {"example_id": "q2", "scores": {"recall": 0.5}},
        {"example_id": "q3", "scores": {"recall": 1.0}},
    ]}  # fmt: skip
    generation = {"examples": [
        {"example_id": "q1", "scores": {"faithfulness": 1.0}},
        {"example_id": "q2", "scores": {"faithfulness": 0.5}},
        {"example_id": "q3", "scores": {"faithfulness": 0.8}},
    ]}  # fmt: skip

    assert hit_vs_faithfulness(retrieval, generation) == (
        ("found every relevant doc", pytest.approx(0.9), 2),
        ("missed a relevant doc", pytest.approx(0.5), 1),
    )
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_query_visuals.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 4: Implement**

`src/rag_eval_platform/playground/query_visuals.py`:

```python
"""Pure data behind the playground's per-question visuals: similarity and re-ranking,
the prompt, the answer's citations, timings and the judge's verdicts."""

import html
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal

from rag_eval_platform.generation.generator import CITATION_PATTERN, Answer
from rag_eval_platform.generation.prompt_templates import SYSTEM_PROMPT
from rag_eval_platform.playground.core import JudgeResult, QueryTrace
from rag_eval_platform.playground.visuals import dot_text
from rag_eval_platform.retrieval.vector_store import SearchResult

SOURCE_COLORS = (
    "#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2",
    "#db2777", "#65a30d", "#ea580c", "#4f46e5", "#0d9488",
)  # fmt: skip
INVALID_COLOR = "#dc2626"
YES_COLOR = "#16a34a"


def source_color(number: int) -> str:
    """The colour of source ``[number]``, shared by the answer and the source list."""
    return SOURCE_COLORS[(number - 1) % len(SOURCE_COLORS)]


def origin(result: SearchResult) -> str:
    chunk = result.chunk
    return chunk.doc_id if chunk.page is None else f"{chunk.doc_id}, page {chunk.page}"


@dataclass(frozen=True)
class SimilarityRow:
    rank: int
    chunk_id: str
    score: float  # cosine similarity from vector search
    kept: bool  # sent to the model
    cited: bool  # cited in the answer


def similarity_rows(trace: QueryTrace) -> tuple[SimilarityRow, ...]:
    kept = {r.chunk.id: n for n, r in enumerate(trace.final_results, start=1)}
    cited = {c.number for c in trace.answer.citations}
    return tuple(
        SimilarityRow(rank, r.chunk.id, r.score, r.chunk.id in kept, kept.get(r.chunk.id) in cited)
        for rank, r in enumerate(trace.vector_results, start=1)
    )


def was_reranked(trace: QueryTrace) -> bool:
    return [r.chunk.id for r in trace.vector_results] != [r.chunk.id for r in trace.final_results]


@dataclass(frozen=True)
class RerankMove:
    chunk_id: str
    before: int  # rank after vector search
    after: int | None  # rank after the cross-encoder; None = dropped


def rerank_moves(trace: QueryTrace) -> tuple[RerankMove, ...]:
    after = {r.chunk.id: n for n, r in enumerate(trace.final_results, start=1)}
    return tuple(
        RerankMove(r.chunk.id, n, after.get(r.chunk.id))
        for n, r in enumerate(trace.vector_results, start=1)
    )


def funnel_stages(total_chunks: int, trace: QueryTrace) -> tuple[tuple[str, int], ...]:
    cited = len({c.number for c in trace.answer.citations})
    return (
        ("all chunks", total_chunks),
        ("candidates", len(trace.vector_results)),
        ("sent to the model", len(trace.final_results)),
        ("cited", cited),
    )


@dataclass(frozen=True)
class PromptBlock:
    kind: Literal["rules", "source", "question"]
    title: str
    text: str


def prompt_blocks(trace: QueryTrace) -> tuple[PromptBlock, ...]:
    """The prompt in the order the model reads it: rules, numbered sources, question."""
    sources = [
        PromptBlock("source", f"[{n}] {origin(r)}", r.chunk.text)
        for n, r in enumerate(trace.final_results, start=1)
    ]
    return (
        PromptBlock("rules", "System rules", SYSTEM_PROMPT),
        *sources,
        PromptBlock("question", "Question", trace.answer.question),
    )


@dataclass(frozen=True)
class AnswerSpan:
    text: str
    number: int | None  # the cited source, None for plain text
    valid: bool  # False for a citation to a source that was never given


def answer_spans(answer: Answer) -> tuple[AnswerSpan, ...]:
    spans: list[AnswerSpan] = []
    position = 0
    for match in CITATION_PATTERN.finditer(answer.text):
        if match.start() > position:
            spans.append(AnswerSpan(answer.text[position : match.start()], None, True))
        for part in match.group(1).split(","):
            number = int(part)
            spans.append(AnswerSpan(f"[{number}]", number, 1 <= number <= len(answer.sources)))
        position = match.end()
    if position < len(answer.text):
        spans.append(AnswerSpan(answer.text[position:], None, True))
    return tuple(spans)


def spans_html(spans: tuple[AnswerSpan, ...]) -> str:
    """The answer as HTML: text escaped, each citation a chip in its source's colour."""
    parts = []
    for span in spans:
        text = html.escape(span.text).replace("\n", "<br>")
        if span.number is None:
            parts.append(text)
        elif span.valid:
            parts.append(
                f'<span style="background:{source_color(span.number)};color:#fff;'
                f'border-radius:4px;padding:0 4px;font-weight:600">{text}</span>'
            )
        else:
            parts.append(
                f'<span title="no such source" style="background:{INVALID_COLOR};color:#fff;'
                f'border-radius:4px;padding:0 4px;text-decoration:line-through">{text}</span>'
            )
    return "".join(parts)


def timing_rows(trace: QueryTrace, judged: JudgeResult | None) -> tuple[tuple[str, float], ...]:
    rows = [("retrieve", trace.retrieval_ms), ("generate", trace.answer.latency_ms)]
    if judged is not None:
        rows.append(("judge", judged.judge_ms))
    return tuple(rows)


def verdict_dot(result: JudgeResult) -> str:
    """Graphviz source: each cited sentence linked to its source, green = supported."""
    verdicts = result.citations.verdicts
    lines = [
        "digraph verdicts {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box style="rounded,filled" fillcolor="#f8fafc" color="#cbd5e1" '
        'fontname="Helvetica" fontsize=10];',
    ]
    if not verdicts:
        lines += ['  none [label="No citations to check"];', "}"]
        return "\n".join(lines)
    sentences = list(dict.fromkeys(v.sentence for v in verdicts))
    for i, sentence in enumerate(sentences):
        lines.append(f'  s{i} [label="{dot_text(sentence)}"];')
    for number, doc_id in dict.fromkeys((v.number, v.doc_id) for v in verdicts):
        lines.append(
            f'  src{number} [label="[{number}] {dot_text(doc_id, 40)}" '
            f'color="{source_color(number)}" penwidth=2];'
        )
    for v in verdicts:
        colour = YES_COLOR if v.supported else INVALID_COLOR
        style = "solid" if v.supported else "dashed"
        word = "supports" if v.supported else "does not support"
        lines.append(
            f"  s{sentences.index(v.sentence)} -> src{v.number} "
            f'[color="{colour}" style={style} label="{word}" dir=back];'
        )
    lines.append("}")
    return "\n".join(lines)


def hit_vs_faithfulness(
    retrieval: Mapping[str, Any], generation: Mapping[str, Any]
) -> tuple[tuple[str, float, int], ...]:
    """Mean faithfulness of answers whose retrieval found every relevant doc vs missed one."""
    recall = {e["example_id"]: e["scores"]["recall"] for e in retrieval.get("examples", [])}
    groups: dict[str, list[float]] = {"found every relevant doc": [], "missed a relevant doc": []}
    for example in generation.get("examples", []):
        faithfulness = example["scores"].get("faithfulness")
        example_recall = recall.get(example["example_id"])
        if faithfulness is None or example_recall is None:
            continue
        key = "found every relevant doc" if example_recall == 1.0 else "missed a relevant doc"
        groups[key].append(float(faithfulness))
    return tuple((label, fmean(values), len(values)) for label, values in groups.items() if values)
```

Note the edge direction: `dir=back` draws the arrow from source to sentence while the DOT text keeps `s{i} -> src{n}` (matched by the tests).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_query_visuals.py tests/unit/test_generator.py -v`
Expected: PASS.

- [ ] **Step 6: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`

---

### Task 6: Plotly chart builders

**Files:**
- Modify: `pyproject.toml` (plotly in `ui` — already added by `uv add --group ui plotly`; mypy override; coverage omit)
- Create: `src/rag_eval_platform/playground/charts.py`
- Test: `tests/unit/test_charts.py`

**Interfaces:**
- Consumes: data classes from Tasks 3–5.
- Produces (all return `plotly.graph_objects.Figure`, all accept `height: int` and where noted `progress: float = 1.0`):
  - `chunk_size_chart(bins, progress, height)`, `chunk_layout_chart(spans, height)`
  - `meaning_map_chart(points, dims, progress, height)`, `embedding_strip_chart(vectors: Mapping[str, Sequence[float]], height)`
  - `similarity_chart(rows, cutoff: int | None, progress, height)`, `rerank_chart(moves, height)`, `funnel_chart(stages, height)`
  - `prompt_chart(blocks, height)`, `timing_chart(rows, progress, height)`
  - `gauge_chart(title, value, threshold: float | None, progress, height)`
  - `scores_chart(names, values, marks, height)`, `per_type_chart(by_type: Mapping[str, Mapping[str, Any]], metrics: Sequence[str], height)`
  - `CHART_CONFIG = {"displayModeBar": False}`

- [ ] **Step 1: Configure pyproject**

Confirm `plotly>=7.1.0` is in the `ui` group (run `uv add --group ui plotly` if not). Add `"plotly", "plotly.*"` to the mypy override list next to `"streamlit", "streamlit.*"`. Change the coverage omit to:

```toml
omit = ["*/playground/app.py", "*/playground/shared.py", "*/playground/tabs/*", "*/playground/charts.py"]
```

and update its comment to: `# Drawing code; its logic lives in core.py, visuals.py and query_visuals.py (unit-tested). charts.py is tested when Plotly is installed.`

- [ ] **Step 2: Write the failing tests**

`tests/unit/test_charts.py`:

```python
"""Tests for playground.charts: each builder returns a figure with the expected data."""

import pytest

pytest.importorskip("plotly")

from rag_eval_platform.playground import charts  # noqa: E402
from rag_eval_platform.playground.query_visuals import (  # noqa: E402
    PromptBlock,
    RerankMove,
    SimilarityRow,
)
from rag_eval_platform.playground.visuals import ChunkSpan, MapPoint, SizeBin  # noqa: E402

ROWS = (
    SimilarityRow(1, "a#0", 0.8, kept=True, cited=True),
    SimilarityRow(2, "b#0", 0.6, kept=True, cited=False),
    SimilarityRow(3, "c#0", 0.4, kept=False, cited=False),
)


def test_chunk_size_chart_grows_with_progress() -> None:
    bins = (SizeBin(100, 200, 4), SizeBin(200, 300, 2))

    half = charts.chunk_size_chart(bins, progress=0.5, height=200)

    assert list(half.data[0].y) == [2.0, 1.0]
    assert half.layout.yaxis.range[1] >= 4  # fixed axis, so bars visibly grow
    assert half.layout.height == 200


def test_chunk_layout_chart_draws_chunks_and_overlaps() -> None:
    spans = (ChunkSpan("d", "d#0", 0, 0, 6, 0), ChunkSpan("d", "d#1", 1, 4, 10, 2))

    fig = charts.chunk_layout_chart(spans, height=200)

    names = [t.name for t in fig.data]
    assert names == ["chunk", "overlap"]
    assert list(fig.data[1].x) == [2]
    assert list(fig.data[1].base) == [4]


def test_meaning_map_chart_2d_and_3d() -> None:
    points = (
        MapPoint("a#0", "chunk", 1.0, 2.0, 0.0, "a"),
        MapPoint("b#0", "retrieved", -1.0, 0.0, 0.0, "b"),
        MapPoint("question", "question", 0.5, 0.5, 0.0, "q"),
    )

    flat = charts.meaning_map_chart(points, dims=2, progress=0.5, height=300)
    deep = charts.meaning_map_chart(points, dims=3, progress=1.0, height=300)

    assert [t.name for t in flat.data] == ["chunks", "retrieved", "your question"]
    assert list(flat.data[0].x) == [0.5]  # spread grows from the centre
    assert deep.data[0].type == "scatter3d"


def test_embedding_strip_chart_has_one_row_per_vector() -> None:
    fig = charts.embedding_strip_chart({"question": [0.1, -0.2], "a#0": [0.3, 0.0]}, height=150)

    assert list(fig.data[0].y) == ["question", "a#0"]


def test_similarity_chart_cutoff_without_reranking() -> None:
    fig = charts.similarity_chart(ROWS, cutoff=2, progress=1.0, height=250)

    assert list(fig.data[0].y) == [0.8, 0.6, 0.4]
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == pytest.approx(1.5)


def test_similarity_chart_cutoff_hidden_with_reranking() -> None:
    fig = charts.similarity_chart(ROWS, cutoff=None, progress=1.0, height=250)

    assert len(fig.layout.shapes) == 0


def test_rerank_chart_has_a_line_per_candidate() -> None:
    fig = charts.rerank_chart((RerankMove("a#0", 1, 2), RerankMove("b#0", 2, None)), height=250)

    assert len(fig.data) == 2
    assert list(fig.data[1].y) == [2, 3]  # dropped: falls below the kept ranks


def test_funnel_and_prompt_and_timing_charts() -> None:
    funnel = charts.funnel_chart((("all chunks", 40), ("candidates", 20)), height=200)
    prompt = charts.prompt_chart(
        (PromptBlock("rules", "System rules", "x" * 10), PromptBlock("question", "Question", "q?")),
        height=150,
    )
    timing = charts.timing_chart((("retrieve", 500.0), ("generate", 4000.0)), progress=0.5,
                                 height=120)  # fmt: skip

    assert list(funnel.data[0].x) == [40, 20]
    assert [t.x[0] for t in prompt.data] == [10, 2]
    assert [t.x[0] for t in timing.data] == [0.25, 2.0]


def test_gauge_and_report_charts() -> None:
    gauge = charts.gauge_chart("faithfulness", 0.9, threshold=0.85, progress=0.5, height=180)
    scores = charts.scores_chart(["faithfulness", "relevance"], [0.9, 0.7], [0.85, None],
                                 height=200)  # fmt: skip
    per_type = charts.per_type_chart(
        {"short": {"faithfulness": 0.9, "answer_relevance": 0.8}},
        ["faithfulness", "answer_relevance"], height=200,
    )  # fmt: skip

    assert gauge.data[0].value == pytest.approx(0.45)
    assert gauge.data[0].gauge.threshold.value == 0.85
    assert list(scores.data[0].y) == [0.9, 0.7]
    assert list(scores.data[1].y) == [0.85]  # only metrics with a pass mark
    assert [t.name for t in per_type.data] == ["faithfulness", "answer_relevance"]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_charts.py -v`
Expected: FAIL (`cannot import name 'charts'`).

- [ ] **Step 4: Implement**

`src/rag_eval_platform/playground/charts.py`:

```python
"""Plotly figures for the playground, built from the pure data in visuals.py and
query_visuals.py. Builders take ``progress`` (0..1) so a chart can grow in, and a fixed
``height`` so the Overview fits on one screen."""

from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

from rag_eval_platform.playground.query_visuals import (
    INVALID_COLOR,
    YES_COLOR,
    PromptBlock,
    RerankMove,
    SimilarityRow,
)
from rag_eval_platform.playground.visuals import ChunkSpan, MapPoint, SizeBin

CHART_CONFIG = {"displayModeBar": False}
NEUTRAL, MUTED, ACCENT, KEPT = "#2563eb", "#cbd5e1", "#d97706", "#60a5fa"
BLOCK_COLORS = {"rules": "#6366f1", "source": "#16a34a", "question": "#d97706"}


def _layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title={"text": title, "font": {"size": 13}} if title else None,
        margin={"l": 8, "r": 8, "t": 32 if title else 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Helvetica, Arial, sans-serif", "size": 11},
    )
    return fig


def chunk_size_chart(
    bins: Sequence[SizeBin], progress: float = 1.0, height: int = 300
) -> go.Figure:
    top = max((b.count for b in bins), default=1)
    fig = go.Figure(go.Bar(
        x=[f"{b.start}–{b.end}" for b in bins],
        y=[b.count * progress for b in bins],
        marker_color=NEUTRAL,
        hovertemplate="%{x} chars: %{y:.0f} chunks<extra></extra>",
    ))  # fmt: skip
    fig.update_yaxes(range=[0, top * 1.15], title="chunks")
    fig.update_xaxes(title="chunk size (characters)")
    return _layout(fig, height)


def chunk_layout_chart(spans: Sequence[ChunkSpan], height: int = 300) -> go.Figure:
    labels = [f"#{s.index}" for s in spans]
    fig = go.Figure()
    fig.add_bar(name="chunk", y=labels, x=[s.end - s.start for s in spans],
                base=[s.start for s in spans], orientation="h", marker_color=KEPT,
                hovertemplate="chars %{base}–%{x}<extra></extra>")  # fmt: skip
    overlapping = [s for s in spans if s.overlap]
    fig.add_bar(name="overlap", y=[f"#{s.index}" for s in overlapping],
                x=[s.overlap for s in overlapping], base=[s.start for s in overlapping],
                orientation="h", marker_color=INVALID_COLOR, opacity=0.8,
                hovertemplate="%{x} chars shared with the previous chunk<extra></extra>")  # fmt: skip
    fig.update_layout(barmode="overlay", legend={"orientation": "h", "y": 1.08})
    fig.update_yaxes(autorange="reversed", title="chunk")
    fig.update_xaxes(title="position in the document (characters)")
    return _layout(fig, height)


def meaning_map_chart(
    points: Sequence[MapPoint], dims: int = 2, progress: float = 1.0, height: int = 400
) -> go.Figure:
    styles = {
        "chunk": ("chunks", MUTED, 6, "circle"),
        "retrieved": ("retrieved", NEUTRAL, 11, "circle"),
        "question": ("your question", ACCENT, 18, "diamond"),
    }
    fig = go.Figure()
    for kind, (name, color, size, symbol) in styles.items():
        group = [p for p in points if p.kind == kind]
        common: dict[str, Any] = {
            "name": name,
            "mode": "markers",
            "hovertext": [p.text for p in group],
            "hoverinfo": "text",
            "marker": {"color": color, "size": size, "symbol": symbol, "line": {"width": 0}},
        }
        xs = [p.x * progress for p in group]
        ys = [p.y * progress for p in group]
        if dims == 3:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[p.z * progress for p in group], **common))
        else:
            fig.add_trace(go.Scatter(x=xs, y=ys, **common))
    extent = max((max(abs(p.x), abs(p.y)) for p in points), default=1.0) * 1.1 or 1.0
    if dims == 2:
        fig.update_xaxes(range=[-extent, extent], showticklabels=False, zeroline=False)
        fig.update_yaxes(range=[-extent, extent], showticklabels=False, zeroline=False)
    fig.update_layout(legend={"orientation": "h", "y": 1.08})
    return _layout(fig, height)


def embedding_strip_chart(vectors: Mapping[str, Sequence[float]], height: int = 160) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=[list(v) for v in vectors.values()], y=list(vectors), colorscale="RdBu", zmid=0,
        showscale=False, hovertemplate="%{y} · dim %{x}: %{z:.3f}<extra></extra>",
    ))  # fmt: skip
    fig.update_xaxes(title="dimension", showgrid=False)
    return _layout(fig, height)


def similarity_chart(
    rows: Sequence[SimilarityRow], cutoff: int | None, progress: float = 1.0, height: int = 300
) -> go.Figure:
    colors = [YES_COLOR if r.cited else KEPT if r.kept else MUTED for r in rows]
    fig = go.Figure(go.Bar(
        x=[f"#{r.rank}" for r in rows], y=[r.score * progress for r in rows],
        marker_color=colors, hovertext=[r.chunk_id for r in rows],
        hovertemplate="%{hovertext}: %{y:.3f}<extra></extra>",
    ))  # fmt: skip
    if cutoff is not None and cutoff < len(rows):
        fig.add_vline(x=cutoff - 0.5, line_dash="dash", line_color=ACCENT,
                      annotation_text=f"top-{cutoff}", annotation_position="top")  # fmt: skip
    top = max((r.score for r in rows), default=1.0)
    fig.update_yaxes(range=[0, top * 1.15], title="cosine similarity")
    return _layout(fig, height)


def rerank_chart(moves: Sequence[RerankMove], height: int = 300) -> go.Figure:
    dropped_rank = len(moves) + 1
    fig = go.Figure()
    for move in moves:
        kept = move.after is not None
        fig.add_trace(go.Scatter(
            x=["vector search", "cross-encoder"], y=[move.before, move.after or dropped_rank],
            mode="lines+markers", name=move.chunk_id, showlegend=False,
            line={"color": NEUTRAL if kept else MUTED, "width": 3 if kept else 1},
            hovertemplate=f"{move.chunk_id}: rank %{{y}}<extra></extra>",
        ))  # fmt: skip
    fig.update_yaxes(autorange="reversed", title="rank", dtick=1)
    return _layout(fig, height)


def funnel_chart(stages: Sequence[tuple[str, int]], height: int = 250) -> go.Figure:
    fig = go.Figure(go.Funnel(
        y=[name for name, _ in stages], x=[value for _, value in stages],
        marker={"color": [MUTED, KEPT, NEUTRAL, YES_COLOR][: len(stages)]},
    ))  # fmt: skip
    return _layout(fig, height)


def prompt_chart(blocks: Sequence[PromptBlock], height: int = 140) -> go.Figure:
    fig = go.Figure()
    for block in blocks:
        fig.add_bar(name=block.title, y=["prompt"], x=[len(block.text)], orientation="h",
                    marker_color=BLOCK_COLORS[block.kind], text=[block.title],
                    textposition="inside", hovertemplate=f"{block.title}: %{{x}} chars<extra></extra>")  # fmt: skip
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_xaxes(title="characters")
    return _layout(fig, height)


def timing_chart(
    rows: Sequence[tuple[str, float]], progress: float = 1.0, height: int = 120
) -> go.Figure:
    colors = {"retrieve": KEPT, "generate": NEUTRAL, "judge": ACCENT}
    total = sum(ms for _, ms in rows) / 1000 or 1.0
    fig = go.Figure()
    for step, ms in rows:
        seconds = ms / 1000
        fig.add_bar(name=step, y=["time"], x=[seconds * progress], orientation="h",
                    marker_color=colors.get(step, MUTED), text=[f"{step} {seconds:.1f}s"],
                    textposition="inside")  # fmt: skip
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_xaxes(range=[0, total * 1.02], title="seconds")
    return _layout(fig, height)


def gauge_chart(
    title: str, value: float, threshold: float | None, progress: float = 1.0, height: int = 200
) -> go.Figure:
    passed = threshold is None or value >= threshold
    gauge: dict[str, Any] = {
        "axis": {"range": [0, 1]},
        "bar": {"color": YES_COLOR if passed else INVALID_COLOR},
    }
    if threshold is not None:
        gauge["threshold"] = {"value": threshold, "line": {"color": "#111827", "width": 3}}
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value * progress,
                                 number={"valueformat": ".2f"}, gauge=gauge,
                                 title={"text": title, "font": {"size": 13}}))  # fmt: skip
    return _layout(fig, height)


def scores_chart(
    names: Sequence[str], values: Sequence[float], marks: Sequence[float | None], height: int = 250
) -> go.Figure:
    colors = [
        NEUTRAL if mark is None else YES_COLOR if value >= mark else INVALID_COLOR
        for value, mark in zip(values, marks, strict=True)
    ]
    fig = go.Figure(go.Bar(x=list(names), y=list(values), marker_color=colors,
                           text=[f"{v:.3f}" for v in values], textposition="outside",
                           name="score"))  # fmt: skip
    with_marks = [(n, m) for n, m in zip(names, marks, strict=True) if m is not None]
    fig.add_trace(go.Scatter(x=[n for n, _ in with_marks], y=[m for _, m in with_marks],
                             mode="markers", name="pass mark",
                             marker={"symbol": "line-ew", "size": 40, "color": "#111827",
                                     "line": {"width": 3}}))  # fmt: skip
    fig.update_yaxes(range=[0, 1.1])
    fig.update_layout(showlegend=False)
    return _layout(fig, height)


def per_type_chart(
    by_type: Mapping[str, Mapping[str, Any]], metrics: Sequence[str], height: int = 250
) -> go.Figure:
    types = list(by_type)
    fig = go.Figure()
    for metric in metrics:
        fig.add_bar(name=metric, x=types, y=[by_type[t].get(metric) or 0.0 for t in types])
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.1})
    fig.update_yaxes(range=[0, 1.1])
    return _layout(fig, height)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_charts.py -v`
Expected: PASS. If a Plotly attribute path differs (e.g. `fig.layout.shapes[0].x0`), adjust the assertion to Plotly's actual structure, not the behaviour.

- [ ] **Step 6: Checkpoint (no commit)**

Run: `uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit -q`

---

### Task 7: App shell, shared helpers, ① Ingest and ② Embed tabs

**Files:**
- Create: `src/rag_eval_platform/playground/shared.py`, `src/rag_eval_platform/playground/tabs/__init__.py`, `tabs/ingest.py`, `tabs/embed.py`
- Rewrite: `src/rag_eval_platform/playground/app.py`
- Modify: `tests/integration/test_playground.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces: `shared.TabContext`, `shared.animated_chart(key, data_key, build, *, height)`, `shared.show_pipeline(active, ctx)`, session keys `TRACE`, `SUMMARY`, `INDEXED_WITH`, `INDEX_VERSION`, `SKIPPED`, `JUDGED`; `tabs.<name>.render(ctx: TabContext) -> None`.

- [ ] **Step 1: Update the smoke test (failing)**

In `tests/integration/test_playground.py`, replace the tab-label assertion:

```python
    assert [t.label for t in app.tabs] == ["Overview", "① Ingest", "② Embed"]
```

(Tasks 8–10 extend this list.) Run `uv run pytest tests/integration/test_playground.py::test_app_renders_without_errors -v` — expected FAIL (old labels).

- [ ] **Step 2: Write `shared.py`**

```python
"""Streamlit helpers shared by the playground tabs: settings, cached resources, session
state, the pipeline header and play-once animated charts. Drawing only; the logic lives in
core.py, visuals.py and query_visuals.py."""

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import get_args

import plotly.graph_objects as go
import streamlit as st

from rag_eval_platform.config.settings import ChunkStrategy, ReasoningEffort, get_settings
from rag_eval_platform.evaluation.citation_validity import CitationChecker, create_citation_checker
from rag_eval_platform.evaluation.judge import RagasJudge
from rag_eval_platform.ingestion.embedding import SentenceTransformerEmbedder
from rag_eval_platform.playground.animation import ease_steps, should_animate
from rag_eval_platform.playground.charts import CHART_CONFIG
from rag_eval_platform.playground.core import (
    PLAYGROUND_COLLECTION,
    IndexSnapshot,
    JudgeResult,
    QueryTrace,
    index_snapshot,
    trace_key,
)
from rag_eval_platform.playground.visuals import PipelineStats, pipeline_dot
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker
from rag_eval_platform.retrieval.vector_store import ChromaVectorStore, VectorStoreError

settings = get_settings()

TRACE, SUMMARY, INDEXED_WITH, INDEX_VERSION = "trace", "summary", "indexed_with", "index_version"
SKIPPED, JUDGED = "skipped", "judged"
FRAMES, FRAME_SECONDS = 8, 0.05
TAB_LABELS = ["Overview", "① Ingest", "② Embed"]  # extended as tabs are added


@dataclass(frozen=True)
class Options:
    strategy: ChunkStrategy
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank: bool
    rerank_candidates: int
    model: str
    reasoning_effort: ReasoningEffort
    temperature: float


@dataclass(frozen=True)
class TabContext:
    options: Options
    store: ChromaVectorStore | None
    models: list[str] | None
    snapshot: IndexSnapshot | None
    trace: QueryTrace | None
    judged: JudgeResult | None

    @property
    def stats(self) -> PipelineStats:
        snapshot, trace, judged = self.snapshot, self.trace, self.judged
        return PipelineStats(
            documents=len(snapshot.doc_ids) if snapshot else 0,
            chunks=len(snapshot.chunks) if snapshot else 0,
            candidates=len(trace.vector_results) if trace else 0,
            top_k=len(trace.final_results) if trace else 0,
            cited=len({c.number for c in trace.answer.citations}) if trace else 0,
            faithfulness=judged.scores.faithfulness if judged else None,
        )


# ---------- cached resources (loaded once per server, not on every click) ----------


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.from_pretrained(model_name)


@st.cache_resource(show_spinner="Loading the re-ranker...")
def load_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker.from_pretrained(model_name)


@st.cache_resource(show_spinner=False)
def connect_store(host: str, port: int) -> ChromaVectorStore:
    return ChromaVectorStore.connect(host, port, PLAYGROUND_COLLECTION)


@st.cache_resource(show_spinner="Loading the judge...")
def load_judge(judge_model: str) -> tuple[RagasJudge, CitationChecker]:
    embedder = load_embedder(settings.embedding_model)
    return RagasJudge.from_settings(settings, embedder), create_citation_checker(settings)


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models(base_url: str) -> list[str] | None:
    """Models available in Ollama, or None when Ollama is not reachable."""
    url = base_url.removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - configured URL
            return sorted(m["name"] for m in json.load(response)["models"])
    except (OSError, ValueError, KeyError):
        return None


@st.cache_data(show_spinner="Reading the index...", max_entries=3)
def _snapshot(_store: ChromaVectorStore, version: int, count: int) -> IndexSnapshot:
    return index_snapshot(_store)  # cached per index version; _store is not hashed


def current_snapshot(store: ChromaVectorStore | None) -> IndexSnapshot | None:
    if store is None:
        return None
    try:
        count = store.count()
        return _snapshot(store, st.session_state.get(INDEX_VERSION, 0), count) if count else None
    except VectorStoreError as exc:
        st.error(str(exc))
        return None


def current_judged(trace: QueryTrace | None) -> JudgeResult | None:
    """The judge result, only if it belongs to the current answer."""
    stored = st.session_state.get(JUDGED)
    if trace is None or stored is None:
        return None
    key, result = stored
    return result if key == trace_key(trace) else None


# ---------- page parts ----------


def sidebar(models: list[str] | None) -> Options:
    st.sidebar.header("Settings")
    st.sidebar.caption("Change these and watch every tab react.")
    st.sidebar.subheader("① Chunking")
    strategy = st.sidebar.segmented_control(
        "Strategy", ["recursive", "fixed"], default=settings.chunk_strategy, required=True
    )
    chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2000, settings.chunk_size, 50)
    chunk_overlap = st.sidebar.slider(
        "Overlap (characters)", 0, min(400, chunk_size - 50), min(settings.chunk_overlap, 400), 10
    )
    st.sidebar.subheader("③ Retrieval")
    top_k = st.sidebar.slider("Top-k chunks sent to the model", 1, 10, settings.top_k)
    rerank = st.sidebar.toggle("Re-rank with a cross-encoder", value=settings.rerank)
    rerank_candidates = st.sidebar.slider(
        "Candidates before re-ranking", top_k, 30, max(settings.rerank_candidates, top_k),
        disabled=not rerank,
    )  # fmt: skip
    st.sidebar.subheader("④ Generation")
    choices = models or [settings.llm_model]
    model = st.sidebar.selectbox(
        "Model", choices,
        index=choices.index(settings.llm_model) if settings.llm_model in choices else 0,
    )  # fmt: skip
    effort = st.sidebar.segmented_control(
        "Thinking (reasoning effort)",
        [e for e in get_args(ReasoningEffort) if e != "default"],
        default=settings.llm_reasoning_effort, required=True,
    )  # fmt: skip
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, settings.llm_temperature, 0.1)
    return Options(strategy, chunk_size, chunk_overlap, top_k, rerank, rerank_candidates,
                   model, effort, temperature)  # fmt: skip


def service_status(models: list[str] | None) -> ChromaVectorStore | None:
    left, right = st.columns(2)
    store: ChromaVectorStore | None
    try:
        store = connect_store(settings.chroma_host, settings.chroma_port)
        left.success(f"Chroma connected · {store.count()} chunks in '{PLAYGROUND_COLLECTION}'")
    except VectorStoreError:
        store = None
        left.error("Chroma is not running: `docker compose -f docker/docker-compose.yml up -d`")
    if models is None:
        right.error("Ollama is not running: `brew services start ollama`")
    else:
        right.success(f"Ollama connected · {len(models)} model(s)")
    return store


def show_pipeline(active: str | None, ctx: TabContext) -> None:
    st.graphviz_chart(pipeline_dot(active, ctx.stats), width="stretch")


def animated_chart(
    key: str, data_key: str, build: Callable[[float], go.Figure], *, height: int | None = None
) -> None:
    """Draw ``build(progress)``; grow it in (progress 0 -> 1) the first time ``data_key`` is shown."""
    placeholder = st.empty()
    if should_animate(st.session_state, key, data_key):
        for i, progress in enumerate(ease_steps(FRAMES)):
            placeholder.plotly_chart(build(progress), key=f"{key}-{i}", config=CHART_CONFIG,
                                     height=height or "content")  # fmt: skip
            time.sleep(FRAME_SECONDS)
    else:
        placeholder.plotly_chart(build(1.0), key=f"{key}-final", config=CHART_CONFIG,
                                 height=height or "content")  # fmt: skip
```

- [ ] **Step 3: Write `tabs/__init__.py`, `tabs/ingest.py`, `tabs/embed.py`**

`tabs/__init__.py`:

```python
"""One module per playground tab; each exposes ``render(ctx)``."""
```

`tabs/ingest.py`:

```python
"""① Ingest: upload, index, and see how documents become chunks."""

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.ingestion.loaders import DocumentLoadError
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import UPLOAD_DIR, UploadError, build_index, save_uploads
from rag_eval_platform.playground.shared import (
    INDEX_VERSION, INDEXED_WITH, JUDGED, SKIPPED, SUMMARY, TRACE, TabContext, animated_chart,
    load_embedder, settings, show_pipeline,
)  # fmt: skip
from rag_eval_platform.playground.visuals import (
    chunk_spans, index_counts, ingest_flow_dot, size_bins,
)  # fmt: skip
from rag_eval_platform.retrieval.vector_store import VectorStoreError


def render(ctx: TabContext) -> None:
    show_pipeline("Ingest", ctx)
    left, right = st.columns([1, 2], gap="large")
    with left:
        _upload(ctx)
    with right:
        _index_views(ctx)


def _upload(ctx: TabContext) -> None:
    st.markdown("**Upload** PDFs (with selectable text), Markdown or text. "
                "Building the index **replaces** the previous upload.")  # fmt: skip
    files = st.file_uploader("Documents", type=["pdf", "md", "txt"], accept_multiple_files=True)
    clicked = st.button("Build index", type="primary", disabled=not files or ctx.store is None)
    options = ctx.options
    if clicked and files and ctx.store is not None:
        try:
            with st.status("Building the index...", expanded=True) as status:
                uploaded = save_uploads([(f.name, f.getvalue()) for f in files], UPLOAD_DIR)
                st.session_state[SKIPPED] = uploaded.skipped
                built = build_index(
                    UPLOAD_DIR, strategy=options.strategy, chunk_size=options.chunk_size,
                    chunk_overlap=options.chunk_overlap,
                    embedder=load_embedder(settings.embedding_model), store=ctx.store,
                    on_progress=st.write,
                )  # fmt: skip
                # Saved before the next st.* call, which could stop the script on a click.
                st.session_state[SUMMARY] = built
                st.session_state[INDEXED_WITH] = (options.strategy, options.chunk_size,
                                                  options.chunk_overlap)  # fmt: skip
                st.session_state[INDEX_VERSION] = st.session_state.get(INDEX_VERSION, 0) + 1
                st.session_state.pop(TRACE, None)  # old answers cite the old index
                st.session_state.pop(JUDGED, None)
                status.update(label="Index ready", state="complete", expanded=False)
        except (UploadError, DocumentLoadError, VectorStoreError, OptionalDependencyError,
                ValueError) as exc:  # fmt: skip
            st.error(str(exc))
            return
        st.rerun()  # redraw every view from the new index
    for note in st.session_state.get(SKIPPED, ()):
        st.warning(f"Skipped duplicate: {note}")
    indexed_with = st.session_state.get(INDEXED_WITH)
    if indexed_with and indexed_with != (
        options.strategy,
        options.chunk_size,
        options.chunk_overlap,
    ):
        st.info("Chunk settings changed since the last build. Click **Build index** to apply them.")


def _index_views(ctx: TabContext) -> None:
    snapshot = ctx.snapshot
    if snapshot is None:
        st.info("Nothing indexed yet: upload documents and click **Build index**.")
        return
    counts = index_counts(snapshot.chunks)
    st.graphviz_chart(ingest_flow_dot(counts), width="stretch")
    st.dataframe(
        [{"document": d, "pages": index_counts([c for c in snapshot.chunks if c.doc_id == d]).pages
          or None, "chunks": sum(1 for c in snapshot.chunks if c.doc_id == d)}
         for d in snapshot.doc_ids],
        hide_index=True, key="ingest-docs",
    )  # fmt: skip
    version = st.session_state.get(INDEX_VERSION, 0)
    sizes = [len(c.text) for c in snapshot.chunks]
    bins = size_bins(sizes)
    st.markdown("**Chunk sizes.** Most chunks should sit just under the chunk-size setting; "
                "small ones are document or section ends.")  # fmt: skip
    animated_chart("ingest-sizes", f"{version}-{len(sizes)}",
                   lambda p: charts.chunk_size_chart(bins, p, height=260))  # fmt: skip

    doc = st.selectbox("Document", snapshot.doc_ids, key="ingest-doc")
    chunks = [c for c in snapshot.chunks if c.doc_id == doc]
    st.markdown("**Where each chunk sits.** Red = characters shared with the previous chunk "
                "(the overlap setting), so a sentence cut at a boundary survives in one piece.")  # fmt: skip
    st.plotly_chart(charts.chunk_layout_chart(chunk_spans(chunks),
                                              height=min(600, 80 + 22 * len(chunks))),
                    key="ingest-layout", config=charts.CHART_CONFIG)  # fmt: skip
    number = st.number_input("Read chunk number", 0, len(chunks) - 1, 0, key="ingest-read")
    chunk = chunks[int(number)]
    st.caption(f"{chunk.id} · page {chunk.page or '-'} · {len(chunk.text)} characters")
    st.text(chunk.text)
```

`tabs/embed.py`:

```python
"""② Embed: every chunk as a point in meaning space, and your question among them."""

import math

import streamlit as st

from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import trace_key
from rag_eval_platform.playground.shared import (
    INDEX_VERSION,
    TabContext,
    animated_chart,
    show_pipeline,
)
from rag_eval_platform.playground.visuals import angle_degrees, meaning_map, vector_for


def render(ctx: TabContext) -> None:
    show_pipeline("Embed", ctx)
    snapshot, trace = ctx.snapshot, ctx.trace
    if snapshot is None:
        st.info("Nothing indexed yet: build an index in ① Ingest.")
        return
    st.markdown(
        "Each chunk became **384 numbers** (its *embedding*). Similar meanings point in similar "
        "directions. The map squeezes 384 dimensions down to 2 or 3 (PCA) so you can see them."
    )
    dims_label = st.segmented_control("Map", ["2-D", "3-D"], default="2-D", key="embed-dims")
    dims = 3 if dims_label == "3-D" else 2
    retrieved = {r.chunk.id for r in trace.final_results} if trace else set()
    query = trace.query_embedding if trace and trace.query_embedding else None
    points = meaning_map(snapshot, query, retrieved, dims=dims)
    data_key = (
        f"{st.session_state.get(INDEX_VERSION, 0)}-{trace_key(trace) if trace else ''}-{dims}"
    )
    animated_chart(f"embed-map-{dims}", data_key,
                   lambda p: charts.meaning_map_chart(points, dims, p, height=460))  # fmt: skip
    if trace is None or not trace.query_embedding:
        st.info("Ask a question (box above) to see where it lands among your chunks.")
        return

    best = trace.final_results[0].chunk.id
    best_vector = vector_for(snapshot, best)
    if best_vector is None:
        return
    st.markdown(f"**The meaning code.** Your question vs the top chunk `{best}`, dimension by "
                "dimension (blue = positive, red = negative). Similar texts share patterns.")  # fmt: skip
    st.plotly_chart(charts.embedding_strip_chart(
        {"your question": trace.query_embedding, best: best_vector}, height=150),
        key="embed-strips", config=charts.CHART_CONFIG)  # fmt: skip
    angle = angle_degrees(trace.query_embedding, best_vector)
    cols = st.columns(3)
    cols[0].metric("Angle", f"{angle:.0f}°")
    cols[1].metric("Cosine similarity", f"{math.cos(math.radians(angle)):.3f}")
    cols[2].metric("Meaning", "close" if angle < 60 else "loose" if angle < 80 else "unrelated")
    st.caption("0° = same direction (same meaning), 90° = unrelated. Retrieval ranks chunks by "
               "this cosine similarity.")  # fmt: skip
```

- [ ] **Step 4: Rewrite `app.py`**

```python
"""RAG Playground: watch every RAG phase work on your own documents.

Run:   uv run streamlit run src/rag_eval_platform/playground/app.py
Needs: Chroma (`docker compose -f docker/docker-compose.yml up -d`), Ollama with a model
       (`brew services start ollama`, `ollama pull qwen3:8b`), `uv sync --all-extras --all-groups`.

Uploads live in data/playground/ (git-ignored) and their own Chroma collection, so the
sample corpus and the golden-dataset evaluation are never affected. Logic lives in core.py,
visuals.py and query_visuals.py; this file and tabs/ only draw.
"""

from collections.abc import Iterator

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.generation.generator import (
    GenerationError,
    Generator,
    OpenAICompatibleClient,
)
from rag_eval_platform.playground.core import QueryTrace, retrieve_step
from rag_eval_platform.playground.shared import (
    JUDGED, TAB_LABELS, TRACE, Options, TabContext, current_judged, current_snapshot,
    load_embedder, load_reranker, ollama_models, service_status, settings, sidebar,
)  # fmt: skip
from rag_eval_platform.playground.tabs import embed, ingest
from rag_eval_platform.retrieval.vector_store import ChromaVectorStore, VectorStoreError

RENDERERS = {"① Ingest": ingest.render, "② Embed": embed.render}


def ask(options: Options, store: ChromaVectorStore | None, models: list[str] | None,
        ready: bool) -> None:  # fmt: skip
    """The question box shared by all tabs; streams the answer as the model writes it."""
    with st.form("ask", border=False):
        question_col, button_col = st.columns([7, 1], vertical_alignment="bottom")
        question = question_col.text_input(
            "Ask a question about your documents",
            placeholder="e.g. What does Newton's second law state?" if ready else
            "Build an index in ① Ingest first",
        )  # fmt: skip
        asked = button_col.form_submit_button(
            "Ask", type="primary", disabled=not ready or not models
        )
    if not asked or not question.strip() or store is None:
        return
    box = st.container(border=True)
    try:
        step = retrieve_step(
            question, embedder=load_embedder(settings.embedding_model), store=store,
            top_k=options.top_k,
            reranker=load_reranker(settings.reranker_model) if options.rerank else None,
            rerank_candidates=options.rerank_candidates,
        )  # fmt: skip
        llm = settings.model_copy(update={
            "llm_provider": "ollama", "llm_model": options.model,
            "llm_reasoning_effort": options.reasoning_effort,
            "llm_temperature": options.temperature,
        })  # fmt: skip
        stream = Generator(OpenAICompatibleClient.from_settings(llm)).stream(
            question, step.final_results
        )
        counter = box.empty()

        def counted() -> Iterator[str]:
            for n, piece in enumerate(stream, start=1):
                counter.caption(f"✍️ writing… {n} pieces received")
                yield piece

        box.write_stream(counted())
    except (GenerationError, VectorStoreError, OptionalDependencyError, ValueError) as exc:
        box.error(str(exc))
        return
    trace = QueryTrace(step.vector_results, step.final_results, stream.answer,
                       step.retrieval_ms, step.query_embedding)  # fmt: skip
    # Saved before any further st.* call, so a click now cannot lose the answer.
    st.session_state[TRACE] = trace
    st.session_state.pop(JUDGED, None)
    counter.caption(f"{trace.answer.model} · {trace.answer.output_tokens} tokens · "
                    f"generation {trace.answer.latency_ms / 1000:.1f} s")  # fmt: skip


st.set_page_config(page_title="RAG Playground", page_icon="📚", layout="wide")
st.title("RAG Playground")
st.caption("Watch every phase work on your own documents: ingest → embed → retrieve → "
           "generate → evaluate.")  # fmt: skip

available_models = ollama_models(settings.ollama_base_url)
options = sidebar(available_models)
store = service_status(available_models)
snapshot = current_snapshot(store)
ask(options, store, available_models, ready=snapshot is not None)

trace = st.session_state.get(TRACE)
ctx = TabContext(options, store, available_models, snapshot, trace, current_judged(trace))
for label, tab in zip(TAB_LABELS, st.tabs(TAB_LABELS, key="phase-tab", on_change="rerun"),
                      strict=True):  # fmt: skip
    if tab.open:
        with tab:
            if label == "Overview":
                st.info("The Overview arrives with the last task; open a phase tab.")
            else:
                RENDERERS[label](ctx)
```

(The Overview message is replaced in Task 10.)

- [ ] **Step 5: Run the smoke test and the unit tests**

Run: `uv run pytest tests/integration/test_playground.py::test_app_renders_without_errors tests/unit -q`
Expected: PASS. Fix any Streamlit API mismatch the smoke test reveals (e.g. `height` values for `plotly_chart`).

- [ ] **Step 6: Try it**

Run: `uv run streamlit run src/rag_eval_platform/playground/app.py`, upload a PDF, build the index, open ① Ingest and ② Embed, ask a question. Expected: the pipeline header highlights the tab; histogram grows in once; meaning map shows the question star; strips and angle appear.

- [ ] **Step 7: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`

---

### Task 8: ③ Retrieve and ④ Generate tabs

**Files:**
- Create: `src/rag_eval_platform/playground/tabs/retrieve.py`, `tabs/generate.py`
- Modify: `src/rag_eval_platform/playground/app.py` (`RENDERERS`), `shared.py` (`TAB_LABELS`), `tests/integration/test_playground.py`

**Interfaces:**
- Consumes: `similarity_rows`, `was_reranked`, `rerank_moves`, `funnel_stages`, `prompt_blocks`, `answer_spans`, `spans_html`, `source_color`, `origin`, `timing_rows` (Task 5); chart builders (Task 6); `TabContext`, `animated_chart`, `show_pipeline` (Task 7).
- Produces: `tabs.retrieve.render`, `tabs.generate.render`.

- [ ] **Step 1: Extend the smoke test (failing)**

Set the expected labels to `["Overview", "① Ingest", "② Embed", "③ Retrieve", "④ Generate"]`; run it — expected FAIL.

- [ ] **Step 2: Write `tabs/retrieve.py`**

```python
"""③ Retrieve: which chunks came closest, where the top-k cut falls, and re-ranking."""

import streamlit as st

from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import trace_key
from rag_eval_platform.playground.query_visuals import (
    funnel_stages, rerank_moves, similarity_rows, was_reranked,
)  # fmt: skip
from rag_eval_platform.playground.shared import TabContext, animated_chart, show_pipeline


def render(ctx: TabContext) -> None:
    show_pipeline("Retrieve", ctx)
    trace = ctx.trace
    if trace is None:
        st.info("Ask a question (box above) to see retrieval at work.")
        return
    rows = similarity_rows(trace)
    reranked = was_reranked(trace)
    cutoff = None if reranked else len(trace.final_results)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**Similarity of every candidate.** Vector search ranks chunks by cosine "
                    "similarity. Blue = sent to the model, green = cited in the answer.")  # fmt: skip
        animated_chart("retrieve-similarity", trace_key(trace),
                       lambda p: charts.similarity_chart(rows, cutoff, p, height=320))  # fmt: skip
    with right:
        st.markdown("**The funnel.** From every chunk to the few the answer cites.")
        total = len(ctx.snapshot.chunks) if ctx.snapshot else len(trace.vector_results)
        st.plotly_chart(charts.funnel_chart(funnel_stages(total, trace), height=320),
                        key="retrieve-funnel", config=charts.CHART_CONFIG)  # fmt: skip
    if reranked:
        st.markdown("**Re-ranking.** The cross-encoder read each candidate *with* the question "
                    "and reordered them. Crossing lines = it disagreed with vector search; grey "
                    "lines fall out of the top-k.")  # fmt: skip
        st.plotly_chart(charts.rerank_chart(rerank_moves(trace), height=360),
                        key="retrieve-rerank", config=charts.CHART_CONFIG)  # fmt: skip
    else:
        st.caption("Turn on **Re-rank with a cross-encoder** in the sidebar and ask again to "
                   "compare rankings.")  # fmt: skip
    st.dataframe(
        [{"rank": r.rank, "chunk": r.chunk_id, "similarity": round(r.score, 3),
          "sent to model": r.kept, "cited": r.cited} for r in rows],
        hide_index=True, key="retrieve-table",
    )  # fmt: skip
```

- [ ] **Step 3: Write `tabs/generate.py`**

```python
"""④ Generate: the prompt the model read, its answer with citations, and the time it took."""

import streamlit as st

from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import trace_key
from rag_eval_platform.playground.query_visuals import (
    answer_spans, origin, prompt_blocks, source_color, spans_html, timing_rows,
)  # fmt: skip
from rag_eval_platform.playground.shared import TabContext, animated_chart, show_pipeline


def render(ctx: TabContext) -> None:
    show_pipeline("Generate", ctx)
    trace = ctx.trace
    if trace is None:
        st.info("Ask a question (box above) to see the prompt and the answer.")
        return
    answer = trace.answer
    blocks = prompt_blocks(trace)
    st.markdown("**The prompt**, in the order the model reads it: rules, numbered sources, "
                "your question (bar length = characters).")  # fmt: skip
    st.plotly_chart(charts.prompt_chart(blocks, height=130), key="generate-prompt",
                    config=charts.CHART_CONFIG)  # fmt: skip
    with st.expander("Read the full prompt"):
        for block in blocks:
            st.caption(block.title)
            st.code(block.text, language=None, wrap_lines=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**The answer.** Each citation has its source's colour.")
        if answer.is_refusal:
            st.warning("The model refused: the retrieved chunks did not contain the answer.")
        st.markdown(spans_html(answer_spans(answer)), unsafe_allow_html=True)
        if answer.invalid_citations:
            st.error(f"Cited numbers that were never provided: {list(answer.invalid_citations)}")
    with right:
        st.markdown("**Sources given to the model**")
        cited = {c.number for c in answer.citations}
        for number, result in enumerate(trace.final_results, start=1):
            mark = " · cited" if number in cited else ""
            st.markdown(
                f'<span style="color:{source_color(number)};font-weight:700">[{number}]</span> '
                f"{origin(result)}{mark}",
                unsafe_allow_html=True,
            )
    st.markdown("**Where the time went**")
    animated_chart("generate-timing", trace_key(trace),
                   lambda p: charts.timing_chart(timing_rows(trace, ctx.judged), p, height=110))  # fmt: skip
    st.caption(f"{answer.model} · {answer.input_tokens} tokens in → {answer.output_tokens} out · "
               f"prompt {answer.prompt_version}")  # fmt: skip
```

`origin` and `source_color` return plain strings; `origin` includes document ids, so escape it: use `html.escape(origin(result))` (add `import html`).

- [ ] **Step 4: Register the tabs**

In `shared.py`: `TAB_LABELS = ["Overview", "① Ingest", "② Embed", "③ Retrieve", "④ Generate"]`.
In `app.py`: `from rag_eval_platform.playground.tabs import embed, generate, ingest, retrieve` and
`RENDERERS = {"① Ingest": ingest.render, "② Embed": embed.render, "③ Retrieve": retrieve.render, "④ Generate": generate.render}`.

- [ ] **Step 5: Run the smoke test and unit tests**

Run: `uv run pytest tests/integration/test_playground.py::test_app_renders_without_errors tests/unit -q`
Expected: PASS.

- [ ] **Step 6: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`

---

### Task 9: ⑤ Evaluate tab — live judge and golden-set reports

**Files:**
- Create: `src/rag_eval_platform/playground/tabs/evaluate.py`
- Modify: `app.py` (`RENDERERS`), `shared.py` (`TAB_LABELS`), `tests/integration/test_playground.py`

**Interfaces:**
- Consumes: `judge_answer`, `JudgeResult`, `load_reports`, `REPORTS_DIR`, `trace_key` (Task 2); `verdict_dot`, `hit_vs_faithfulness` (Task 5); `gauge_chart`, `scores_chart`, `per_type_chart` (Task 6); `load_judge`, `JUDGED` (Task 7).
- Produces: `tabs.evaluate.render`.

- [ ] **Step 1: Extend the smoke test (failing)**

Expected labels: `["Overview", "① Ingest", "② Embed", "③ Retrieve", "④ Generate", "⑤ Evaluate"]`.

- [ ] **Step 2: Write `tabs/evaluate.py`**

```python
"""⑤ Evaluate: grade this answer with the LLM judge, and see the golden-set reports."""

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.evaluation.judge import JudgeError
from rag_eval_platform.generation.generator import GenerationError
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import REPORTS_DIR, judge_answer, load_reports, trace_key
from rag_eval_platform.playground.query_visuals import hit_vs_faithfulness, verdict_dot
from rag_eval_platform.playground.shared import (
    JUDGED, TabContext, animated_chart, load_judge, settings, show_pipeline,
)  # fmt: skip


def render(ctx: TabContext) -> None:
    show_pipeline("Evaluate", ctx)
    judge_col, reports_col = st.columns(2, gap="large")
    with judge_col:
        _judge_this_answer(ctx)
    with reports_col:
        _golden_reports()


def _judge_this_answer(ctx: TabContext) -> None:
    st.subheader("Judge this answer")
    st.caption(f"A second model ({settings.judge_model}) checks whether every claim is backed "
               "by the retrieved chunks, whether it answers the question, and whether each "
               "citation points to a chunk that supports it. Takes 1–2 minutes.")  # fmt: skip
    trace = ctx.trace
    if trace is None:
        st.info("Ask a question first (box above).")
        return
    if trace.answer.is_refusal:
        st.info("This answer is a refusal: it makes no claims, so there is nothing to judge.")
        return
    if ctx.judged is None:
        if st.button("Judge this answer", type="primary"):
            try:
                with st.status("Judging…", expanded=True) as status:
                    st.write("Loading the judge model…")
                    judge, checker = load_judge(settings.judge_model)
                    st.write("Scoring faithfulness and relevance, checking citations…")
                    result = judge_answer(trace, judge, checker)
                    # Saved before the next st.* call, so a click now cannot lose the result.
                    st.session_state[JUDGED] = (trace_key(trace), result)
                    status.update(label="Judged", state="complete")
            except OptionalDependencyError:
                st.error(
                    "The judge needs the evaluation extra: `uv sync --all-extras --all-groups`"
                )
                return
            except (JudgeError, GenerationError) as exc:
                st.error(str(exc))
                return
            st.rerun()
        return

    result, key = ctx.judged, trace_key(trace)
    scores = result.scores
    gauges = [
        ("faithfulness", scores.faithfulness, settings.min_faithfulness),
        ("answer relevance", scores.answer_relevance, settings.min_answer_relevance),
        ("citation validity", result.citations.score, None),
    ]
    cols = st.columns(3)
    for col, (name, value, mark) in zip(cols, gauges, strict=True):
        with col:
            if value is None:
                st.metric(name, "not scored")
            else:
                animated_chart(f"judge-{name}", key,
                               lambda p, n=name, v=value, m=mark: charts.gauge_chart(n, v, m, p, height=190))  # fmt: skip
    if result.note:
        st.warning(result.note)
    st.markdown("**Each citation's verdict** (green = the chunk supports the sentence).")
    st.graphviz_chart(verdict_dot(result), width="stretch")
    st.caption(f"Judged in {result.judge_ms / 1000:.0f} s")


def _golden_reports() -> None:
    st.subheader("Golden-set reports")
    reports = load_reports(REPORTS_DIR)
    if reports.generation is None and reports.retrieval is None:
        st.info("No reports yet. Create them (the generation run takes about an hour):")
        st.code("uv run python scripts/run_evaluation.py\n"
                "uv run python scripts/run_generation_evaluation.py", language="bash")  # fmt: skip
        return
    if reports.generation is not None:
        g = reports.generation
        st.markdown(f"**Answers** ({len(g['examples'])} questions, judge {g['judge_model']})")
        overall, marks = g["overall"], g["thresholds"]
        st.plotly_chart(charts.scores_chart(
            ["faithfulness", "answer relevance", "citation validity"],
            [overall["faithfulness"] or 0.0, overall["answer_relevance"] or 0.0,
             overall["citation_validity"] or 0.0],
            [marks["min_faithfulness"], marks["min_answer_relevance"], None], height=230),
            key="eval-gen-scores", config=charts.CHART_CONFIG)  # fmt: skip
        st.plotly_chart(charts.per_type_chart(
            g["by_query_type"], ["faithfulness", "answer_relevance", "citation_validity"],
            height=230), key="eval-gen-types", config=charts.CHART_CONFIG)  # fmt: skip
    if reports.retrieval is not None:
        r = reports.retrieval
        overall, marks = r["overall"], r["thresholds"]
        st.markdown(f"**Retrieval** (top-{r['k']})")
        st.plotly_chart(charts.scores_chart(
            ["recall@k", "mrr", "ndcg@k", "precision@k"],
            [overall["recall"], overall["mrr"], overall["ndcg"], overall["precision"]],
            [marks["min_recall_at_k"], marks["min_mrr"], marks["min_ndcg_at_k"], None],
            height=230), key="eval-ret-scores", config=charts.CHART_CONFIG)  # fmt: skip
    if reports.retrieval is not None and reports.generation is not None:
        joined = hit_vs_faithfulness(reports.retrieval, reports.generation)
        st.markdown("**Does bad retrieval cause unsupported answers?**")
        st.plotly_chart(charts.scores_chart(
            [f"{label} (n={n})" for label, _, n in joined], [mean for _, mean, _ in joined],
            [None] * len(joined), height=230), key="eval-join", config=charts.CHART_CONFIG)  # fmt: skip
```

- [ ] **Step 3: Register the tab**

`shared.py`: append `"⑤ Evaluate"` to `TAB_LABELS`. `app.py`: import `evaluate` and add `"⑤ Evaluate": evaluate.render` to `RENDERERS`.

- [ ] **Step 4: Run the smoke test and unit tests**

Run: `uv run pytest tests/integration/test_playground.py::test_app_renders_without_errors tests/unit -q`
Expected: PASS.

- [ ] **Step 5: Try the live judge**

In the running app: ask a question, open ⑤ Evaluate, click **Judge this answer**. Expected: status steps, then three gauges fill up once, verdict diagram, judge time. Switch tabs and back: gauges do not replay. Ask a new question: the judge result disappears (it belonged to the old answer).

- [ ] **Step 6: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`

---

### Task 10: Overview tab — everything on one screen

**Files:**
- Create: `src/rag_eval_platform/playground/tabs/overview.py`
- Modify: `app.py` (use `overview.render` for "Overview")

**Interfaces:**
- Consumes: everything above.
- Produces: `tabs.overview.render`, `overview.PANEL_HEIGHT = 250`.

- [ ] **Step 1: Write `tabs/overview.py`**

```python
"""Overview: all five phases for the latest question, on one screen with no scrolling."""

import streamlit as st

from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import REPORTS_DIR, load_reports, trace_key
from rag_eval_platform.playground.query_visuals import (
    answer_spans, similarity_rows, spans_html, timing_rows, was_reranked,
)  # fmt: skip
from rag_eval_platform.playground.shared import (
    INDEX_VERSION, TabContext, animated_chart, settings, show_pipeline,
)  # fmt: skip
from rag_eval_platform.playground.visuals import meaning_map, size_bins

PANEL_HEIGHT = 250  # each of the six panels; tuned so the page fits 1440x900 without scrolling
CHART_HEIGHT = PANEL_HEIGHT - 45


def render(ctx: TabContext) -> None:
    show_pipeline(None, ctx)
    top = st.columns(3)
    bottom = st.columns(3)
    panels = [
        (top[0], "① Ingest · chunk sizes", _ingest),
        (top[1], "② Embed · meaning map", _embed),
        (top[2], "③ Retrieve · similarity", _retrieve),
        (bottom[0], "④ Generate · answer", _generate),
        (bottom[1], "⑤ Evaluate · scores", _evaluate),
        (bottom[2], "⏱ Timing", _timing),
    ]
    for column, title, draw in panels:
        with column, st.container(height=PANEL_HEIGHT, border=True):
            st.markdown(f"**{title}**")
            draw(ctx)


def _ingest(ctx: TabContext) -> None:
    if ctx.snapshot is None:
        st.caption("Nothing indexed yet.")
        return
    bins = size_bins([len(c.text) for c in ctx.snapshot.chunks])
    animated_chart("ov-ingest", str(st.session_state.get(INDEX_VERSION, 0)),
                   lambda p: charts.chunk_size_chart(bins, p, height=CHART_HEIGHT))  # fmt: skip


def _embed(ctx: TabContext) -> None:
    if ctx.snapshot is None:
        st.caption("Nothing indexed yet.")
        return
    trace = ctx.trace
    retrieved = {r.chunk.id for r in trace.final_results} if trace else set()
    query = trace.query_embedding if trace and trace.query_embedding else None
    points = meaning_map(ctx.snapshot, query, retrieved, max_points=800)
    key = f"{st.session_state.get(INDEX_VERSION, 0)}-{trace_key(trace) if trace else ''}"
    animated_chart("ov-embed", key,
                   lambda p: charts.meaning_map_chart(points, 2, p, height=CHART_HEIGHT))  # fmt: skip


def _retrieve(ctx: TabContext) -> None:
    trace = ctx.trace
    if trace is None:
        st.caption("Ask a question to see retrieval.")
        return
    rows = similarity_rows(trace)
    cutoff = None if was_reranked(trace) else len(trace.final_results)
    animated_chart("ov-retrieve", trace_key(trace),
                   lambda p: charts.similarity_chart(rows, cutoff, p, height=CHART_HEIGHT))  # fmt: skip


def _generate(ctx: TabContext) -> None:
    trace = ctx.trace
    if trace is None:
        st.caption("Ask a question to see the answer.")
        return
    st.markdown(spans_html(answer_spans(trace.answer)), unsafe_allow_html=True)


def _evaluate(ctx: TabContext) -> None:
    if ctx.judged is not None and ctx.judged.scores.faithfulness is not None:
        s = ctx.judged.scores
        animated_chart("ov-evaluate", trace_key(ctx.trace) if ctx.trace else "",
                       lambda p: charts.scores_chart(
                           ["faithfulness", "relevance", "citations"],
                           [(s.faithfulness or 0.0) * p, (s.answer_relevance or 0.0) * p,
                            (ctx.judged.citations.score or 0.0) * p],  # type: ignore[union-attr]
                           [settings.min_faithfulness, settings.min_answer_relevance, None],
                           height=CHART_HEIGHT))  # fmt: skip
        return
    generation = load_reports(REPORTS_DIR).generation
    if generation is None:
        st.caption("Judge an answer in ⑤ Evaluate, or create the golden-set report.")
        return
    o, m = generation["overall"], generation["thresholds"]
    st.caption("Golden-set baseline (judge this answer in ⑤ Evaluate)")
    st.plotly_chart(charts.scores_chart(
        ["faithfulness", "relevance", "citations"],
        [o["faithfulness"] or 0.0, o["answer_relevance"] or 0.0, o["citation_validity"] or 0.0],
        [m["min_faithfulness"], m["min_answer_relevance"], None], height=CHART_HEIGHT - 20),
        key="ov-baseline", config=charts.CHART_CONFIG)  # fmt: skip


def _timing(ctx: TabContext) -> None:
    trace = ctx.trace
    if trace is None:
        st.caption("Ask a question to see the timings.")
        return
    animated_chart("ov-timing", trace_key(trace),
                   lambda p: charts.timing_chart(timing_rows(trace, ctx.judged), p,
                                                 height=CHART_HEIGHT))  # fmt: skip
```

- [ ] **Step 2: Use it in `app.py`**

Replace the Overview branch:

```python
    if tab.open:
        with tab:
            RENDERERS[label](ctx)
```

with `from rag_eval_platform.playground.tabs import embed, evaluate, generate, ingest, overview, retrieve` and `"Overview": overview.render` added to `RENDERERS`.

- [ ] **Step 3: Run the smoke test and unit tests**

Run: `uv run pytest tests/integration/test_playground.py::test_app_renders_without_errors tests/unit -q`
Expected: PASS.

- [ ] **Step 4: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests`

---

### Task 11: Animation polish — try Plotly's native transitions

**Files:**
- Modify (only if the spike succeeds): `src/rag_eval_platform/playground/charts.py`, `shared.py`

- [ ] **Step 1: Spike (throwaway, in the scratchpad)**

Write a two-button Streamlit script outside the repo that redraws one `go.Bar` with a stable `key` and `fig.update_layout(transition={"duration": 500, "easing": "cubic-in-out"})`, switching between two sets of y values on each click. Run it and click in a headless browser (screenshots at 0 ms and 250 ms after a click).

- [ ] **Step 2: Decide**

- If bars visibly glide between values: set the same `transition` in `charts._layout`, and in `animated_chart` skip the frame loop when the chart already exists for this key (draw once with `key=f"{key}-final"`), keeping the frame loop only for the first appearance.
- If not (Streamlit replaces the figure without Plotly.react): keep the frame loop as is. Record the result in the spec's §5 in one sentence.

- [ ] **Step 3: Checkpoint (no commit)**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit -q`

---

### Task 12: Verify in a real browser, then document

**Files:**
- Modify: `readme.md`, `CLAUDE.md`, `docs/architecture.md`, `docs/learning/phase-4.md` (or a new `docs/learning/playground.md`), the spec's §5 if Task 11 changed it

- [ ] **Step 1: Drive the real app**

Start: `uv run streamlit run src/rag_eval_platform/playground/app.py --server.headless true`. With the gstack browser (`$B = ~/.claude/skills/gstack/browse/dist/browse`) at viewport 1440x900:
1. Upload `tests`-built or real PDF, build the index, screenshot ① Ingest.
2. Ask a question; screenshot Overview (must not scroll: compare page height to 900 px with `$B js "document.documentElement.scrollHeight"`), ② Embed (2-D and 3-D), ③ Retrieve (with re-ranking on and off), ④ Generate.
3. Judge the answer; screenshot ⑤ Evaluate.
4. Switch tabs back and forth; confirm charts do not replay (screenshots 100 ms after switching show final frames).
Read every screenshot; fix layout issues (e.g. tune `PANEL_HEIGHT`) and repeat.

- [ ] **Step 2: Full checks**

Run: `uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit --cov -q && uv run pytest -m integration -q`
Expected: clean; unit coverage ≥ 80%.

- [ ] **Step 3: Documentation**

- `readme.md`: replace the playground paragraph with the six tabs and what each shows; add a screenshot-free Mermaid diagram of the tab flow.
- `CLAUDE.md`: update the playground bullet — tabs/ layout, lazy tabs, `animated_chart` play-once rule, unique chart keys, Plotly in `ui`, coverage omits.
- `docs/architecture.md`: Plotly row in the tooling table (`ui` group, in use).
- Learning notes: a "The visual playground" section mapping each tab to the concepts it shows.

- [ ] **Step 4: Hand back**

Report to the user with screenshots of every tab, what was verified, and ask whether to commit and push.
