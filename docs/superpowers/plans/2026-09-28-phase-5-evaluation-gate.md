# Phase 5 Evaluation Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Block any pull request that makes retrieval worse, or that changes what the generation scores depend on without a fresh, locally judged baseline. Show the gate working in a new ⑥ Gate playground tab.

**Architecture:** A pure core in `evaluation/` has four parts: a fingerprint of everything the scores depend on, committed baselines, the gate rules with their Markdown report, and a runner that rebuilds the index and scores retrieval. A thin CLI (`scripts/run_gate.py`) runs it locally and in a new GitHub Actions workflow, which also comments on the pull request. The playground reuses the same core for a what-if gate in a separate Chroma collection.

**Tech Stack:** Python 3.12, uv, pydantic-settings, Chroma (Docker), sentence-transformers (CPU in CI), GitHub Actions, Streamlit and Plotly (`ui` group), Graphviz DOT.

**Spec:** `docs/superpowers/specs/2026-09-28-phase-5-evaluation-gate-design.md`

## Global Constraints

- No new dependencies. CI installs the existing extra: `uv sync --locked --extra local-embeddings`.
- Configuration comes only from `get_settings()`. The new setting is `gate_max_drop: Score = 0.02` (`RAG_GATE_MAX_DROP`); thresholds stay `min_recall_at_k 0.80`, `min_mrr 0.70`, `min_ndcg_at_k 0.70`, `min_faithfulness 0.85`, `min_answer_relevance 0.80`.
- Exit codes: 0 pass, 1 blocked or below threshold, 2 could not run (same as the existing scripts).
- Baselines live in the tracked `baselines/` folder: `baselines/retrieval.json` and `baselines/generation.json`. `reports/` stays git-ignored.
- The what-if gate uses the Chroma collection `gate_preview`, never `rag_documents` or `playground`.
- mypy is `strict`: every function fully annotated. Ruff line length is 100. Ruff flags any constant named `*TOKEN*` (S105), en dashes (RUF001) and asserts outside tests (S101).
- Streamlit rules: every `st.plotly_chart` needs a unique `key`; save a long result to `st.session_state` before any further `st.*` call; HTML-escape text from documents or the LLM before `unsafe_allow_html`.
- Every phase updates the docs and adds a playground view (user rule).
- Commits use `<type>: <description>` with no Co-Authored-By lines. Stage files by name, read `git diff --cached` before each commit, and never use `--no-verify`. **Commit only when the user says so.** The commit steps below are prepared commands, run when the user asks.

## Review Focus

1. **Hidden files in `data/raw/`** (a macOS `.DS_Store`): the corpus hash must ignore them, or a Mac-made baseline is always stale in CI. Tested in Task 1.
2. **Re-rank details while re-ranking is off**: changing `rerank_candidates` or `reranker_model` with `rerank=False` must not make the baseline stale. Tested in Task 1.
3. **A drop of exactly `max_drop`** (0.93 → 0.91, where float subtraction gives 0.020000000000000018) must pass. Tested in Task 3.
4. **A baseline with `null` or missing metrics** must fail its check as "never scored", not crash. Tested in Task 3.
5. **The pull-request comment must be found again on the next run**: the Markdown must start with the hidden marker. Tested in Task 3. Also, the what-if collection name must differ from `rag_documents` and `playground`. Tested in Task 8.

**Deviations from the spec, decided here:**
- `Baseline.fingerprint` is the raw `dict[str, Any]` read from JSON, not an `EvalFingerprint`. Section 3.1 requires that a missing or unknown field in an older baseline counts as a change, which a typed object cannot represent.
- `run_gate` takes a `GateLimits` (thresholds plus `max_drop`, built with `GateLimits.from_settings`) instead of the whole `Settings`, which keeps the rules easy to test.

---

### Task 1: Gate setting and the evaluation fingerprint

**Files:**
- Modify: `src/rag_eval_platform/config/settings.py` (after `min_answer_relevance`)
- Modify: `.env.example` (after `RAG_MIN_ANSWER_RELEVANCE` or the thresholds block)
- Create: `src/rag_eval_platform/evaluation/fingerprint.py`
- Test: `tests/unit/test_fingerprint.py`, `tests/unit/test_settings.py`

**Interfaces:**
- Produces: `EvalFingerprint` (frozen dataclass, fields listed below) with `.to_dict() -> dict[str, Any]`; `FieldChange(field: str, saved: Any, current: Any)` whose `str()` is `"field: saved → current"`; `MISSING = "(missing)"`; `file_sha256(path: Path) -> str`; `tree_sha256(root: Path) -> str`; `current_fingerprint(settings: Settings) -> EvalFingerprint`; `fingerprint_changes(saved: Mapping[str, Any], current: EvalFingerprint) -> tuple[FieldChange, ...]`; `Settings.gate_max_drop`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_fingerprint.py`:

```python
"""Tests for evaluation.fingerprint: what the scores depend on, and what changed."""

from pathlib import Path
from typing import Any

import pytest

from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.evaluation.citation_validity import CITATION_PROMPT_VERSION
from rag_eval_platform.evaluation.fingerprint import (
    MISSING,
    FieldChange,
    current_fingerprint,
    fingerprint_changes,
    tree_sha256,
)
from rag_eval_platform.generation.prompt_templates import PROMPT_VERSION


def make_settings(tmp_path: Path, **overrides: Any) -> Settings:
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    (raw / "a.md").write_text("alpha", encoding="utf-8")
    golden = tmp_path / "qa.json"
    golden.write_text("[]", encoding="utf-8")
    return get_settings().model_copy(
        update={"raw_data_dir": raw, "golden_dataset_path": golden, **overrides}
    )


def test_fingerprint_records_settings_versions_and_hashes(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)

    fp = current_fingerprint(settings)

    assert fp.chunk_size == settings.chunk_size
    assert fp.embedding_model == settings.embedding_model
    assert fp.prompt_version == PROMPT_VERSION
    assert fp.citation_prompt_version == CITATION_PROMPT_VERSION
    assert len(fp.golden_sha256) == len(fp.corpus_sha256) == 64


def test_corpus_hash_ignores_hidden_files_but_sees_edits_and_renames(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    raw = settings.raw_data_dir
    first = tree_sha256(raw)

    (raw / ".DS_Store").write_bytes(b"finder noise")
    assert tree_sha256(raw) == first

    (raw / "a.md").rename(raw / "b.md")
    renamed = tree_sha256(raw)
    assert renamed != first

    (raw / "b.md").write_text("changed", encoding="utf-8")
    assert tree_sha256(raw) != renamed


def test_corpus_hash_needs_the_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        tree_sha256(tmp_path / "nope")


def test_reranker_details_only_count_while_reranking(tmp_path: Path) -> None:
    off_20 = current_fingerprint(make_settings(tmp_path, rerank=False, rerank_candidates=20))
    off_30 = current_fingerprint(make_settings(tmp_path, rerank=False, rerank_candidates=30))
    on_20 = current_fingerprint(make_settings(tmp_path, rerank=True, rerank_candidates=20))
    on_30 = current_fingerprint(make_settings(tmp_path, rerank=True, rerank_candidates=30))

    assert off_20 == off_30
    assert on_20 != on_30


def test_changes_list_each_differing_field(tmp_path: Path) -> None:
    fp = current_fingerprint(make_settings(tmp_path))
    saved = {**fp.to_dict(), "chunk_size": fp.chunk_size + 1}

    changes = fingerprint_changes(saved, fp)

    assert changes == (FieldChange("chunk_size", fp.chunk_size + 1, fp.chunk_size),)
    assert str(changes[0]) == f"chunk_size: {fp.chunk_size + 1} → {fp.chunk_size}"
    assert fingerprint_changes(fp.to_dict(), fp) == ()


def test_missing_or_unknown_fields_count_as_changes(tmp_path: Path) -> None:
    fp = current_fingerprint(make_settings(tmp_path))
    saved = {k: v for k, v in fp.to_dict().items() if k != "top_k"} | {"old_field": 1}

    changes = {c.field: c for c in fingerprint_changes(saved, fp)}

    assert set(changes) == {"top_k", "old_field"}
    assert changes["top_k"].saved == MISSING
    assert changes["old_field"].current == MISSING
```

In `tests/unit/test_settings.py`, in the test that asserts the defaults (the one containing `assert settings.llm_max_tokens == 1024`), add after that line:

```python
    assert settings.gate_max_drop == 0.02
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_fingerprint.py tests/unit/test_settings.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'rag_eval_platform.evaluation.fingerprint'`, and the settings test fails with `AttributeError: 'Settings' object has no attribute 'gate_max_drop'`.

- [ ] **Step 3: Implement**

In `settings.py`, after `min_answer_relevance: Score = 0.80`:

```python
    # Evaluation gate: the most a retrieval metric may drop below the committed baseline.
    gate_max_drop: Score = 0.02
```

In `.env.example`, next to the threshold lines (keep the value commented):

```
# RAG_GATE_MAX_DROP=0.02             # gate: max drop of a retrieval metric below baselines/retrieval.json
```

`src/rag_eval_platform/evaluation/fingerprint.py`:

```python
"""What the evaluation scores depend on, so a saved baseline can be checked for freshness.

A generation baseline is judged locally (CI has no GPU) and committed. The gate compares
its fingerprint with the current one: any difference means the scores may no longer hold,
and the baseline must be re-made.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.citation_validity import CITATION_PROMPT_VERSION
from rag_eval_platform.generation.prompt_templates import PROMPT_VERSION

MISSING = "(missing)"


@dataclass(frozen=True)
class EvalFingerprint:
    generator_model: str
    llm_temperature: float
    llm_reasoning_effort: str
    prompt_version: str
    judge_model: str
    citation_prompt_version: str
    embedding_model: str
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank: bool
    rerank_candidates: int  # 0 when not re-ranking (then it cannot affect the scores)
    reranker_model: str  # "" when not re-ranking
    golden_sha256: str
    corpus_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FieldChange:
    field: str
    saved: Any
    current: Any

    def __str__(self) -> str:
        return f"{self.field}: {self.saved} → {self.current}"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(root: Path) -> str:
    """One hash over every file's relative path and bytes; hidden files are ignored."""
    if not root.is_dir():
        raise FileNotFoundError(f"corpus folder not found: {root}")
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue  # .DS_Store and friends differ between machines
        digest.update(relative.as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def current_fingerprint(settings: Settings) -> EvalFingerprint:
    reranking = settings.rerank
    return EvalFingerprint(
        generator_model=settings.llm_model,
        llm_temperature=settings.llm_temperature,
        llm_reasoning_effort=settings.llm_reasoning_effort,
        prompt_version=PROMPT_VERSION,
        judge_model=settings.judge_model,
        citation_prompt_version=CITATION_PROMPT_VERSION,
        embedding_model=settings.embedding_model,
        chunk_strategy=settings.chunk_strategy,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        rerank=reranking,
        rerank_candidates=settings.rerank_candidates if reranking else 0,
        reranker_model=settings.reranker_model if reranking else "",
        golden_sha256=file_sha256(settings.golden_dataset_path),
        corpus_sha256=tree_sha256(settings.raw_data_dir),
    )


def fingerprint_changes(
    saved: Mapping[str, Any], current: EvalFingerprint
) -> tuple[FieldChange, ...]:
    """Every field that differs; a field missing on either side counts as a change."""
    now = current.to_dict()
    names = [*now, *(name for name in saved if name not in now)]
    return tuple(
        FieldChange(name, saved.get(name, MISSING), now.get(name, MISSING))
        for name in names
        if saved.get(name, MISSING) != now.get(name, MISSING)
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_fingerprint.py tests/unit/test_settings.py -q`
Expected: all pass.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/config/settings.py .env.example src/rag_eval_platform/evaluation/fingerprint.py tests/unit/test_fingerprint.py tests/unit/test_settings.py
git diff --cached
git commit -m "feat: fingerprint what evaluation scores depend on"
```

---

### Task 2: Committed baselines

**Files:**
- Create: `src/rag_eval_platform/evaluation/baselines.py`
- Test: `tests/unit/test_baselines.py`

**Interfaces:**
- Consumes: `EvalFingerprint`, `.to_dict()` (Task 1).
- Produces: `BASELINE_DIR = Path("baselines")`, `RETRIEVAL_BASELINE`, `GENERATION_BASELINE`; `class BaselineError(Exception)`; `Baseline(report: dict[str, Any], fingerprint: dict[str, Any])`; `load_baseline(path: Path) -> Baseline | None`; `save_baseline(path: Path, report: Mapping[str, Any], fingerprint: EvalFingerprint) -> None`; `check_adoptable(report: Mapping[str, Any], fingerprint: EvalFingerprint, golden_size: int) -> tuple[str, ...]`; test helper `BASE: EvalFingerprint` in `tests/unit/test_baselines.py`, reused by later tests.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_baselines.py`:

```python
"""Tests for evaluation.baselines: committed, reviewed evaluation results."""

from dataclasses import replace
from pathlib import Path

import pytest

from rag_eval_platform.evaluation.baselines import (
    Baseline,
    BaselineError,
    check_adoptable,
    load_baseline,
    save_baseline,
)
from rag_eval_platform.evaluation.fingerprint import EvalFingerprint

BASE = EvalFingerprint(
    generator_model="qwen3:8b", llm_temperature=0.0, llm_reasoning_effort="none",
    prompt_version="v1", judge_model="gemma3:12b", citation_prompt_version="v1",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2", chunk_strategy="recursive",
    chunk_size=800, chunk_overlap=100, top_k=5, rerank=False, rerank_candidates=0,
    reranker_model="", golden_sha256="g" * 64, corpus_sha256="c" * 64,
)  # fmt: skip

GENERATION_REPORT = {
    "generator_model": "qwen3:8b", "prompt_version": "v1", "judge_model": "gemma3:12b",
    "citation_prompt_version": "v1",
    "overall": {"faithfulness": 0.93, "answer_relevance": 0.81},
    "examples": [{"example_id": f"q{i}"} for i in range(30)],
}  # fmt: skip


def test_saved_baseline_loads_back(tmp_path: Path) -> None:
    path = tmp_path / "baselines" / "generation.json"

    save_baseline(path, GENERATION_REPORT, BASE)

    assert load_baseline(path) == Baseline(GENERATION_REPORT, BASE.to_dict())
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_missing_baseline_is_none(tmp_path: Path) -> None:
    assert load_baseline(tmp_path / "nope.json") is None


@pytest.mark.parametrize(
    "content", ["{not json", '{"report": {}}', '{"report": [], "fingerprint": {}}']
)
def test_malformed_baseline_names_the_file(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(BaselineError, match="bad.json"):
        load_baseline(path)


def test_a_full_report_made_with_the_current_versions_can_be_adopted() -> None:
    assert check_adoptable(GENERATION_REPORT, BASE, golden_size=30) == ()


def test_adopting_refuses_other_versions_or_a_partial_run() -> None:
    problems = check_adoptable(GENERATION_REPORT, replace(BASE, judge_model="other"), 31)

    assert any("judge_model" in p for p in problems)
    assert any("scored 30 of 31" in p for p in problems)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_baselines.py -q`
Expected: collection error `No module named 'rag_eval_platform.evaluation.baselines'`.

- [ ] **Step 3: Implement** `src/rag_eval_platform/evaluation/baselines.py`:

```python
"""Committed evaluation baselines: the reviewed scores the gate compares against.

Unlike ``reports/`` (git-ignored scratch output), ``baselines/`` is tracked: updating a
baseline shows up in the pull request's diff, like changing a threshold.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_eval_platform.evaluation.fingerprint import EvalFingerprint

BASELINE_DIR = Path("baselines")
RETRIEVAL_BASELINE = BASELINE_DIR / "retrieval.json"
GENERATION_BASELINE = BASELINE_DIR / "generation.json"

# Report fields that must match the current settings before a report is adopted.
_ADOPT_FIELDS = ("generator_model", "prompt_version", "judge_model", "citation_prompt_version")


class BaselineError(Exception):
    """A baseline file exists but cannot be read."""


@dataclass(frozen=True)
class Baseline:
    report: dict[str, Any]
    fingerprint: dict[str, Any]  # raw, so fields an older version lacked read as changed


def load_baseline(path: Path) -> Baseline | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        report, fingerprint = data["report"], data["fingerprint"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise BaselineError(f"{path} is not a valid baseline file: {exc}") from exc
    if not isinstance(report, dict) or not isinstance(fingerprint, dict):
        raise BaselineError(f"{path} is not a valid baseline file: report and fingerprint "
                            "must be objects")  # fmt: skip
    return Baseline(report, fingerprint)


def save_baseline(path: Path, report: Mapping[str, Any], fingerprint: EvalFingerprint) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"fingerprint": fingerprint.to_dict(), "report": dict(report)}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def check_adoptable(
    report: Mapping[str, Any], fingerprint: EvalFingerprint, golden_size: int
) -> tuple[str, ...]:
    """Why an existing generation report cannot become the baseline (empty means it can)."""
    problems = [
        f"{name} is {report.get(name)!r} in the report but {getattr(fingerprint, name)!r} now"
        for name in _ADOPT_FIELDS
        if report.get(name) != getattr(fingerprint, name)
    ]
    scored = len(report.get("examples", []))
    if scored != golden_size:
        problems.append(f"the report scored {scored} of {golden_size} golden questions")
    return tuple(problems)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_baselines.py -q`
Expected: all pass.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/evaluation/baselines.py tests/unit/test_baselines.py
git diff --cached
git commit -m "feat: committed evaluation baselines"
```

---

### Task 3: Gate rules and the Markdown report

**Files:**
- Create: `src/rag_eval_platform/evaluation/gate.py`
- Test: `tests/unit/test_gate.py`

**Interfaces:**
- Consumes: `Baseline` (Task 2), `EvalFingerprint`, `FieldChange`, `fingerprint_changes` (Task 1), `RetrievalReport` (`evaluation/evaluator.py`).
- Produces: `GATE_MARKER = "<!-- rag-eval-gate -->"`; `FIX_HINT: str`; `GateLimits(min_recall_at_k, min_mrr, min_ndcg_at_k, min_faithfulness, min_answer_relevance, max_drop)` with `from_settings(settings)`; `Check(group, name, value, threshold, baseline, status, reason)`; `GateResult(checks, changes, passed)`; `run_gate(retrieval: RetrievalReport, retrieval_baseline: Baseline | None, generation_baseline: Baseline | None, current: EvalFingerprint, limits: GateLimits) -> GateResult`; `gate_markdown(result: GateResult, retrieval: RetrievalReport, retrieval_baseline: Baseline | None) -> str`; `gate_to_dict(result) -> dict[str, Any]`; `gate_from_dict(data: Mapping[str, Any]) -> GateResult`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_gate.py`:

```python
"""Tests for evaluation.gate: the rules that pass or block a change, and its report."""

from dataclasses import replace

from tests.unit.test_baselines import BASE

from rag_eval_platform.evaluation.baselines import Baseline
from rag_eval_platform.evaluation.evaluator import RetrievalReport, RetrievalThresholds
from rag_eval_platform.evaluation.fingerprint import EvalFingerprint, FieldChange
from rag_eval_platform.evaluation.gate import (
    GATE_MARKER,
    GateLimits,
    gate_from_dict,
    gate_markdown,
    gate_to_dict,
    run_gate,
)
from rag_eval_platform.evaluation.metrics import RetrievalScores

LIMITS = GateLimits(0.80, 0.70, 0.70, 0.85, 0.80, max_drop=0.02)


def retrieval(recall: float = 0.93, mrr: float = 0.88, ndcg: float = 0.88) -> RetrievalReport:
    overall = RetrievalScores(precision=0.3, recall=recall, mrr=mrr, ndcg=ndcg)
    return RetrievalReport(
        k=5, thresholds=RetrievalThresholds(0.8, 0.7, 0.7), overall=overall,
        by_query_type={"short": overall}, examples=(), failures=(),
    )  # fmt: skip


def retrieval_baseline(recall: float = 0.93, mrr: float = 0.88, ndcg: float = 0.88) -> Baseline:
    scores = {"precision": 0.3, "recall": recall, "mrr": mrr, "ndcg": ndcg}
    report = {"overall": scores, "by_query_type": {"short": scores}, "examples": []}
    return Baseline(report, BASE.to_dict())


def generation_baseline(
    faith: float | None = 0.93, relevance: float | None = 0.81, fp: EvalFingerprint = BASE
) -> Baseline:
    report = {"overall": {"faithfulness": faith, "answer_relevance": relevance}}
    return Baseline(report, fp.to_dict())


def test_everything_in_order_passes() -> None:
    result = run_gate(retrieval(), retrieval_baseline(), generation_baseline(), BASE, LIMITS)

    assert result.passed
    assert [c.name for c in result.checks] == [
        "recall@k", "mrr", "ndcg@k", "freshness", "faithfulness", "answer_relevance",
    ]  # fmt: skip
    assert result.changes == ()


def test_a_metric_below_its_threshold_blocks() -> None:
    result = run_gate(retrieval(recall=0.79), retrieval_baseline(), generation_baseline(), BASE,
                      LIMITS)  # fmt: skip

    recall = result.checks[0]
    assert not result.passed
    assert (recall.status, recall.reason) == ("fail", "0.790 < 0.80")


def test_a_drop_beyond_max_drop_blocks_even_above_the_threshold() -> None:
    result = run_gate(retrieval(mrr=0.84), retrieval_baseline(mrr=0.88), generation_baseline(),
                      BASE, LIMITS)  # fmt: skip

    mrr = result.checks[1]
    assert mrr.status == "fail"
    assert mrr.reason == "dropped 0.040 from 0.880 (max 0.02)"


def test_a_drop_of_exactly_max_drop_passes() -> None:
    result = run_gate(retrieval(recall=0.91), retrieval_baseline(recall=0.93),
                      generation_baseline(), BASE, LIMITS)  # fmt: skip

    assert result.checks[0].status == "pass"


def test_without_a_retrieval_baseline_only_thresholds_apply() -> None:
    result = run_gate(retrieval(), None, generation_baseline(), BASE, LIMITS)

    assert result.passed
    assert "no baseline to compare" in result.checks[0].reason


def test_a_missing_generation_baseline_blocks() -> None:
    result = run_gate(retrieval(), retrieval_baseline(), None, BASE, LIMITS)

    by_name = {c.name: c for c in result.checks}
    assert not result.passed
    assert "no generation baseline" in by_name["freshness"].reason
    assert by_name["faithfulness"].reason == "never scored"


def test_a_stale_generation_baseline_blocks_and_lists_what_changed() -> None:
    stale = generation_baseline(fp=replace(BASE, chunk_size=400))

    result = run_gate(retrieval(), retrieval_baseline(), stale, BASE, LIMITS)

    freshness = result.checks[3]
    assert result.changes == (FieldChange("chunk_size", 400, 800),)
    assert freshness.status == "fail"
    assert "chunk_size: 400 → 800" in freshness.reason
    assert "--save-baseline" in freshness.reason


def test_null_or_missing_generation_metrics_fail_as_never_scored() -> None:
    nulls = run_gate(retrieval(), None, generation_baseline(faith=None), BASE, LIMITS)
    empty = run_gate(retrieval(), None, Baseline({}, BASE.to_dict()), BASE, LIMITS)

    assert nulls.checks[4].reason == "never scored"
    assert [c.reason for c in empty.checks[4:]] == ["never scored", "never scored"]


def test_low_generation_scores_block() -> None:
    result = run_gate(retrieval(), None, generation_baseline(faith=0.80), BASE, LIMITS)

    assert (result.checks[4].status, result.checks[4].reason) == ("fail", "0.800 < 0.85")


def test_markdown_starts_with_the_marker_and_shows_before_and_after() -> None:
    result = run_gate(retrieval(recall=0.79), retrieval_baseline(), None, BASE, LIMITS)

    text = gate_markdown(result, retrieval(recall=0.79), retrieval_baseline())

    assert text.startswith(GATE_MARKER)
    assert "⛔ Evaluation gate blocked this change" in text
    assert "| recall@k | 0.930 | 0.790 | -0.140 | 0.80 | ⛔ 0.790 < 0.80 |" in text
    assert "| short | 0.930 → 0.790 | 0.880 → 0.880 | 0.880 → 0.880 |" in text


def test_markdown_lists_stale_changes() -> None:
    stale = generation_baseline(fp=replace(BASE, top_k=3))
    result = run_gate(retrieval(), None, stale, BASE, LIMITS)

    text = gate_markdown(result, retrieval(), None)

    assert "- `top_k: 3 → 5`" in text
    assert "| short | - → 0.930 | - → 0.880 | - → 0.880 |" in text  # no retrieval baseline


def test_result_round_trips_through_json_form() -> None:
    result = run_gate(retrieval(), None, generation_baseline(fp=replace(BASE, top_k=3)), BASE,
                      LIMITS)  # fmt: skip

    assert gate_from_dict(gate_to_dict(result)) == result
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_gate.py -q`
Expected: collection error `No module named 'rag_eval_platform.evaluation.gate'`.

- [ ] **Step 3: Implement** `src/rag_eval_platform/evaluation/gate.py`:

```python
"""The evaluation gate: decide whether a change may merge.

Retrieval is scored live on every pull request. Generation needs a local GPU, so its scores
come from the committed baseline, which must be fresh: made with the current settings,
golden set and corpus (see fingerprint.py).
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal, Self

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.baselines import Baseline
from rag_eval_platform.evaluation.evaluator import RetrievalReport
from rag_eval_platform.evaluation.fingerprint import (
    EvalFingerprint,
    FieldChange,
    fingerprint_changes,
)

GATE_MARKER = "<!-- rag-eval-gate -->"  # lets CI find and update its own PR comment
FIX_HINT = (
    "Fix: run `uv run python scripts/run_generation_evaluation.py --save-baseline` "
    "and commit `baselines/generation.json`."
)
_EPSILON = 1e-9  # 0.93 - 0.91 is 0.020000000000000018 in floating point

CheckStatus = Literal["pass", "fail"]


@dataclass(frozen=True)
class GateLimits:
    min_recall_at_k: float
    min_mrr: float
    min_ndcg_at_k: float
    min_faithfulness: float
    min_answer_relevance: float
    max_drop: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.min_recall_at_k, settings.min_mrr, settings.min_ndcg_at_k,
                   settings.min_faithfulness, settings.min_answer_relevance,
                   settings.gate_max_drop)  # fmt: skip


@dataclass(frozen=True)
class Check:
    group: Literal["retrieval", "generation"]
    name: str
    value: float | None
    threshold: float | None
    baseline: float | None
    status: CheckStatus
    reason: str


@dataclass(frozen=True)
class GateResult:
    checks: tuple[Check, ...]
    changes: tuple[FieldChange, ...]
    passed: bool


def run_gate(
    retrieval: RetrievalReport,
    retrieval_baseline: Baseline | None,
    generation_baseline: Baseline | None,
    current: EvalFingerprint,
    limits: GateLimits,
) -> GateResult:
    saved = _overall(retrieval_baseline)
    checks = [
        _retrieval_check(
            name,
            getattr(retrieval.overall, field),
            getattr(limits, limit),
            _number(saved.get(field)) if retrieval_baseline else None,
            limits.max_drop,
        )  # fmt: skip
        for name, field, limit in (
            ("recall@k", "recall", "min_recall_at_k"),
            ("mrr", "mrr", "min_mrr"),
            ("ndcg@k", "ndcg", "min_ndcg_at_k"),
        )
    ]
    changes: tuple[FieldChange, ...] = ()
    if generation_baseline is None:
        checks.append(_freshness("fail", f"no generation baseline. {FIX_HINT}"))
    else:
        changes = fingerprint_changes(generation_baseline.fingerprint, current)
        if changes:
            listed = "; ".join(str(change) for change in changes)
            checks.append(_freshness("fail", f"stale ({listed}). {FIX_HINT}"))
        else:
            checks.append(_freshness("pass", "made with the current settings"))
    scores = _overall(generation_baseline)
    for name, limit in (("faithfulness", limits.min_faithfulness),
                        ("answer_relevance", limits.min_answer_relevance)):  # fmt: skip
        checks.append(_generation_check(name, _number(scores.get(name)), limit))
    return GateResult(tuple(checks), changes, all(c.status == "pass" for c in checks))


def gate_markdown(
    result: GateResult, retrieval: RetrievalReport, retrieval_baseline: Baseline | None
) -> str:
    headline = ("## ✅ Evaluation gate passed" if result.passed
                else "## ⛔ Evaluation gate blocked this change")  # fmt: skip
    lines = [GATE_MARKER, headline, "",
             "| Check | Baseline | Now | Change | Minimum | Result |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    for c in result.checks:
        change = (f"{c.value - c.baseline:+.3f}"
                  if c.value is not None and c.baseline is not None else "-")  # fmt: skip
        mark = "✅" if c.status == "pass" else "⛔"
        lines.append(f"| {c.name} | {_fmt(c.baseline)} | {_fmt(c.value)} | {change} | "
                     f"{_fmt(c.threshold, 2)} | {mark} {c.reason} |")  # fmt: skip
    lines += ["", "Generation scores come from `baselines/generation.json`, judged locally "
              "(CI has no GPU)."]  # fmt: skip
    if result.changes:
        lines += ["", "### Stale generation baseline", ""]
        lines += [f"- `{change}`" for change in result.changes]
        lines += ["", FIX_HINT]
    lines += ["", "### Retrieval by query type", "",
              "| Query type | Recall before → now | MRR before → now | NDCG before → now |",
              "|---|---|---|---|"]  # fmt: skip
    before = retrieval_baseline.report.get("by_query_type", {}) if retrieval_baseline else {}
    for query_type, scores in sorted(retrieval.by_query_type.items()):
        old = before.get(query_type, {}) if isinstance(before, dict) else {}
        cells = [f"{_fmt(_number(old.get(m)))} → {getattr(scores, m):.3f}"
                 for m in ("recall", "mrr", "ndcg")]  # fmt: skip
        lines.append(f"| {query_type} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def gate_to_dict(result: GateResult) -> dict[str, Any]:
    return {
        "passed": result.passed,
        "checks": [asdict(c) for c in result.checks],
        "changes": [asdict(c) for c in result.changes],
    }


def gate_from_dict(data: Mapping[str, Any]) -> GateResult:
    return GateResult(
        tuple(Check(**c) for c in data["checks"]),
        tuple(FieldChange(**c) for c in data["changes"]),
        bool(data["passed"]),
    )


def _retrieval_check(
    name: str, value: float, threshold: float, baseline: float | None, max_drop: float
) -> Check:
    def check(status: CheckStatus, reason: str) -> Check:
        return Check("retrieval", name, value, threshold, baseline, status, reason)

    if value < threshold:
        return check("fail", f"{value:.3f} < {threshold:.2f}")
    if baseline is None:
        return check("pass", f"{value:.3f} >= {threshold:.2f} (no baseline to compare)")
    drop = baseline - value
    if drop > max_drop + _EPSILON:
        return check("fail", f"dropped {drop:.3f} from {baseline:.3f} (max {max_drop:.2f})")
    return check("pass", f"{value:.3f} >= {threshold:.2f}, within {max_drop:.2f} of baseline")


def _freshness(status: CheckStatus, reason: str) -> Check:
    return Check("generation", "freshness", None, None, None, status, reason)


def _generation_check(name: str, value: float | None, threshold: float) -> Check:
    if value is None:
        return Check("generation", name, None, threshold, None, "fail", "never scored")
    status: CheckStatus = "pass" if value >= threshold else "fail"
    comparison = ">=" if status == "pass" else "<"
    return Check("generation", name, value, threshold, None, status,
                 f"{value:.3f} {comparison} {threshold:.2f}")  # fmt: skip


def _overall(baseline: Baseline | None) -> Mapping[str, Any]:
    overall = baseline.report.get("overall", {}) if baseline else {}
    return overall if isinstance(overall, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_gate.py -q`
Expected: all pass. If `test_everything_in_order_passes` fails on the reason text of a passing check, fix the code, not the test's check names.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/evaluation/gate.py tests/unit/test_gate.py
git diff --cached
git commit -m "feat: evaluation gate rules and before/after report"
```

---

### Task 4: Gate runner, CLI and `scripts/run_gate.py`

**Files:**
- Create: `src/rag_eval_platform/evaluation/gate_runner.py`, `src/rag_eval_platform/evaluation/gate_cli.py`, `scripts/run_gate.py`
- Test: `tests/unit/test_gate_runner.py`, `tests/unit/test_gate_cli.py`, `tests/integration/test_gate_pipeline.py`

**Interfaces:**
- Consumes: Tasks 1–3; `load_documents`, `chunk_documents`, `seed`, `Retriever`, `evaluate_retrieval`, `RetrievalThresholds`, `report_to_dict`, `load_golden_dataset`, `create_embedder`, `create_vector_store`, `CrossEncoderReranker`.
- Produces: `rebuild_and_evaluate(examples: Sequence[GoldenExample], *, raw_dir: Path, strategy: ChunkStrategy, chunk_size: int, chunk_overlap: int, embedder: Embedder, store: VectorStore, top_k: int, reranker: Reranker | None, rerank_candidates: int, thresholds: RetrievalThresholds, on_progress: Callable[[str], None] | None = None) -> RetrievalReport`; `gate_cli.main(argv) -> int`; `DEFAULT_GATE_REPORT = Path("reports/gate_report.json")`, `DEFAULT_GATE_SUMMARY = Path("reports/gate_summary.md")`; test helpers `WordEmbedder`, `MemoryStore`, `write_corpus(root)`, `EXAMPLES` in `tests/unit/test_gate_runner.py`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_gate_runner.py`:

```python
"""Tests for evaluation.gate_runner: rebuild the index from the corpus and score it."""

from collections.abc import Sequence
from pathlib import Path

from rag_eval_platform.evaluation.evaluator import RetrievalThresholds
from rag_eval_platform.evaluation.gate_runner import rebuild_and_evaluate
from rag_eval_platform.evaluation.golden_dataset import GoldenExample
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult

VOCAB = ("alpha", "beta")
THRESHOLDS = RetrievalThresholds(0.8, 0.7, 0.7)
EXAMPLES = (
    GoldenExample(id="q1", question="alpha please", expected_answer="a",
                  relevant_doc_ids=("alpha.md",), query_type="short"),
    GoldenExample(id="q2", question="beta", expected_answer="b",
                  relevant_doc_ids=("beta.md",), query_type="short"),
)  # fmt: skip


class WordEmbedder:
    """Counts the vocabulary words: enough meaning for a two-document corpus."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        words = text.lower().split()
        return [words.count(word) + 0.01 for word in VOCAB]


class MemoryStore:
    def __init__(self) -> None:
        self.items: list[tuple[Chunk, list[float]]] = []

    def replace_all(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        self.items = [(c, list(e)) for c, e in zip(chunks, embeddings, strict=True)]

    def search(self, query_embedding: Sequence[float], k: int) -> list[SearchResult]:
        scored = [(sum(a * b for a, b in zip(query_embedding, e, strict=True)), c)
                  for c, e in self.items]  # fmt: skip
        scored.sort(key=lambda pair: -pair[0])
        return [SearchResult(chunk=c, score=s) for s, c in scored[:k]]

    def count(self) -> int:
        return len(self.items)


def write_corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "alpha.md").write_text("alpha alpha alpha", encoding="utf-8")
    (root / "beta.md").write_text("beta beta beta", encoding="utf-8")
    return root


def test_rebuilds_the_index_and_scores_every_question(tmp_path: Path) -> None:
    store, progress = MemoryStore(), []

    report = rebuild_and_evaluate(
        EXAMPLES, raw_dir=write_corpus(tmp_path / "raw"), strategy="recursive",
        chunk_size=200, chunk_overlap=0, embedder=WordEmbedder(), store=store, top_k=1,
        reranker=None, rerank_candidates=20, thresholds=THRESHOLDS, on_progress=progress.append,
    )  # fmt: skip

    assert store.count() == 2
    assert report.overall.recall == 1.0
    assert report.k == 1
    assert len(progress) == 2
```

`tests/unit/test_gate_cli.py`:

```python
"""Tests for evaluation.gate_cli (the logic behind scripts/run_gate.py)."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from tests.unit.test_gate_runner import MemoryStore, WordEmbedder, write_corpus

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation import gate_cli
from rag_eval_platform.evaluation.baselines import save_baseline
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
from rag_eval_platform.evaluation.gate import GATE_MARKER
from rag_eval_platform.retrieval.vector_store import VectorStoreError

GOLDEN = [
    {"id": "q1", "question": "alpha please", "expected_answer": "a",
     "relevant_doc_ids": ["alpha.md"], "query_type": "short"},
    {"id": "q2", "question": "beta", "expected_answer": "b",
     "relevant_doc_ids": ["beta.md"], "query_type": "short"},
]  # fmt: skip


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    write_corpus(tmp_path / "raw")
    (tmp_path / "qa.json").write_text(json.dumps(GOLDEN), encoding="utf-8")
    monkeypatch.setenv("RAG_RAW_DATA_DIR", str(tmp_path / "raw"))
    monkeypatch.setenv("RAG_GOLDEN_DATASET_PATH", str(tmp_path / "qa.json"))
    monkeypatch.setenv("RAG_TOP_K", "1")
    monkeypatch.setenv("RAG_RERANK", "false")
    get_settings.cache_clear()
    monkeypatch.setattr(gate_cli, "create_embedder", lambda settings: WordEmbedder())
    monkeypatch.setattr(gate_cli, "create_vector_store", lambda settings: MemoryStore())
    yield tmp_path
    get_settings.cache_clear()


def fresh_generation_baseline(root: Path) -> None:
    report = {"overall": {"faithfulness": 0.95, "answer_relevance": 0.9}}
    save_baseline(root / "baselines" / "generation.json", report,
                  current_fingerprint(get_settings()))  # fmt: skip


def run(root: Path, *extra: str) -> int:
    return gate_cli.main(["--baselines", str(root / "baselines"),
                          "--report", str(root / "gate.json"),
                          "--summary-out", str(root / "gate.md"), *extra])  # fmt: skip


def test_passing_gate_writes_report_and_summary_and_returns_0(workspace: Path) -> None:
    fresh_generation_baseline(workspace)

    assert run(workspace) == 0
    report = json.loads((workspace / "gate.json").read_text(encoding="utf-8"))
    assert report["gate"]["passed"] is True
    assert report["retrieval"]["overall"]["recall"] == 1.0
    assert (workspace / "gate.md").read_text(encoding="utf-8").startswith(GATE_MARKER)


def test_missing_generation_baseline_blocks_with_1_and_appends_the_summary(
    workspace: Path,
) -> None:
    step_summary = workspace / "step_summary.md"
    step_summary.write_text("earlier step\n", encoding="utf-8")

    assert run(workspace, "--summary", str(step_summary)) == 1
    text = step_summary.read_text(encoding="utf-8")
    assert text.startswith("earlier step\n")
    assert "no generation baseline" in text


def test_a_store_failure_returns_2(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(settings: object) -> MemoryStore:
        raise VectorStoreError("Chroma is down")

    monkeypatch.setattr(gate_cli, "create_vector_store", broken)

    assert run(workspace) == 2
    assert not (workspace / "gate.json").exists()


def test_a_malformed_baseline_returns_2(workspace: Path) -> None:
    (workspace / "baselines").mkdir()
    (workspace / "baselines" / "retrieval.json").write_text("{nope", encoding="utf-8")

    assert run(workspace) == 2
```

`tests/integration/test_gate_pipeline.py`:

```python
"""The gate's rebuild-and-score step against a real Chroma server (fake embedder).

Skipped when no server is reachable at RAG_CHROMA_HOST:RAG_CHROMA_PORT.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import chromadb
import pytest
from tests.unit.test_gate_runner import EXAMPLES, THRESHOLDS, WordEmbedder, write_corpus

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.gate_runner import rebuild_and_evaluate
from rag_eval_platform.retrieval.vector_store import ChromaVectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store() -> Iterator[ChromaVectorStore]:
    settings = Settings()
    try:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
    except Exception:
        pytest.skip(f"no Chroma server at {settings.chroma_host}:{settings.chroma_port}")
    name = f"test_gate_{uuid.uuid4().hex[:12]}"
    yield ChromaVectorStore(client, collection_name=name)
    client.delete_collection(name)


def test_rebuild_and_evaluate_scores_the_corpus_in_chroma(
    store: ChromaVectorStore, tmp_path: Path
) -> None:
    report = rebuild_and_evaluate(
        EXAMPLES, raw_dir=write_corpus(tmp_path / "raw"), strategy="recursive",
        chunk_size=200, chunk_overlap=0, embedder=WordEmbedder(), store=store, top_k=1,
        reranker=None, rerank_candidates=20, thresholds=THRESHOLDS,
    )  # fmt: skip

    assert store.count() == 2
    assert report.overall.recall == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_gate_runner.py tests/unit/test_gate_cli.py -q`
Expected: collection errors, `No module named 'rag_eval_platform.evaluation.gate_runner'` (and `gate_cli`).

- [ ] **Step 3: Implement**

`src/rag_eval_platform/evaluation/gate_runner.py`:

```python
"""Rebuild the index from the corpus with given settings, then score retrieval.

Shared by the CI gate (scripts/run_gate.py) and the playground's what-if gate, which
passes its own settings and a separate collection.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from rag_eval_platform.config.settings import ChunkStrategy
from rag_eval_platform.evaluation.evaluator import (
    RetrievalReport,
    RetrievalThresholds,
    evaluate_retrieval,
)
from rag_eval_platform.evaluation.golden_dataset import GoldenExample
from rag_eval_platform.ingestion.chunking import chunk_documents
from rag_eval_platform.ingestion.embedding import Embedder
from rag_eval_platform.ingestion.loaders import load_documents
from rag_eval_platform.retrieval.reranker import Reranker
from rag_eval_platform.retrieval.retriever import Retriever
from rag_eval_platform.retrieval.seed import seed
from rag_eval_platform.retrieval.vector_store import VectorStore


def rebuild_and_evaluate(
    examples: Sequence[GoldenExample],
    *,
    raw_dir: Path,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Embedder,
    store: VectorStore,
    top_k: int,
    reranker: Reranker | None,
    rerank_candidates: int,
    thresholds: RetrievalThresholds,
    on_progress: Callable[[str], None] | None = None,
) -> RetrievalReport:
    report = on_progress or (lambda message: None)
    documents = load_documents(raw_dir)
    chunks = chunk_documents(
        documents, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    report(f"Indexed {len(documents)} documents as {len(chunks)} chunks")
    seed(chunks, embedder, store)
    retriever = Retriever(embedder, store, top_k, reranker, rerank_candidates)
    result = evaluate_retrieval(examples, retriever, k=top_k, thresholds=thresholds)
    report(f"Scored {len(examples)} golden questions")
    return result
```

`src/rag_eval_platform/evaluation/gate_cli.py`:

```python
"""Evaluation gate command: rebuild the index, score retrieval, check the baselines.

Run with ``uv run python scripts/run_gate.py`` (Chroma running, ``--extra
local-embeddings``). Writes ``reports/gate_report.json`` and ``reports/gate_summary.md``,
prints the summary, and exits with 0 = pass, 1 = blocked, 2 = could not run.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.evaluation.baselines import BASELINE_DIR, BaselineError, load_baseline
from rag_eval_platform.evaluation.evaluator import RetrievalThresholds, report_to_dict
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
from rag_eval_platform.evaluation.gate import GateLimits, gate_markdown, gate_to_dict, run_gate
from rag_eval_platform.evaluation.gate_runner import rebuild_and_evaluate
from rag_eval_platform.evaluation.golden_dataset import GoldenDatasetError, load_golden_dataset
from rag_eval_platform.ingestion.embedding import EmbeddingError, create_embedder
from rag_eval_platform.ingestion.loaders import DocumentLoadError
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker, Reranker
from rag_eval_platform.retrieval.vector_store import VectorStoreError, create_vector_store

logger = logging.getLogger(__name__)

DEFAULT_GATE_REPORT = Path("reports/gate_report.json")
DEFAULT_GATE_SUMMARY = Path("reports/gate_summary.md")
EXIT_PASSED, EXIT_BLOCKED, EXIT_ERROR = 0, 1, 2
_KNOWN_ERRORS = (GoldenDatasetError, DocumentLoadError, VectorStoreError, EmbeddingError,
                 OptionalDependencyError, BaselineError, OSError, ValueError)  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Run the evaluation gate.")
    parser.add_argument("--baselines", type=Path, default=BASELINE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_GATE_SUMMARY)
    parser.add_argument("--summary", type=Path, help="also append the Markdown here "
                        "(CI passes $GITHUB_STEP_SUMMARY)")  # fmt: skip
    args = parser.parse_args(argv)

    try:
        retrieval_baseline = load_baseline(args.baselines / "retrieval.json")
        generation_baseline = load_baseline(args.baselines / "generation.json")
        current = current_fingerprint(settings)
        retrieval = rebuild_and_evaluate(
            load_golden_dataset(settings.golden_dataset_path),
            raw_dir=settings.raw_data_dir, strategy=settings.chunk_strategy,
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
            embedder=create_embedder(settings), store=create_vector_store(settings),
            top_k=settings.top_k, reranker=_reranker(settings),
            rerank_candidates=settings.rerank_candidates,
            thresholds=RetrievalThresholds.from_settings(settings),
            on_progress=lambda message: logger.info(message),
        )  # fmt: skip
    except _KNOWN_ERRORS as exc:
        logger.error("The evaluation gate could not run: %s", exc)
        return EXIT_ERROR

    result = run_gate(retrieval, retrieval_baseline, generation_baseline, current,
                      GateLimits.from_settings(settings))  # fmt: skip
    markdown = gate_markdown(result, retrieval, retrieval_baseline)
    report = {"gate": gate_to_dict(result), "retrieval": report_to_dict(retrieval)}
    for path, text in ((args.report, json.dumps(report, indent=2)),
                       (args.summary_out, markdown)):  # fmt: skip
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if args.summary is not None:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(markdown)
    print(markdown)
    logger.info("Evaluation gate finished", extra={"passed": result.passed,
                "report": str(args.report)})  # fmt: skip
    return EXIT_PASSED if result.passed else EXIT_BLOCKED


def _reranker(settings: Settings) -> Reranker | None:
    return (CrossEncoderReranker.from_pretrained(settings.reranker_model)
            if settings.rerank else None)  # fmt: skip
```

`scripts/run_gate.py`:

```python
"""Run the evaluation gate: rebuild the index, score retrieval, check the baselines.

Usage: uv run python scripts/run_gate.py [--summary $GITHUB_STEP_SUMMARY]
Exit 0 = pass, 1 = blocked, 2 = could not run.
"""

from rag_eval_platform.evaluation.gate_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_gate_runner.py tests/unit/test_gate_cli.py -q && uv run pytest tests/integration/test_gate_pipeline.py -m integration -q`
Expected: unit tests pass. The integration test passes with Chroma running (`docker compose -f docker/docker-compose.yml up -d`) or is skipped without it.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/evaluation/gate_runner.py src/rag_eval_platform/evaluation/gate_cli.py scripts/run_gate.py tests/unit/test_gate_runner.py tests/unit/test_gate_cli.py tests/integration/test_gate_pipeline.py
git diff --cached
git commit -m "feat: run_gate command rebuilds the index, scores retrieval and checks baselines"
```

---

### Task 5: Saving and adopting baselines

**Files:**
- Modify: `src/rag_eval_platform/evaluation/cli.py` (argument parser and after the report is written)
- Modify: `src/rag_eval_platform/evaluation/generation_cli.py` (argument parser and after the report is written)
- Create: `src/rag_eval_platform/evaluation/baseline_cli.py`, `scripts/save_baseline.py`
- Test: `tests/unit/test_evaluation_cli.py`, `tests/unit/test_generation_cli.py`, `tests/unit/test_baseline_cli.py`

**Interfaces:**
- Consumes: `save_baseline`, `check_adoptable`, `RETRIEVAL_BASELINE`, `GENERATION_BASELINE` (Task 2); `current_fingerprint` (Task 1).
- Produces: a `--save-baseline` flag on both evaluation CLIs; `baseline_cli.main(argv) -> int` (0 saved, 1 refused, 2 unreadable).

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_evaluation_cli.py`:

```python
def test_save_baseline_writes_the_report_and_fingerprint(
    golden: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_retriever(monkeypatch, "forces.md")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "forces.md").write_text("F = ma", encoding="utf-8")

    exit_code = cli.main(["--golden", str(golden), "--report", str(tmp_path / "r.json"),
                          "--save-baseline"])  # fmt: skip

    assert exit_code == 0
    saved = json.loads((tmp_path / "baselines" / "retrieval.json").read_text(encoding="utf-8"))
    assert saved["report"]["overall"]["recall"] == 1.0
    assert len(saved["fingerprint"]["golden_sha256"]) == 64
```

Add to `tests/unit/test_generation_cli.py`:

```python
def test_save_baseline_writes_generation_json(fakes: Fakes, tmp_path: Path) -> None:
    (tmp_path / "data" / "raw").mkdir(parents=True)

    assert run(tmp_path, "--save-baseline") == 0

    saved = json.loads((tmp_path / "baselines" / "generation.json").read_text(encoding="utf-8"))
    assert saved["report"]["judge_model"] == "gemma3:12b"
    assert saved["fingerprint"]["judge_model"] == "gemma3:12b"


def test_save_baseline_refuses_a_limited_run(fakes: Fakes, tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        run(tmp_path, "--limit", "1", "--save-baseline")

    assert exit_info.value.code == 2
    assert not (tmp_path / "baselines").exists()
```

`tests/unit/test_baseline_cli.py`:

```python
"""Tests for evaluation.baseline_cli (scripts/save_baseline.py)."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation import baseline_cli

GOLDEN = [{"id": "q1", "question": "What is force?", "expected_answer": "F = ma",
           "relevant_doc_ids": ["forces.md"], "query_type": "short"}]  # fmt: skip


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "forces.md").write_text("F = ma", encoding="utf-8")
    (tmp_path / "qa.json").write_text(json.dumps(GOLDEN), encoding="utf-8")
    monkeypatch.setenv("RAG_GOLDEN_DATASET_PATH", str(tmp_path / "qa.json"))
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def write_report(root: Path, **overrides: object) -> Path:
    settings = get_settings()
    report = {"generator_model": settings.llm_model, "prompt_version": "v1",
              "judge_model": settings.judge_model, "citation_prompt_version": "v1",
              "overall": {"faithfulness": 0.9}, "examples": [{"example_id": "q1"}],
              **overrides}  # fmt: skip
    path = root / "generation_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_adopts_a_matching_full_report(workspace: Path) -> None:
    assert baseline_cli.main(["generation", str(write_report(workspace))]) == 0

    saved = json.loads((workspace / "baselines" / "generation.json").read_text(encoding="utf-8"))
    assert saved["report"]["overall"]["faithfulness"] == 0.9


def test_refuses_a_report_from_another_judge(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_report(workspace, judge_model="other-judge")

    assert baseline_cli.main(["generation", str(path)]) == 1
    assert "judge_model" in capsys.readouterr().out
    assert not (workspace / "baselines").exists()


def test_unreadable_report_returns_2(workspace: Path) -> None:
    (workspace / "bad.json").write_text("{nope", encoding="utf-8")

    assert baseline_cli.main(["generation", str(workspace / "bad.json")]) == 2
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_evaluation_cli.py tests/unit/test_generation_cli.py tests/unit/test_baseline_cli.py -q`
Expected: the two CLI files fail with `error: unrecognized arguments: --save-baseline` (SystemExit 2), and `test_baseline_cli.py` fails to collect with `cannot import name 'baseline_cli'`.

- [ ] **Step 3: Implement**

In `evaluation/cli.py`, add the imports:

```python
from rag_eval_platform.evaluation.baselines import RETRIEVAL_BASELINE, save_baseline
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
```

After `parser.add_argument("--report", ...)`:

```python
    parser.add_argument("--save-baseline", action="store_true",
                        help=f"also save the result as {RETRIEVAL_BASELINE} (commit it)")  # fmt: skip
```

Directly after `args.report.write_text(...)`:

```python
if args.save_baseline:
    fingerprint = current_fingerprint(
        settings.model_copy(update={"golden_dataset_path": args.golden})
    )
    save_baseline(RETRIEVAL_BASELINE, report_to_dict(report), fingerprint)
    print(f"Saved the retrieval baseline to {RETRIEVAL_BASELINE}; commit it.")
```

(Wrap the `fingerprint = ...` line to stay under 100 characters.)

In `evaluation/generation_cli.py`, add the imports:

```python
from rag_eval_platform.evaluation.baselines import GENERATION_BASELINE, save_baseline
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
```

After the `--full` argument:

```python
    parser.add_argument("--save-baseline", action="store_true",
                        help=f"also save the result as {GENERATION_BASELINE} (commit it)")  # fmt: skip
```

After `args = parser.parse_args(argv)`:

```python
    if args.save_baseline and args.limit is not None:
        parser.error("--save-baseline needs every golden question; drop --limit")
```

Directly after `args.report.write_text(report_json, encoding="utf-8")`:

```python
if args.save_baseline:
    fingerprint = current_fingerprint(
        settings.model_copy(update={"golden_dataset_path": args.golden})
    )
    save_baseline(GENERATION_BASELINE, generation_report_to_dict(report), fingerprint)
    print(f"Saved the generation baseline to {GENERATION_BASELINE}; commit it.")
```

(Wrap the long line as above.)

`src/rag_eval_platform/evaluation/baseline_cli.py`:

```python
"""Adopt an existing report as a committed baseline, after checking it still applies.

Usage: uv run python scripts/save_baseline.py generation reports/generation_report.json

The report must come from the current generator, prompt, judge and citation prompt and
cover every golden question. The current fingerprint is stamped on it: this trusts that
the report was made with the current retrieval settings, golden set and corpus.
Exit 0 = saved, 1 = refused, 2 = the report could not be read.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation.baselines import (
    GENERATION_BASELINE,
    RETRIEVAL_BASELINE,
    check_adoptable,
    save_baseline,
)
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
from rag_eval_platform.evaluation.golden_dataset import GoldenDatasetError, load_golden_dataset

EXIT_SAVED, EXIT_REFUSED, EXIT_ERROR = 0, 1, 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adopt a report as a committed baseline.")
    parser.add_argument("kind", choices=["generation", "retrieval"])
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        fingerprint = current_fingerprint(settings)
        golden_size = len(load_golden_dataset(settings.golden_dataset_path))
    except (OSError, json.JSONDecodeError, GoldenDatasetError) as exc:
        print(f"Could not read the inputs: {exc}")
        return EXIT_ERROR
    if args.kind == "generation":
        problems = check_adoptable(report, fingerprint, golden_size)
        if problems:
            print("Not adopted:\n" + "\n".join(f"  - {p}" for p in problems))
            return EXIT_REFUSED
    target = GENERATION_BASELINE if args.kind == "generation" else RETRIEVAL_BASELINE
    save_baseline(target, report, fingerprint)
    print(f"Saved {target}; commit it.")
    return EXIT_SAVED
```

`scripts/save_baseline.py`:

```python
"""Adopt an existing evaluation report as a committed baseline.

Usage: uv run python scripts/save_baseline.py generation reports/generation_report.json
"""

from rag_eval_platform.evaluation.baseline_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_evaluation_cli.py tests/unit/test_generation_cli.py tests/unit/test_baseline_cli.py -q`
Expected: all pass.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/evaluation/cli.py src/rag_eval_platform/evaluation/generation_cli.py src/rag_eval_platform/evaluation/baseline_cli.py scripts/save_baseline.py tests/unit/test_evaluation_cli.py tests/unit/test_generation_cli.py tests/unit/test_baseline_cli.py
git diff --cached
git commit -m "feat: save and adopt evaluation baselines"
```

---

### Task 6: The first committed baselines

**Files:**
- Create: `baselines/retrieval.json`, `baselines/generation.json`

**Interfaces:**
- Consumes: the scripts from Tasks 4 and 5.
- Produces: the two tracked baselines that CI compares against.

- [ ] **Step 1: Make sure the local settings are the defaults.** The baselines must match what CI runs with, and CI uses only the defaults. Check `.env` for any `RAG_CHUNK_*`, `RAG_TOP_K`, `RAG_RERANK*`, `RAG_EMBEDDING_MODEL`, `RAG_LLM_*` or `RAG_JUDGE_*` override:

Run: `grep -E "^RAG_(CHUNK|TOP_K|RERANK|EMBEDDING|LLM|JUDGE)" .env || echo "no overrides"`
Expected: `no overrides`. If there are overrides, ask the user before commenting them out for this step.

- [ ] **Step 2: Build the retrieval baseline** (Chroma running):

Run: `uv run python scripts/run_ingestion.py && uv run python scripts/seed_vector_store.py && uv run python scripts/run_evaluation.py --save-baseline`
Expected: exit 0, recall@k 0.933, MRR 0.878, NDCG@k 0.876 (the documented 2026-09-26 baseline), and `baselines/retrieval.json` written.

- [ ] **Step 3: Adopt the generation baseline** from the existing full run (2026-09-27):

Run: `uv run python scripts/save_baseline.py generation reports/generation_report.json`
Expected: `Saved baselines/generation.json; commit it.` If it refuses, show the listed problems to the user; the fallback is a full `run_generation_evaluation.py --save-baseline` (about an hour).

- [ ] **Step 4: Run the gate locally**

Run: `uv run python scripts/run_gate.py`
Expected: exit 0 and `## ✅ Evaluation gate passed`, with all six checks passing.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add baselines/retrieval.json baselines/generation.json
git diff --cached --stat
git commit -m "chore: first committed retrieval and generation baselines"
```

---

### Task 7: The CI workflow

**Files:**
- Modify: `.github/workflows/evaluation_gate.yml` (replace the placeholder)

**Interfaces:**
- Consumes: `scripts/run_gate.py` (Task 4), `GATE_MARKER` (Task 3; the comment step matches on it).

- [ ] **Step 1: Write the workflow** (the whole file):

```yaml
# Evaluation gate: every pull request is scored against the golden dataset.
# Retrieval is evaluated live (CPU embeddings); generation scores come from the committed
# baselines/generation.json, which must be fresh. See docs/learning/phase-5.md.
# To block merges, mark "Evaluate against golden dataset" as a required status check in
# the repository's branch protection settings.
name: Evaluation gate

on:
  pull_request:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pull-requests: write

concurrency:
  group: gate-${{ github.ref }}
  cancel-in-progress: true

jobs:
  evaluate:
    name: Evaluate against golden dataset
    runs-on: ubuntu-latest
    services:
      chroma:
        # Same image and version as docker/docker-compose.yml
        image: chromadb/chroma:1.5.9
        ports:
          - 8001:8000
    env:
      RAG_CHROMA_HOST: localhost
      RAG_CHROMA_PORT: "8001"
      HF_HUB_DISABLE_TELEMETRY: "1"
    steps:
      - uses: actions/checkout@v7

      - uses: astral-sh/setup-uv@v7
        with:
          enable-cache: true

      - name: Install dependencies (CPU embeddings)
        run: uv sync --locked --extra local-embeddings

      - name: Cache embedding models
        uses: actions/cache@v6
        with:
          path: ~/.cache/huggingface
          key: hf-${{ runner.os }}-${{ hashFiles('src/rag_eval_platform/config/settings.py') }}

      - name: Wait for Chroma
        run: |
          for i in $(seq 1 30); do
            curl -sf http://localhost:8001/api/v2/heartbeat && exit 0
            sleep 2
          done
          echo "Chroma did not become ready" && exit 1

      - name: Run the gate
        run: uv run python scripts/run_gate.py --summary "$GITHUB_STEP_SUMMARY"

      - name: Upload reports
        if: always()
        uses: actions/upload-artifact@v7
        with:
          name: evaluation-reports
          path: reports/
          if-no-files-found: ignore

      - name: Comment on the pull request
        # Forks get a read-only token; their job summary still has the table.
        if: >-
          always() && github.event_name == 'pull_request' &&
          github.event.pull_request.head.repo.full_name == github.repository &&
          hashFiles('reports/gate_summary.md') != ''
        env:
          GH_TOKEN: ${{ github.token }}
          PR: ${{ github.event.pull_request.number }}
        run: |
          id=$(gh api "repos/$GITHUB_REPOSITORY/issues/$PR/comments" --paginate \
            --jq '.[] | select(.body | startswith("<!-- rag-eval-gate -->")) | .id' | head -n 1)
          if [ -n "$id" ]; then
            gh api -X PATCH "repos/$GITHUB_REPOSITORY/issues/comments/$id" \
              -F body=@reports/gate_summary.md
          else
            gh pr comment "$PR" --body-file reports/gate_summary.md
          fi
```

- [ ] **Step 2: Validate the YAML syntax locally**

Run: `uv run python -c "import yaml, sys; yaml.safe_load(open('.github/workflows/evaluation_gate.yml')); print('ok')"`
Expected: `ok`. If PyYAML is not installed, run `ruby -ryaml -e "YAML.load_file('.github/workflows/evaluation_gate.yml'); puts 'ok'"` instead (macOS ships Ruby). Do not add a dependency for this.

- [ ] **Step 3: Commit (when the user asks).** The real test is the workflow's own run on the pull request.

```bash
git add .github/workflows/evaluation_gate.yml
git diff --cached
git commit -m "ci: evaluation gate scores every pull request and comments the result"
```

---

### Task 8: Playground gate logic (pure) and the comparison chart

**Files:**
- Create: `src/rag_eval_platform/playground/gate_view.py`
- Modify: `src/rag_eval_platform/playground/charts.py` (add `type_compare_chart` at the end)
- Test: `tests/unit/test_gate_view.py`, `tests/unit/test_charts.py`

**Interfaces:**
- Consumes: `GateResult`, `Check` (Task 3); `Baseline` (Task 2); `current_fingerprint` (Task 1); `RetrievalReport`; `PLAYGROUND_COLLECTION` (`playground/core.py`).
- Produces: `GATE_PREVIEW_COLLECTION = "gate_preview"`; `GATE_STEPS: tuple[str, ...]`; `gate_flow_dot(result: GateResult | None) -> str`; `Flip(example_id, question, query_type, relevant_doc_ids, before_hit, now_hit)`; `flipped_questions(retrieval: RetrievalReport, baseline: Baseline | None, questions: Mapping[str, str]) -> tuple[Flip, ...]`; `TypeCompare(query_type, metric, before, now)`; `type_comparison(retrieval, baseline) -> tuple[TypeCompare, ...]`; `preview_settings(settings, *, strategy, chunk_size, chunk_overlap, top_k, rerank, rerank_candidates) -> Settings`; `charts.type_compare_chart(rows: Sequence[TypeCompare], height: int = 260) -> go.Figure`.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_gate_view.py`:

```python
"""Tests for playground.gate_view: what the ⑥ Gate tab draws."""

from pathlib import Path

from tests.unit.test_baselines import BASE
from tests.unit.test_gate import (
    LIMITS,
    generation_baseline,
    retrieval,
    retrieval_baseline,
)

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation.baselines import Baseline
from rag_eval_platform.evaluation.evaluator import (
    ExampleResult,
    RetrievalReport,
    RetrievalThresholds,
)
from rag_eval_platform.evaluation.gate import run_gate
from rag_eval_platform.evaluation.metrics import RetrievalScores
from rag_eval_platform.playground.core import PLAYGROUND_COLLECTION
from rag_eval_platform.playground.gate_view import (
    GATE_PREVIEW_COLLECTION,
    Flip,
    TypeCompare,
    flipped_questions,
    gate_flow_dot,
    preview_settings,
    type_comparison,
)


def test_the_what_if_gate_never_touches_the_real_collections() -> None:
    assert GATE_PREVIEW_COLLECTION not in {get_settings().collection_name, PLAYGROUND_COLLECTION}


def test_flow_is_grey_before_any_run_and_coloured_after() -> None:
    assert "#16a34a" not in gate_flow_dot(None)  # no green before a run

    passed = gate_flow_dot(run_gate(retrieval(), None, generation_baseline(), BASE, LIMITS))
    blocked = gate_flow_dot(run_gate(retrieval(recall=0.5), None, None, BASE, LIMITS))

    assert "Pass" in passed and "#16a34a" in passed
    assert "Block" in blocked and "#dc2626" in blocked


def example(example_id: str, recall: float) -> ExampleResult:
    return ExampleResult(example_id, "short", ("a.md",), ("a.md",),
                         RetrievalScores(0.2, recall, recall, recall))  # fmt: skip


def test_flipped_questions_list_hits_that_became_misses_and_back() -> None:
    now = RetrievalReport(5, RetrievalThresholds(0.8, 0.7, 0.7), RetrievalScores(0, 0, 0, 0),
                          {}, (example("q1", 0.0), example("q2", 1.0), example("q3", 1.0)), ())  # fmt: skip
    before = Baseline({"examples": [
        {"example_id": "q1", "scores": {"recall": 1.0}},
        {"example_id": "q2", "scores": {"recall": 0.5}},
        {"example_id": "q3", "scores": {"recall": 1.0}},
    ]}, BASE.to_dict())  # fmt: skip

    flips = flipped_questions(now, before, {"q1": "Q one?", "q2": "Q two?"})

    assert flips == (
        Flip("q1", "Q one?", "short", ("a.md",), before_hit=True, now_hit=False),
        Flip("q2", "Q two?", "short", ("a.md",), before_hit=False, now_hit=True),
    )
    assert flipped_questions(now, None, {}) == ()


def test_type_comparison_pairs_baseline_and_now() -> None:
    rows = type_comparison(retrieval(recall=0.8), retrieval_baseline(recall=0.93))

    assert TypeCompare("short", "recall", 0.93, 0.8) in rows
    assert all(r.before is None for r in type_comparison(retrieval(), None))


def test_preview_settings_apply_the_sidebar_choices(tmp_path: Path) -> None:
    settings = preview_settings(get_settings(), strategy="fixed", chunk_size=300,
                                chunk_overlap=30, top_k=3, rerank=False,
                                rerank_candidates=20)  # fmt: skip

    assert (settings.chunk_strategy, settings.chunk_size, settings.top_k) == ("fixed", 300, 3)
```

Add to `tests/unit/test_charts.py` (it already skips when Plotly is missing):

```python
def test_type_compare_chart_has_a_before_and_a_now_series() -> None:
    from rag_eval_platform.playground.gate_view import TypeCompare

    fig = charts.type_compare_chart(
        [TypeCompare("short", "recall", 0.93, 0.8), TypeCompare("short", "mrr", None, 0.7)]
    )

    assert [trace.name for trace in fig.data] == ["baseline", "now"]
```

(Use the module alias that `test_charts.py` already imports for `charts`.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_gate_view.py tests/unit/test_charts.py -q`
Expected: `No module named 'rag_eval_platform.playground.gate_view'`.

- [ ] **Step 3: Implement** `src/rag_eval_platform/playground/gate_view.py`:

```python
"""Pure pieces of the ⑥ Gate tab: the CI flow diagram, flipped questions, comparisons."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rag_eval_platform.config.settings import ChunkStrategy, Settings
from rag_eval_platform.evaluation.baselines import Baseline
from rag_eval_platform.evaluation.evaluator import RetrievalReport
from rag_eval_platform.evaluation.gate import GateResult

GATE_PREVIEW_COLLECTION = "gate_preview"  # the what-if index; never the real collections
GATE_STEPS = ("Pull request", "Build index", "Evaluate retrieval", "Generation baseline",
              "Compare")  # fmt: skip
_GREEN, _RED, _GREY = "#16a34a", "#dc2626", "#9ca3af"
_FILL = {_GREEN: "#dcfce7", _RED: "#fee2e2", _GREY: "#f3f4f6"}


@dataclass(frozen=True)
class Flip:
    example_id: str
    question: str
    query_type: str
    relevant_doc_ids: tuple[str, ...]
    before_hit: bool
    now_hit: bool


@dataclass(frozen=True)
class TypeCompare:
    query_type: str
    metric: str
    before: float | None
    now: float


def gate_flow_dot(result: GateResult | None) -> str:
    """Graphviz source for the CI flow, each step coloured by the latest result."""
    if result is None:
        colours = [_GREY] * len(GATE_STEPS)
        verdict, verdict_colour = "Pass or block", _GREY
    else:
        retrieval_ok = all(c.status == "pass" for c in result.checks if c.group == "retrieval")
        generation_ok = all(c.status == "pass" for c in result.checks
                            if c.group == "generation")  # fmt: skip
        final = _GREEN if result.passed else _RED
        colours = [_GREEN, _GREEN, _GREEN if retrieval_ok else _RED,
                   _GREEN if generation_ok else _RED, final]  # fmt: skip
        verdict = "✅ Pass: may merge" if result.passed else "⛔ Block: fix first"
        verdict_colour = final
    nodes = [
        f'  s{i} [label="{step}", color="{colour}", fillcolor="{_FILL[colour]}"];'
        for i, (step, colour) in enumerate(zip(GATE_STEPS, colours, strict=True))
    ]
    edges = [f"  s{i} -> s{i + 1};" for i in range(len(GATE_STEPS) - 1)]
    return "\n".join([
        "digraph gate {", "  rankdir=LR;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [color="#9ca3af"];', *nodes,
        f'  verdict [label="{verdict}", color="{verdict_colour}", '
        f'fillcolor="{_FILL[verdict_colour]}", penwidth=2];',
        *edges, f"  s{len(GATE_STEPS) - 1} -> verdict;", "}",
    ])  # fmt: skip


def flipped_questions(
    retrieval: RetrievalReport, baseline: Baseline | None, questions: Mapping[str, str]
) -> tuple[Flip, ...]:
    """Questions whose every relevant document was found before but not now, or back."""
    if baseline is None:
        return ()
    before = {
        e.get("example_id"): _recall(e)
        for e in baseline.report.get("examples", [])
        if isinstance(e, dict)
    }
    flips = []
    for result in retrieval.examples:
        old = before.get(result.example_id)
        if old is None:
            continue
        before_hit, now_hit = old >= 1.0, result.scores.recall >= 1.0
        if before_hit != now_hit:
            flips.append(Flip(result.example_id, questions.get(result.example_id, ""),
                              result.query_type, result.relevant_doc_ids, before_hit,
                              now_hit))  # fmt: skip
    return tuple(flips)


def type_comparison(
    retrieval: RetrievalReport, baseline: Baseline | None
) -> tuple[TypeCompare, ...]:
    saved = baseline.report.get("by_query_type", {}) if baseline else {}
    rows = []
    for query_type, scores in sorted(retrieval.by_query_type.items()):
        old = saved.get(query_type, {}) if isinstance(saved, dict) else {}
        for metric in ("recall", "mrr", "ndcg"):
            value = old.get(metric) if isinstance(old, dict) else None
            before = float(value) if isinstance(value, int | float) else None
            rows.append(TypeCompare(query_type, metric, before, getattr(scores, metric)))
    return tuple(rows)


def preview_settings(
    settings: Settings,
    *,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    rerank: bool,
    rerank_candidates: int,
) -> Settings:
    """The settings a pull request would have if it changed only these values."""
    return settings.model_copy(update={
        "chunk_strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "top_k": top_k, "rerank": rerank, "rerank_candidates": rerank_candidates,
    })  # fmt: skip


def _recall(example: Mapping[str, Any]) -> float | None:
    scores = example.get("scores", {})
    value = scores.get("recall") if isinstance(scores, dict) else None
    return float(value) if isinstance(value, int | float) else None
```

Append to `src/rag_eval_platform/playground/charts.py` (add `from rag_eval_platform.playground.gate_view import TypeCompare` to its imports, grouped with the other project imports):

```python
def type_compare_chart(rows: Sequence[TypeCompare], height: int = 260) -> go.Figure:
    """Baseline vs now for each query type and metric (grouped bars)."""
    labels = [f"{r.query_type} · {r.metric}" for r in rows]
    fig = go.Figure()
    fig.add_bar(name="baseline", x=labels, y=[r.before or 0.0 for r in rows],
                marker_color=MUTED)  # fmt: skip
    fig.add_bar(name="now", x=labels, y=[r.now for r in rows], marker_color=NEUTRAL)
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.12})
    fig.update_yaxes(range=[0, 1.1])
    return _layout(fig, height)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/unit/test_gate_view.py tests/unit/test_charts.py -q`
Expected: all pass.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/playground/gate_view.py src/rag_eval_platform/playground/charts.py tests/unit/test_gate_view.py tests/unit/test_charts.py
git diff --cached
git commit -m "feat: playground gate view logic and baseline comparison chart"
```

---

### Task 9: The ⑥ Gate tab

**Files:**
- Create: `src/rag_eval_platform/playground/tabs/gate.py`
- Modify: `src/rag_eval_platform/playground/shared.py` (`TAB_LABELS`, a `GATE_RUN` key, a cached `connect_gate_store`)
- Modify: `src/rag_eval_platform/playground/app.py` (`RENDERERS`, import)
- Modify: `tests/integration/test_playground.py` (the tab labels list)

**Interfaces:**
- Consumes: everything from Tasks 1–4 and 8; `TabContext`, `load_embedder`, `load_reranker`, `settings`, `CHART_CONFIG`, `charts.gauge_chart`.
- Produces: `tabs.gate.render(ctx: TabContext) -> None`; `shared.GATE_RUN = "gate_run"`; `shared.connect_gate_store(host: str, port: int) -> ChromaVectorStore`.

- [ ] **Step 1: Update the smoke test first** (red): in `tests/integration/test_playground.py`, add `"⑥ Gate"` to the end of the expected tab labels list.

Run: `uv run pytest tests/integration/test_playground.py::test_app_renders_without_errors -q`
Expected: FAIL. The labels list lacks `"⑥ Gate"`.

- [ ] **Step 2: Wire the tab**

In `shared.py`:

```python
TAB_LABELS = [
    "Overview", "① Ingest", "② Embed", "③ Retrieve", "④ Generate", "⑤ Evaluate", "⑥ Gate",
]  # fmt: skip
GATE_RUN = "gate_run"  # the what-if gate's result, saved before the next st.* call
```

and next to `connect_store`:

```python
@st.cache_resource(show_spinner=False)
def connect_gate_store(host: str, port: int) -> ChromaVectorStore:
    return ChromaVectorStore.connect(host, port, GATE_PREVIEW_COLLECTION)
```

(import `GATE_PREVIEW_COLLECTION` from `rag_eval_platform.playground.gate_view`; match the decorator style `connect_store` already uses.)

In `app.py`, import `gate` from `rag_eval_platform.playground.tabs` and add `"⑥ Gate": gate.render,` to `RENDERERS`.

`src/rag_eval_platform/playground/tabs/gate.py`:

```python
"""⑥ Gate: how CI decides whether a change may merge, and a what-if run of it."""

import html
import json
from dataclasses import dataclass

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.evaluation.baselines import (
    GENERATION_BASELINE,
    RETRIEVAL_BASELINE,
    Baseline,
    BaselineError,
    load_baseline,
)
from rag_eval_platform.evaluation.evaluator import RetrievalReport, RetrievalThresholds
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
from rag_eval_platform.evaluation.gate import (
    GateLimits,
    GateResult,
    gate_from_dict,
    run_gate,
)
from rag_eval_platform.evaluation.gate_runner import rebuild_and_evaluate
from rag_eval_platform.evaluation.golden_dataset import GoldenDatasetError, load_golden_dataset
from rag_eval_platform.ingestion.embedding import EmbeddingError
from rag_eval_platform.ingestion.loaders import DocumentLoadError
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import REPORTS_DIR
from rag_eval_platform.playground.gate_view import (
    flipped_questions,
    gate_flow_dot,
    preview_settings,
    type_comparison,
)
from rag_eval_platform.playground.shared import (
    GATE_RUN,
    TabContext,
    connect_gate_store,
    load_embedder,
    load_reranker,
    settings,
)
from rag_eval_platform.retrieval.vector_store import VectorStoreError


@dataclass(frozen=True)
class GateRun:
    result: GateResult
    retrieval: RetrievalReport
    baseline: Baseline | None
    questions: dict[str, str]


def render(ctx: TabContext) -> None:
    run: GateRun | None = st.session_state.get(GATE_RUN)
    latest = run.result if run else _latest_ci_result()
    st.graphviz_chart(gate_flow_dot(latest), width="stretch")
    left, right = st.columns([1, 2], gap="large")
    with left:
        _explain_and_run(ctx)
    with right:
        if latest is None:
            st.info("No gate result yet: click **Run the gate** to try your sidebar settings.")
            return
        _board(latest)
        if run is not None:
            _comparison(run)


def _explain_and_run(ctx: TabContext) -> None:
    st.markdown(
        "Every pull request is scored on the **30 golden questions** before it can merge. "
        "Retrieval is measured **live**; answer quality comes from a **committed baseline** "
        "judged locally, which must match the current settings."
    )
    o = ctx.options
    st.caption(f"What-if with your sidebar: {o.strategy} chunks of {o.chunk_size} "
               f"(overlap {o.chunk_overlap}), top-{o.top_k}, "
               f"re-rank {'on' if o.rerank else 'off'}.")  # fmt: skip
    if st.button("Run the gate", type="primary", key="gate-run"):
        _run(ctx)


def _run(ctx: TabContext) -> None:
    o = ctx.options
    what_if = preview_settings(settings, strategy=o.strategy, chunk_size=o.chunk_size,
                               chunk_overlap=o.chunk_overlap, top_k=o.top_k, rerank=o.rerank,
                               rerank_candidates=o.rerank_candidates)  # fmt: skip
    try:
        with st.status("Running the gate...", expanded=True) as status:
            examples = load_golden_dataset(what_if.golden_dataset_path)
            retrieval = rebuild_and_evaluate(
                examples, raw_dir=what_if.raw_data_dir, strategy=o.strategy,
                chunk_size=o.chunk_size, chunk_overlap=o.chunk_overlap,
                embedder=load_embedder(what_if.embedding_model),
                store=connect_gate_store(what_if.chroma_host, what_if.chroma_port),
                top_k=o.top_k,
                reranker=load_reranker(what_if.reranker_model) if o.rerank else None,
                rerank_candidates=o.rerank_candidates,
                thresholds=RetrievalThresholds.from_settings(what_if),
                on_progress=st.write,
            )  # fmt: skip
            baseline = load_baseline(RETRIEVAL_BASELINE)
            result = run_gate(retrieval, baseline, load_baseline(GENERATION_BASELINE),
                              current_fingerprint(what_if), GateLimits.from_settings(what_if))  # fmt: skip
            # Saved before the next st.* call, so a click now cannot lose the result.
            st.session_state[GATE_RUN] = GateRun(result, retrieval, baseline,
                                                 {e.id: e.question for e in examples})  # fmt: skip
            status.update(label="Gate finished", state="complete", expanded=False)
    except (GoldenDatasetError, DocumentLoadError, VectorStoreError, EmbeddingError,
            OptionalDependencyError, BaselineError, OSError, ValueError) as exc:  # fmt: skip
        st.error(str(exc))
        return
    st.rerun()


def _board(result: GateResult) -> None:
    if result.passed:
        st.success("✅ The gate passes: this change may merge.")
    else:
        st.error("⛔ The gate blocks this change.")
    scored = [c for c in result.checks if c.value is not None]
    for i, column in enumerate(st.columns(max(len(scored), 1))):
        if i < len(scored):
            check = scored[i]
            column.plotly_chart(
                charts.gauge_chart(check.name, check.value or 0.0, check.threshold, height=170),
                key=f"gate-gauge-{check.name}", config=charts.CHART_CONFIG,
            )  # fmt: skip
    for check in result.checks:
        mark = "✅" if check.status == "pass" else "⛔"
        st.markdown(f"{mark} **{html.escape(check.name)}**: {html.escape(check.reason)}")
    if result.changes:
        st.warning("The generation baseline is stale. Changed since it was judged:\n\n"
                   + "\n".join(f"- `{change}`" for change in result.changes))  # fmt: skip


def _comparison(run: GateRun) -> None:
    st.markdown("**Baseline vs now, by question type**")
    st.plotly_chart(charts.type_compare_chart(type_comparison(run.retrieval, run.baseline)),
                    key="gate-types", config=charts.CHART_CONFIG)  # fmt: skip
    flips = flipped_questions(run.retrieval, run.baseline, run.questions)
    st.markdown(f"**Questions that flipped: {len(flips)}**")
    for flip in flips:
        arrow = "hit → miss" if flip.before_hit else "miss → hit"
        st.markdown(f"- `{flip.example_id}` ({arrow}) {html.escape(flip.question)} · "
                    f"expects {', '.join(flip.relevant_doc_ids)}")  # fmt: skip


def _latest_ci_result() -> GateResult | None:
    path = REPORTS_DIR / "gate_report.json"
    try:
        return gate_from_dict(json.loads(path.read_text(encoding="utf-8"))["gate"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
```

- [ ] **Step 3: Run the checks**

Run: `uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit tests/integration/test_playground.py -q`
Expected: all clean; the smoke test now passes with `"⑥ Gate"`.

- [ ] **Step 4: Check it in the browser** (restart the app first; the file watcher is off). Open the ⑥ Gate tab at 1440×900, click **Run the gate** with the defaults, and confirm it passes. Then set the chunk size to 150 in the sidebar, run again, and confirm it blocks with flipped questions and a stale-baseline warning (`chunk_size: 800 → 150`). Screenshot both states.

- [ ] **Step 5: Commit (when the user asks)**

```bash
git add src/rag_eval_platform/playground/tabs/gate.py src/rag_eval_platform/playground/shared.py src/rag_eval_platform/playground/app.py tests/integration/test_playground.py
git diff --cached
git commit -m "feat: playground ⑥ Gate tab with a live what-if gate"
```

---

### Task 10: Docs

**Files:**
- Create: `docs/learning/phase-5.md`
- Modify: `docs/evaluation_methodology.md` (§4), `readme.md`, `CLAUDE.md`, `docs/architecture.md`, `docs/learning/playground.md`

- [ ] **Step 1: Write `docs/learning/phase-5.md`** in the style of the other learning pages (short sections, Mermaid diagrams GitHub can parse, no `&quot;` in labels). Required sections:
  1. *Why a gate*: quality drops silently. A chunk-size change can lower Recall on multi-hop questions without any test failing.
  2. *What runs where*: a `sequenceDiagram` of Developer → GitHub → Runner (Chroma service) → run_gate → PR comment.
  3. *The rules*: a `flowchart TD` with threshold → max drop → baseline fresh? → generation thresholds → pass or block.
  4. *The fingerprint*: a table of its fields and why each one matters.
  5. *Updating a baseline*: the exact commands (`run_evaluation.py --save-baseline`, `run_generation_evaluation.py --save-baseline`, `save_baseline.py generation ...`).
  6. *Making the check required*: GitHub → Settings → Branches → branch protection for `main` → require the status check "Evaluate against golden dataset".
  7. *Try it in the ⑥ Gate tab*: change the chunk size to 150 and watch the gate block.
- [ ] **Step 2: Rewrite `evaluation_methodology.md` §4** to match what's built: live retrieval, the committed generation baseline with its freshness rule, `RAG_GATE_MAX_DROP`, the PR comment and artifact, and the required-check note. Keep the baseline tables.
- [ ] **Step 3: Update the other docs.**
  - `readme.md`: the Phase 5 status, the `run_gate.py` and `save_baseline.py` commands, the ⑥ Gate tab in the playground table, and `phase-5.md` in the docs list.
  - `CLAUDE.md`: the Project status (Phase 5 done, `evaluation_gate.yml` no longer a placeholder), the commands, a Gate bullet in the Evaluation section (fingerprint, baselines, freshness, max drop, `GATE_MARKER`), and seven tabs in the playground bullet.
  - `docs/architecture.md`: the Phase 5 rows become *in use*, and the NumPy row becomes *in use*.
  - `docs/learning/playground.md`: a ⑥ Gate row in the tabs table.
- [ ] **Step 4: Verify.**
Run: `grep -rn "evaluation_gate.yml\|placeholder" CLAUDE.md readme.md docs/*.md | head` and check that no line still calls the gate a placeholder.
- [ ] **Step 5: Commit (when the user asks)**

```bash
git add docs/learning/phase-5.md docs/evaluation_methodology.md readme.md CLAUDE.md docs/architecture.md docs/learning/playground.md docs/superpowers/specs/2026-09-28-phase-5-evaluation-gate-design.md docs/superpowers/plans/2026-09-28-phase-5-evaluation-gate.md
git diff --cached
git commit -m "docs: phase 5 evaluation gate guide, methodology, spec and plan"
```

---

### Task 11: Final verification

- [ ] **Step 1:** Run: `uv lock --check && uv run ruff check . && uv run ruff format --check . && uv run mypy src tests && uv run pytest tests/unit --cov -q && uv run pytest -m integration -q`
Expected: all green; the integration tests skip only where a model or Ollama is missing.
- [ ] **Step 2:** Run: `uv run python scripts/run_gate.py; echo "exit $?"`
Expected: `exit 0`.
- [ ] **Step 3:** Prove the gate blocks: `RAG_CHUNK_SIZE=150 RAG_CHUNK_OVERLAP=20 uv run python scripts/run_gate.py; echo "exit $?"`
Expected: `exit 1`, with a stale-baseline change `chunk_size: 800 → 150`. Then re-run step 2 so Chroma holds the default index again.
- [ ] **Step 4:** Report to the user, and push and open the pull request only when they ask. The pull request's own `Evaluation gate` run is the end-to-end test of the workflow.
