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
    step_summary = workspace / "step_summary.md"

    assert run(workspace, "--summary", str(step_summary)) == 2
    assert not (workspace / "gate.json").exists()
    # The PR comment is replaced, so an older "passed" comment cannot linger.
    summary = (workspace / "gate.md").read_text(encoding="utf-8")
    assert summary.startswith(GATE_MARKER)
    assert "could not run" in summary
    assert "Chroma is down" in summary
    assert step_summary.read_text(encoding="utf-8") == summary


def test_a_malformed_baseline_returns_2(workspace: Path) -> None:
    (workspace / "baselines").mkdir()
    (workspace / "baselines" / "retrieval.json").write_text("{nope", encoding="utf-8")

    assert run(workspace) == 2
