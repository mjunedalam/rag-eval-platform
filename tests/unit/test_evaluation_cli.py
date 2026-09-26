"""Tests for evaluation.cli (the logic behind scripts/run_evaluation.py)."""

import json
from pathlib import Path

import pytest

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation import cli
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult, VectorStoreError

GOLDEN = [
    {
        "id": "q1",
        "question": "What is force?",
        "expected_answer": "F = ma",
        "relevant_doc_ids": ["forces.md"],
        "query_type": "short",
    }
]


class FakeRetriever:
    def __init__(self, doc_id: str) -> None:
        self.doc_id = doc_id

    def retrieve(self, query: str) -> list[SearchResult]:
        chunk = Chunk(id=f"{self.doc_id}#0", doc_id=self.doc_id, index=0, text="t", start_index=0)
        return [SearchResult(chunk=chunk, score=0.9)]


@pytest.fixture
def golden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    path = tmp_path / "qa_pairs.json"
    path.write_text(json.dumps(GOLDEN), encoding="utf-8")
    return path


def use_retriever(monkeypatch: pytest.MonkeyPatch, doc_id: str) -> None:
    monkeypatch.setattr(cli, "create_retriever", lambda settings: FakeRetriever(doc_id))


def test_passing_run_writes_report_prints_table_and_returns_0(
    golden: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_retriever(monkeypatch, "forces.md")
    report_path = tmp_path / "reports" / "retrieval.json"

    exit_code = cli.main(["--golden", str(golden), "--report", str(report_path)])

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    stdout = capsys.readouterr().out
    assert "recall@k" in stdout
    assert "PASS" in stdout


def test_below_threshold_returns_1_and_lists_misses(
    golden: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_retriever(monkeypatch, "wrong.md")

    exit_code = cli.main(["--golden", str(golden), "--report", str(tmp_path / "r.json")])

    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "FAIL" in stdout
    assert "q1" in stdout
    assert "wrong.md" in stdout


def test_setup_error_returns_2(
    golden: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def down(settings: object) -> None:
        raise VectorStoreError("Cannot reach Chroma")

    monkeypatch.setattr(cli, "create_retriever", down)

    assert cli.main(["--golden", str(golden), "--report", str(tmp_path / "r.json")]) == 2
    assert "Cannot reach Chroma" in capsys.readouterr().err


def test_invalid_golden_dataset_returns_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    use_retriever(monkeypatch, "forces.md")

    assert cli.main(["--golden", str(tmp_path / "missing.json")]) == 2
