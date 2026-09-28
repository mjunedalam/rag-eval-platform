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


def test_adopts_a_matching_legacy_report_only_with_the_flag(workspace: Path) -> None:
    path = write_report(workspace)

    assert baseline_cli.main(["generation", str(path)]) == 1
    assert baseline_cli.main(["generation", str(path), "--legacy"]) == 0

    saved = json.loads((workspace / "baselines" / "generation.json").read_text(encoding="utf-8"))
    assert saved["report"]["overall"]["faithfulness"] == 0.9


def test_refuses_a_report_from_another_judge(
    workspace: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_report(workspace, judge_model="other-judge")

    assert baseline_cli.main(["generation", str(path), "--legacy"]) == 1
    assert "judge_model" in capsys.readouterr().out
    assert not (workspace / "baselines").exists()


def test_unreadable_report_returns_2(workspace: Path) -> None:
    (workspace / "bad.json").write_text("{nope", encoding="utf-8")

    assert baseline_cli.main(["generation", str(workspace / "bad.json")]) == 2
