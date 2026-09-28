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

    with pytest.raises(BaselineError, match=r"bad\.json"):
        load_baseline(path)


def test_a_legacy_report_made_with_the_current_versions_can_be_adopted() -> None:
    assert check_adoptable(GENERATION_REPORT, BASE, golden_size=30, legacy=True) == ()


def test_adopting_refuses_other_versions_or_a_partial_run() -> None:
    problems = check_adoptable(
        GENERATION_REPORT, replace(BASE, judge_model="other"), 31, legacy=True
    )

    assert any("judge_model" in p for p in problems)
    assert any("scored 30 of 31" in p for p in problems)


def test_a_report_without_a_fingerprint_needs_the_legacy_flag() -> None:
    problems = check_adoptable(GENERATION_REPORT, BASE, golden_size=30)

    assert len(problems) == 1
    assert "--legacy" in problems[0]


def test_a_report_with_a_fingerprint_must_match_it_exactly() -> None:
    stamped = {**GENERATION_REPORT, "fingerprint": replace(BASE, chunk_size=400).to_dict()}

    problems = check_adoptable(stamped, BASE, golden_size=30, legacy=True)

    assert problems == ("the report was made with chunk_size: 400 → 800",)
    assert check_adoptable({**GENERATION_REPORT, "fingerprint": BASE.to_dict()}, BASE, 30) == ()
