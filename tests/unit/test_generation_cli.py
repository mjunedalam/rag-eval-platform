"""Tests for evaluation.generation_cli (scripts/run_generation_evaluation.py) with fakes."""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.evaluation import generation_cli as cli
from rag_eval_platform.evaluation.citation_validity import CitationValidity, CitationVerdict
from rag_eval_platform.evaluation.judge import JudgeError, JudgeSample, JudgeScores, RagasJudge
from rag_eval_platform.generation.generator import Answer, Completion, Generator
from rag_eval_platform.generation.prompt_templates import Message
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult

GOLDEN = [
    {
        "id": f"q{i}",
        "question": f"Question {i}?",
        "expected_answer": "reference",
        "relevant_doc_ids": ["a.md"],
        "query_type": "short" if i == 1 else "multi_hop",
    }
    for i in (1, 2)
]


@dataclass
class ScriptedLlm:
    model: str = "qwen3:8b"

    def complete(self, messages: Sequence[Message]) -> Completion:
        return Completion(text="Claim [1].", input_tokens=1, output_tokens=1)

    def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
        raise NotImplementedError


class FakePipeline:
    def __init__(self) -> None:
        self.asked: list[str] = []

    def ask(self, question: str) -> Answer:
        self.asked.append(question)
        chunk = Chunk(id="a.md#0", doc_id="a.md", index=0, text="chunk", start_index=0)
        return Generator(ScriptedLlm()).generate(question, [SearchResult(chunk, 0.9)])


@dataclass
class FakeJudge:
    faithfulness: float = 1.0
    error: Exception | None = None
    model: str = "gemma3:12b"
    full_flags: list[bool] = field(default_factory=list)

    def score(self, sample: JudgeSample, *, full: bool = False) -> JudgeScores:
        if self.error is not None:
            raise self.error
        self.full_flags.append(full)
        return JudgeScores(faithfulness=self.faithfulness, answer_relevance=0.9)


class FakeChecker:
    model = "gemma3:12b"

    def check(self, answer: Answer) -> CitationValidity:
        return CitationValidity((CitationVerdict("Claim.", 1, "a.md", False),))


@dataclass
class Fakes:
    pipeline: FakePipeline
    judge: FakeJudge


@pytest.fixture
def fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fakes:
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    (tmp_path / "qa_pairs.json").write_text(json.dumps(GOLDEN), encoding="utf-8")
    built = Fakes(FakePipeline(), FakeJudge())
    monkeypatch.setattr(cli, "create_pipeline", lambda settings: built.pipeline)
    monkeypatch.setattr(cli, "create_judge", lambda settings: built.judge)
    monkeypatch.setattr(cli, "create_citation_checker", lambda settings: FakeChecker())
    return built


def run(tmp_path: Path, *extra: str) -> int:
    return cli.main(
        ["--golden", str(tmp_path / "qa_pairs.json"), "--report", str(tmp_path / "gen.json"),
         *extra]
    )  # fmt: skip


def test_passing_run_writes_report_prints_table_and_returns_0(
    fakes: Fakes, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = run(tmp_path)

    assert exit_code == 0
    report = json.loads((tmp_path / "gen.json").read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["judge_model"] == "gemma3:12b"
    stdout = capsys.readouterr().out
    assert "judge gemma3:12b" in stdout
    assert "faithfulness" in stdout
    assert "PASS" in stdout
    assert "q1 [1]: Claim." in stdout  # unsupported citation listed
    assert "PASSED" in stdout


def test_below_threshold_returns_1_and_lists_unfaithful_answers(
    fakes: Fakes, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.judge.faithfulness = 0.5

    exit_code = run(tmp_path)

    assert exit_code == 1
    stdout = capsys.readouterr().out
    assert "FAILED: faithfulness 0.500 < 0.850" in stdout
    assert "q1: faithfulness 0.50" in stdout


def test_limit_scores_only_the_first_questions(fakes: Fakes, tmp_path: Path) -> None:
    run(tmp_path, "--limit", "1")

    assert fakes.pipeline.asked == ["Question 1?"]


def test_full_adds_context_metrics(fakes: Fakes, tmp_path: Path) -> None:
    run(tmp_path, "--full")

    assert fakes.judge.full_flags == [True, True]


def test_judge_failure_returns_2_without_a_report(
    fakes: Fakes, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fakes.judge.error = JudgeError("judge is down")

    exit_code = run(tmp_path)

    assert exit_code == 2
    assert not (tmp_path / "gen.json").exists()
    assert "judge is down" in capsys.readouterr().err


def test_limit_must_be_positive(fakes: Fakes, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run(tmp_path, "--limit", "0")


def test_create_judge_builds_ragas_judge_with_the_project_embedder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def fake_from_settings(settings: Settings, embedder: object) -> str:
        calls.append(embedder)
        return "judge"

    monkeypatch.setattr(cli, "create_embedder", lambda settings: "embedder")
    monkeypatch.setattr(RagasJudge, "from_settings", staticmethod(fake_from_settings))

    judge: object = cli.create_judge(Settings())

    assert judge == "judge"
    assert calls == ["embedder"]
