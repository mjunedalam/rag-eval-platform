"""Tests for observability.trulens_cli: the catch-up command for unscored answers."""

from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.unit.fake_tracer import FakeTracer
from tests.unit.observe_records import make_query, make_triad

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability import trulens_cli
from rag_eval_platform.observability.score_worker import Outcome
from rag_eval_platform.observability.store import ObserveStore


@dataclass
class FakeWorker:
    outcomes: dict[str, Outcome] = field(default_factory=dict)
    scored: list[str] = field(default_factory=list)

    def score_now(self, query_id: str) -> Outcome:
        self.scored.append(query_id)
        return self.outcomes.get(query_id, "scored")


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ObserveStore:
    monkeypatch.setenv("RAG_OBSERVE_DB", str(tmp_path / "rag.db"))
    store = ObserveStore(tmp_path / "rag.db")
    for n in range(1, 4):
        store.add_query(make_query(n))
    store.save_triad_scores([make_triad("q2")])
    return store


def run(argv: list[str], worker: FakeWorker | None, monkeypatch: pytest.MonkeyPatch,
        tracer: FakeTracer | None = None) -> int:  # fmt: skip
    monkeypatch.setattr(trulens_cli, "build_worker", lambda settings: worker)
    monkeypatch.setattr(trulens_cli, "create_tracer", lambda settings: tracer or FakeTracer())
    return trulens_cli.main(argv)


def test_pending_scores_every_unscored_answer_newest_first(
    store: ObserveStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worker, tracer = FakeWorker(), FakeTracer()

    code = run(["--pending"], worker, monkeypatch, tracer)

    assert code == 0
    assert worker.scored == ["q3", "q1"]
    assert tracer.flushed == 1  # scores reach Langfuse before the process exits
    assert "2 scored, 0 skipped, 0 failed" in capsys.readouterr().out


def test_last_limits_how_many(store: ObserveStore, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = FakeWorker()

    assert run(["--pending", "--last", "1"], worker, monkeypatch) == 0
    assert worker.scored == ["q3"]


def test_named_answers_are_scored_even_if_already_scored(
    store: ObserveStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = FakeWorker()

    assert run(["q2"], worker, monkeypatch) == 0
    assert worker.scored == ["q2"]


def test_failures_stay_pending_and_exit_1(
    store: ObserveStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    worker = FakeWorker(outcomes={"q3": "failed"})

    assert run(["--pending"], worker, monkeypatch) == 1
    assert "1 scored, 0 skipped, 1 failed" in capsys.readouterr().out


def test_cannot_run_without_a_scorer(
    store: ObserveStore, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(["--pending"], None, monkeypatch) == 2
    assert "uv sync --extra monitoring" in capsys.readouterr().out


def test_cannot_run_with_observability_off(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RAG_OBSERVE_ENABLED", "false")

    assert run(["--pending"], FakeWorker(), monkeypatch) == 2
    assert "RAG_OBSERVE_ENABLED" in capsys.readouterr().out


def test_needs_pending_or_query_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit):
        run([], FakeWorker(), monkeypatch)


def test_the_worker_is_built_even_when_background_scoring_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running the command is an explicit request, so RAG_TRULENS_ENABLED=false doesn't block it."""
    seen: list[Settings] = []
    monkeypatch.setattr(trulens_cli, "create_triad_scorer", lambda s: seen.append(s))

    trulens_cli.build_worker(Settings(_env_file=None, trulens_enabled=False))

    assert seen[0].trulens_enabled is True
