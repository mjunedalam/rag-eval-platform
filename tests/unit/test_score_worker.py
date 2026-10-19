"""Tests for observability.score_worker: background TruLens scoring of logged answers."""

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from tests.unit.fake_tracer import FakeTracer
from tests.unit.observe_records import T0, make_query

from rag_eval_platform.observability.score_worker import ScoreWorker
from rag_eval_platform.observability.store import ObserveStore
from rag_eval_platform.observability.trulens_scorer import MetricScore, TriadScoreError

TEXTS = {"a.md#0": "MRR is the reciprocal rank.", "b.md#1": "NDCG discounts gains."}


@dataclass
class FakeScorer:
    judge: str = "gemma3:4b"
    error: Exception | None = None
    calls: list[tuple[str, tuple[str, ...], str]] = field(default_factory=list)
    gate: threading.Event | None = None

    def score(self, question: str, contexts: Sequence[str], answer: str) -> tuple[MetricScore, ...]:
        if self.gate is not None:
            self.gate.wait(5)
        self.calls.append((question, tuple(contexts), answer))
        if self.error is not None:
            raise self.error
        return (
            MetricScore("context_relevance", 0.5, "half the chunks help"),
            MetricScore("groundedness", 1.0, "all supported"),
            MetricScore("answer_relevance", 0.9, "on topic"),
        )


@dataclass
class FakeTexts:
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    def __call__(self, collection: str, chunk_ids: Sequence[str]) -> Sequence[str]:
        self.calls.append((collection, tuple(chunk_ids)))
        return [TEXTS[i] for i in chunk_ids if i in TEXTS]


@pytest.fixture
def store(tmp_path: Path) -> ObserveStore:
    return ObserveStore(tmp_path / "rag.db")


def worker(store: ObserveStore, scorer: FakeScorer | None = None,
           tracer: FakeTracer | None = None, **kwargs: object) -> ScoreWorker:  # fmt: skip
    return ScoreWorker(scorer or FakeScorer(), store, FakeTexts(), tracer or FakeTracer(),
                       clock=lambda: T0, **kwargs)  # type: ignore[arg-type]  # fmt: skip


class TestScoreNow:
    def test_scores_an_answer_saves_rows_and_sends_them_to_langfuse(
        self, store: ObserveStore
    ) -> None:
        store.add_query(make_query(1, collection="playground"))
        scorer, tracer = FakeScorer(), FakeTracer()
        texts = FakeTexts()
        w = ScoreWorker(scorer, store, texts, tracer, clock=lambda: T0)

        assert w.score_now("q1") == "scored"

        assert texts.calls == [("playground", ("a.md#0", "b.md#1"))]
        assert scorer.calls == [("Question 1?", tuple(TEXTS.values()), "Answer [1].")]
        rows = {r.metric: r for r in store.triad_scores_for(["q1"])}
        assert {m: r.value for m, r in rows.items()} == {
            "context_relevance": 0.5, "groundedness": 1.0, "answer_relevance": 0.9,
        }  # fmt: skip
        assert rows["groundedness"].reason == "all supported"
        assert rows["groundedness"].judge == "gemma3:4b"
        assert rows["groundedness"].ts == T0
        assert tracer.scores == [
            {"query_id": "q1", "name": "trulens_context_relevance", "value": 0.5,
             "comment": "half the chunks help", "data_type": "NUMERIC"},
            {"query_id": "q1", "name": "trulens_groundedness", "value": 1.0,
             "comment": "all supported", "data_type": "NUMERIC"},
            {"query_id": "q1", "name": "trulens_answer_relevance", "value": 0.9,
             "comment": "on topic", "data_type": "NUMERIC"},
        ]  # fmt: skip

    @pytest.mark.parametrize(
        "changes",
        [
            {"refusal": True},
            {"status": "error", "answer": None, "error": "LLM down"},
            {"question": "sha256:abc"},  # text storage was off: nothing to judge
        ],
    )
    def test_skips_what_cannot_be_judged(
        self, store: ObserveStore, changes: dict[str, object]
    ) -> None:
        store.add_query(make_query(1, **changes))
        scorer = FakeScorer()

        assert worker(store, scorer).score_now("q1") == "skipped"
        assert scorer.calls == []
        assert store.triad_scores_for(["q1"]) == []

    def test_unknown_query_is_skipped(self, store: ObserveStore) -> None:
        assert worker(store).score_now("missing") == "skipped"

    def test_a_failed_score_saves_nothing_so_it_can_be_retried(
        self, store: ObserveStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        store.add_query(make_query(1))
        tracer = FakeTracer()

        with caplog.at_level(logging.WARNING):
            outcome = worker(
                store, FakeScorer(error=TriadScoreError("bad reply")), tracer
            ).score_now("q1")

        assert outcome == "failed"
        assert store.triad_scores_for(["q1"]) == []
        assert tracer.scores == []
        assert "bad reply" in caplog.text

    def test_judge_crashes_are_failures_not_exceptions(self, store: ObserveStore) -> None:
        store.add_query(make_query(1))

        assert worker(store, FakeScorer(error=ConnectionError("ollama down"))).score_now("q1") == (
            "failed"
        )


class TestBackground:
    def test_submitted_answers_are_scored_on_the_worker_thread(self, store: ObserveStore) -> None:
        store.add_query(make_query(1))
        store.add_query(make_query(2))
        w = worker(store)

        assert w.submit("q1")
        assert w.submit("q2")
        w.stop(timeout=5)

        assert {r.query_id for r in store.triad_scores_for(["q1", "q2"])} == {"q1", "q2"}

    def test_a_full_queue_drops_the_job_with_a_warning(
        self, store: ObserveStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        for n in range(3):
            store.add_query(make_query(n))
        gate = threading.Event()
        w = worker(store, FakeScorer(gate=gate), queue_size=1)

        with caplog.at_level(logging.WARNING):
            first = w.submit("q0")  # taken by the thread, which waits at the gate
            w.wait_until_busy(timeout=5)
            second = w.submit("q1")  # fills the queue
            third = w.submit("q2")  # dropped
        gate.set()
        w.stop(timeout=5)

        assert (first, second, third) == (True, True, False)
        assert "queue is full" in caplog.text
        assert store.unscored_query_ids() == ["q2"]  # the catch-up command picks it up later

    def test_the_thread_survives_a_failing_job(self, store: ObserveStore) -> None:
        store.add_query(make_query(1))
        store.add_query(make_query(2))
        scorer = FakeScorer(error=RuntimeError("boom"))
        w = worker(store, scorer)

        w.submit("q1")
        w.wait_until_idle(timeout=5)
        scorer.error = None
        w.submit("q2")
        w.stop(timeout=5)

        assert [r.query_id for r in store.triad_scores_for(["q2"])][:1] == ["q2"]

    def test_submit_after_stop_is_refused(self, store: ObserveStore) -> None:
        w = worker(store)
        w.stop(timeout=5)

        assert w.submit("q1") is False
