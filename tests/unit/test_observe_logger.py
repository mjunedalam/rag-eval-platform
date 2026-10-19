"""Tests for observability.logger: answers become rows, and failures never escape."""

import hashlib
from pathlib import Path

import pytest
from tests.unit.fake_tracer import FakeTracer
from tests.unit.observe_records import T0
from tests.unit.test_api_app import answer

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import GenerationError
from rag_eval_platform.generation.prompt_templates import NO_ANSWER
from rag_eval_platform.observability import logger as logger_module
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
    assert run.fingerprint is not None
    assert len(run.fingerprint) == 16
    assert record_eval_run(None, "gate", True, {}, None) is None


def test_record_eval_run_never_raises(tmp_path: Path) -> None:
    (tmp_path / "rag.db").write_bytes(b"not a database" * 100)

    assert record_eval_run(ObserveStore(tmp_path / "rag.db"), "gate", True, {}, None) is None


def test_record_uses_the_query_id_chosen_when_the_question_arrived(tmp_path: Path) -> None:
    store = ObserveStore(tmp_path / "o.db")
    log = logger_for(store)

    assert (
        log.record(answer(), retrieve_ms=1, total_ms=2, query_id="chosen-early") == "chosen-early"
    )
    assert log.record_error("q?", GenerationError("x"), total_ms=1, query_id="err-id") == "err-id"
    assert store.get_query("chosen-early") is not None
    assert store.get_query("err-id") is not None


def test_record_feedback_also_scores_the_trace(tmp_path: Path) -> None:
    store = ObserveStore(tmp_path / "o.db")
    query_id = logger_for(store).record(answer(), retrieve_ms=1, total_ms=2)
    assert query_id is not None
    tracer = FakeTracer()

    record_feedback(store, query_id, 0, "wrong source", "api", tracer=tracer)

    assert tracer.scores == [{"query_id": query_id, "name": "user_feedback", "value": 0,
                              "comment": "wrong source", "data_type": "BOOLEAN"}]  # fmt: skip


def test_unknown_query_feedback_is_not_scored(tmp_path: Path) -> None:
    tracer = FakeTracer()

    with pytest.raises(UnknownQueryError):
        record_feedback(ObserveStore(tmp_path / "o.db"), "ghost", 1, None, "api", tracer=tracer)

    assert tracer.scores == []


class TestScoringHandOff:
    def test_a_recorded_answer_is_handed_to_the_scorer(self, tmp_path: Path) -> None:
        handed: list[str] = []
        recorder = logger_for(ObserveStore(tmp_path / "rag.db"), on_recorded=handed.append)

        query_id = recorder.record(answer(), retrieve_ms=1, total_ms=2, query_id="q-1")

        assert handed == [query_id] == ["q-1"]

    def test_refusals_and_errors_are_not_handed_over(self, tmp_path: Path) -> None:
        handed: list[str] = []
        recorder = logger_for(ObserveStore(tmp_path / "rag.db"), on_recorded=handed.append)

        recorder.record(answer(NO_ANSWER), retrieve_ms=1, total_ms=2)
        recorder.record_error("q?", GenerationError("down"), total_ms=1)

        assert handed == []

    def test_a_failing_hand_off_never_breaks_recording(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        def broken(query_id: str) -> None:
            raise RuntimeError("worker gone")

        recorder = logger_for(ObserveStore(tmp_path / "rag.db"), on_recorded=broken)

        assert recorder.record(answer(), retrieve_ms=1, total_ms=2, query_id="q-1") == "q-1"
        assert "worker gone" in caplog.text

    @pytest.mark.parametrize(("source", "scored"), [("api", True), ("playground", True),
                                                    ("cli", False)])  # fmt: skip
    def test_the_factory_scores_long_running_sources_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, scored: bool
    ) -> None:
        """ask.py exits right after answering, so run_trulens.py --pending scores it later."""
        handed: list[str] = []

        class Worker:
            def submit(self, query_id: str) -> bool:
                handed.append(query_id)
                return True

        monkeypatch.setattr(logger_module, "get_score_worker", lambda settings: Worker())
        recorder = create_query_logger(Settings(_env_file=None, observe_db=tmp_path / "rag.db"),
                                       source)  # type: ignore[arg-type]  # fmt: skip

        assert recorder is not None
        recorder.record(answer(), retrieve_ms=1, total_ms=2, query_id="q-1")
        assert handed == (["q-1"] if scored else [])
