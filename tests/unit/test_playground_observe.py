"""Tests for the playground's observability helpers (core.py)."""

from pathlib import Path

from tests.unit.test_api_app import answer

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.store import ObserveStore
from rag_eval_platform.playground.chat import Turn
from rag_eval_platform.playground.core import (
    PLAYGROUND_COLLECTION,
    record_answer,
    record_failure,
    save_rating,
)


def on(tmp_path: Path) -> Settings:
    return Settings(_env_file=None, observe_db=tmp_path / "rag.db")


def test_answers_are_logged_as_playground_queries(tmp_path: Path) -> None:
    settings = on(tmp_path)

    query_id = record_answer(settings, answer(), retrieve_ms=5, total_ms=30, style="detailed")

    saved = ObserveStore(settings.observe_db).get_query(query_id or "")
    assert saved is not None
    assert (saved.source, saved.collection, saved.style) == (
        "playground", PLAYGROUND_COLLECTION, "detailed")  # fmt: skip


def test_failures_are_logged(tmp_path: Path) -> None:
    settings = on(tmp_path)

    query_id = record_failure(settings, "q?", ValueError("bad"), total_ms=3, style="concise")

    saved = ObserveStore(settings.observe_db).get_query(query_id or "")
    assert saved is not None
    assert saved.status == "error"


def test_a_rating_is_saved_for_a_logged_answer(tmp_path: Path) -> None:
    settings = on(tmp_path)
    query_id = record_answer(settings, answer(), retrieve_ms=5, total_ms=30, style="concise")

    assert save_rating(settings, query_id, 0) is True
    [saved] = ObserveStore(settings.observe_db).feedback()
    assert (saved.query_id, saved.rating, saved.source) == (query_id, 0, "playground")


def test_rating_an_unlogged_answer_does_nothing(tmp_path: Path) -> None:
    settings = on(tmp_path)

    assert save_rating(settings, None, 1) is False
    assert save_rating(settings, "never-logged", 1) is False
    assert save_rating(Settings(_env_file=None, observe_enabled=False), "x", 1) is False


def test_logging_off_records_nothing(tmp_path: Path) -> None:
    off = Settings(_env_file=None, observe_enabled=False, observe_db=tmp_path / "rag.db")

    assert record_answer(off, answer(), retrieve_ms=1, total_ms=2, style="concise") is None
    assert not (tmp_path / "rag.db").exists()


def test_a_turn_keeps_its_query_id() -> None:
    assert Turn.__dataclass_fields__["query_id"].default is None
