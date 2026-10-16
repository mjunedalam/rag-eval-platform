"""Tests for observability.review."""

from tests.unit.observe_records import T0, make_feedback, make_query

from rag_eval_platform.observability.review import latest_feedback, review_queue


def test_the_latest_rating_wins() -> None:
    history = [make_feedback("q1", 1, minutes=1), make_feedback("q1", 0, minutes=5)]

    assert latest_feedback(history)["q1"].rating == 0


def test_each_reason_puts_an_answer_in_the_queue() -> None:
    queries = [
        make_query(1),  # fine: not queued
        make_query(2, refusal=True),
        make_query(3, invalid_citations=(9,)),
        make_query(4, status="error", error="GenerationError: down", total_ms=10.0),
        make_query(5, total_ms=45000.0),
        make_query(6),
    ]

    items = review_queue(queries, [make_feedback("q6", 0, comment="wrong")], slow_ms=30000)

    reasons = {item.query.id: item.reasons for item in items}
    assert reasons == {"q2": ("refused",), "q3": ("invalid_citation",), "q4": ("error",),
                       "q5": ("slow",), "q6": ("thumbs_down",)}  # fmt: skip
    assert next(i for i in items if i.query.id == "q6").comment == "wrong"


def test_thumbs_down_first_then_errors_then_newest() -> None:
    queries = [make_query(1, refusal=True), make_query(2, status="error", total_ms=None),
               make_query(3, refusal=True), make_query(4)]  # fmt: skip

    items = review_queue(queries, [make_feedback("q4", 0)], slow_ms=30000)

    assert [i.query.id for i in items] == ["q4", "q2", "q3", "q1"]


def test_reviewed_answers_leave_the_queue_unless_asked_for() -> None:
    queries = [make_query(1, refusal=True, reviewed_at=T0)]

    assert review_queue(queries, [], slow_ms=30000) == []
    assert len(review_queue(queries, [], slow_ms=30000, include_reviewed=True)) == 1


def test_a_thumbs_up_after_a_thumbs_down_clears_it() -> None:
    history = [make_feedback("q1", 0, minutes=1), make_feedback("q1", 1, minutes=2)]

    assert review_queue([make_query(1)], history, slow_ms=30000) == []
