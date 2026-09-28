"""Tests for playground.session_metrics: how the conversation is going, question by question."""

import pytest
from tests.unit.test_answer_metrics import trace

from rag_eval_platform.generation.prompt_templates import NO_ANSWER
from rag_eval_platform.playground.chat import Turn
from rag_eval_platform.playground.session_metrics import (
    SessionSummary,
    session_kpis,
    session_points,
    session_summary,
)

HISTORY = (
    Turn(trace("Mean reciprocal rank averages the inverse rank [1].", latency_ms=3000.0), 1.0,
         feedback=1),
    Turn(trace("Pasta needs salted boiling water always.", latency_ms=5000.0), 1.0, feedback=0),
    Turn(trace(NO_ANSWER, latency_ms=500.0), None),
)  # fmt: skip


def test_points_follow_the_conversation_in_order() -> None:
    points = session_points(HISTORY)

    assert [p.number for p in points] == [1, 2, 3]
    assert points[0].total_s == pytest.approx(3.5)  # 0.5 s search + 3.0 s generating
    assert points[0].coverage == 1.0
    assert points[1].coverage == 0.0  # a claim without a citation
    assert points[2].refused is True
    assert points[0].feedback == 1


def test_summary_averages_only_what_was_measured() -> None:
    summary = session_summary(session_points(HISTORY))

    assert summary.avg_total_s == pytest.approx((3.5 + 5.5 + 1.0) / 3)
    assert summary == SessionSummary(
        questions=3, avg_total_s=summary.avg_total_s, avg_coverage=0.5, avg_grounding=1.0,
        refusals=1, thumbs_up=1, thumbs_down=1,
    )  # fmt: skip


def test_empty_session() -> None:
    summary = session_summary(())

    assert summary.questions == 0
    assert summary.avg_total_s is None
    assert session_kpis(summary) == ()


def test_kpis_for_the_header() -> None:
    kpis = session_kpis(session_summary(session_points(HISTORY)))

    assert kpis == ("3 questions", "avg 3.3 s", "coverage 50%", "👍 1 · 👎 1")
