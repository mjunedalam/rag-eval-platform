"""Tests for observability.dashboard: what the ⑧ Observe tab shows."""

import json
from datetime import timedelta
from pathlib import Path

import pytest
from tests.unit.observe_records import T0, make_feedback, make_query, make_run, make_triad

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.cost import Prices
from rag_eval_platform.observability.dashboard import (
    TRIAD_LABELS,
    baseline_values,
    bucket_size,
    buckets,
    cost_label,
    eval_history,
    metric_thresholds,
    query_rows,
    since_for,
    summarize,
    triad_buckets,
    triad_means,
)


def test_time_ranges() -> None:
    assert since_for("24 h", T0) == T0 - timedelta(hours=24)
    assert since_for("All", T0) is None
    assert bucket_size("24 h") == timedelta(hours=1)
    assert bucket_size("7 days") == timedelta(days=1)


def test_summary_of_an_empty_log() -> None:
    summary = summarize([], [])

    assert summary.queries == 0
    assert (summary.p50_ms, summary.p95_ms, summary.error_rate) == (None, None, None)
    assert summary.thumbs_up_rate is None


def test_summary_tiles() -> None:
    queries = [
        make_query(1, total_ms=1000.0),
        make_query(2, total_ms=3000.0, refusal=True),
        make_query(3, status="error", total_ms=None, input_tokens=None, output_tokens=None),
        make_query(4, total_ms=2000.0, cost_usd=0.5),
    ]
    feedback = [make_feedback("q1", 1), make_feedback("q2", 0), make_feedback("q4", 1)]

    s = summarize(queries, feedback)

    assert (s.queries, s.p50_ms, s.p95_ms) == (4, 2000.0, 3000.0)
    assert (s.error_rate, s.refusal_rate) == (0.25, 1 / 3)
    assert (s.thumbs_up_rate, s.rated) == (2 / 3, 3)
    assert (s.tokens, s.cost_usd) == (3 * 340, 0.5)


def test_cost_label() -> None:
    s = summarize([make_query(1, cost_usd=0.0123)], [])

    assert cost_label(s, Prices()) == "local · $0"
    assert cost_label(s, Prices(0.15, 0.6)) == "$0.0123"


def test_buckets_group_by_hour_and_source() -> None:
    queries = [make_query(1), make_query(2, source="playground"), make_query(70)]

    rows = buckets(queries, timedelta(hours=1))

    assert [b.start for b in rows] == [T0, T0 + timedelta(hours=1)]
    assert dict(rows[0].counts) == {"api": 1, "playground": 1}
    assert rows[0].p95_ms == 2000.0
    assert rows[1].tokens == 340


def test_eval_history_by_metric_and_the_baselines(tmp_path: Path) -> None:
    runs = [
        make_run("retrieval", True, {"recall": 0.93, "mrr": 0.88}, minutes=1),
        make_run("gate", True, {"recall": 0.9}, minutes=2),
        make_run("generation", True, {"faithfulness": 0.95}, minutes=3),
    ]
    (tmp_path / "retrieval.json").write_text(
        json.dumps(
            {"report": {"overall": {"recall": 0.93, "mrr": 0.88, "ndcg": 0.87, "precision": 0.3}}}
        )
    )
    (tmp_path / "generation.json").write_text("not json")

    history = eval_history(runs)

    assert [(p.kind, p.value) for p in history["recall"]] == [("retrieval", 0.93), ("gate", 0.9)]
    assert [p.value for p in history["faithfulness"]] == [0.95]
    assert baseline_values(tmp_path) == {"recall": 0.93, "mrr": 0.88, "ndcg": 0.87}
    assert baseline_values(tmp_path / "missing") == {}


def test_metric_thresholds_come_from_settings() -> None:
    assert metric_thresholds(Settings(_env_file=None)) == {
        "recall": 0.80,
        "mrr": 0.70,
        "ndcg": 0.70,
        "faithfulness": 0.85,
        "answer_relevance": 0.80,
    }


def test_query_rows_are_newest_first_and_filter_by_source() -> None:
    queries = [make_query(1, question="x" * 200), make_query(2, source="cli")]

    rows = query_rows(queries, [make_feedback("q1", 0)])

    assert [r["id"] for r in rows] == ["q2", "q1"]
    assert rows[1]["rating"] == "👎"
    assert len(str(rows[1]["question"])) <= 80
    assert [r["id"] for r in query_rows(queries, [], source="cli")] == ["q2"]


def test_triad_means_average_each_metric_over_scored_answers() -> None:
    scores = [
        make_triad("q1", "groundedness", 1.0), make_triad("q2", "groundedness", 0.5),
        make_triad("q1", "answer_relevance", 0.8),
    ]  # fmt: skip

    means = triad_means(scores)

    assert means == {"context_relevance": None, "groundedness": 0.75, "answer_relevance": 0.8}
    assert list(TRIAD_LABELS) == list(means)


def test_triad_buckets_follow_the_answers_time() -> None:
    queries = [make_query(0), make_query(1), make_query(90)]  # minutes after T0
    scores = [make_triad("q0", value=0.4), make_triad("q1", value=0.6),
              make_triad("q90", value=1.0)]  # fmt: skip

    rows = triad_buckets(queries, scores, timedelta(hours=1))

    assert [r.start for r in rows] == [T0, T0 + timedelta(hours=1)]
    assert [r.means["groundedness"] for r in rows] == [pytest.approx(0.5), 1.0]
    assert rows[0].means["context_relevance"] is None


def test_query_rows_show_triad_scores_blank_while_pending() -> None:
    queries = [make_query(1), make_query(2)]
    scores = [
        make_triad("q1", metric, value)
        for metric, value in (
            ("context_relevance", 0.666),
            ("groundedness", 1.0),
            ("answer_relevance", 0.9),
        )
    ]

    rows = {r["id"]: r for r in query_rows(queries, [], scores=scores)}

    assert (rows["q1"]["context"], rows["q1"]["grounded"], rows["q1"]["relevant"]) == (
        0.67, 1.0, 0.9
    )  # fmt: skip
    assert (rows["q2"]["context"], rows["q2"]["grounded"], rows["q2"]["relevant"]) == (
        None, None, None
    )  # fmt: skip
