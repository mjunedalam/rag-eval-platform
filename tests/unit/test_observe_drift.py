"""Tests for observability.drift: last N answers against the N before."""

from tests.unit.observe_records import make_feedback, make_query, make_run

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.drift import (
    DriftThresholds,
    drift_report,
    percentile,
)
from rag_eval_platform.observability.store import QueryRecord

T = DriftThresholds(window=4)


def windows(previous: list[QueryRecord], recent: list[QueryRecord]) -> list[QueryRecord]:
    """Renumber so the previous window is older than the recent one."""
    rows = [*previous, *recent]
    return [make_query(n, **{k: getattr(q, k) for k in ("status", "total_ms", "refusal",
                                                          "invalid_citations")})
            for n, q in enumerate(rows)]  # fmt: skip


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 0.95) is None
    assert percentile([5.0], 0.5) == 5.0
    assert percentile([float(n) for n in range(1, 21)], 0.95) == 19.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.0


def test_thresholds_come_from_settings() -> None:
    t = DriftThresholds.from_settings(Settings(_env_file=None, drift_window=7, gate_max_drop=0.05))

    assert (t.window, t.max_drop, t.latency_rise) == (7, 0.05, 0.5)


def test_not_enough_data_gives_no_traffic_warnings() -> None:
    report = drift_report([make_query(n, total_ms=99999.0) for n in range(7)], [], [], T)

    assert not report.enough_data
    assert (report.have, report.need) == (7, 8)
    assert report.warnings == ()


def test_steady_traffic_gives_no_warnings() -> None:
    report = drift_report([make_query(n) for n in range(8)], [], [], T)

    assert report.enough_data
    assert report.warnings == ()


def test_latency_rise_warns_with_both_values() -> None:
    rows = windows([make_query(total_ms=2000.0)] * 4, [make_query(total_ms=3100.0)] * 4)

    [warning] = drift_report(rows, [], [], T).warnings

    assert warning == "p95 latency rose from 2.0 s to 3.1 s (last 4 vs previous 4 answers)."


def test_latency_at_exactly_the_limit_does_not_warn() -> None:
    rows = windows([make_query(total_ms=2000.0)] * 4, [make_query(total_ms=3000.0)] * 4)

    assert drift_report(rows, [], [], T).warnings == ()


def test_error_rate_rise_warns_at_the_threshold() -> None:
    ok, bad = make_query(), make_query(status="error", total_ms=None)
    t = DriftThresholds(window=10, error_rise=0.10)
    rows = windows([ok] * 10, [bad] + [ok] * 9)

    assert drift_report(rows, [], [], t).warnings == (
        "Error rate rose from 0% to 10% (last 10 vs previous 10 answers).",
    )


def test_refusal_and_invalid_citation_rises_warn() -> None:
    ok = make_query()
    rows = windows([ok] * 4, [make_query(refusal=True), make_query(invalid_citations=(9,)), ok, ok])

    warnings = drift_report(rows, [], [], T).warnings

    assert "Refusal rate rose from 0% to 25% (last 4 vs previous 4 answers)." in warnings
    assert "Invalid-citation rate rose from 0% to 25% (last 4 vs previous 4 answers)." in warnings


def test_windows_full_of_errors_do_not_crash() -> None:
    bad = make_query(status="error", total_ms=None)

    report = drift_report(windows([bad] * 4, [bad] * 4), [], [], T)

    assert report.enough_data
    assert report.warnings == ()  # 100% -> 100% is no rise; no latency, no refusal data


def test_downvote_rise_needs_enough_ratings() -> None:
    rows = [make_query(n) for n in range(10)]
    t = DriftThresholds(window=5)
    up_before = [make_feedback(f"q{n}", 1) for n in range(5)]
    down_now = [make_feedback(f"q{n}", 0) for n in range(5, 10)]

    assert drift_report(rows, [*up_before, *down_now], [], t).warnings == (
        "👎 rate rose from 0% to 100% (last 5 vs previous 5 answers).",
    )
    assert drift_report(rows, [*up_before, *down_now[:4]], [], t).warnings == ()


def test_total_overrides_have_but_not_the_windowed_warnings() -> None:
    report = drift_report([make_query(n) for n in range(3)], [], [], T, total=500)

    assert (report.have, report.need) == (500, 8)
    assert report.enough_data  # "have" comes from total, not from the 3 rows passed in
    assert report.warnings == ()  # too few rows were passed to compute the traffic windows


def test_total_defaults_to_the_length_of_queries_when_not_given() -> None:
    report = drift_report([make_query(n) for n in range(3)], [], [], T)

    assert report.have == 3


def test_evaluation_failures_and_drops_warn_even_without_traffic() -> None:
    runs = [
        make_run("retrieval", True, {"recall": 0.93, "mrr": 0.88}, minutes=1),
        make_run("retrieval", True, {"recall": 0.90, "mrr": 0.88}, minutes=2),
        make_run("gate", False, {"recall": 0.9}, minutes=3),
    ]

    warnings = drift_report([], [], runs, T).warnings

    assert warnings == (
        "Retrieval Recall@k fell from 0.93 to 0.90 since the previous run.",
        "The latest gate run failed (2026-09-29 12:03 UTC).",
    )
