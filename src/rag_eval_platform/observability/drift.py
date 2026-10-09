"""Drift: is recent traffic worse than the traffic just before it?

Windows count answers, not hours (local traffic is light): the last ``window`` answers
against the ``window`` before them. Evaluation runs are checked on their own.
"""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.review import latest_feedback
from rag_eval_platform.observability.store import EvalRunRecord, FeedbackRecord, QueryRecord

MIN_RATED = 5  # ratings needed in each window before the 👎 rate is compared
_RETRIEVAL_METRICS = {"recall": "Recall@k", "mrr": "MRR", "ndcg": "NDCG@k"}


@dataclass(frozen=True)
class DriftThresholds:
    window: int = 20
    latency_rise: float = 0.5  # relative: p95 may grow by 50%
    error_rise: float = 0.10  # absolute: 10 points
    refusal_rise: float = 0.15
    citation_rise: float = 0.10
    downvote_rise: float = 0.20
    max_drop: float = 0.02  # a retrieval metric's allowed fall between evaluation runs

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            settings.drift_window,
            settings.drift_latency_rise,
            settings.drift_error_rise,
            settings.drift_refusal_rise,
            settings.drift_citation_rise,
            settings.drift_downvote_rise,
            settings.gate_max_drop,
        )


@dataclass(frozen=True)
class DriftReport:
    warnings: tuple[str, ...]
    have: int  # answers logged
    need: int  # answers needed for traffic checks (2 x window)

    @property
    def enough_data(self) -> bool:
        return self.have >= self.need


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile; None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, math.ceil(q * len(ordered))) - 1]


def _rate(rows: Sequence[QueryRecord], hit: Callable[[QueryRecord], bool]) -> float | None:
    return sum(1 for r in rows if hit(r)) / len(rows) if rows else None


def _rose(before: float | None, now: float | None, rise: float) -> bool:
    return before is not None and now is not None and round(now - before, 9) >= rise


def _traffic(
    previous: list[QueryRecord],
    recent: list[QueryRecord],
    feedback: Sequence[FeedbackRecord],
    t: DriftThresholds,
) -> list[str]:
    span = f"(last {t.window} vs previous {t.window} answers)."
    warnings = []

    def p95(rows: list[QueryRecord]) -> float | None:
        values = [r.total_ms for r in rows if r.status == "ok" and r.total_ms is not None]
        return percentile(values, 0.95)

    before, now = p95(previous), p95(recent)
    if before is not None and now is not None and now > before * (1 + t.latency_rise):
        warnings.append(f"p95 latency rose from {before / 1000:.1f} s to {now / 1000:.1f} s {span}")

    ratings = latest_feedback(feedback)

    def ok(rows: list[QueryRecord]) -> list[QueryRecord]:
        return [r for r in rows if r.status == "ok"]

    def downvote_rate(rows: list[QueryRecord]) -> float | None:
        rated = [ratings[r.id].rating for r in rows if r.id in ratings]
        return rated.count(0) / len(rated) if len(rated) >= MIN_RATED else None

    rates = (
        ("Error rate", _rate(previous, lambda r: r.status == "error"),
         _rate(recent, lambda r: r.status == "error"), t.error_rise),
        ("Refusal rate", _rate(ok(previous), lambda r: r.refusal),
         _rate(ok(recent), lambda r: r.refusal), t.refusal_rise),
        ("Invalid-citation rate", _rate(ok(previous), lambda r: bool(r.invalid_citations)),
         _rate(ok(recent), lambda r: bool(r.invalid_citations)), t.citation_rise),
        ("👎 rate", downvote_rate(previous), downvote_rate(recent), t.downvote_rise),
    )  # fmt: skip
    for label, before_rate, now_rate, rise in rates:
        if before_rate is not None and now_rate is not None and _rose(before_rate, now_rate, rise):
            warnings.append(f"{label} rose from {before_rate:.0%} to {now_rate:.0%} {span}")
    return warnings


def _evaluation(runs: Sequence[EvalRunRecord], max_drop: float) -> list[str]:
    warnings = []
    for kind in ("retrieval", "generation", "gate"):
        of_kind = [r for r in runs if r.kind == kind]
        if not of_kind:
            continue
        latest = of_kind[-1]
        if len(of_kind) >= 2 and kind != "generation":
            before = of_kind[-2].metrics
            for key, label in _RETRIEVAL_METRICS.items():
                if (
                    key in before
                    and key in latest.metrics
                    and round(before[key] - latest.metrics[key], 9) > max_drop
                ):
                    before_val = before[key]
                    after_val = latest.metrics[key]
                    warnings.append(
                        f"{kind.capitalize()} {label} fell from {before_val:.2f} "
                        f"to {after_val:.2f} since the previous run."
                    )
        if not latest.passed:
            warnings.append(f"The latest {kind} run failed ({latest.ts:%Y-%m-%d %H:%M} UTC).")
    return warnings


def drift_report(
    queries: Sequence[QueryRecord],
    feedback: Sequence[FeedbackRecord],
    runs: Sequence[EvalRunRecord],
    t: DriftThresholds,
    total: int | None = None,
) -> DriftReport:
    """Warnings for recent traffic and evaluation runs (all inputs oldest first).

    ``queries`` may be a bounded, most-recent slice (callers can fetch just enough for the
    traffic windows); pass the true logged count as ``total`` so ``have`` still reports it,
    rather than the size of the slice. ``None`` falls back to ``len(queries)``.
    """
    need = 2 * t.window
    warnings: list[str] = []
    if len(queries) >= need:
        rows = list(queries[-need:])
        warnings += _traffic(rows[: t.window], rows[t.window :], feedback, t)
    warnings += _evaluation(runs, t.max_drop)
    have = len(queries) if total is None else total
    return DriftReport(tuple(warnings), have, need)
