"""The review queue: answers that deserve a human look, with the reasons why."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from rag_eval_platform.observability.store import FeedbackRecord, QueryRecord

Reason = Literal["thumbs_down", "error", "refused", "invalid_citation", "slow"]
REASON_LABELS: dict[str, str] = {
    "thumbs_down": "👎 thumbs down", "error": "error", "refused": "refused",
    "invalid_citation": "invalid citation", "slow": "slow",
}  # fmt: skip


@dataclass(frozen=True)
class ReviewItem:
    query: QueryRecord
    reasons: tuple[Reason, ...]
    comment: str | None  # the 👎 comment, if any


def latest_feedback(feedback: Sequence[FeedbackRecord]) -> dict[str, FeedbackRecord]:
    """The newest rating per query (a re-rating replaces the shown one; history is kept)."""
    return {f.query_id: f for f in sorted(feedback, key=lambda f: (f.ts, f.id))}


def _reasons(
    query: QueryRecord, rating: FeedbackRecord | None, slow_ms: float
) -> tuple[Reason, ...]:
    checks: tuple[tuple[Reason, bool], ...] = (
        ("thumbs_down", rating is not None and rating.rating == 0),
        ("error", query.status == "error"),
        ("refused", query.refusal),
        ("invalid_citation", bool(query.invalid_citations)),
        ("slow", query.total_ms is not None and query.total_ms > slow_ms),
    )
    return tuple(reason for reason, hit in checks if hit)


def review_queue(queries: Sequence[QueryRecord], feedback: Sequence[FeedbackRecord], *,
                 slow_ms: float, include_reviewed: bool = False) -> list[ReviewItem]:  # fmt: skip
    """👎 first, then errors, then the rest; newest first within each group."""
    latest = latest_feedback(feedback)
    items = [
        ReviewItem(q, reasons, latest[q.id].comment if q.id in latest else None)
        for q in queries
        if (include_reviewed or q.reviewed_at is None)
        and (reasons := _reasons(q, latest.get(q.id), slow_ms))
    ]

    def group(item: ReviewItem) -> int:
        return 0 if "thumbs_down" in item.reasons else 1 if "error" in item.reasons else 2

    return sorted(items, key=lambda i: (group(i), -i.query.ts.timestamp()))
