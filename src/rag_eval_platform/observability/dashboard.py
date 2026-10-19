"""Data for the ⑧ Observe tab: tiles, time buckets, evaluation history and table rows.

Pure functions over store records, so they are tested without Streamlit.
"""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.cost import Prices
from rag_eval_platform.observability.drift import percentile
from rag_eval_platform.observability.review import latest_feedback
from rag_eval_platform.observability.store import (
    TRIAD_METRICS,
    EvalRunRecord,
    FeedbackRecord,
    QueryRecord,
    TriadScoreRecord,
)

RANGES: dict[str, timedelta | None] = {
    "24 h": timedelta(hours=24),
    "7 days": timedelta(days=7),
    "30 days": timedelta(days=30),
    "All": None,
}
EVAL_METRICS: dict[str, str] = {
    "recall": "Recall@k",
    "mrr": "MRR",
    "ndcg": "NDCG@k",
    "faithfulness": "Faithfulness",
    "answer_relevance": "Answer relevance",
}
TRIAD_LABELS: dict[str, str] = {
    "context_relevance": "Context relevance",
    "groundedness": "Groundedness",
    "answer_relevance": "Answer relevance",
}
# Short column names for the recent-queries table, in TRIAD_METRICS order.
_TRIAD_COLUMNS = dict(zip(TRIAD_METRICS, ("context", "grounded", "relevant"), strict=True))
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def since_for(label: str, now: datetime) -> datetime | None:
    span = RANGES[label]
    return None if span is None else now - span


def bucket_size(label: str) -> timedelta:
    return timedelta(hours=1) if label == "24 h" else timedelta(days=1)


@dataclass(frozen=True)
class Summary:
    queries: int
    p50_ms: float | None
    p95_ms: float | None
    error_rate: float | None
    refusal_rate: float | None  # among answered (ok) queries
    thumbs_up_rate: float | None  # among rated queries
    rated: int
    tokens: int
    cost_usd: float


def _latencies(queries: Sequence[QueryRecord]) -> list[float]:
    return [q.total_ms for q in queries if q.status == "ok" and q.total_ms is not None]


def _tokens(q: QueryRecord) -> int:
    return (q.input_tokens or 0) + (q.output_tokens or 0)


def summarize(queries: Sequence[QueryRecord], feedback: Sequence[FeedbackRecord]) -> Summary:
    ok = [q for q in queries if q.status == "ok"]
    ids = {q.id for q in queries}
    rated = [f.rating for f in latest_feedback(feedback).values() if f.query_id in ids]
    latencies = _latencies(queries)
    return Summary(
        queries=len(queries),
        p50_ms=percentile(latencies, 0.5),
        p95_ms=percentile(latencies, 0.95),
        error_rate=(len(queries) - len(ok)) / len(queries) if queries else None,
        refusal_rate=sum(q.refusal for q in ok) / len(ok) if ok else None,
        thumbs_up_rate=rated.count(1) / len(rated) if rated else None,
        rated=len(rated),
        tokens=sum(_tokens(q) for q in queries),
        cost_usd=sum(q.cost_usd for q in queries),
    )


def cost_label(summary: Summary, prices: Prices) -> str:
    return "local · $0" if prices.is_free else f"${summary.cost_usd:.4f}"


@dataclass(frozen=True)
class Bucket:
    start: datetime
    counts: Mapping[str, int]  # queries per source
    p50_ms: float | None
    p95_ms: float | None
    tokens: int
    cost_usd: float


def buckets(queries: Sequence[QueryRecord], size: timedelta) -> list[Bucket]:
    groups: dict[datetime, list[QueryRecord]] = {}
    for q in queries:
        start = _EPOCH + ((q.ts - _EPOCH) // size) * size
        groups.setdefault(start, []).append(q)
    return [
        Bucket(
            start,
            dict(Counter(q.source for q in rows)),
            percentile(_latencies(rows), 0.5),
            percentile(_latencies(rows), 0.95),
            sum(_tokens(q) for q in rows),
            sum(q.cost_usd for q in rows),
        )
        for start, rows in sorted(groups.items())
    ]


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def triad_means(scores: Sequence[TriadScoreRecord]) -> dict[str, float | None]:
    """Each triad metric's average over the scored answers (``None`` when none is scored)."""
    return {m: _mean([s.value for s in scores if s.metric == m]) for m in TRIAD_METRICS}


@dataclass(frozen=True)
class TriadBucket:
    start: datetime
    means: Mapping[str, float | None]


def triad_buckets(queries: Sequence[QueryRecord], scores: Sequence[TriadScoreRecord],
                  size: timedelta) -> list[TriadBucket]:  # fmt: skip
    """Triad averages per time bucket, by when each scored answer was asked."""
    asked = {q.id: q.ts for q in queries}
    groups: dict[datetime, list[TriadScoreRecord]] = {}
    for score in scores:
        if score.query_id in asked:
            start = _EPOCH + ((asked[score.query_id] - _EPOCH) // size) * size
            groups.setdefault(start, []).append(score)
    return [TriadBucket(start, triad_means(rows)) for start, rows in sorted(groups.items())]


@dataclass(frozen=True)
class EvalPoint:
    ts: datetime
    kind: str
    value: float


def eval_history(runs: Sequence[EvalRunRecord]) -> dict[str, list[EvalPoint]]:
    """Metric -> its values over time (retrieval metrics from retrieval and gate runs)."""
    return {
        metric: [EvalPoint(r.ts, r.kind, r.metrics[metric]) for r in runs if metric in r.metrics]
        for metric in EVAL_METRICS
    }


def baseline_values(baselines_dir: Path) -> dict[str, float]:
    """The committed baselines' overall scores, as reference lines (unreadable files skipped)."""
    values: dict[str, float] = {}
    for name in ("retrieval.json", "generation.json"):
        try:
            overall = json.loads((baselines_dir / name).read_text(encoding="utf-8"))["report"][
                "overall"
            ]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        values |= {k: float(v) for k, v in overall.items() if k in EVAL_METRICS and v is not None}
    return values


def metric_thresholds(settings: Settings) -> dict[str, float]:
    return {
        "recall": settings.min_recall_at_k,
        "mrr": settings.min_mrr,
        "ndcg": settings.min_ndcg_at_k,
        "faithfulness": settings.min_faithfulness,
        "answer_relevance": settings.min_answer_relevance,
    }


def query_rows(
    queries: Sequence[QueryRecord],
    feedback: Sequence[FeedbackRecord],
    source: str | None = None,
    scores: Sequence[TriadScoreRecord] = (),
) -> list[dict[str, object]]:
    """Rows for the recent-queries table, newest first; triad cells are blank while pending."""
    ratings = latest_feedback(feedback)
    triad = {(s.query_id, s.metric): round(s.value, 2) for s in scores}
    rows = [q for q in queries if source is None or q.source == source]
    return [
        {
            "id": q.id,
            "time": f"{q.ts:%Y-%m-%d %H:%M}",
            "source": q.source,
            "question": (q.question if len(q.question) <= 80 else q.question[:79] + "…"),
            "status": "refused" if q.refusal else q.status,
            "seconds": None if q.total_ms is None else round(q.total_ms / 1000, 1),
            "tokens": _tokens(q),
            "rating": ({1: "👍", 0: "👎"}.get(ratings[q.id].rating, "") if q.id in ratings else ""),
            **{column: triad.get((q.id, metric)) for metric, column in _TRIAD_COLUMNS.items()},
        }
        for q in sorted(rows, key=lambda q: q.ts, reverse=True)
    ]
