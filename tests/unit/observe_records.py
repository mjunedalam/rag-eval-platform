"""Record factories for observability tests."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from rag_eval_platform.observability.store import (
    EvalRunRecord,
    FeedbackRecord,
    QueryRecord,
    TriadScoreRecord,
)

T0 = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)


def make_query(n: int = 0, **changes: object) -> QueryRecord:
    """Query number ``n``, one minute after query ``n - 1``; override any field."""
    fields: dict[str, object] = {
        "id": f"q{n}", "ts": T0 + timedelta(minutes=n), "source": "api",
        "collection": "rag_documents", "question": f"Question {n}?", "answer": "Answer [1].",
        "style": "concise", "prompt_version": "v1", "model": "qwen3:8b", "refusal": False,
        "retrieved_chunk_ids": ("a.md#0", "b.md#1"), "cited_doc_ids": ("a.md",),
        "invalid_citations": (), "input_tokens": 300, "output_tokens": 40,
        "retrieve_ms": 200.0, "generate_ms": 1800.0, "total_ms": 2000.0, "cost_usd": 0.0,
        "status": "ok",
    }  # fmt: skip
    fields.update(changes)
    return QueryRecord(**fields)  # type: ignore[arg-type]


def make_feedback(query_id: str, rating: int, minutes: int = 0,
                  comment: str | None = None) -> FeedbackRecord:  # fmt: skip
    return FeedbackRecord(id=f"f-{query_id}-{minutes}", query_id=query_id,
                          ts=T0 + timedelta(minutes=minutes), rating=rating, comment=comment,
                          source="playground")  # fmt: skip


def make_run(kind: str, passed: bool, metrics: Mapping[str, float],
             minutes: int = 0) -> EvalRunRecord:  # fmt: skip
    return EvalRunRecord(id=f"r-{kind}-{minutes}", ts=T0 + timedelta(minutes=minutes), kind=kind,
                         passed=passed, metrics=dict(metrics), fingerprint="abc123")  # fmt: skip


def make_triad(query_id: str, metric: str = "groundedness", value: float = 0.8,
               minutes: int = 0) -> TriadScoreRecord:  # fmt: skip
    return TriadScoreRecord(query_id=query_id, metric=metric, value=value,
                            reason=f"{metric} reason", judge="gemma3:4b",
                            ts=T0 + timedelta(minutes=minutes))  # fmt: skip
