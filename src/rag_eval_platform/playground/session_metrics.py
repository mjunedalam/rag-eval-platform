"""How the conversation is going: one point per question, and a session summary.

Built from the chat history (``chat.Turn``), so it costs nothing extra: speed, instant
answer health (see answer_metrics.py), refusals and your 👍/👎 feedback.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rag_eval_platform.playground.answer_metrics import answer_health
from rag_eval_platform.playground.chat import Turn
from rag_eval_platform.playground.live import LiveStats


@dataclass(frozen=True)
class SessionPoint:
    number: int
    question: str
    total_s: float  # search + generation
    tokens_per_s: float | None
    coverage: float | None
    grounding: float | None
    refused: bool
    feedback: int | None  # 1 = 👍, 0 = 👎


@dataclass(frozen=True)
class SessionSummary:
    questions: int
    avg_total_s: float | None
    avg_coverage: float | None
    avg_grounding: float | None
    refusals: int
    thumbs_up: int
    thumbs_down: int


def session_points(history: Sequence[Turn]) -> tuple[SessionPoint, ...]:
    points = []
    for number, turn in enumerate(history, start=1):
        answer = turn.trace.answer
        health = answer_health(turn.trace, turn.first_token_s)
        speed = LiveStats(tokens=answer.output_tokens, generate_s=answer.latency_ms / 1000,
                          first_token_s=turn.first_token_s).tokens_per_s  # fmt: skip
        points.append(SessionPoint(
            number=number, question=answer.question,
            total_s=(turn.trace.retrieval_ms + answer.latency_ms) / 1000,
            tokens_per_s=speed, coverage=health.citation_coverage,
            grounding=health.grounding, refused=answer.is_refusal, feedback=turn.feedback,
        ))  # fmt: skip
    return tuple(points)


def session_summary(points: Sequence[SessionPoint]) -> SessionSummary:
    return SessionSummary(
        questions=len(points),
        avg_total_s=_mean([p.total_s for p in points]),
        avg_coverage=_mean([p.coverage for p in points]),
        avg_grounding=_mean([p.grounding for p in points]),
        refusals=sum(p.refused for p in points),
        thumbs_up=sum(p.feedback == 1 for p in points),
        thumbs_down=sum(p.feedback == 0 for p in points),
    )


def session_kpis(summary: SessionSummary) -> tuple[str, ...]:
    """Short labels for the header, e.g. ("3 questions", "avg 3.3 s", "coverage 50%")."""
    if not summary.questions:
        return ()
    kpis = [f"{summary.questions} question{'s' if summary.questions != 1 else ''}"]
    if summary.avg_total_s is not None:
        kpis.append(f"avg {summary.avg_total_s:.1f} s")
    if summary.avg_coverage is not None:
        kpis.append(f"coverage {summary.avg_coverage:.0%}")
    if summary.thumbs_up or summary.thumbs_down:
        kpis.append(f"👍 {summary.thumbs_up} · 👎 {summary.thumbs_down}")
    return tuple(kpis)


def _mean(values: Sequence[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None
