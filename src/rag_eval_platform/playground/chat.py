"""Pure pieces of the playground chat: the steps behind each reply, starter suggestions,
the conversation history, and the source cards shown under a reply."""

import html
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import PurePosixPath

from rag_eval_platform.playground.core import QueryTrace
from rag_eval_platform.playground.live import LiveStats
from rag_eval_platform.playground.query_visuals import (
    origin,
    source_color,
    spans_markdown,
    text_spans,
    was_reranked,
)

_SUGGESTIONS = (
    "Summarise {name}",
    "What are the key ideas in {name}?",
    "Explain an important concept from {name}",
)


@dataclass(frozen=True)
class Step:
    """One thing the pipeline did for a reply, like a tool step in Claude Code."""

    label: str
    detail: str
    ms: float | None  # how long it took, when measured


@dataclass(frozen=True)
class Turn:
    """One question and its answer in the chat."""

    trace: QueryTrace
    first_token_s: float | None
    feedback: int | None = None  # 1 = thumbs up, 0 = thumbs down


@dataclass(frozen=True)
class SourceCard:
    number: int
    origin: str
    color: str
    cited: bool
    text: str


def run_steps(
    trace: QueryTrace, chunks: int | None, first_token_s: float | None
) -> tuple[Step, ...]:
    answer = trace.answer
    total = f"{chunks:,} chunks" if chunks is not None else "the index"
    steps = [
        Step(
            "Searched your documents",
            f"{total} → {len(trace.final_results)} sources",
            trace.retrieval_ms,
        ),
    ]
    if was_reranked(trace):
        steps.append(
            Step("Re-ranked the candidates",
                 f"{len(trace.vector_results)} candidates → top {len(trace.final_results)}", None)
        )  # fmt: skip
    speed = LiveStats(
        tokens=answer.output_tokens, generate_s=answer.latency_ms / 1000,
        first_token_s=first_token_s,
    ).tokens_per_s  # fmt: skip
    written = f"{answer.output_tokens} tokens" if answer.output_tokens is not None else "done"
    if speed is not None:
        written += f" · {speed:.0f} tok/s"
    steps.append(Step("Wrote the answer", written, answer.latency_ms))
    if answer.is_refusal:
        steps.append(Step("Found no answer in the sources", "the model refused to guess", None))
    else:
        cited = len({c.number for c in answer.citations})
        detail = f"{cited} of {len(trace.final_results)} sources cited"
        if answer.invalid_citations:
            detail += f" · {len(answer.invalid_citations)} invalid"
        steps.append(Step("Checked citations", detail, None))
    return tuple(steps)


def steps_summary(steps: Sequence[Step]) -> str:
    seconds = sum(s.ms for s in steps if s.ms is not None) / 1000
    return f"{len(steps)} steps · {seconds:.1f} s"


def suggestions(doc_ids: Sequence[str], limit: int = 3) -> tuple[str, ...]:
    """Starter questions built from the indexed documents' names."""
    if not doc_ids:
        return ()
    names = [PurePosixPath(doc).stem for doc in doc_ids]
    return tuple(
        _SUGGESTIONS[i % len(_SUGGESTIONS)].format(name=names[i % len(names)]) for i in range(limit)
    )


def add_turn(history: tuple[Turn, ...], turn: Turn) -> tuple[Turn, ...]:
    return (*history, turn)


def replace_turn(history: tuple[Turn, ...], index: int, turn: Turn) -> tuple[Turn, ...]:
    return (*history[:index], turn, *history[index + 1 :])


def with_feedback(history: tuple[Turn, ...], index: int, value: int | None) -> tuple[Turn, ...]:
    return replace_turn(history, index, replace(history[index], feedback=value))


def source_cards(trace: QueryTrace) -> tuple[SourceCard, ...]:
    cited = {c.number for c in trace.answer.citations}
    return tuple(
        SourceCard(n, origin(r), source_color(n), n in cited, r.chunk.text)
        for n, r in enumerate(trace.final_results, start=1)
    )


def user_html(question: str) -> str:
    """The user's message as a right-aligned bubble, the way Claude shows it."""
    return f'<div class="rag-user"><span>{html.escape(question)}</span></div>'


def streaming_markdown(text: str, sources: int) -> str:
    """An answer still being written, as Markdown with chips, ending in a soft cursor."""
    return spans_markdown(text_spans(text, sources)) + '<span class="rag-cursor">●</span>'


def cited_sources_html(trace: QueryTrace) -> str:
    """A visible "Sources" line under the answer: each cited source in its chip colour."""
    items = [
        f'<span class="rag-source"><b style="color:{card.color}">[{card.number}]</b> '
        f"{html.escape(card.origin)}</span>"
        for card in source_cards(trace)
        if card.cited
    ]
    if not items:
        return ""
    return (
        '<div class="rag-sources"><span class="label">Sources</span> ' + " ".join(items) + "</div>"
    )
