"""Pure data behind the playground's per-question visuals: similarity and re-ranking,
the prompt, the answer's citations, timings and the judge's verdicts."""

import html
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal

from rag_eval_platform.generation.generator import CITATION_PATTERN, Answer
from rag_eval_platform.generation.prompt_templates import SYSTEM_PROMPT
from rag_eval_platform.playground.core import JudgeResult, QueryTrace
from rag_eval_platform.playground.streaming import fade_markdown_text, fade_style, progress_at
from rag_eval_platform.playground.visuals import dot_text
from rag_eval_platform.retrieval.vector_store import SearchResult

SOURCE_COLORS = (
    "#2563eb", "#16a34a", "#d97706", "#9333ea", "#0891b2",
    "#db2777", "#65a30d", "#ea580c", "#4f46e5", "#0d9488",
)  # fmt: skip
INVALID_COLOR = "#dc2626"
YES_COLOR = "#16a34a"


def source_color(number: int) -> str:
    """The colour of source ``[number]``, shared by the answer and the source list."""
    return SOURCE_COLORS[(number - 1) % len(SOURCE_COLORS)]


def origin(result: SearchResult) -> str:
    chunk = result.chunk
    return chunk.doc_id if chunk.page is None else f"{chunk.doc_id}, page {chunk.page}"


@dataclass(frozen=True)
class SimilarityRow:
    rank: int
    chunk_id: str
    score: float  # cosine similarity from vector search
    kept: bool  # sent to the model
    cited: bool  # cited in the answer


def similarity_rows(trace: QueryTrace) -> tuple[SimilarityRow, ...]:
    kept = {r.chunk.id: n for n, r in enumerate(trace.final_results, start=1)}
    cited = {c.number for c in trace.answer.citations}
    return tuple(
        SimilarityRow(rank, r.chunk.id, r.score, r.chunk.id in kept, kept.get(r.chunk.id) in cited)
        for rank, r in enumerate(trace.vector_results, start=1)
    )


def was_reranked(trace: QueryTrace) -> bool:
    """True when the chunks sent to the model are not simply the top of the vector ranking."""
    final = [r.chunk.id for r in trace.final_results]
    return [r.chunk.id for r in trace.vector_results[: len(final)]] != final


@dataclass(frozen=True)
class RerankMove:
    chunk_id: str
    before: int  # rank after vector search
    after: int | None  # rank after the cross-encoder; None = dropped


def rerank_moves(trace: QueryTrace) -> tuple[RerankMove, ...]:
    after = {r.chunk.id: n for n, r in enumerate(trace.final_results, start=1)}
    return tuple(
        RerankMove(r.chunk.id, n, after.get(r.chunk.id))
        for n, r in enumerate(trace.vector_results, start=1)
    )


def funnel_stages(total_chunks: int, trace: QueryTrace) -> tuple[tuple[str, int], ...]:
    cited = len({c.number for c in trace.answer.citations})
    return (
        ("all chunks", total_chunks),
        ("candidates", len(trace.vector_results)),
        ("sent to the model", len(trace.final_results)),
        ("cited", cited),
    )


@dataclass(frozen=True)
class PromptBlock:
    kind: Literal["rules", "source", "question"]
    title: str
    text: str


def prompt_blocks(trace: QueryTrace) -> tuple[PromptBlock, ...]:
    """The prompt in the order the model reads it: rules, numbered sources, question."""
    sources = [
        PromptBlock("source", f"[{n}] {origin(r)}", r.chunk.text)
        for n, r in enumerate(trace.final_results, start=1)
    ]
    return (
        PromptBlock("rules", "System rules", SYSTEM_PROMPT),
        *sources,
        PromptBlock("question", "Question", trace.answer.question),
    )


@dataclass(frozen=True)
class AnswerSpan:
    text: str
    number: int | None  # the cited source, None for plain text
    valid: bool  # False for a citation to a source that was never given


def answer_spans(answer: Answer) -> tuple[AnswerSpan, ...]:
    return text_spans(answer.text, len(answer.sources))


def text_spans(text: str, sources: int) -> tuple[AnswerSpan, ...]:
    """Split text (a whole answer or one still streaming) into plain text and citations."""
    spans: list[AnswerSpan] = []
    position = 0
    for match in CITATION_PATTERN.finditer(text):
        if match.start() > position:
            spans.append(AnswerSpan(text[position : match.start()], None, True))
        for part in match.group(1).split(","):
            number = int(part)
            spans.append(AnswerSpan(f"[{number}]", number, 1 <= number <= sources))
        position = match.end()
    if position < len(text):
        spans.append(AnswerSpan(text[position:], None, True))
    return tuple(spans)


def _chip(span: AnswerSpan) -> str:
    if span.valid and span.number is not None:
        return (
            f'<span style="background:{source_color(span.number)};color:#fff;'
            f'border-radius:4px;padding:0 4px;font-weight:600">{span.text}</span>'
        )
    return (
        f'<span title="no such source" style="background:{INVALID_COLOR};color:#fff;'
        f'border-radius:4px;padding:0 4px;text-decoration:line-through">{span.text}</span>'
    )


def spans_html(spans: tuple[AnswerSpan, ...]) -> str:
    """The answer as HTML: text escaped, each citation a chip in its source's colour."""
    return "".join(
        html.escape(span.text).replace("\n", "<br>") if span.number is None else _chip(span)
        for span in spans
    )


def spans_markdown(spans: tuple[AnswerSpan, ...], levels: Sequence[tuple[int, float]] = ()) -> str:
    """The answer as Markdown (headings, lists, tables) with coloured citation chips.

    Only ``&`` and ``<`` are escaped: that is enough to keep any HTML from the model inert,
    while Markdown syntax, including ``>`` for quotes, still renders. ``levels`` fades the
    newest words while an answer is being written (see streaming.fade_levels).
    """
    out, position = [], 0
    for span in spans:
        if span.number is None:
            out.append(fade_markdown_text(span.text, position, levels))
        else:
            progress = progress_at(position, levels)
            chip = _chip(span)
            out.append(chip if progress is None or progress >= 1
                       else f'<span style="{fade_style(progress)}">{chip}</span>')  # fmt: skip
        position += len(span.text)
    return "".join(out)


def timing_rows(trace: QueryTrace, judged: JudgeResult | None) -> tuple[tuple[str, float], ...]:
    rows = [("retrieve", trace.retrieval_ms), ("generate", trace.answer.latency_ms)]
    if judged is not None:
        rows.append(("judge", judged.judge_ms))
    return tuple(rows)


def verdict_dot(result: JudgeResult) -> str:
    """Graphviz source: each cited sentence linked to its source, green = supported."""
    verdicts = result.citations.verdicts
    lines = [
        "digraph verdicts {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box style="rounded,filled" fillcolor="#f8fafc" color="#cbd5e1" '
        'fontname="Helvetica" fontsize=10];',
    ]
    if not verdicts:
        lines += ['  none [label="No citations to check"];', "}"]
        return "\n".join(lines)
    sentences = list(dict.fromkeys(v.sentence for v in verdicts))
    for i, sentence in enumerate(sentences):
        lines.append(f'  s{i} [label="{dot_text(sentence)}"];')
    for number, doc_id in dict.fromkeys((v.number, v.doc_id) for v in verdicts):
        lines.append(
            f'  src{number} [label="[{number}] {dot_text(doc_id, 40)}" '
            f'color="{source_color(number)}" penwidth=2];'
        )
    for v in verdicts:
        colour = YES_COLOR if v.supported else INVALID_COLOR
        style = "solid" if v.supported else "dashed"
        word = "supports" if v.supported else "does not support"
        lines.append(
            f"  s{sentences.index(v.sentence)} -> src{v.number} "
            f'[color="{colour}" style={style} label="{word}" dir=back];'
        )
    lines.append("}")
    return "\n".join(lines)


def hit_vs_faithfulness(
    retrieval: Mapping[str, Any], generation: Mapping[str, Any]
) -> tuple[tuple[str, float, int], ...]:
    """Mean faithfulness of answers whose retrieval found every relevant doc vs missed one."""
    recall = {e["example_id"]: e["scores"]["recall"] for e in retrieval.get("examples", [])}
    groups: dict[str, list[float]] = {"found every relevant doc": [], "missed a relevant doc": []}
    for example in generation.get("examples", []):
        faithfulness = example["scores"].get("faithfulness")
        example_recall = recall.get(example["example_id"])
        if faithfulness is None or example_recall is None:
            continue
        key = "found every relevant doc" if example_recall == 1.0 else "missed a relevant doc"
        groups[key].append(float(faithfulness))
    return tuple((label, fmean(values), len(values)) for label, values in groups.items() if values)
