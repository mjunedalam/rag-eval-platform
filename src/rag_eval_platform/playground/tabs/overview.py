"""Overview: all five phases on one screen with no scrolling, updating live while a
question runs.

The layout (pipeline, stats tiles, six panels) is drawn first as empty slots. A newly asked
question then runs *here*: each slot fills as its phase finishes, the answer streams in with
a shimmering status line and a cursor, and the tiles tick up. Later reruns redraw the same
slots from the saved trace without replaying any animation.
"""

import time
from collections.abc import Iterable
from typing import Any

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.generation.generator import (
    Answer,
    GenerationError,
    Generator,
    OpenAICompatibleClient,
)
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.chat import Turn, add_turn, replace_turn, streaming_markdown
from rag_eval_platform.playground.core import (
    REPORTS_DIR,
    QueryTrace,
    load_reports,
    retrieve_step,
    trace_key,
)
from rag_eval_platform.playground.live import (
    LiveStats,
    Pace,
    pace,
    split_for_typing,
    stat_tiles,
    stats_from_trace,
    status_html,
    status_text,
    tail_for_display,
    tiles_html,
    typing_html,
)
from rag_eval_platform.playground.query_visuals import (
    answer_spans,
    similarity_rows,
    spans_html,
    timing_rows,
    was_reranked,
)
from rag_eval_platform.playground.shared import (
    FIRST_OUTPUT_WAIT,
    HISTORY,
    INDEX_VERSION,
    PENDING_QUESTION,
    PENDING_REPLACE,
    TRACE,
    TabContext,
    animated_chart,
    load_embedder,
    load_reranker,
    mark_shown,
    settings,
)
from rag_eval_platform.playground.visuals import (
    PipelineStats,
    meaning_map,
    pipeline_dot,
    size_bins,
)
from rag_eval_platform.retrieval.vector_store import VectorStoreError

PANEL_HEIGHT = 168  # each of the six panels (3 rows of 2); tuned to fit 1440x900 unscrolled
CHART_HEIGHT = PANEL_HEIGHT - 45
REFRESH_EVERY = 0.25  # seconds between live tile/timing updates while the answer streams
# Characters the answer panel shows while writing (about 4 lines), so the cursor stays in view.
LIVE_TAIL = 150
TITLES = (
    "① Ingest · chunk sizes",
    "② Embed · meaning map",
    "③ Retrieve · similarity",
    "④ Generate · answer",
    "⑤ Evaluate · scores",
    "⏱ Timing",
)


def render(ctx: TabContext) -> None:
    pipeline = st.empty()
    tiles = st.empty()
    cells = [
        column.container(height=PANEL_HEIGHT, border=True)
        for column in [*st.columns(2), *st.columns(2), *st.columns(2)]
    ]
    for cell, title in zip(cells, TITLES, strict=True):
        cell.markdown(f"**{title}**")
    slots = [cell.empty() for cell in cells]

    question = st.session_state.pop(PENDING_QUESTION, None)
    if question and ctx.store is not None and ctx.snapshot is not None:
        ctx = _run_live(question, ctx, pipeline, tiles, slots)
    else:
        _draw_static(ctx, pipeline, tiles, slots)


# ---------- the live run ----------


def _run_live(
    question: str, ctx: TabContext, pipeline: Any, tiles: Any, slots: list[Any]
) -> TabContext:
    snapshot, store, options = ctx.snapshot, ctx.store, ctx.options
    if snapshot is None or store is None:  # checked by the caller; keeps the types narrow
        return ctx
    speed = pace(slow=options.slow_motion)
    bubble = ctx.bubble  # the chat reply being written on the left, if any
    started = time.perf_counter()
    base = PipelineStats(documents=len(snapshot.doc_ids), chunks=len(snapshot.chunks))
    stats = LiveStats(chunks=len(snapshot.chunks))

    def show_phase(active: str | None, pipe_stats: PipelineStats) -> None:
        # The highlight travels through the pipeline as each phase really runs.
        pipeline.graphviz_chart(pipeline_dot(active, pipe_stats), width="stretch")

    show_phase("Embed", base)
    _show_tiles(tiles, stats)
    _ingest(ctx, slots[0])
    with slots[3].container():
        status = st.empty()
        text = st.empty()
    slots[4].caption("Waiting for the answer…")
    slots[5].caption("Timing starts when the answer does.")
    _say(status, status_text("embed"), bubble)
    time.sleep(speed.step_pause)

    try:
        show_phase("Retrieve", base)
        _say(status, status_text("search", chunks=len(snapshot.chunks)), bubble)
        step = retrieve_step(
            question,
            embedder=load_embedder(settings.embedding_model),
            store=store,
            top_k=options.top_k,
            reranker=load_reranker(settings.reranker_model) if options.rerank else None,
            rerank_candidates=options.rerank_candidates,
        )
    except (VectorStoreError, OptionalDependencyError, ValueError) as exc:
        _fail(status, bubble, str(exc))
        return ctx

    retrieved = {r.chunk.id for r in step.final_results}
    points = meaning_map(snapshot, step.query_embedding, retrieved, max_points=800)
    animated_chart("live-embed", question, lambda p: charts.meaning_map_chart(
        points, 2, p, height=CHART_HEIGHT), placeholder=slots[1], frames=speed.frames,
        frame_seconds=speed.frame_seconds)  # fmt: skip
    partial = QueryTrace(step.vector_results, step.final_results, _empty_answer(question),
                         step.retrieval_ms, step.query_embedding)  # fmt: skip
    rows = similarity_rows(partial)
    cutoff = None if was_reranked(partial) else len(step.final_results)
    animated_chart("live-retrieve", question, lambda p: charts.similarity_chart(
        rows, cutoff, p, height=CHART_HEIGHT), placeholder=slots[2], frames=speed.frames,
        frame_seconds=speed.frame_seconds)  # fmt: skip
    stats = LiveStats(
        chunks=len(snapshot.chunks), candidates=len(step.vector_results),
        top_similarity=step.vector_results[0].score if step.vector_results else None,
        sources=len(step.final_results), elapsed_s=time.perf_counter() - started,
    )  # fmt: skip
    _show_tiles(tiles, stats)
    retrieved_stats = PipelineStats(
        documents=base.documents, chunks=base.chunks,
        candidates=len(step.vector_results), top_k=len(step.final_results),
    )  # fmt: skip

    show_phase("Generate", retrieved_stats)
    _say(status, status_text("read", sources=len(step.final_results)), bubble)
    time.sleep(speed.step_pause)
    _say(status, status_text("think"), bubble)
    llm = settings.model_copy(update={
        "llm_provider": "ollama", "llm_model": options.model,
        "llm_reasoning_effort": options.reasoning_effort, "llm_temperature": options.temperature,
        "llm_max_tokens": settings.llm_detailed_max_tokens if options.answer_style == "detailed"
        else settings.llm_max_tokens,
    })  # fmt: skip
    try:
        generator = Generator(OpenAICompatibleClient.from_settings(llm), options.answer_style)
        stream = generator.stream(question, step.final_results)
        first_token_s = _type_out(
            stream, status, text, tiles, slots[5], stats, step.retrieval_ms, started, speed,
            bubble, len(step.final_results),
        )  # fmt: skip
    except (GenerationError, OptionalDependencyError, ValueError) as exc:
        _fail(status, bubble, str(exc))
        return ctx

    answer = stream.answer
    trace = QueryTrace(step.vector_results, step.final_results, answer, step.retrieval_ms,
                       step.query_embedding)  # fmt: skip
    # Saved before the next st.* call, so a click now cannot lose the answer.
    turn = Turn(trace, first_token_s)
    history: tuple[Turn, ...] = st.session_state.get(HISTORY, ())
    replace_at = st.session_state.pop(PENDING_REPLACE, None)
    st.session_state[HISTORY] = (
        replace_turn(history, replace_at, turn)
        if replace_at is not None and replace_at < len(history)
        else add_turn(history, turn)
    )
    st.session_state[TRACE] = trace
    st.session_state[FIRST_OUTPUT_WAIT] = (trace_key(trace), first_token_s)

    _say(status, status_text("cite"), bubble)
    time.sleep(speed.step_pause)
    # This run already animated these panels; the redraw below shows them settled.
    key = trace_key(trace)
    for chart in ("ov-embed", "ov-retrieve", "ov-timing"):
        mark_shown(chart, _data_key(chart, key))
    st.rerun()  # redraw the chat turn (steps, sources, actions) and the settled dashboard
    return ctx  # not reached: st.rerun() stops this run


def _type_out(
    stream: Iterable[str],
    status: Any,
    text: Any,
    tiles: Any,
    timing: Any,
    stats: LiveStats,
    retrieval_ms: float,
    started: float,
    speed: Pace,
    bubble: Any,
    sources: int,
) -> float | None:
    """Stream the answer smoothly (newest text in view), updating tiles and timing as it goes.

    Returns the seconds waited for the first token (model loading, reading the prompt).
    """
    written, frame = "", 0
    generation_started = time.perf_counter()
    first_token_s: float | None = None
    last_refresh = 0.0
    for pieces, piece in enumerate(stream, start=1):
        if pieces == 1:
            first_token_s = time.perf_counter() - generation_started
            _say(status, status_text("write"), bubble)
        for part in split_for_typing(piece, speed.typing_size):
            written += part
            text.markdown(typing_html(tail_for_display(written, LIVE_TAIL)), unsafe_allow_html=True)
            if bubble is not None:  # the chat scrolls, so it shows the whole answer so far
                bubble.text.markdown(streaming_markdown(written, sources), unsafe_allow_html=True)
            if speed.typing_pause:
                time.sleep(speed.typing_pause)
        now = time.perf_counter()
        if now - last_refresh >= REFRESH_EVERY:
            last_refresh = now
            generating = now - generation_started
            live = LiveStats(**{**stats.__dict__, "tokens": pieces, "generate_s": generating,
                                "first_token_s": first_token_s,
                                "elapsed_s": now - started})  # fmt: skip
            _show_tiles(tiles, live)
            rows = (("retrieve", retrieval_ms), ("generate", generating * 1000))
            timing.plotly_chart(charts.timing_chart(rows, height=CHART_HEIGHT),
                                key=f"live-timing-{frame}", config=charts.CHART_CONFIG)  # fmt: skip
            frame += 1
    return first_token_s


def _empty_answer(question: str) -> Answer:
    """A stand-in answer so retrieval-only views can be drawn before the model answers."""
    return Answer(question=question, text="", citations=(), invalid_citations=(), sources=(),
                  model="", prompt_version="", input_tokens=None, output_tokens=None,
                  latency_ms=0.0)  # fmt: skip


def _say(status: Any, text: str, bubble: Any = None, *, done: bool = False) -> None:
    status.markdown(status_html(text, done=done), unsafe_allow_html=True)
    if bubble is not None:
        bubble.status.markdown(status_html(text, done=done), unsafe_allow_html=True)


def _fail(status: Any, bubble: Any, message: str) -> None:
    status.error(message)
    if bubble is not None:
        bubble.status.error(message)


def _show_tiles(tiles: Any, stats: LiveStats) -> None:
    tiles.markdown(tiles_html(stat_tiles(stats)), unsafe_allow_html=True)


# ---------- the settled view (every rerun after a run, or before any question) ----------


def _draw_static(ctx: TabContext, pipeline: Any, tiles: Any, slots: list[Any]) -> None:
    pipeline.graphviz_chart(pipeline_dot(None, ctx.stats), width="stretch")
    chunks = len(ctx.snapshot.chunks) if ctx.snapshot else None
    stats = LiveStats(chunks=chunks)
    if ctx.trace is not None:
        stored = st.session_state.get(FIRST_OUTPUT_WAIT)
        first = stored[1] if stored and stored[0] == trace_key(ctx.trace) else None
        stats = stats_from_trace(ctx.trace, chunks, first)
    _show_tiles(tiles, stats)
    _ingest(ctx, slots[0])
    _embed(ctx, slots[1])
    _retrieve(ctx, slots[2])
    _generate(ctx, slots[3])
    _evaluate(ctx, slots[4])
    _timing(ctx, slots[5])


def _data_key(chart: str, trace_part: str) -> str:
    version = st.session_state.get(INDEX_VERSION, 0)
    return f"{version}-{trace_part}" if chart == "ov-embed" else trace_part


def _ingest(ctx: TabContext, slot: Any) -> None:
    if ctx.snapshot is None:
        slot.caption("Nothing indexed yet.")
        return
    bins = size_bins([len(c.text) for c in ctx.snapshot.chunks])
    animated_chart("ov-ingest", str(st.session_state.get(INDEX_VERSION, 0)),
                   lambda p: charts.chunk_size_chart(bins, p, height=CHART_HEIGHT),
                   placeholder=slot)  # fmt: skip


def _embed(ctx: TabContext, slot: Any) -> None:
    if ctx.snapshot is None:
        slot.caption("Nothing indexed yet.")
        return
    trace = ctx.trace
    retrieved = {r.chunk.id for r in trace.final_results} if trace else set()
    query = trace.query_embedding if trace and trace.query_embedding else None
    points = meaning_map(ctx.snapshot, query, retrieved, max_points=800)
    animated_chart("ov-embed", _data_key("ov-embed", trace_key(trace) if trace else ""),
                   lambda p: charts.meaning_map_chart(points, 2, p, height=CHART_HEIGHT),
                   placeholder=slot)  # fmt: skip


def _retrieve(ctx: TabContext, slot: Any) -> None:
    trace = ctx.trace
    if trace is None:
        slot.caption("Ask a question to see retrieval.")
        return
    rows = similarity_rows(trace)
    cutoff = None if was_reranked(trace) else len(trace.final_results)
    animated_chart("ov-retrieve", trace_key(trace),
                   lambda p: charts.similarity_chart(rows, cutoff, p, height=CHART_HEIGHT),
                   placeholder=slot)  # fmt: skip


def _generate(ctx: TabContext, slot: Any) -> None:
    if ctx.trace is None:
        slot.caption("Ask a question to see the answer.")
        return
    slot.markdown(spans_html(answer_spans(ctx.trace.answer)), unsafe_allow_html=True)


def _evaluate(ctx: TabContext, slot: Any) -> None:
    judged, trace = ctx.judged, ctx.trace
    if judged is not None and trace is not None:
        values = [
            judged.scores.faithfulness or 0.0,
            judged.scores.answer_relevance or 0.0,
            judged.citations.score or 0.0,
        ]
        marks = [settings.min_faithfulness, settings.min_answer_relevance, None]
        animated_chart("ov-evaluate", trace_key(trace), lambda p: charts.scores_chart(
            ["faithfulness", "relevance", "citations"], [v * p for v in values], marks,
            height=CHART_HEIGHT), placeholder=slot)  # fmt: skip
        return
    generation = load_reports(REPORTS_DIR).generation
    if generation is None:
        slot.caption("Judge an answer in ⑤ Evaluate, or create the golden-set report.")
        return
    overall, thresholds = generation["overall"], generation["thresholds"]
    with slot.container():
        st.caption("Golden-set baseline · judge this answer in ⑤ Evaluate")
        st.plotly_chart(
            charts.scores_chart(
                ["faithfulness", "relevance", "citations"],
                [overall["faithfulness"] or 0.0, overall["answer_relevance"] or 0.0,
                 overall["citation_validity"] or 0.0],
                [thresholds["min_faithfulness"], thresholds["min_answer_relevance"], None],
                height=CHART_HEIGHT - 25,
            ),
            key="ov-baseline", config=charts.CHART_CONFIG,
        )  # fmt: skip


def _timing(ctx: TabContext, slot: Any) -> None:
    trace = ctx.trace
    if trace is None:
        slot.caption("Ask a question to see the timings.")
        return
    animated_chart("ov-timing", trace_key(trace),
                   lambda p: charts.timing_chart(timing_rows(trace, ctx.judged), p,
                                                 height=CHART_HEIGHT),
                   placeholder=slot)  # fmt: skip
