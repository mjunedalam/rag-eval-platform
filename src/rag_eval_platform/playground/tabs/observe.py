"""⑧ Observe: traffic, latency, cost, quality over time, drift warnings and a review queue."""

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from rag_eval_platform.observability.cost import Prices
from rag_eval_platform.observability.dashboard import (
    EVAL_METRICS,
    RANGES,
    baseline_values,
    bucket_size,
    buckets,
    cost_label,
    eval_history,
    metric_thresholds,
    query_rows,
    since_for,
    summarize,
)
from rag_eval_platform.observability.drift import DriftThresholds, drift_report
from rag_eval_platform.observability.logger import create_logset
from rag_eval_platform.observability.logset import LogSet
from rag_eval_platform.observability.review import REASON_LABELS, review_queue
from rag_eval_platform.observability.store import (
    EvalRunRecord,
    FeedbackRecord,
    ObserveStoreError,
    QueryRecord,
)
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.query_visuals import inert_markdown_images
from rag_eval_platform.playground.shared import TabContext, settings

BASELINES = Path("baselines")
EMPTY = "No queries yet — ask one in the chat or the API."
REVIEW_LIMIT = 20


def render(ctx: TabContext) -> None:
    store = create_logset(settings)
    if store is None:
        st.info("Observability is off (RAG_OBSERVE_ENABLED=false).")
        return
    try:
        _render(store)
    except ObserveStoreError as exc:
        st.error(str(exc))


def _render(store: LogSet) -> None:
    thresholds = DriftThresholds.from_settings(settings)
    drift_limit = 2 * thresholds.window
    recent_queries = store.recent_queries(drift_limit)
    recent_feedback = store.feedback_for([q.id for q in recent_queries])
    runs = store.eval_runs()
    report = drift_report(recent_queries, recent_feedback, runs, thresholds,
                          total=store.count_queries())  # fmt: skip
    for warning in report.warnings:
        st.warning(warning, icon="⚠️")
    if not report.enough_data:
        st.info(f"Drift checks need {report.need} answers; {report.have} logged so far.")

    label = st.segmented_control("Time range", list(RANGES), default="7 days",
                                 key="obs-range", required=True) or "7 days"  # fmt: skip
    since = since_for(label, datetime.now(UTC))
    queries = store.queries(since=since)
    all_feedback = store.feedback_for([q.id for q in queries])
    summary = summarize(queries, all_feedback)
    tiles = st.columns(8)
    values = (
        ("Queries", f"{summary.queries:,}"),
        ("p50", _seconds(summary.p50_ms)), ("p95", _seconds(summary.p95_ms)),
        ("Errors", _percent(summary.error_rate)), ("Refusals", _percent(summary.refusal_rate)),
        ("👍 rate", _percent(summary.thumbs_up_rate)), ("Tokens", f"{summary.tokens:,}"),
        ("Cost", cost_label(summary, Prices.from_settings(settings))),
    )  # fmt: skip
    for tile, (name, value) in zip(tiles, values, strict=True):
        tile.metric(name, value)

    if queries:
        rows = buckets(queries, bucket_size(label))
        left, middle, right = st.columns(3)
        left.plotly_chart(charts.observe_latency_chart(rows), key="obs-latency",
                          config=charts.CHART_CONFIG)  # fmt: skip
        middle.plotly_chart(charts.observe_volume_chart(rows), key="obs-volume",
                            config=charts.CHART_CONFIG)  # fmt: skip
        right.plotly_chart(charts.observe_tokens_chart(rows), key="obs-tokens",
                           config=charts.CHART_CONFIG)  # fmt: skip
    else:
        st.info(EMPTY)
    _evaluation(runs)
    _review(store, queries, all_feedback)
    _recent(store, queries, all_feedback, label)


def _evaluation(runs: list[EvalRunRecord]) -> None:
    st.subheader("Evaluation history")
    history, baselines = eval_history(runs), baseline_values(BASELINES)
    thresholds = metric_thresholds(settings)
    if not runs:
        st.caption(
            "No evaluation runs logged yet; the lines show the committed baselines. "
            "Run scripts/run_evaluation.py or scripts/run_gate.py to add points."
        )
    columns = st.columns(len(EVAL_METRICS))
    for column, (metric, name) in zip(columns, EVAL_METRICS.items(), strict=True):
        column.plotly_chart(
            charts.eval_history_chart(name, history[metric], baselines.get(metric),
                                      thresholds.get(metric)),
            key=f"obs-eval-{metric}", config=charts.CHART_CONFIG,
        )  # fmt: skip


def _review(store: LogSet, queries: list[QueryRecord], feedback: list[FeedbackRecord]) -> None:
    st.subheader("Review queue")
    show_reviewed = st.toggle("Show reviewed", key="obs-show-reviewed")
    items = review_queue(queries, feedback, slow_ms=settings.review_slow_ms,
                         include_reviewed=show_reviewed)  # fmt: skip
    if not items:
        st.caption("Nothing to review: no 👎, refusals, invalid citations, errors or slow answers.")
    for item in items[:REVIEW_LIMIT]:
        with st.container(border=True):
            badges = " · ".join(REASON_LABELS[r] for r in item.reasons)
            st.markdown(f"**{badges}** — {item.query.ts:%Y-%m-%d %H:%M} · {item.query.source}")
            st.text(item.query.question)
            if item.comment:
                st.text(f"Comment: {item.comment}")
            if item.query.answer:
                with st.expander("Answer"):
                    st.text(item.query.answer)
            if item.query.error:
                st.text(item.query.error)
            if item.query.reviewed_at is None and st.button(
                    "Mark reviewed", key=f"obs-reviewed-{item.query.id}"):  # fmt: skip
                store.mark_reviewed(item.query.id, datetime.now(UTC))
                st.rerun()


def _selected_query_id(rows: list[dict[str, object]], selected: list[int]) -> str | None:
    """The id of the selected row, or None without a valid selection.

    Bounds-checked: the dataframe's key changes with the range and source filters, but on
    the rerun where a filter changes, a leftover selection index from the previous (possibly
    longer) table must never be used to index the new, possibly shorter one.
    """
    if not selected or selected[0] >= len(rows):
        return None
    return str(rows[selected[0]]["id"])


def _recent(
    store: LogSet, queries: list[QueryRecord], feedback: list[FeedbackRecord], label: str
) -> None:
    st.subheader("Recent queries")
    source = st.segmented_control("Source", ["all", "api", "playground", "cli"], default="all",
                                  key="obs-source", required=True) or "all"  # fmt: skip
    rows = query_rows(queries, feedback, None if source == "all" else source)
    if not rows:
        st.caption(EMPTY)
        return
    picked = st.dataframe(rows, hide_index=True, key=f"obs-recent-{label}-{source}",
                          on_select="rerun", selection_mode="single-row",
                          column_config={"id": None})  # fmt: skip
    query_id = _selected_query_id(rows, picked.selection.rows if picked else [])
    if query_id is not None:
        query = store.get_query(query_id)
        if query is not None:
            with st.container(border=True):
                st.text(query.question)
                if query.answer:
                    st.markdown(inert_markdown_images(query.answer))  # no remote images
                elif query.error:
                    st.text(query.error)
                st.caption(f"Cited: {', '.join(query.cited_doc_ids) or 'none'} · "
                           f"model {query.model} · prompt {query.prompt_version} · "
                           f"retrieve {_seconds(query.retrieve_ms)} · "
                           f"generate {_seconds(query.generate_ms)} · "
                           f"total {_seconds(query.total_ms)}")  # fmt: skip


def _seconds(ms: float | None) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f} s"


def _percent(rate: float | None) -> str:
    return "—" if rate is None else f"{rate:.0%}"
