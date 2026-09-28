"""⑤ Evaluate: grade this answer with the LLM judge, and see the golden-set reports."""

from typing import Any

import plotly.graph_objects as go
import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.evaluation.judge import JudgeError
from rag_eval_platform.generation.generator import GenerationError
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import REPORTS_DIR, judge_answer, load_reports, trace_key
from rag_eval_platform.playground.header import panel_title_html
from rag_eval_platform.playground.query_visuals import hit_vs_faithfulness, verdict_dot
from rag_eval_platform.playground.session_metrics import (
    session_points,
)
from rag_eval_platform.playground.shared import (
    HISTORY,
    TabContext,
    animated_chart,
    load_judge,
    save_judged,
    settings,
    show_pipeline,
)


def render(ctx: TabContext) -> None:
    show_pipeline("Evaluate", ctx)
    _session()
    judge_col, reports_col = st.columns(2, gap="large")
    with judge_col:
        _judge_this_answer(ctx)
    with reports_col:
        _golden_reports()


def _session() -> None:
    """How this conversation is going: time, coverage and grounding for every question."""
    points = session_points(st.session_state.get(HISTORY, ()))
    if not points:
        return
    st.markdown(panel_title_html("This session", "health"), unsafe_allow_html=True)
    time_col, quality_col = st.columns(2, gap="large")
    time_col.plotly_chart(charts.session_time_chart(points), key="session-time",
                          config=charts.CHART_CONFIG)  # fmt: skip
    quality_col.plotly_chart(charts.session_quality_chart(points), key="session-quality",
                             config=charts.CHART_CONFIG)  # fmt: skip
    st.caption("Coverage and grounding are instant word-level checks on every answer; the "
               "judge below grades one answer with a second model.")  # fmt: skip


def _judge_this_answer(ctx: TabContext) -> None:
    st.subheader("Judge this answer")
    st.caption(
        f"A second model ({settings.judge_model}) checks whether every claim is backed by the "
        "retrieved chunks, whether it answers the question, and whether each citation points "
        "to a chunk that supports it. Takes 1 to 2 minutes."
    )
    trace = ctx.trace
    if trace is None:
        st.info("Ask a question first (box above).")
        return
    if trace.answer.is_refusal:
        st.info("This answer is a refusal: it makes no claims, so there is nothing to judge.")
        return
    if ctx.judged is None:
        if st.button("Judge this answer", type="primary"):
            try:
                with st.status("Judging…", expanded=True) as status:
                    st.write("Loading the judge model…")
                    judge, checker = load_judge(settings.judge_model)
                    st.write("Scoring faithfulness and relevance, checking citations…")
                    result = judge_answer(trace, judge, checker)
                    # Saved before the next st.* call, so a click now cannot lose the result.
                    save_judged(trace, result)
                    status.update(label="Judged", state="complete")
            except OptionalDependencyError:
                st.error(
                    "The judge needs the evaluation extra: `uv sync --all-extras --all-groups`"
                )
                return
            except (JudgeError, GenerationError) as exc:
                st.error(str(exc))
                return
            st.rerun()
        return

    result, key = ctx.judged, trace_key(trace)
    gauges = [
        ("faithfulness", result.scores.faithfulness, settings.min_faithfulness),
        ("answer relevance", result.scores.answer_relevance, settings.min_answer_relevance),
        ("citation validity", result.citations.score, None),
    ]
    for column, (name, value, mark) in zip(st.columns(3), gauges, strict=True):
        with column:
            if value is None:
                st.metric(name, "not scored")
            else:
                animated_chart(f"judge-{name}", key, _gauge(name, value, mark))
    if result.note:
        st.warning(result.note)
    st.markdown("**Each citation's verdict** (green = the chunk supports the sentence).")
    st.graphviz_chart(verdict_dot(result), width="stretch")
    st.caption(f"Judged in {result.judge_ms / 1000:.0f} s")


def _gauge(name: str, value: float, mark: float | None) -> Any:
    def build(progress: float) -> go.Figure:
        return charts.gauge_chart(name, value, mark, progress, height=190)

    return build


def _golden_reports() -> None:
    st.subheader("Golden-set reports")
    reports = load_reports(REPORTS_DIR)
    if reports.generation is None and reports.retrieval is None:
        st.info("No reports yet. Create them (the generation run takes about an hour):")
        st.code(
            "uv run python scripts/run_evaluation.py\n"
            "uv run python scripts/run_generation_evaluation.py",
            language="bash",
        )
        return
    if reports.generation is not None:
        gen = reports.generation
        overall, marks = gen["overall"], gen["thresholds"]
        st.markdown(f"**Answers** ({len(gen['examples'])} questions, judge {gen['judge_model']})")
        st.plotly_chart(
            charts.scores_chart(
                ["faithfulness", "answer relevance", "citation validity"],
                [
                    overall["faithfulness"] or 0.0,
                    overall["answer_relevance"] or 0.0,
                    overall["citation_validity"] or 0.0,
                ],
                [marks["min_faithfulness"], marks["min_answer_relevance"], None],
                height=230,
            ),
            key="eval-gen-scores",
            config=charts.CHART_CONFIG,
        )
        st.plotly_chart(
            charts.per_type_chart(
                gen["by_query_type"],
                ["faithfulness", "answer_relevance", "citation_validity"],
                height=230,
            ),
            key="eval-gen-types",
            config=charts.CHART_CONFIG,
        )
    if reports.retrieval is not None:
        ret = reports.retrieval
        overall, marks = ret["overall"], ret["thresholds"]
        st.markdown(f"**Retrieval** (top-{ret['k']})")
        st.plotly_chart(
            charts.scores_chart(
                ["recall@k", "mrr", "ndcg@k", "precision@k"],
                [overall["recall"], overall["mrr"], overall["ndcg"], overall["precision"]],
                [marks["min_recall_at_k"], marks["min_mrr"], marks["min_ndcg_at_k"], None],
                height=230,
            ),
            key="eval-ret-scores",
            config=charts.CHART_CONFIG,
        )
    if reports.retrieval is not None and reports.generation is not None:
        joined = hit_vs_faithfulness(reports.retrieval, reports.generation)
        st.markdown("**Does bad retrieval cause unsupported answers?** (mean faithfulness)")
        st.plotly_chart(
            charts.scores_chart(
                [f"{label} (n={n})" for label, _, n in joined],
                [mean for _, mean, _ in joined],
                [None] * len(joined),
                height=230,
            ),
            key="eval-join",
            config=charts.CHART_CONFIG,
        )
