"""⑥ Gate: how CI decides whether a change may merge, and a what-if run of it."""

import html
import json
from dataclasses import dataclass

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.evaluation.baselines import (
    GENERATION_BASELINE,
    RETRIEVAL_BASELINE,
    Baseline,
    BaselineError,
    load_baseline,
)
from rag_eval_platform.evaluation.evaluator import RetrievalReport, RetrievalThresholds
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
from rag_eval_platform.evaluation.gate import (
    GateLimits,
    GateResult,
    gate_from_dict,
    run_gate,
)
from rag_eval_platform.evaluation.gate_runner import rebuild_and_evaluate
from rag_eval_platform.evaluation.golden_dataset import GoldenDatasetError, load_golden_dataset
from rag_eval_platform.ingestion.embedding import EmbeddingError
from rag_eval_platform.ingestion.loaders import DocumentLoadError
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import REPORTS_DIR
from rag_eval_platform.playground.gate_view import (
    flipped_questions,
    gate_flow_dot,
    preview_settings,
    type_comparison,
)
from rag_eval_platform.playground.shared import (
    GATE_RUN,
    TabContext,
    connect_gate_store,
    load_embedder,
    load_reranker,
    settings,
)
from rag_eval_platform.retrieval.vector_store import VectorStoreError

GAUGES_PER_ROW = 3  # five gauges in one row clip their titles


@dataclass(frozen=True)
class GateRun:
    result: GateResult
    retrieval: RetrievalReport
    baseline: Baseline | None
    questions: dict[str, str]


def render(ctx: TabContext) -> None:
    run: GateRun | None = st.session_state.get(GATE_RUN)
    latest = run.result if run else _latest_ci_result()
    st.graphviz_chart(gate_flow_dot(latest), width="stretch")
    left, right = st.columns([1, 2], gap="large")
    with left:
        _explain_and_run(ctx)
    with right:
        if latest is None:
            st.info("No gate result yet: click **Run the gate** to try your sidebar settings.")
            return
        _board(latest)
        if run is not None:
            _comparison(run)


def _explain_and_run(ctx: TabContext) -> None:
    st.markdown(
        "Every pull request is scored on the **30 golden questions** before it can merge. "
        "Retrieval is measured **live**; answer quality comes from a **committed baseline** "
        "judged locally, which must match the current settings."
    )
    o = ctx.options
    st.caption(f"What-if with your sidebar: {o.strategy} chunks of {o.chunk_size} "
               f"(overlap {o.chunk_overlap}), top-{o.top_k}, "
               f"re-rank {'on' if o.rerank else 'off'}.")  # fmt: skip
    if st.button("Run the gate", type="primary", key="gate-run"):
        _run(ctx)


def _run(ctx: TabContext) -> None:
    o = ctx.options
    what_if = preview_settings(settings, strategy=o.strategy, chunk_size=o.chunk_size,
                               chunk_overlap=o.chunk_overlap, top_k=o.top_k, rerank=o.rerank,
                               rerank_candidates=o.rerank_candidates)  # fmt: skip
    try:
        with st.status("Running the gate...", expanded=True) as status:
            examples = load_golden_dataset(what_if.golden_dataset_path)
            retrieval = rebuild_and_evaluate(
                examples, raw_dir=what_if.raw_data_dir, strategy=o.strategy,
                chunk_size=o.chunk_size, chunk_overlap=o.chunk_overlap,
                embedder=load_embedder(what_if.embedding_model),
                store=connect_gate_store(what_if.chroma_host, what_if.chroma_port),
                top_k=o.top_k,
                reranker=load_reranker(what_if.reranker_model) if o.rerank else None,
                rerank_candidates=o.rerank_candidates,
                thresholds=RetrievalThresholds.from_settings(what_if),
                on_progress=st.write,
            )  # fmt: skip
            baseline = load_baseline(RETRIEVAL_BASELINE)
            generation = load_baseline(GENERATION_BASELINE)
            limits = GateLimits.from_settings(what_if)
            result = run_gate(retrieval, baseline, generation, current_fingerprint(what_if), limits)
            # Saved before the next st.* call, so a click now cannot lose the result.
            st.session_state[GATE_RUN] = GateRun(result, retrieval, baseline,
                                                 {e.id: e.question for e in examples})  # fmt: skip
            status.update(label="Gate finished", state="complete", expanded=False)
    except (GoldenDatasetError, DocumentLoadError, VectorStoreError, EmbeddingError,
            OptionalDependencyError, BaselineError, OSError, ValueError) as exc:  # fmt: skip
        st.error(str(exc))
        return
    st.rerun()


def _board(result: GateResult) -> None:
    if result.passed:
        st.success("✅ The gate passes: this change may merge.")
    else:
        st.error("⛔ The gate blocks this change.")
    scored = [c for c in result.checks if c.value is not None]
    for start in range(0, len(scored), GAUGES_PER_ROW):
        row = scored[start : start + GAUGES_PER_ROW]
        for column, check in zip(st.columns(GAUGES_PER_ROW), row, strict=False):
            column.plotly_chart(
                charts.gauge_chart(check.name, check.value or 0.0, check.threshold, height=170,
                                   passed=check.status == "pass"),
                key=f"gate-gauge-{check.name}", config=charts.CHART_CONFIG,
            )  # fmt: skip
    for check in result.checks:
        mark = "✅" if check.status == "pass" else "⛔"
        st.markdown(f"{mark} **{html.escape(check.name)}**: {html.escape(check.reason)}")
    if result.changes:
        st.warning("The generation baseline is stale. Changed since it was judged:\n\n"
                   + "\n".join(f"- `{change}`" for change in result.changes))  # fmt: skip


def _comparison(run: GateRun) -> None:
    st.markdown("**Baseline vs now, by question type**")
    st.plotly_chart(charts.type_compare_chart(type_comparison(run.retrieval, run.baseline)),
                    key="gate-types", config=charts.CHART_CONFIG)  # fmt: skip
    flips = flipped_questions(run.retrieval, run.baseline, run.questions)
    st.markdown(f"**Questions that flipped: {len(flips)}**")
    for flip in flips:
        arrow = "hit → miss" if flip.before_hit else "miss → hit"
        st.markdown(f"- `{flip.example_id}` ({arrow}) {html.escape(flip.question)} · "
                    f"expects {', '.join(flip.relevant_doc_ids)}")  # fmt: skip


def _latest_ci_result() -> GateResult | None:
    path = REPORTS_DIR / "gate_report.json"
    try:
        return gate_from_dict(json.loads(path.read_text(encoding="utf-8"))["gate"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
