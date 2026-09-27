"""④ Generate: the prompt the model read, its answer with citations, and the time it took."""

import html

import streamlit as st

from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import trace_key
from rag_eval_platform.playground.query_visuals import (
    answer_spans,
    origin,
    prompt_blocks,
    source_color,
    spans_html,
    timing_rows,
)
from rag_eval_platform.playground.shared import TabContext, animated_chart, show_pipeline


def render(ctx: TabContext) -> None:
    show_pipeline("Generate", ctx)
    trace = ctx.trace
    if trace is None:
        st.info("Ask a question (box above) to see the prompt and the answer.")
        return
    answer = trace.answer
    blocks = prompt_blocks(trace)
    st.markdown(
        "**The prompt**, in the order the model reads it: rules, numbered sources, your "
        "question (bar length = characters)."
    )
    st.plotly_chart(
        charts.prompt_chart(blocks, height=130), key="generate-prompt", config=charts.CHART_CONFIG
    )
    with st.expander("Read the full prompt"):
        for block in blocks:
            st.caption(block.title)
            st.code(block.text, language=None, wrap_lines=True)

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown("**The answer.** Each citation has its source's colour.")
        if answer.is_refusal:
            st.warning("The model refused: the retrieved chunks did not contain the answer.")
        st.markdown(spans_html(answer_spans(answer)), unsafe_allow_html=True)
        if answer.invalid_citations:
            st.error(f"Cited numbers that were never provided: {list(answer.invalid_citations)}")
    with right:
        st.markdown("**Sources given to the model**")
        cited = {c.number for c in answer.citations}
        for number, result in enumerate(trace.final_results, start=1):
            mark = " · cited" if number in cited else ""
            st.markdown(
                f'<span style="color:{source_color(number)};font-weight:700">[{number}]</span> '
                f"{html.escape(origin(result))}{mark}",
                unsafe_allow_html=True,
            )
    st.markdown("**Where the time went**")
    animated_chart(
        "generate-timing",
        trace_key(trace),
        lambda p: charts.timing_chart(timing_rows(trace, ctx.judged), p, height=110),
    )
    st.caption(
        f"{answer.model} · {answer.input_tokens} tokens in → {answer.output_tokens} out · "
        f"prompt {answer.prompt_version}"
    )
