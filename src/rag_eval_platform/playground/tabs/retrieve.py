"""③ Retrieve: which chunks came closest, where the top-k cut falls, and re-ranking."""

import streamlit as st

from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import trace_key
from rag_eval_platform.playground.query_visuals import (
    funnel_stages,
    rerank_moves,
    similarity_rows,
    was_reranked,
)
from rag_eval_platform.playground.shared import TabContext, animated_chart, show_pipeline


def render(ctx: TabContext) -> None:
    show_pipeline("Retrieve", ctx)
    trace = ctx.trace
    if trace is None:
        st.info("Ask a question (box above) to see retrieval at work.")
        return
    rows = similarity_rows(trace)
    reranked = was_reranked(trace)
    cutoff = None if reranked else len(trace.final_results)
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown(
            "**Similarity of every candidate.** Vector search ranks chunks by cosine "
            "similarity. Blue = sent to the model, green = cited in the answer."
        )
        animated_chart(
            "retrieve-similarity",
            trace_key(trace),
            lambda p: charts.similarity_chart(rows, cutoff, p, height=320),
        )
    with right:
        st.markdown("**The funnel.** From every chunk to the few the answer cites.")
        total = len(ctx.snapshot.chunks) if ctx.snapshot else len(trace.vector_results)
        st.plotly_chart(
            charts.funnel_chart(funnel_stages(total, trace), height=320),
            key="retrieve-funnel",
            config=charts.CHART_CONFIG,
        )
    if reranked:
        st.markdown(
            "**Re-ranking.** The cross-encoder read each candidate *with* the question and "
            "reordered them. Crossing lines = it disagreed with vector search; grey lines fall "
            "out of the top-k."
        )
        st.plotly_chart(
            charts.rerank_chart(rerank_moves(trace), height=360),
            key="retrieve-rerank",
            config=charts.CHART_CONFIG,
        )
    else:
        st.caption(
            "Turn on **Re-rank with a cross-encoder** in the sidebar and ask again to compare "
            "rankings."
        )
    st.dataframe(
        [
            {
                "rank": r.rank,
                "chunk": r.chunk_id,
                "similarity": round(r.score, 3),
                "sent to model": r.kept,
                "cited": r.cited,
            }
            for r in rows
        ],
        hide_index=True,
        key="retrieve-table",
    )
