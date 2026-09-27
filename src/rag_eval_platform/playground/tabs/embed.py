"""② Embed: every chunk as a point in meaning space, and your question among them."""

import math

import streamlit as st

from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import trace_key
from rag_eval_platform.playground.shared import (
    INDEX_VERSION,
    TabContext,
    animated_chart,
    show_pipeline,
)
from rag_eval_platform.playground.visuals import angle_degrees, meaning_map, vector_for


def render(ctx: TabContext) -> None:
    show_pipeline("Embed", ctx)
    snapshot, trace = ctx.snapshot, ctx.trace
    if snapshot is None:
        st.info("Nothing indexed yet: build an index in ① Ingest.")
        return
    st.markdown(
        "Each chunk became **384 numbers** (its *embedding*). Similar meanings point in similar "
        "directions. The map squeezes 384 dimensions down to 2 or 3 (PCA) so you can see them."
    )
    dims = (
        3
        if st.segmented_control("Map", ["2-D", "3-D"], default="2-D", key="embed-dims") == "3-D"
        else 2
    )
    retrieved = {r.chunk.id for r in trace.final_results} if trace else set()
    query = trace.query_embedding if trace and trace.query_embedding else None
    points = meaning_map(snapshot, query, retrieved, dims=dims)
    version = st.session_state.get(INDEX_VERSION, 0)
    data_key = f"{version}-{trace_key(trace) if trace else ''}-{dims}"
    animated_chart(
        f"embed-map-{dims}",
        data_key,
        lambda p: charts.meaning_map_chart(points, dims, p, height=460),
    )
    if trace is None or not trace.query_embedding:
        st.info("Ask a question (box above) to see where it lands among your chunks.")
        return

    best = trace.final_results[0].chunk.id
    best_vector = vector_for(snapshot, best)
    if best_vector is None:
        return
    st.markdown(
        f"**The meaning code.** Your question vs the top chunk `{best}`, dimension by dimension "
        "(blue = positive, red = negative). Similar texts share patterns."
    )
    st.plotly_chart(
        charts.embedding_strip_chart(
            {"your question": trace.query_embedding, best: best_vector}, height=150
        ),
        key="embed-strips",
        config=charts.CHART_CONFIG,
    )
    angle = angle_degrees(trace.query_embedding, best_vector)
    cols = st.columns(3)
    cols[0].metric("Angle", f"{angle:.0f}°")
    cols[1].metric("Cosine similarity", f"{math.cos(math.radians(angle)):.3f}")
    cols[2].metric("Meaning", "close" if angle < 60 else "loose" if angle < 80 else "unrelated")
    st.caption(
        "0° = same direction (same meaning), 90° = unrelated. Retrieval ranks chunks by this "
        "cosine similarity."
    )
