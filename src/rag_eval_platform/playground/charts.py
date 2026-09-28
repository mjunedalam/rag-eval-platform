"""Plotly figures for the playground, built from the pure data in visuals.py and
query_visuals.py. Builders take ``progress`` (0..1) so a chart can grow in, and a fixed
``height`` so the Overview fits on one screen."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import plotly.graph_objects as go

from rag_eval_platform.playground.gate_view import TypeCompare
from rag_eval_platform.playground.query_visuals import (
    INVALID_COLOR,
    YES_COLOR,
    PromptBlock,
    RerankMove,
    SimilarityRow,
)
from rag_eval_platform.playground.session_metrics import SessionPoint
from rag_eval_platform.playground.visuals import ChunkSpan, MapPoint, SizeBin

CHART_CONFIG = {"displayModeBar": False}
NEUTRAL, MUTED, ACCENT, KEPT = "#2563eb", "#cbd5e1", "#d97706", "#60a5fa"
BLOCK_COLORS = {"rules": "#6366f1", "source": "#16a34a", "question": "#d97706"}


FONT = "Inter, Helvetica, Arial, sans-serif"  # the app's font (set in .streamlit/config.toml)


def _layout(fig: go.Figure, height: int, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title={"text": title, "font": {"size": 13}} if title else None,
        margin={"l": 8, "r": 8, "t": 32 if title else 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Text and grid colours are left to Streamlit's chart theme, so they suit light and dark.
        font={"family": FONT, "size": 11},
        hoverlabel={"font": {"family": FONT}},
    )
    return fig


def chunk_size_chart(
    bins: Sequence[SizeBin], progress: float = 1.0, height: int = 300
) -> go.Figure:
    top = max((b.count for b in bins), default=1)
    fig = go.Figure(
        go.Bar(
            x=[f"{b.start}-{b.end}" for b in bins],
            y=[b.count * progress for b in bins],
            marker_color=NEUTRAL,
            hovertemplate="%{x} chars: %{y:.0f} chunks<extra></extra>",
        )
    )
    fig.update_yaxes(range=[0, top * 1.15], title="chunks")
    fig.update_xaxes(title="chunk size (characters)")
    return _layout(fig, height)


def chunk_layout_chart(spans: Sequence[ChunkSpan], height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        name="chunk",
        y=[f"#{s.index}" for s in spans],
        x=[s.end - s.start for s in spans],
        base=[s.start for s in spans],
        orientation="h",
        marker_color=KEPT,
        hovertemplate="chars %{base} to %{x}<extra></extra>",
    )
    overlapping = [s for s in spans if s.overlap]
    fig.add_bar(
        name="overlap",
        y=[f"#{s.index}" for s in overlapping],
        x=[s.overlap for s in overlapping],
        base=[s.start for s in overlapping],
        orientation="h",
        marker_color=INVALID_COLOR,
        opacity=0.8,
        hovertemplate="%{x} chars shared with the previous chunk<extra></extra>",
    )
    fig.update_layout(barmode="overlay", legend={"orientation": "h", "y": 1.08})
    fig.update_yaxes(autorange="reversed", title="chunk")
    fig.update_xaxes(title="position in the document (characters)")
    return _layout(fig, height)


def meaning_map_chart(
    points: Sequence[MapPoint], dims: int = 2, progress: float = 1.0, height: int = 400
) -> go.Figure:
    styles = {
        "chunk": ("chunks", MUTED, 6, "circle"),
        "retrieved": ("retrieved", NEUTRAL, 11, "circle"),
        "question": ("your question", ACCENT, 18, "diamond"),
    }
    fig = go.Figure()
    for kind, (name, color, size, symbol) in styles.items():
        group = [p for p in points if p.kind == kind]
        common: dict[str, Any] = {
            "name": name,
            "mode": "markers",
            "hovertext": [p.text for p in group],
            "hoverinfo": "text",
            "marker": {"color": color, "size": size, "symbol": symbol, "line": {"width": 0}},
        }
        xs = [p.x * progress for p in group]
        ys = [p.y * progress for p in group]
        if dims == 3:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[p.z * progress for p in group], **common))
        else:
            fig.add_trace(go.Scatter(x=xs, y=ys, **common))
    extent = max((max(abs(p.x), abs(p.y)) for p in points), default=1.0) * 1.1 or 1.0
    if dims == 2:
        fig.update_xaxes(range=[-extent, extent], showticklabels=False, zeroline=False)
        fig.update_yaxes(range=[-extent, extent], showticklabels=False, zeroline=False)
    fig.update_layout(legend={"orientation": "h", "y": 1.08})
    return _layout(fig, height)


def embedding_strip_chart(vectors: Mapping[str, Sequence[float]], height: int = 160) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=[list(v) for v in vectors.values()],
            y=list(vectors),
            colorscale="RdBu",
            zmid=0,
            showscale=False,
            hovertemplate="%{y} · dim %{x}: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="dimension", showgrid=False)
    return _layout(fig, height)


def similarity_chart(
    rows: Sequence[SimilarityRow], cutoff: int | None, progress: float = 1.0, height: int = 300
) -> go.Figure:
    colors = [YES_COLOR if r.cited else KEPT if r.kept else MUTED for r in rows]
    fig = go.Figure(
        go.Bar(
            x=[f"#{r.rank}" for r in rows],
            y=[r.score * progress for r in rows],
            marker_color=colors,
            hovertext=[r.chunk_id for r in rows],
            hovertemplate="%{hovertext}: %{y:.3f}<extra></extra>",
        )
    )
    if cutoff is not None and cutoff < len(rows):
        fig.add_vline(
            x=cutoff - 0.5,
            line_dash="dash",
            line_color=ACCENT,
            annotation_text=f"top-{cutoff}",
            annotation_position="top",
        )
    top = max((r.score for r in rows), default=1.0)
    fig.update_yaxes(range=[0, top * 1.15], title="cosine similarity")
    return _layout(fig, height)


def rerank_chart(moves: Sequence[RerankMove], height: int = 300) -> go.Figure:
    dropped_rank = len(moves) + 1
    fig = go.Figure()
    for move in moves:
        kept = move.after is not None
        fig.add_trace(
            go.Scatter(
                x=["vector search", "cross-encoder"],
                y=[move.before, move.after or dropped_rank],
                mode="lines+markers",
                name=move.chunk_id,
                showlegend=False,
                line={"color": NEUTRAL if kept else MUTED, "width": 3 if kept else 1},
                hovertemplate=f"{move.chunk_id}: rank %{{y}}<extra></extra>",
            )
        )
    fig.update_yaxes(autorange="reversed", title="rank", dtick=1)
    return _layout(fig, height)


def funnel_chart(stages: Sequence[tuple[str, int]], height: int = 250) -> go.Figure:
    """Stage widths on a log scale (1,505 chunks vs 2 cited would hide the small stages);
    the labels show the true counts."""
    fig = go.Figure(
        go.Funnel(
            y=[name for name, _ in stages],
            x=[math.log10(value + 1) + 0.3 for _, value in stages],
            text=[str(value) for _, value in stages],
            textinfo="text",
            hoverinfo="y+text",
            marker={"color": [MUTED, KEPT, NEUTRAL, YES_COLOR][: len(stages)]},
        )
    )
    return _layout(fig, height)


def prompt_chart(blocks: Sequence[PromptBlock], height: int = 140) -> go.Figure:
    fig = go.Figure()
    for block in blocks:
        fig.add_bar(
            name=block.title,
            y=["prompt"],
            x=[len(block.text)],
            orientation="h",
            marker_color=BLOCK_COLORS[block.kind],
            text=[block.title],
            textposition="inside",
            hovertemplate=f"{block.title}: %{{x}} chars<extra></extra>",
        )
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_xaxes(title="characters")
    return _layout(fig, height)


def timing_chart(
    rows: Sequence[tuple[str, float]], progress: float = 1.0, height: int = 120
) -> go.Figure:
    colors = {"retrieve": KEPT, "generate": NEUTRAL, "judge": ACCENT}
    total = sum(ms for _, ms in rows) / 1000 or 1.0
    fig = go.Figure()
    for step, ms in rows:
        seconds = ms / 1000
        fig.add_bar(
            name=step,
            y=["time"],
            x=[seconds * progress],
            orientation="h",
            marker_color=colors.get(step, MUTED),
            text=[f"{step} {seconds:.1f}s"],
            textposition="inside",
        )
    fig.update_layout(barmode="stack", showlegend=False)
    fig.update_xaxes(range=[0, total * 1.02], title="seconds")
    return _layout(fig, height)


def gauge_chart(
    title: str,
    value: float,
    threshold: float | None,
    progress: float = 1.0,
    height: int = 200,
    *,
    passed: bool | None = None,  # override, e.g. a gate check that failed on another rule
) -> go.Figure:
    if passed is None:
        passed = threshold is None or value >= threshold
    gauge: dict[str, Any] = {
        "axis": {"range": [0, 1]},
        "bar": {"color": YES_COLOR if passed else INVALID_COLOR},
    }
    if threshold is not None:
        gauge["threshold"] = {"value": threshold, "line": {"color": "#111827", "width": 3}}
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value * progress,
            number={"valueformat": ".2f"},
            gauge=gauge,
            title={"text": title, "font": {"size": 13}},
        )
    )
    return _layout(fig, height)


def scores_chart(
    names: Sequence[str], values: Sequence[float], marks: Sequence[float | None], height: int = 250
) -> go.Figure:
    colors = [
        NEUTRAL if mark is None else YES_COLOR if value >= mark else INVALID_COLOR
        for value, mark in zip(values, marks, strict=True)
    ]
    fig = go.Figure(
        go.Bar(
            x=list(names),
            y=list(values),
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition="outside",
            name="score",
        )
    )
    with_marks = [(n, m) for n, m in zip(names, marks, strict=True) if m is not None]
    fig.add_trace(
        go.Scatter(
            x=[n for n, _ in with_marks],
            y=[m for _, m in with_marks],
            mode="markers",
            name="pass mark",
            marker={"symbol": "line-ew", "size": 40, "color": "#111827", "line": {"width": 3}},
        )
    )
    fig.update_yaxes(range=[0, 1.1])
    fig.update_layout(showlegend=False)
    return _layout(fig, height)


def per_type_chart(
    by_type: Mapping[str, Mapping[str, Any]], metrics: Sequence[str], height: int = 250
) -> go.Figure:
    types = list(by_type)
    palette = [NEUTRAL, KEPT, ACCENT, MUTED]  # never the pass/fail colours
    fig = go.Figure()
    for i, metric in enumerate(metrics):
        fig.add_bar(name=metric, x=types, y=[by_type[t].get(metric) or 0.0 for t in types],
                    marker_color=palette[i % len(palette)])  # fmt: skip
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.1})
    fig.update_yaxes(range=[0, 1.1])
    return _layout(fig, height)


def type_compare_chart(rows: Sequence[TypeCompare], height: int = 260) -> go.Figure:
    """Baseline vs now for each query type and metric (grouped bars)."""
    labels = [f"{r.query_type} · {r.metric}" for r in rows]
    fig = go.Figure()
    fig.add_bar(name="baseline", x=labels, y=[r.before or 0.0 for r in rows],
                marker_color=MUTED)  # fmt: skip
    fig.add_bar(name="now", x=labels, y=[r.now for r in rows], marker_color=NEUTRAL)
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.12})
    fig.update_yaxes(range=[0, 1.1])
    return _layout(fig, height)


def session_time_chart(points: Sequence[SessionPoint], height: int = 220) -> go.Figure:
    """Seconds per question; refusals greyed out."""
    fig = go.Figure(
        go.Bar(
            x=[f"Q{p.number}" for p in points],
            y=[p.total_s for p in points],
            marker_color=[MUTED if p.refused else NEUTRAL for p in points],
            text=[f"{p.total_s:.1f}s" for p in points],
            textposition="outside",
            hovertext=[p.question for p in points],
            name="seconds",
        )
    )
    fig.update_yaxes(title_text="seconds")
    return _layout(fig, height)


def session_quality_chart(points: Sequence[SessionPoint], height: int = 220) -> go.Figure:
    """Citation coverage and grounding per question (gaps where they do not apply)."""
    x = [f"Q{p.number}" for p in points]
    fig = go.Figure()
    fig.add_scatter(x=x, y=[p.coverage for p in points], name="citation coverage",
                    mode="lines+markers", line={"color": NEUTRAL, "width": 2})  # fmt: skip
    fig.add_scatter(x=x, y=[p.grounding for p in points], name="grounding",
                    mode="lines+markers", line={"color": ACCENT, "width": 2})  # fmt: skip
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%")
    fig.update_layout(legend={"orientation": "h", "y": 1.15})
    return _layout(fig, height)
