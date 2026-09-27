"""Tests for playground.charts: each builder returns a figure with the expected data."""

import pytest

pytest.importorskip("plotly")

from rag_eval_platform.playground import charts
from rag_eval_platform.playground.query_visuals import (
    INVALID_COLOR,
    PromptBlock,
    RerankMove,
    SimilarityRow,
)
from rag_eval_platform.playground.visuals import ChunkSpan, MapPoint, SizeBin

ROWS = (
    SimilarityRow(1, "a#0", 0.8, kept=True, cited=True),
    SimilarityRow(2, "b#0", 0.6, kept=True, cited=False),
    SimilarityRow(3, "c#0", 0.4, kept=False, cited=False),
)


def test_chunk_size_chart_grows_with_progress() -> None:
    bins = (SizeBin(100, 200, 4), SizeBin(200, 300, 2))

    half = charts.chunk_size_chart(bins, progress=0.5, height=200)

    assert list(half.data[0].y) == [2.0, 1.0]
    assert half.layout.yaxis.range[1] >= 4  # fixed axis, so bars visibly grow
    assert half.layout.height == 200


def test_chunk_layout_chart_draws_chunks_and_overlaps() -> None:
    spans = (ChunkSpan("d", "d#0", 0, 0, 6, 0), ChunkSpan("d", "d#1", 1, 4, 10, 2))

    fig = charts.chunk_layout_chart(spans, height=200)

    assert [t.name for t in fig.data] == ["chunk", "overlap"]
    assert list(fig.data[1].x) == [2]
    assert list(fig.data[1].base) == [4]


def test_meaning_map_chart_2d_and_3d() -> None:
    points = (
        MapPoint("a#0", "chunk", 1.0, 2.0, 0.0, "a"),
        MapPoint("b#0", "retrieved", -1.0, 0.0, 0.0, "b"),
        MapPoint("question", "question", 0.5, 0.5, 0.0, "q"),
    )

    flat = charts.meaning_map_chart(points, dims=2, progress=0.5, height=300)
    deep = charts.meaning_map_chart(points, dims=3, progress=1.0, height=300)

    assert [t.name for t in flat.data] == ["chunks", "retrieved", "your question"]
    assert list(flat.data[0].x) == [0.5]  # spread grows from the centre
    assert deep.data[0].type == "scatter3d"


def test_embedding_strip_chart_has_one_row_per_vector() -> None:
    fig = charts.embedding_strip_chart({"question": [0.1, -0.2], "a#0": [0.3, 0.0]}, height=150)

    assert list(fig.data[0].y) == ["question", "a#0"]


def test_similarity_chart_cutoff_without_reranking() -> None:
    fig = charts.similarity_chart(ROWS, cutoff=2, progress=1.0, height=250)

    assert list(fig.data[0].y) == [0.8, 0.6, 0.4]
    assert len(fig.layout.shapes) == 1
    assert fig.layout.shapes[0].x0 == pytest.approx(1.5)


def test_similarity_chart_cutoff_hidden_with_reranking() -> None:
    fig = charts.similarity_chart(ROWS, cutoff=None, progress=1.0, height=250)

    assert len(fig.layout.shapes) == 0


def test_rerank_chart_has_a_line_per_candidate() -> None:
    fig = charts.rerank_chart((RerankMove("a#0", 1, 2), RerankMove("b#0", 2, None)), height=250)

    assert len(fig.data) == 2
    assert list(fig.data[1].y) == [2, 3]  # dropped: falls below the kept ranks


def test_funnel_and_prompt_and_timing_charts() -> None:
    funnel = charts.funnel_chart((("all chunks", 40), ("candidates", 20)), height=200)
    prompt = charts.prompt_chart(
        (PromptBlock("rules", "System rules", "x" * 10), PromptBlock("question", "Question", "q?")),
        height=150,
    )
    timing = charts.timing_chart(
        (("retrieve", 500.0), ("generate", 4000.0)), progress=0.5, height=120
    )

    assert list(funnel.data[0].text) == ["40", "20"]  # true counts as labels
    assert [t.x[0] for t in prompt.data] == [10, 2]
    assert [t.x[0] for t in timing.data] == [0.25, 2.0]


def test_gauge_and_report_charts() -> None:
    gauge = charts.gauge_chart("faithfulness", 0.9, threshold=0.85, progress=0.5, height=180)
    scores = charts.scores_chart(
        ["faithfulness", "relevance"], [0.9, 0.7], [0.85, None], height=200
    )
    per_type = charts.per_type_chart(
        {"short": {"faithfulness": 0.9, "answer_relevance": 0.8}},
        ["faithfulness", "answer_relevance"],
        height=200,
    )

    assert gauge.data[0].value == pytest.approx(0.45)
    assert gauge.data[0].gauge.threshold.value == 0.85
    assert list(scores.data[0].y) == [0.9, 0.7]
    assert list(scores.data[1].y) == [0.85]  # only metrics with a pass mark
    assert [t.name for t in per_type.data] == ["faithfulness", "answer_relevance"]


def test_funnel_keeps_small_stages_visible() -> None:
    fig = charts.funnel_chart((("all chunks", 1505), ("sent to the model", 5), ("cited", 2)), 200)

    widths = list(fig.data[0].x)
    assert widths[0] > widths[1] > widths[2] > 0
    assert widths[0] / widths[2] < 10  # log scale: 2 cited chunks stay visible beside 1505


def test_per_type_chart_never_uses_the_fail_colour() -> None:
    fig = charts.per_type_chart(
        {"short": {"a": 0.9, "b": 0.8, "c": 0.7}}, ["a", "b", "c"], height=200
    )

    assert all(t.marker.color not in (None, INVALID_COLOR) for t in fig.data)
