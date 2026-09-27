"""Tests for playground.visuals (pure data behind the index-level charts)."""

import math

import pytest

from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import IndexSnapshot
from rag_eval_platform.playground.visuals import (
    ChunkSpan,
    IndexCounts,
    PipelineStats,
    angle_degrees,
    chunk_spans,
    dot_text,
    index_counts,
    ingest_flow_dot,
    meaning_map,
    phases_with_data,
    pipeline_dot,
    size_bins,
    vector_for,
)

FULL = PipelineStats(documents=2, chunks=40, candidates=20, top_k=5, cited=2, faithfulness=0.93)


def _node_line(dot: str, phase: str) -> str:
    return next(line for line in dot.splitlines() if line.strip().startswith(f"{phase} ["))


def test_phases_with_data_follow_the_pipeline() -> None:
    assert phases_with_data(PipelineStats()) == frozenset()
    assert phases_with_data(PipelineStats(documents=1, chunks=3)) == {"Ingest", "Embed"}
    assert phases_with_data(FULL) == {"Ingest", "Embed", "Retrieve", "Generate", "Evaluate"}


def test_pipeline_dot_highlights_the_active_phase_with_live_numbers() -> None:
    dot = pipeline_dot("Retrieve", FULL)

    assert dot.startswith("digraph pipeline {")
    assert "#d97706" in _node_line(dot, "Retrieve")  # active colour
    assert "top-5 of 20" in _node_line(dot, "Retrieve")
    assert "cited 2/5" in _node_line(dot, "Generate")
    assert "faithfulness 0.93" in _node_line(dot, "Evaluate")
    assert "#16a34a" in _node_line(dot, "Ingest")  # has data, not active
    assert 'label="40 chunks"' in dot  # Ingest -> Embed edge


def test_pipeline_dot_greys_out_phases_without_data() -> None:
    dot = pipeline_dot(None, PipelineStats(documents=1, chunks=3))

    assert "#9ca3af" in _node_line(dot, "Retrieve")
    assert "not judged" in _node_line(dot, "Evaluate")


def test_pipeline_dot_rejects_unknown_phases() -> None:
    with pytest.raises(ValueError, match="unknown phase"):
        pipeline_dot("Deploy", FULL)


def test_pipeline_dot_escapes_quotes() -> None:
    assert dot_text('say "hi" \\ bye') == 'say \\"hi\\" \\\\ bye'
    assert dot_text("x" * 100, limit=10) == "xxxxxxxxx…"


def _chunk(doc: str, index: int, start: int, text: str, page: int | None = None) -> Chunk:
    return Chunk(id=f"{doc}#{index}", doc_id=doc, index=index, text=text, start_index=start,
                 page=page)  # fmt: skip


CHUNKS = (
    _chunk("a.pdf", 0, 0, "abcdef", page=1),
    _chunk("a.pdf", 1, 4, "efghij", page=1),
    _chunk("a.pdf", 2, 10, "klmn", page=2),
    _chunk("b.md", 0, 0, "xyz"),
)


def test_index_counts() -> None:
    assert index_counts(CHUNKS) == IndexCounts(
        documents=2, pages=2, characters=17, chunks=4, mean_chars=19 / 4
    )
    assert index_counts(()) == IndexCounts(0, 0, 0, 0, 0.0)


def test_ingest_flow_dot_shows_each_stage_count() -> None:
    dot = ingest_flow_dot(index_counts(CHUNKS))

    for text in ("2 files", "2 PDF pages", "17 characters", "4 chunks"):
        assert text in dot


def test_chunk_spans_measure_overlap_with_the_previous_chunk() -> None:
    spans = chunk_spans(tuple(reversed(CHUNKS)))  # input order does not matter

    assert spans == (
        ChunkSpan("a.pdf", "a.pdf#0", 0, 0, 6, 0),
        ChunkSpan("a.pdf", "a.pdf#1", 1, 4, 10, 2),  # shares chars 4-5 with #0
        ChunkSpan("a.pdf", "a.pdf#2", 2, 10, 14, 0),
        ChunkSpan("b.md", "b.md#0", 0, 0, 3, 0),  # a new document starts fresh
    )


def test_size_bins_cover_every_size() -> None:
    bins = size_bins([100, 120, 480, 500], bins=2)

    assert [(b.start, b.end, b.count) for b in bins] == [(100, 300, 2), (300, 500, 2)]
    assert size_bins([]) == ()


def _snapshot(vectors: list[tuple[float, ...]]) -> IndexSnapshot:
    chunks = tuple(_chunk("d.md", i, i, f"text {i}") for i in range(len(vectors)))
    return IndexSnapshot(chunks, tuple(vectors))


def test_meaning_map_places_similar_chunks_close_and_projects_the_question() -> None:
    snapshot = _snapshot([(1.0, 0.0, 0.0), (0.9, 0.1, 0.0), (0.0, 0.0, 1.0)])

    points = meaning_map(snapshot, (1.0, 0.05, 0.0), {"d.md#1"})

    by_id = {p.chunk_id: p for p in points}
    assert [p.kind for p in points] == ["chunk", "retrieved", "chunk", "question"]
    question = (by_id["question"].x, by_id["question"].y)
    near = math.dist(question, (by_id["d.md#0"].x, by_id["d.md#0"].y))
    far = math.dist(question, (by_id["d.md#2"].x, by_id["d.md#2"].y))
    assert near < far
    assert all(p.z == 0.0 for p in points)  # 2-D map


def test_meaning_map_in_3d_has_depth() -> None:
    snapshot = _snapshot([(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)])

    points = meaning_map(snapshot, None, set(), dims=3)

    assert len(points) == 4
    assert any(p.z != 0.0 for p in points)


def test_meaning_map_samples_but_keeps_retrieved() -> None:
    snapshot = _snapshot([(float(i), float(i % 7), 1.0) for i in range(50)])

    points = meaning_map(snapshot, None, {"d.md#49", "d.md#3"}, max_points=10)

    ids = {p.chunk_id for p in points}
    assert len(points) == 10
    assert {"d.md#49", "d.md#3"} <= ids


def test_meaning_map_edge_cases() -> None:
    assert meaning_map(IndexSnapshot((), ()), (1.0,), set()) == ()
    single = meaning_map(_snapshot([(1.0, 2.0)]), None, set())
    assert len(single) == 1
    with pytest.raises(ValueError, match="dims"):
        meaning_map(_snapshot([(1.0, 2.0)]), None, set(), dims=4)


def test_vector_for_and_angle() -> None:
    snapshot = _snapshot([(1.0, 0.0), (0.0, 1.0)])

    assert vector_for(snapshot, "d.md#1") == (0.0, 1.0)
    assert vector_for(snapshot, "missing") is None
    assert angle_degrees((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)
    assert angle_degrees((1.0, 1.0), (2.0, 2.0)) == pytest.approx(0.0, abs=1e-3)
    with pytest.raises(ValueError, match="zero"):
        angle_degrees((0.0, 0.0), (1.0, 0.0))


def test_pipeline_dot_marks_the_active_phase_for_motion() -> None:
    dot = pipeline_dot("Retrieve", FULL)

    assert 'class="active"' in _node_line(dot, "Retrieve")
    assert 'class="active"' not in _node_line(dot, "Embed")
    flow = next(line for line in dot.splitlines() if "Embed -> Retrieve" in line)
    assert 'class="flow"' in flow  # the arrow feeding the active phase
    idle = next(line for line in dot.splitlines() if "Retrieve -> Generate" in line)
    assert 'class="flow"' not in idle
