"""Pure data behind the playground's index-level visuals: the pipeline diagram, chunks and
their overlaps, and the meaning map. No Streamlit or Plotly here, so it is unit-tested."""

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import fmean
from typing import Literal

import numpy as np
import numpy.typing as npt

from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import IndexSnapshot

PHASES = ("Ingest", "Embed", "Retrieve", "Generate", "Evaluate")

_ACTIVE = 'fillcolor="#fef3c7" color="#d97706" penwidth=2 fontcolor="#78350f"'
_DONE = 'fillcolor="#dcfce7" color="#16a34a" fontcolor="#14532d"'
_EMPTY = 'fillcolor="#f3f4f6" color="#d1d5db" fontcolor="#9ca3af"'


@dataclass(frozen=True)
class PipelineStats:
    documents: int = 0
    chunks: int = 0
    candidates: int = 0
    top_k: int = 0
    cited: int = 0
    faithfulness: float | None = None


def phases_with_data(stats: PipelineStats) -> frozenset[str]:
    ready: set[str] = set()
    if stats.chunks:
        ready |= {"Ingest", "Embed"}
    if stats.top_k:
        ready |= {"Retrieve", "Generate"}
    if stats.faithfulness is not None:
        ready.add("Evaluate")
    return frozenset(ready)


def dot_text(text: str, limit: int = 60) -> str:
    """Text safe inside a double-quoted DOT label, cut to ``limit`` characters."""
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text.replace("\\", "\\\\").replace('"', '\\"')


def pipeline_dot(active: str | None, stats: PipelineStats) -> str:
    """Graphviz source: the five phases, ``active`` highlighted, live numbers as captions."""
    if active is not None and active not in PHASES:
        raise ValueError(f"unknown phase: {active}")
    ready = phases_with_data(stats)
    captions = {
        "Ingest": f"{stats.documents} docs" if stats.chunks else "no documents",
        "Embed": f"{stats.chunks} vectors" if stats.chunks else "",
        "Retrieve": f"top-{stats.top_k} of {stats.candidates}" if stats.top_k else "",
        "Generate": f"cited {stats.cited}/{stats.top_k}" if stats.top_k else "",
        "Evaluate": (
            f"faithfulness {stats.faithfulness:.2f}"
            if stats.faithfulness is not None
            else "not judged"
        ),
    }
    lines = [
        "digraph pipeline {",
        '  rankdir=LR; bgcolor="transparent"; nodesep=0.3;',
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11];',
        '  edge [color="#9ca3af" fontname="Helvetica" fontsize=9 fontcolor="#6b7280"];',
    ]
    for number, phase in enumerate(PHASES, start=1):
        style = _ACTIVE if phase == active else _DONE if phase in ready else _EMPTY
        motion = ' class="active"' if phase == active else ""  # pulses, see live.LIVE_CSS
        caption = captions[phase]
        label = f"{number} · {phase}\\n{dot_text(caption)}" if caption else f"{number} · {phase}"
        lines.append(f'  {phase} [label="{label}" {style}{motion}];')
    edge_labels = [
        f"{stats.chunks} chunks" if stats.chunks else "",
        "cosine search" if stats.chunks else "",
        f"{stats.top_k} chunks" if stats.top_k else "",
        "answer" if stats.top_k else "",
    ]
    for (source, target), label in zip(pairwise(PHASES), edge_labels, strict=True):
        flow = ' class="flow"' if target == active else ""  # marching dashes into the active phase
        lines.append(f'  {source} -> {target} [label="{label}"{flow}];')
    lines.append("}")
    return "\n".join(lines)


@dataclass(frozen=True)
class IndexCounts:
    documents: int
    pages: int
    characters: int
    chunks: int
    mean_chars: float


def index_counts(chunks: Sequence[Chunk]) -> IndexCounts:
    """Counts for the ingest flow: files, PDF pages, text length and chunks."""
    pages: dict[str, set[int]] = {}
    ends: dict[str, int] = {}
    for c in chunks:
        if c.page is not None:
            pages.setdefault(c.doc_id, set()).add(c.page)
        ends[c.doc_id] = max(ends.get(c.doc_id, 0), c.start_index + len(c.text))
    return IndexCounts(
        documents=len(ends),
        pages=sum(len(p) for p in pages.values()),
        characters=sum(ends.values()),
        chunks=len(chunks),
        mean_chars=fmean(len(c.text) for c in chunks) if chunks else 0.0,
    )


def ingest_flow_dot(counts: IndexCounts) -> str:
    """Graphviz source: files -> pages -> text -> chunks, with counts."""
    stages = [
        ("files", f"📄 {counts.documents} files"),
        ("pages", f"{counts.pages} PDF pages" if counts.pages else "no PDF pages"),
        ("text", f"{counts.characters:,} characters"),
        ("chunks", f"✂️ {counts.chunks} chunks\\n~{counts.mean_chars:.0f} chars each"),
    ]
    lines = [
        "digraph ingest {",
        '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box style="rounded,filled" fillcolor="#eef2ff" color="#6366f1" '
        'fontname="Helvetica" fontsize=11];',
        '  edge [color="#6366f1"];',
    ]
    lines += [f'  {name} [label="{label}"];' for name, label in stages]
    lines.append("  files -> pages -> text -> chunks;")
    lines.append("}")
    return "\n".join(lines)


@dataclass(frozen=True)
class ChunkSpan:
    doc_id: str
    chunk_id: str
    index: int
    start: int
    end: int
    overlap: int  # characters shared with the previous chunk of the same document


def chunk_spans(chunks: Sequence[Chunk]) -> tuple[ChunkSpan, ...]:
    """Where each chunk sits in its document, and how much it overlaps the previous one."""
    spans: list[ChunkSpan] = []
    previous: Chunk | None = None
    for c in sorted(chunks, key=lambda c: (c.doc_id, c.index)):
        end = c.start_index + len(c.text)
        overlap = 0
        if previous is not None and previous.doc_id == c.doc_id:
            previous_end = previous.start_index + len(previous.text)
            overlap = max(0, previous_end - c.start_index)
        spans.append(ChunkSpan(c.doc_id, c.id, c.index, c.start_index, end, overlap))
        previous = c
    return tuple(spans)


@dataclass(frozen=True)
class SizeBin:
    start: int
    end: int
    count: int


def size_bins(sizes: Sequence[int], bins: int = 12) -> tuple[SizeBin, ...]:
    """A histogram of chunk sizes (characters)."""
    if not sizes:
        return ()
    counts, edges = np.histogram(np.asarray(sizes), bins=bins)
    return tuple(
        SizeBin(round(float(edges[i])), round(float(edges[i + 1])), int(count))
        for i, count in enumerate(counts)
    )


@dataclass(frozen=True)
class MapPoint:
    chunk_id: str  # "question" for the question itself
    kind: Literal["chunk", "retrieved", "question"]
    x: float
    y: float
    z: float
    text: str


def meaning_map(
    snapshot: IndexSnapshot,
    query: Sequence[float] | None,
    retrieved_ids: Collection[str],
    *,
    dims: int = 2,
    max_points: int = 3000,
    seed: int = 0,
) -> tuple[MapPoint, ...]:
    """Chunks (and the question) squeezed from embedding space to 2 or 3 dimensions with PCA.

    Big indexes are sampled down to ``max_points``, always keeping the retrieved chunks.
    """
    if dims not in (2, 3):
        raise ValueError(f"dims must be 2 or 3, got {dims}")
    if not snapshot.chunks:
        return ()
    keep = _sample(snapshot, retrieved_ids, max_points, seed)
    matrix = np.asarray([snapshot.vectors[i] for i in keep], dtype=np.float64)
    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    components = vt[:dims]

    def coords(vector: Sequence[float]) -> tuple[float, float, float]:
        centred = np.asarray(vector, dtype=np.float64) - mean
        values: npt.NDArray[np.float64] = centred @ components.T
        padded = [float(v) for v in values] + [0.0, 0.0, 0.0]
        return padded[0], padded[1], padded[2]

    points = []
    for i in keep:
        chunk = snapshot.chunks[i]
        kind: Literal["chunk", "retrieved"] = "retrieved" if chunk.id in retrieved_ids else "chunk"
        points.append(MapPoint(chunk.id, kind, *coords(snapshot.vectors[i]), text=chunk.text[:160]))
    if query is not None:
        points.append(MapPoint("question", "question", *coords(query), text="your question"))
    return tuple(points)


def _sample(
    snapshot: IndexSnapshot, retrieved_ids: Collection[str], max_points: int, seed: int
) -> list[int]:
    total = len(snapshot.chunks)
    if total <= max_points:
        return list(range(total))
    must = [i for i, c in enumerate(snapshot.chunks) if c.id in retrieved_ids]
    must_set = set(must)
    others = [i for i in range(total) if i not in must_set]
    rng = np.random.default_rng(seed)
    picked = rng.choice(others, size=max(0, max_points - len(must)), replace=False)
    return sorted([*must, *(int(i) for i in picked)])


def vector_for(snapshot: IndexSnapshot, chunk_id: str) -> tuple[float, ...] | None:
    for chunk, vector in zip(snapshot.chunks, snapshot.vectors, strict=True):
        if chunk.id == chunk_id:
            return vector
    return None


def angle_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    """The angle between two vectors: 0° = same direction, 90° = unrelated."""
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    norms = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if norms == 0.0:
        raise ValueError("cannot measure the angle of a zero vector")
    cosine = float(np.clip(va @ vb / norms, -1.0, 1.0))
    return math.degrees(math.acos(cosine))
