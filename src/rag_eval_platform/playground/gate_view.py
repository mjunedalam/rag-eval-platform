"""Pure pieces of the ⑥ Gate tab: the CI flow diagram, flipped questions, comparisons."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rag_eval_platform.config.settings import ChunkStrategy, Settings
from rag_eval_platform.evaluation.baselines import Baseline
from rag_eval_platform.evaluation.evaluator import RetrievalReport
from rag_eval_platform.evaluation.gate import GateResult

GATE_PREVIEW_COLLECTION = "gate_preview"  # the what-if index; never the real collections
GATE_STEPS = ("Pull request", "Build index", "Evaluate retrieval", "Generation baseline",
              "Compare")  # fmt: skip
_GREEN, _RED, _GREY = "#16a34a", "#dc2626", "#9ca3af"
_FILL = {_GREEN: "#dcfce7", _RED: "#fee2e2", _GREY: "#f3f4f6"}


@dataclass(frozen=True)
class Flip:
    example_id: str
    question: str
    query_type: str
    relevant_doc_ids: tuple[str, ...]
    before_hit: bool
    now_hit: bool


@dataclass(frozen=True)
class TypeCompare:
    query_type: str
    metric: str
    before: float | None
    now: float


def gate_flow_dot(result: GateResult | None) -> str:
    """Graphviz source for the CI flow, each step coloured by the latest result."""
    if result is None:
        colours = [_GREY] * len(GATE_STEPS)
        verdict, verdict_colour = "Pass or block", _GREY
    else:
        retrieval_ok = all(c.status == "pass" for c in result.checks if c.group == "retrieval")
        generation_ok = all(c.status == "pass" for c in result.checks
                            if c.group == "generation")  # fmt: skip
        final = _GREEN if result.passed else _RED
        colours = [_GREEN, _GREEN, _GREEN if retrieval_ok else _RED,
                   _GREEN if generation_ok else _RED, final]  # fmt: skip
        verdict = "✅ Pass: may merge" if result.passed else "⛔ Block: fix first"
        verdict_colour = final
    nodes = [
        f'  s{i} [label="{step}", color="{colour}", fillcolor="{_FILL[colour]}"];'
        for i, (step, colour) in enumerate(zip(GATE_STEPS, colours, strict=True))
    ]
    edges = [f"  s{i} -> s{i + 1};" for i in range(len(GATE_STEPS) - 1)]
    return "\n".join([
        "digraph gate {", '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [color="#9ca3af"];', *nodes,
        f'  verdict [label="{verdict}", color="{verdict_colour}", '
        f'fillcolor="{_FILL[verdict_colour]}", penwidth=2];',
        *edges, f"  s{len(GATE_STEPS) - 1} -> verdict;", "}",
    ])  # fmt: skip


def flipped_questions(
    retrieval: RetrievalReport, baseline: Baseline | None, questions: Mapping[str, str]
) -> tuple[Flip, ...]:
    """Questions whose every relevant document was found before but not now, or back."""
    if baseline is None:
        return ()
    before = {
        e.get("example_id"): _recall(e)
        for e in baseline.report.get("examples", [])
        if isinstance(e, dict)
    }
    flips = []
    for result in retrieval.examples:
        old = before.get(result.example_id)
        if old is None:
            continue
        before_hit, now_hit = old >= 1.0, result.scores.recall >= 1.0
        if before_hit != now_hit:
            flips.append(Flip(result.example_id, questions.get(result.example_id, ""),
                              result.query_type, result.relevant_doc_ids, before_hit,
                              now_hit))  # fmt: skip
    return tuple(flips)


def type_comparison(
    retrieval: RetrievalReport, baseline: Baseline | None
) -> tuple[TypeCompare, ...]:
    saved = baseline.report.get("by_query_type", {}) if baseline else {}
    rows = []
    for query_type, scores in sorted(retrieval.by_query_type.items()):
        old = saved.get(query_type, {}) if isinstance(saved, dict) else {}
        for metric in ("recall", "mrr", "ndcg"):
            value = old.get(metric) if isinstance(old, dict) else None
            before = float(value) if isinstance(value, int | float) else None
            rows.append(TypeCompare(query_type, metric, before, getattr(scores, metric)))
    return tuple(rows)


def preview_settings(
    settings: Settings,
    *,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    rerank: bool,
    rerank_candidates: int,
) -> Settings:
    """The settings a pull request would have if it changed only these values."""
    return settings.model_copy(update={
        "chunk_strategy": strategy, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "top_k": top_k, "rerank": rerank, "rerank_candidates": rerank_candidates,
    })  # fmt: skip


def _recall(example: Mapping[str, Any]) -> float | None:
    scores = example.get("scores", {})
    value = scores.get("recall") if isinstance(scores, dict) else None
    return float(value) if isinstance(value, int | float) else None
