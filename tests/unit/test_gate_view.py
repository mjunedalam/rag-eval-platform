"""Tests for playground.gate_view: what the ⑥ Gate tab draws."""

from pathlib import Path

from tests.unit.test_baselines import BASE
from tests.unit.test_gate import (
    LIMITS,
    generation_baseline,
    retrieval,
    retrieval_baseline,
)

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation.baselines import Baseline
from rag_eval_platform.evaluation.evaluator import (
    ExampleResult,
    RetrievalReport,
    RetrievalThresholds,
)
from rag_eval_platform.evaluation.gate import run_gate
from rag_eval_platform.evaluation.metrics import RetrievalScores
from rag_eval_platform.playground.core import PLAYGROUND_COLLECTION
from rag_eval_platform.playground.gate_view import (
    GATE_PREVIEW_COLLECTION,
    Flip,
    TypeCompare,
    flipped_questions,
    gate_flow_dot,
    preview_settings,
    type_comparison,
)


def test_the_what_if_gate_never_touches_the_real_collections() -> None:
    assert GATE_PREVIEW_COLLECTION not in {get_settings().collection_name, PLAYGROUND_COLLECTION}


def test_flow_is_grey_before_any_run_and_coloured_after() -> None:
    assert "#16a34a" not in gate_flow_dot(None)  # no green before a run

    passed = gate_flow_dot(run_gate(retrieval(), None, generation_baseline(), BASE, LIMITS))
    blocked = gate_flow_dot(run_gate(retrieval(recall=0.5), None, None, BASE, LIMITS))

    assert "Pass" in passed
    assert "#16a34a" in passed
    assert "Block" in blocked
    assert "#dc2626" in blocked


def example(example_id: str, recall: float) -> ExampleResult:
    return ExampleResult(example_id, "short", ("a.md",), ("a.md",),
                         RetrievalScores(0.2, recall, recall, recall))  # fmt: skip


def test_flipped_questions_list_hits_that_became_misses_and_back() -> None:
    examples = (example("q1", 0.0), example("q2", 1.0), example("q3", 1.0))
    now = RetrievalReport(5, RetrievalThresholds(0.8, 0.7, 0.7), RetrievalScores(0, 0, 0, 0),
                          {}, examples, ())  # fmt: skip
    before = Baseline({"examples": [
        {"example_id": "q1", "scores": {"recall": 1.0}},
        {"example_id": "q2", "scores": {"recall": 0.5}},
        {"example_id": "q3", "scores": {"recall": 1.0}},
    ]}, BASE.to_dict())  # fmt: skip

    flips = flipped_questions(now, before, {"q1": "Q one?", "q2": "Q two?"})

    assert flips == (
        Flip("q1", "Q one?", "short", ("a.md",), before_hit=True, now_hit=False),
        Flip("q2", "Q two?", "short", ("a.md",), before_hit=False, now_hit=True),
    )
    assert flipped_questions(now, None, {}) == ()


def test_type_comparison_pairs_baseline_and_now() -> None:
    rows = type_comparison(retrieval(recall=0.8), retrieval_baseline(recall=0.93))

    assert TypeCompare("short", "recall", 0.93, 0.8) in rows
    assert all(r.before is None for r in type_comparison(retrieval(), None))


def test_preview_settings_apply_the_sidebar_choices(tmp_path: Path) -> None:
    settings = preview_settings(get_settings(), strategy="fixed", chunk_size=300,
                                chunk_overlap=30, top_k=3, rerank=False,
                                rerank_candidates=20)  # fmt: skip

    assert (settings.chunk_strategy, settings.chunk_size, settings.top_k) == ("fixed", 300, 3)
