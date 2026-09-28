"""Tests for evaluation.gate: the rules that pass or block a change, and its report."""

from dataclasses import replace

from tests.unit.test_baselines import BASE

from rag_eval_platform.evaluation.baselines import Baseline
from rag_eval_platform.evaluation.evaluator import RetrievalReport, RetrievalThresholds
from rag_eval_platform.evaluation.fingerprint import EvalFingerprint, FieldChange
from rag_eval_platform.evaluation.gate import (
    GATE_MARKER,
    GateLimits,
    gate_error_markdown,
    gate_from_dict,
    gate_markdown,
    gate_to_dict,
    run_gate,
)
from rag_eval_platform.evaluation.metrics import RetrievalScores

LIMITS = GateLimits(0.80, 0.70, 0.70, 0.85, 0.80, max_drop=0.02)


def retrieval(recall: float = 0.93, mrr: float = 0.88, ndcg: float = 0.88) -> RetrievalReport:
    overall = RetrievalScores(precision=0.3, recall=recall, mrr=mrr, ndcg=ndcg)
    return RetrievalReport(
        k=5, thresholds=RetrievalThresholds(0.8, 0.7, 0.7), overall=overall,
        by_query_type={"short": overall}, examples=(), failures=(),
    )  # fmt: skip


def retrieval_baseline(recall: float = 0.93, mrr: float = 0.88, ndcg: float = 0.88) -> Baseline:
    scores = {"precision": 0.3, "recall": recall, "mrr": mrr, "ndcg": ndcg}
    report = {"overall": scores, "by_query_type": {"short": scores}, "examples": []}
    return Baseline(report, BASE.to_dict())


def generation_baseline(
    faith: float | None = 0.93, relevance: float | None = 0.81, fp: EvalFingerprint = BASE
) -> Baseline:
    report = {"overall": {"faithfulness": faith, "answer_relevance": relevance}}
    return Baseline(report, fp.to_dict())


def test_everything_in_order_passes() -> None:
    result = run_gate(retrieval(), retrieval_baseline(), generation_baseline(), BASE, LIMITS)

    assert result.passed
    assert [c.name for c in result.checks] == [
        "recall@k", "mrr", "ndcg@k", "freshness", "faithfulness", "answer_relevance",
    ]  # fmt: skip
    assert result.changes == ()


def test_a_metric_below_its_threshold_blocks() -> None:
    result = run_gate(retrieval(recall=0.79), retrieval_baseline(), generation_baseline(), BASE,
                      LIMITS)  # fmt: skip

    recall = result.checks[0]
    assert not result.passed
    assert (recall.status, recall.reason) == ("fail", "0.790 < 0.80")


def test_a_drop_beyond_max_drop_blocks_even_above_the_threshold() -> None:
    result = run_gate(retrieval(mrr=0.84), retrieval_baseline(mrr=0.88), generation_baseline(),
                      BASE, LIMITS)  # fmt: skip

    mrr = result.checks[1]
    assert mrr.status == "fail"
    assert mrr.reason == "dropped 0.040 from 0.880 (max 0.02)"


def test_a_drop_of_exactly_max_drop_passes() -> None:
    result = run_gate(retrieval(recall=0.91), retrieval_baseline(recall=0.93),
                      generation_baseline(), BASE, LIMITS)  # fmt: skip

    assert result.checks[0].status == "pass"


def test_without_a_retrieval_baseline_only_thresholds_apply() -> None:
    result = run_gate(retrieval(), None, generation_baseline(), BASE, LIMITS)

    assert result.passed
    assert "no baseline to compare" in result.checks[0].reason


def test_a_missing_generation_baseline_blocks() -> None:
    result = run_gate(retrieval(), retrieval_baseline(), None, BASE, LIMITS)

    by_name = {c.name: c for c in result.checks}
    assert not result.passed
    assert "no generation baseline" in by_name["freshness"].reason
    assert by_name["faithfulness"].reason == "never scored"


def test_a_stale_generation_baseline_blocks_and_lists_what_changed() -> None:
    stale = generation_baseline(fp=replace(BASE, chunk_size=400))

    result = run_gate(retrieval(), retrieval_baseline(), stale, BASE, LIMITS)

    freshness = result.checks[3]
    assert result.changes == (FieldChange("chunk_size", 400, 800),)
    assert freshness.status == "fail"
    assert "chunk_size: 400 → 800" in freshness.reason
    assert "--save-baseline" in freshness.reason


def test_null_or_missing_generation_metrics_fail_as_never_scored() -> None:
    nulls = run_gate(retrieval(), None, generation_baseline(faith=None), BASE, LIMITS)
    empty = run_gate(retrieval(), None, Baseline({}, BASE.to_dict()), BASE, LIMITS)

    assert nulls.checks[4].reason == "never scored"
    assert [c.reason for c in empty.checks[4:]] == ["never scored", "never scored"]


def test_low_generation_scores_block() -> None:
    result = run_gate(retrieval(), None, generation_baseline(faith=0.80), BASE, LIMITS)

    assert (result.checks[4].status, result.checks[4].reason) == ("fail", "0.800 < 0.85")


def test_markdown_starts_with_the_marker_and_shows_before_and_after() -> None:
    result = run_gate(retrieval(recall=0.79), retrieval_baseline(), None, BASE, LIMITS)

    text = gate_markdown(result, retrieval(recall=0.79), retrieval_baseline())

    assert text.startswith(GATE_MARKER)
    assert "⛔ Evaluation gate blocked this change" in text
    assert "| recall@k | 0.930 | 0.790 | -0.140 | 0.80 | ⛔ 0.790 < 0.80 |" in text
    assert "| short | 0.930 → 0.790 | 0.880 → 0.880 | 0.880 → 0.880 |" in text


def test_markdown_lists_stale_changes() -> None:
    stale = generation_baseline(fp=replace(BASE, top_k=3))
    result = run_gate(retrieval(), None, stale, BASE, LIMITS)

    text = gate_markdown(result, retrieval(), None)

    assert "- `top_k: 3 → 5`" in text
    assert "| short | - → 0.930 | - → 0.880 | - → 0.880 |" in text  # no retrieval baseline


def test_result_round_trips_through_json_form() -> None:
    result = run_gate(retrieval(), None, generation_baseline(fp=replace(BASE, top_k=3)), BASE,
                      LIMITS)  # fmt: skip

    assert gate_from_dict(gate_to_dict(result)) == result


def test_error_markdown_replaces_the_comment_and_escapes_table_pipes() -> None:
    text = gate_error_markdown("Chroma | down")

    assert text.startswith(GATE_MARKER)
    assert "⚠️ Evaluation gate could not run" in text
    assert "Chroma \\| down" in text
