"""Tests for evaluation.metrics."""

import math

import pytest

from rag_eval_platform.evaluation.metrics import (
    RetrievalScores,
    mean_scores,
    mean_scores_by_group,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    score_retrieval,
)

RELEVANT = {"a", "c"}


class TestPrecisionAtK:
    def test_counts_relevant_share_of_retrieved_top_k(self) -> None:
        assert precision_at_k(["a", "b", "c", "d"], RELEVANT, k=4) == 0.5

    def test_ignores_results_beyond_k(self) -> None:
        assert precision_at_k(["a", "b", "c"], RELEVANT, k=2) == 0.5

    def test_divides_by_retrieved_count_when_fewer_than_k(self) -> None:
        assert precision_at_k(["a"], RELEVANT, k=5) == 1.0

    def test_is_zero_when_nothing_retrieved(self) -> None:
        assert precision_at_k([], RELEVANT, k=5) == 0.0


class TestRecallAtK:
    def test_counts_share_of_relevant_found(self) -> None:
        assert recall_at_k(["a", "b"], RELEVANT, k=2) == 0.5

    def test_is_one_when_all_relevant_found(self) -> None:
        assert recall_at_k(["c", "b", "a"], RELEVANT, k=3) == 1.0

    def test_ignores_results_beyond_k(self) -> None:
        assert recall_at_k(["b", "d", "a"], RELEVANT, k=2) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        ("retrieved", "expected"),
        [(["a", "b"], 1.0), (["b", "a"], 0.5), (["b", "d", "e", "c"], 0.25), (["b", "d"], 0.0)],
    )
    def test_is_inverse_rank_of_first_relevant(self, retrieved: list[str], expected: float) -> None:
        assert reciprocal_rank(retrieved, RELEVANT, k=10) == expected

    def test_relevant_beyond_k_scores_zero(self) -> None:
        assert reciprocal_rank(["b", "d", "a"], RELEVANT, k=2) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["a", "c", "b"], RELEVANT, k=3) == pytest.approx(1.0)

    def test_matches_hand_computed_value(self) -> None:
        # relevant at ranks 2 and 3; ideal has them at ranks 1 and 2
        dcg = 1 / math.log2(3) + 1 / math.log2(4)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)

        assert ndcg_at_k(["b", "a", "c"], RELEVANT, k=3) == pytest.approx(dcg / idcg)

    def test_ideal_is_capped_at_k(self) -> None:
        # only one slot, filled with a relevant doc: best possible
        assert ndcg_at_k(["a", "c"], RELEVANT, k=1) == pytest.approx(1.0)

    def test_no_relevant_scores_zero(self) -> None:
        assert ndcg_at_k(["b", "d"], RELEVANT, k=2) == 0.0


class TestDuplicateDocIds:
    def test_duplicates_are_collapsed_in_rank_order(self) -> None:
        # chunks from the same document map to the same doc id
        retrieved = ["a", "a", "b", "a", "c"]

        scores = score_retrieval(retrieved, RELEVANT, k=3)

        assert scores == score_retrieval(["a", "b", "c"], RELEVANT, k=3)
        assert scores.precision == pytest.approx(2 / 3)
        assert scores.recall == 1.0


@pytest.mark.parametrize(
    "metric", [precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k, score_retrieval]
)
class TestInputValidation:
    def test_rejects_k_below_one(self, metric: object) -> None:
        with pytest.raises(ValueError, match="k must be"):
            metric(["a"], RELEVANT, k=0)  # type: ignore[operator]

    def test_rejects_empty_relevant_set(self, metric: object) -> None:
        with pytest.raises(ValueError, match="relevant"):
            metric(["a"], set(), k=1)  # type: ignore[operator]


def test_score_retrieval_bundles_all_metrics() -> None:
    scores = score_retrieval(["b", "a"], RELEVANT, k=2)

    assert (scores.precision, scores.recall, scores.mrr) == (0.5, 0.5, 0.5)
    assert scores.ndcg == ndcg_at_k(["b", "a"], RELEVANT, k=2)


def test_mean_scores_averages_each_metric() -> None:
    result = mean_scores([RetrievalScores(1.0, 0.5, 1.0, 0.8), RetrievalScores(0.0, 0.5, 0.0, 0.4)])

    assert (result.precision, result.recall, result.mrr) == (0.5, 0.5, 0.5)
    assert result.ndcg == pytest.approx(0.6)


def test_mean_scores_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mean_scores([])


def test_mean_scores_by_group_averages_each_group_separately() -> None:
    perfect = RetrievalScores(1.0, 1.0, 1.0, 1.0)
    miss = RetrievalScores(0.0, 0.0, 0.0, 0.0)

    result = mean_scores_by_group([("short", perfect), ("multi_hop", miss), ("short", miss)])

    assert result == {
        "short": RetrievalScores(0.5, 0.5, 0.5, 0.5),
        "multi_hop": miss,
    }
