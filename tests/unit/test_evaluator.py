"""Tests for evaluation.evaluator with a fake retriever."""

import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.evaluator import (
    RetrievalThresholds,
    check_thresholds,
    evaluate_retrieval,
    report_to_dict,
)
from rag_eval_platform.evaluation.golden_dataset import GoldenExample
from rag_eval_platform.evaluation.metrics import RetrievalScores
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult


def example(id_: str, question: str, relevant: list[str], query_type: str) -> GoldenExample:
    return GoldenExample.model_validate(
        {
            "id": id_,
            "question": question,
            "expected_answer": "answer",
            "relevant_doc_ids": relevant,
            "query_type": query_type,
        }
    )


def hit(doc_id: str, n: int = 0) -> SearchResult:
    chunk = Chunk(id=f"{doc_id}#{n}", doc_id=doc_id, index=n, text="t", start_index=0)
    return SearchResult(chunk=chunk, score=0.5)


class FakeRetriever:
    def __init__(self, answers: dict[str, list[SearchResult]]) -> None:
        self.answers = answers

    def retrieve(self, query: str) -> list[SearchResult]:
        return self.answers[query]


EXAMPLES = (
    example("q1", "perfect?", ["a.md"], "short"),
    example("q2", "second?", ["b.md"], "short"),
    example("q3", "two docs?", ["a.md", "c.md"], "multi_hop"),
)
RETRIEVER = FakeRetriever(
    {
        # two chunks of a.md collapse into one document hit at rank 1
        "perfect?": [hit("a.md", 0), hit("a.md", 1), hit("x.md")],
        "second?": [hit("x.md"), hit("b.md")],
        "two docs?": [hit("a.md"), hit("y.md")],
    }
)
LENIENT = RetrievalThresholds(min_recall_at_k=0.0, min_mrr=0.0, min_ndcg_at_k=0.0)


def test_scores_each_example_on_distinct_doc_ids() -> None:
    report = evaluate_retrieval(EXAMPLES, RETRIEVER, k=3, thresholds=LENIENT)

    q1, q2, q3 = report.examples
    assert q1.retrieved_doc_ids == ("a.md", "x.md")
    assert (q1.scores.recall, q1.scores.mrr) == (1.0, 1.0)
    assert (q2.scores.recall, q2.scores.mrr) == (1.0, 0.5)
    assert (q3.scores.recall, q3.scores.mrr) == (0.5, 1.0)


def test_averages_overall_and_per_query_type() -> None:
    report = evaluate_retrieval(EXAMPLES, RETRIEVER, k=3, thresholds=LENIENT)

    assert report.overall.recall == pytest.approx(2.5 / 3)
    assert report.overall.mrr == pytest.approx(2.5 / 3)
    assert report.by_query_type["short"].mrr == pytest.approx(0.75)
    assert report.by_query_type["multi_hop"].recall == pytest.approx(0.5)
    assert report.passed


def test_fails_when_a_metric_is_below_threshold() -> None:
    strict = RetrievalThresholds(min_recall_at_k=0.9, min_mrr=0.5, min_ndcg_at_k=0.5)

    report = evaluate_retrieval(EXAMPLES, RETRIEVER, k=3, thresholds=strict)

    assert not report.passed
    assert report.failures == ("recall@k 0.833 < 0.900",)


def test_check_thresholds_lists_every_failing_metric() -> None:
    scores = RetrievalScores(precision=0.1, recall=0.5, mrr=0.4, ndcg=0.9)
    thresholds = RetrievalThresholds(min_recall_at_k=0.8, min_mrr=0.7, min_ndcg_at_k=0.7)

    assert check_thresholds(scores, thresholds) == ("recall@k 0.500 < 0.800", "mrr 0.400 < 0.700")


def test_thresholds_come_from_settings() -> None:
    thresholds = RetrievalThresholds.from_settings(
        Settings(min_recall_at_k=0.9, min_mrr=0.6, min_ndcg_at_k=0.5)
    )

    assert thresholds == RetrievalThresholds(0.9, 0.6, 0.5)


def test_rejects_empty_example_set() -> None:
    with pytest.raises(ValueError, match="at least one"):
        evaluate_retrieval((), RETRIEVER, k=3, thresholds=LENIENT)


def test_report_serialises_to_plain_dict() -> None:
    report = evaluate_retrieval(EXAMPLES, RETRIEVER, k=3, thresholds=LENIENT)

    data = report_to_dict(report)

    assert data["k"] == 3
    assert data["passed"] is True
    assert data["overall"]["recall"] == pytest.approx(2.5 / 3)
    assert data["by_query_type"]["multi_hop"]["recall"] == 0.5
    assert data["examples"][0]["retrieved_doc_ids"] == ["a.md", "x.md"]
    assert data["thresholds"] == {"min_recall_at_k": 0.0, "min_mrr": 0.0, "min_ndcg_at_k": 0.0}
