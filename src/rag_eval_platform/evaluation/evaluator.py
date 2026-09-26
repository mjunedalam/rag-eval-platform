"""Run the golden dataset and check score thresholds.

Each golden question goes through the retriever; the retrieved chunks are mapped to
their source documents and scored against ``relevant_doc_ids`` (see metrics.py). The
averages are compared with the thresholds from settings, which is what the CI
evaluation gate enforces.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Self

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.golden_dataset import GoldenExample
from rag_eval_platform.evaluation.metrics import (
    RetrievalScores,
    mean_scores,
    mean_scores_by_group,
    score_retrieval,
)
from rag_eval_platform.retrieval.vector_store import SearchResult


class _Retriever(Protocol):
    def retrieve(self, query: str) -> list[SearchResult]: ...


@dataclass(frozen=True)
class RetrievalThresholds:
    min_recall_at_k: float
    min_mrr: float
    min_ndcg_at_k: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.min_recall_at_k, settings.min_mrr, settings.min_ndcg_at_k)


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    query_type: str
    relevant_doc_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    scores: RetrievalScores


@dataclass(frozen=True)
class RetrievalReport:
    k: int
    thresholds: RetrievalThresholds
    overall: RetrievalScores
    by_query_type: Mapping[str, RetrievalScores]
    examples: tuple[ExampleResult, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_retrieval(
    examples: Sequence[GoldenExample],
    retriever: _Retriever,
    k: int,
    thresholds: RetrievalThresholds,
) -> RetrievalReport:
    """Score every golden example and compare the averages with ``thresholds``."""
    if not examples:
        raise ValueError("evaluate_retrieval needs at least one golden example")

    results = tuple(_evaluate_example(example, retriever, k) for example in examples)
    overall = mean_scores([r.scores for r in results])
    return RetrievalReport(
        k=k,
        thresholds=thresholds,
        overall=overall,
        by_query_type=mean_scores_by_group((r.query_type, r.scores) for r in results),
        examples=results,
        failures=check_thresholds(overall, thresholds),
    )


def check_thresholds(scores: RetrievalScores, thresholds: RetrievalThresholds) -> tuple[str, ...]:
    """Describe every metric that is below its minimum (empty means pass)."""
    checks = (
        ("recall@k", scores.recall, thresholds.min_recall_at_k),
        ("mrr", scores.mrr, thresholds.min_mrr),
        ("ndcg@k", scores.ndcg, thresholds.min_ndcg_at_k),
    )
    return tuple(
        f"{name} {value:.3f} < {minimum:.3f}" for name, value, minimum in checks if value < minimum
    )


def report_to_dict(report: RetrievalReport) -> dict[str, Any]:
    """Plain-JSON form of the report (for the report file and CI artifacts)."""
    return {
        "k": report.k,
        "passed": report.passed,
        "failures": list(report.failures),
        "thresholds": asdict(report.thresholds),
        "overall": asdict(report.overall),
        "by_query_type": {name: asdict(s) for name, s in report.by_query_type.items()},
        "examples": [
            {**asdict(r), "relevant_doc_ids": list(r.relevant_doc_ids),
             "retrieved_doc_ids": list(r.retrieved_doc_ids)}
            for r in report.examples
        ],
    }  # fmt: skip


def _evaluate_example(example: GoldenExample, retriever: _Retriever, k: int) -> ExampleResult:
    doc_ids = [r.chunk.doc_id for r in retriever.retrieve(example.question)]
    return ExampleResult(
        example_id=example.id,
        query_type=example.query_type,
        relevant_doc_ids=example.relevant_doc_ids,
        retrieved_doc_ids=tuple(dict.fromkeys(doc_ids)),
        scores=score_retrieval(doc_ids, set(example.relevant_doc_ids), k),
    )
