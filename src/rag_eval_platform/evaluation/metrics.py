"""Retrieval metrics: precision, recall, MRR, NDCG.

All metrics use binary, document-level relevance (see docs/evaluation_methodology.md).
``retrieved`` is the ranked list of source-document ids for the retrieved chunks; several
chunks from one document collapse to that document's first (best) rank before scoring.
"""

import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from statistics import fmean


@dataclass(frozen=True)
class RetrievalScores:
    """Retrieval metrics for one query, or averaged over many."""

    precision: float
    recall: float
    mrr: float
    ndcg: float


def _top_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> list[str]:
    """Validate inputs and return the first k distinct doc ids in rank order."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant:
        raise ValueError("relevant must contain at least one document id")
    return list(dict.fromkeys(retrieved))[:k]


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Relevant documents in the top k divided by documents retrieved in the top k."""
    top = _top_k(retrieved, relevant, k)
    if not top:
        return 0.0
    return sum(doc in relevant for doc in top) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Relevant documents in the top k divided by all relevant documents."""
    top = _top_k(retrieved, relevant, k)
    return sum(doc in relevant for doc in top) / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1 / rank of the first relevant document in the top k, or 0 if there is none."""
    top = _top_k(retrieved, relevant, k)
    return next((1 / rank for rank, doc in enumerate(top, start=1) if doc in relevant), 0.0)


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """DCG of the top k divided by the DCG of an ideal ranking."""
    top = _top_k(retrieved, relevant, k)
    dcg = sum(1 / math.log2(rank + 1) for rank, doc in enumerate(top, start=1) if doc in relevant)
    ideal_hits = min(k, len(set(relevant)))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def score_retrieval(retrieved: Sequence[str], relevant: Collection[str], k: int) -> RetrievalScores:
    """Compute every retrieval metric for one query."""
    return RetrievalScores(
        precision=precision_at_k(retrieved, relevant, k),
        recall=recall_at_k(retrieved, relevant, k),
        mrr=reciprocal_rank(retrieved, relevant, k),
        ndcg=ndcg_at_k(retrieved, relevant, k),
    )


def mean_scores(scores: Sequence[RetrievalScores]) -> RetrievalScores:
    """Average each metric over a set of queries (MRR is the mean reciprocal rank)."""
    if not scores:
        raise ValueError("mean_scores needs at least one RetrievalScores")
    return RetrievalScores(
        precision=fmean(s.precision for s in scores),
        recall=fmean(s.recall for s in scores),
        mrr=fmean(s.mrr for s in scores),
        ndcg=fmean(s.ndcg for s in scores),
    )


def mean_scores_by_group(
    grouped_scores: Iterable[tuple[str, RetrievalScores]],
) -> dict[str, RetrievalScores]:
    """Average scores per group, e.g. per golden-dataset query_type."""
    groups: defaultdict[str, list[RetrievalScores]] = defaultdict(list)
    for group, scores in grouped_scores:
        groups[group].append(scores)
    return {group: mean_scores(members) for group, members in groups.items()}
