"""Tests for retrieval.reranker (with a fake cross-encoder; no model download)."""

from typing import Any

import pytest

from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker
from rag_eval_platform.retrieval.vector_store import SearchResult


def result(text: str, score: float = 0.5) -> SearchResult:
    chunk = Chunk(id=f"{text}#0", doc_id=f"{text}.md", index=0, text=text, start_index=0)
    return SearchResult(chunk=chunk, score=score)


class FakeArray:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeCrossEncoder:
    """Scores a pair by how many query words appear in the passage."""

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> FakeArray:
        self.pairs.extend(pairs)
        return FakeArray(
            [float(sum(word in passage for word in query.split())) for query, passage in pairs]
        )


def test_reorders_by_cross_encoder_score_and_keeps_top_n() -> None:
    model = FakeCrossEncoder()
    candidates = [result("gravity"), result("force mass"), result("force")]

    reranked = CrossEncoderReranker(model).rerank("force mass", candidates, top_n=2)

    assert [r.chunk.text for r in reranked] == ["force mass", "force"]
    assert [r.score for r in reranked] == [2.0, 1.0]
    assert model.pairs[0] == ("force mass", "gravity")


def test_empty_candidates_skip_the_model() -> None:
    model = FakeCrossEncoder()

    assert CrossEncoderReranker(model).rerank("q", [], top_n=3) == []
    assert model.pairs == []


def test_rejects_top_n_below_one() -> None:
    with pytest.raises(ValueError, match="top_n"):
        CrossEncoderReranker(FakeCrossEncoder()).rerank("q", [result("a")], top_n=0)
