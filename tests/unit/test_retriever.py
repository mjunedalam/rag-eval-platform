"""Tests for retrieval.retriever using fake embedder, store and re-ranker."""

from collections.abc import Sequence
from typing import Any

import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval import retriever as retriever_module
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker
from rag_eval_platform.retrieval.retriever import Retriever, create_retriever
from rag_eval_platform.retrieval.vector_store import SearchResult


def result(name: str, score: float) -> SearchResult:
    chunk = Chunk(id=f"{name}#0", doc_id=f"{name}.md", index=0, text=name, start_index=0)
    return SearchResult(chunk=chunk, score=score)


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        raise AssertionError("retriever must only embed the query")

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0, 0.0]


class FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.requested_k: list[int] = []

    def replace_all(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        raise AssertionError("not used")

    def search(self, query_embedding: Sequence[float], k: int) -> list[SearchResult]:
        self.requested_k.append(k)
        return self.results[:k]

    def count(self) -> int:
        return len(self.results)


class ReverseReranker:
    def __init__(self) -> None:
        self.received: list[SearchResult] = []

    def rerank(self, query: str, results: Sequence[SearchResult], top_n: int) -> list[SearchResult]:
        self.received = list(results)
        return list(reversed(results))[:top_n]


RESULTS = [result(name, score) for name, score in [("a", 0.9), ("b", 0.8), ("c", 0.7), ("d", 0.6)]]


def test_embeds_query_and_returns_top_k() -> None:
    embedder, store = FakeEmbedder(), FakeStore(RESULTS)

    results = Retriever(embedder, store, top_k=2).retrieve("What is force?")

    assert results == RESULTS[:2]
    assert embedder.queries == ["What is force?"]
    assert store.requested_k == [2]


def test_reranker_sees_larger_candidate_set_then_top_k_is_kept() -> None:
    store, reranker = FakeStore(RESULTS), ReverseReranker()

    results = Retriever(
        FakeEmbedder(), store, top_k=2, reranker=reranker, rerank_candidates=4
    ).retrieve("q")

    assert store.requested_k == [4]
    assert len(reranker.received) == 4
    assert [r.chunk.text for r in results] == ["d", "c"]


def test_candidate_count_is_never_below_top_k() -> None:
    store = FakeStore(RESULTS)

    Retriever(
        FakeEmbedder(), store, top_k=3, reranker=ReverseReranker(), rerank_candidates=1
    ).retrieve("q")

    assert store.requested_k == [3]


def test_rejects_blank_query() -> None:
    with pytest.raises(ValueError, match="query"):
        Retriever(FakeEmbedder(), FakeStore(RESULTS), top_k=2).retrieve("   ")


def test_rejects_top_k_below_one() -> None:
    with pytest.raises(ValueError, match="top_k"):
        Retriever(FakeEmbedder(), FakeStore(RESULTS), top_k=0)


class TestCreateRetriever:
    @pytest.fixture
    def built(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        built: dict[str, Any] = {}
        monkeypatch.setattr(
            retriever_module, "create_embedder", lambda s: built.setdefault("embedder", object())
        )
        monkeypatch.setattr(
            retriever_module, "create_vector_store", lambda s: built.setdefault("store", object())
        )

        def fake_reranker(name: str) -> object:
            built["reranker_model"] = name
            return built.setdefault("reranker", object())

        monkeypatch.setattr(CrossEncoderReranker, "from_pretrained", staticmethod(fake_reranker))
        return built

    def test_without_rerank(self, built: dict[str, Any]) -> None:
        retriever = create_retriever(Settings(top_k=7))

        assert retriever.top_k == 7
        assert retriever.reranker is None
        assert "reranker_model" not in built

    def test_with_rerank_loads_configured_model(self, built: dict[str, Any]) -> None:
        retriever = create_retriever(
            Settings(rerank=True, reranker_model="my/cross-encoder", rerank_candidates=30)
        )

        assert retriever.reranker is built["reranker"]
        assert built["reranker_model"] == "my/cross-encoder"
        assert retriever.rerank_candidates == 30
