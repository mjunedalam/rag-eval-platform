"""Optional cross-encoder re-ranking.

A cross-encoder reads the query and one candidate passage together and scores their
relevance. It is more accurate than vector similarity but too slow to run over the whole
corpus, so the retriever fetches a larger candidate set and the re-ranker keeps the best.
Needs ``uv sync --extra local-embeddings``.
"""

from collections.abc import Sequence
from typing import Any, Protocol, Self

from rag_eval_platform._optional import import_optional
from rag_eval_platform.retrieval.vector_store import SearchResult


class Reranker(Protocol):
    def rerank(
        self, query: str, results: Sequence[SearchResult], top_n: int
    ) -> list[SearchResult]: ...


class _Array(Protocol):
    def tolist(self) -> Any: ...


class _PairScorer(Protocol):
    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> _Array: ...


class CrossEncoderReranker:
    """Re-rank with a Sentence Transformers cross-encoder, e.g. ms-marco-MiniLM-L-6-v2."""

    def __init__(self, model: _PairScorer) -> None:
        self._model = model

    @classmethod
    def from_pretrained(cls, model_name: str) -> Self:
        """Load a model by name (downloaded from Hugging Face on first use, then cached)."""
        module = import_optional("sentence_transformers", extra="local-embeddings")
        return cls(module.CrossEncoder(model_name))

    def rerank(self, query: str, results: Sequence[SearchResult], top_n: int) -> list[SearchResult]:
        """Return the ``top_n`` results by cross-encoder score (the new ``score``)."""
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        if not results:
            return []

        scores = self._model.predict(
            [(query, r.chunk.text) for r in results], show_progress_bar=False
        ).tolist()
        rescored = [
            SearchResult(chunk=r.chunk, score=float(s))
            for r, s in zip(results, scores, strict=True)
        ]
        return sorted(rescored, key=lambda r: r.score, reverse=True)[:top_n]
