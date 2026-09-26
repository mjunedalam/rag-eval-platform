"""Query-time retrieval with configurable top-k.

Embed the query with the same model used at ingestion, fetch the nearest chunks from the
vector store and, when re-ranking is on, let the cross-encoder pick the best ``top_k``
from a larger candidate set.
"""

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.ingestion.embedding import Embedder, create_embedder
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker, Reranker
from rag_eval_platform.retrieval.vector_store import (
    SearchResult,
    VectorStore,
    create_vector_store,
)


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        top_k: int,
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates

    def retrieve(self, query: str) -> list[SearchResult]:
        """Return up to ``top_k`` chunks, most relevant first."""
        if not query.strip():
            raise ValueError("query must not be blank")

        query_embedding = self.embedder.embed_query(query)
        if self.reranker is None:
            return self.store.search(query_embedding, k=self.top_k)

        candidates = self.store.search(query_embedding, k=max(self.rerank_candidates, self.top_k))
        return self.reranker.rerank(query, candidates, top_n=self.top_k)


def create_retriever(settings: Settings) -> Retriever:
    """Build the retriever (embedder, store and optional re-ranker) from settings."""
    reranker = (
        CrossEncoderReranker.from_pretrained(settings.reranker_model) if settings.rerank else None
    )
    return Retriever(
        embedder=create_embedder(settings),
        store=create_vector_store(settings),
        top_k=settings.top_k,
        reranker=reranker,
        rerank_candidates=settings.rerank_candidates,
    )
