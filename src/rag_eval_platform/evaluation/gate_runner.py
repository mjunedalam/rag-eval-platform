"""Rebuild the index from the corpus with given settings, then score retrieval.

Shared by the CI gate (scripts/run_gate.py) and the playground's what-if gate, which
passes its own settings and a separate collection.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

from rag_eval_platform.config.settings import ChunkStrategy, Settings
from rag_eval_platform.evaluation.evaluator import (
    RetrievalReport,
    RetrievalThresholds,
    evaluate_retrieval,
)
from rag_eval_platform.evaluation.golden_dataset import GoldenExample
from rag_eval_platform.ingestion.chunking import chunk_documents
from rag_eval_platform.ingestion.embedding import Embedder, create_embedder
from rag_eval_platform.ingestion.loaders import load_documents
from rag_eval_platform.retrieval.reranker import Reranker
from rag_eval_platform.retrieval.retriever import Retriever
from rag_eval_platform.retrieval.seed import seed
from rag_eval_platform.retrieval.vector_store import VectorStore, create_vector_store


def rebuild_index(
    raw_dir: Path,
    *,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Embedder,
    store: VectorStore,
    on_progress: Callable[[str], None] | None = None,
) -> int:
    """Replace the store's contents with ``raw_dir`` chunked this way; returns the chunk count."""
    report = on_progress or (lambda message: None)
    documents = load_documents(raw_dir)
    chunks = chunk_documents(
        documents, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    report(f"Indexed {len(documents)} documents as {len(chunks)} chunks")
    return seed(chunks, embedder, store)


def rebuild_index_from_settings(settings: Settings) -> int:
    """Rebuild the configured collection from the corpus with the current settings.

    Saving a baseline calls this first, so the scores always come from an index that matches
    the fingerprint stamped on them.
    """
    return rebuild_index(
        settings.raw_data_dir, strategy=settings.chunk_strategy, chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap, embedder=create_embedder(settings),
        store=create_vector_store(settings),
    )  # fmt: skip


def rebuild_and_evaluate(
    examples: Sequence[GoldenExample],
    *,
    raw_dir: Path,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Embedder,
    store: VectorStore,
    top_k: int,
    reranker: Reranker | None,
    rerank_candidates: int,
    thresholds: RetrievalThresholds,
    on_progress: Callable[[str], None] | None = None,
) -> RetrievalReport:
    report = on_progress or (lambda message: None)
    rebuild_index(raw_dir, strategy=strategy, chunk_size=chunk_size,
                  chunk_overlap=chunk_overlap, embedder=embedder, store=store,
                  on_progress=report)  # fmt: skip
    retriever = Retriever(embedder, store, top_k, reranker, rerank_candidates)
    result = evaluate_retrieval(examples, retriever, k=top_k, thresholds=thresholds)
    report(f"Scored {len(examples)} golden questions")
    return result
