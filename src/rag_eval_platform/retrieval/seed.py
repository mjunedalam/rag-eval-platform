"""Seed the vector store: embed the processed chunks and replace the collection's contents.

Run with ``uv run python scripts/seed_vector_store.py`` after ingestion. Re-run it after
changing chunking or the embedding model; the collection is rebuilt from scratch.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.ingestion.chunk_io import ChunkFileError, load_chunks
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.ingestion.cli import CHUNKS_FILE_NAME
from rag_eval_platform.ingestion.embedding import Embedder, EmbeddingError, create_embedder
from rag_eval_platform.retrieval.vector_store import (
    VectorStore,
    VectorStoreError,
    create_vector_store,
)

logger = logging.getLogger(__name__)


def seed(chunks: Sequence[Chunk], embedder: Embedder, store: VectorStore) -> int:
    """Embed ``chunks`` and store them, replacing what was there. Returns the stored count."""
    if not chunks:
        raise ValueError("No chunks to seed; run ingestion first")
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.replace_all(chunks, embeddings)
    return store.count()


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Embed processed chunks into the vector store.")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=settings.processed_data_dir / CHUNKS_FILE_NAME,
        help="chunks .jsonl file written by run_ingestion.py",
    )
    args = parser.parse_args(argv)

    try:
        chunks = load_chunks(args.chunks)
        stored = seed(chunks, create_embedder(settings), create_vector_store(settings))
    except (
        ChunkFileError,
        EmbeddingError,
        OptionalDependencyError,
        VectorStoreError,
        ValueError,
    ) as exc:
        logger.error("Seeding failed: %s", exc)
        return 1

    logger.info(
        "Vector store seeded",
        extra={
            "chunks": stored,
            "collection": settings.collection_name,
            "embedding_model": settings.embedding_model,
        },
    )
    return 0
