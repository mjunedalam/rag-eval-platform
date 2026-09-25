"""Ingestion command: load documents from the corpus, chunk them, save the chunks.

Run with ``uv run python scripts/run_ingestion.py``. Defaults come from settings
(``RAG_RAW_DATA_DIR``, ``RAG_CHUNK_STRATEGY``, ...); command-line flags override them.
Embedding happens when the vector store is seeded, not here.
"""

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import get_args

from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import ChunkStrategy, get_settings
from rag_eval_platform.ingestion.chunk_io import save_chunks
from rag_eval_platform.ingestion.chunking import chunk_documents
from rag_eval_platform.ingestion.loaders import DocumentLoadError, load_documents

logger = logging.getLogger(__name__)

CHUNKS_FILE_NAME = "chunks.jsonl"


@dataclass(frozen=True)
class IngestionSummary:
    documents: int
    chunks: int
    mean_chunk_chars: float
    output: Path


def ingest(
    raw_dir: Path,
    output: Path,
    *,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestionSummary:
    """Load every document in ``raw_dir``, chunk it and write the chunks to ``output``."""
    documents = load_documents(raw_dir)
    chunks = chunk_documents(
        documents, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    save_chunks(chunks, output)
    return IngestionSummary(
        documents=len(documents),
        chunks=len(chunks),
        mean_chunk_chars=fmean(len(c.text) for c in chunks),
        output=output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    args = _parse_args(argv, default_output=settings.processed_data_dir / CHUNKS_FILE_NAME)

    try:
        summary = ingest(
            args.raw_dir or settings.raw_data_dir,
            args.output,
            strategy=args.strategy or settings.chunk_strategy,
            chunk_size=args.chunk_size or settings.chunk_size,
            chunk_overlap=(
                settings.chunk_overlap if args.chunk_overlap is None else args.chunk_overlap
            ),
        )
    except (DocumentLoadError, ValueError, NotImplementedError) as exc:
        logger.error("Ingestion failed: %s", exc)
        return 1

    logger.info(
        "Ingestion complete",
        extra={
            "documents": summary.documents,
            "chunks": summary.chunks,
            "mean_chunk_chars": round(summary.mean_chunk_chars, 1),
            "output": str(summary.output),
        },
    )
    return 0


def _parse_args(argv: Sequence[str] | None, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load, chunk and save the document corpus.")
    parser.add_argument("--raw-dir", type=Path, help="corpus folder (default: settings)")
    parser.add_argument("--output", type=Path, default=default_output, help="chunks .jsonl file")
    parser.add_argument("--strategy", choices=get_args(ChunkStrategy))
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--chunk-overlap", type=int)
    return parser.parse_args(argv)
