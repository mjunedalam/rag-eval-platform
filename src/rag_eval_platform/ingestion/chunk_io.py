"""Save and load chunks as JSON Lines (one chunk per line) in ``data/processed/``.

Ingestion writes this file; seeding the vector store reads it, so the two steps can run
separately and the chunked output can be inspected by hand.
"""

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from rag_eval_platform.ingestion.chunking import Chunk

_CHUNK_ADAPTER = TypeAdapter(Chunk)


class ChunkFileError(Exception):
    """The chunk file is missing or malformed."""


def save_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    """Write chunks to ``path``, replacing the file; parent folders are created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> tuple[Chunk, ...]:
    """Read and validate every chunk in ``path``."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ChunkFileError(f"Chunk file not found: {path} (run ingestion first)") from exc

    chunks = []
    for line_number, line in enumerate(lines, start=1):
        try:
            chunks.append(_CHUNK_ADAPTER.validate_json(line, strict=True))
        except ValidationError as exc:
            raise ChunkFileError(f"Invalid chunk on line {line_number} of {path}: {exc}") from exc
    return tuple(chunks)
