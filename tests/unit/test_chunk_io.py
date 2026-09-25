"""Tests for ingestion.chunk_io."""

from pathlib import Path

import pytest

from rag_eval_platform.ingestion.chunk_io import ChunkFileError, load_chunks, save_chunks
from rag_eval_platform.ingestion.chunking import Chunk

CHUNKS = (
    Chunk(id="a.md#0", doc_id="a.md", index=0, text="Alpha", start_index=0),
    Chunk(id="book.pdf#0", doc_id="book.pdf", index=0, text="Café π", start_index=5, page=2),
)


def test_round_trips_chunks_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "out" / "chunks.jsonl"

    save_chunks(CHUNKS, path)

    assert load_chunks(path) == CHUNKS
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ChunkFileError, match="not found"):
        load_chunks(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    "bad_line",
    [
        "{not json",
        '{"id": "x"}',
        '{"id": "x", "doc_id": "a", "index": "one", "text": "t", "start_index": 0, "page": null}',
    ],
)
def test_invalid_line_reports_line_number(tmp_path: Path, bad_line: str) -> None:
    path = tmp_path / "chunks.jsonl"
    save_chunks(CHUNKS[:1], path)
    with path.open("a", encoding="utf-8") as f:
        f.write(bad_line + "\n")

    with pytest.raises(ChunkFileError, match="line 2"):
        load_chunks(path)
