"""Chunking strategies (fixed-size, recursive, semantic).

- ``recursive`` (default): split on paragraph breaks, then line breaks, then spaces, and
  only cut mid-word as a last resort (LangChain's RecursiveCharacterTextSplitter).
- ``fixed``: a sliding window of ``chunk_size`` characters, stepping by
  ``chunk_size - chunk_overlap``.
- ``semantic``: planned; it needs the embedder to find topic shifts.

Sizes are in characters. Every chunk keeps its source ``doc_id`` (for document-level
evaluation) and its ``start_index`` in the document text (and, for PDFs, its page).
"""

from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_eval_platform.config.settings import ChunkStrategy
from rag_eval_platform.ingestion.loaders import Document


@dataclass(frozen=True)
class Chunk:
    """A passage of a document, the unit that is embedded and retrieved."""

    id: str
    doc_id: str
    index: int
    text: str
    start_index: int
    page: int | None = None


def chunk_document(
    document: Document, *, strategy: ChunkStrategy, chunk_size: int, chunk_overlap: int
) -> tuple[Chunk, ...]:
    """Split one document into chunks with ids ``<doc_id>#<index>``."""
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap must be >= 0 and smaller than chunk_size, got {chunk_overlap}"
        )

    pieces: Iterable[tuple[str, int]]
    if strategy == "recursive":
        pieces = _recursive_pieces(document.text, chunk_size, chunk_overlap)
    elif strategy == "fixed":
        pieces = _fixed_pieces(document.text, chunk_size, chunk_overlap)
    else:
        raise NotImplementedError(f"{strategy} chunking is not implemented yet")

    return tuple(
        Chunk(
            id=f"{document.id}#{index}",
            doc_id=document.id,
            index=index,
            text=text,
            start_index=start,
            page=_page_at(document.page_starts, start),
        )
        for index, (text, start) in enumerate(pieces)
    )


def chunk_documents(
    documents: Iterable[Document], *, strategy: ChunkStrategy, chunk_size: int, chunk_overlap: int
) -> tuple[Chunk, ...]:
    """Chunk every document, keeping document order."""
    return tuple(
        chunk
        for document in documents
        for chunk in chunk_document(
            document, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    )


def _recursive_pieces(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    return [(d.page_content, d.metadata["start_index"]) for d in splitter.create_documents([text])]


def _fixed_pieces(text: str, chunk_size: int, chunk_overlap: int) -> Iterator[tuple[str, int]]:
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        window = text[start : start + chunk_size]
        stripped = window.strip()
        if stripped:
            yield stripped, start + (len(window) - len(window.lstrip()))
        if start + chunk_size >= len(text):
            break


def _page_at(page_starts: tuple[int, ...], offset: int) -> int | None:
    """1-based page number containing ``offset``, or None for documents without pages."""
    if not page_starts:
        return None
    return bisect_right(page_starts, offset)
