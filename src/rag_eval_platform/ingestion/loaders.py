"""Load raw documents from disk.

Supported formats are Markdown, plain text and PDFs that have a text layer. A document's
id is its path relative to the corpus root (POSIX style), which is what the golden
dataset's ``relevant_doc_ids`` refer to. Problems raise ``DocumentLoadError`` instead of
being skipped, so a broken file never silently disappears from the index.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)

DocumentFormat = Literal["markdown", "text", "pdf"]

SUFFIX_FORMATS: dict[str, DocumentFormat] = {".md": "markdown", ".txt": "text", ".pdf": "pdf"}

# Pages are joined with a blank line so paragraph-aware chunking prefers page breaks.
PAGE_SEPARATOR = "\n\n"


class DocumentLoadError(Exception):
    """A document or corpus directory could not be loaded."""


@dataclass(frozen=True)
class Document:
    """One source document's full text.

    ``page_starts`` holds the character offset in ``text`` where each PDF page begins
    (empty for non-PDF formats), so chunks can be mapped back to page numbers.
    """

    id: str
    text: str
    format: DocumentFormat
    page_starts: tuple[int, ...]


def load_document(path: Path, root: Path) -> Document:
    """Load one file; ``root`` is the corpus directory used to build the document id."""
    doc_format = SUFFIX_FORMATS.get(path.suffix.lower())
    if doc_format is None:
        raise DocumentLoadError(f"Unsupported file type: {path}")

    doc_id = path.relative_to(root).as_posix()
    if doc_format == "pdf":
        text, page_starts = _read_pdf(path)
    else:
        text, page_starts = _read_text(path), ()

    if not text.strip():
        hint = " (scanned PDF? run OCR first)" if doc_format == "pdf" else ""
        raise DocumentLoadError(f"Document has no text{hint}: {path}")
    return Document(id=doc_id, text=text, format=doc_format, page_starts=page_starts)


def load_documents(root: Path) -> tuple[Document, ...]:
    """Load every supported file under ``root`` recursively, sorted by id.

    Hidden files and folders (names starting with ".") are ignored; other unsupported
    files are skipped with a warning.
    """
    if not root.is_dir():
        raise DocumentLoadError(f"Corpus path is not a directory: {root}")

    paths: list[Path] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part.startswith(".") for part in relative.parts):
            continue
        if path.suffix.lower() not in SUFFIX_FORMATS:
            logger.warning("Skipping unsupported file %s", relative, extra={"path": str(relative)})
            continue
        paths.append(path)

    if not paths:
        raise DocumentLoadError(f"No supported documents (.md, .txt, .pdf) found in {root}")
    return tuple(sorted((load_document(p, root) for p in paths), key=lambda d: d.id))


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise DocumentLoadError(f"File is not valid UTF-8: {path}") from exc


def _read_pdf(path: Path) -> tuple[str, tuple[int, ...]]:
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PyPdfError, OSError, ValueError) as exc:
        raise DocumentLoadError(f"Could not read PDF {path}: {exc}") from exc

    page_starts: list[int] = []
    offset = 0
    for page_text in pages:
        page_starts.append(offset)
        offset += len(page_text) + len(PAGE_SEPARATOR)
    return PAGE_SEPARATOR.join(pages), tuple(page_starts)
