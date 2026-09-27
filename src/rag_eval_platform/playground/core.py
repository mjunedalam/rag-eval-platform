"""UI-independent logic for the playground app, kept here so it can be unit-tested.

Uploaded files are saved under ``data/playground/`` (git-ignored) and indexed into their
own Chroma collection, so the sample corpus and the golden-dataset evaluation are never
touched.
"""

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

from rag_eval_platform.config.settings import ChunkStrategy
from rag_eval_platform.generation.generator import Answer, Generator
from rag_eval_platform.ingestion.chunking import Chunk, chunk_documents
from rag_eval_platform.ingestion.embedding import Embedder
from rag_eval_platform.ingestion.loaders import (
    SUFFIX_FORMATS,
    Document,
    DocumentLoadError,
    load_document,
)
from rag_eval_platform.retrieval.reranker import Reranker
from rag_eval_platform.retrieval.seed import seed
from rag_eval_platform.retrieval.vector_store import SearchResult, VectorStore

PLAYGROUND_COLLECTION = "playground"
UPLOAD_DIR = Path("data/playground")


class UploadError(ValueError):
    """An uploaded file cannot be accepted."""


@dataclass(frozen=True)
class UploadResult:
    saved: tuple[Path, ...]
    skipped: tuple[str, ...]  # duplicates, with the reason


@dataclass(frozen=True)
class IndexSummary:
    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]

    @property
    def page_count(self) -> int:
        return sum(len(d.page_starts) for d in self.documents)

    @property
    def mean_chunk_chars(self) -> float:
        return fmean(len(c.text) for c in self.chunks) if self.chunks else 0.0


@dataclass(frozen=True)
class QueryTrace:
    """Every intermediate step of one question, so the UI can show what happened."""

    vector_results: tuple[SearchResult, ...]  # straight from vector search
    final_results: tuple[SearchResult, ...]  # after re-ranking (same as above if off)
    answer: Answer
    retrieval_ms: float


def save_uploads(files: Sequence[tuple[str, bytes]], upload_dir: Path) -> UploadResult:
    """Replace the contents of ``upload_dir`` with ``files`` (name, content) pairs.

    Duplicates are skipped, not indexed twice: a repeated file name, or the same content
    under another name. Duplicate chunks would fill the top-k slots with copies.
    """
    if not files:
        raise UploadError("No files uploaded")

    checked = [(_safe_name(name), content) for name, content in files]
    for name, content in checked:
        if not content:
            raise UploadError(f"File is empty: {name}")

    unique: dict[str, bytes] = {}
    first_name_by_hash: dict[str, str] = {}
    skipped = []
    for name, content in checked:
        digest = hashlib.sha256(content).hexdigest()
        if name in unique:
            skipped.append(f"{name} (a file with this name was already uploaded)")
        elif digest in first_name_by_hash:
            skipped.append(f"{name} (same content as {first_name_by_hash[digest]})")
        else:
            unique[name] = content
            first_name_by_hash[digest] = name

    upload_dir.mkdir(parents=True, exist_ok=True)
    for old in upload_dir.iterdir():
        if old.is_file():
            old.unlink()

    saved = []
    for name, content in unique.items():
        path = upload_dir / name
        path.write_bytes(content)
        saved.append(path)
    return UploadResult(saved=tuple(saved), skipped=tuple(skipped))


def build_index(
    upload_dir: Path,
    *,
    strategy: ChunkStrategy,
    chunk_size: int,
    chunk_overlap: int,
    embedder: Embedder,
    store: VectorStore,
    on_progress: Callable[[str], None] | None = None,
) -> IndexSummary:
    """Load, chunk, embed and store every uploaded document (replacing the old index).

    ``on_progress`` receives a short message after each document and at the end, so a UI
    can show that a long build (large PDFs take tens of seconds) is still working.
    """
    report = on_progress or (lambda message: None)
    paths = sorted(
        p for p in upload_dir.iterdir() if p.is_file() and p.suffix.lower() in SUFFIX_FORMATS
    )
    if not paths:
        raise DocumentLoadError(f"No supported documents (.md, .txt, .pdf) found in {upload_dir}")

    documents = []
    for path in paths:
        document = load_document(path, upload_dir)
        documents.append(document)
        pages = f" ({len(document.page_starts)} pages)" if document.page_starts else ""
        report(f"Read {document.id}{pages}")

    chunks = chunk_documents(
        documents, strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    report(f"Split into {len(chunks)} chunks; embedding and storing...")
    seed(chunks, embedder, store)
    return IndexSummary(documents=tuple(documents), chunks=chunks)


def _safe_name(filename: str) -> str:
    # Keep only the final name part, so "../../x.md" cannot escape the upload folder.
    name = Path(filename.replace("\\", "/")).name
    if not Path(name).stem or name.startswith("."):
        raise UploadError(f"Invalid file name: {filename!r}")
    if Path(name).suffix.lower() not in SUFFIX_FORMATS:
        raise UploadError(f"Unsupported file type: {name} (use .pdf, .md or .txt)")
    return name


@dataclass(frozen=True)
class RetrievalStep:
    vector_results: tuple[SearchResult, ...]  # straight from vector search
    final_results: tuple[SearchResult, ...]  # after re-ranking (same as above if off)
    retrieval_ms: float


def retrieve_step(
    question: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    top_k: int,
    reranker: Reranker | None = None,
    rerank_candidates: int = 20,
) -> RetrievalStep:
    """Vector search, then optional re-ranking, keeping both rankings for comparison."""
    if not question.strip():
        raise ValueError("question must not be blank")

    started = time.perf_counter()
    candidates = top_k if reranker is None else max(rerank_candidates, top_k)
    vector_results = tuple(store.search(embedder.embed_query(question), k=candidates))
    final_results = (
        vector_results
        if reranker is None
        else tuple(reranker.rerank(question, vector_results, top_n=top_k))
    )
    return RetrievalStep(vector_results, final_results, (time.perf_counter() - started) * 1000)


def run_query(
    question: str,
    *,
    embedder: Embedder,
    store: VectorStore,
    generator: Generator,
    top_k: int,
    reranker: Reranker | None = None,
    rerank_candidates: int = 20,
) -> QueryTrace:
    """Retrieve (and optionally re-rank), then generate, keeping each step's output."""
    step = retrieve_step(
        question,
        embedder=embedder,
        store=store,
        top_k=top_k,
        reranker=reranker,
        rerank_candidates=rerank_candidates,
    )
    answer = generator.generate(question, step.final_results)
    return QueryTrace(step.vector_results, step.final_results, answer, step.retrieval_ms)
