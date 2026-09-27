"""Tests for the playground's UI-independent logic (uploads and index building)."""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from tests.pdf_builder import build_pdf

from rag_eval_platform.generation.generator import Completion, Generator
from rag_eval_platform.generation.prompt_templates import Message
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import (
    PLAYGROUND_COLLECTION,
    UploadError,
    build_index,
    retrieve_step,
    run_query,
    save_uploads,
)
from rag_eval_platform.retrieval.vector_store import SearchResult


class FakeEmbedder:
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


class FakeStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def replace_all(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        self.chunks = list(chunks)

    def search(self, query_embedding: Sequence[float], k: int) -> list[SearchResult]:
        return []

    def count(self) -> int:
        return len(self.chunks)


class TestSaveUploads:
    def test_writes_files_into_upload_dir(self, tmp_path: Path) -> None:
        result = save_uploads([("notes.md", b"# Notes"), ("book.pdf", b"%PDF")], tmp_path)

        assert [p.name for p in result.saved] == ["notes.md", "book.pdf"]
        assert result.skipped == ()
        assert (tmp_path / "notes.md").read_bytes() == b"# Notes"

    def test_strips_directories_from_names(self, tmp_path: Path) -> None:
        upload_dir = tmp_path / "uploads"

        (saved,) = save_uploads([("../../etc/evil.md", b"text")], upload_dir).saved

        assert saved == upload_dir / "evil.md"
        assert not (tmp_path / "etc").exists()

    def test_replaces_previous_uploads(self, tmp_path: Path) -> None:
        save_uploads([("old.md", b"old")], tmp_path)

        save_uploads([("new.md", b"new")], tmp_path)

        assert sorted(p.name for p in tmp_path.iterdir()) == ["new.md"]

    def test_skips_identical_content_under_another_name(self, tmp_path: Path) -> None:
        result = save_uploads(
            [("book.pdf", b"%PDF same"), ("book (1).pdf", b"%PDF same"), ("other.md", b"x")],
            tmp_path,
        )

        assert [p.name for p in result.saved] == ["book.pdf", "other.md"]
        assert result.skipped == ("book (1).pdf (same content as book.pdf)",)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["book.pdf", "other.md"]

    def test_skips_a_second_file_with_the_same_name(self, tmp_path: Path) -> None:
        result = save_uploads([("notes.md", b"first"), ("notes.md", b"second")], tmp_path)

        assert [p.name for p in result.saved] == ["notes.md"]
        assert result.skipped == ("notes.md (a file with this name was already uploaded)",)
        assert (tmp_path / "notes.md").read_bytes() == b"first"

    @pytest.mark.parametrize(
        ("files", "message"),
        [
            ([], "No files"),
            ([("slides.pptx", b"x")], "Unsupported"),
            ([("empty.md", b"")], "empty"),
            ([(".md", b"x")], "file name"),
        ],
    )
    def test_rejects_bad_uploads(
        self, tmp_path: Path, files: list[tuple[str, bytes]], message: str
    ) -> None:
        with pytest.raises(UploadError, match=message):
            save_uploads(files, tmp_path)


def test_build_index_loads_chunks_and_stores_everything(tmp_path: Path) -> None:
    save_uploads(
        [
            ("notes.md", b"First paragraph.\n\nSecond paragraph."),
            ("book.pdf", build_pdf([["Chapter 1", "Motion"], ["Chapter 2", "Forces"]])),
        ],
        tmp_path,
    )
    store = FakeStore()
    progress: list[str] = []

    summary = build_index(
        tmp_path,
        strategy="recursive",
        chunk_size=20,
        chunk_overlap=0,
        embedder=FakeEmbedder(),
        store=store,
        on_progress=progress.append,
    )

    assert progress[0] == "Read book.pdf (2 pages)"
    assert progress[1] == "Read notes.md"
    assert progress[-1] == f"Split into {len(summary.chunks)} chunks; embedding and storing..."

    assert [d.id for d in summary.documents] == ["book.pdf", "notes.md"]
    assert summary.page_count == 2
    assert len(summary.chunks) == store.count() > 2
    assert {c.page for c in summary.chunks if c.doc_id == "book.pdf"} == {1, 2}
    assert summary.mean_chunk_chars == pytest.approx(
        sum(len(c.text) for c in summary.chunks) / len(summary.chunks)
    )


def test_playground_uses_its_own_collection() -> None:
    assert PLAYGROUND_COLLECTION == "playground"


class RankedStore(FakeStore):
    def __init__(self, results: list[SearchResult]) -> None:
        super().__init__()
        self.results = results
        self.requested_k: list[int] = []

    def search(self, query_embedding: Sequence[float], k: int) -> list[SearchResult]:
        self.requested_k.append(k)
        return self.results[:k]


class ReverseReranker:
    def rerank(self, query: str, results: Sequence[SearchResult], top_n: int) -> list[SearchResult]:
        return [SearchResult(r.chunk, 9.0) for r in reversed(results)][:top_n]


class FakeLlm:
    model = "fake-8b"

    def complete(self, messages: Sequence[Message]) -> Completion:
        return Completion(text="Answer [1].", input_tokens=10, output_tokens=3)

    def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
        yield Completion(text="Answer [1].", input_tokens=None, output_tokens=None)
        yield Completion(text="", input_tokens=10, output_tokens=3)


def ranked(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            Chunk(id=f"d#{i}", doc_id="d.md", index=i, text="t", start_index=0), 1 - i / 10
        )
        for i in range(n)
    ]


def test_run_query_without_rerank_keeps_vector_order() -> None:
    store = RankedStore(ranked(6))

    trace = run_query(
        "q?", embedder=FakeEmbedder(), store=store, generator=Generator(FakeLlm()), top_k=3
    )

    assert store.requested_k == [3]
    assert trace.vector_results == trace.final_results == tuple(ranked(6)[:3])
    assert [r.chunk.id for r in trace.answer.sources] == ["d#0", "d#1", "d#2"]
    assert trace.retrieval_ms >= 0


def test_run_query_with_rerank_shows_before_and_after() -> None:
    store = RankedStore(ranked(6))

    trace = run_query(
        "q?",
        embedder=FakeEmbedder(),
        store=store,
        generator=Generator(FakeLlm()),
        top_k=2,
        reranker=ReverseReranker(),
        rerank_candidates=5,
    )

    assert store.requested_k == [5]
    assert [r.chunk.id for r in trace.vector_results] == ["d#0", "d#1", "d#2", "d#3", "d#4"]
    assert [r.chunk.id for r in trace.final_results] == ["d#4", "d#3"]
    assert [r.chunk.id for r in trace.answer.sources] == ["d#4", "d#3"]


def test_run_query_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="question"):
        run_query(
            " ", embedder=FakeEmbedder(), store=FakeStore(), generator=Generator(FakeLlm()), top_k=3
        )


def test_retrieve_step_returns_rankings_without_generating() -> None:
    store = RankedStore(ranked(6))

    step = retrieve_step(
        "q?",
        embedder=FakeEmbedder(),
        store=store,
        top_k=2,
        reranker=ReverseReranker(),
        rerank_candidates=4,
    )

    assert [r.chunk.id for r in step.vector_results] == ["d#0", "d#1", "d#2", "d#3"]
    assert [r.chunk.id for r in step.final_results] == ["d#3", "d#2"]
    assert step.retrieval_ms >= 0
