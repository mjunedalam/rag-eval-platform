"""Tests for ingestion.loaders."""

import logging
from pathlib import Path

import pytest
from tests.pdf_builder import build_pdf

from rag_eval_platform.ingestion.loaders import (
    Document,
    DocumentFormat,
    DocumentLoadError,
    load_document,
    load_documents,
)


def write(path: Path, content: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


class TestLoadTextFiles:
    @pytest.mark.parametrize(("name", "fmt"), [("notes.md", "markdown"), ("notes.txt", "text")])
    def test_reads_text_and_format(self, tmp_path: Path, name: str, fmt: DocumentFormat) -> None:
        path = write(tmp_path / name, "# Title\n\nBody text.")

        document = load_document(path, root=tmp_path)

        assert document == Document(
            id=name, text="# Title\n\nBody text.", format=fmt, page_starts=()
        )

    def test_id_is_posix_path_relative_to_root(self, tmp_path: Path) -> None:
        path = write(tmp_path / "physics" / "02_newtons_laws.md", "F = ma")

        assert load_document(path, root=tmp_path).id == "physics/02_newtons_laws.md"

    def test_blank_file_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path / "empty.md", "  \n\n ")

        with pytest.raises(DocumentLoadError, match="no text"):
            load_document(path, root=tmp_path)

    def test_non_utf8_file_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path / "latin.txt", "café".encode("latin-1"))

        with pytest.raises(DocumentLoadError, match="UTF-8"):
            load_document(path, root=tmp_path)

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path / "slides.pptx", b"binary")

        with pytest.raises(DocumentLoadError, match="Unsupported"):
            load_document(path, root=tmp_path)


class TestLoadPdf:
    def test_joins_pages_and_records_page_starts(self, tmp_path: Path) -> None:
        path = write(
            tmp_path / "book.pdf",
            build_pdf([["Chapter 1", "Motion"], ["Chapter 2 (Forces)"]]),
        )

        document = load_document(path, root=tmp_path)

        assert document.format == "pdf"
        assert document.id == "book.pdf"
        assert "Motion" in document.text
        assert "Chapter 2 (Forces)" in document.text
        assert len(document.page_starts) == 2
        assert document.page_starts[0] == 0
        second_page = document.text[document.page_starts[1] :]
        assert second_page.startswith("Chapter 2")

    def test_blank_pages_keep_page_numbering(self, tmp_path: Path) -> None:
        path = write(tmp_path / "book.pdf", build_pdf([["Page one"], [], ["Page three"]]))

        document = load_document(path, root=tmp_path)

        assert len(document.page_starts) == 3
        assert document.text[document.page_starts[2] :].startswith("Page three")

    def test_pdf_without_text_layer_raises_with_ocr_hint(self, tmp_path: Path) -> None:
        path = write(tmp_path / "scan.pdf", build_pdf([[], []]))

        with pytest.raises(DocumentLoadError, match="OCR"):
            load_document(path, root=tmp_path)

    def test_corrupt_pdf_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path / "broken.pdf", b"%PDF-1.4 this is not a pdf")

        with pytest.raises(DocumentLoadError, match=r"broken\.pdf"):
            load_document(path, root=tmp_path)


class TestLoadDocuments:
    def test_loads_supported_files_recursively_sorted_by_id(self, tmp_path: Path) -> None:
        write(tmp_path / "b.md", "B")
        write(tmp_path / "a.txt", "A")
        write(tmp_path / "sub" / "c.pdf", build_pdf([["C"]]))

        documents = load_documents(tmp_path)

        assert [d.id for d in documents] == ["a.txt", "b.md", "sub/c.pdf"]

    def test_skips_hidden_files_and_warns_on_unsupported(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        write(tmp_path / "a.md", "A")
        write(tmp_path / ".gitkeep", "")
        write(tmp_path / ".cache" / "x.md", "hidden dir")
        write(tmp_path / "image.png", b"png")

        with caplog.at_level(logging.WARNING):
            documents = load_documents(tmp_path)

        assert [d.id for d in documents] == ["a.md"]
        assert "image.png" in caplog.text
        assert ".gitkeep" not in caplog.text

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DocumentLoadError, match="not a directory"):
            load_documents(tmp_path / "nope")

    def test_directory_without_documents_raises(self, tmp_path: Path) -> None:
        write(tmp_path / "image.png", b"png")

        with pytest.raises(DocumentLoadError, match="No supported documents"):
            load_documents(tmp_path)


def test_repository_corpus_loads() -> None:
    corpus = Path(__file__).resolve().parents[2] / "data" / "raw"

    documents = load_documents(corpus)

    assert len(documents) == 14
    assert all(d.format == "markdown" for d in documents)
