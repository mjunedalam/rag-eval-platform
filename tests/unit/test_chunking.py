"""Tests for ingestion.chunking."""

import pytest

from rag_eval_platform.ingestion.chunking import Chunk, chunk_document, chunk_documents
from rag_eval_platform.ingestion.loaders import Document


def markdown(text: str, doc_id: str = "doc.md") -> Document:
    return Document(id=doc_id, text=text, format="markdown", page_starts=())


PARAGRAPHS = "First paragraph about motion.\n\nSecond paragraph about force.\n\nThird about energy."


class TestRecursiveChunking:
    def test_short_document_is_one_chunk(self) -> None:
        chunks = chunk_document(
            markdown("Short text."), strategy="recursive", chunk_size=100, chunk_overlap=10
        )

        assert chunks == (
            Chunk(id="doc.md#0", doc_id="doc.md", index=0, text="Short text.", start_index=0),
        )

    def test_splits_on_paragraph_boundaries_first(self) -> None:
        chunks = chunk_document(
            markdown(PARAGRAPHS), strategy="recursive", chunk_size=35, chunk_overlap=0
        )

        assert [c.text for c in chunks] == [
            "First paragraph about motion.",
            "Second paragraph about force.",
            "Third about energy.",
        ]

    def test_respects_chunk_size(self) -> None:
        text = " ".join(f"word{i}" for i in range(200))

        chunks = chunk_document(
            markdown(text), strategy="recursive", chunk_size=50, chunk_overlap=10
        )

        assert len(chunks) > 1
        assert all(len(c.text) <= 50 for c in chunks)


class TestFixedChunking:
    def test_windows_step_by_size_minus_overlap(self) -> None:
        chunks = chunk_document(
            markdown("abcdefghij"), strategy="fixed", chunk_size=4, chunk_overlap=1
        )

        assert [(c.text, c.start_index) for c in chunks] == [
            ("abcd", 0),
            ("defg", 3),
            ("ghij", 6),
        ]

    def test_strips_whitespace_and_skips_blank_windows(self) -> None:
        chunks = chunk_document(
            markdown("ab      cd"), strategy="fixed", chunk_size=4, chunk_overlap=0
        )

        assert [(c.text, c.start_index) for c in chunks] == [("ab", 0), ("cd", 8)]


@pytest.mark.parametrize("strategy", ["recursive", "fixed"])
class TestChunkInvariants:
    def test_start_index_points_at_chunk_text(self, strategy: str) -> None:
        document = markdown(PARAGRAPHS * 3)

        chunks = chunk_document(document, strategy=strategy, chunk_size=40, chunk_overlap=8)  # type: ignore[arg-type]

        for chunk in chunks:
            assert document.text[chunk.start_index :].startswith(chunk.text)

    def test_ids_are_stable_and_sequential(self, strategy: str) -> None:
        chunks = chunk_document(
            markdown(PARAGRAPHS, doc_id="physics/ch1.md"),
            strategy=strategy,  # type: ignore[arg-type]
            chunk_size=30,
            chunk_overlap=5,
        )

        assert [c.index for c in chunks] == list(range(len(chunks)))
        assert [c.id for c in chunks] == [f"physics/ch1.md#{i}" for i in range(len(chunks))]
        assert {c.doc_id for c in chunks} == {"physics/ch1.md"}


class TestPdfPages:
    def test_chunks_record_the_page_they_start_on(self) -> None:
        page_one = "Kinematics describes motion."
        page_two = "Newton's laws describe forces."
        text = f"{page_one}\n\n{page_two}"
        document = Document(
            id="book.pdf", text=text, format="pdf", page_starts=(0, len(page_one) + 2)
        )

        chunks = chunk_document(document, strategy="recursive", chunk_size=35, chunk_overlap=0)

        assert [(c.text, c.page) for c in chunks] == [(page_one, 1), (page_two, 2)]

    def test_non_pdf_chunks_have_no_page(self) -> None:
        (chunk,) = chunk_document(
            markdown("Text."), strategy="recursive", chunk_size=100, chunk_overlap=0
        )

        assert chunk.page is None


class TestValidation:
    @pytest.mark.parametrize(("size", "overlap"), [(0, 0), (10, -1), (10, 10), (10, 20)])
    def test_rejects_invalid_size_and_overlap(self, size: int, overlap: int) -> None:
        with pytest.raises(ValueError, match="chunk"):
            chunk_document(
                markdown("text"), strategy="recursive", chunk_size=size, chunk_overlap=overlap
            )

    def test_semantic_strategy_is_not_implemented_yet(self) -> None:
        with pytest.raises(NotImplementedError, match="semantic"):
            chunk_document(markdown("text"), strategy="semantic", chunk_size=10, chunk_overlap=0)


def test_chunk_documents_concatenates_in_document_order() -> None:
    documents = [markdown("Alpha.", "a.md"), markdown("Beta.", "b.md")]

    chunks = chunk_documents(documents, strategy="recursive", chunk_size=100, chunk_overlap=0)

    assert [c.id for c in chunks] == ["a.md#0", "b.md#0"]


@pytest.mark.parametrize("strategy", ["recursive", "fixed"])
def test_empty_text_gives_no_chunks(strategy: str) -> None:
    chunks = chunk_document(markdown(""), strategy=strategy, chunk_size=10, chunk_overlap=0)  # type: ignore[arg-type]

    assert chunks == ()
