"""Tests for retrieval.seed (the logic behind scripts/seed_vector_store.py)."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.ingestion.chunk_io import save_chunks
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval import seed as seed_module
from rag_eval_platform.retrieval.seed import main, seed
from rag_eval_platform.retrieval.vector_store import SearchResult, VectorStoreError

CHUNKS = (
    Chunk(id="a.md#0", doc_id="a.md", index=0, text="alpha", start_index=0),
    Chunk(id="b.md#0", doc_id="b.md", index=0, text="beta", start_index=0),
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        return [[float(len(t))] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        raise AssertionError("seeding must not embed queries")


class FakeStore:
    def __init__(self) -> None:
        self.stored: list[tuple[Chunk, list[float]]] = []

    def replace_all(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        self.stored = [(c, list(e)) for c, e in zip(chunks, embeddings, strict=True)]

    def search(self, query_embedding: Sequence[float], k: int) -> list[SearchResult]:
        raise AssertionError("not used")

    def count(self) -> int:
        return len(self.stored)


def test_seed_embeds_chunk_texts_and_replaces_store_contents() -> None:
    embedder, store = FakeEmbedder(), FakeStore()

    stored = seed(CHUNKS, embedder, store)

    assert stored == 2
    assert embedder.batches == [["alpha", "beta"]]
    assert store.stored == [(CHUNKS[0], [5.0]), (CHUNKS[1], [4.0])]


def test_seed_rejects_empty_chunks() -> None:
    with pytest.raises(ValueError, match="No chunks"):
        seed((), FakeEmbedder(), FakeStore())


class TestMain:
    @pytest.fixture(autouse=True)
    def isolated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeStore:
        monkeypatch.chdir(tmp_path)
        get_settings.cache_clear()
        store = FakeStore()
        monkeypatch.setattr(seed_module, "create_embedder", lambda settings: FakeEmbedder())
        monkeypatch.setattr(seed_module, "create_vector_store", lambda settings: store)
        return store

    def test_seeds_from_chunks_file(self, tmp_path: Path, isolated: FakeStore) -> None:
        path = tmp_path / "chunks.jsonl"
        save_chunks(CHUNKS, path)

        assert main(["--chunks", str(path)]) == 0
        assert isolated.count() == 2

    def test_default_chunks_path_comes_from_settings(
        self, tmp_path: Path, isolated: FakeStore
    ) -> None:
        save_chunks(CHUNKS, tmp_path / "data" / "processed" / "chunks.jsonl")

        assert main([]) == 0
        assert isolated.count() == 2

    def test_missing_chunks_file_returns_1_with_hint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--chunks", str(tmp_path / "missing.jsonl")]) == 1
        assert "run ingestion first" in capsys.readouterr().err

    def test_unreachable_store_returns_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def down(settings: object) -> None:
            raise VectorStoreError("Cannot reach Chroma")

        monkeypatch.setattr(seed_module, "create_vector_store", down)
        path = tmp_path / "chunks.jsonl"
        save_chunks(CHUNKS, path)

        assert main(["--chunks", str(path)]) == 1
