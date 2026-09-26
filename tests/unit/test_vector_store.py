"""Tests for retrieval.vector_store against an in-memory fake of the Chroma client."""

from dataclasses import dataclass, field
from typing import Any

import chromadb
import pytest
from chromadb.errors import NotFoundError

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import (
    ChromaVectorStore,
    VectorStoreError,
    create_vector_store,
)


@dataclass
class FakeCollection:
    configuration: dict[str, Any]
    rows: dict[str, tuple[list[float], str, dict[str, Any]]] = field(default_factory=dict)
    upsert_batch_sizes: list[int] = field(default_factory=list)

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.upsert_batch_sizes.append(len(ids))
        for row in zip(ids, embeddings, documents, metadatas, strict=True):
            self.rows[row[0]] = row[1:]

    def count(self) -> int:
        return len(self.rows)

    def query(
        self, query_embeddings: list[list[float]], n_results: int, include: list[str]
    ) -> dict[str, Any]:
        (query,) = query_embeddings
        ranked = sorted(
            self.rows.items(),
            key=lambda item: -sum(a * b for a, b in zip(query, item[1][0], strict=True)),
        )[:n_results]
        return {
            "ids": [[id_ for id_, _ in ranked]],
            "documents": [[row[1] for _, row in ranked]],
            "metadatas": [[row[2] for _, row in ranked]],
            "distances": [
                [1 - sum(a * b for a, b in zip(query, row[0], strict=True)) for _, row in ranked]
            ],
        }


@dataclass
class FakeChromaClient:
    max_batch_size: int = 2
    collections: dict[str, FakeCollection] = field(default_factory=dict)

    def get_collection(self, name: str) -> FakeCollection:
        if name not in self.collections:
            raise NotFoundError(f"Collection [{name}] does not exist")
        return self.collections[name]

    def create_collection(
        self, name: str, configuration: dict[str, Any], embedding_function: None
    ) -> FakeCollection:
        self.collections[name] = FakeCollection(configuration)
        return self.collections[name]

    def delete_collection(self, name: str) -> None:
        if name not in self.collections:
            raise NotFoundError(f"Collection [{name}] does not exist")
        del self.collections[name]

    def get_max_batch_size(self) -> int:
        return self.max_batch_size


CHUNKS = (
    Chunk(id="a.md#0", doc_id="a.md", index=0, text="About force", start_index=0),
    Chunk(id="b.pdf#0", doc_id="b.pdf", index=0, text="About energy", start_index=0, page=3),
    Chunk(id="c.md#0", doc_id="c.md", index=0, text="About motion", start_index=10),
)
EMBEDDINGS = ([1.0, 0.0], [0.0, 1.0], [0.6, 0.8])


def make_store(client: FakeChromaClient | None = None) -> ChromaVectorStore:
    return ChromaVectorStore(client or FakeChromaClient(), collection_name="test")  # type: ignore[arg-type]


class TestReplaceAll:
    def test_stores_every_chunk_in_batches_with_cosine_space(self) -> None:
        client = FakeChromaClient(max_batch_size=2)
        store = make_store(client)

        store.replace_all(CHUNKS, EMBEDDINGS)

        collection = client.collections["test"]
        assert store.count() == 3
        assert collection.upsert_batch_sizes == [2, 1]
        assert collection.configuration == {"hnsw": {"space": "cosine"}}

    def test_replaces_previous_contents(self) -> None:
        store = make_store()
        store.replace_all(CHUNKS, EMBEDDINGS)

        store.replace_all(CHUNKS[:1], EMBEDDINGS[:1])

        assert store.count() == 1

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            make_store().replace_all(CHUNKS, EMBEDDINGS[:2])


class TestSearch:
    def test_returns_chunks_ranked_with_cosine_similarity_scores(self) -> None:
        store = make_store()
        store.replace_all(CHUNKS, EMBEDDINGS)

        results = store.search([1.0, 0.0], k=2)

        assert [r.chunk for r in results] == [CHUNKS[0], CHUNKS[2]]
        assert [r.score for r in results] == pytest.approx([1.0, 0.6])

    def test_round_trips_optional_page(self) -> None:
        store = make_store()
        store.replace_all(CHUNKS, EMBEDDINGS)

        (result,) = store.search([0.0, 1.0], k=1)

        assert result.chunk == CHUNKS[1]
        assert result.chunk.page == 3
        assert result.score == pytest.approx(1.0)

    def test_empty_or_missing_collection_raises_with_seed_hint(self) -> None:
        with pytest.raises(VectorStoreError, match="seed_vector_store"):
            make_store().search([1.0, 0.0], k=1)

    def test_count_is_zero_without_collection(self) -> None:
        assert make_store().count() == 0

    def test_rejects_k_below_one(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            make_store().search([1.0], k=0)


class TestConnect:
    def test_unreachable_server_raises_with_docker_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def refuse(**kwargs: Any) -> None:
            raise ValueError("Could not connect to a Chroma server")

        monkeypatch.setattr(chromadb, "HttpClient", refuse)

        with pytest.raises(VectorStoreError, match="docker compose"):
            ChromaVectorStore.connect("localhost", 1, "test")

    def test_create_vector_store_uses_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, Any] = {}

        def fake_http_client(**kwargs: Any) -> FakeChromaClient:
            received.update(kwargs)
            return FakeChromaClient()

        monkeypatch.setattr(chromadb, "HttpClient", fake_http_client)

        store = create_vector_store(
            Settings(chroma_host="chroma", chroma_port=9000, collection_name="docs")
        )

        assert isinstance(store, ChromaVectorStore)
        assert (received["host"], received["port"]) == ("chroma", 9000)

    def test_qdrant_is_not_implemented_yet(self) -> None:
        with pytest.raises(NotImplementedError, match="qdrant"):
            create_vector_store(Settings(vector_store="qdrant"))


def test_malformed_chroma_response_raises() -> None:
    class BrokenCollection(FakeCollection):
        def query(
            self, query_embeddings: list[list[float]], n_results: int, include: list[str]
        ) -> dict[str, Any]:
            return {"ids": [["a.md#0"]]}

    client = FakeChromaClient()
    store = make_store(client)
    store.replace_all(CHUNKS, EMBEDDINGS)
    client.collections["test"].__class__ = BrokenCollection

    with pytest.raises(VectorStoreError, match="Unexpected response"):
        store.search([1.0, 0.0], k=1)
