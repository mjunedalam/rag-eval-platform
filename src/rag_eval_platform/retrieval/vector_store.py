"""Vector store abstraction (Chroma / Qdrant).

The rest of the pipeline depends only on the ``VectorStore`` protocol. Chroma runs as a
server in Docker (``docker compose -f docker/docker-compose.yml up -d``) and is reached
over HTTP; Qdrant is planned for production.

Scores are cosine similarities (higher is more similar). Embeddings are expected to be
normalised, which both embedders in ``ingestion.embedding`` guarantee.
"""

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, Self

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.ingestion.chunking import Chunk

# The exact list type Chroma's upsert/query accept (list is invariant, so spell it out).
_ChromaVectors = list[Sequence[float] | Sequence[int]]


class VectorStoreError(Exception):
    """The vector store is unreachable, empty or returned unusable data."""


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def replace_all(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    def search(self, query_embedding: Sequence[float], k: int) -> list[SearchResult]: ...

    def count(self) -> int: ...


class ChromaVectorStore:
    """Stores chunks in one Chroma collection that uses cosine distance."""

    def __init__(self, client: ClientAPI, collection_name: str) -> None:
        self._client = client
        self._collection_name = collection_name

    @classmethod
    def connect(cls, host: str, port: int, collection_name: str) -> Self:
        try:
            client = chromadb.HttpClient(host=host, port=port)
        except Exception as exc:  # the client raises ValueError, ConnectError, ... when down
            raise VectorStoreError(
                f"Cannot reach Chroma at {host}:{port}. Is it running? "
                "Start it with: docker compose -f docker/docker-compose.yml up -d"
            ) from exc
        return cls(client, collection_name)

    def replace_all(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Drop the collection and store ``chunks`` fresh, so no stale chunks survive."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        with suppress(NotFoundError):
            self._client.delete_collection(self._collection_name)
        collection = self._client.create_collection(
            self._collection_name,
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            batch_embeddings: _ChromaVectors = [
                list(e) for e in embeddings[start : start + batch_size]
            ]
            collection.upsert(
                ids=[c.id for c in batch],
                embeddings=batch_embeddings,
                documents=[c.text for c in batch],
                metadatas=[_to_metadata(c) for c in batch],
            )

    def search(self, query_embedding: Sequence[float], k: int) -> list[SearchResult]:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        collection = self._collection()
        if collection is None or collection.count() == 0:
            raise VectorStoreError(
                f"Collection '{self._collection_name}' is empty. "
                "Run: uv run python scripts/seed_vector_store.py"
            )

        query_embeddings: _ChromaVectors = [list(query_embedding)]
        response = collection.query(
            query_embeddings=query_embeddings,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        return _to_results(response)

    def count(self) -> int:
        collection = self._collection()
        return 0 if collection is None else collection.count()

    def _collection(self) -> Collection | None:
        try:
            return self._client.get_collection(self._collection_name)
        except NotFoundError:
            return None


def create_vector_store(settings: Settings) -> VectorStore:
    """Build the store selected by ``RAG_VECTOR_STORE``."""
    if settings.vector_store == "qdrant":
        raise NotImplementedError("qdrant vector store is not implemented yet")
    return ChromaVectorStore.connect(
        settings.chroma_host, settings.chroma_port, settings.collection_name
    )


def _to_metadata(chunk: Chunk) -> dict[str, str | int]:
    metadata: dict[str, str | int] = {
        "doc_id": chunk.doc_id,
        "index": chunk.index,
        "start_index": chunk.start_index,
    }
    if chunk.page is not None:  # Chroma metadata values cannot be None
        metadata["page"] = chunk.page
    return metadata


def _to_results(response: Any) -> list[SearchResult]:
    try:
        rows = zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
            strict=True,
        )
        return [
            SearchResult(
                chunk=Chunk(
                    id=chunk_id,
                    doc_id=str(metadata["doc_id"]),
                    index=int(metadata["index"]),
                    text=document,
                    start_index=int(metadata["start_index"]),
                    page=int(metadata["page"]) if "page" in metadata else None,
                ),
                score=1.0 - float(distance),
            )
            for chunk_id, document, metadata, distance in rows
        ]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise VectorStoreError(f"Unexpected response from Chroma: {exc}") from exc
