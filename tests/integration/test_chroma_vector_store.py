"""Integration tests against a real Chroma server.

Start it with: docker compose -f docker/docker-compose.yml up -d
Skipped when no server is reachable at RAG_CHROMA_HOST:RAG_CHROMA_PORT (default localhost:8001).
"""

import uuid
from collections.abc import Iterator

import chromadb
import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import ChromaVectorStore

pytestmark = pytest.mark.integration

CHUNKS = (
    Chunk(id="forces.md#0", doc_id="forces.md", index=0, text="F = ma", start_index=0),
    Chunk(id="book.pdf#3", doc_id="book.pdf", index=3, text="Energy", start_index=90, page=2),
)


@pytest.fixture
def store() -> Iterator[ChromaVectorStore]:
    settings = Settings()
    try:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
    except Exception:
        pytest.skip(f"no Chroma server at {settings.chroma_host}:{settings.chroma_port}")
    name = f"test_{uuid.uuid4().hex[:12]}"
    yield ChromaVectorStore(client, collection_name=name)
    client.delete_collection(name)


def test_stores_searches_and_round_trips_chunks(store: ChromaVectorStore) -> None:
    store.replace_all(CHUNKS, [[1.0, 0.0], [0.0, 1.0]])

    results = store.search([0.0, 1.0], k=2)

    assert store.count() == 2
    assert [r.chunk for r in results] == [CHUNKS[1], CHUNKS[0]]
    assert results[0].score == pytest.approx(1.0, abs=1e-4)
    assert results[1].score == pytest.approx(0.0, abs=1e-4)


def test_replace_all_removes_stale_chunks(store: ChromaVectorStore) -> None:
    store.replace_all(CHUNKS, [[1.0, 0.0], [0.0, 1.0]])

    store.replace_all(CHUNKS[:1], [[1.0, 0.0]])

    assert store.count() == 1
