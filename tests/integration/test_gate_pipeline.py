"""The gate's rebuild-and-score step against a real Chroma server (fake embedder).

Skipped when no server is reachable at RAG_CHROMA_HOST:RAG_CHROMA_PORT.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import chromadb
import pytest
from tests.unit.test_gate_runner import EXAMPLES, THRESHOLDS, WordEmbedder, write_corpus

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.gate_runner import rebuild_and_evaluate
from rag_eval_platform.retrieval.vector_store import ChromaVectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store() -> Iterator[ChromaVectorStore]:
    settings = Settings()
    try:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
    except Exception:
        pytest.skip(f"no Chroma server at {settings.chroma_host}:{settings.chroma_port}")
    name = f"test_gate_{uuid.uuid4().hex[:12]}"
    yield ChromaVectorStore(client, collection_name=name)
    client.delete_collection(name)


def test_rebuild_and_evaluate_scores_the_corpus_in_chroma(
    store: ChromaVectorStore, tmp_path: Path
) -> None:
    report = rebuild_and_evaluate(
        EXAMPLES, raw_dir=write_corpus(tmp_path / "raw"), strategy="recursive",
        chunk_size=200, chunk_overlap=0, embedder=WordEmbedder(), store=store, top_k=1,
        reranker=None, rerank_candidates=20, thresholds=THRESHOLDS,
    )  # fmt: skip

    assert store.count() == 2
    assert report.overall.recall == 1.0
