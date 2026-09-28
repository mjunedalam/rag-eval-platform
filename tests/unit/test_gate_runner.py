"""Tests for evaluation.gate_runner: rebuild the index from the corpus and score it."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation import gate_runner
from rag_eval_platform.evaluation.evaluator import RetrievalThresholds
from rag_eval_platform.evaluation.gate_runner import (
    rebuild_and_evaluate,
    rebuild_index_from_settings,
)
from rag_eval_platform.evaluation.golden_dataset import GoldenExample
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult

VOCAB = ("alpha", "beta")
THRESHOLDS = RetrievalThresholds(0.8, 0.7, 0.7)
EXAMPLES = (
    GoldenExample(id="q1", question="alpha please", expected_answer="a",
                  relevant_doc_ids=("alpha.md",), query_type="short"),
    GoldenExample(id="q2", question="beta", expected_answer="b",
                  relevant_doc_ids=("beta.md",), query_type="short"),
)  # fmt: skip


class WordEmbedder:
    """Counts the vocabulary words: enough meaning for a two-document corpus."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        words = text.lower().split()
        return [words.count(word) + 0.01 for word in VOCAB]


class MemoryStore:
    def __init__(self) -> None:
        self.items: list[tuple[Chunk, list[float]]] = []

    def replace_all(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        self.items = [(c, list(e)) for c, e in zip(chunks, embeddings, strict=True)]

    def search(self, query_embedding: Sequence[float], k: int) -> list[SearchResult]:
        scored = [(sum(a * b for a, b in zip(query_embedding, e, strict=True)), c)
                  for c, e in self.items]  # fmt: skip
        scored.sort(key=lambda pair: -pair[0])
        return [SearchResult(chunk=c, score=s) for s, c in scored[:k]]

    def count(self) -> int:
        return len(self.items)


def write_corpus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "alpha.md").write_text("alpha alpha alpha", encoding="utf-8")
    (root / "beta.md").write_text("beta beta beta", encoding="utf-8")
    return root


def test_rebuilds_the_index_and_scores_every_question(tmp_path: Path) -> None:
    store = MemoryStore()
    progress: list[str] = []

    report = rebuild_and_evaluate(
        EXAMPLES, raw_dir=write_corpus(tmp_path / "raw"), strategy="recursive",
        chunk_size=200, chunk_overlap=0, embedder=WordEmbedder(), store=store, top_k=1,
        reranker=None, rerank_candidates=20, thresholds=THRESHOLDS, on_progress=progress.append,
    )  # fmt: skip

    assert store.count() == 2
    assert report.overall.recall == 1.0
    assert report.k == 1
    assert len(progress) == 2


def test_rebuild_from_settings_indexes_the_configured_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = MemoryStore()
    monkeypatch.setattr(gate_runner, "create_embedder", lambda settings: WordEmbedder())
    monkeypatch.setattr(gate_runner, "create_vector_store", lambda settings: store)
    settings = get_settings().model_copy(
        update={"raw_data_dir": write_corpus(tmp_path / "raw"), "chunk_size": 200,
                "chunk_overlap": 0}
    )  # fmt: skip

    assert rebuild_index_from_settings(settings) == 2
    assert store.count() == 2
