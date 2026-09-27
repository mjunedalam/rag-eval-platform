"""Tests for the playground core pieces behind the visual tabs: the question's embedding,
the index snapshot, judging one answer, and loading the golden-set reports."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from rag_eval_platform.evaluation.citation_validity import CitationValidity, CitationVerdict
from rag_eval_platform.evaluation.judge import JudgeError, JudgeSample, JudgeScores
from rag_eval_platform.generation.generator import Answer
from rag_eval_platform.generation.prompt_templates import NO_ANSWER
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import (
    IndexSnapshot,
    QueryTrace,
    Reports,
    index_snapshot,
    judge_answer,
    load_reports,
    retrieve_step,
    trace_key,
)
from rag_eval_platform.retrieval.vector_store import SearchResult


def hit(chunk_id: str, text: str = "text", score: float = 0.5) -> SearchResult:
    doc_id = chunk_id.split("#")[0]
    return SearchResult(Chunk(id=chunk_id, doc_id=doc_id, index=0, text=text, start_index=0), score)


def make_answer(text: str, sources: tuple[SearchResult, ...], question: str = "q?") -> Answer:
    return Answer(
        question=question, text=text, citations=(), invalid_citations=(), sources=sources,
        model="m", prompt_version="v1", input_tokens=1, output_tokens=1, latency_ms=10.0,
    )  # fmt: skip


def make_trace(text: str = "A [1].", question: str = "q?") -> QueryTrace:
    sources = (hit("a.md#0", "alpha"), hit("b.md#0", "beta"))
    return QueryTrace(sources, sources, make_answer(text, sources, question), 5.0, (0.1, 0.2))


class VectorEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.6, 0.8]


class OneHitStore:
    def search(self, query_embedding: list[float], k: int) -> list[SearchResult]:
        return [hit("a.md#0")]


def test_retrieve_step_keeps_the_query_embedding() -> None:
    step = retrieve_step("q?", embedder=VectorEmbedder(), store=OneHitStore(), top_k=1)  # type: ignore[arg-type]

    assert step.query_embedding == (0.6, 0.8)


def test_query_trace_embedding_defaults_to_empty() -> None:
    sources = (hit("a.md#0"),)
    trace = QueryTrace(sources, sources, make_answer("A [1].", sources), 1.0)

    assert trace.query_embedding == ()


class ExportingStore:
    def export(self) -> tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]:
        chunks = (
            Chunk(id="a.md#0", doc_id="a.md", index=0, text="x", start_index=0),
            Chunk(id="b.md#0", doc_id="b.md", index=0, text="y", start_index=0),
            Chunk(id="a.md#1", doc_id="a.md", index=1, text="z", start_index=1),
        )
        return chunks, ((1.0,), (2.0,), (3.0,))


def test_index_snapshot_reads_the_store() -> None:
    snapshot = index_snapshot(ExportingStore())

    assert len(snapshot.chunks) == 3
    assert snapshot.vectors[1] == (2.0,)
    assert snapshot.doc_ids == ("a.md", "b.md")


def test_index_snapshot_of_an_empty_store_is_empty() -> None:
    class Empty:
        def export(self) -> tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]:
            return (), ()

    assert index_snapshot(Empty()) == IndexSnapshot((), ())


def test_trace_key_changes_with_question_and_answer() -> None:
    assert trace_key(make_trace()) == trace_key(make_trace())
    assert trace_key(make_trace(question="other?")) != trace_key(make_trace())
    assert trace_key(make_trace(text="B [2].")) != trace_key(make_trace())


@dataclass
class FakeJudge:
    model: str = "judge"
    samples: list[JudgeSample] | None = None

    def score(self, sample: JudgeSample, *, full: bool = False) -> JudgeScores:
        self.samples = [sample]
        return JudgeScores(faithfulness=0.5, answer_relevance=0.9)


class FakeChecker:
    def __init__(self, error: bool = False) -> None:
        self.error = error

    def check(self, answer: Answer) -> CitationValidity:
        if self.error:
            raise JudgeError("unreadable")
        return CitationValidity((CitationVerdict("A.", 1, "a.md", True),))


def test_judge_answer_scores_the_answer_against_its_chunks() -> None:
    judge = FakeJudge()

    result = judge_answer(make_trace(), judge, FakeChecker())

    assert result.scores == JudgeScores(faithfulness=0.5, answer_relevance=0.9)
    assert result.citations.score == 1.0
    assert result.judge_ms >= 0
    assert result.note == ""
    assert judge.samples is not None
    assert judge.samples[0].contexts == ("alpha", "beta")


def test_judge_answer_keeps_scores_when_citation_verdicts_fail() -> None:
    result = judge_answer(make_trace(), FakeJudge(), FakeChecker(error=True))

    assert result.citations.score is None
    assert "unreadable" in result.note


def test_judge_answer_rejects_refusals() -> None:
    with pytest.raises(ValueError, match="refusal"):
        judge_answer(make_trace(text=NO_ANSWER), FakeJudge(), FakeChecker())


def test_load_reports_reads_both_files(tmp_path: Path) -> None:
    (tmp_path / "retrieval_report.json").write_text(json.dumps({"k": 5}), encoding="utf-8")
    (tmp_path / "generation_report.json").write_text(json.dumps({"passed": True}), encoding="utf-8")

    assert load_reports(tmp_path) == Reports({"k": 5}, {"passed": True})


def test_load_reports_tolerates_missing_and_broken_files(tmp_path: Path) -> None:
    (tmp_path / "generation_report.json").write_text("{not json", encoding="utf-8")

    assert load_reports(tmp_path) == Reports(None, None)
