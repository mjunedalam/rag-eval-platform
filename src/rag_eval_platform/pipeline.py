"""End-to-end RAG pipeline: question -> retrieve -> generate -> cited answer.

Every answered question is logged as one structured line (what was retrieved, what was
cited, latency and token counts), which is the raw material for observability.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Answer, AnswerStream, create_generator
from rag_eval_platform.retrieval.retriever import create_retriever
from rag_eval_platform.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


class _Retriever(Protocol):
    def retrieve(self, query: str) -> list[SearchResult]: ...


class _Generator(Protocol):
    def generate(self, question: str, results: Sequence[SearchResult]) -> Answer: ...

    def stream(
        self,
        question: str,
        results: Sequence[SearchResult],
        on_complete: Callable[[Answer], None] | None = None,
    ) -> AnswerStream: ...


@dataclass(frozen=True)
class TimedAnswer:
    answer: Answer
    retrieve_ms: float
    total_ms: float


class RagPipeline:
    def __init__(self, retriever: _Retriever, generator: _Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    def ask(self, question: str) -> Answer:
        return self.ask_timed(question).answer

    def ask_timed(self, question: str) -> TimedAnswer:
        """Answer and report how long retrieval and the whole request took."""
        started = time.perf_counter()
        results = self.retriever.retrieve(question)
        retrieved = time.perf_counter()
        answer = self.generator.generate(question, results)
        total_ms = (time.perf_counter() - started) * 1000
        _log_answer(answer, total_ms=total_ms)
        return TimedAnswer(answer, (retrieved - started) * 1000, total_ms)

    def ask_stream(self, question: str) -> AnswerStream:
        """Retrieve now, then stream the answer; the log line is written once it finishes."""
        started = time.perf_counter()
        results = self.retriever.retrieve(question)
        return self.generator.stream(
            question,
            results,
            on_complete=lambda answer: _log_answer(
                answer, total_ms=(time.perf_counter() - started) * 1000
            ),
        )


def _log_answer(answer: Answer, total_ms: float) -> None:
    logger.info(
        "Question answered",
        extra={
            "question": answer.question,
            "retrieved_chunk_ids": [r.chunk.id for r in answer.sources],
            "cited_doc_ids": list(dict.fromkeys(c.source.chunk.doc_id for c in answer.citations)),
            "invalid_citations": list(answer.invalid_citations),
            "refusal": answer.is_refusal,
            "model": answer.model,
            "prompt_version": answer.prompt_version,
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "generation_ms": round(answer.latency_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    )


def create_pipeline(settings: Settings) -> RagPipeline:
    return RagPipeline(create_retriever(settings), create_generator(settings))
