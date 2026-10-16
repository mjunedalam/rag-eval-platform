"""End-to-end RAG pipeline: question -> retrieve -> generate -> cited answer.

Every answered question is logged as one structured line, and, when a recorder is given,
stored in the observability log (see ``observability/logger.py``).
"""

import logging
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Answer, AnswerStream, create_generator
from rag_eval_platform.observability.logger import create_query_logger
from rag_eval_platform.observability.store import Source
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


class QueryRecorder(Protocol):
    def record(self, answer: Answer, *, retrieve_ms: float, total_ms: float) -> str | None: ...

    def record_error(
        self, question: str, error: BaseException, *, total_ms: float
    ) -> str | None: ...


@dataclass(frozen=True)
class TimedAnswer:
    answer: Answer
    retrieve_ms: float
    total_ms: float
    query_id: str | None = None  # the observability log's id, when recorded


class RecordedStream:
    """Iterate for the answer text; once it finishes, ``query_id`` holds the log's id."""

    def __init__(self, stream: AnswerStream, question: str, recorder: QueryRecorder | None,
                 started: float, retrieve_ms: float) -> None:  # fmt: skip
        self._stream, self._question, self._recorder = stream, question, recorder
        self._started, self._retrieve_ms = started, retrieve_ms
        self.query_id: str | None = None

    @property
    def answer(self) -> Answer:
        return self._stream.answer

    def __iter__(self) -> Iterator[str]:
        try:
            yield from self._stream
        except Exception as exc:
            _record_error(self._recorder, self._question, exc, self._started)
            raise
        if self._recorder is not None:
            recorder, answer, retrieve_ms = self._recorder, self._stream.answer, self._retrieve_ms
            self.query_id = _safely(
                lambda: recorder.record(
                    answer, retrieve_ms=retrieve_ms, total_ms=_ms_since(self._started)
                )
            )


class RagPipeline:
    def __init__(self, retriever: _Retriever, generator: _Generator,
                 recorder: QueryRecorder | None = None) -> None:  # fmt: skip
        self.retriever = retriever
        self.generator = generator
        self.recorder = recorder

    def ask(self, question: str) -> Answer:
        return self.ask_timed(question).answer

    def ask_timed(self, question: str) -> TimedAnswer:
        """Answer, report how long retrieval and the whole request took, and record it."""
        started = time.perf_counter()
        try:
            results = self.retriever.retrieve(question)
            retrieve_ms = _ms_since(started)
            answer = self.generator.generate(question, results)
        except Exception as exc:
            _record_error(self.recorder, question, exc, started)
            raise
        total_ms = _ms_since(started)
        _log_answer(answer, total_ms=total_ms)
        recorder = self.recorder
        query_id = (_safely(lambda: recorder.record(answer, retrieve_ms=retrieve_ms,
                                                     total_ms=total_ms))
                    if recorder is not None else None)  # fmt: skip
        return TimedAnswer(answer, retrieve_ms, total_ms, query_id)

    def ask_stream(self, question: str) -> RecordedStream:
        """Retrieve now, then stream the answer; it is logged and recorded once it finishes."""
        started = time.perf_counter()
        try:
            results = self.retriever.retrieve(question)
            stream = self.generator.stream(
                question, results,
                on_complete=lambda answer: _log_answer(answer, total_ms=_ms_since(started)),
            )  # fmt: skip
        except Exception as exc:
            _record_error(self.recorder, question, exc, started)
            raise
        return RecordedStream(stream, question, self.recorder, started, _ms_since(started))


def _ms_since(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _safely(call: Callable[[], str | None]) -> str | None:
    """A recorder that fails must never break an answer."""
    try:
        return call()
    except Exception as exc:
        logger.warning("Could not record the query", extra={"error": str(exc)})
        return None


def _record_error(recorder: QueryRecorder | None, question: str, error: Exception,
                  started: float) -> None:  # fmt: skip
    if recorder is not None:
        _safely(lambda: recorder.record_error(question, error, total_ms=_ms_since(started)))


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


def create_pipeline(settings: Settings, source: Source | None = None) -> RagPipeline:
    """The pipeline for ``settings``; with a ``source``, answers go to the observability log."""
    recorder = create_query_logger(settings, source) if source is not None else None
    return RagPipeline(create_retriever(settings), create_generator(settings), recorder)
