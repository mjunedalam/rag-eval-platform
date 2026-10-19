"""End-to-end RAG pipeline: question -> retrieve -> generate -> cited answer.

Every answered question is logged as one structured line, and, when a recorder is given,
stored in the observability log (see ``observability/logger.py``).
"""

import logging
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Answer, AnswerStream, create_generator
from rag_eval_platform.observability.langfuse_tracer import create_tracer
from rag_eval_platform.observability.logger import create_query_logger, new_id
from rag_eval_platform.observability.store import Source
from rag_eval_platform.observability.tracing import (
    NOOP_TRACER,
    LlmTraceRef,
    QuestionTrace,
    Tracer,
    chunk_rows,
)
from rag_eval_platform.retrieval.retriever import create_retriever
from rag_eval_platform.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


class _Retriever(Protocol):
    def retrieve(self, query: str) -> list[SearchResult]: ...


@runtime_checkable
class _TracedRetriever(Protocol):
    def retrieve_traced(self, query: str, trace: QuestionTrace) -> list[SearchResult]: ...


class _Generator(Protocol):
    def generate(
        self, question: str, results: Sequence[SearchResult], trace: LlmTraceRef | None = None
    ) -> Answer: ...

    def stream(
        self,
        question: str,
        results: Sequence[SearchResult],
        on_complete: Callable[[Answer], None] | None = None,
        trace: LlmTraceRef | None = None,
    ) -> AnswerStream: ...


class QueryRecorder(Protocol):
    def record(self, answer: Answer, *, retrieve_ms: float, total_ms: float,
               query_id: str | None = None) -> str | None: ...  # fmt: skip

    def record_error(self, question: str, error: BaseException, *, total_ms: float,
                     query_id: str | None = None) -> str | None: ...  # fmt: skip


@dataclass(frozen=True)
class TimedAnswer:
    answer: Answer
    retrieve_ms: float
    total_ms: float
    query_id: str | None = None  # the observability log's id, when recorded


class RecordedStream:
    """Iterate for the answer text; once it finishes, ``query_id`` holds the log's id."""

    def __init__(self, stream: AnswerStream, question: str, recorder: QueryRecorder | None,
                 started: float, retrieve_ms: float, trace: QuestionTrace,
                 query_id: str) -> None:  # fmt: skip
        self._stream, self._question, self._recorder = stream, question, recorder
        self._started, self._retrieve_ms = started, retrieve_ms
        self._trace, self._new_id = trace, query_id
        self.query_id: str | None = None

    @property
    def answer(self) -> Answer:
        return self._stream.answer

    def __iter__(self) -> Iterator[str]:
        try:
            yield from self._stream
        except Exception as exc:
            self._trace.finish(error=exc)
            _record_error(self._recorder, self._question, exc, self._started, self._new_id)
            raise
        answer = self._stream.answer
        self._trace.finish(answer=answer.text, metadata=answer_metadata(answer))
        if self._recorder is not None:
            recorder, retrieve_ms, query_id = self._recorder, self._retrieve_ms, self._new_id
            self.query_id = _safely(
                lambda: recorder.record(answer, retrieve_ms=retrieve_ms,
                                        total_ms=_ms_since(self._started), query_id=query_id)
            )  # fmt: skip


class RagPipeline:
    """Retrieve, then generate. Each question gets a ``query_id`` up front: it names the
    question's trace and its row in the observability log, so the two always match."""

    def __init__(self, retriever: _Retriever, generator: _Generator,
                 recorder: QueryRecorder | None = None, *, tracer: Tracer = NOOP_TRACER,
                 tags: Sequence[str] = (),
                 trace_metadata: Mapping[str, Any] | None = None) -> None:  # fmt: skip
        self.retriever = retriever
        self.generator = generator
        self.recorder = recorder
        self.tracer = tracer
        self.tags = tuple(tags)
        self.trace_metadata = dict(trace_metadata or {})

    def ask(self, question: str) -> Answer:
        return self.ask_timed(question).answer

    def ask_timed(self, question: str) -> TimedAnswer:
        """Answer, report how long retrieval and the whole request took, and record it."""
        started, query_id = time.perf_counter(), new_id()
        trace = self._start_trace(query_id, question)
        try:
            results = self._retrieve(question, trace)
            retrieve_ms = _ms_since(started)
            answer = self.generator.generate(question, results, trace=trace.llm_ref)
        except Exception as exc:
            trace.finish(error=exc)
            _record_error(self.recorder, question, exc, started, query_id)
            raise
        total_ms = _ms_since(started)
        trace.finish(answer=answer.text, metadata=answer_metadata(answer))
        _log_answer(answer, total_ms=total_ms)
        recorder = self.recorder
        recorded = (_safely(lambda: recorder.record(answer, retrieve_ms=retrieve_ms,
                                                     total_ms=total_ms, query_id=query_id))
                    if recorder is not None else None)  # fmt: skip
        return TimedAnswer(answer, retrieve_ms, total_ms, recorded)

    def ask_stream(self, question: str) -> RecordedStream:
        """Retrieve now, then stream the answer; it is logged and recorded once it finishes."""
        started, query_id = time.perf_counter(), new_id()
        trace = self._start_trace(query_id, question)
        try:
            results = self._retrieve(question, trace)
            stream = self.generator.stream(
                question, results,
                on_complete=lambda answer: _log_answer(answer, total_ms=_ms_since(started)),
                trace=trace.llm_ref,
            )  # fmt: skip
        except Exception as exc:
            trace.finish(error=exc)
            _record_error(self.recorder, question, exc, started, query_id)
            raise
        return RecordedStream(stream, question, self.recorder, started, _ms_since(started),
                              trace, query_id)  # fmt: skip

    def _start_trace(self, query_id: str, question: str) -> QuestionTrace:
        return self.tracer.start(query_id, question, tags=self.tags,
                                 metadata=self.trace_metadata)  # fmt: skip

    def _retrieve(self, question: str, trace: QuestionTrace) -> list[SearchResult]:
        if isinstance(self.retriever, _TracedRetriever):
            return self.retriever.retrieve_traced(question, trace)
        stage = trace.stage("retrieve-context", input=question)
        try:
            results = self.retriever.retrieve(question)
        except Exception as exc:
            stage.end(error=exc)
            raise
        stage.end(output=chunk_rows(results))
        return results


def answer_metadata(answer: Answer) -> dict[str, Any]:
    """What a reviewer filters traces by, beyond the answer text itself."""
    return {
        "refusal": answer.is_refusal,
        "cited_doc_ids": list(dict.fromkeys(c.source.chunk.doc_id for c in answer.citations)),
        "invalid_citations": list(answer.invalid_citations),
        "prompt_version": answer.prompt_version,
        "sources": len(answer.sources),
    }


def trace_metadata(settings: Settings, *, collection: str | None = None) -> dict[str, Any]:
    """The retrieval settings that shaped an answer, recorded on its trace."""
    return {
        "collection": collection or settings.collection_name,
        "top_k": settings.top_k,
        "rerank": settings.rerank,
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
    }


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
                  started: float, query_id: str | None = None) -> None:  # fmt: skip
    if recorder is not None:
        _safely(lambda: recorder.record_error(question, error, total_ms=_ms_since(started),
                                              query_id=query_id))  # fmt: skip


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
    """The pipeline for ``settings``; with a ``source``, answers go to the observability log.

    Every question is traced when Langfuse is configured (see observability/langfuse_tracer).
    """
    recorder = create_query_logger(settings, source) if source is not None else None
    generator = create_generator(settings)
    return RagPipeline(
        create_retriever(settings), generator, recorder, tracer=create_tracer(settings),
        tags=(source or "library", generator.style), trace_metadata=trace_metadata(settings),
    )  # fmt: skip
