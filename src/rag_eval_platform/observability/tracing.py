"""Tracing interface: one trace per answered question, with a span per pipeline stage.

The pipeline only talks to these protocols, never to a tracing vendor. ``NOOP_TRACER`` is
the default and does nothing; ``langfuse_tracer.LangfuseTracer`` sends traces to Langfuse.
Handles are explicit objects, not "the current span", so a trace started on one thread can
be finished on another (the playground streams answers on a background thread).

Stage names are verb-first and stable, because Langfuse filters, dashboards and evaluators
refer to them by name; run-specific values belong in metadata, never in a name.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from rag_eval_platform.retrieval.vector_store import SearchResult

StageName = Literal["retrieve-context", "rerank-chunks"]
ScoreType = Literal["NUMERIC", "BOOLEAN"]


@dataclass(frozen=True)
class LlmTraceRef:
    """Where the LLM call's generation belongs: this trace, under this observation."""

    trace_id: str
    parent_observation_id: str
    public_key: str


class Stage(Protocol):
    def end(
        self,
        *,
        output: Any = None,
        metadata: Mapping[str, Any] | None = None,
        error: BaseException | None = None,
    ) -> None: ...


class QuestionTrace(Protocol):
    @property
    def llm_ref(self) -> LlmTraceRef | None: ...

    def stage(
        self, name: StageName, *, input: Any = None, metadata: Mapping[str, Any] | None = None
    ) -> Stage: ...

    def finish(
        self,
        *,
        answer: str | None = None,
        error: BaseException | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None: ...


class Tracer(Protocol):
    @property
    def enabled(self) -> bool: ...

    def start(
        self,
        query_id: str,
        question: str,
        *,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> QuestionTrace: ...

    def score(
        self,
        query_id: str,
        name: str,
        value: float,
        *,
        comment: str | None = None,
        data_type: ScoreType = "NUMERIC",
    ) -> None: ...

    def trace_url(self, query_id: str) -> str | None: ...

    def flush(self) -> None: ...


class _NoopStage:
    def end(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None,
            error: BaseException | None = None) -> None:  # fmt: skip
        return None


class _NoopQuestion:
    @property
    def llm_ref(self) -> LlmTraceRef | None:
        return None

    def stage(self, name: StageName, *, input: Any = None,
              metadata: Mapping[str, Any] | None = None) -> Stage:  # fmt: skip
        return NOOP_STAGE

    def finish(self, *, answer: str | None = None, error: BaseException | None = None,
               metadata: Mapping[str, Any] | None = None) -> None:  # fmt: skip
        return None


class _NoopTracer:
    @property
    def enabled(self) -> bool:
        return False

    def start(self, query_id: str, question: str, *, tags: Sequence[str] = (),
              metadata: Mapping[str, Any] | None = None) -> QuestionTrace:  # fmt: skip
        return NOOP_QUESTION

    def score(self, query_id: str, name: str, value: float, *, comment: str | None = None,
              data_type: ScoreType = "NUMERIC") -> None:  # fmt: skip
        return None

    def trace_url(self, query_id: str) -> str | None:
        return None

    def flush(self) -> None:
        return None


NOOP_STAGE: Stage = _NoopStage()
NOOP_QUESTION: QuestionTrace = _NoopQuestion()
NOOP_TRACER: Tracer = _NoopTracer()


def chunk_rows(results: Sequence[SearchResult]) -> list[dict[str, Any]]:
    """Retrieved chunks as a reviewer needs them in a trace: where from, how close, the text."""
    return [
        {
            "chunk_id": r.chunk.id,
            "doc_id": r.chunk.doc_id,
            "page": r.chunk.page,
            "score": round(r.score, 4),
            "text": r.chunk.text,
        }
        for r in results
    ]
