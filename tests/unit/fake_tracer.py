"""A tracer that records events instead of sending them, for pipeline/API/playground tests."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_eval_platform.observability.tracing import (
    LlmTraceRef,
    QuestionTrace,
    ScoreType,
    Stage,
    StageName,
)


@dataclass
class FakeStage:
    name: StageName
    input: Any
    metadata: Mapping[str, Any] | None
    output: Any = None
    error: BaseException | None = None
    ended: bool = False

    def end(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None,
            error: BaseException | None = None) -> None:  # fmt: skip
        self.output, self.error, self.ended = output, error, True


@dataclass
class FakeQuestion:
    query_id: str
    question: str
    tags: Sequence[str]
    metadata: Mapping[str, Any] | None
    stages: list[FakeStage] = field(default_factory=list)
    answer: str | None = None
    error: BaseException | None = None
    finish_metadata: Mapping[str, Any] | None = None
    finished: bool = False

    @property
    def llm_ref(self) -> LlmTraceRef | None:
        return LlmTraceRef(f"trace-{self.query_id}", f"root-{self.query_id}", "fake-public-value")

    def stage(self, name: StageName, *, input: Any = None,
              metadata: Mapping[str, Any] | None = None) -> Stage:  # fmt: skip
        stage = FakeStage(name, input, metadata)
        self.stages.append(stage)
        return stage

    def finish(self, *, answer: str | None = None, error: BaseException | None = None,
               metadata: Mapping[str, Any] | None = None) -> None:  # fmt: skip
        self.answer, self.error, self.finish_metadata, self.finished = answer, error, metadata, True


@dataclass
class FakeTracer:
    questions: list[FakeQuestion] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    flushed: int = 0

    @property
    def enabled(self) -> bool:
        return True

    def start(self, query_id: str, question: str, *, tags: Sequence[str] = (),
              metadata: Mapping[str, Any] | None = None) -> QuestionTrace:  # fmt: skip
        trace = FakeQuestion(query_id, question, tuple(tags), metadata)
        self.questions.append(trace)
        return trace

    def score(self, query_id: str, name: str, value: float, *, comment: str | None = None,
              data_type: ScoreType = "NUMERIC") -> None:  # fmt: skip
        self.scores.append({"query_id": query_id, "name": name, "value": value,
                            "comment": comment, "data_type": data_type})  # fmt: skip

    def trace_url(self, query_id: str) -> str | None:
        return f"https://lf.example/{query_id}"

    def flush(self) -> None:
        self.flushed += 1
