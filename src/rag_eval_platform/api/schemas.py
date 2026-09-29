"""What the API accepts and returns (pydantic models), and the mapping from an Answer."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from rag_eval_platform.generation.prompt_templates import AnswerStyle
from rag_eval_platform.pipeline import TimedAnswer

MAX_QUESTION_CHARS = 2000
Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1,
                                            max_length=MAX_QUESTION_CHARS)]  # fmt: skip


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    answer_style: AnswerStyle = "concise"


class CitationOut(BaseModel):
    number: int
    doc_id: str
    page: int | None
    chunk_id: str


class SourceOut(BaseModel):
    number: int
    doc_id: str
    page: int | None
    chunk_id: str
    score: float
    text: str


class Timings(BaseModel):
    retrieve_ms: float
    generate_ms: float
    total_ms: float


class QueryResponse(BaseModel):
    question: str
    answer: str
    refusal: bool
    citations: list[CitationOut]
    invalid_citations: list[int]
    sources: list[SourceOut]
    model: str
    prompt_version: str
    input_tokens: int | None
    output_tokens: int | None
    timings: Timings


class CheckOut(BaseModel):
    ok: bool
    detail: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    checks: dict[str, CheckOut]


class ErrorResponse(BaseModel):
    error: str
    hint: str | None = None


def to_response(timed: TimedAnswer) -> QueryResponse:
    answer = timed.answer
    return QueryResponse(
        question=answer.question,
        answer=answer.text,
        refusal=answer.is_refusal,
        citations=[CitationOut(number=c.number, doc_id=c.source.chunk.doc_id,
                               page=c.source.chunk.page, chunk_id=c.source.chunk.id)
                   for c in answer.citations],
        invalid_citations=list(answer.invalid_citations),
        sources=[SourceOut(number=n, doc_id=s.chunk.doc_id, page=s.chunk.page,
                           chunk_id=s.chunk.id, score=s.score, text=s.chunk.text)
                 for n, s in enumerate(answer.sources, start=1)],
        model=answer.model,
        prompt_version=answer.prompt_version,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        timings=Timings(retrieve_ms=round(timed.retrieve_ms, 1),
                        generate_ms=round(answer.latency_ms, 1),
                        total_ms=round(timed.total_ms, 1)),
    )  # fmt: skip
