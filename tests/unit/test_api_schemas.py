"""Tests for api.schemas: what the API accepts and returns."""

import pytest
from pydantic import ValidationError

from rag_eval_platform.api.schemas import QueryRequest, to_response
from rag_eval_platform.generation.generator import Answer, Citation
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.pipeline import TimedAnswer
from rag_eval_platform.retrieval.vector_store import SearchResult

SOURCE = SearchResult(Chunk(id="book.pdf#4", doc_id="book.pdf", index=4, text="F=ma",
                            start_index=0, page=3), 0.41)  # fmt: skip


def test_request_strips_the_question_and_defaults_to_concise() -> None:
    request = QueryRequest(question="  What is MRR?  ")

    assert (request.question, request.answer_style) == ("What is MRR?", "concise")


@pytest.mark.parametrize("body", [
    {"question": "   "}, {"question": "x" * 2001}, {"question": "q", "answer_style": "long"},
    {"question": "q", "extra": 1},
])  # fmt: skip
def test_bad_requests_are_rejected(body: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(body)


def test_to_response_maps_the_answer_with_citations_sources_and_timings() -> None:
    answer = Answer(
        question="q?", text="F equals ma [1].", citations=(Citation(1, SOURCE),),
        invalid_citations=(7,), sources=(SOURCE,), model="qwen3:8b", prompt_version="v1",
        input_tokens=100, output_tokens=12, latency_ms=900.0,
    )  # fmt: skip

    response = to_response(TimedAnswer(answer, retrieve_ms=40.0, total_ms=950.0))

    assert response.answer == "F equals ma [1]."
    assert response.citations[0].model_dump() == {"number": 1, "doc_id": "book.pdf", "page": 3,
                                                  "chunk_id": "book.pdf#4"}  # fmt: skip
    assert response.sources[0].score == 0.41
    assert response.invalid_citations == [7]
    assert response.refusal is False
    assert response.timings.model_dump() == {"retrieve_ms": 40.0, "generate_ms": 900.0,
                                             "total_ms": 950.0}  # fmt: skip
