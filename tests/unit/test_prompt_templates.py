"""Tests for generation.prompt_templates."""

from rag_eval_platform.generation.prompt_templates import (
    NO_ANSWER,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    Message,
    build_messages,
    format_context,
)
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult


def result(doc_id: str, text: str, page: int | None = None) -> SearchResult:
    chunk = Chunk(id=f"{doc_id}#0", doc_id=doc_id, index=0, text=text, start_index=0, page=page)
    return SearchResult(chunk=chunk, score=0.8)


RESULTS = [
    result("retrieval_metrics.md", "MRR is the mean reciprocal rank."),
    result("physics.pdf", "F = ma.", page=12),
]


def test_context_numbers_sources_from_one_with_their_origin() -> None:
    context = format_context(RESULTS)

    assert context == (
        "[1] (source: retrieval_metrics.md)\nMRR is the mean reciprocal rank.\n\n"
        "[2] (source: physics.pdf, page 12)\nF = ma."
    )


def test_messages_are_system_then_user_with_question_after_sources() -> None:
    system, user = build_messages("What is MRR?", RESULTS)

    assert system == Message(role="system", content=SYSTEM_PROMPT)
    assert user.role == "user"
    assert user.content.index("[1] (source: retrieval_metrics.md)") < user.content.index(
        "Question: What is MRR?"
    )


def test_system_prompt_demands_grounding_citations_and_refusal() -> None:
    assert "only" in SYSTEM_PROMPT.lower()
    assert "[1]" in SYSTEM_PROMPT
    assert NO_ANSWER in SYSTEM_PROMPT
    # retrieved text is data, never instructions (prompt-injection defence)
    assert "instructions" in SYSTEM_PROMPT.lower()


def test_prompt_version_is_pinned() -> None:
    assert PROMPT_VERSION == "v1"
