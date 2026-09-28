"""Tests for playground.answer_metrics: instant answer health, no judge needed."""

import pytest

from rag_eval_platform.generation.generator import Answer, Citation
from rag_eval_platform.generation.prompt_templates import NO_ANSWER
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.answer_metrics import (
    AnswerHealth,
    answer_health,
    band,
    claims,
    health_html,
    health_rows,
)
from rag_eval_platform.playground.core import QueryTrace
from rag_eval_platform.retrieval.vector_store import SearchResult

SOURCES = (
    SearchResult(Chunk(id="metrics.md#0", doc_id="metrics.md", index=0, start_index=0,
                       text="Mean reciprocal rank averages the inverse rank of the first "
                            "relevant document."), 0.8),
    SearchResult(Chunk(id="book.pdf#3", doc_id="book.pdf", index=3, start_index=0, page=4,
                       text="Recall counts how many relevant documents were retrieved."), 0.6),
    SearchResult(Chunk(id="book.pdf#5", doc_id="book.pdf", index=5, start_index=0, page=4,
                       text="Unrelated text about cooking pasta."), 0.4),
)  # fmt: skip


def trace(text: str, *, input_tokens: int | None = 120, latency_ms: float = 3000.0) -> QueryTrace:
    cited = sorted({int(n) for n in __import__("re").findall(r"\[(\d)\]", text)} & {1, 2, 3})
    answer = Answer(
        question="What is MRR?", text=text,
        citations=tuple(Citation(n, SOURCES[n - 1]) for n in cited), invalid_citations=(),
        sources=SOURCES, model="qwen3:8b", prompt_version="v1-detailed",
        input_tokens=input_tokens, output_tokens=40, latency_ms=latency_ms,
    )  # fmt: skip
    return QueryTrace(SOURCES, SOURCES, answer, 500.0)


def test_claims_skip_headings_table_rules_and_fragments() -> None:
    text = (
        "## Key ideas\n\n- MRR averages inverse ranks [1].\n|---|---|\n| a | b |\nOk.\n"
        "Recall counts documents [2]."
    )

    assert claims(text) == ("- MRR averages inverse ranks [1].", "Recall counts documents [2].")


def test_coverage_is_the_share_of_claims_with_a_citation() -> None:
    health = answer_health(trace("MRR averages inverse ranks [1]. Recall is useful for search."),
                           first_token_s=1.0)  # fmt: skip

    assert health.citation_coverage == 0.5


def test_grounding_measures_word_overlap_with_the_cited_source() -> None:
    supported = answer_health(trace("Mean reciprocal rank averages the inverse rank [1]."), None)
    invented = answer_health(trace("Pasta needs salted boiling water always [1]."), None)

    assert supported.grounding == pytest.approx(1.0)
    assert invented.grounding == pytest.approx(0.0)


def test_sources_context_and_time_breakdown() -> None:
    health = answer_health(trace("MRR averages ranks [1]. Recall counts [2]."), first_token_s=1.2)

    assert (health.distinct_docs, health.distinct_pages) == (2, 2)  # metrics.md, book.pdf p.4
    assert health.context_chars == sum(len(s.chunk.text) for s in SOURCES)
    assert health.prompt_tokens == 120
    assert (health.retrieve_s, health.first_token_s) == (0.5, 1.2)
    assert health.write_s == pytest.approx(1.8)  # 3.0 s generating - 1.2 s waiting
    assert health.answer_words == 5  # citation markers are not words


def test_a_refusal_has_no_coverage_or_grounding() -> None:
    health = answer_health(trace(NO_ANSWER), first_token_s=None)

    assert (health.citation_coverage, health.grounding, health.write_s) == (None, None, None)


def test_bands_colour_scores() -> None:
    assert band(0.9) == "good"
    assert band(0.6) == "fair"
    assert band(0.2) == "poor"
    assert band(None) == "none"


def test_rows_and_html_show_every_metric_escaped() -> None:
    health = AnswerHealth(
        citation_coverage=0.75, grounding=0.5, distinct_docs=2, distinct_pages=3,
        context_chars=2400, prompt_tokens=None, answer_words=180, retrieve_s=0.4,
        first_token_s=2.0, write_s=6.5,
    )  # fmt: skip

    rows = health_rows(health)
    html = health_html(rows)

    assert [r.label for r in rows] == ["coverage", "grounding", "sources", "context",
                                       "answer", "time"]  # fmt: skip
    assert rows[0].value == "75%"
    assert rows[1].band == "fair"
    assert (rows[2].value, rows[2].detail) == ("3 pages", "from 2 documents")
    assert rows[3].value == "2,400 chars"
    assert rows[5].value == "8.9 s"
    assert rows[5].detail == "0.4 s search + 2.0 s first token + 6.5 s writing"
    assert 'title="0.4 s search' in html  # the breakdown shows on hover
    assert html.count('class="rag-health-card') == 6
    assert "<script" not in health_html(rows[:1]).lower()
