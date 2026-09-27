"""Tests for playground.query_visuals (pure data behind the per-question charts)."""

import pytest

from rag_eval_platform.evaluation.citation_validity import CitationValidity, CitationVerdict
from rag_eval_platform.evaluation.judge import JudgeScores
from rag_eval_platform.generation.generator import Answer, Citation
from rag_eval_platform.generation.prompt_templates import SYSTEM_PROMPT
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import JudgeResult, QueryTrace
from rag_eval_platform.playground.query_visuals import (
    INVALID_COLOR,
    AnswerSpan,
    RerankMove,
    SimilarityRow,
    answer_spans,
    funnel_stages,
    hit_vs_faithfulness,
    origin,
    prompt_blocks,
    rerank_moves,
    similarity_rows,
    source_color,
    spans_html,
    spans_markdown,
    text_spans,
    timing_rows,
    verdict_dot,
    was_reranked,
)
from rag_eval_platform.retrieval.vector_store import SearchResult


def hit(chunk_id: str, score: float, page: int | None = None) -> SearchResult:
    doc = chunk_id.split("#")[0]
    chunk = Chunk(
        id=chunk_id, doc_id=doc, index=0, text=f"text of {chunk_id}", start_index=0, page=page
    )
    return SearchResult(chunk, score)


def answer(text: str, sources: tuple[SearchResult, ...], cited: tuple[int, ...] = ()) -> Answer:
    return Answer(
        question="What is MRR?", text=text,
        citations=tuple(Citation(n, sources[n - 1]) for n in cited),
        invalid_citations=tuple(n for n in (9,) if f"[{n}]" in text), sources=sources,
        model="qwen3:8b", prompt_version="v1", input_tokens=10, output_tokens=5, latency_ms=4000.0,
    )  # fmt: skip


VECTOR = (hit("a#0", 0.9), hit("b#0", 0.7), hit("c#0", 0.5))


def test_similarity_rows_without_reranking() -> None:
    final = VECTOR[:2]
    trace = QueryTrace(VECTOR, final, answer("A [1].", final, cited=(1,)), 20.0)

    assert similarity_rows(trace) == (
        SimilarityRow(1, "a#0", 0.9, kept=True, cited=True),
        SimilarityRow(2, "b#0", 0.7, kept=True, cited=False),
        SimilarityRow(3, "c#0", 0.5, kept=False, cited=False),
    )
    assert was_reranked(trace) is False


def test_similarity_rows_and_moves_with_reranking() -> None:
    final = (hit("c#0", 5.1), hit("a#0", 2.0))  # cross-encoder promoted c
    trace = QueryTrace(VECTOR, final, answer("C [1].", final, cited=(1,)), 20.0)

    rows = similarity_rows(trace)
    assert [(r.chunk_id, r.kept, r.cited) for r in rows] == [
        ("a#0", True, False),
        ("b#0", False, False),
        ("c#0", True, True),
    ]
    assert was_reranked(trace) is True
    assert rerank_moves(trace) == (
        RerankMove("a#0", 1, 2),
        RerankMove("b#0", 2, None),
        RerankMove("c#0", 3, 1),
    )


def test_funnel_stages() -> None:
    final = VECTOR[:2]
    trace = QueryTrace(VECTOR, final, answer("A [1][2].", final, cited=(1, 2)), 20.0)

    assert funnel_stages(40, trace) == (
        ("all chunks", 40),
        ("candidates", 3),
        ("sent to the model", 2),
        ("cited", 2),
    )


def test_prompt_blocks_follow_the_prompt_order() -> None:
    final = (hit("a#0", 0.9, page=3), hit("b#0", 0.7))
    trace = QueryTrace(final, final, answer("A [1].", final), 1.0)

    blocks = prompt_blocks(trace)

    assert [b.kind for b in blocks] == ["rules", "source", "source", "question"]
    assert blocks[0].text == SYSTEM_PROMPT
    assert blocks[1].title == "[1] a, page 3"
    assert blocks[-1].text == "What is MRR?"
    assert origin(final[1]) == "b"


def test_answer_spans_split_text_and_citations() -> None:
    sources = VECTOR[:2]

    spans = answer_spans(answer("MRR ranks [1, 2]. Also [9].", sources))

    assert spans == (
        AnswerSpan("MRR ranks ", None, True),
        AnswerSpan("[1]", 1, True),
        AnswerSpan("[2]", 2, True),
        AnswerSpan(". Also ", None, True),
        AnswerSpan("[9]", 9, False),
        AnswerSpan(".", None, True),
    )


def test_answer_spans_without_citations() -> None:
    assert answer_spans(answer("No citations.", VECTOR)) == (
        AnswerSpan("No citations.", None, True),
    )


def test_spans_html_colours_citations_and_escapes_text() -> None:
    html = spans_html(
        (
            AnswerSpan("a < b & c\n", None, True),
            AnswerSpan("[1]", 1, True),
            AnswerSpan("[9]", 9, False),
        )
    )

    assert "a &lt; b &amp; c<br>" in html
    assert source_color(1) in html
    assert INVALID_COLOR in html
    assert "<script" not in spans_html((AnswerSpan("<script>x</script>", None, True),))


def test_spans_html_escapes_text() -> None:
    assert "&lt;b&gt;" in spans_html((AnswerSpan("<b>", None, True),))


def test_source_colors_cycle() -> None:
    assert source_color(1) != source_color(2)
    assert source_color(11) == source_color(1)


def _judged(verdicts: tuple[CitationVerdict, ...]) -> JudgeResult:
    scores = JudgeScores(faithfulness=1.0, answer_relevance=0.9)
    return JudgeResult(scores, CitationValidity(verdicts), 60000.0)


def test_timing_rows() -> None:
    trace = QueryTrace(VECTOR, VECTOR, answer("A.", VECTOR), 250.0)

    assert timing_rows(trace, None) == (("retrieve", 250.0), ("generate", 4000.0))
    assert timing_rows(trace, _judged(()))[-1] == ("judge", 60000.0)


def test_verdict_dot_links_sentences_to_sources() -> None:
    dot = verdict_dot(
        _judged(
            (
                CitationVerdict('He said "hi".', 1, "a.md", True),
                CitationVerdict("Other claim.", 2, "b.md", False),
            )
        )
    )

    assert 's0 -> src1 [color="#16a34a"' in dot
    assert 's1 -> src2 [color="#dc2626"' in dot
    assert 'He said \\"hi\\".' in dot


def test_verdict_dot_escapes_quotes() -> None:
    dot = verdict_dot(_judged((CitationVerdict('"q"', 1, 'doc"x', True),)))

    assert '\\"q\\"' in dot
    assert 'doc\\"x' in dot


def test_verdict_dot_without_citations() -> None:
    assert "No citations to check" in verdict_dot(_judged(()))


def test_hit_vs_faithfulness_joins_the_two_reports() -> None:
    retrieval = {
        "examples": [
            {"example_id": "q1", "scores": {"recall": 1.0}},
            {"example_id": "q2", "scores": {"recall": 0.5}},
            {"example_id": "q3", "scores": {"recall": 1.0}},
        ]
    }
    generation = {
        "examples": [
            {"example_id": "q1", "scores": {"faithfulness": 1.0}},
            {"example_id": "q2", "scores": {"faithfulness": 0.5}},
            {"example_id": "q3", "scores": {"faithfulness": 0.8}},
        ]
    }

    (hit_label, hit_mean, hit_n), (miss_label, miss_mean, miss_n) = hit_vs_faithfulness(
        retrieval, generation
    )
    assert (hit_label, hit_n) == ("found every relevant doc", 2)
    assert hit_mean == pytest.approx(0.9)
    assert (miss_label, miss_n) == ("missed a relevant doc", 1)
    assert miss_mean == pytest.approx(0.5)


def test_text_spans_work_on_a_partial_streamed_answer() -> None:
    spans = text_spans("Recall [1] and [3", sources=2)

    assert spans == (
        AnswerSpan("Recall ", None, True),
        AnswerSpan("[1]", 1, True),
        AnswerSpan(" and [3", None, True),  # not a citation until the bracket closes
    )


def test_spans_markdown_keeps_markdown_and_chips_but_disarms_html() -> None:
    text = "## Key ideas\n\n| a | b |\n|---|---|\n| **x** [1] | y [9] |\n\n<script>x</script> & > q"

    rendered = spans_markdown(text_spans(text, sources=2))

    assert rendered.startswith("## Key ideas\n\n| a | b |\n|---|---|\n| **x** ")
    assert f"background:{source_color(1)}" in rendered
    assert "line-through" in rendered  # [9] is not a real source
    assert "<script>" not in rendered
    assert "&lt;script>x&lt;/script> &amp; > q" in rendered  # inert; ">" kept for quotes
