"""Tests for playground.chat: steps, suggestions, conversation history and source cards."""

from tests.unit.test_query_visuals import VECTOR, answer, hit

from rag_eval_platform.generation.prompt_templates import NO_ANSWER
from rag_eval_platform.playground.chat import (
    SourceCard,
    Step,
    Turn,
    add_turn,
    cited_sources_html,
    replace_turn,
    run_steps,
    source_cards,
    steps_summary,
    streaming_markdown,
    suggestions,
    user_html,
    with_feedback,
)
from rag_eval_platform.playground.core import QueryTrace
from rag_eval_platform.playground.query_visuals import source_color


def trace_with(text: str, final: tuple = VECTOR[:2], cited: tuple[int, ...] = (1,)) -> QueryTrace:  # type: ignore[type-arg]
    return QueryTrace(VECTOR, final, answer(text, final, cited=cited), 500.0)


def test_run_steps_without_reranking() -> None:
    steps = run_steps(trace_with("A [1]."), chunks=1505, first_token_s=1.0)

    assert steps == (
        Step("Searched your documents", "1,505 chunks → 2 sources", 500.0),
        Step("Wrote the answer", "5 tokens · 2 tok/s", 4000.0),
        Step("Checked citations", "1 of 2 sources cited", None),
    )


def test_run_steps_with_reranking_and_invalid_citations() -> None:
    reranked = (hit("c#0", 5.0), hit("a#0", 2.0))
    steps = run_steps(trace_with("C [1]. X [9].", final=reranked), chunks=40, first_token_s=None)

    labels = [s.label for s in steps]
    assert labels == [
        "Searched your documents", "Re-ranked the candidates", "Wrote the answer",
        "Checked citations",
    ]  # fmt: skip
    assert steps[1].detail == "3 candidates → top 2"
    assert steps[-1].detail == "1 of 2 sources cited · 1 invalid"


def test_run_steps_for_a_refusal() -> None:
    steps = run_steps(trace_with(NO_ANSWER, cited=()), chunks=40, first_token_s=None)

    assert steps[-1] == Step("Found no answer in the sources", "the model refused to guess", None)


def test_steps_summary_counts_steps_and_time() -> None:
    steps = (Step("a", "", 500.0), Step("b", "", 4000.0), Step("c", "", None))

    assert steps_summary(steps) == "3 steps · 4.5 s"


def test_suggestions_come_from_document_names() -> None:
    picks = suggestions(("Math-For-ML.pdf", "notes/rag_overview.md"))

    assert picks == (
        "Summarise Math-For-ML",
        "What are the key ideas in rag_overview?",
        "Explain an important concept from Math-For-ML",
    )
    assert suggestions(()) == ()
    assert len(suggestions(("a.md",), limit=2)) == 2


def _turn(text: str) -> Turn:
    return Turn(trace_with(text), first_token_s=0.5)


def test_history_operations_return_new_tuples() -> None:
    first, second = _turn("A [1]."), _turn("B [1].")

    history = add_turn(add_turn((), first), second)
    replaced = replace_turn(history, 0, _turn("C [1]."))
    rated = with_feedback(history, 1, 1)

    assert history == (first, second)
    assert replaced[0].trace.answer.text == "C [1]."
    assert replaced[1] is second
    assert rated[1].feedback == 1
    assert history[1].feedback is None  # the original is untouched


def test_source_cards_mark_cited_sources_with_their_colour() -> None:
    trace = trace_with("A [1].", final=(hit("a#0", 0.9, page=3), hit("b#0", 0.7)))

    cards = source_cards(trace)

    assert cards == (
        SourceCard(1, "a, page 3", source_color(1), True, "text of a#0"),
        SourceCard(2, "b", source_color(2), False, "text of b#0"),
    )


def test_user_html_is_an_escaped_bubble() -> None:
    bubble = user_html("Is <b> safe?")

    assert 'class="rag-user"' in bubble
    assert "Is &lt;b&gt; safe?" in bubble


def test_cited_sources_html_lists_only_cited_sources_escaped() -> None:
    trace = trace_with(
        "A [2]. B [1][2].",
        final=(hit("a<b>#0", 0.9, page=3), hit("c#0", 0.7), hit("d#0", 0.5)),
        cited=(2, 1),
    )

    listed = cited_sources_html(trace)

    assert listed.index("[1]") < listed.index("[2]")
    assert "a&lt;b&gt;, page 3" in listed
    assert listed.count('class="rag-source"') == 2  # [3] was never cited
    assert cited_sources_html(trace_with("No citations.", cited=())) == ""


def test_streaming_markdown_renders_chips_and_ends_with_the_cursor() -> None:
    rendered = streaming_markdown("**MRR** [1] and <b>", sources=2)

    assert rendered.startswith("**MRR** <span")
    assert "&lt;b>" in rendered
    assert rendered.endswith('<span class="rag-cursor">●</span>')
