"""Tests for playground.streaming: the Claude-like feel of a reply being written."""

import threading
from collections.abc import Iterator

import pytest

from rag_eval_platform.playground.streaming import (
    VERBS,
    StreamPump,
    fade_levels,
    fade_markdown_text,
    fade_style,
    reveal_count,
    snap_to_word,
    status_line,
    steps_html,
    verb,
)


def test_every_step_has_several_words_ending_without_dots() -> None:
    for step in ("embed", "search", "rerank", "read", "think", "write", "cite"):
        assert len(VERBS[step]) >= 3
        assert all(not word.endswith(("…", ".")) for word in VERBS[step])


def test_verb_rotates_with_time_and_wraps_around() -> None:
    words = VERBS["think"]

    assert verb("think", 0.0) == words[0]
    assert verb("think", 1.6) == words[1]
    assert verb("think", 1.5 * len(words)) == words[0]  # back to the start
    assert verb("think", 3.1, every_s=3.0) == words[1]  # slower in slow motion


def test_verb_rejects_an_unknown_step() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        verb("dance", 0.0)


def test_status_line_adds_detail_then_time_and_tokens_like_claude_code() -> None:
    assert status_line("search", 0.4, detail="19 chunks") == "Searching 19 chunks…"
    assert status_line("think", 3.2) == f"{verb('think', 3.2)}… (3s)"
    assert (
        status_line("write", 12.9, tokens=1234) == f"{verb('write', 12.9)}… (12s · ↓ 1,234 tokens)"
    )


def test_reveal_keeps_a_steady_minimum_pace() -> None:
    assert reveal_count(0, 100, 0.04, min_cps=50, catch_up_s=0.3) == 14  # backlog-driven
    assert reveal_count(0, 3, 0.04, min_cps=50, catch_up_s=0.3) == 2  # ceil(50 * 0.04)
    assert reveal_count(10, 10, 0.04, min_cps=50, catch_up_s=0.3) == 10  # nothing new


def test_reveal_catches_up_with_a_big_backlog_but_never_passes_it() -> None:
    assert reveal_count(0, 1000, 0.3, min_cps=50, catch_up_s=0.3) == 1000
    assert reveal_count(990, 1000, 0.3, min_cps=50, catch_up_s=0.3) == 1000


def test_slow_motion_reveals_at_a_fixed_pace_without_catching_up() -> None:
    assert reveal_count(0, 1000, 0.1, min_cps=20, catch_up_s=None) == 2


def test_pump_reads_the_stream_in_the_background() -> None:
    release = threading.Event()

    def slow_stream() -> Iterator[str]:
        yield "Hel"
        release.wait(timeout=5)
        yield "lo"

    pump = StreamPump(slow_stream())
    assert pump.wait_for_pieces(timeout=5) == ["Hel"]
    assert not pump.finished

    release.set()
    assert pump.wait_for_pieces(timeout=5) == ["lo"]
    pump.join(timeout=5)
    assert pump.finished
    assert pump.poll() == []


def test_pump_hands_a_stream_error_back_to_the_caller() -> None:
    def broken() -> Iterator[str]:
        yield "a"
        raise RuntimeError("model crashed")

    pump = StreamPump(broken())
    pump.join(timeout=5)

    assert pump.poll() == ["a"]
    with pytest.raises(RuntimeError, match="model crashed"):
        pump.raise_error()


def test_steps_html_draws_a_timeline_of_finished_steps_escaped() -> None:
    html = steps_html(["Searched 19 chunks → best match 0.22", "Read <5> sources"])

    assert html.startswith('<div class="rag-timeline">')
    assert html.count('<div class="rag-step">') == 2  # each gets a green dot (CSS)
    assert "Read &lt;5&gt; sources" in html
    assert steps_html([]) == ""


def test_fade_levels_keep_only_text_still_settling() -> None:
    reveals = [(0, 0.0), (10, 0.5), (20, 0.9)]

    levels = fade_levels(reveals, now=1.0, fade_s=0.9)

    assert [start for start, _ in levels] == [10, 20]  # the first piece has settled
    assert levels[0][1] == pytest.approx(0.5 / 0.9)
    assert levels[1][1] == pytest.approx(0.1 / 0.9)


def test_snap_to_word_never_shows_half_a_word() -> None:
    assert snap_to_word("hello world", 3) == 5
    assert snap_to_word("hello world", 5) == 5
    assert snap_to_word("hello world", 11) == 11
    assert snap_to_word("a" * 40, 3) == 3  # no word end nearby: keep the steady pace


def test_fade_style_goes_from_faint_and_blurred_to_sharp() -> None:
    assert fade_style(0.0) == "opacity:0.20;filter:blur(3.0px)"
    assert fade_style(0.5) == "opacity:0.60;filter:blur(1.5px)"
    assert fade_style(1.0) == ""


def test_fade_wraps_only_the_new_words() -> None:
    faded = fade_markdown_text("Hello brave world", 0, ((6, 0.0),))

    assert faded.startswith("Hello ")
    assert faded.count('<span style="opacity:0.20;filter:blur(3.0px)">') == 2
    assert fade_markdown_text("Hello", 0, ()) == "Hello"


@pytest.mark.parametrize(
    ("text", "kept"),
    [("- item one", "- "), ("## Title words", "## "), ("1. first step", "1. "),
     ("> quoted text", "> "), ("| a | b |", "| ")],
)  # fmt: skip
def test_fade_leaves_markdown_markers_alone(text: str, kept: str) -> None:
    faded = fade_markdown_text(text, 0, ((0, 0.5),))

    assert faded.startswith(kept)
    assert "<span" in faded


def test_fade_skips_inline_code_and_escapes_html() -> None:
    faded = fade_markdown_text("use `code here` now a<b", 0, ((0, 0.5),))

    assert "`code here`" in faded  # HTML inside backticks would show as text
    assert "&lt;" in faded
    assert faded.count("<span") == 3  # use, now, a&lt;b
