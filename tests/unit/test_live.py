"""Tests for playground.live: status wording, smooth typing and the live stats tiles."""

import pytest
from tests.unit.test_query_visuals import answer

from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.playground.core import QueryTrace
from rag_eval_platform.playground.live import (
    LiveStats,
    pace,
    split_for_typing,
    stat_tiles,
    stats_from_trace,
    status_html,
    status_text,
    tail_for_display,
    tiles_html,
    typing_html,
)
from rag_eval_platform.retrieval.vector_store import SearchResult


def test_status_text_describes_each_step() -> None:
    assert status_text("embed") == "Understanding your question…"
    assert status_text("search", chunks=1505) == "Searching 1,505 chunks…"
    assert status_text("rerank", candidates=20) == "Re-ranking 20 candidates…"
    assert status_text("read", sources=5) == "Reading 5 sources…"
    assert status_text("think") == "Thinking…"
    assert status_text("write") == "Writing the answer…"
    assert status_text("cite") == "Checking citations…"
    assert status_text("done", seconds=6.25) == "Done in 6.2 s"


def test_status_text_rejects_unknown_steps() -> None:
    with pytest.raises(ValueError, match="unknown step"):
        status_text("deploy")


def test_split_for_typing_keeps_every_character() -> None:
    assert split_for_typing("Hello world", size=4) == ["Hell", "o wo", "rld"]
    assert "".join(split_for_typing("MRR averages [1].", size=3)) == "MRR averages [1]."
    assert split_for_typing("", size=4) == []
    with pytest.raises(ValueError, match="size"):
        split_for_typing("x", size=0)


def test_stat_tiles_show_placeholders_until_known() -> None:
    tiles = dict(stat_tiles(LiveStats(chunks=1505)))

    assert tiles["chunks searched"] == "1,505"
    assert tiles["candidates"] == "…"
    assert tiles["speed"] == "…"


def test_stat_tiles_when_complete() -> None:
    stats = LiveStats(
        chunks=1505, candidates=20, top_similarity=0.624, sources=5, tokens=70,
        generate_s=25.0, first_token_s=20.0, cited=2, elapsed_s=26.3,
    )  # fmt: skip

    assert stats.tokens_per_s == pytest.approx(14.0)  # 70 tokens over 5 s of writing
    assert dict(stat_tiles(stats)) == {
        "chunks searched": "1,505",
        "candidates": "20",
        "top similarity": "0.62",
        "sources sent": "5",
        "first token": "20.0 s",
        "tokens": "70",
        "speed": "14 tok/s",
        "cited": "2/5",
        "elapsed": "26.3 s",
    }


def test_speed_falls_back_to_total_time_without_first_token() -> None:
    assert LiveStats(tokens=70, generate_s=5.0).tokens_per_s == pytest.approx(14.0)


def test_tail_for_display_keeps_the_newest_text_whole_words() -> None:
    text = "one two three four five six"

    assert tail_for_display(text, limit=100) == text
    tail = tail_for_display(text, limit=12)
    assert tail.startswith("…")
    assert text.endswith(tail[1:])
    assert tail == "…five six"  # starts at a word, not mid-word
    with pytest.raises(ValueError, match="limit"):
        tail_for_display(text, limit=0)


def test_slow_motion_stretches_every_timing() -> None:
    normal, slow = pace(slow=False), pace(slow=True)

    assert slow.step_pause > normal.step_pause
    assert slow.typing_pause > normal.typing_pause
    assert slow.typing_size is not None
    assert slow.frames > normal.frames
    assert slow.frame_seconds > normal.frame_seconds


def test_normal_pace_writes_at_the_model_speed() -> None:
    normal = pace(slow=False)

    assert normal.typing_pause == 0
    assert normal.typing_size is None


def test_split_for_typing_without_a_size_keeps_the_piece_whole() -> None:
    assert split_for_typing("Hello world", size=None) == ["Hello world"]
    assert split_for_typing("", size=None) == []


def test_tokens_per_second_needs_time() -> None:
    assert LiveStats(tokens=10, generate_s=0.0).tokens_per_s is None
    assert LiveStats(tokens=None, generate_s=2.0).tokens_per_s is None


def _hit(chunk_id: str, score: float) -> SearchResult:
    chunk = Chunk(id=chunk_id, doc_id="d", index=0, text="t", start_index=0)
    return SearchResult(chunk, score)


def test_stats_from_a_finished_trace() -> None:
    vector = (_hit("a#0", 0.62), _hit("b#0", 0.5), _hit("c#0", 0.4))
    final = vector[:2]
    trace = QueryTrace(vector, final, answer("A [1].", final, cited=(1,)), 500.0)

    stats = stats_from_trace(trace, chunks=40)

    assert stats == LiveStats(
        chunks=40, candidates=3, top_similarity=0.62, sources=2, tokens=5,
        generate_s=4.0, cited=1, elapsed_s=4.5,
    )  # fmt: skip
    assert stats_from_trace(trace, chunks=40, first_token_s=3.0).first_token_s == 3.0


def test_html_helpers_escape_text() -> None:
    assert "&lt;b&gt;" in status_html("<b>")
    assert "rag-status" in status_html("Thinking…")
    typed = typing_html("a < b")
    assert "a &lt; b" in typed
    assert "rag-cursor" in typed
    assert "rag-cursor" not in typing_html("done", cursor=False)
    tiles = tiles_html((("tokens", "<70>"),))
    assert "&lt;70&gt;" in tiles
    assert "tokens" in tiles


def test_running_status_has_a_spinning_spark() -> None:
    assert "rag-spark" in status_html("Thinking…")
    assert "rag-spark" not in status_html("Done", done=True)
