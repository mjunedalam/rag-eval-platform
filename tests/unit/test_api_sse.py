"""Tests for api.sse: server-sent event formatting."""

from rag_eval_platform.api.sse import sse_event


def test_one_event_per_block_and_one_data_line_per_text_line() -> None:
    assert sse_event("token", "Hello") == "event: token\ndata: Hello\n\n"
    assert sse_event("token", "a\nb") == "event: token\ndata: a\ndata: b\n\n"
