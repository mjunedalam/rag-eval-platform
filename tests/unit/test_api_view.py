"""Tests for playground.api_view: the ⑦ API tab's pure pieces."""

import io
import json
import urllib.error
from typing import Any

from rag_eval_platform.playground.api_view import (
    SseEvent,
    api_flow_dot,
    curl_command,
    get_json,
    parse_sse,
    post_json,
    stream_events,
)


def test_curl_command_quotes_the_body_and_masks_the_key() -> None:
    body = {"question": "What's MRR?", "answer_style": "concise"}

    shown = curl_command("http://127.0.0.1:8000", body, api_key="s3cret-long-key")
    real = curl_command("http://127.0.0.1:8000", body, api_key="s3cret-long-key", reveal_key=True)

    assert shown.startswith("curl -s -X POST http://127.0.0.1:8000/query")
    assert "s3cret-long-key" not in shown
    assert "X-API-Key: s3cr…" in shown
    assert "X-API-Key: s3cret-long-key" in real
    assert "'\"'\"'" in shown  # the apostrophe in What's is shell-quoted
    assert "/query/stream" in curl_command("http://x", body, api_key=None, stream=True)
    assert "X-API-Key" not in curl_command("http://x", body, api_key=None)


def test_parse_sse_joins_data_lines_per_event() -> None:
    lines = ["event: status\n", "data: retrieving\n", "\n", "event: token\n", "data: a\n",
             "data: b\n", "\n"]  # fmt: skip

    assert list(parse_sse(lines)) == [SseEvent("status", "retrieving"), SseEvent("token", "a\nb")]


class Response(io.BytesIO):
    def __init__(self, body: bytes, status: int = 200) -> None:
        super().__init__(body)
        self.status = status

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_post_json_returns_status_body_and_time() -> None:
    sent: list[Any] = []

    def opener(request: Any, timeout: float) -> Response:
        sent.append(request)
        return Response(b'{"answer": "x"}')

    result = post_json("http://x/query", {"question": "q"}, "k", opener=opener)

    assert (result.status, json.loads(result.body)) == (200, {"answer": "x"})
    assert result.ms >= 0
    assert sent[0].get_header("X-api-key") == "k"


def test_http_errors_keep_their_status_and_body() -> None:
    def opener(request: Any, timeout: float) -> Response:
        raise urllib.error.HTTPError(
            "http://x",
            429,
            "Too Many",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error": "slow down"}'),
        )

    result = post_json("http://x/query", {"question": "q"}, None, opener=opener)

    assert result.status == 429
    assert "slow down" in result.body


def test_an_unreachable_api_is_status_0() -> None:
    def opener(request: Any, timeout: float) -> Response:
        raise urllib.error.URLError("connection refused")

    assert get_json("http://x/health", opener=opener).status == 0


def test_stream_events_reads_the_event_stream() -> None:
    def opener(request: Any, timeout: float) -> Response:
        return Response(b"event: token\ndata: Hi\n\nevent: done\ndata: {}\n\n")

    events = list(stream_events("http://x/query/stream", {"question": "q"}, None, opener=opener))

    assert events == [SseEvent("token", "Hi"), SseEvent("done", "{}")]


def test_flow_diagram_shows_each_stage_time_after_a_request() -> None:
    assert "ms" not in api_flow_dot(None)
    dot = api_flow_dot({"retrieve_ms": 38.0, "generate_ms": 4200.0, "total_ms": 4300.0})
    assert "38 ms" in dot
    assert "4.2 s" in dot


def test_only_http_urls_are_allowed() -> None:
    import pytest

    with pytest.raises(ValueError, match="http"):
        post_json("file:///etc/passwd", {"question": "q"}, None)


def test_short_keys_are_fully_masked() -> None:
    shown = curl_command("http://x", {"question": "q"}, api_key="abc")

    assert "abc" not in shown
    assert "X-API-Key: ••••" in shown


def test_stream_http_errors_become_events_with_their_status() -> None:
    def opener(request: Any, timeout: float) -> Response:
        raise urllib.error.HTTPError(
            "http://x",
            429,
            "Too Many",
            {},  # type: ignore[arg-type]
            io.BytesIO(b'{"error": "slow down"}'),
        )

    events = list(stream_events("http://x/query/stream", {"question": "q"}, None, opener=opener))

    assert events == [SseEvent("status_code", "429"), SseEvent("error", '{"error": "slow down"}')]


def test_stream_connection_failures_become_status_0() -> None:
    def opener(request: Any, timeout: float) -> Response:
        raise urllib.error.URLError("connection refused")

    events = list(stream_events("http://x/query/stream", {"question": "q"}, None, opener=opener))

    assert events[0] == SseEvent("status_code", "0")
    assert events[1].event == "error"
