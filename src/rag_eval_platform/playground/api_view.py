"""Pure pieces of the ⑦ API tab: the curl command, HTTP calls and the flow diagram.

Plain urllib (no new dependency); ``opener`` is injectable so tests need no server.
"""

import json
import shlex
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

API_URL = "http://127.0.0.1:8000"
Opener = Callable[..., Any]


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: str


@dataclass(frozen=True)
class HttpResult:
    status: int  # 0 when the API could not be reached
    body: str
    ms: float


def curl_command(
    base_url: str, body: Mapping[str, object], *, api_key: str | None, stream: bool = False,
    reveal_key: bool = False,
) -> str:  # fmt: skip
    """The same request as a curl command; the key is masked unless ``reveal_key``."""
    path = "/query/stream" if stream else "/query"
    parts = ["curl", "-s", "-N"] if stream else ["curl", "-s"]
    parts += ["-X", "POST", f"{base_url}{path}", "-H", "Content-Type: application/json"]
    if api_key:
        # A short key would be given away by any prefix, so it is masked completely.
        shown = api_key if reveal_key else api_key[:4] + "…" if len(api_key) >= 12 else "••••"
        parts += ["-H", f"X-API-Key: {shown}"]
    parts += ["-d", json.dumps(dict(body))]
    return " ".join(shlex.quote(p) if i else p for i, p in enumerate(parts))


def parse_sse(lines: Iterable[str]) -> Iterator[SseEvent]:
    event = "message"
    data: list[str] = []
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            if data:
                yield SseEvent(event, "\n".join(data))
            event, data = "message", []  # a blank line ends one event
        elif line.startswith("event: "):
            event = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data.append(line.removeprefix("data: "))
    if data:
        yield SseEvent(event, "\n".join(data))


def _request(url: str, body: Mapping[str, object] | None, api_key: str | None) -> Any:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not an http(s) URL: {url}")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    data = json.dumps(dict(body)).encode() if body is not None else None
    method = "POST" if body is not None else "GET"
    return urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310 - http(s) only, checked above


def _call(request: Any, timeout: float, opener: Opener) -> HttpResult:
    started = time.perf_counter()
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read().decode()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        body, status = exc.read().decode(), exc.code
    except (urllib.error.URLError, OSError) as exc:
        body, status = str(exc), 0
    return HttpResult(status, body, (time.perf_counter() - started) * 1000)


def post_json(url: str, body: Mapping[str, object], api_key: str | None, timeout: float = 120,
              opener: Opener = urllib.request.urlopen) -> HttpResult:  # fmt: skip
    return _call(_request(url, body, api_key), timeout, opener)


def get_json(url: str, timeout: float = 3, opener: Opener = urllib.request.urlopen) -> HttpResult:
    return _call(_request(url, None, None), timeout, opener)


def stream_events(url: str, body: Mapping[str, object], api_key: str | None,
                  opener: Opener = urllib.request.urlopen) -> Iterator[SseEvent]:  # fmt: skip
    """The events of /query/stream as they arrive (the response is read line by line).

    A failed request becomes a ``status_code`` event (0 when unreachable), then ``error``.
    """
    try:
        with opener(_request(url, body, api_key), timeout=300) as response:
            yield from parse_sse(line.decode() for line in response)
    except urllib.error.HTTPError as exc:
        yield SseEvent("status_code", str(exc.code))
        yield SseEvent("error", exc.read().decode())
    except (urllib.error.URLError, OSError) as exc:
        yield SseEvent("status_code", "0")
        yield SseEvent("error", str(exc))


def api_flow_dot(timings: Mapping[str, float] | None) -> str:
    """Client → FastAPI (auth, rate limit) → retrieve → generate → JSON, with times if known."""

    def took(ms: float) -> str:
        return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"

    retrieve = f"\\n{took(timings['retrieve_ms'])}" if timings else ""
    generate = f"\\n{took(timings['generate_ms'])}" if timings else ""
    total = f"total {took(timings['total_ms'])}" if timings else ""
    return "\n".join([
        "digraph api {", '  rankdir=LR; bgcolor="transparent";',
        '  node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 '
        'fillcolor="#eef2ff" color="#6366f1"];',
        '  edge [color="#9ca3af" fontname="Helvetica" fontsize=9 fontcolor="#6b7280"];',
        '  client [label="Client\\ncurl · app"];',
        '  api [label="FastAPI\\nkey · rate limit"];',
        f'  retrieve [label="Retrieve{retrieve}"];',
        f'  generate [label="Generate{generate}"];',
        '  json [label="JSON\\ncited answer" fillcolor="#dcfce7" color="#16a34a"];',
        f'  client -> api [label="POST /query"]; api -> retrieve; retrieve -> generate;'
        f' generate -> json [label="{total}"];',
        "}",
    ])  # fmt: skip
