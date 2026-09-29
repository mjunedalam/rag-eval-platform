"""⑦ API: the same pipeline over HTTP — status, a request builder, curl, and live streaming."""

import json
import time

import streamlit as st

from rag_eval_platform.playground.api_view import (
    API_URL,
    api_flow_dot,
    curl_command,
    get_json,
    post_json,
    stream_events,
)
from rag_eval_platform.playground.header import Pill, header_html
from rag_eval_platform.playground.shared import TabContext, settings

START_HINT = (
    "Start it with `uv run python scripts/serve_api.py`, or in Docker: "
    "`docker compose -f docker/docker-compose.yml --profile api up -d --build`."
)
API_LAST = "api_last"  # (status, body, ms) of the last request, kept across reruns


def render(ctx: TabContext) -> None:
    health = get_json(f"{API_URL}/health")
    up = health.status in (200, 503)
    label = f"API up at {API_URL}" if up else "API not running"
    st.markdown(header_html((Pill(label, ok=health.status == 200),), ()), unsafe_allow_html=True)
    if not up:
        st.info(START_HINT)
    last = st.session_state.get(API_LAST)
    timings = json.loads(last[1]).get("timings") if last and last[0] == 200 else None
    st.graphviz_chart(api_flow_dot(timings), width="stretch")
    left, right = st.columns([1, 1], gap="large")
    with left:
        _builder(up)
    with right:
        _response()


def _builder(up: bool) -> None:
    question = st.text_area("Question", "What does MRR measure?", key="api-question")
    style = st.segmented_control("Answer style", ["concise", "detailed"], default="concise",
                                 key="api-style", required=True)  # fmt: skip
    stream = st.toggle("Stream (/query/stream)", key="api-stream")
    key = settings.api_key.get_secret_value() if settings.api_key else None
    body = {"question": question, "answer_style": style}
    st.code(curl_command(API_URL, body, api_key=key, stream=stream), language="bash",
            wrap_lines=True)  # fmt: skip
    st.caption(f"Interactive docs: {API_URL}/docs")
    if st.button("Send", type="primary", disabled=not up, key="api-send"):
        if stream:
            _stream(body, key)
        else:
            result = post_json(f"{API_URL}/query", body, key)
            st.session_state[API_LAST] = (result.status, result.body, result.ms)
            st.rerun()


def _stream(body: dict[str, object], key: str | None) -> None:
    status, text = st.empty(), st.empty()
    written, code, started = "", 200, time.perf_counter()
    for events, event in enumerate(stream_events(f"{API_URL}/query/stream", body, key), 1):
        if event.event == "status_code":  # the request failed before streaming (401, 429...)
            code = int(event.data)
            continue
        if event.event == "status":
            status.caption(f"event: status · {event.data}")
        elif event.event == "token":
            written += event.data
            text.markdown(written)
            status.caption(f"event: token · {events} events so far")
        elif event.event in ("done", "error"):
            # Saved before the next st.* call, so a click cannot lose it. An error event on a
            # 200 stream is a failure after streaming began: shown as "stream error".
            ms = (time.perf_counter() - started) * 1000
            st.session_state[API_LAST] = (code, event.data, ms)
            status.caption(f"event: {event.event}")
    st.rerun()


def _response() -> None:
    last = st.session_state.get(API_LAST)
    if last is None:
        st.info("Send a request to see the JSON response.")
        return
    status, body, ms = last
    failed_midway = status == 200 and '"error"' in body and '"answer"' not in body
    label = "stream error" if failed_midway else "unreachable" if status == 0 else str(status)
    colour = "green" if status == 200 and not failed_midway else "red"
    timing = f" · {ms:.0f} ms" if ms else ""
    st.markdown(f"**Response** :{colour}[{label}]{timing}")
    try:
        st.json(json.loads(body), expanded=2)
    except ValueError:
        st.code(body)
