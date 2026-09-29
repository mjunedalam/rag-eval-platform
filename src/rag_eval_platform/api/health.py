"""Health checks for /health: each returns a CheckOut and never raises."""

import json
import urllib.request
from collections.abc import Callable
from typing import Protocol

from rag_eval_platform.api.schemas import CheckOut
from rag_eval_platform.retrieval.vector_store import VectorStoreError

HealthCheck = Callable[[], CheckOut]


class _Countable(Protocol):
    def count(self) -> int: ...


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=3) as response:  # noqa: S310 - configured URL
        data: bytes = response.read()
        return data


def chroma_check(store: _Countable) -> HealthCheck:
    def check() -> CheckOut:
        try:
            return CheckOut(ok=True, detail=f"{store.count():,} chunks")
        except VectorStoreError as exc:
            return CheckOut(ok=False, detail=f"Chroma unavailable: {exc}")

    return check


def llm_check(base_url: str, model: str, fetch: Callable[[str], bytes] = _fetch) -> HealthCheck:
    """The OpenAI-compatible server lists the configured model (Ollama or OpenAI)."""

    def check() -> CheckOut:
        try:
            listed = {m["id"] for m in json.loads(fetch(base_url.rstrip("/") + "/models"))["data"]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return CheckOut(ok=False, detail=f"LLM server unreachable at {base_url}: {exc}")
        if model not in listed:
            return CheckOut(ok=False, detail=f"model {model} not found; run: ollama pull {model}")
        return CheckOut(ok=True, detail=model)

    return check
