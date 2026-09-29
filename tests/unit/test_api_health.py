"""Tests for api.health: checks that never raise."""

import json

from rag_eval_platform.api.health import chroma_check, llm_check
from rag_eval_platform.retrieval.vector_store import VectorStoreError


class Store:
    def __init__(self, count: int | None) -> None:
        self._count = count

    def count(self) -> int:
        if self._count is None:
            raise VectorStoreError("connection refused")
        return self._count


def test_chroma_check_reports_the_chunk_count_or_the_error() -> None:
    assert chroma_check(Store(40))().model_dump() == {"ok": True, "detail": "40 chunks"}
    down = chroma_check(Store(None))()
    assert not down.ok
    assert "connection refused" in down.detail


def models(*names: str) -> bytes:
    return json.dumps({"data": [{"id": n} for n in names]}).encode()


def test_llm_check_needs_the_configured_model() -> None:
    ok = llm_check("http://x/v1", "qwen3:8b", fetch=lambda url: models("qwen3:8b"))()
    missing = llm_check("http://x/v1", "qwen3:8b", fetch=lambda url: models("other"))()

    assert ok.ok
    assert not missing.ok
    assert "ollama pull qwen3:8b" in missing.detail


def test_llm_check_reports_an_unreachable_server() -> None:
    def refuse(url: str) -> bytes:
        raise OSError("connection refused")

    check = llm_check("http://x/v1", "qwen3:8b", fetch=refuse)()

    assert not check.ok
    assert "unreachable" in check.detail
