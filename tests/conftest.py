"""Shared pytest configuration.

Tests marked ``evaluation`` call a real LLM judge and take minutes, so they only run when
selected explicitly: ``uv run pytest -m evaluation``.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from rag_eval_platform.config.settings import get_settings


@pytest.fixture(autouse=True)
def _isolated_observe_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test logs to its own temporary database, never data/observability/.

    ``RAG_OBSERVE_READONLY_DBS`` is emptied too, so a real ``data/observability/rag-api.db``
    left over from running the Docker API locally can never leak extra rows into a test that
    builds its ``LogSet`` from ``get_settings()``.
    """
    monkeypatch.setenv("RAG_OBSERVE_DB", str(tmp_path / "observe.db"))
    monkeypatch.setenv("RAG_OBSERVE_READONLY_DBS", "[]")
    # The developer's .env may hold real Langfuse keys: no test may send traces to Langfuse
    # Cloud unless it switches this back on itself (tests/integration/test_langfuse_cloud.py).
    monkeypatch.setenv("RAG_LANGFUSE_ENABLED", "false")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if "evaluation" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="slow LLM-judge evaluation; run with: pytest -m evaluation")
    for item in items:
        if item.get_closest_marker("evaluation"):
            item.add_marker(skip)
