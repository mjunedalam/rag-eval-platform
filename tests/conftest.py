"""Shared pytest configuration.

Tests marked ``evaluation`` call a real LLM judge and take minutes, so they only run when
selected explicitly: ``uv run pytest -m evaluation``.
"""

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if "evaluation" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="slow LLM-judge evaluation; run with: pytest -m evaluation")
    for item in items:
        if item.get_closest_marker("evaluation"):
            item.add_marker(skip)
