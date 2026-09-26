"""Tests for the optional-dependency import helper."""

import pytest

from rag_eval_platform._optional import OptionalDependencyError, import_optional


def test_returns_installed_module() -> None:
    assert import_optional("json", extra="none").dumps([1]) == "[1]"


def test_missing_module_raises_with_install_hint() -> None:
    with pytest.raises(OptionalDependencyError, match="uv sync --extra my-extra"):
        import_optional("definitely_not_installed_pkg", extra="my-extra")
