"""Import packages that belong to an optional extra, with an install hint on failure."""

from importlib import import_module
from typing import Any


class OptionalDependencyError(ImportError):
    """A package from an optional extra (see pyproject.toml) is not installed."""


def import_optional(module_name: str, extra: str) -> Any:
    """Import ``module_name`` or raise with the ``uv sync --extra`` command that installs it."""
    try:
        return import_module(module_name)
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            f"The '{module_name}' package is not installed. Run: uv sync --extra {extra}"
        ) from exc
