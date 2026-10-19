"""Whether Langfuse tracing and TruLens scoring are working, for the playground header.

"off" means switched off on purpose (no keys, or a setting); "problem" means it should work
but cannot, and ``detail`` says how to fix it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal

from rag_eval_platform.config.settings import Settings

State = Literal["on", "off", "problem"]


@dataclass(frozen=True)
class ToolStatus:
    label: str
    state: State
    detail: str


def _installed(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except ModuleNotFoundError:  # a missing parent package
        return False


def langfuse_status(settings: Settings, *,
                    installed: Callable[[str], bool] = _installed) -> ToolStatus:  # fmt: skip
    if not settings.langfuse_enabled:
        return ToolStatus("Langfuse off", "off", "RAG_LANGFUSE_ENABLED=false")
    if not settings.langfuse_ready:
        return ToolStatus("Langfuse · no keys", "off",
                          "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env")  # fmt: skip
    if not installed("langfuse"):
        return ToolStatus("Langfuse · not installed", "problem", "Run: uv sync --extra monitoring")
    return ToolStatus("Langfuse on", "on",
                      f"Questions, answers and retrieved text are sent to Langfuse Cloud "
                      f"({settings.langfuse_base_url})")  # fmt: skip


def _pulled(model: str, models: Sequence[str]) -> bool:
    names = set(models) | {m.removesuffix(":latest") for m in models}
    return model in names


def trulens_status(settings: Settings, models: Sequence[str] | None, *,
                   installed: Callable[[str], bool] = _installed) -> ToolStatus:  # fmt: skip
    judge = settings.trulens_judge_model
    if not settings.trulens_enabled:
        return ToolStatus("TruLens off", "off", "RAG_TRULENS_ENABLED=false")
    if not installed("trulens.providers.litellm"):
        return ToolStatus("TruLens · not installed", "problem", "Run: uv sync --extra monitoring")
    if models is None:
        return ToolStatus("TruLens · Ollama down", "problem", "Start Ollama to score answers")
    if not _pulled(judge, models):
        return ToolStatus("TruLens · judge missing", "problem", f"Run: ollama pull {judge}")
    return ToolStatus(f"TruLens · {judge}", "on",
                      f"Every answer is scored in the background by {judge}")  # fmt: skip
