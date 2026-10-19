"""Tests for observability.status: what the header says about Langfuse and TruLens."""

from typing import Any

import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.status import ToolStatus, langfuse_status, trulens_status

KEYS = {"langfuse_public_key": "fake-public-value", "langfuse_secret_key": "fake-secret-value"}


def make(**options: Any) -> Settings:
    """Settings from these options only, never the developer's .env (it may hold real keys)."""
    return Settings(_env_file=None, **options)


def everything_installed(module: str) -> bool:
    return True


def nothing_installed(module: str) -> bool:
    return False


class TestLangfuse:
    def test_on_with_keys(self) -> None:
        status = langfuse_status(make(langfuse_enabled=True, **KEYS),
                                 installed=everything_installed)  # fmt: skip

        assert status == ToolStatus("Langfuse on", "on", status.detail)
        assert "Langfuse Cloud" in status.detail  # where questions and answers are sent

    @pytest.mark.parametrize(
        ("options", "label", "state"),
        [
            ({"langfuse_enabled": False, **KEYS}, "Langfuse off", "off"),
            ({"langfuse_enabled": True}, "Langfuse · no keys", "off"),
        ],
    )
    def test_off_on_purpose_is_not_a_problem(
        self, options: dict[str, object], label: str, state: str
    ) -> None:
        status = langfuse_status(make(**options), installed=everything_installed)

        assert (status.label, status.state) == (label, state)

    def test_keys_without_the_package_is_a_problem(self) -> None:
        status = langfuse_status(make(langfuse_enabled=True, **KEYS),
                                 installed=nothing_installed)  # fmt: skip

        assert (status.label, status.state) == ("Langfuse · not installed", "problem")
        assert "uv sync --extra monitoring" in status.detail


class TestTrulens:
    def test_on_when_the_judge_is_pulled(self) -> None:
        settings = make(trulens_enabled=True)

        status = trulens_status(settings, ["qwen3:8b", "gemma3:4b"], installed=everything_installed)

        assert (status.label, status.state) == ("TruLens · gemma3:4b", "on")

    def test_a_latest_tag_counts_as_pulled(self) -> None:
        settings = make(trulens_enabled=True, trulens_judge_model="phi4")

        status = trulens_status(settings, ["phi4:latest"], installed=everything_installed)

        assert status.state == "on"

    @pytest.mark.parametrize(
        ("models", "installed", "label", "hint"),
        [
            (
                ["qwen3:8b"],
                everything_installed,
                "TruLens · judge missing",
                "ollama pull gemma3:4b",
            ),
            (None, everything_installed, "TruLens · Ollama down", "Ollama"),
            (["gemma3:4b"], nothing_installed, "TruLens · not installed", "--extra monitoring"),
        ],
    )
    def test_problems_say_how_to_fix_them(
        self, models: list[str] | None, installed: object, label: str, hint: str
    ) -> None:
        status = trulens_status(make(trulens_enabled=True), models,
                                installed=installed)  # type: ignore[arg-type]  # fmt: skip

        assert (status.label, status.state) == (label, "problem")
        assert hint in status.detail

    def test_off(self) -> None:
        status = trulens_status(make(trulens_enabled=False), ["gemma3:4b"],
                                installed=everything_installed)  # fmt: skip

        assert (status.label, status.state) == ("TruLens off", "off")
