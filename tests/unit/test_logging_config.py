"""Tests for config.logging_config."""

import json
import logging
import sys
from collections.abc import Iterator

import pytest

from rag_eval_platform.config.logging_config import JsonFormatter, configure_logging


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def test_json_formatter_emits_one_json_object_with_core_fields() -> None:
    record = logging.LogRecord("rag.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "rag.test"
    assert payload["message"] == "hello world"
    assert "timestamp" in payload


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.LogRecord("rag.test", logging.INFO, __file__, 1, "query", (), None)
    record.latency_ms = 42

    payload = json.loads(JsonFormatter().format(record))

    assert payload["latency_ms"] == 42


def test_json_formatter_includes_exception_text() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord(
            "rag.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info()
        )

    payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_sets_level_and_single_handler() -> None:
    configure_logging("DEBUG")
    configure_logging("DEBUG")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, JsonFormatter)


def test_configure_logging_rejects_unknown_level() -> None:
    with pytest.raises(ValueError, match="log level"):
        configure_logging("LOUD")
