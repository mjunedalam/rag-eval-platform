"""Logging configuration.

Every process (API, scripts, evaluation runs) logs one JSON object per line to
stderr, so query logs and evaluation runs can be parsed by the observability layer.
Pass structured fields with ``logger.info("query", extra={"latency_ms": 42})``.
"""

import json
import logging
from datetime import UTC, datetime
from typing import Any

# Attributes every LogRecord has; anything else was passed through ``extra``.
_RESERVED_ATTRS = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", (), None)).keys()
    | {"message", "asctime", "taskName"}
)


class JsonFormatter(logging.Formatter):
    """Format a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {key: value for key, value in vars(record).items() if key not in _RESERVED_ATTRS}
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Replace the root logger's handlers with a single JSON stderr handler."""
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(f"Unknown log level: {level!r}")

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(numeric_level)
