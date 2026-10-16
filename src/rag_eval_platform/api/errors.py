"""What an API caller is told when a request fails, and what stays in the server log.

Exception messages can name internal hosts, ports, model URLs or file paths, so they are
never sent to callers: each failure type maps to a fixed public message and a fix hint,
and the full exception goes only to the server log.
"""

from typing import NamedTuple

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.api.schemas import ErrorResponse
from rag_eval_platform.generation.generator import GenerationError
from rag_eval_platform.observability.store import ObserveStoreError, UnknownQueryError
from rag_eval_platform.retrieval.vector_store import VectorStoreError


class PublicFailure(NamedTuple):
    status: int
    error: str
    hint: str


FAILURES: dict[type[Exception], PublicFailure] = {
    GenerationError: PublicFailure(
        502, "The language model is unavailable",
        "Is Ollama running with the model pulled? See GET /health.",
    ),
    VectorStoreError: PublicFailure(
        503, "The vector store is unavailable",
        "Start Chroma: docker compose -f docker/docker-compose.yml up -d",
    ),
    OptionalDependencyError: PublicFailure(
        503, "A required package is not installed on the server",
        "Install the extras: uv sync --all-extras",
    ),
    UnknownQueryError: PublicFailure(
        404, "Unknown query_id", "Use the query_id from a /query response"
    ),
    ObserveStoreError: PublicFailure(
        503, "Observability is unavailable", "Check RAG_OBSERVE_ENABLED and RAG_OBSERVE_DB"
    ),
}  # fmt: skip

UNEXPECTED = PublicFailure(500, "Internal error", "See the server log")


def public_failure(exc: BaseException) -> PublicFailure:
    """The caller-safe status, message and hint for ``exc`` (never its own text)."""
    for failure_type, failure in FAILURES.items():
        if isinstance(exc, failure_type):
            return failure
    return UNEXPECTED


def public_error_body(exc: BaseException) -> ErrorResponse:
    failure = public_failure(exc)
    return ErrorResponse(error=failure.error, hint=failure.hint)
