"""FastAPI application: ``create_app`` wires injected services, ``create_app_from_settings``
the real ones (the models load once, at startup)."""

import logging
from collections.abc import Mapping
from typing import get_args

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.api.health import HealthCheck, chroma_check, llm_check
from rag_eval_platform.api.routes import router
from rag_eval_platform.api.schemas import CheckOut, ErrorResponse
from rag_eval_platform.api.security import RateLimiter
from rag_eval_platform.api.services import ApiServices, QueryPipeline
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import GenerationError, create_generator
from rag_eval_platform.generation.prompt_templates import AnswerStyle
from rag_eval_platform.pipeline import RagPipeline
from rag_eval_platform.retrieval.retriever import create_retriever
from rag_eval_platform.retrieval.vector_store import VectorStoreError, create_vector_store

logger = logging.getLogger(__name__)

_REQUEST_HINT = 'Send {"question": "...", "answer_style": "concise"}'
_FAILURES: dict[type[Exception], tuple[int, str]] = {
    GenerationError: (502, "Is Ollama running with the model pulled? See GET /health."),
    VectorStoreError: (503, "Start Chroma: docker compose -f docker/docker-compose.yml up -d"),
    OptionalDependencyError: (503, "Install the extras: uv sync --all-extras"),
}


def _error(status: int, error: str, hint: str | None = None,
           headers: Mapping[str, str] | None = None) -> JSONResponse:  # fmt: skip
    body = ErrorResponse(error=error, hint=hint).model_dump()
    return JSONResponse(body, status_code=status, headers=dict(headers) if headers else None)


def create_app(services: ApiServices) -> FastAPI:
    app = FastAPI(
        title="RAG Eval Platform API",
        version="1.0.0",
        description="Cited answers from your documents: retrieve, then generate.",
    )
    app.state.services = services
    app.include_router(router)

    # Starlette's HTTPException also covers routing errors (404, 405), not only our own.
    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail), headers=exc.headers)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unexpected error", extra={"path": request.url.path})
        return _error(500, "Internal error", hint="See the server log")

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = f"Invalid request: {where} {first.get('msg', '')}".strip()
        return _error(422, message, hint=_REQUEST_HINT)

    for failure, (status, hint) in _FAILURES.items():

        def handler(request: Request, exc: Exception, status: int = status,
                    hint: str = hint) -> JSONResponse:  # fmt: skip
            logger.warning("Query failed", extra={"error": str(exc), "status": status})
            return _error(status, str(exc), hint)

        app.add_exception_handler(failure, handler)
    return app


def create_app_from_settings(settings: Settings) -> FastAPI:
    """The real services: one retriever (models load once) shared by both answer styles."""
    retriever = create_retriever(settings)
    pipelines: dict[AnswerStyle, QueryPipeline] = {
        style: RagPipeline(retriever, create_generator(settings, style))
        for style in get_args(AnswerStyle)
    }
    key = settings.api_key.get_secret_value() if settings.api_key else None
    services = ApiServices(
        pipelines=pipelines,
        checks={"chroma": chroma_check(create_vector_store(settings)), "llm": _llm_check(settings)},
        api_key=key, limiter=RateLimiter(settings.api_rate_limit),
    )  # fmt: skip
    return create_app(services)


def _llm_check(settings: Settings) -> HealthCheck:
    """Ollama must list the model; a hosted provider is not probed (it needs a paid call)."""
    if settings.llm_provider == "ollama":
        return llm_check(settings.ollama_base_url, settings.llm_model)
    return lambda: CheckOut(ok=True, detail=f"{settings.llm_provider} (not probed)")
