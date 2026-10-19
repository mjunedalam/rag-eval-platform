"""FastAPI application: ``create_app`` wires injected services, ``create_app_from_settings``
the real ones (the models load once, at startup)."""

import logging
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import get_args

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from rag_eval_platform.api.errors import FAILURES, public_failure
from rag_eval_platform.api.health import HealthCheck, chroma_check, llm_check
from rag_eval_platform.api.routes import router
from rag_eval_platform.api.schemas import CheckOut, ErrorResponse
from rag_eval_platform.api.security import RateLimiter
from rag_eval_platform.api.services import ApiServices, QueryPipeline
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import create_generator
from rag_eval_platform.generation.prompt_templates import AnswerStyle
from rag_eval_platform.observability.drift import DriftThresholds
from rag_eval_platform.observability.langfuse_tracer import create_tracer
from rag_eval_platform.observability.logger import create_query_logger, create_store
from rag_eval_platform.pipeline import RagPipeline, trace_metadata
from rag_eval_platform.retrieval.retriever import create_retriever
from rag_eval_platform.retrieval.vector_store import create_vector_store

logger = logging.getLogger(__name__)

_REQUEST_HINT = 'Send {"question": "...", "answer_style": "concise"}'
_FEEDBACK_HINT = 'Send {"query_id": "...", "rating": "up" or "down", "comment": "..."}'
_HINTS_BY_PATH: dict[str, str] = {"/feedback": _FEEDBACK_HINT}


def _error(status: int, error: str, hint: str | None = None,
           headers: Mapping[str, str] | None = None) -> JSONResponse:  # fmt: skip
    body = ErrorResponse(error=error, hint=hint).model_dump()
    return JSONResponse(body, status_code=status, headers=dict(headers) if headers else None)


def create_app(services: ApiServices) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        services.tracer.flush()  # send buffered traces and scores before the server stops

    app = FastAPI(
        title="RAG Eval Platform API",
        version="1.0.0",
        description="Cited answers from your documents: retrieve, then generate.",
        lifespan=lifespan,
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
        hint = _HINTS_BY_PATH.get(request.url.path, _REQUEST_HINT)
        return _error(422, message, hint=hint)

    for failure_type in FAILURES:

        def handler(request: Request, exc: Exception) -> JSONResponse:
            failure = public_failure(exc)
            # The full message goes to the log only; callers get the fixed public text.
            logger.warning("Request failed", extra={"error": str(exc), "status": failure.status})
            return _error(failure.status, failure.error, failure.hint)

        app.add_exception_handler(failure_type, handler)
    return app


def create_app_from_settings(settings: Settings) -> FastAPI:
    """The real services: one retriever (models load once) shared by both answer styles."""
    retriever, tracer = create_retriever(settings), create_tracer(settings)
    pipelines: dict[AnswerStyle, QueryPipeline] = {
        style: RagPipeline(retriever, create_generator(settings, style),
                           create_query_logger(settings, "api", style=style), tracer=tracer,
                           tags=("api", style), trace_metadata=trace_metadata(settings))
        for style in get_args(AnswerStyle)
    }  # fmt: skip
    key = settings.api_key.get_secret_value() if settings.api_key else None
    services = ApiServices(
        pipelines=pipelines,
        checks={"chroma": chroma_check(create_vector_store(settings)), "llm": _llm_check(settings)},
        api_key=key, limiter=RateLimiter(settings.api_rate_limit),
        store=create_store(settings), drift=DriftThresholds.from_settings(settings),
        tracer=tracer,
    )  # fmt: skip
    return create_app(services)


def _llm_check(settings: Settings) -> HealthCheck:
    """Ollama must list the model; a hosted provider is not probed (it needs a paid call)."""
    if settings.llm_provider == "ollama":
        return llm_check(settings.ollama_base_url, settings.llm_model)
    return lambda: CheckOut(ok=True, detail=f"{settings.llm_provider} (not probed)")
