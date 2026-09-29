"""What the API needs at run time, injected so tests can use fakes."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from fastapi import HTTPException, Request

from rag_eval_platform.api.health import HealthCheck
from rag_eval_platform.api.security import RateLimiter, check_api_key
from rag_eval_platform.generation.generator import Answer
from rag_eval_platform.generation.prompt_templates import AnswerStyle
from rag_eval_platform.pipeline import TimedAnswer


class _AnswerStream(Iterable[str], Protocol):
    @property
    def answer(self) -> Answer: ...


class QueryPipeline(Protocol):
    def ask_timed(self, question: str) -> TimedAnswer: ...

    def ask_stream(self, question: str) -> _AnswerStream: ...


@dataclass(frozen=True)
class ApiServices:
    pipelines: Mapping[AnswerStyle, QueryPipeline]
    checks: Mapping[str, HealthCheck]
    api_key: str | None
    limiter: RateLimiter


def services_of(request: Request) -> ApiServices:
    services: ApiServices = request.app.state.services
    return services


def guard(request: Request) -> None:
    """Rate limit first, then the API key (both skipped for /health and the docs)."""
    services = services_of(request)
    client = request.client.host if request.client else "unknown"
    allowed, retry_after = services.limiter.allow(client)
    if not allowed:
        raise HTTPException(429, detail="Too many requests; slow down",
                            headers={"Retry-After": str(max(1, round(retry_after)))})  # fmt: skip
    if not check_api_key(request.headers.get("X-API-Key"), services.api_key):
        raise HTTPException(401, detail="Missing or wrong X-API-Key header")
