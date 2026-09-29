"""API routes: /health, /query and /query/stream."""

import json
import logging
import time
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from rag_eval_platform.api.schemas import (
    ErrorResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    to_response,
)
from rag_eval_platform.api.services import QueryPipeline, guard, services_of
from rag_eval_platform.api.sse import sse_event
from rag_eval_platform.pipeline import TimedAnswer

logger = logging.getLogger(__name__)
router = APIRouter()
_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorResponse} for code in (401, 422, 429, 502, 503)
}


@router.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health(request: Request) -> JSONResponse:
    checks = {name: check() for name, check in services_of(request).checks.items()}
    ok = all(c.ok for c in checks.values())
    body = HealthResponse(status="ok" if ok else "degraded", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)


@router.post("/query", response_model=QueryResponse, responses=_ERRORS,
             dependencies=[Depends(guard)])  # fmt: skip
def query(body: QueryRequest, request: Request) -> QueryResponse:
    pipeline = services_of(request).pipelines[body.answer_style]
    return to_response(pipeline.ask_timed(body.question))


@router.post("/query/stream", responses={200: {"content": {"text/event-stream": {}}}, **_ERRORS},
             dependencies=[Depends(guard)])  # fmt: skip
def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
    pipeline = services_of(request).pipelines[body.answer_style]
    return StreamingResponse(_events(pipeline, body.question), media_type="text/event-stream")


def _events(pipeline: QueryPipeline, question: str) -> Iterator[str]:
    started = time.perf_counter()
    try:
        yield sse_event("status", "retrieving")
        stream = pipeline.ask_stream(question)
        retrieve_ms = (time.perf_counter() - started) * 1000
        yield sse_event("status", "generating")
        for piece in stream:
            yield sse_event("token", piece)
        timed = TimedAnswer(stream.answer, retrieve_ms, (time.perf_counter() - started) * 1000)
        yield sse_event("done", to_response(timed).model_dump_json())
    except Exception as exc:  # the response has started: report the failure as an event
        logger.warning("Stream failed", extra={"error": str(exc)})
        yield sse_event("error", json.dumps(ErrorResponse(error=str(exc)).model_dump()))
