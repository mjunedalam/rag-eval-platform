"""Langfuse implementation of the tracing interface (Langfuse Python SDK v5).

Each question is one trace: a root ``answer-question`` observation (trace input = the
question, output = the answer), ``retrieve-context`` and ``rerank-chunks`` retriever
observations, and a ``generate-answer`` generation that the official ``langfuse.openai``
integration records (see generation/generator.py), nested here through ``llm_ref``.

The trace id is derived from the observability log's ``query_id`` (``create_trace_id(seed=
query_id)``), so scores that arrive later (user feedback, TruLens) attach to the right trace
from the ``query_id`` alone. The root gets that id from the client's OpenTelemetry id
generator (``seeded_id_generator``), not from ``trace_context``: ``trace_context`` without a
parent span makes the SDK invent a random remote parent, and Langfuse then finds no logical
root for the trace (found by fetching a real trace back with the Langfuse CLI).

Tracing never breaks an answer: every SDK call is wrapped, and
a failure only logs one warning per kind. The client is built from typed settings with
explicit arguments, never from ``os.environ``, so import order cannot matter.
"""

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from rag_eval_platform._optional import OptionalDependencyError, import_optional
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.tracing import (
    NOOP_QUESTION,
    NOOP_STAGE,
    NOOP_TRACER,
    LlmTraceRef,
    QuestionTrace,
    ScoreType,
    Stage,
    StageName,
    Tracer,
)

logger = logging.getLogger(__name__)

ROOT_NAME = "answer-question"
HIDDEN_TEXT = "[hidden: RAG_OBSERVE_STORE_TEXT=false]"

_warned: set[str] = set()
# The trace id the next new root should get (per thread and async task, so it is safe).
_next_trace_id: ContextVar[int | None] = ContextVar("rag_next_trace_id", default=None)
_clients: dict[tuple[str, str, str, bool], "LangfuseTracer"] = {}


def _warn_once(kind: str, exc: BaseException) -> None:
    if kind not in _warned:
        _warned.add(kind)
        logger.warning("Langfuse %s failed; tracing continues without it", kind,
                       extra={"error": f"{type(exc).__name__}: {exc}"})  # fmt: skip


def _error_fields(error: BaseException) -> dict[str, Any]:
    return {"level": "ERROR", "status_message": f"{type(error).__name__}: {error}"}


class _LangfuseStage:
    def __init__(self, observation: Any) -> None:
        self._observation = observation

    def end(self, *, output: Any = None, metadata: Mapping[str, Any] | None = None,
            error: BaseException | None = None) -> None:  # fmt: skip
        try:
            if error is not None:
                self._observation.update(**_error_fields(error))
            else:
                self._observation.update(output=output, metadata=dict(metadata or {}))
            self._observation.end()
        except Exception as exc:
            _warn_once("stage", exc)


class _LangfuseQuestion:
    def __init__(self, root: Any, trace_id: str, public_key: str) -> None:
        self._root, self._trace_id, self._public_key = root, trace_id, public_key

    @property
    def llm_ref(self) -> LlmTraceRef | None:
        return LlmTraceRef(self._trace_id, self._root.id, self._public_key)

    def stage(self, name: StageName, *, input: Any = None,
              metadata: Mapping[str, Any] | None = None) -> Stage:  # fmt: skip
        try:
            return _LangfuseStage(self._root.start_observation(
                name=name, as_type="retriever", input=input, metadata=dict(metadata or {})
            ))  # fmt: skip
        except Exception as exc:
            _warn_once("stage", exc)
            return NOOP_STAGE

    def finish(self, *, answer: str | None = None, error: BaseException | None = None,
               metadata: Mapping[str, Any] | None = None) -> None:  # fmt: skip
        try:
            if error is not None:
                self._root.update(**_error_fields(error))
            else:
                self._root.update(output=answer, metadata=dict(metadata or {}))
            self._root.end()
        except Exception as exc:
            _warn_once("trace", exc)


Propagate = Callable[..., AbstractContextManager[Any]]
SeedTrace = Callable[[str], AbstractContextManager[None]]


@contextmanager
def next_trace_id(trace_id: str) -> Iterator[None]:
    """Within this block, a new root observation gets ``trace_id`` (32 hex characters)."""
    token = _next_trace_id.set(int(trace_id, 16))
    try:
        yield
    finally:
        _next_trace_id.reset(token)


def seeded_id_generator(random_generator: type[Any]) -> Any:
    """An OpenTelemetry id generator that uses the id set by ``next_trace_id`` once.

    ``random_generator`` is OpenTelemetry's ``RandomIdGenerator`` (passed in, so this module
    imports nothing from the optional ``monitoring`` extra at load time).
    """

    class SeededIdGenerator(random_generator):  # type: ignore[misc]
        def generate_trace_id(self) -> int:
            seeded = _next_trace_id.get()
            if seeded is None:
                return int(super().generate_trace_id())
            _next_trace_id.set(None)  # one root per seeded id
            return seeded

    return SeededIdGenerator()


class LangfuseTracer:
    """Traces questions in Langfuse; ``client`` is a ``langfuse.Langfuse`` instance."""

    def __init__(self, client: Any, *, public_key: str, propagate: Propagate,
                 seed_trace: SeedTrace = next_trace_id) -> None:  # fmt: skip
        self._client, self._public_key, self._propagate = client, public_key, propagate
        self._seed_trace = seed_trace

    @property
    def enabled(self) -> bool:
        return True

    def start(self, query_id: str, question: str, *, tags: Sequence[str] = (),
              metadata: Mapping[str, Any] | None = None) -> QuestionTrace:  # fmt: skip
        try:
            trace_id = self._client.create_trace_id(seed=query_id)
            # Tags are fixed when an observation is created, so they wrap the root's creation.
            with self._propagate(tags=list(tags)), self._seed_trace(trace_id):
                root = self._client.start_observation(
                    name=ROOT_NAME,
                    as_type="chain",
                    input=question,
                    metadata=dict(metadata or {}),
                )
        except Exception as exc:
            _warn_once("trace", exc)
            return NOOP_QUESTION
        return _LangfuseQuestion(root, trace_id, self._public_key)

    def score(self, query_id: str, name: str, value: float, *, comment: str | None = None,
              data_type: ScoreType = "NUMERIC") -> None:  # fmt: skip
        try:
            self._client.create_score(
                name=name,
                value=value,
                trace_id=self._client.create_trace_id(seed=query_id),
                data_type=data_type,
                comment=comment,
            )
        except Exception as exc:
            _warn_once("score", exc)

    def trace_url(self, query_id: str) -> str | None:
        try:
            url: str | None = self._client.get_trace_url(
                trace_id=self._client.create_trace_id(seed=query_id)
            )
        except Exception as exc:
            _warn_once("trace URL", exc)
            return None
        return url

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception as exc:
            _warn_once("flush", exc)


def _hide_text(*, data: Any, **kwargs: Any) -> Any:
    """Langfuse mask hook for ``RAG_OBSERVE_STORE_TEXT=false``: no text leaves the machine."""
    return None if data is None else HIDDEN_TEXT


def _release() -> str:
    try:
        return version("rag-eval-platform")
    except PackageNotFoundError:
        return "unknown"


def create_tracer(settings: Settings) -> Tracer:
    """The process's Langfuse tracer, or ``NOOP_TRACER`` when tracing is off or unavailable."""
    if not settings.langfuse_ready or settings.langfuse_secret_key is None:
        return NOOP_TRACER
    public_key = settings.langfuse_public_key or ""
    key = (public_key, settings.langfuse_base_url, settings.langfuse_environment,
           settings.observe_store_text)  # fmt: skip
    if key in _clients:
        return _clients[key]
    try:
        module = import_optional("langfuse", extra="monitoring")
        id_generators = import_optional("opentelemetry.sdk.trace.id_generator", extra="monitoring")
        client = module.Langfuse(
            public_key=public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_base_url,
            environment=settings.langfuse_environment,
            release=_release(),
            mask=None if settings.observe_store_text else _hide_text,
            id_generator=seeded_id_generator(id_generators.RandomIdGenerator),
        )
    except OptionalDependencyError as exc:
        _warn_once("setup", exc)
        return NOOP_TRACER
    except Exception as exc:  # a bad key format or URL must not stop the app
        _warn_once("setup", exc)
        return NOOP_TRACER
    tracer = LangfuseTracer(client, public_key=public_key, propagate=module.propagate_attributes)
    _clients[key] = tracer
    return tracer


def clear_tracer_cache() -> None:
    """For tests: forget the per-process clients."""
    _clients.clear()
