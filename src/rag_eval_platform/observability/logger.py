"""Record answered questions, feedback and evaluation runs in the observability store.

Recording never breaks the caller: ``QueryLogger`` and ``record_eval_run`` log a warning
and return None when the store fails. ``record_feedback`` raises, because the API must
tell the client that the query id was unknown.
"""

import hashlib
import json
import logging
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Answer
from rag_eval_platform.observability.cost import Prices, cost_usd
from rag_eval_platform.observability.logset import LogSet
from rag_eval_platform.observability.store import (
    EvalKind,
    EvalRunRecord,
    FeedbackRecord,
    ObserveStore,
    QueryRecord,
    Source,
)

logger = logging.getLogger(__name__)
Clock = Callable[[], datetime]
_DEFAULT_PRICES = Prices()


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return uuid.uuid4().hex


class QueryLogger:
    """Turns an answer (or a failure) into one ``queries`` row."""

    def __init__(self, store: ObserveStore, *, source: Source, collection: str, style: str,
                 prices: Prices = _DEFAULT_PRICES, store_text: bool = True,
                 clock: Clock = utc_now) -> None:  # fmt: skip
        self._store, self._source, self._collection, self._style = store, source, collection, style
        self._prices, self._store_text, self._clock = prices, store_text, clock

    def record(self, answer: Answer, *, retrieve_ms: float, total_ms: float) -> str | None:
        return self._save(QueryRecord(
            id=new_id(), ts=self._clock(), source=self._source, collection=self._collection,
            question=self._question(answer.question),
            answer=answer.text if self._store_text else None, style=self._style,
            prompt_version=answer.prompt_version, model=answer.model, refusal=answer.is_refusal,
            retrieved_chunk_ids=tuple(r.chunk.id for r in answer.sources),
            cited_doc_ids=tuple(dict.fromkeys(c.source.chunk.doc_id for c in answer.citations)),
            invalid_citations=tuple(answer.invalid_citations),
            input_tokens=answer.input_tokens, output_tokens=answer.output_tokens,
            retrieve_ms=round(retrieve_ms, 1), generate_ms=round(answer.latency_ms, 1),
            total_ms=round(total_ms, 1),
            cost_usd=cost_usd(answer.input_tokens, answer.output_tokens, self._prices),
            status="ok",
        ))  # fmt: skip

    def record_error(self, question: str, error: BaseException, *, total_ms: float) -> str | None:
        return self._save(QueryRecord(
            id=new_id(), ts=self._clock(), source=self._source, collection=self._collection,
            question=self._question(question), answer=None, style=self._style,
            prompt_version=None, model=None, refusal=False, retrieved_chunk_ids=(),
            cited_doc_ids=(), invalid_citations=(), input_tokens=None, output_tokens=None,
            retrieve_ms=None, generate_ms=None, total_ms=round(total_ms, 1), cost_usd=0.0,
            status="error", error=f"{type(error).__name__}: {error}"[:500],
        ))  # fmt: skip

    def _question(self, text: str) -> str:
        if self._store_text:
            return text
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save(self, record: QueryRecord) -> str | None:
        try:
            self._store.add_query(record)
        except Exception as exc:  # recording must never break answering
            logger.warning("Could not record the query", extra={"error": str(exc)})
            return None
        return record.id


def create_store(settings: Settings) -> ObserveStore | None:
    return ObserveStore(settings.observe_db) if settings.observe_enabled else None


def create_logset(settings: Settings) -> LogSet | None:
    """The host's writable store plus any read-only logs (e.g. the Docker API's own file),
    for the ⑧ Observe tab. ``None`` when observability is off. A read-only path that
    resolves to the same file as ``settings.observe_db`` is skipped: the host already
    writes and reads that file directly."""
    host = create_store(settings)
    if host is None:
        return None
    host_path = settings.observe_db.resolve()
    readonly = tuple(
        ObserveStore(path, read_only=True)
        for path in settings.observe_readonly_dbs
        if path.resolve() != host_path
    )
    return LogSet(host, readonly)


def create_query_logger(settings: Settings, source: Source, *, collection: str | None = None,
                        style: str = "concise") -> QueryLogger | None:  # fmt: skip
    store = create_store(settings)
    if store is None:
        return None
    return QueryLogger(store, source=source, collection=collection or settings.collection_name,
                       style=style, prices=Prices.from_settings(settings),
                       store_text=settings.observe_store_text)  # fmt: skip


def record_feedback(store: ObserveStore, query_id: str, rating: int, comment: str | None,
                    source: Source, clock: Clock = utc_now) -> str:  # fmt: skip
    record = FeedbackRecord(new_id(), query_id, clock(), rating, comment, source)
    store.add_feedback(record)
    return record.id


def record_eval_run(store: ObserveStore | None, kind: EvalKind, passed: bool,
                    metrics: Mapping[str, float | None], fingerprint: Mapping[str, Any] | None,
                    clock: Clock = utc_now) -> str | None:  # fmt: skip
    """One row per evaluation run; unscored metrics are dropped. Never raises."""
    if store is None:
        return None
    digest = (hashlib.sha256(json.dumps(fingerprint, sort_keys=True, default=str).encode())
              .hexdigest()[:16] if fingerprint else None)  # fmt: skip
    record = EvalRunRecord(
        new_id(), clock(), kind, passed,
        {k: float(v) for k, v in metrics.items() if v is not None}, digest)  # fmt: skip
    try:
        store.add_eval_run(record)
    except Exception as exc:  # evaluation results matter more than their history
        logger.warning("Could not record the evaluation run", extra={"error": str(exc)})
        return None
    return record.id
