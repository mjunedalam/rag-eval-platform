"""LogSet: the host's writable observability store plus any read-only ones, merged.

Read-only stores exist because of the one-writer-per-file rule in ``store.py``'s
docstring: the Docker API writes only its own ``rag-api.db``, so the ⑧ Observe tab (which
runs on the host) can never write it, but still wants to show what it holds. ``LogSet``
gives the tab one object with the same read API as ``ObserveStore``, merging every
store's rows; writes (``mark_reviewed``) always go to the host store, whose own errors are
never swallowed. A broken read-only store is skipped for that one read, with a warning.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TypeVar

from rag_eval_platform.observability.store import (
    EvalRunRecord,
    FeedbackRecord,
    ObserveStore,
    ObserveStoreError,
    QueryRecord,
    TriadScoreRecord,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")


@dataclass(frozen=True)
class LogSet:
    host: ObserveStore
    readonly: tuple[ObserveStore, ...] = ()

    def _stores(self) -> tuple[ObserveStore, ...]:
        return (self.host, *self.readonly)

    def _safe(self, store: ObserveStore, read: Callable[[ObserveStore], T]) -> T | None:
        try:
            return read(store)
        except ObserveStoreError:
            if store is self.host:
                raise
            logger.warning("Skipping a broken read-only observability log: %s", store.path)
            return None

    def _apply_reviews(self, records: Sequence[QueryRecord]) -> list[QueryRecord]:
        reviews = self.host.reviews()
        return [
            replace(record, reviewed_at=reviews[record.id])
            if record.reviewed_at is None and record.id in reviews
            else record
            for record in records
        ]

    def get_query(self, query_id: str) -> QueryRecord | None:
        for store in self._stores():
            record = self._safe(store, lambda s: s.get_query(query_id))
            if record is not None:
                return self._apply_reviews([record])[0]
        return None

    def queries(self, since: datetime | None = None) -> list[QueryRecord]:
        merged: list[QueryRecord] = []
        for store in self._stores():
            merged.extend(self._safe(store, lambda s: s.queries(since)) or [])
        merged.sort(key=lambda q: (q.ts, q.id))
        return self._apply_reviews(merged)

    def recent_queries(self, limit: int) -> list[QueryRecord]:
        """The newest ``limit`` queries across every store, returned oldest first."""
        if limit <= 0:
            return []
        merged: list[QueryRecord] = []
        for store in self._stores():
            merged.extend(self._safe(store, lambda s: s.recent_queries(limit)) or [])
        merged.sort(key=lambda q: (q.ts, q.id))
        return self._apply_reviews(merged[-limit:])

    def count_queries(self) -> int:
        return sum(self._safe(store, lambda s: s.count_queries()) or 0 for store in self._stores())

    def feedback_for(self, query_ids: Sequence[str]) -> list[FeedbackRecord]:
        merged: list[FeedbackRecord] = []
        for store in self._stores():
            merged.extend(self._safe(store, lambda s: s.feedback_for(query_ids)) or [])
        merged.sort(key=lambda f: (f.ts, f.id))
        return merged

    def triad_scores_for(self, query_ids: Sequence[str]) -> list[TriadScoreRecord]:
        merged: list[TriadScoreRecord] = []
        for store in self._stores():
            merged.extend(self._safe(store, lambda s: s.triad_scores_for(query_ids)) or [])
        return merged

    def eval_runs(self, kind: str | None = None) -> list[EvalRunRecord]:
        merged: list[EvalRunRecord] = []
        for store in self._stores():
            merged.extend(self._safe(store, lambda s: s.eval_runs(kind)) or [])
        merged.sort(key=lambda r: (r.ts, r.id))
        return merged

    def mark_reviewed(self, query_id: str, at: datetime) -> bool:
        """Always writes to the host store, regardless of which store logged the query."""
        return self.host.mark_reviewed(query_id, at)
