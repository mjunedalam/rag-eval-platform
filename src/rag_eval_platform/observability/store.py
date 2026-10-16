"""SQLite store for the query log, feedback and evaluation runs (Phase 7).

**One writer per file.** A stress test proved that the API container (a Linux VM under
Docker Desktop) and the host (macOS) writing the *same* SQLite file through a Docker
Desktop bind mount corrupts it within a fraction of a second: Docker Desktop's file
sharing does not carry SQLite's POSIX locks between the host and the VM. Two writers on
the host alone are fine (see ``test_two_writers_on_one_file_lose_nothing`` below). So the
Docker API writes only its own file (``rag-api.db``, via ``RAG_OBSERVE_DB`` in
``docker-compose.yml``); everything on the host — the playground, ``ask.py``, a locally
run API, the evaluation CLIs — writes ``rag.db``. ``ObserveStore(path, read_only=True)``
opens another process's file (typically ``rag-api.db``) through a read-only, immutable
SQLite URI, so it never takes a lock and can never corrupt the writer's file; see
``LogSet`` (``logset.py``) for how the ⑧ Observe tab merges the host's file with any such
read-only logs.

One short connection per operation, the classic rollback journal and a 5 s busy timeout.
If the log is ever reported corrupt, move ``data/observability/rag.db`` aside to start
fresh. The schema version is SQLite's ``user_version``. Timestamps are stored as UTC
ISO-8601 text, so they sort as text.
"""

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import astuple, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote

Source = Literal["api", "playground", "cli"]
EvalKind = Literal["retrieval", "generation", "gate"]
SCHEMA_VERSION = 2
_FEEDBACK_FOR_CHUNK = 500  # SQLite's `IN (...)` list stays well under its variable limit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, source TEXT NOT NULL, collection TEXT NOT NULL,
    question TEXT NOT NULL, answer TEXT, style TEXT NOT NULL, prompt_version TEXT,
    model TEXT, refusal INTEGER NOT NULL, retrieved_chunk_ids TEXT NOT NULL,
    cited_doc_ids TEXT NOT NULL, invalid_citations TEXT NOT NULL, input_tokens INTEGER,
    output_tokens INTEGER, retrieve_ms REAL, generate_ms REAL, total_ms REAL,
    cost_usd REAL NOT NULL, status TEXT NOT NULL, error TEXT, reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY, query_id TEXT NOT NULL REFERENCES queries(id), ts TEXT NOT NULL,
    rating INTEGER NOT NULL, comment TEXT, source TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, passed INTEGER NOT NULL,
    metrics TEXT NOT NULL, fingerprint TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    query_id TEXT PRIMARY KEY, ts TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS queries_ts ON queries(ts);
CREATE INDEX IF NOT EXISTS queries_source_ts ON queries(source, ts);
CREATE INDEX IF NOT EXISTS feedback_query ON feedback(query_id);
CREATE INDEX IF NOT EXISTS eval_runs_kind_ts ON eval_runs(kind, ts);
"""

# Every query column, qualified and aliased so a LEFT JOIN with `reviews` never shadows one
# name with another (see `_query`, which reads columns by name from the row). `reviewed_at`
# prefers the `reviews` table (settable from any store, e.g. the host marking an API-only
# row) and falls back to the legacy `queries.reviewed_at` column for rows upgraded from v1.
_QUERY_SELECT = """
SELECT queries.id AS id, queries.ts AS ts, queries.source AS source,
       queries.collection AS collection, queries.question AS question,
       queries.answer AS answer, queries.style AS style,
       queries.prompt_version AS prompt_version, queries.model AS model,
       queries.refusal AS refusal, queries.retrieved_chunk_ids AS retrieved_chunk_ids,
       queries.cited_doc_ids AS cited_doc_ids, queries.invalid_citations AS invalid_citations,
       queries.input_tokens AS input_tokens, queries.output_tokens AS output_tokens,
       queries.retrieve_ms AS retrieve_ms, queries.generate_ms AS generate_ms,
       queries.total_ms AS total_ms, queries.cost_usd AS cost_usd, queries.status AS status,
       queries.error AS error, COALESCE(reviews.ts, queries.reviewed_at) AS reviewed_at
FROM queries LEFT JOIN reviews ON reviews.query_id = queries.id
"""


class ObserveStoreError(Exception):
    """The observability database cannot be opened or used."""


class UnknownQueryError(LookupError):
    """Feedback for a query id that is not in the log."""


@dataclass(frozen=True)
class QueryRecord:
    id: str
    ts: datetime
    source: str
    collection: str
    question: str  # or "sha256:<hex>" when text storage is off
    answer: str | None
    style: str
    prompt_version: str | None
    model: str | None
    refusal: bool
    retrieved_chunk_ids: tuple[str, ...]
    cited_doc_ids: tuple[str, ...]
    invalid_citations: tuple[int, ...]
    input_tokens: int | None
    output_tokens: int | None
    retrieve_ms: float | None
    generate_ms: float | None
    total_ms: float | None
    cost_usd: float
    status: str  # "ok" or "error"
    error: str | None = None
    reviewed_at: datetime | None = None


@dataclass(frozen=True)
class FeedbackRecord:
    id: str
    query_id: str
    ts: datetime
    rating: int  # 1 = 👍, 0 = 👎
    comment: str | None
    source: str


@dataclass(frozen=True)
class EvalRunRecord:
    id: str
    ts: datetime
    kind: str
    passed: bool
    metrics: Mapping[str, float]
    fingerprint: str | None


def utc(ts: datetime) -> datetime:
    """Naive timestamps are taken as UTC; aware ones are converted to UTC."""
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)


def _text(ts: datetime | None) -> str | None:
    return None if ts is None else utc(ts).isoformat()


def _time(text: str | None) -> datetime | None:
    return None if text is None else datetime.fromisoformat(text)


class ObserveStore:
    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = path
        self.read_only = read_only

    def _missing(self) -> bool:
        """True for a read-only store whose file does not exist: "no rows", not an error."""
        return self.read_only and not self.path.exists()

    def _require_writable(self) -> None:
        if self.read_only:
            raise ObserveStoreError(f"read-only log: {self.path}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        hint = f"Move {self.path} aside to start a fresh log"
        try:
            if self.read_only:
                # `mode=ro&immutable=1`: never creates the file, never takes a lock, and
                # SQLite skips its usual change-detection bookkeeping (safe here because
                # nothing on this side ever writes the file).
                uri = f"file:{quote(str(self.path.resolve(strict=False)))}?mode=ro&immutable=1"
                conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(self.path, timeout=5.0)
        except (OSError, sqlite3.Error) as exc:
            raise ObserveStoreError(f"Cannot open {self.path}: {exc}. {hint}") from exc
        conn.row_factory = sqlite3.Row
        try:
            if self.read_only:
                with conn:
                    yield conn
            else:
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("PRAGMA foreign_keys=ON")
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                if version > SCHEMA_VERSION:
                    raise ObserveStoreError(
                        f"{self.path} has schema v{version}, newer than this code (v"
                        f"{SCHEMA_VERSION}). {hint}"
                    )
                if version < SCHEMA_VERSION:
                    conn.executescript(_SCHEMA)
                    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                with conn:  # one transaction: commit on success, roll back on error
                    yield conn
        except sqlite3.Error as exc:
            raise ObserveStoreError(f"{self.path}: {exc}. {hint}") from exc
        finally:
            conn.close()

    def add_query(self, record: QueryRecord) -> None:
        self._require_writable()
        values = astuple(record)
        row = (
            *values[:1],
            _text(record.ts),
            *values[2:10],
            json.dumps(record.retrieved_chunk_ids),
            json.dumps(record.cited_doc_ids),
            json.dumps(record.invalid_citations),
            *values[13:21],
            _text(record.reviewed_at),
        )  # fmt: skip
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO queries VALUES ({', '.join('?' * len(row))})",  # noqa: S608
                row,
            )

    def get_query(self, query_id: str) -> QueryRecord | None:
        if self._missing():
            return None
        with self._connect() as conn:
            row = conn.execute(_QUERY_SELECT + " WHERE queries.id = ?", (query_id,)).fetchone()
        return None if row is None else _query(row)

    def queries(self, since: datetime | None = None) -> list[QueryRecord]:
        if self._missing():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                _QUERY_SELECT + " WHERE queries.ts >= ? ORDER BY queries.ts, queries.id",
                (_text(since) or "",),
            ).fetchall()  # fmt: skip
        return [_query(row) for row in rows]

    def recent_queries(self, limit: int) -> list[QueryRecord]:
        """The newest ``limit`` queries, returned oldest first."""
        if self._missing():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                _QUERY_SELECT + " ORDER BY queries.ts DESC, queries.id DESC LIMIT ?",
                (limit,),
            ).fetchall()  # fmt: skip
        return [_query(row) for row in reversed(rows)]

    def count_queries(self) -> int:
        """The total number of logged queries, regardless of any limit."""
        if self._missing():
            return 0
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM queries").fetchone()
        return int(row[0])

    def mark_reviewed(self, query_id: str, at: datetime) -> bool:
        """Record ``query_id`` as reviewed. Always succeeds: the query need not live in this
        file, so a host store can mark a query answered only in a read-only log (e.g. the
        Docker API's own file) as reviewed."""
        self._require_writable()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO reviews (query_id, ts) VALUES (?, ?) "
                "ON CONFLICT(query_id) DO UPDATE SET ts = excluded.ts",
                (query_id, _text(at)),
            )  # fmt: skip
        return True

    def reviews(self) -> dict[str, datetime]:
        """Every reviewed query id in this file, mapped to when it was reviewed."""
        if self._missing():
            return {}
        with self._connect() as conn:
            rows = conn.execute("SELECT query_id, ts FROM reviews").fetchall()
        return {row["query_id"]: _utc_time(row["ts"]) for row in rows}

    def add_feedback(self, record: FeedbackRecord) -> None:
        self._require_writable()
        with self._connect() as conn:
            if (
                conn.execute(
                    "SELECT 1 FROM queries WHERE id = ?",
                    (record.query_id,),
                ).fetchone()
                is None
            ):
                raise UnknownQueryError(f"Unknown query_id: {record.query_id}")
            conn.execute(
                "INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    record.query_id,
                    _text(record.ts),
                    record.rating,
                    record.comment,
                    record.source,
                ),
            )  # fmt: skip

    def feedback(self, since: datetime | None = None) -> list[FeedbackRecord]:
        if self._missing():
            return []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE ts >= ? ORDER BY ts, id",
                (_text(since) or "",),
            ).fetchall()  # fmt: skip
        return [_feedback(row) for row in rows]

    def feedback_for(self, query_ids: Sequence[str]) -> list[FeedbackRecord]:
        """Feedback for these query ids, oldest first; ``[]`` for empty input.

        Chunked into groups of at most ``_FEEDBACK_FOR_CHUNK`` ids per ``IN (...)`` list, well
        under SQLite's limit on the number of variables in one statement.
        """
        if not query_ids or self._missing():
            return []
        results: list[FeedbackRecord] = []
        with self._connect() as conn:
            for start in range(0, len(query_ids), _FEEDBACK_FOR_CHUNK):
                chunk = query_ids[start : start + _FEEDBACK_FOR_CHUNK]
                placeholders = ", ".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT * FROM feedback WHERE query_id IN ({placeholders}) "  # noqa: S608
                    "ORDER BY ts, id",
                    tuple(chunk),
                ).fetchall()  # fmt: skip
                results.extend(_feedback(row) for row in rows)
        results.sort(key=lambda f: (f.ts, f.id))
        return results

    def add_eval_run(self, record: EvalRunRecord) -> None:
        self._require_writable()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO eval_runs VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.id,
                    _text(record.ts),
                    record.kind,
                    int(record.passed),
                    json.dumps(dict(record.metrics)),
                    record.fingerprint,
                ),
            )  # fmt: skip

    def eval_runs(self, kind: str | None = None) -> list[EvalRunRecord]:
        if self._missing():
            return []
        sql = (
            "SELECT * FROM eval_runs"  # noqa: S608
            + (" WHERE kind = ?" if kind else "")
            + " ORDER BY ts, id"
        )
        with self._connect() as conn:
            rows = conn.execute(sql, (kind,) if kind else ()).fetchall()
        return [
            EvalRunRecord(
                row["id"],
                _utc_time(row["ts"]),
                row["kind"],
                bool(row["passed"]),
                json.loads(row["metrics"]),
                row["fingerprint"],
            )
            for row in rows
        ]  # fmt: skip


def _feedback(row: sqlite3.Row) -> FeedbackRecord:
    return FeedbackRecord(row["id"], row["query_id"], _utc_time(row["ts"]), row["rating"],
                          row["comment"], row["source"])  # fmt: skip


def _utc_time(text: str) -> datetime:
    parsed = _time(text)
    assert parsed is not None  # noqa: S101 - NOT NULL column
    return parsed


def _query(row: sqlite3.Row) -> QueryRecord:
    return QueryRecord(
        id=row["id"],
        ts=_utc_time(row["ts"]),
        source=row["source"],
        collection=row["collection"],
        question=row["question"],
        answer=row["answer"],
        style=row["style"],
        prompt_version=row["prompt_version"],
        model=row["model"],
        refusal=bool(row["refusal"]),
        retrieved_chunk_ids=tuple(json.loads(row["retrieved_chunk_ids"])),
        cited_doc_ids=tuple(json.loads(row["cited_doc_ids"])),
        invalid_citations=tuple(json.loads(row["invalid_citations"])),
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        retrieve_ms=row["retrieve_ms"],
        generate_ms=row["generate_ms"],
        total_ms=row["total_ms"],
        cost_usd=row["cost_usd"],
        status=row["status"],
        error=row["error"],
        reviewed_at=_time(row["reviewed_at"]),
    )  # fmt: skip
