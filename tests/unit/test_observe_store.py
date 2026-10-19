"""Tests for observability.store: a real SQLite file in a temporary folder."""

import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from tests.unit.observe_records import T0, make_feedback, make_query, make_run, make_triad

from rag_eval_platform.observability.store import (
    SCHEMA_VERSION,
    TRIAD_METRICS,
    ObserveStore,
    ObserveStoreError,
    UnknownQueryError,
)


@pytest.fixture
def store(tmp_path: Path) -> ObserveStore:
    return ObserveStore(tmp_path / "nested" / "rag.db")


def test_a_query_round_trips_every_field(store: ObserveStore) -> None:
    record = make_query(1, invalid_citations=(7,), error=None)

    store.add_query(record)

    assert store.get_query("q1") == record
    assert store.get_query("missing") is None


def test_creates_the_folder_and_stamps_the_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "rag.db"

    ObserveStore(path).add_query(make_query())

    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_queries_are_oldest_first_and_filter_by_time(store: ObserveStore) -> None:
    for n in (3, 1, 2):
        store.add_query(make_query(n))

    assert [q.id for q in store.queries()] == ["q1", "q2", "q3"]
    assert [q.id for q in store.queries(since=T0 + timedelta(minutes=2))] == ["q2", "q3"]


def test_timestamps_are_stored_as_utc(store: ObserveStore) -> None:
    plus_two = timezone(timedelta(hours=2))
    store.add_query(make_query(1, ts=datetime(2026, 9, 29, 14, 30, tzinfo=plus_two)))  # 12:30 UTC
    store.add_query(make_query(2, ts=datetime(2026, 9, 29, 12, 15)))  # naive: taken as UTC

    rows = store.queries()

    assert [q.id for q in rows] == ["q2", "q1"]
    assert rows[1].ts == datetime(2026, 9, 29, 12, 30, tzinfo=UTC)


def test_long_and_non_ascii_text_round_trips(store: ObserveStore) -> None:
    question = "Why? 🤔 ما هو «RAG»? 'quotes' \"double\" " + "x" * 2000
    store.add_query(make_query(1, question=question, answer="Réponse ✅ [1]."))

    saved = store.get_query("q1")

    assert saved is not None
    assert (saved.question, saved.answer) == (question, "Réponse ✅ [1].")


def test_mark_reviewed(store: ObserveStore) -> None:
    store.add_query(make_query(1))

    assert store.mark_reviewed("q1", T0) is True
    # No longer requires the query to be in this file: a host store can mark a query
    # answered by the API (and only present in the API's own read-only file) as reviewed.
    assert store.mark_reviewed("nope", T0) is True
    saved = store.get_query("q1")
    assert saved is not None
    assert saved.reviewed_at == T0


def test_mark_reviewed_again_replaces_the_timestamp(store: ObserveStore) -> None:
    store.add_query(make_query(1))
    store.mark_reviewed("q1", T0)

    assert store.mark_reviewed("q1", T0 + timedelta(minutes=5)) is True

    saved = store.get_query("q1")
    assert saved is not None
    assert saved.reviewed_at == T0 + timedelta(minutes=5)


def test_reviews_returns_every_marked_query(store: ObserveStore) -> None:
    store.add_query(make_query(1))
    store.add_query(make_query(2))
    store.mark_reviewed("q1", T0)
    store.mark_reviewed("ghost-not-in-this-file", T0 + timedelta(minutes=1))

    reviews = store.reviews()

    assert reviews == {
        "q1": T0,
        "ghost-not-in-this-file": T0 + timedelta(minutes=1),
    }


def test_feedback_needs_a_known_query(store: ObserveStore) -> None:
    store.add_query(make_query(1))
    store.add_feedback(make_feedback("q1", 0, comment="wrong source"))

    with pytest.raises(UnknownQueryError):
        store.add_feedback(make_feedback("ghost", 1))
    assert store.feedback() == [make_feedback("q1", 0, comment="wrong source")]


def test_recent_queries_returns_the_newest_n_oldest_first(store: ObserveStore) -> None:
    for n in range(5):
        store.add_query(make_query(n))

    assert [q.id for q in store.recent_queries(3)] == ["q2", "q3", "q4"]
    assert [q.id for q in store.recent_queries(100)] == [f"q{n}" for n in range(5)]
    assert store.recent_queries(0) == []


def test_feedback_for_filters_by_query_id_oldest_first(store: ObserveStore) -> None:
    store.add_query(make_query(1))
    store.add_query(make_query(2))
    store.add_query(make_query(3))
    store.add_feedback(make_feedback("q1", 0, minutes=2))
    store.add_feedback(make_feedback("q2", 1, minutes=1))

    assert [f.query_id for f in store.feedback_for(["q1", "q2"])] == ["q2", "q1"]
    assert store.feedback_for(["q3"]) == []
    assert store.feedback_for([]) == []


def test_count_queries_is_the_total_regardless_of_any_limit(store: ObserveStore) -> None:
    assert store.count_queries() == 0

    for n in range(3):
        store.add_query(make_query(n))

    assert store.count_queries() == 3


def test_eval_runs_round_trip_and_filter_by_kind(store: ObserveStore) -> None:
    store.add_eval_run(make_run("retrieval", True, {"recall": 0.93}, minutes=1))
    store.add_eval_run(make_run("gate", False, {"recall": 0.7}, minutes=2))

    assert [r.kind for r in store.eval_runs()] == ["retrieval", "gate"]
    assert store.eval_runs("gate") == [make_run("gate", False, {"recall": 0.7}, minutes=2)]


def test_a_corrupt_file_raises_with_a_hint(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    path.write_bytes(b"this is not a database" * 100)

    with pytest.raises(ObserveStoreError, match="aside"):
        ObserveStore(path).queries()


def test_a_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    with sqlite3.connect(path) as conn:
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")

    with pytest.raises(ObserveStoreError, match="newer"):
        ObserveStore(path).queries()


def test_feedback_for_chunks_ids_past_the_sqlite_variable_limit(store: ObserveStore) -> None:
    ids = [f"q{n}" for n in range(1200)]
    for query_id in ids:
        store.add_query(make_query(0, id=query_id))
    store.add_feedback(make_feedback(ids[0], 1, minutes=0))
    store.add_feedback(make_feedback(ids[1199], 0, minutes=1))

    result = store.feedback_for(ids)

    assert {f.query_id for f in result} == {ids[0], ids[1199]}
    assert [f.query_id for f in result] == [ids[0], ids[1199]]  # oldest first


# --- Read-only stores (the ⑧ tab's view of the Docker API's own file) -----------------


@pytest.fixture
def readonly_missing(tmp_path: Path) -> ObserveStore:
    return ObserveStore(tmp_path / "rag-api.db", read_only=True)


def test_readonly_store_with_a_missing_file_reads_as_empty(readonly_missing: ObserveStore) -> None:
    assert readonly_missing.queries() == []
    assert readonly_missing.recent_queries(10) == []
    assert readonly_missing.count_queries() == 0
    assert readonly_missing.get_query("q1") is None
    assert readonly_missing.feedback() == []
    assert readonly_missing.feedback_for(["q1"]) == []
    assert readonly_missing.eval_runs() == []
    assert readonly_missing.reviews() == {}


def test_readonly_store_never_creates_the_file_or_folder(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "rag-api.db"
    store = ObserveStore(path, read_only=True)

    store.queries()
    store.get_query("q1")
    store.count_queries()

    assert not path.exists()
    assert not path.parent.exists()


@pytest.mark.parametrize(
    "action",
    [
        lambda s: s.add_query(make_query(1)),
        lambda s: s.add_feedback(make_feedback("q1", 1)),
        lambda s: s.add_eval_run(make_run("gate", True, {"recall": 0.9})),
        lambda s: s.mark_reviewed("q1", T0),
    ],
)
def test_readonly_store_writes_raise(
    tmp_path: Path, action: Callable[[ObserveStore], object]
) -> None:
    store = ObserveStore(tmp_path / "rag-api.db", read_only=True)

    with pytest.raises(ObserveStoreError, match="read-only"):
        action(store)


def test_readonly_store_reads_rows_written_by_a_normal_store(tmp_path: Path) -> None:
    path = tmp_path / "rag-api.db"
    writer = ObserveStore(path)
    writer.add_query(make_query(1))
    writer.add_query(make_query(2))
    writer.add_feedback(make_feedback("q1", 0, comment="meh"))
    writer.add_eval_run(make_run("gate", True, {"recall": 0.9}))

    reader = ObserveStore(path, read_only=True)

    assert [q.id for q in reader.queries()] == ["q1", "q2"]
    assert [q.id for q in reader.recent_queries(1)] == ["q2"]
    assert reader.count_queries() == 2
    assert reader.get_query("q1") is not None
    assert [f.query_id for f in reader.feedback_for(["q1"])] == ["q1"]
    assert [r.kind for r in reader.eval_runs()] == ["gate"]


def test_readonly_store_never_migrates_or_writes_the_schema(tmp_path: Path) -> None:
    """A read-only open of a file that does not exist yet must never create one; opening a
    read-only store must never bump ``user_version`` on a file created by someone else."""
    path = tmp_path / "rag-api.db"
    writer = ObserveStore(path)
    writer.add_query(make_query(1))
    before = path.read_bytes()

    ObserveStore(path, read_only=True).queries()

    assert path.read_bytes() == before


def test_readonly_store_path_with_spaces_works(tmp_path: Path) -> None:
    folder = tmp_path / "has spaces here"
    folder.mkdir()
    path = folder / "rag-api.db"
    ObserveStore(path).add_query(make_query(1))

    reader = ObserveStore(path, read_only=True)

    assert [q.id for q in reader.queries()] == ["q1"]


def test_a_v1_file_with_rows_upgrades_to_v2_and_keeps_them(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    v1_schema = """
    CREATE TABLE queries (
        id TEXT PRIMARY KEY, ts TEXT NOT NULL, source TEXT NOT NULL, collection TEXT NOT NULL,
        question TEXT NOT NULL, answer TEXT, style TEXT NOT NULL, prompt_version TEXT,
        model TEXT, refusal INTEGER NOT NULL, retrieved_chunk_ids TEXT NOT NULL,
        cited_doc_ids TEXT NOT NULL, invalid_citations TEXT NOT NULL, input_tokens INTEGER,
        output_tokens INTEGER, retrieve_ms REAL, generate_ms REAL, total_ms REAL,
        cost_usd REAL NOT NULL, status TEXT NOT NULL, error TEXT, reviewed_at TEXT
    );
    CREATE TABLE feedback (
        id TEXT PRIMARY KEY, query_id TEXT NOT NULL REFERENCES queries(id), ts TEXT NOT NULL,
        rating INTEGER NOT NULL, comment TEXT, source TEXT NOT NULL
    );
    CREATE TABLE eval_runs (
        id TEXT PRIMARY KEY, ts TEXT NOT NULL, kind TEXT NOT NULL, passed INTEGER NOT NULL,
        metrics TEXT NOT NULL, fingerprint TEXT
    );
    """
    with sqlite3.connect(path) as conn:
        conn.executescript(v1_schema)
        conn.execute(
            "INSERT INTO queries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?)",
            ("q1", "2026-09-29T12:00:00+00:00", "api", "rag_documents", "Q?", "A [1].",
             "concise", "v1", "qwen3:8b", 0, "[]", "[]", "[]", None, None, None, None, None,
             0.0, "ok", None, "2026-09-29T12:05:00+00:00"),
        )  # fmt: skip
        conn.execute("PRAGMA user_version = 1")

    store = ObserveStore(path)
    saved = store.get_query("q1")

    assert saved is not None
    assert saved.id == "q1"
    assert saved.reviewed_at == datetime(2026, 9, 29, 12, 5, tzinfo=UTC)  # kept from the old column
    assert store.mark_reviewed("q1", T0) is True  # the new `reviews` table works post-upgrade
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_two_writers_on_one_file_lose_nothing(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    ObserveStore(path).queries()  # create the schema first

    def write(offset: int) -> None:
        writer = ObserveStore(path)
        for n in range(offset, offset + 25):
            writer.add_query(make_query(n))

    threads = [threading.Thread(target=write, args=(start,)) for start in (0, 100)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ObserveStore(path).queries()) == 50


class TestTriadScores:
    def test_scores_round_trip_for_the_requested_queries(self, store: ObserveStore) -> None:
        store.add_query(make_query(1))
        store.add_query(make_query(2))
        rows = [make_triad("q1", metric) for metric in TRIAD_METRICS]

        store.save_triad_scores(rows)
        store.save_triad_scores([make_triad("q2")])

        assert sorted(store.triad_scores_for(["q1"]), key=lambda r: r.metric) == sorted(
            rows, key=lambda r: r.metric
        )
        assert store.triad_scores_for([]) == []

    def test_scoring_again_replaces_the_old_value(self, store: ObserveStore) -> None:
        store.add_query(make_query(1))
        store.save_triad_scores([make_triad("q1", value=0.2)])

        store.save_triad_scores([make_triad("q1", value=0.9, minutes=5)])

        (row,) = store.triad_scores_for(["q1"])
        assert row.value == 0.9

    def test_unscored_lists_answered_questions_without_scores_newest_first(
        self, store: ObserveStore
    ) -> None:
        store.add_query(make_query(1))
        store.add_query(make_query(2, refusal=True))  # nothing to judge
        store.add_query(make_query(3, status="error", answer=None, error="LLM down"))
        store.add_query(make_query(4))
        store.add_query(make_query(5))
        store.save_triad_scores([make_triad("q4")])

        assert store.unscored_query_ids() == ["q5", "q1"]
        assert store.unscored_query_ids(limit=1) == ["q5"]

    def test_metric_names_are_the_rag_triad(self) -> None:
        assert TRIAD_METRICS == ("context_relevance", "groundedness", "answer_relevance")

    def test_a_read_only_store_cannot_save_scores(self, tmp_path: Path) -> None:
        ObserveStore(tmp_path / "rag.db").add_query(make_query(1))

        with pytest.raises(ObserveStoreError):
            ObserveStore(tmp_path / "rag.db", read_only=True).save_triad_scores([make_triad("q1")])


def test_a_v2_file_upgrades_to_v3_and_keeps_its_rows(tmp_path: Path) -> None:
    path = tmp_path / "rag.db"
    ObserveStore(path).add_query(make_query(1))
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE trulens_scores")
        conn.execute("PRAGMA user_version = 2")

    store = ObserveStore(path)

    assert store.get_query("q1") is not None
    store.save_triad_scores([make_triad("q1")])
    assert len(store.triad_scores_for(["q1"])) == 1
    with sqlite3.connect(path) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION == 3
