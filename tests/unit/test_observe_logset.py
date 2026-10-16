"""Tests for observability.logset.LogSet: one writable host store plus read-only logs,
merged for reading (the ⑧ Observe tab's view across the host's rag.db and the Docker API's
own rag-api.db, which the tab can never write through Docker Desktop's bind mount).
"""

import logging
from pathlib import Path

import pytest
from tests.unit.observe_records import T0, make_feedback, make_query, make_run

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.logger import create_logset
from rag_eval_platform.observability.logset import LogSet
from rag_eval_platform.observability.store import ObserveStore, ObserveStoreError


def test_queries_merges_and_sorts_across_stores(tmp_path: Path) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    readonly = ObserveStore(tmp_path / "rag-api.db")  # written as a normal store, read read-only
    host.add_query(make_query(1))
    host.add_query(make_query(3))
    readonly.add_query(make_query(2))
    logset = LogSet(host, (ObserveStore(tmp_path / "rag-api.db", read_only=True),))

    assert [q.id for q in logset.queries()] == ["q1", "q2", "q3"]


def test_recent_queries_keeps_the_newest_limit_overall_oldest_first(tmp_path: Path) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    api_writer = ObserveStore(tmp_path / "rag-api.db")
    for n in (0, 2, 4):
        host.add_query(make_query(n))
    for n in (1, 3, 5):
        api_writer.add_query(make_query(n))
    logset = LogSet(host, (ObserveStore(tmp_path / "rag-api.db", read_only=True),))

    assert [q.id for q in logset.recent_queries(3)] == ["q3", "q4", "q5"]
    assert [q.id for q in logset.recent_queries(100)] == [f"q{n}" for n in range(6)]
    assert logset.recent_queries(0) == []


def test_count_queries_sums_every_store(tmp_path: Path) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    api_writer = ObserveStore(tmp_path / "rag-api.db")
    host.add_query(make_query(1))
    api_writer.add_query(make_query(2))
    api_writer.add_query(make_query(3))
    logset = LogSet(host, (ObserveStore(tmp_path / "rag-api.db", read_only=True),))

    assert logset.count_queries() == 3


def test_get_query_finds_a_row_in_any_store(tmp_path: Path) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    api_writer = ObserveStore(tmp_path / "rag-api.db")
    host.add_query(make_query(1))
    api_writer.add_query(make_query(2))
    logset = LogSet(host, (ObserveStore(tmp_path / "rag-api.db", read_only=True),))

    assert logset.get_query("q1") is not None
    assert logset.get_query("q2") is not None
    assert logset.get_query("missing") is None


def test_reviews_from_the_host_store_are_applied_to_every_query(tmp_path: Path) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    api_writer = ObserveStore(tmp_path / "rag-api.db")
    host.add_query(make_query(1))
    api_writer.add_query(make_query(2))  # only exists in the read-only file
    host.mark_reviewed("q2", T0)  # marked from the host, per the new design
    logset = LogSet(host, (ObserveStore(tmp_path / "rag-api.db", read_only=True),))

    by_id = {q.id: q for q in logset.queries()}

    assert by_id["q2"].reviewed_at == T0
    assert by_id["q1"].reviewed_at is None


def test_a_querys_own_reviewed_at_wins_over_the_hosts_reviews_table(tmp_path: Path) -> None:
    """A query already carrying reviewed_at (e.g. upgraded from a v1 file) is not overwritten
    by an absent host review."""
    host = ObserveStore(tmp_path / "rag.db")
    host.add_query(make_query(1, reviewed_at=T0))
    logset = LogSet(host)

    [query] = logset.queries()

    assert query.reviewed_at == T0


def test_mark_reviewed_always_writes_to_the_host_store(tmp_path: Path) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    readonly_store = ObserveStore(tmp_path / "rag-api.db", read_only=True)
    logset = LogSet(host, (readonly_store,))

    assert logset.mark_reviewed("some-query-id", T0) is True
    assert host.reviews() == {"some-query-id": T0}


def test_a_broken_readonly_store_is_skipped_with_one_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    host.add_query(make_query(1))
    broken_path = tmp_path / "rag-api.db"
    broken_path.write_bytes(b"not a database" * 100)
    logset = LogSet(host, (ObserveStore(broken_path, read_only=True),))

    with caplog.at_level(logging.WARNING):
        result = logset.queries()

    assert [q.id for q in result] == ["q1"]
    assert "read-only" in caplog.text.lower() or "observability" in caplog.text.lower()


def test_the_hosts_own_errors_propagate(tmp_path: Path) -> None:
    broken_path = tmp_path / "rag.db"
    broken_path.write_bytes(b"not a database" * 100)
    logset = LogSet(ObserveStore(broken_path))

    with pytest.raises(ObserveStoreError):
        logset.queries()


def test_feedback_for_merges_and_sorts_across_stores(tmp_path: Path) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    api_writer = ObserveStore(tmp_path / "rag-api.db")
    host.add_query(make_query(1))
    api_writer.add_query(make_query(2))
    host.add_feedback(make_feedback("q1", 1, minutes=2))
    api_writer.add_feedback(make_feedback("q2", 0, minutes=1))
    logset = LogSet(host, (ObserveStore(tmp_path / "rag-api.db", read_only=True),))

    result = logset.feedback_for(["q1", "q2"])

    assert [f.query_id for f in result] == ["q2", "q1"]


def test_eval_runs_merges_and_sorts_across_stores(tmp_path: Path) -> None:
    host = ObserveStore(tmp_path / "rag.db")
    api_writer = ObserveStore(tmp_path / "rag-api.db")
    host.add_eval_run(make_run("gate", True, {"recall": 0.9}, minutes=2))
    api_writer.add_eval_run(make_run("retrieval", True, {"recall": 0.8}, minutes=1))
    logset = LogSet(host, (ObserveStore(tmp_path / "rag-api.db", read_only=True),))

    assert [r.kind for r in logset.eval_runs()] == ["retrieval", "gate"]


def test_create_logset_is_none_when_observability_is_off(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, observe_enabled=False, observe_db=tmp_path / "rag.db")

    assert create_logset(settings) is None


def test_create_logset_includes_the_configured_readonly_stores(tmp_path: Path) -> None:
    api_path = tmp_path / "rag-api.db"
    ObserveStore(api_path).add_query(make_query(1))
    settings = Settings(
        _env_file=None, observe_db=tmp_path / "rag.db", observe_readonly_dbs=[api_path]
    )

    logset = create_logset(settings)

    assert logset is not None
    assert [q.id for q in logset.queries()] == ["q1"]


def test_create_logset_skips_a_readonly_path_equal_to_the_host_db(tmp_path: Path) -> None:
    same_path = tmp_path / "rag.db"
    settings = Settings(_env_file=None, observe_db=same_path, observe_readonly_dbs=[same_path])

    logset = create_logset(settings)

    assert logset is not None
    assert logset.readonly == ()
