"""Tests for the pure logic in playground.tabs.observe (the ⑧ Observe tab)."""

import pytest

pytest.importorskip("streamlit")  # the tab module imports Streamlit; CI does not install it

from rag_eval_platform.playground.tabs.observe import _selected_query_id


def test_selected_query_id_returns_the_row_at_the_selected_index() -> None:
    rows: list[dict[str, object]] = [{"id": "q1"}, {"id": "q2"}]

    assert _selected_query_id(rows, [1]) == "q2"


def test_selected_query_id_is_none_without_a_selection() -> None:
    rows: list[dict[str, object]] = [{"id": "q1"}]

    assert _selected_query_id(rows, []) is None


def test_selected_query_id_ignores_a_stale_selection_past_the_new_rows() -> None:
    """The dataframe key changes with the range/source filter, but on the rerun where the
    filter changes and the widget's selection has not yet reset, a leftover index from the
    longer previous table must not be used to index the shorter new one."""
    rows: list[dict[str, object]] = [{"id": "q1"}]

    assert _selected_query_id(rows, [5]) is None
