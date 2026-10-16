"""Tests for playground.tabs.overview's live run: a retrieval failure must be logged.

``_run_live`` is Streamlit-heavy (placeholders, session state, timing), but every ``st.*``
call it makes works in "bare mode" (no running app), just with a warning printed to stderr.
So the retrieve-step exception path can be exercised directly, with the drawing surfaces
(``pipeline``, ``tiles``, ``slots``) replaced by plain mocks and the heavy model loaders
replaced by fakes.
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# The tab draws with Streamlit and Plotly; CI does not install the UI group.
pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Completion, OpenAICompatibleClient
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.observability.store import ObserveStore
from rag_eval_platform.playground import core
from rag_eval_platform.playground.chat import Turn
from rag_eval_platform.playground.shared import HISTORY, Options, TabContext
from rag_eval_platform.playground.tabs import overview
from rag_eval_platform.retrieval.vector_store import SearchResult, VectorStoreError


def _options() -> Options:
    return Options(
        strategy="recursive", chunk_size=200, chunk_overlap=0, top_k=2, rerank=False,
        rerank_candidates=10, model="qwen3:8b", reasoning_effort="none", temperature=0.0,
        slow_motion=False, answer_style="concise",
    )  # fmt: skip


def _snapshot() -> core.IndexSnapshot:
    chunks = (
        Chunk(id="a.md#0", doc_id="a.md", index=0, text="chunk one", start_index=0),
        Chunk(id="a.md#1", doc_id="a.md", index=1, text="chunk two", start_index=10),
    )
    return core.IndexSnapshot(chunks=chunks, vectors=((0.1, 0.2), (0.3, 0.4)))


def test_a_retrieval_failure_in_the_live_run_is_logged_before_it_is_shown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh_settings = Settings(_env_file=None, observe_db=tmp_path / "rag.db")
    monkeypatch.setattr(overview, "settings", fresh_settings)
    monkeypatch.setattr(overview, "load_embedder", lambda model_name: object())

    def broken_retrieve_step(*args: object, **kwargs: object) -> None:
        raise VectorStoreError("Chroma is down")

    monkeypatch.setattr(overview, "retrieve_step", broken_retrieve_step)

    ctx = TabContext(
        options=_options(), store=MagicMock(), models=None, snapshot=_snapshot(), trace=None,
        judged=None, bubble=None,
    )  # fmt: skip
    slots = [MagicMock() for _ in range(6)]

    overview._run_live("What is F?", ctx, overview.Dashboard(MagicMock(), MagicMock(), slots))

    [saved] = ObserveStore(fresh_settings.observe_db).queries()
    assert saved.status == "error"
    assert "Chroma is down" in (saved.error or "")


class _FakeLlm:
    model = "fake-8b"

    def complete(self, messages: object) -> Completion:
        return Completion(text="Chunk one says so [1].", input_tokens=30, output_tokens=6)

    def stream(self, messages: object) -> Iterator[Completion]:
        yield Completion(text="Chunk one says so [1].", input_tokens=None, output_tokens=None)
        yield Completion(text="", input_tokens=30, output_tokens=6)


def test_a_question_can_run_from_the_chat_without_touching_the_dashboard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh_settings = Settings(_env_file=None, observe_db=tmp_path / "rag.db")
    snapshot = _snapshot()
    step = core.RetrievalStep(
        vector_results=(SearchResult(snapshot.chunks[0], 0.9),),
        final_results=(SearchResult(snapshot.chunks[0], 0.9),),
        retrieval_ms=12.0, query_embedding=(0.1, 0.2),
    )  # fmt: skip
    monkeypatch.setattr(overview, "settings", fresh_settings)
    monkeypatch.setattr(overview, "load_embedder", lambda model_name: object())
    monkeypatch.setattr(overview, "retrieve_step", lambda *args, **kwargs: step)
    monkeypatch.setattr(OpenAICompatibleClient, "from_settings",
                        staticmethod(lambda settings: _FakeLlm()))  # fmt: skip

    def no_dashboard(*args: object, **kwargs: object) -> None:
        raise AssertionError("the dashboard must not be drawn from another tab")

    monkeypatch.setattr(overview, "animated_chart", no_dashboard)
    monkeypatch.setattr("streamlit.rerun", lambda: None)  # the run ends by redrawing the page
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    st_state: dict[str, object] = {}
    monkeypatch.setattr("streamlit.session_state", st_state)
    bubble = MagicMock()
    ctx = TabContext(
        options=_options(), store=MagicMock(), models=None, snapshot=snapshot, trace=None,
        judged=None, bubble=bubble,
    )  # fmt: skip

    overview._run_live("What does chunk one say?", ctx, None)

    history = st_state[HISTORY]
    assert isinstance(history, tuple)
    [turn] = history
    assert isinstance(turn, Turn)
    assert turn.trace.answer.text == "Chunk one says so [1]."
    assert turn.query_id is not None
    assert bubble.text.markdown.called  # the reply streamed into the chat
    [saved] = ObserveStore(fresh_settings.observe_db).queries()
    assert saved.source == "playground"
