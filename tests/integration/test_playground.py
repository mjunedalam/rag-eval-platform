"""Integration tests for the playground: the page renders, and the full upload-to-answer
flow works with real Chroma, a real embedding model and a real local LLM.

Each test skips when its tools are missing (Streamlit, Chroma, Ollama or models).
"""

import json
import urllib.request
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from tests.pdf_builder import build_pdf
from tests.unit.observe_records import make_feedback, make_query, make_run

from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.generation.generator import Generator, OpenAICompatibleClient
from rag_eval_platform.observability.store import ObserveStore
from rag_eval_platform.playground.core import build_index, run_query, save_uploads

pytestmark = pytest.mark.integration

APP = Path(__file__).resolve().parents[2] / "src" / "rag_eval_platform" / "playground" / "app.py"


def test_app_renders_without_errors() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")

    app = testing.AppTest.from_file(str(APP), default_timeout=60).run()

    assert not app.exception
    assert app.title[0].value == "RAG Playground"
    assert [t.label for t in app.tabs] == [
        "Overview",
        "① Ingest",
        "② Embed",
        "③ Retrieve",
        "④ Generate",
        "⑤ Evaluate",
        "⑥ Gate",
        "⑦ API",
        "⑧ Observe",
    ]


def _reload_observe_script() -> None:
    """Rebuild ``observe.render`` inside the AppTest script.

    ``playground.shared`` reads ``settings = get_settings()`` once at import time, and
    ``observe.py`` imports that name, so a stale module (already imported by an earlier test
    with a different ``RAG_OBSERVE_DB``) would keep pointing at the wrong database. Reloading
    both, after clearing the settings cache, picks up the env var this test just set.
    """
    import importlib

    from rag_eval_platform.config.settings import get_settings

    get_settings.cache_clear()
    from rag_eval_platform.playground import shared

    importlib.reload(shared)
    from rag_eval_platform.playground.tabs import observe

    importlib.reload(observe)
    observe.render(None)  # type: ignore[arg-type] # unused by render; avoids importing TabContext


def test_observe_tab_with_no_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("RAG_OBSERVE_DB", str(tmp_path / "rag.db"))
    get_settings.cache_clear()

    app = testing.AppTest.from_function(_reload_observe_script, default_timeout=60).run()

    assert not app.exception
    infos = [info.value for info in app.info]
    assert any("No queries yet — ask one in the chat or the API." in text for text in infos)
    assert any("Drift checks need" in text for text in infos)
    metrics = {metric.label: metric.value for metric in app.get("metric")}
    assert len(metrics) == 8
    assert metrics["Cost"] == "local · $0"


def test_observe_tab_with_seeded_queries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    db_path = tmp_path / "rag.db"
    monkeypatch.setenv("RAG_OBSERVE_DB", str(db_path))
    get_settings.cache_clear()

    now = datetime.now(UTC)
    queries = [
        make_query(n, ts=now - timedelta(minutes=(44 - n) * 5), total_ms=1000.0 + n * 10.0)
        for n in range(45)
    ]  # later queries (higher n, more recent) are a little slower
    queries[10] = replace(
        queries[10], status="error", total_ms=None, input_tokens=None, output_tokens=None,
        retrieve_ms=None, generate_ms=None, cost_usd=0.0,
    )  # fmt: skip
    store = ObserveStore(db_path)
    for query in queries:
        store.add_query(query)
    store.add_feedback(make_feedback("q5", 0, comment="Too vague."))
    store.add_eval_run(make_run("gate", False, {"recall": 0.5, "mrr": 0.4}))

    app = testing.AppTest.from_function(_reload_observe_script, default_timeout=60).run()

    assert not app.exception
    warnings = [warning.value for warning in app.warning]
    assert any("The latest gate run failed" in text for text in warnings)
    buttons = [button.label for button in app.button]
    assert any(label == "Mark reviewed" for label in buttons)
    metrics = {metric.label: metric.value for metric in app.get("metric")}
    assert metrics["Queries"] == f"{len(queries):,}"


def test_uploaded_pdf_can_be_indexed_and_answered_with_page_citation(tmp_path: Path) -> None:
    pytest.importorskip("sentence_transformers")
    pytest.importorskip("openai")
    import chromadb

    from rag_eval_platform.ingestion.embedding import SentenceTransformerEmbedder
    from rag_eval_platform.retrieval.vector_store import ChromaVectorStore

    settings = Settings()
    try:
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        tags = settings.ollama_base_url.removesuffix("/v1") + "/api/tags"
        with urllib.request.urlopen(tags, timeout=3) as response:  # noqa: S310 - local URL
            models = {m["name"] for m in json.load(response)["models"]}
    except Exception:
        pytest.skip("Chroma or Ollama is not running")
    if settings.llm_model not in models:
        pytest.skip(f"model {settings.llm_model} is not pulled")

    pdf = build_pdf(
        [
            ["Chapter 1: Motion", "Velocity is displacement divided by time."],
            ["Chapter 2: Forces", "Newton's second law states that force equals mass times",
             "acceleration, written F = m a."],
            ["Chapter 3: Energy", "Kinetic energy equals one half m v squared."],
        ]
    )  # fmt: skip
    save_uploads([("physics.pdf", pdf)], tmp_path)
    store = ChromaVectorStore(client, collection_name="test_playground_flow")
    embedder = SentenceTransformerEmbedder.from_pretrained(settings.embedding_model)
    try:
        summary = build_index(
            tmp_path, strategy="recursive", chunk_size=120, chunk_overlap=0,
            embedder=embedder, store=store,
        )  # fmt: skip
        trace = run_query(
            "What does Newton's second law state?",
            embedder=embedder,
            store=store,
            generator=Generator(OpenAICompatibleClient.from_settings(settings)),
            top_k=2,
        )
    finally:
        client.delete_collection("test_playground_flow")

    assert summary.page_count == 3
    assert trace.final_results[0].chunk.page == 2
    assert not trace.answer.is_refusal
    assert {c.source.chunk.page for c in trace.answer.citations} == {2}
