"""Integration tests for the playground: the page renders, and the full upload-to-answer
flow works with real Chroma, a real embedding model and a real local LLM.

Each test skips when its tools are missing (Streamlit, Chroma, Ollama or models).
"""

import json
import urllib.request
from pathlib import Path

import pytest
from tests.pdf_builder import build_pdf

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Generator, OpenAICompatibleClient
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
    ]


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
