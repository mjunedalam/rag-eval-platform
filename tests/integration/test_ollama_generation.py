"""Integration tests against a real local LLM served by Ollama.

Needs `ollama serve` with the configured model pulled (default: `ollama pull qwen3:8b`)
and `uv sync --extra openai`. Skipped otherwise, e.g. in CI.
"""

import json
import urllib.request

import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import Generator, OpenAICompatibleClient
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def generator() -> Generator:
    pytest.importorskip("openai")
    settings = Settings(llm_provider="ollama")
    tags_url = settings.ollama_base_url.removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=3) as response:  # noqa: S310 - local URL
            models = {m["name"] for m in json.load(response)["models"]}
    except OSError:
        pytest.skip("Ollama is not running")
    if settings.llm_model not in models:
        pytest.skip(f"model {settings.llm_model} is not pulled")
    return Generator(OpenAICompatibleClient.from_settings(settings))


def source(doc_id: str, text: str) -> SearchResult:
    return SearchResult(
        Chunk(id=f"{doc_id}#0", doc_id=doc_id, index=0, text=text, start_index=0), 0.5
    )


def test_answers_from_the_source_and_cites_it(generator: Generator) -> None:
    sources = [
        source("weather.md", "Rain is common in autumn."),
        source("metrics.md", "MRR is the mean over queries of 1 divided by the rank of the "
               "first relevant document."),
    ]  # fmt: skip

    answer = generator.generate("What does MRR measure?", sources)

    assert not answer.is_refusal
    assert [c.source.chunk.doc_id for c in answer.citations] == ["metrics.md"]
    assert answer.invalid_citations == ()


def test_refuses_when_sources_do_not_contain_the_answer(generator: Generator) -> None:
    answer = generator.generate(
        "Who won the 2018 football World Cup?", [source("sky.md", "The sky is blue.")]
    )

    assert answer.is_refusal


def test_streams_the_answer_in_pieces(generator: Generator) -> None:
    sources = [
        source("metrics.md", "MRR is the mean over queries of 1 divided by the rank of the "
               "first relevant document."),
    ]  # fmt: skip
    stream = generator.stream("What does MRR measure?", sources)

    pieces = list(stream)

    assert len(pieces) > 3  # arrived gradually, not as one block
    assert "".join(pieces) == stream.answer.text
    assert "<think>" not in stream.answer.text
    assert [c.source.chunk.doc_id for c in stream.answer.citations] == ["metrics.md"]
    assert stream.answer.output_tokens is not None
