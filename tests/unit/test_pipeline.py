"""Tests for the end-to-end pipeline and the ask command (all fakes)."""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from rag_eval_platform import ask as ask_module
from rag_eval_platform import pipeline as pipeline_module
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.generation.generator import Completion, GenerationError, Generator
from rag_eval_platform.generation.prompt_templates import Message
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.pipeline import RagPipeline, create_pipeline
from rag_eval_platform.retrieval.vector_store import SearchResult

RESULTS = [
    SearchResult(
        Chunk(id="metrics.md#0", doc_id="metrics.md", index=0, text="MRR...", start_index=0),
        0.83,
    ),
    SearchResult(
        Chunk(id="book.pdf#4", doc_id="book.pdf", index=4, text="F=ma", start_index=9, page=3),
        0.41,
    ),
]


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[SearchResult]:
        self.queries.append(query)
        return RESULTS


class FakeLlm:
    model = "fake-8b"

    def __init__(self, reply: str) -> None:
        self.reply = reply

    def complete(self, messages: Sequence[Message]) -> Completion:
        return Completion(text=self.reply, input_tokens=300, output_tokens=20)

    def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
        yield Completion(text=self.reply, input_tokens=None, output_tokens=None)
        yield Completion(text="", input_tokens=300, output_tokens=20)


def make_pipeline(reply: str = "MRR is the mean reciprocal rank [1].") -> RagPipeline:
    return RagPipeline(FakeRetriever(), Generator(FakeLlm(reply)))


def test_ask_retrieves_then_generates_from_those_results() -> None:
    retriever = FakeRetriever()
    pipeline = RagPipeline(retriever, Generator(FakeLlm("Answer [2].")))

    answer = pipeline.ask("What is F?")

    assert retriever.queries == ["What is F?"]
    assert answer.sources == tuple(RESULTS)
    assert [c.source.chunk.doc_id for c in answer.citations] == ["book.pdf"]


def test_ask_logs_one_structured_line(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    make_pipeline("Claim [1]. Bad [9].").ask("What is MRR?")

    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["message"] == "Question answered"
    assert record["retrieved_chunk_ids"] == ["metrics.md#0", "book.pdf#4"]
    assert record["cited_doc_ids"] == ["metrics.md"]
    assert record["invalid_citations"] == [9]
    assert record["model"] == "fake-8b"
    assert record["refusal"] is False
    assert record["total_ms"] >= record["generation_ms"] >= 0


def test_ask_stream_yields_text_and_logs_once_finished(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging("INFO")
    stream = make_pipeline("Streamed answer [1].").ask_stream("What is MRR?")

    assert capsys.readouterr().err == ""  # nothing logged before the answer is complete
    assert "".join(stream) == "Streamed answer [1]."

    record = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert record["message"] == "Question answered"
    assert record["cited_doc_ids"] == ["metrics.md"]
    assert record["output_tokens"] == 20
    assert stream.answer.text == "Streamed answer [1]."


def test_create_pipeline_builds_retriever_and_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever, generator = FakeRetriever(), Generator(FakeLlm("x"))
    monkeypatch.setattr(pipeline_module, "create_retriever", lambda s: retriever)
    monkeypatch.setattr(pipeline_module, "create_generator", lambda s: generator)

    pipeline = create_pipeline(Settings())

    assert pipeline.retriever is retriever
    assert pipeline.generator is generator


class TestAskCommand:
    @pytest.fixture(autouse=True)
    def isolated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        get_settings.cache_clear()

    def test_prints_answer_sources_and_model(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(ask_module, "create_pipeline", lambda s: make_pipeline())

        assert ask_module.main(["What", "is", "MRR?"]) == 0

        out = capsys.readouterr().out
        assert "MRR is the mean reciprocal rank [1]." in out
        assert "[1] metrics.md" in out
        assert "[2] book.pdf, page 3" in out
        assert "cited" in out
        assert "fake-8b" in out

    def test_warns_about_invalid_citations(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(ask_module, "create_pipeline", lambda s: make_pipeline("X [5]."))

        ask_module.main(["q?"])

        assert "not among the sources: [5]" in capsys.readouterr().out

    def test_pipeline_error_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def down(settings: object) -> None:
            raise GenerationError("LLM request failed. Is Ollama running?")

        monkeypatch.setattr(ask_module, "create_pipeline", down)

        assert ask_module.main(["q?"]) == 1
        assert "Is Ollama running?" in capsys.readouterr().err

    def test_streams_answer_before_listing_sources(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(ask_module, "create_pipeline", lambda s: make_pipeline("Streamed [1]."))

        assert ask_module.main(["q?"]) == 0

        out = capsys.readouterr().out
        assert out.startswith("Streamed [1].\n")
        assert out.index("Streamed [1].") < out.index("Sources:")

    def test_failure_while_streaming_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        class DroppingLlm(FakeLlm):
            def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
                yield Completion(text="Partial ", input_tokens=None, output_tokens=None)
                raise GenerationError("connection dropped. Is Ollama running?")

        pipeline = RagPipeline(FakeRetriever(), Generator(DroppingLlm("x")))
        monkeypatch.setattr(ask_module, "create_pipeline", lambda s: pipeline)

        assert ask_module.main(["q?"]) == 1
        assert "connection dropped" in capsys.readouterr().err


def test_ask_timed_reports_retrieval_and_total_time() -> None:
    timed = make_pipeline().ask_timed("What is MRR?")

    assert timed.answer.text == "MRR is the mean reciprocal rank [1]."
    assert 0 <= timed.retrieve_ms <= timed.total_ms
