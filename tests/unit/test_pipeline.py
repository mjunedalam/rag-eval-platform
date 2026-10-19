"""Tests for the end-to-end pipeline and the ask command (all fakes)."""

import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from tests.unit.fake_tracer import FakeTracer

from rag_eval_platform import ask as ask_module
from rag_eval_platform import pipeline as pipeline_module
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.generation.generator import (
    Answer,
    AnswerStream,
    Completion,
    GenerationError,
    Generator,
)
from rag_eval_platform.generation.prompt_templates import Message
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.observability.tracing import LlmTraceRef
from rag_eval_platform.pipeline import RagPipeline, create_pipeline
from rag_eval_platform.retrieval.vector_store import SearchResult, VectorStoreError

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
        monkeypatch.setattr(ask_module, "create_pipeline", lambda s, **kw: make_pipeline())

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
        monkeypatch.setattr(ask_module, "create_pipeline", lambda s, **kw: make_pipeline("X [5]."))

        ask_module.main(["q?"])

        assert "not among the sources: [5]" in capsys.readouterr().out

    def test_pipeline_error_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        def down(settings: object, **kwargs: object) -> None:
            raise GenerationError("LLM request failed. Is Ollama running?")

        monkeypatch.setattr(ask_module, "create_pipeline", down)

        assert ask_module.main(["q?"]) == 1
        assert "Is Ollama running?" in capsys.readouterr().err

    def test_streams_answer_before_listing_sources(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(
            ask_module, "create_pipeline", lambda s, **kw: make_pipeline("Streamed [1].")
        )

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
        monkeypatch.setattr(ask_module, "create_pipeline", lambda s, **kw: pipeline)

        assert ask_module.main(["q?"]) == 1
        assert "connection dropped" in capsys.readouterr().err


def test_ask_timed_reports_retrieval_and_total_time() -> None:
    timed = make_pipeline().ask_timed("What is MRR?")

    assert timed.answer.text == "MRR is the mean reciprocal rank [1]."
    assert 0 <= timed.retrieve_ms <= timed.total_ms


class FakeRecorder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.rows: list[tuple[str, object]] = []
        self.ids: list[str | None] = []

    def record(self, answer: Answer, *, retrieve_ms: float, total_ms: float,
               query_id: str | None = None) -> str | None:  # fmt: skip
        if self.fail:
            raise RuntimeError("store down")
        self.rows.append(("ok", answer.text))
        self.ids.append(query_id)
        return query_id or f"id-{len(self.rows)}"

    def record_error(self, question: str, error: BaseException, *, total_ms: float,
                     query_id: str | None = None) -> str | None:  # fmt: skip
        self.rows.append(("error", type(error).__name__))
        self.ids.append(query_id)
        return query_id or f"id-{len(self.rows)}"


class DownRetriever:
    def retrieve(self, query: str) -> list[SearchResult]:
        raise VectorStoreError("Chroma is down")


def test_ask_timed_records_the_answer_and_returns_its_id() -> None:
    recorder = FakeRecorder()
    pipeline = RagPipeline(FakeRetriever(), Generator(FakeLlm("Answer [1].")), recorder)

    timed = pipeline.ask_timed("q?")

    # The id is chosen when the question arrives (it also names the trace) and recorded.
    assert timed.query_id is not None
    assert recorder.ids == [timed.query_id]
    assert recorder.rows == [("ok", "Answer [1].")]


def test_a_failed_question_is_recorded_then_re_raised() -> None:
    recorder = FakeRecorder()
    pipeline = RagPipeline(DownRetriever(), Generator(FakeLlm("x")), recorder)

    with pytest.raises(VectorStoreError):
        pipeline.ask_timed("q?")
    with pytest.raises(VectorStoreError):
        pipeline.ask_stream("q?")
    assert recorder.rows == [("error", "VectorStoreError"), ("error", "VectorStoreError")]


def test_a_stream_is_recorded_once_it_finishes() -> None:
    recorder = FakeRecorder()
    pipeline = RagPipeline(FakeRetriever(), Generator(FakeLlm("Streamed [1].")), recorder)

    stream = pipeline.ask_stream("q?")
    assert stream.query_id is None
    text = "".join(stream)

    assert text == "Streamed [1]."
    assert stream.query_id is not None
    assert recorder.ids == [stream.query_id]
    assert stream.answer.text == "Streamed [1]."


def test_a_stream_that_fails_is_recorded_as_an_error() -> None:
    class BrokenLlm(FakeLlm):
        def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
            yield Completion(text="Half", input_tokens=None, output_tokens=None)
            raise GenerationError("model crashed")

    recorder = FakeRecorder()
    pipeline = RagPipeline(FakeRetriever(), Generator(BrokenLlm("x")), recorder)

    with pytest.raises(GenerationError):
        list(pipeline.ask_stream("q?"))
    assert recorder.rows == [("error", "GenerationError")]


def test_a_failing_recorder_never_breaks_an_answer() -> None:
    pipeline = RagPipeline(
        FakeRetriever(), Generator(FakeLlm("Fine [1].")), FakeRecorder(fail=True)
    )

    timed = pipeline.ask_timed("q?")
    stream = pipeline.ask_stream("q?")

    assert (timed.answer.text, timed.query_id) == ("Fine [1].", None)
    assert "".join(stream) == "Fine [1]."
    assert stream.query_id is None


def test_create_pipeline_records_only_with_a_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "create_retriever", lambda settings: FakeRetriever())
    monkeypatch.setattr(pipeline_module, "create_generator",
                        lambda settings: Generator(FakeLlm("A [1].")))  # fmt: skip

    assert create_pipeline(Settings()).recorder is None
    recorded = create_pipeline(Settings(), source="cli")
    assert recorded.recorder is not None
    assert recorded.ask_timed("q?").query_id is not None


def test_a_blank_question_is_recorded_as_an_error_through_ask_stream() -> None:
    recorder = FakeRecorder()
    pipeline = RagPipeline(FakeRetriever(), Generator(FakeLlm("x")), recorder)

    with pytest.raises(ValueError, match="blank"):
        pipeline.ask_stream("   ")

    assert recorder.rows == [("error", "ValueError")]


class FailingGenerator:
    def generate(self, question: str, results: Sequence[SearchResult],
                 trace: LlmTraceRef | None = None) -> Answer:  # fmt: skip
        raise ValueError("boom")

    def stream(
        self,
        question: str,
        results: Sequence[SearchResult],
        on_complete: Callable[[Answer], None] | None = None,
        trace: LlmTraceRef | None = None,
    ) -> AnswerStream:
        raise NotImplementedError


def test_a_failing_generator_is_recorded_then_re_raised_through_ask_timed() -> None:
    recorder = FakeRecorder()
    pipeline = RagPipeline(FakeRetriever(), FailingGenerator(), recorder)

    with pytest.raises(ValueError, match="boom"):
        pipeline.ask_timed("q?")

    assert recorder.rows == [("error", "ValueError")]


class TraceableLlm(FakeLlm):
    def __init__(self, reply: str) -> None:
        super().__init__(reply)
        self.refs: list[LlmTraceRef] = []

    def traced(self, ref: LlmTraceRef) -> "TraceableLlm":
        self.refs.append(ref)
        return self


class TestPipelineTracing:
    def traced(self, llm: FakeLlm | None = None, retriever: object | None = None
               ) -> tuple[RagPipeline, FakeTracer, FakeRecorder]:  # fmt: skip
        tracer, recorder = FakeTracer(), FakeRecorder()
        pipeline = RagPipeline(
            retriever or FakeRetriever(), Generator(llm or FakeLlm("Answer [1].")), recorder,  # type: ignore[arg-type]
            tracer=tracer, tags=("api", "concise"), trace_metadata={"top_k": 2},
        )  # fmt: skip
        return pipeline, tracer, recorder

    def test_one_trace_per_question_sharing_the_query_id_with_the_log(self) -> None:
        pipeline, tracer, recorder = self.traced()

        timed = pipeline.ask_timed("What is MRR?")

        (trace,) = tracer.questions
        assert trace.question == "What is MRR?"
        assert trace.tags == ("api", "concise")
        assert trace.metadata == {"top_k": 2}
        assert recorder.ids == [trace.query_id]
        assert timed.query_id == trace.query_id

    def test_retrieval_is_a_stage_and_the_trace_ends_with_the_answer(self) -> None:
        pipeline, tracer, _ = self.traced()

        pipeline.ask_timed("What is MRR?")

        trace = tracer.questions[0]
        (stage,) = trace.stages
        assert stage.name == "retrieve-context"
        assert stage.input == "What is MRR?"
        assert [row["chunk_id"] for row in stage.output] == ["metrics.md#0", "book.pdf#4"]
        assert trace.finished
        assert trace.answer == "Answer [1]."
        assert trace.finish_metadata is not None
        assert trace.finish_metadata["refusal"] is False
        assert trace.finish_metadata["cited_doc_ids"] == ["metrics.md"]

    def test_the_llm_call_is_nested_in_the_same_trace(self) -> None:
        llm = TraceableLlm("Answer [1].")
        pipeline, tracer, _ = self.traced(llm)

        pipeline.ask_timed("q?")

        assert llm.refs == [tracer.questions[0].llm_ref]

    def test_a_failure_ends_the_trace_with_the_error_and_logs_the_same_id(self) -> None:
        pipeline, tracer, recorder = self.traced(retriever=DownRetriever())

        with pytest.raises(VectorStoreError):
            pipeline.ask_timed("q?")

        trace = tracer.questions[0]
        assert isinstance(trace.error, VectorStoreError)
        assert isinstance(trace.stages[0].error, VectorStoreError)
        assert recorder.ids == [trace.query_id]

    def test_a_stream_ends_its_trace_once_fully_read(self) -> None:
        llm = TraceableLlm("Streamed [1].")
        pipeline, tracer, recorder = self.traced(llm)

        stream = pipeline.ask_stream("q?")
        trace = tracer.questions[0]
        assert not trace.finished

        assert "".join(stream) == "Streamed [1]."
        assert trace.finished
        assert trace.answer == "Streamed [1]."
        assert llm.refs == [trace.llm_ref]
        assert recorder.ids == [trace.query_id]
        assert stream.query_id == trace.query_id

    def test_a_stream_that_breaks_ends_its_trace_with_the_error(self) -> None:
        class DroppingLlm(FakeLlm):
            def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
                yield Completion(text="Partial ", input_tokens=None, output_tokens=None)
                raise GenerationError("connection dropped")

        pipeline, tracer, _ = self.traced(DroppingLlm("x"))

        with pytest.raises(GenerationError):
            list(pipeline.ask_stream("q?"))

        assert isinstance(tracer.questions[0].error, GenerationError)


def test_create_pipeline_traces_with_the_source_and_style_as_tags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = FakeTracer()
    monkeypatch.setattr(pipeline_module, "create_retriever", lambda settings: FakeRetriever())
    monkeypatch.setattr(pipeline_module, "create_tracer", lambda settings: tracer)
    monkeypatch.setattr(pipeline_module, "create_generator",
                        lambda settings: Generator(FakeLlm("A [1].")))  # fmt: skip

    create_pipeline(Settings(), source="cli").ask_timed("q?")

    trace = tracer.questions[0]
    assert trace.tags == ("cli", "concise")
    assert trace.metadata is not None
    assert trace.metadata["collection"] == "rag_documents"
    assert trace.metadata["top_k"] == 5


class TestAskCommandFlushes:
    @pytest.fixture(autouse=True)
    def isolated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.chdir(tmp_path)
        get_settings.cache_clear()

    def test_traces_are_flushed_before_the_script_exits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracer = FakeTracer()
        monkeypatch.setattr(ask_module, "create_tracer", lambda settings: tracer)
        monkeypatch.setattr(ask_module, "create_pipeline",
                            lambda settings, source: make_pipeline())  # fmt: skip

        assert ask_module.main(["q?"]) == 0
        assert tracer.flushed == 1

    def test_traces_are_flushed_even_when_answering_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tracer = FakeTracer()

        def down(settings: object, source: object) -> None:
            raise GenerationError("LLM down")

        monkeypatch.setattr(ask_module, "create_tracer", lambda settings: tracer)
        monkeypatch.setattr(ask_module, "create_pipeline", down)

        assert ask_module.main(["q?"]) == 1
        assert tracer.flushed == 1
