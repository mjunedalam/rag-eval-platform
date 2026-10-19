"""Tests for observability.tracing (the interface) and observability.langfuse_tracer.

A fake Langfuse client records every call, so nothing is ever sent to Langfuse.
"""

import contextlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from rag_eval_platform import _optional
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.observability import langfuse_tracer
from rag_eval_platform.observability.langfuse_tracer import LangfuseTracer, create_tracer
from rag_eval_platform.observability.tracing import (
    NOOP_TRACER,
    LlmTraceRef,
    chunk_rows,
)
from rag_eval_platform.retrieval.vector_store import SearchResult

PRIVATE_VALUE = "fake-secret-value"  # an obvious placeholder, never a real key


@dataclass
class FakeObservation:
    id: str
    kwargs: dict[str, Any]
    updates: list[dict[str, Any]] = field(default_factory=list)
    children: list["FakeObservation"] = field(default_factory=list)
    ended: bool = False
    fail_on: str | None = None

    def start_observation(self, **kwargs: Any) -> "FakeObservation":
        if self.fail_on == "child":
            raise RuntimeError("child failed")
        child = FakeObservation(id=f"{self.id}-c{len(self.children)}", kwargs=kwargs)
        self.children.append(child)
        return child

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def end(self) -> None:
        self.ended = True


@dataclass
class FakeClient:
    roots: list[FakeObservation] = field(default_factory=list)
    scores: list[dict[str, Any]] = field(default_factory=list)
    flushed: int = 0
    fail: bool = False

    @staticmethod
    def create_trace_id(*, seed: str | None = None) -> str:
        return f"trace-of-{seed}"

    def start_observation(self, **kwargs: Any) -> FakeObservation:
        if self.fail:
            raise RuntimeError("network down")
        root = FakeObservation(id=f"root{len(self.roots)}", kwargs=kwargs)
        self.roots.append(root)
        return root

    def create_score(self, **kwargs: Any) -> None:
        if self.fail:
            raise RuntimeError("network down")
        self.scores.append(kwargs)

    def get_trace_url(self, *, trace_id: str | None = None) -> str:
        return f"https://lf.example/trace/{trace_id}"

    def flush(self) -> None:
        self.flushed += 1


@dataclass
class FakePropagate:
    calls: list[dict[str, Any]] = field(default_factory=list)

    @contextlib.contextmanager
    def __call__(self, **kwargs: Any) -> Iterator[None]:
        self.calls.append(kwargs)
        yield


@dataclass
class FakeSeed:
    """Records which trace id was seeded around each root's creation."""

    seeded: list[str] = field(default_factory=list)
    active: str | None = None

    @contextlib.contextmanager
    def __call__(self, trace_id: str) -> Iterator[None]:
        self.seeded.append(trace_id)
        self.active = trace_id
        yield
        self.active = None


def tracer(
    client: FakeClient | None = None, *, store_text: bool = True
) -> tuple[LangfuseTracer, FakeClient, FakePropagate]:
    client = client or FakeClient()
    propagate = FakePropagate()
    lf = LangfuseTracer(client, public_key="fake-public-value", propagate=propagate,
                        seed_trace=FakeSeed(), store_text=store_text)  # fmt: skip
    return lf, client, propagate


def result(n: int, page: int | None = None) -> SearchResult:
    chunk = Chunk(
        id=f"d.md#{n}", doc_id="d.md", index=n, text=f"text {n}", start_index=0, page=page
    )
    return SearchResult(chunk=chunk, score=0.9 - n / 10)


class TestNoopTracer:
    def test_does_nothing_and_never_raises(self) -> None:
        trace = NOOP_TRACER.start("q1", "What is MRR?", tags=["cli"], metadata={"k": 1})
        stage = trace.stage("retrieve-context", input={"question": "x"})
        stage.end(output=[], error=RuntimeError("x"))
        trace.finish(answer="a")
        NOOP_TRACER.score("q1", "user_feedback", 1, data_type="BOOLEAN")
        NOOP_TRACER.flush()

        assert NOOP_TRACER.enabled is False
        assert trace.llm_ref is None
        assert NOOP_TRACER.trace_url("q1") is None


def test_chunk_rows_keep_what_a_reviewer_needs() -> None:
    assert chunk_rows([result(0, page=3)]) == [
        {"chunk_id": "d.md#0", "doc_id": "d.md", "page": 3, "score": 0.9, "text": "text 0"}
    ]


class TestLangfuseTracer:
    def test_question_becomes_a_root_observation_with_a_seeded_trace_id(self) -> None:
        lf, client, propagate = tracer()

        lf.start("q1", "What is MRR?", tags=["api", "concise"], metadata={"top_k": 5})

        (root,) = client.roots
        # A genuine root: no trace_context (that would invent a parent span); the seeded id
        # comes from the id generator instead.
        assert "trace_context" not in root.kwargs
        assert root.kwargs["name"] == "answer-question"
        assert root.kwargs["as_type"] == "chain"
        assert root.kwargs["input"] == "What is MRR?"
        assert root.kwargs["metadata"] == {"top_k": 5}
        assert propagate.calls == [{"tags": ["api", "concise"]}]

    def test_stages_are_retriever_children_that_end_with_their_output(self) -> None:
        lf, client, _ = tracer()
        trace = lf.start("q1", "q?")

        stage = trace.stage("retrieve-context", input={"question": "q?"}, metadata={"top_k": 2})
        stage.end(output=chunk_rows([result(0)]), metadata={"returned": 1})

        (child,) = client.roots[0].children
        assert child.kwargs == {
            "name": "retrieve-context", "as_type": "retriever",
            "input": {"question": "q?"}, "metadata": {"top_k": 2},
        }  # fmt: skip
        assert child.updates == [{"output": chunk_rows([result(0)]), "metadata": {"returned": 1}}]
        assert child.ended

    def test_a_failed_stage_sends_only_the_error_type(self) -> None:
        lf, client, _ = tracer()
        stage = lf.start("q1", "q?").stage("rerank-chunks")

        stage.end(error=ValueError("bad candidates"))

        (child,) = client.roots[0].children
        # The message can name internal hosts or quote text; it stays in the local log.
        assert child.updates == [{"level": "ERROR", "status_message": "ValueError"}]
        assert child.ended

    def test_finish_sets_the_trace_output_and_ends_the_root(self) -> None:
        lf, client, _ = tracer()
        trace = lf.start("q1", "q?")

        trace.finish(answer="MRR is ... [1].", metadata={"refusal": False})

        root = client.roots[0]
        assert root.updates == [{"output": "MRR is ... [1].", "metadata": {"refusal": False}}]
        assert root.ended

    def test_finish_with_an_error_marks_the_trace_with_only_the_error_type(self) -> None:
        lf, client, _ = tracer()

        lf.start("q1", "q?").finish(error=RuntimeError("LLM down"))

        assert client.roots[0].updates == [{"level": "ERROR", "status_message": "RuntimeError"}]

    def test_llm_ref_points_the_generation_at_this_trace_and_root(self) -> None:
        lf, _, _ = tracer()

        ref = lf.start("q1", "q?").llm_ref

        assert ref == LlmTraceRef(
            trace_id="trace-of-q1", parent_observation_id="root0", public_key="fake-public-value"
        )

    def test_scores_attach_to_the_trace_seeded_from_the_query_id(self) -> None:
        lf, client, _ = tracer()

        lf.score("q1", "user_feedback", 1, comment="great", data_type="BOOLEAN")

        assert client.scores == [{
            "name": "user_feedback", "value": 1, "trace_id": "trace-of-q1",
            "data_type": "BOOLEAN", "comment": "great",
        }]  # fmt: skip

    def test_feedback_comments_stay_local_when_the_privacy_switch_is_off(self) -> None:
        lf, client, _ = tracer(store_text=False)

        lf.score("q1", "user_feedback", 0, comment="my private note", data_type="BOOLEAN")

        assert client.scores[0]["comment"] is None
        assert client.scores[0]["value"] == 0

    def test_trace_url_and_flush(self) -> None:
        lf, client, _ = tracer()

        assert lf.trace_url("q1") == "https://lf.example/trace/trace-of-q1"
        lf.flush()
        assert client.flushed == 1
        assert lf.enabled is True

    def test_sdk_failures_never_reach_the_caller(self, caplog: pytest.LogCaptureFixture) -> None:
        lf, client, _ = tracer(FakeClient(fail=True))

        with caplog.at_level(logging.WARNING):
            trace = lf.start("q1", "q?")
            trace.stage("retrieve-context").end(output=[])
            trace.finish(answer="a")
            lf.score("q1", "user_feedback", 0)

        assert trace.llm_ref is None  # no root: the generation is not traced
        assert client.scores == []
        assert any("Langfuse" in r.message for r in caplog.records)

    def test_a_failing_child_does_not_break_the_question(self) -> None:
        client = FakeClient()
        lf, _, _ = tracer(client)
        trace = lf.start("q1", "q?")
        client.roots[0].fail_on = "child"

        trace.stage("retrieve-context").end(output=[])  # no exception
        trace.finish(answer="a")

        assert client.roots[0].ended


@dataclass
class FakeLangfuseModule:
    built: list[dict[str, Any]] = field(default_factory=list)

    def Langfuse(self, **kwargs: Any) -> FakeClient:
        self.built.append(kwargs)
        return FakeClient()

    propagate_attributes = FakePropagate()

    class RandomIdGenerator:  # stands in for opentelemetry's, imported from the same fake
        def generate_trace_id(self) -> int:
            return 1

        def generate_span_id(self) -> int:
            return 2


class TestCreateTracer:
    @pytest.fixture(autouse=True)
    def isolated(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No real .env keys: run outside the repo, with Langfuse variables cleared."""
        monkeypatch.chdir(tmp_path)
        for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL"):
            monkeypatch.delenv(name, raising=False)
        langfuse_tracer.clear_tracer_cache()

    def ready(self, **extra: Any) -> Settings:
        return Settings(
            langfuse_public_key="fake-public-value",
            langfuse_secret_key=SecretStr(PRIVATE_VALUE),
            langfuse_base_url="https://us.cloud.langfuse.com",
            langfuse_enabled=True,
            **extra,
        )

    def test_not_ready_gives_the_noop_tracer(self) -> None:
        assert create_tracer(Settings(langfuse_enabled=False)) is NOOP_TRACER
        assert create_tracer(Settings(langfuse_enabled=True)) is NOOP_TRACER  # no keys

    def test_missing_package_gives_the_noop_tracer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(name: str, extra: str) -> None:
            raise _optional.OptionalDependencyError("not installed")

        monkeypatch.setattr(langfuse_tracer, "import_optional", missing)

        assert create_tracer(self.ready()) is NOOP_TRACER

    def test_builds_the_client_from_settings_with_explicit_arguments(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = FakeLangfuseModule()
        monkeypatch.setattr(langfuse_tracer, "import_optional", lambda name, extra: module)

        built = create_tracer(self.ready(langfuse_environment="staging"))

        assert isinstance(built, LangfuseTracer)
        (kwargs,) = module.built
        assert kwargs["public_key"] == "fake-public-value"
        assert kwargs["secret_key"] == PRIVATE_VALUE
        assert kwargs["base_url"] == "https://us.cloud.langfuse.com"
        assert kwargs["environment"] == "staging"
        assert kwargs["release"]
        assert kwargs["mask"] is None

    def test_hides_text_when_the_privacy_switch_is_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module = FakeLangfuseModule()
        monkeypatch.setattr(langfuse_tracer, "import_optional", lambda name, extra: module)

        built = create_tracer(self.ready(observe_store_text=False))

        assert isinstance(built, LangfuseTracer)
        assert built.store_text is False  # so feedback comments stay local too
        mask = module.built[0]["mask"]
        assert mask(data="my private question") == langfuse_tracer.HIDDEN_TEXT
        assert mask(data=None) is None

    def test_one_client_per_process(self, monkeypatch: pytest.MonkeyPatch) -> None:
        module = FakeLangfuseModule()
        monkeypatch.setattr(langfuse_tracer, "import_optional", lambda name, extra: module)

        assert create_tracer(self.ready()) is create_tracer(self.ready())
        assert len(module.built) == 1


def test_the_root_is_created_while_its_trace_id_is_seeded() -> None:
    seed = FakeSeed()
    seen: list[str | None] = []

    class SpyClient(FakeClient):
        def start_observation(self, **kwargs: Any) -> FakeObservation:
            seen.append(seed.active)
            return super().start_observation(**kwargs)

    LangfuseTracer(SpyClient(), public_key="p", propagate=FakePropagate(),
                   seed_trace=seed).start("q1", "q?")  # fmt: skip

    assert seed.seeded == ["trace-of-q1"]
    assert seen == ["trace-of-q1"]


def test_seeded_ids_are_used_inside_the_block_then_random_again() -> None:
    otel = pytest.importorskip("opentelemetry.sdk.trace.id_generator")
    generator = langfuse_tracer.seeded_id_generator(otel.RandomIdGenerator)

    with langfuse_tracer.next_trace_id("0" * 31 + "a"):
        assert generator.generate_trace_id() == 10
    assert generator.generate_trace_id() != 10
    assert generator.generate_span_id() > 0


def test_the_client_gets_the_seeded_id_generator(monkeypatch: pytest.MonkeyPatch,
                                                tmp_path: Path) -> None:  # fmt: skip
    monkeypatch.chdir(tmp_path)
    langfuse_tracer.clear_tracer_cache()
    module = FakeLangfuseModule()
    monkeypatch.setattr(langfuse_tracer, "import_optional", lambda name, extra: module)

    create_tracer(Settings(langfuse_public_key="p", langfuse_secret_key=SecretStr(PRIVATE_VALUE),
                           langfuse_enabled=True))  # fmt: skip

    generator = module.built[0]["id_generator"]
    with langfuse_tracer.next_trace_id("0" * 31 + "f"):
        assert generator.generate_trace_id() == 15
    assert generator.generate_trace_id() == 1
