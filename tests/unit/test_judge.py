"""Tests for evaluation.judge with fake RAGAS metrics (no ragas package needed)."""

import asyncio
import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation import judge as judge_module
from rag_eval_platform.evaluation.judge import (
    JudgeError,
    JudgeSample,
    JudgeScores,
    RagasJudge,
)

SAMPLE = JudgeSample(
    question="What does MRR measure?",
    answer="How high the first relevant document ranks [1].",
    contexts=("MRR is the mean of 1/rank.", "Recall counts hits."),
    reference="How high the first relevant document appears.",
)


@dataclass
class FakeMetric:
    value: float = 1.0
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def ascore(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(value=self.value)


def fake_metrics(**values: float) -> dict[str, FakeMetric]:
    names = ("faithfulness", "answer_relevance", "context_precision", "context_recall")
    return {name: FakeMetric(values.get(name, 1.0)) for name in names}


class TestRagasJudge:
    def test_default_run_scores_faithfulness_and_relevance_only(self) -> None:
        metrics = fake_metrics(faithfulness=0.75, answer_relevance=0.9)

        scores = RagasJudge("judge", metrics).score(SAMPLE)

        assert scores == JudgeScores(faithfulness=0.75, answer_relevance=0.9)
        assert metrics["context_precision"].calls == []
        assert metrics["context_recall"].calls == []

    def test_full_run_adds_context_metrics(self) -> None:
        metrics = fake_metrics(context_precision=0.5, context_recall=0.25)

        scores = RagasJudge("judge", metrics).score(SAMPLE, full=True)

        assert scores.context_precision == 0.5
        assert scores.context_recall == 0.25

    def test_each_metric_gets_the_inputs_ragas_expects(self) -> None:
        metrics = fake_metrics()

        RagasJudge("judge", metrics).score(SAMPLE, full=True)

        contexts = list(SAMPLE.contexts)
        expected_faithfulness_call = {
            "user_input": SAMPLE.question,
            "response": SAMPLE.answer,
            "retrieved_contexts": contexts,
        }
        assert metrics["faithfulness"].calls == [expected_faithfulness_call]
        assert metrics["answer_relevance"].calls == [
            {"user_input": SAMPLE.question, "response": SAMPLE.answer}
        ]
        expected_context_call = {
            "user_input": SAMPLE.question,
            "reference": SAMPLE.reference,
            "retrieved_contexts": contexts,
        }
        assert metrics["context_precision"].calls == [expected_context_call]
        assert metrics["context_recall"].calls == [expected_context_call]

    def test_every_call_runs_on_the_same_event_loop(self) -> None:
        loops: list[asyncio.AbstractEventLoop] = []

        class LoopRecordingMetric(FakeMetric):
            async def ascore(self, **kwargs: Any) -> SimpleNamespace:
                loops.append(asyncio.get_running_loop())
                return await super().ascore(**kwargs)

        metrics = {name: LoopRecordingMetric() for name in fake_metrics()}
        judge = RagasJudge("judge", metrics)

        judge.score(SAMPLE)
        judge.score(SAMPLE)

        assert len(loops) == 4
        assert len(set(map(id, loops))) == 1

    def test_nan_from_the_judge_becomes_none(self) -> None:
        scores = RagasJudge("judge", fake_metrics(faithfulness=math.nan)).score(SAMPLE)

        assert scores.faithfulness is None
        assert scores.answer_relevance == 1.0

    def test_metric_failure_becomes_judge_error_with_hint(self) -> None:
        metrics = fake_metrics()
        metrics["faithfulness"].error = ConnectionError("refused")

        with pytest.raises(JudgeError, match=r"faithfulness.*refused.*ollama serve"):
            RagasJudge("gemma3:12b", metrics, hint="Is Ollama running (`ollama serve`)?").score(
                SAMPLE
            )


class TestFromSettings:
    @pytest.fixture
    def fake_ragas(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        class BaseRagasEmbedding:
            pass

        def llm_factory(model: str, **kwargs: Any) -> str:
            captured["model"] = model
            captured.update(kwargs)
            return "llm"

        def metric(name: str) -> Any:
            def build(**kwargs: Any) -> tuple[str, dict[str, Any]]:
                return name, kwargs

            return build

        modules = {
            "openai": SimpleNamespace(AsyncOpenAI=lambda **kw: ("async-client", kw)),
            "ragas.llms": SimpleNamespace(llm_factory=llm_factory),
            "ragas.embeddings.base": SimpleNamespace(BaseRagasEmbedding=BaseRagasEmbedding),
            "ragas.metrics.collections": SimpleNamespace(
                Faithfulness=metric("faithfulness"),
                AnswerRelevancy=metric("answer_relevancy"),
                ContextPrecision=metric("context_precision"),
                ContextRecall=metric("context_recall"),
            ),
        }
        monkeypatch.setattr(judge_module, "import_optional", lambda name, extra: modules[name])
        return captured

    def test_builds_ragas_metrics_on_the_judge_model(self, fake_ragas: dict[str, Any]) -> None:
        judge = RagasJudge.from_settings(Settings(judge_max_tokens=2048), FakeEmbedder())

        assert judge.model == "gemma3:12b"
        assert fake_ragas["model"] == "gemma3:12b"
        assert fake_ragas["provider"] == "openai"
        assert fake_ragas["temperature"] == 0.0
        assert fake_ragas["max_tokens"] == 2048
        client, options = fake_ragas["client"]
        assert client == "async-client"
        assert options["base_url"] == "http://localhost:11434/v1"
        assert options["timeout"] == 300.0

    def test_answer_relevance_uses_the_project_embedder(self, fake_ragas: dict[str, Any]) -> None:
        judge = RagasJudge.from_settings(Settings(), FakeEmbedder())

        built: Any = judge.metrics["answer_relevance"]
        name, kwargs = built
        assert name == "answer_relevancy"
        assert kwargs["embeddings"].embed_text("hi") == [2.0, 0.0]


@dataclass
class FakeEmbedder:
    def embed_documents(self, texts: Any) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0]


def test_embedding_adapter_async_matches_sync() -> None:
    class Base:
        pass

    adapter = judge_module.make_ragas_embeddings(FakeEmbedder(), Base)

    assert asyncio.run(adapter.aembed_text("abc")) == adapter.embed_text("abc") == [3.0, 0.0]
