"""Tests for observability.trulens_scorer: the RAG triad against a fake TruLens provider."""

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from rag_eval_platform import _optional
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability import trulens_scorer
from rag_eval_platform.observability.trulens_scorer import (
    MetricScore,
    TriadScoreError,
    TriadScorer,
    create_triad_scorer,
)


@dataclass
class FakeProvider:
    """Answers like TruLens' LLM providers: ``(score 0..1, {"reason": ...})``."""

    context_scores: list[float] = field(default_factory=lambda: [1.0, 0.5])
    grounded: float = 0.75
    relevant: float = 0.9
    calls: list[tuple[str, dict[str, str]]] = field(default_factory=list)

    def context_relevance_with_cot_reasons(self, question: str, context: str) -> tuple[float, Any]:
        self.calls.append(("context", {"question": question, "context": context}))
        score = self.context_scores[len([c for c in self.calls if c[0] == "context"]) - 1]
        return score, {"reason": f"chunk says {context}"}

    def groundedness_measure_with_cot_reasons(
        self, source: str, statement: str
    ) -> tuple[float, Any]:
        self.calls.append(("grounded", {"source": source, "statement": statement}))
        return self.grounded, {"reasons": "Criteria: supported\nScore: 3"}

    def relevance_with_cot_reasons(self, prompt: str, response: str) -> tuple[float, Any]:
        self.calls.append(("relevant", {"prompt": prompt, "response": response}))
        return self.relevant, {"reason": "answers the question"}


def test_scores_the_three_triad_metrics_with_their_reasons() -> None:
    provider = FakeProvider()
    scorer = TriadScorer(provider, judge="gemma3:4b")

    scores = scorer.score("What is MRR?", ["MRR is rank based.", "Unrelated."], "MRR is ... [1].")

    assert scores == (
        MetricScore("context_relevance", 0.75,
                    "[1] 1.00: chunk says MRR is rank based.\n[2] 0.50: chunk says Unrelated."),
        MetricScore("groundedness", 0.75, "Criteria: supported\nScore: 3"),
        MetricScore("answer_relevance", 0.9, "answers the question"),
    )  # fmt: skip
    assert scorer.judge == "gemma3:4b"


def test_each_metric_sees_the_right_inputs() -> None:
    provider = FakeProvider()

    TriadScorer(provider, judge="j").score("Q?", ["c1", "c2"], "A.")

    assert provider.calls == [
        ("context", {"question": "Q?", "context": "c1"}),
        ("context", {"question": "Q?", "context": "c2"}),
        ("grounded", {"source": "c1\n\nc2", "statement": "A."}),
        ("relevant", {"prompt": "Q?", "response": "A."}),
    ]


def test_values_are_clamped_to_zero_to_one() -> None:
    provider = FakeProvider(context_scores=[1.4], grounded=-0.2)

    scores = TriadScorer(provider, judge="j").score("Q?", ["c"], "A.")

    assert [s.value for s in scores] == [1.0, 0.0, 0.9]


def test_an_unreadable_judge_reply_fails_the_whole_answer() -> None:
    """TruLens gives NaN when it cannot parse the judge; saving nothing lets a re-run retry."""
    provider = FakeProvider(grounded=math.nan)

    with pytest.raises(TriadScoreError, match="groundedness"):
        TriadScorer(provider, judge="j").score("Q?", ["c"], "A.")


def test_no_context_cannot_be_judged() -> None:
    with pytest.raises(TriadScoreError, match="no retrieved text"):
        TriadScorer(FakeProvider(), judge="j").score("Q?", [], "A.")


def test_long_reasons_are_shortened() -> None:
    class Wordy(FakeProvider):
        def relevance_with_cot_reasons(self, prompt: str, response: str) -> tuple[float, Any]:
            return 0.5, {"reason": "x" * 5000}

    scores = TriadScorer(Wordy(context_scores=[0.5]), judge="j").score("Q?", ["c"], "A.")

    assert len(scores[2].reason or "") <= trulens_scorer.MAX_REASON_CHARS


class FakeLiteLLM:
    built: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, **kwargs: Any) -> None:
        FakeLiteLLM.built.append(kwargs)


class FakeModule:
    LiteLLM = FakeLiteLLM


class TestCreateTriadScorer:
    def test_off_gives_none(self) -> None:
        assert create_triad_scorer(Settings(_env_file=None, trulens_enabled=False)) is None

    def test_missing_package_gives_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def missing(name: str, extra: str) -> Any:
            raise _optional.OptionalDependencyError(f"{name} missing; uv sync --extra {extra}")

        monkeypatch.setattr(trulens_scorer, "import_optional", missing)

        assert create_triad_scorer(Settings(_env_file=None, trulens_enabled=True)) is None

    def test_points_litellm_at_the_ollama_judge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        FakeLiteLLM.built.clear()
        monkeypatch.setattr(trulens_scorer, "import_optional", lambda name, extra: FakeModule)
        settings = Settings(
            trulens_enabled=True,
            trulens_judge_model="gemma3:4b",
            ollama_base_url="http://host.docker.internal:11434/v1",
        )

        scorer = create_triad_scorer(settings)

        assert scorer is not None
        assert scorer.judge == "gemma3:4b"
        assert FakeLiteLLM.built == [{
            "model_engine": "ollama/gemma3:4b", "api_base": "http://host.docker.internal:11434",
        }]  # fmt: skip


def test_groundedness_reasons_list_each_claim_with_its_evidence() -> None:
    """The real shape from TruLens 2.16: one check per claim in the answer."""

    class PerClaim(FakeProvider):
        def groundedness_measure_with_cot_reasons(
            self, source: str, statement: str
        ) -> tuple[float, Any]:
            return 0.5, {"reasons": [
                {"criteria": "MRR uses the first relevant rank [1].",
                 "supporting_evidence": "Criteria: MRR uses ...\nSupporting Evidence: MRR averages "
                                        "one divided by the rank.", "score": 1.0},
                {"criteria": "It was invented in 1999.", "supporting_evidence": "NOTHING FOUND",
                 "score": 0.0},
            ]}  # fmt: skip

    scores = TriadScorer(PerClaim(context_scores=[1.0]), judge="j").score("Q?", ["c"], "A.")

    assert scores[1].reason == (
        "1.00 MRR uses the first relevant rank [1]. — MRR averages one divided by the rank.\n"
        "0.00 It was invented in 1999. — NOTHING FOUND"
    )
