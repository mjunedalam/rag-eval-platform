"""Tests for evaluation.generation_evaluator with a fake pipeline, judge and citation checker."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import pytest

from rag_eval_platform.evaluation.citation_validity import CitationValidity, CitationVerdict
from rag_eval_platform.evaluation.generation_evaluator import (
    GenerationReport,
    GenerationSummary,
    GenerationThresholds,
    check_generation_thresholds,
    evaluate_generation,
    generation_report_to_dict,
    summarize,
)
from rag_eval_platform.evaluation.golden_dataset import GoldenExample
from rag_eval_platform.evaluation.judge import JudgeError, JudgeSample, JudgeScores
from rag_eval_platform.generation.generator import Answer, Completion, Generator
from rag_eval_platform.generation.prompt_templates import NO_ANSWER, Message
from rag_eval_platform.ingestion.chunking import Chunk
from rag_eval_platform.retrieval.vector_store import SearchResult


def example(id_: str, question: str, query_type: str = "short") -> GoldenExample:
    return GoldenExample.model_validate(
        {
            "id": id_,
            "question": question,
            "expected_answer": f"reference for {id_}",
            "relevant_doc_ids": ["a.md"],
            "query_type": query_type,
        }
    )


def hit(doc_id: str, text: str) -> SearchResult:
    chunk = Chunk(id=f"{doc_id}#0", doc_id=doc_id, index=0, text=text, start_index=0)
    return SearchResult(chunk=chunk, score=0.5)


@dataclass
class ScriptedLlm:
    reply: str
    model: str = "qwen3:8b"

    def complete(self, messages: Sequence[Message]) -> Completion:
        return Completion(text=self.reply, input_tokens=10, output_tokens=5)

    def stream(self, messages: Sequence[Message]) -> Iterator[Completion]:
        raise NotImplementedError


class FakePipeline:
    """Answers each question with a scripted reply over two fixed chunks."""

    def __init__(self, replies: dict[str, str]) -> None:
        self.replies = replies
        self.asked: list[str] = []

    def ask(self, question: str) -> Answer:
        self.asked.append(question)
        sources = [hit("a.md", "chunk A"), hit("b.md", "chunk B")]
        return Generator(ScriptedLlm(self.replies[question])).generate(question, sources)


@dataclass
class FakeJudge:
    scores: dict[str, JudgeScores]
    model: str = "gemma3:12b"
    samples: list[JudgeSample] = field(default_factory=list)
    full_flags: list[bool] = field(default_factory=list)

    def score(self, sample: JudgeSample, *, full: bool = False) -> JudgeScores:
        self.samples.append(sample)
        self.full_flags.append(full)
        return self.scores[sample.question]


@dataclass
class FakeChecker:
    supported: dict[str, list[bool]]
    model: str = "gemma3:12b"
    checked: list[str] = field(default_factory=list)
    unreadable: set[str] = field(default_factory=set)

    def check(self, answer: Answer) -> CitationValidity:
        self.checked.append(answer.question)
        if answer.question in self.unreadable:
            raise JudgeError("unreadable citation verdicts")
        flags = self.supported.get(answer.question, [])
        return CitationValidity(tuple(CitationVerdict("s.", 1, "a.md", ok) for ok in flags))


EXAMPLES = (
    example("q1", "first?"),
    example("q2", "second?"),
    example("q3", "third?", "multi_hop"),
)
REPLIES = {"first?": "A is true [1].", "second?": "B is true [2][7].", "third?": NO_ANSWER}
JUDGE_SCORES = {
    "first?": JudgeScores(faithfulness=1.0, answer_relevance=0.9),
    "second?": JudgeScores(faithfulness=0.5, answer_relevance=0.7),
}
CITATIONS = {"first?": [True], "second?": [True, False]}
LENIENT = GenerationThresholds(min_faithfulness=0.0, min_answer_relevance=0.0)


def run(
    thresholds: GenerationThresholds = LENIENT, full: bool = False
) -> tuple[GenerationReport, FakePipeline, FakeJudge, FakeChecker]:
    pipeline, judge = FakePipeline(REPLIES), FakeJudge(JUDGE_SCORES)
    checker = FakeChecker(CITATIONS)
    report = evaluate_generation(EXAMPLES, pipeline, judge, checker, thresholds, full=full)
    return report, pipeline, judge, checker


class TestEvaluateGeneration:
    def test_generates_every_answer_before_judging_any(self) -> None:
        events: list[str] = []

        class OrderedPipeline(FakePipeline):
            def ask(self, question: str) -> Answer:
                events.append(f"ask {question}")
                return super().ask(question)

        class OrderedJudge(FakeJudge):
            def score(self, sample: JudgeSample, *, full: bool = False) -> JudgeScores:
                events.append(f"judge {sample.question}")
                return super().score(sample, full=full)

        evaluate_generation(
            EXAMPLES, OrderedPipeline(REPLIES), OrderedJudge(JUDGE_SCORES),
            FakeChecker(CITATIONS), LENIENT,
        )  # fmt: skip

        asks = ["ask first?", "ask second?", "ask third?"]
        assert events == [*asks, "judge first?", "judge second?"]

    def test_judge_sees_question_answer_chunks_and_reference(self) -> None:
        _, _, judge, _ = run()

        sample = judge.samples[0]
        assert sample == JudgeSample(
            question="first?",
            answer="A is true [1].",
            contexts=("chunk A", "chunk B"),
            reference="reference for q1",
        )

    def test_full_flag_is_passed_to_the_judge(self) -> None:
        _, _, judge, _ = run(full=True)

        assert judge.full_flags == [True, True]

    def test_refusals_are_not_judged_and_count_as_irrelevant(self) -> None:
        report, _, judge, checker = run()

        refused = report.examples[2]
        assert refused.refusal is True
        assert refused.scores == JudgeScores(answer_relevance=0.0)
        assert refused.citation_validity is None
        assert "third?" not in [s.question for s in judge.samples]
        assert "third?" not in checker.checked

    def test_per_example_details(self) -> None:
        report, *_ = run()

        second = report.examples[1]
        assert second.example_id == "q2"
        assert second.answer == "B is true [2][7]."
        assert second.invalid_citations == (7,)
        assert second.cited_doc_ids == ("b.md",)
        assert second.citation_validity == 0.5
        assert second.unsupported_citations == ((1, "s."),)

    def test_overall_averages_and_rates(self) -> None:
        report, *_ = run()

        overall = report.overall
        assert overall.faithfulness == pytest.approx(0.75)  # refusal has no faithfulness
        assert overall.answer_relevance == pytest.approx((0.9 + 0.7 + 0.0) / 3)
        assert overall.citation_validity == pytest.approx(0.75)
        assert overall.hallucination_rate == pytest.approx(0.5)  # q2 has faithfulness < 1
        assert overall.refusal_rate == pytest.approx(1 / 3)
        assert overall.invalid_citation_rate == pytest.approx(1 / 3)
        assert overall.context_precision is None

    def test_breakdown_by_query_type(self) -> None:
        report, *_ = run()

        assert set(report.by_query_type) == {"short", "multi_hop"}
        assert report.by_query_type["short"].faithfulness == pytest.approx(0.75)
        assert report.by_query_type["multi_hop"].refusal_rate == 1.0
        assert report.by_query_type["multi_hop"].faithfulness is None

    def test_records_models_and_prompt_versions(self) -> None:
        report, *_ = run()

        assert report.generator_model == "qwen3:8b"
        assert report.judge_model == "gemma3:12b"
        assert report.prompt_version == "v1"
        assert report.citation_prompt_version == "v1"

    def test_passes_or_fails_on_thresholds(self) -> None:
        strict = GenerationThresholds(min_faithfulness=0.85, min_answer_relevance=0.5)

        report, *_ = run(thresholds=strict)

        assert report.passed is False
        assert report.failures == ("faithfulness 0.750 < 0.850",)

    def test_unreadable_citations_leave_that_answer_unscored_and_the_run_continues(self) -> None:
        checker = FakeChecker(CITATIONS, unreadable={"first?"})

        report = evaluate_generation(
            EXAMPLES, FakePipeline(REPLIES), FakeJudge(JUDGE_SCORES), checker, LENIENT
        )

        assert report.examples[0].citation_validity is None
        assert report.examples[0].scores.faithfulness == 1.0
        assert report.examples[1].citation_validity == 0.5
        assert report.overall.citation_validity == 0.5

    def test_needs_at_least_one_example(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            evaluate_generation((), FakePipeline({}), FakeJudge({}), FakeChecker({}), LENIENT)


def test_summarize_ignores_unscored_values() -> None:
    summary = summarize([])

    assert summary == GenerationSummary()


def test_unscored_threshold_metric_is_a_failure() -> None:
    failures = check_generation_thresholds(
        GenerationSummary(answer_relevance=0.9),
        GenerationThresholds(min_faithfulness=0.85, min_answer_relevance=0.8),
    )

    assert failures == ("faithfulness was not scored",)


def test_report_to_dict_is_plain_json() -> None:
    report, *_ = run()

    data = generation_report_to_dict(report)

    assert data["passed"] is True
    assert data["judge_model"] == "gemma3:12b"
    assert data["overall"]["faithfulness"] == pytest.approx(0.75)
    assert data["by_query_type"]["multi_hop"]["refusal_rate"] == 1.0
    assert data["examples"][1]["invalid_citations"] == [7]
    assert data["examples"][1]["unsupported_citations"] == [[1, "s."]]
    assert data["examples"][2]["scores"]["faithfulness"] is None
