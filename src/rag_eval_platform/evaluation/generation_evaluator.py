"""Score generated answers on the golden dataset with an LLM judge.

Two passes, so a local machine only needs one model in memory at a time:

1. the pipeline answers every golden question (the model being evaluated);
2. the judge scores every answer (faithfulness, answer relevance, optionally context
   precision/recall) and checks each ``[n]`` citation against its source.

Refusals are not sent to the judge: every golden question is answerable, so a refusal gets
answer relevance 0 and no faithfulness score (it makes no claims). Averages skip unscored
values; a thresholded metric with no score at all fails the check. Citation validity has no
threshold, so an answer whose citation verdicts stay unreadable is logged and left unscored
instead of stopping a long run.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Self

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.evaluation.citation_validity import CITATION_PROMPT_VERSION, CitationValidity
from rag_eval_platform.evaluation.golden_dataset import GoldenExample
from rag_eval_platform.evaluation.judge import Judge, JudgeError, JudgeSample, JudgeScores
from rag_eval_platform.generation.generator import Answer

logger = logging.getLogger(__name__)


class _Pipeline(Protocol):
    def ask(self, question: str) -> Answer: ...


class _CitationChecker(Protocol):
    def check(self, answer: Answer) -> CitationValidity: ...


@dataclass(frozen=True)
class GenerationThresholds:
    min_faithfulness: float
    min_answer_relevance: float

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.min_faithfulness, settings.min_answer_relevance)


@dataclass(frozen=True)
class GenerationExampleResult:
    example_id: str
    query_type: str
    question: str
    answer: str
    refusal: bool
    invalid_citations: tuple[int, ...]
    cited_doc_ids: tuple[str, ...]
    scores: JudgeScores
    citation_validity: float | None
    unsupported_citations: tuple[tuple[int, str], ...]
    generation_ms: float


@dataclass(frozen=True)
class GenerationSummary:
    """Averages over a set of answers; ``None`` when nothing in the set had that score."""

    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    citation_validity: float | None = None
    hallucination_rate: float | None = None  # share of judged answers with an unsupported claim
    refusal_rate: float | None = None
    invalid_citation_rate: float | None = None  # share of answers citing a missing source


@dataclass(frozen=True)
class GenerationReport:
    generator_model: str
    prompt_version: str
    judge_model: str
    citation_prompt_version: str
    full: bool
    thresholds: GenerationThresholds
    overall: GenerationSummary
    by_query_type: Mapping[str, GenerationSummary]
    examples: tuple[GenerationExampleResult, ...]
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_generation(
    examples: Sequence[GoldenExample],
    pipeline: _Pipeline,
    judge: Judge,
    citation_checker: _CitationChecker,
    thresholds: GenerationThresholds,
    *,
    full: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> GenerationReport:
    """Answer every golden question, then judge every answer and check the thresholds."""
    if not examples:
        raise ValueError("evaluate_generation needs at least one golden example")
    progress = on_progress or (lambda message: logger.info(message))

    answers = []
    for i, example in enumerate(examples, start=1):
        answers.append(pipeline.ask(example.question))
        progress(f"Answered {i}/{len(examples)}: {example.id}")

    results = []
    for i, (example, answer) in enumerate(zip(examples, answers, strict=True), start=1):
        results.append(_judge_example(example, answer, judge, citation_checker, full))
        progress(f"Judged {i}/{len(examples)}: {example.id}")

    overall = summarize(results)
    groups: dict[str, list[GenerationExampleResult]] = defaultdict(list)
    for result in results:
        groups[result.query_type].append(result)
    return GenerationReport(
        generator_model=answers[0].model,
        prompt_version=answers[0].prompt_version,
        judge_model=judge.model,
        citation_prompt_version=CITATION_PROMPT_VERSION,
        full=full,
        thresholds=thresholds,
        overall=overall,
        by_query_type={name: summarize(group) for name, group in sorted(groups.items())},
        examples=tuple(results),
        failures=check_generation_thresholds(overall, thresholds),
    )


def summarize(results: Sequence[GenerationExampleResult]) -> GenerationSummary:
    faithfulness = [r.scores.faithfulness for r in results]
    return GenerationSummary(
        faithfulness=_mean(faithfulness),
        answer_relevance=_mean([r.scores.answer_relevance for r in results]),
        context_precision=_mean([r.scores.context_precision for r in results]),
        context_recall=_mean([r.scores.context_recall for r in results]),
        citation_validity=_mean([r.citation_validity for r in results]),
        hallucination_rate=_mean([float(f < 1.0) for f in faithfulness if f is not None]),
        refusal_rate=_mean([float(r.refusal) for r in results]),
        invalid_citation_rate=_mean([float(bool(r.invalid_citations)) for r in results]),
    )


def check_generation_thresholds(
    summary: GenerationSummary, thresholds: GenerationThresholds
) -> tuple[str, ...]:
    """Describe every thresholded metric that is below its minimum or was not scored."""
    checks = (
        ("faithfulness", summary.faithfulness, thresholds.min_faithfulness),
        ("answer_relevance", summary.answer_relevance, thresholds.min_answer_relevance),
    )
    failures = []
    for name, value, minimum in checks:
        if value is None:
            failures.append(f"{name} was not scored")
        elif value < minimum:
            failures.append(f"{name} {value:.3f} < {minimum:.3f}")
    return tuple(failures)


def generation_report_to_dict(report: GenerationReport) -> dict[str, Any]:
    """Plain-JSON form of the report (for the report file and CI artifacts)."""
    data = asdict(report)
    data["passed"] = report.passed
    data["by_query_type"] = {name: asdict(s) for name, s in report.by_query_type.items()}
    for example in data["examples"]:
        for name in ("invalid_citations", "cited_doc_ids"):
            example[name] = list(example[name])
        example["unsupported_citations"] = [list(pair) for pair in example["unsupported_citations"]]
    data["failures"] = list(report.failures)
    return data


def _judge_example(
    example: GoldenExample,
    answer: Answer,
    judge: Judge,
    checker: _CitationChecker,
    full: bool,
) -> GenerationExampleResult:
    if answer.is_refusal:
        scores, validity = JudgeScores(answer_relevance=0.0), CitationValidity(())
    else:
        sample = JudgeSample(
            question=example.question,
            answer=answer.text,
            contexts=tuple(source.chunk.text for source in answer.sources),
            reference=example.expected_answer,
        )
        scores, validity = judge.score(sample, full=full), _check_citations(checker, answer)

    return GenerationExampleResult(
        example_id=example.id,
        query_type=example.query_type,
        question=example.question,
        answer=answer.text,
        refusal=answer.is_refusal,
        invalid_citations=answer.invalid_citations,
        cited_doc_ids=tuple(dict.fromkeys(c.source.chunk.doc_id for c in answer.citations)),
        scores=scores,
        citation_validity=validity.score,
        unsupported_citations=validity.unsupported,
        generation_ms=answer.latency_ms,
    )


def _check_citations(checker: _CitationChecker, answer: Answer) -> CitationValidity:
    try:
        return checker.check(answer)
    except JudgeError as exc:
        logger.warning("Citation validity not scored", extra={"question": answer.question,
                                                              "error": str(exc)})  # fmt: skip
        return CitationValidity(())


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
