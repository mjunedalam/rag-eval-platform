"""Generation evaluation command: judge the pipeline's answers on the golden dataset.

Run with ``uv run python scripts/run_generation_evaluation.py`` (Chroma seeded, Ollama
running with the generator and judge models pulled, ``uv sync --all-extras``). Prints a
summary, writes the full JSON report, and exits with:
0 = all thresholds met, 1 = a metric is below its threshold, 2 = could not run.

A full run is slow with a local judge (about a minute per question); use ``--limit`` for a
quick check and ``--full`` to add context precision and recall.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.evaluation.citation_validity import create_citation_checker
from rag_eval_platform.evaluation.generation_evaluator import (
    GenerationReport,
    GenerationSummary,
    GenerationThresholds,
    evaluate_generation,
    generation_report_to_dict,
)
from rag_eval_platform.evaluation.golden_dataset import GoldenDatasetError, load_golden_dataset
from rag_eval_platform.evaluation.judge import Judge, JudgeError, RagasJudge
from rag_eval_platform.generation.generator import GenerationError
from rag_eval_platform.ingestion.embedding import EmbeddingError, create_embedder
from rag_eval_platform.pipeline import create_pipeline
from rag_eval_platform.retrieval.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = Path("reports/generation_report.json")
EXIT_PASSED, EXIT_BELOW_THRESHOLD, EXIT_ERROR = 0, 1, 2
_KNOWN_ERRORS = (
    GoldenDatasetError,
    VectorStoreError,
    EmbeddingError,
    GenerationError,
    JudgeError,
    OptionalDependencyError,
)


def create_judge(settings: Settings) -> Judge:
    """The RAGAS judge; answer relevance embeds with the same model as retrieval."""
    return RagasJudge.from_settings(settings, create_embedder(settings))


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Judge generated answers on the golden dataset.")
    parser.add_argument("--golden", type=Path, default=settings.golden_dataset_path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--limit", type=_positive_int, help="score only the first N questions")
    parser.add_argument("--full", action="store_true", help="add context precision and recall")
    args = parser.parse_args(argv)

    try:
        examples = load_golden_dataset(args.golden)[: args.limit]
        report = evaluate_generation(
            examples,
            create_pipeline(settings),
            create_judge(settings),
            create_citation_checker(settings),
            GenerationThresholds.from_settings(settings),
            full=args.full,
        )
    except _KNOWN_ERRORS as exc:
        logger.error("Generation evaluation could not run: %s", exc)
        return EXIT_ERROR

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(generation_report_to_dict(report), indent=2)
    args.report.write_text(report_json, encoding="utf-8")
    print(format_generation_report(report))
    logger.info(
        "Generation evaluation finished",
        extra={
            "passed": report.passed,
            "failures": list(report.failures),
            "judge_model": report.judge_model,
            "report": str(args.report),
        },
    )
    return EXIT_PASSED if report.passed else EXIT_BELOW_THRESHOLD


def format_generation_report(report: GenerationReport) -> str:
    """Human-readable summary: scores vs thresholds, per query type, and what to look at."""
    t, o = report.thresholds, report.overall
    rows: list[tuple[str, float | None, float | None]] = [
        ("faithfulness", o.faithfulness, t.min_faithfulness),
        ("answer_relevance", o.answer_relevance, t.min_answer_relevance),
        ("citation_validity", o.citation_validity, None),
        ("hallucination_rate", o.hallucination_rate, None),
        ("refusal_rate", o.refusal_rate, None),
        ("invalid_citation_rate", o.invalid_citation_rate, None),
    ]
    if report.full:
        rows += [("context_precision", o.context_precision, None),
                 ("context_recall", o.context_recall, None)]  # fmt: skip

    lines = [
        f"Generation evaluation: {len(report.examples)} questions · generator "
        f"{report.generator_model} (prompt {report.prompt_version}) · judge {report.judge_model}",
        "",
        f"{'metric':<23}{'score':>7}{'min':>7}",
    ]
    for name, value, minimum in rows:
        verdict = "" if minimum is None or value is None else (
            "  PASS" if value >= minimum else "  FAIL")  # fmt: skip
        min_text = "-" if minimum is None else f"{minimum:.2f}"
        lines.append(f"{name:<23}{_fmt(value):>7}{min_text:>7}{verdict}")

    lines += ["", f"{'query type':<13}{'n':>4}{'faith':>8}{'relev':>8}{'cite':>8}"]
    for query_type, summary in report.by_query_type.items():
        n = sum(1 for r in report.examples if r.query_type == query_type)
        lines.append(f"{query_type:<13}{n:>4}{_row(summary)}")

    unfaithful = [r for r in report.examples if (r.scores.faithfulness or 1.0) < 1.0]
    lines += ["", f"Answers with an unsupported claim: {len(unfaithful)}"]
    lines += [f"  {r.example_id}: faithfulness {r.scores.faithfulness:.2f}  {r.answer[:90]!r}"
              for r in unfaithful]  # fmt: skip

    unsupported = [(r.example_id, n, s) for r in report.examples
                   for n, s in r.unsupported_citations]  # fmt: skip
    lines += ["", f"Citations the judge found unsupported: {len(unsupported)}"]
    lines += [f"  {example_id} [{n}]: {sentence[:90]}" for example_id, n, sentence in unsupported]

    lines += ["", "PASSED" if report.passed else "FAILED: " + "; ".join(report.failures)]
    return "\n".join(lines)


def _row(summary: GenerationSummary) -> str:
    values = (summary.faithfulness, summary.answer_relevance, summary.citation_validity)
    return "".join(f"{_fmt(v):>8}" for v in values)


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value
