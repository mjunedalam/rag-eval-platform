"""Evaluation command: score retrieval on the golden dataset against the thresholds.

Run with ``uv run python scripts/run_evaluation.py`` (Chroma running and seeded).
Prints a summary table, writes the full JSON report, and exits with:
0 = all thresholds met, 1 = a metric is below its threshold, 2 = could not run.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation.evaluator import (
    RetrievalReport,
    RetrievalThresholds,
    evaluate_retrieval,
    report_to_dict,
)
from rag_eval_platform.evaluation.golden_dataset import GoldenDatasetError, load_golden_dataset
from rag_eval_platform.ingestion.embedding import EmbeddingError
from rag_eval_platform.retrieval.retriever import create_retriever
from rag_eval_platform.retrieval.vector_store import VectorStoreError

logger = logging.getLogger(__name__)

DEFAULT_REPORT_PATH = Path("reports/retrieval_report.json")
EXIT_PASSED, EXIT_BELOW_THRESHOLD, EXIT_ERROR = 0, 1, 2


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Evaluate retrieval on the golden dataset.")
    parser.add_argument("--golden", type=Path, default=settings.golden_dataset_path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args(argv)

    try:
        examples = load_golden_dataset(args.golden)
        report = evaluate_retrieval(
            examples,
            create_retriever(settings),
            k=settings.top_k,
            thresholds=RetrievalThresholds.from_settings(settings),
        )
    except (
        GoldenDatasetError,
        VectorStoreError,
        EmbeddingError,
        OptionalDependencyError,
    ) as exc:
        logger.error("Evaluation could not run: %s", exc)
        return EXIT_ERROR

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    print(format_report(report))
    logger.info(
        "Retrieval evaluation finished",
        extra={
            "passed": report.passed,
            "failures": list(report.failures),
            "report": str(args.report),
        },
    )
    return EXIT_PASSED if report.passed else EXIT_BELOW_THRESHOLD


def format_report(report: RetrievalReport) -> str:
    """Human-readable summary: overall scores vs thresholds, per query type, and misses."""
    t, o = report.thresholds, report.overall
    rows = [
        ("recall@k", o.recall, t.min_recall_at_k),
        ("mrr", o.mrr, t.min_mrr),
        ("ndcg@k", o.ndcg, t.min_ndcg_at_k),
        ("precision@k", o.precision, None),
    ]
    lines = [
        f"Retrieval evaluation: k={report.k}, {len(report.examples)} questions",
        "",
        f"{'metric':<13}{'score':>7}{'min':>7}",
    ]
    for name, value, minimum in rows:
        verdict = "" if minimum is None else ("  PASS" if value >= minimum else "  FAIL")
        min_text = "-" if minimum is None else f"{minimum:.2f}"
        lines.append(f"{name:<13}{value:>7.3f}{min_text:>7}{verdict}")

    lines += ["", f"{'query type':<13}{'n':>4}{'recall':>8}{'mrr':>7}{'ndcg':>7}"]
    for query_type, scores in sorted(report.by_query_type.items()):
        n = sum(1 for r in report.examples if r.query_type == query_type)
        lines.append(
            f"{query_type:<13}{n:>4}{scores.recall:>8.3f}{scores.mrr:>7.3f}{scores.ndcg:>7.3f}"
        )

    misses = [r for r in report.examples if r.scores.recall < 1.0]
    lines += ["", f"Questions missing a relevant document: {len(misses)}"]
    for r in misses:
        lines.append(
            f"  {r.example_id}: expected {list(r.relevant_doc_ids)}, "
            f"got {list(r.retrieved_doc_ids)}"
        )
    lines += ["", "PASSED" if report.passed else "FAILED: " + "; ".join(report.failures)]
    return "\n".join(lines)
