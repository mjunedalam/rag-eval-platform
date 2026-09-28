"""Evaluation gate command: rebuild the index, score retrieval, check the baselines.

Run with ``uv run python scripts/run_gate.py`` (Chroma running, ``--extra
local-embeddings``). Writes ``reports/gate_report.json`` and ``reports/gate_summary.md``,
prints the summary, and exits with 0 = pass, 1 = blocked, 2 = could not run.
"""

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.evaluation.baselines import BASELINE_DIR, BaselineError, load_baseline
from rag_eval_platform.evaluation.evaluator import RetrievalThresholds, report_to_dict
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
from rag_eval_platform.evaluation.gate import (
    GateLimits,
    gate_error_markdown,
    gate_markdown,
    gate_to_dict,
    run_gate,
)
from rag_eval_platform.evaluation.gate_runner import rebuild_and_evaluate
from rag_eval_platform.evaluation.golden_dataset import GoldenDatasetError, load_golden_dataset
from rag_eval_platform.ingestion.embedding import EmbeddingError, create_embedder
from rag_eval_platform.ingestion.loaders import DocumentLoadError
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker, Reranker
from rag_eval_platform.retrieval.vector_store import VectorStoreError, create_vector_store

logger = logging.getLogger(__name__)

DEFAULT_GATE_REPORT = Path("reports/gate_report.json")
DEFAULT_GATE_SUMMARY = Path("reports/gate_summary.md")
EXIT_PASSED, EXIT_BLOCKED, EXIT_ERROR = 0, 1, 2
_KNOWN_ERRORS = (GoldenDatasetError, DocumentLoadError, VectorStoreError, EmbeddingError,
                 OptionalDependencyError, BaselineError, OSError, ValueError)  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Run the evaluation gate.")
    parser.add_argument("--baselines", type=Path, default=BASELINE_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_GATE_REPORT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_GATE_SUMMARY)
    parser.add_argument("--summary", type=Path, help="also append the Markdown here "
                        "(CI passes $GITHUB_STEP_SUMMARY)")  # fmt: skip
    args = parser.parse_args(argv)

    try:
        retrieval_baseline = load_baseline(args.baselines / "retrieval.json")
        generation_baseline = load_baseline(args.baselines / "generation.json")
        current = current_fingerprint(settings)
        retrieval = rebuild_and_evaluate(
            load_golden_dataset(settings.golden_dataset_path),
            raw_dir=settings.raw_data_dir, strategy=settings.chunk_strategy,
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
            embedder=create_embedder(settings), store=create_vector_store(settings),
            top_k=settings.top_k, reranker=_reranker(settings),
            rerank_candidates=settings.rerank_candidates,
            thresholds=RetrievalThresholds.from_settings(settings),
            on_progress=lambda message: logger.info(message),
        )  # fmt: skip
    except _KNOWN_ERRORS as exc:
        logger.error("The evaluation gate could not run: %s", exc)
        _write_summary(gate_error_markdown(str(exc)), args.summary_out, args.summary)
        return EXIT_ERROR

    result = run_gate(retrieval, retrieval_baseline, generation_baseline, current,
                      GateLimits.from_settings(settings))  # fmt: skip
    markdown = gate_markdown(result, retrieval, retrieval_baseline)
    report = {"gate": gate_to_dict(result), "retrieval": report_to_dict(retrieval)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary(markdown, args.summary_out, args.summary)
    print(markdown)
    logger.info("Evaluation gate finished", extra={"passed": result.passed,
                "report": str(args.report)})  # fmt: skip
    return EXIT_PASSED if result.passed else EXIT_BLOCKED


def _write_summary(markdown: str, summary_out: Path, append_to: Path | None) -> None:
    """Write the Markdown report, and append it to the CI job summary when given."""
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(markdown, encoding="utf-8")
    if append_to is not None:
        with append_to.open("a", encoding="utf-8") as summary:
            summary.write(markdown)


def _reranker(settings: Settings) -> Reranker | None:
    return (CrossEncoderReranker.from_pretrained(settings.reranker_model)
            if settings.rerank else None)  # fmt: skip
