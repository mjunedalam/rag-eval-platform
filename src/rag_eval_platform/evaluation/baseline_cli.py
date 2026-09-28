"""Adopt an existing report as a committed baseline, after checking it still applies.

Usage: uv run python scripts/save_baseline.py generation reports/generation_report.json

The report must cover every golden question, and the fingerprint it records must match
the current one. A report from before fingerprints existed needs ``--legacy``: then only
its generator, prompt, judge and citation-prompt versions are checked, and it is trusted
to have been made with the current retrieval settings, golden set and corpus.
Exit 0 = saved, 1 = refused, 2 = the report could not be read.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.evaluation.baselines import (
    GENERATION_BASELINE,
    RETRIEVAL_BASELINE,
    check_adoptable,
    save_baseline,
)
from rag_eval_platform.evaluation.fingerprint import current_fingerprint
from rag_eval_platform.evaluation.golden_dataset import GoldenDatasetError, load_golden_dataset

EXIT_SAVED, EXIT_REFUSED, EXIT_ERROR = 0, 1, 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Adopt a report as a committed baseline.")
    parser.add_argument("kind", choices=["generation", "retrieval"])
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--legacy", action="store_true", help="adopt a report made before fingerprints existed"
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        fingerprint = current_fingerprint(settings)
        golden_size = len(load_golden_dataset(settings.golden_dataset_path))
    except (OSError, json.JSONDecodeError, GoldenDatasetError) as exc:
        print(f"Could not read the inputs: {exc}")
        return EXIT_ERROR
    if args.kind == "generation":
        problems = check_adoptable(report, fingerprint, golden_size, legacy=args.legacy)
        if problems:
            print("Not adopted:\n" + "\n".join(f"  - {p}" for p in problems))
            return EXIT_REFUSED
    target = GENERATION_BASELINE if args.kind == "generation" else RETRIEVAL_BASELINE
    save_baseline(target, report, fingerprint)
    print(f"Saved {target}; commit it.")
    return EXIT_SAVED
