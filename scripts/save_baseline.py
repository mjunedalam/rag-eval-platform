"""Adopt an existing evaluation report as a committed baseline.

Usage: uv run python scripts/save_baseline.py generation reports/generation_report.json
"""

from rag_eval_platform.evaluation.baseline_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
