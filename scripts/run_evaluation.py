"""Score retrieval on the golden dataset; exit 1 if a metric is below its threshold.

Usage: uv run python scripts/run_evaluation.py [--golden qa_pairs.json] [--report out.json]
"""

from rag_eval_platform.evaluation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
