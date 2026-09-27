"""Judge generated answers on the golden dataset; exit 1 if a metric is below its threshold.

Usage: uv run python scripts/run_generation_evaluation.py [--limit N] [--full]
       [--golden qa_pairs.json] [--report out.json]
"""

from rag_eval_platform.evaluation.generation_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
