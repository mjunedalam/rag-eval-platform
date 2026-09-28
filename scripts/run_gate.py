"""Run the evaluation gate: rebuild the index, score retrieval, check the baselines.

Usage: uv run python scripts/run_gate.py [--summary $GITHUB_STEP_SUMMARY]
Exit 0 = pass, 1 = blocked, 2 = could not run.
"""

from rag_eval_platform.evaluation.gate_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
