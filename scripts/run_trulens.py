"""Score logged answers that have no TruLens scores yet.

Usage: uv run python scripts/run_trulens.py --pending [--last N]
Needs the monitoring extra and the judge on Ollama (ollama pull gemma3:4b).
"""

from rag_eval_platform.observability.trulens_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
