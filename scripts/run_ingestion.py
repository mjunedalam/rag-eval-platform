"""Load and chunk documents from data/raw into data/processed/chunks.jsonl.

Usage: uv run python scripts/run_ingestion.py [--strategy fixed] [--chunk-size 500] ...
"""

from rag_eval_platform.ingestion.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
