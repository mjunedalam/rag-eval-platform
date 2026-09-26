"""Embed data/processed/chunks.jsonl and load it into the vector store (Chroma).

Usage: uv run python scripts/seed_vector_store.py [--chunks path/to/chunks.jsonl]
Needs Chroma running: docker compose -f docker/docker-compose.yml up -d
"""

from rag_eval_platform.retrieval.seed import main

if __name__ == "__main__":
    raise SystemExit(main())
