"""Serve the RAG API on RAG_API_HOST:RAG_API_PORT (default 127.0.0.1:8000); docs at /docs.

Usage: uv run python scripts/serve_api.py   (Chroma running; --all-extras for the models)
"""

from rag_eval_platform.api.serve import main

if __name__ == "__main__":
    main()
