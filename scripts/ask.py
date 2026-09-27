"""Ask a question and get a cited answer from the corpus.

Usage: uv run python scripts/ask.py "What does MRR measure?"
Needs Chroma seeded and the LLM available (default: Ollama with qwen3:8b).
"""

from rag_eval_platform.ask import main

if __name__ == "__main__":
    raise SystemExit(main())
