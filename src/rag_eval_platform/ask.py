"""Ask command: answer one question from the corpus, with cited sources.

Run with ``uv run python scripts/ask.py "What does MRR measure?"`` (Chroma seeded and the
LLM running, by default Ollama). The answer is printed as it is generated (streaming).
Exit code 0 on success, 1 if the pipeline cannot run.
"""

import argparse
import logging
from collections.abc import Sequence

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.config.logging_config import configure_logging
from rag_eval_platform.config.settings import get_settings
from rag_eval_platform.generation.generator import Answer, GenerationError
from rag_eval_platform.ingestion.embedding import EmbeddingError
from rag_eval_platform.pipeline import create_pipeline
from rag_eval_platform.retrieval.vector_store import VectorStoreError

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    parser = argparse.ArgumentParser(description="Answer a question from the document corpus.")
    parser.add_argument("question", nargs="+", help="the question (quotes optional)")
    args = parser.parse_args(argv)

    try:
        stream = create_pipeline(settings).ask_stream(" ".join(args.question))
        for piece in stream:  # print the answer as the model writes it
            print(piece, end="", flush=True)
        print()
    except (
        VectorStoreError,
        GenerationError,
        EmbeddingError,
        OptionalDependencyError,
        ValueError,
    ) as exc:
        logger.error("Could not answer: %s", exc)
        return 1

    print(format_details(stream.answer))
    return 0


def format_details(answer: Answer) -> str:
    """The numbered sources (marking which were cited) and run details."""
    cited = {c.number for c in answer.citations}
    lines = ["", "Sources:"]
    for number, result in enumerate(answer.sources, start=1):
        chunk = result.chunk
        origin = chunk.doc_id if chunk.page is None else f"{chunk.doc_id}, page {chunk.page}"
        mark = "  cited" if number in cited else ""
        lines.append(f"  [{number}] {origin}  (score {result.score:.2f}){mark}")
    if answer.invalid_citations:
        lines.append(
            f"Warning: cited numbers not among the sources: {list(answer.invalid_citations)}"
        )
    tokens = (
        f"{answer.input_tokens} -> {answer.output_tokens} tokens" if answer.input_tokens else ""
    )
    lines += ["", f"{answer.model} · {answer.latency_ms / 1000:.1f} s · {tokens}".rstrip(" ·")]
    return "\n".join(lines)
