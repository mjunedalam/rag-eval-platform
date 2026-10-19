"""Score logged answers that have no TruLens scores yet (Phase 7b catch-up).

Usage:
    uv run python scripts/run_trulens.py --pending            # every unscored answer
    uv run python scripts/run_trulens.py --pending --last 10  # only the newest 10
    uv run python scripts/run_trulens.py QUERY_ID ...         # these answers (scores again)

Covers what the background worker never saw: ``ask.py`` answers, jobs dropped by a full
queue, and anything cut off when a process stopped. Needs the ``monitoring`` extra and the
judge model on Ollama (``ollama pull gemma3:4b``).
Exit 0 = done, 1 = some answers could not be scored (they stay pending), 2 = cannot run.
"""

import argparse
from collections import Counter
from collections.abc import Sequence

from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.observability.langfuse_tracer import create_tracer
from rag_eval_platform.observability.score_worker import ScoreWorker, chroma_texts
from rag_eval_platform.observability.store import ObserveStore
from rag_eval_platform.observability.trulens_scorer import create_triad_scorer

EXIT_DONE, EXIT_FAILURES, EXIT_ERROR = 0, 1, 2


def build_worker(settings: Settings) -> ScoreWorker | None:
    """A worker for scoring on this thread; ignores ``RAG_TRULENS_ENABLED`` (an explicit run)."""
    settings = settings.model_copy(update={"trulens_enabled": True})
    scorer = create_triad_scorer(settings)
    if scorer is None:
        return None
    return ScoreWorker(scorer, ObserveStore(settings.observe_db),
                       chroma_texts(settings.chroma_host, settings.chroma_port),
                       create_tracer(settings))  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score logged answers with the TruLens triad.")
    parser.add_argument("query_ids", nargs="*", help="answers to score (again)")
    parser.add_argument("--pending", action="store_true", help="every answer with no scores")
    parser.add_argument("--last", type=int, metavar="N", help="with --pending: the newest N")
    args = parser.parse_args(argv)
    if not args.pending and not args.query_ids:
        parser.error("give --pending or one or more query ids")
    settings = get_settings()
    if not settings.observe_enabled:
        print("The query log is off (RAG_OBSERVE_ENABLED=false): nothing to score.")
        return EXIT_ERROR
    worker = build_worker(settings)
    if worker is None:
        print("TruLens is not available. Run: uv sync --extra monitoring")
        return EXIT_ERROR
    ids = list(args.query_ids) or ObserveStore(settings.observe_db).unscored_query_ids(args.last)
    print(f"Scoring {len(ids)} answer(s) with {settings.trulens_judge_model}...")
    outcomes: Counter[str] = Counter()
    for n, query_id in enumerate(ids, 1):
        outcome = worker.score_now(query_id)
        outcomes[outcome] += 1
        print(f"  [{n}/{len(ids)}] {query_id}: {outcome}")
    create_tracer(settings).flush()
    print(f"{outcomes['scored']} scored, {outcomes['skipped']} skipped, "
          f"{outcomes['failed']} failed")  # fmt: skip
    return EXIT_FAILURES if outcomes["failed"] else EXIT_DONE
