"""Background TruLens scoring (Phase 7b): one thread, a bounded queue of ``query_id``s.

The answer is delivered first; scoring happens afterwards, so answers keep their speed.
For each id the worker loads the logged question from SQLite, fetches the retrieved chunks'
text, asks the judge for the RAG triad, saves the three rows and sends them to the question's
Langfuse trace. Nothing here ever raises into the caller: a full queue drops the job with a
warning and a failed score saves nothing, so ``scripts/run_trulens.py --pending`` can retry
it later.
"""

import logging
import queue
import threading
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Literal, Protocol

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.langfuse_tracer import create_tracer
from rag_eval_platform.observability.store import ObserveStore, TriadScoreRecord
from rag_eval_platform.observability.tracing import NOOP_TRACER, Tracer
from rag_eval_platform.observability.trulens_scorer import MetricScore, create_triad_scorer
from rag_eval_platform.retrieval.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

Outcome = Literal["scored", "skipped", "failed"]
# (collection, chunk ids) -> the text of each chunk that still exists.
ChunkTexts = Callable[[str, Sequence[str]], Sequence[str]]
_STOP = object()


class Scorer(Protocol):
    judge: str

    def score(
        self, question: str, contexts: Sequence[str], answer: str
    ) -> tuple[MetricScore, ...]: ...


class ScoreWorker:
    def __init__(self, scorer: Scorer, store: ObserveStore, texts: ChunkTexts,
                 tracer: Tracer = NOOP_TRACER, *, queue_size: int = 100,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:  # fmt: skip
        self._scorer, self._store, self._texts, self._tracer = scorer, store, texts, tracer
        self._clock = clock
        self._queue: queue.Queue[object] = queue.Queue(maxsize=queue_size)
        self._lock = threading.Condition()
        self._pending = 0  # queued or running
        self._busy = threading.Event()
        self._stopped = False
        self._thread: threading.Thread | None = None

    def score_now(self, query_id: str) -> Outcome:
        """Scores one logged answer on the calling thread; never raises."""
        try:
            record = self._store.get_query(query_id)
            if record is None or record.status != "ok" or record.refusal or not record.answer:
                return "skipped"
            if record.question.startswith("sha256:"):  # RAG_OBSERVE_STORE_TEXT=false
                return "skipped"
            contexts = self._texts(record.collection, record.retrieved_chunk_ids)
            scores = self._scorer.score(record.question, contexts, record.answer)
            now = self._clock()
            self._store.save_triad_scores([
                TriadScoreRecord(query_id, s.metric, s.value, s.reason, self._scorer.judge, now)
                for s in scores
            ])  # fmt: skip
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("TruLens could not score %s (%s); it stays pending", query_id, error,
                           extra={"query_id": query_id, "error": error})  # fmt: skip
            return "failed"
        for s in scores:
            self._tracer.score(query_id, f"trulens_{s.metric}", s.value, comment=s.reason)
        return "scored"

    def submit(self, query_id: str) -> bool:
        """Queues an answer for scoring; ``False`` (and a warning) when full or stopped."""
        with self._lock:
            if self._stopped:
                return False
            try:
                self._queue.put_nowait(query_id)
            except queue.Full:
                logger.warning("TruLens queue is full; %s is left for run_trulens.py --pending",
                               query_id, extra={"query_id": query_id})  # fmt: skip
                return False
            self._pending += 1
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="trulens-scorer",
                                                daemon=True)  # fmt: skip
                self._thread.start()
        return True

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            self._busy.set()
            try:
                self.score_now(str(item))
            finally:
                with self._lock:
                    self._pending -= 1
                    self._lock.notify_all()

    def wait_until_busy(self, timeout: float) -> bool:
        """For tests and the catch-up command: wait until the thread has taken a job."""
        return self._busy.wait(timeout)

    def wait_until_idle(self, timeout: float) -> bool:
        """Waits until every queued job has finished."""
        with self._lock:
            return self._lock.wait_for(lambda: self._pending == 0, timeout)

    def stop(self, timeout: float = 30.0) -> None:
        """Refuses new jobs, lets queued ones finish (up to ``timeout``), then ends the thread."""
        with self._lock:
            self._stopped = True
            thread = self._thread
        if thread is None:
            return
        self.wait_until_idle(timeout)
        try:
            self._queue.put(_STOP, timeout=1)
        except queue.Full:  # still busy after the timeout: the daemon thread ends with the process
            return
        thread.join(timeout=1)


def chroma_texts(host: str, port: int) -> ChunkTexts:
    """Looks chunks up in whichever collection the answer was retrieved from."""

    def texts(collection: str, chunk_ids: Sequence[str]) -> Sequence[str]:
        return ChromaVectorStore.connect(host, port, collection).texts(chunk_ids)

    return texts


_workers: dict[tuple[str, ...], ScoreWorker | None] = {}
_workers_lock = threading.Lock()


def get_score_worker(settings: Settings) -> ScoreWorker | None:
    """The process's worker (one per database and judge), or ``None`` when scoring is off.

    Cached, ``None`` included, so a missing package warns once rather than on every answer.
    """
    if not (settings.trulens_enabled and settings.observe_enabled):
        return None
    key = (str(settings.observe_db.resolve()), settings.trulens_judge_model,
           settings.ollama_root_url, settings.chroma_host, str(settings.chroma_port))  # fmt: skip
    with _workers_lock:
        if key not in _workers:
            scorer = create_triad_scorer(settings)
            _workers[key] = None if scorer is None else ScoreWorker(
                scorer, ObserveStore(settings.observe_db),
                chroma_texts(settings.chroma_host, settings.chroma_port), create_tracer(settings),
                queue_size=settings.trulens_queue_size,
            )  # fmt: skip
        return _workers[key]


def stop_score_workers(timeout: float = 30.0) -> None:
    """Finishes queued scores (up to ``timeout`` each); for the API's shutdown."""
    with _workers_lock:
        workers = [w for w in _workers.values() if w is not None]
        _workers.clear()
    for worker in workers:
        worker.stop(timeout)
