"""TruLens' RAG triad for one answer (Phase 7b), judged by a small local model on Ollama.

- **Context relevance:** is each retrieved chunk relevant to the question? (mean over chunks)
- **Groundedness:** is every claim in the answer supported by the retrieved text?
- **Answer relevance:** does the answer address the question?

Only TruLens' feedback functions are used, called directly through its LiteLLM provider:
no ``TruApp`` instrumentation, no TruLens database and no dashboard (it pins ``plotly<7``).
The provider is imported lazily from the optional ``monitoring`` extra.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from rag_eval_platform._optional import OptionalDependencyError, import_optional
from rag_eval_platform.config.settings import Settings

logger = logging.getLogger(__name__)

MAX_REASON_CHARS = 2000


class TriadScoreError(Exception):
    """The judge could not score this answer; nothing is saved, so a later run retries it."""


class TriadProvider(Protocol):
    """The three feedback functions of a TruLens LLM provider (``trulens.providers.litellm``)."""

    def context_relevance_with_cot_reasons(
        self, question: str, context: str
    ) -> tuple[float, Any]: ...

    def groundedness_measure_with_cot_reasons(
        self, source: str, statement: str
    ) -> tuple[float, Any]: ...

    def relevance_with_cot_reasons(self, prompt: str, response: str) -> tuple[float, Any]: ...


@dataclass(frozen=True)
class MetricScore:
    metric: str  # one of store.TRIAD_METRICS
    value: float  # 0..1
    reason: str | None


def _claim_line(check: Mapping[str, Any]) -> str:
    """One groundedness check: ``score claim — evidence`` (the evidence without its header)."""
    evidence = str(check.get("supporting_evidence", "")).split("Supporting Evidence:")[-1]
    score = check.get("score")
    prefix = f"{float(score):.2f} " if isinstance(score, int | float) else ""
    return f"{prefix}{check.get('criteria', '')} — {evidence.strip()}"


def _reason(details: Any) -> str | None:
    """TruLens' reasons: ``{"reason": "..."}``, or for groundedness ``{"reasons": [check, ...]}``
    with one check per claim in the answer."""
    parts: list[str] = []
    for value in details.values() if isinstance(details, Mapping) else [details]:
        if isinstance(value, str) and value.strip():
            parts.append(value)
        elif isinstance(value, list):
            parts.extend(_claim_line(v) if isinstance(v, Mapping) else str(v) for v in value)
    return "\n".join(parts).strip()[:MAX_REASON_CHARS] or None


def _value(metric: str, raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise TriadScoreError(f"{metric}: the judge gave no score ({raw!r})") from exc
    if math.isnan(value):
        raise TriadScoreError(f"{metric}: the judge's reply could not be read")
    return min(1.0, max(0.0, value))


class TriadScorer:
    def __init__(self, provider: TriadProvider, *, judge: str) -> None:
        self._provider = provider
        self.judge = judge

    def score(self, question: str, contexts: Sequence[str], answer: str) -> tuple[MetricScore, ...]:
        """The three scores, or ``TriadScoreError`` (the judge's errors propagate as they are)."""
        if not contexts:
            raise TriadScoreError("no retrieved text to judge (the chunks may have been re-seeded)")
        per_chunk = []
        for context in contexts:
            raw, details = self._provider.context_relevance_with_cot_reasons(
                question=question, context=context
            )
            per_chunk.append((_value("context_relevance", raw), _reason(details)))
        chunk_lines = "\n".join(
            f"[{n}] {value:.2f}: {reason or '-'}" for n, (value, reason) in enumerate(per_chunk, 1)
        )
        grounded, grounded_details = self._provider.groundedness_measure_with_cot_reasons(
            source="\n\n".join(contexts), statement=answer
        )
        relevant, relevant_details = self._provider.relevance_with_cot_reasons(
            prompt=question, response=answer
        )
        return (
            MetricScore("context_relevance", sum(v for v, _ in per_chunk) / len(per_chunk),
                        chunk_lines[:MAX_REASON_CHARS]),
            MetricScore("groundedness", _value("groundedness", grounded),
                        _reason(grounded_details)),
            MetricScore("answer_relevance", _value("answer_relevance", relevant),
                        _reason(relevant_details)),
        )  # fmt: skip


def create_triad_scorer(settings: Settings) -> TriadScorer | None:
    """A scorer for ``RAG_TRULENS_JUDGE_MODEL`` on Ollama, or ``None`` when off or not installed."""
    if not settings.trulens_enabled:
        return None
    try:
        module = import_optional("trulens.providers.litellm", extra="monitoring")
        provider = module.LiteLLM(
            model_engine=f"ollama/{settings.trulens_judge_model}",
            api_base=settings.ollama_root_url,
        )
    except OptionalDependencyError as exc:
        logger.warning("TruLens scoring is off: %s", exc)
        return None
    except Exception as exc:  # a broken install must not stop the app
        logger.warning("TruLens scoring is off: %s: %s", type(exc).__name__, exc)
        return None
    return TriadScorer(provider, judge=settings.trulens_judge_model)
