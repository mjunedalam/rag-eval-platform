"""LLM-as-a-judge scores for generated answers, computed with RAGAS.

A judge model (``RAG_JUDGE_MODEL``, by default ``gemma3:12b`` on Ollama) reads the question,
the answer and the retrieved chunks and scores:

- faithfulness: share of the answer's claims that the retrieved chunks support;
- answer relevance: how directly the answer addresses the question;
- context precision / recall (``full=True``): whether the chunks were relevant, and whether
  they contain what the reference answer needs.

The judge must differ from the model being evaluated (checked in settings). Needs
``uv sync --extra evaluation`` (RAGAS) and ``--extra openai`` (the client RAGAS drives).
"""

import asyncio
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Self

from rag_eval_platform._optional import import_optional
from rag_eval_platform.config.settings import Settings
from rag_eval_platform.generation.generator import client_options
from rag_eval_platform.ingestion.embedding import Embedder

# Always scored; these two have thresholds in settings.
CORE_METRICS = ("faithfulness", "answer_relevance")
# Scored with full=True only: two more judge calls per question.
CONTEXT_METRICS = ("context_precision", "context_recall")


class JudgeError(Exception):
    """The judge model could not be reached, or its reply could not be used."""


@dataclass(frozen=True)
class JudgeSample:
    question: str
    answer: str
    contexts: tuple[str, ...]  # retrieved chunk texts, in rank order
    reference: str  # the golden expected answer


@dataclass(frozen=True)
class JudgeScores:
    """Scores in [0, 1]; ``None`` means not scored (not requested, or the judge gave no score)."""

    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class Judge(Protocol):
    @property
    def model(self) -> str: ...

    def score(self, sample: JudgeSample, *, full: bool = False) -> JudgeScores: ...


class _Metric(Protocol):
    async def ascore(self, **kwargs: Any) -> Any: ...


class RagasJudge:
    def __init__(self, model: str, metrics: Mapping[str, _Metric], hint: str = "") -> None:
        self._model = model
        self._metrics = metrics
        self._hint = hint
        # One loop for the judge's lifetime: the async HTTP client RAGAS drives keeps
        # connections bound to the loop it first ran on, so a new loop per call breaks it.
        # It runs on its own thread, so scoring also works where a loop is already running
        # (a Jupyter notebook), which would refuse a second loop on the same thread.
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, name="ragas-judge", daemon=True).start()

    @property
    def model(self) -> str:
        return self._model

    @property
    def metrics(self) -> Mapping[str, _Metric]:
        return self._metrics

    @classmethod
    def from_settings(cls, settings: Settings, embedder: Embedder) -> Self:
        """RAGAS metrics driven by the judge model; answer relevance reuses ``embedder``."""
        openai = import_optional("openai", extra="openai")
        ragas_llms = import_optional("ragas.llms", extra="evaluation")
        ragas_base = import_optional("ragas.embeddings.base", extra="evaluation")
        collections = import_optional("ragas.metrics.collections", extra="evaluation")

        options, hint = client_options(
            settings, settings.judge_provider, settings.judge_model, settings.judge_timeout_seconds
        )
        llm = ragas_llms.llm_factory(
            settings.judge_model,
            provider="openai",  # Ollama speaks the OpenAI API
            client=openai.AsyncOpenAI(**options),
            temperature=settings.judge_temperature,
            max_tokens=settings.judge_max_tokens,
        )
        embeddings = make_ragas_embeddings(embedder, ragas_base.BaseRagasEmbedding)
        metrics = {
            "faithfulness": collections.Faithfulness(llm=llm),
            "answer_relevance": collections.AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": collections.ContextPrecision(llm=llm),
            "context_recall": collections.ContextRecall(llm=llm),
        }
        return cls(settings.judge_model, metrics, hint)

    def score(self, sample: JudgeSample, *, full: bool = False) -> JudgeScores:
        names = CORE_METRICS + (CONTEXT_METRICS if full else ())
        future = asyncio.run_coroutine_threadsafe(self._score_all(sample, names), self._loop)
        return JudgeScores(**future.result())

    async def _score_all(
        self, sample: JudgeSample, names: tuple[str, ...]
    ) -> dict[str, float | None]:
        # One at a time: a local judge serves one request at a time anyway.
        return {name: await self._score_one(name, sample) for name in names}

    async def _score_one(self, name: str, sample: JudgeSample) -> float | None:
        try:
            result = await self._metrics[name].ascore(**_metric_inputs(name, sample))
        except Exception as exc:  # network, unknown model, unparsable judge output
            raise JudgeError(
                f"Judge metric '{name}' with model '{self._model}' failed: {exc}. {self._hint}"
            ) from exc
        value = float(result.value)
        return None if math.isnan(value) else value


def make_ragas_embeddings(embedder: Embedder, base: type) -> Any:
    """Wrap the project's embedder in the RAGAS embeddings interface (``base``)."""

    class _ProjectEmbeddings(base):  # type: ignore[misc]
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return list(embedder.embed_query(text))

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.embed_text(text)

    return _ProjectEmbeddings()


def _metric_inputs(name: str, sample: JudgeSample) -> dict[str, Any]:
    contexts = list(sample.contexts)
    if name == "faithfulness":
        return {"user_input": sample.question, "response": sample.answer,
                "retrieved_contexts": contexts}  # fmt: skip
    if name == "answer_relevance":
        return {"user_input": sample.question, "response": sample.answer}
    return {"user_input": sample.question, "reference": sample.reference,
            "retrieved_contexts": contexts}  # fmt: skip
