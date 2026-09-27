"""DeepEval checks: real answers from the pipeline, graded by the pinned judge model.

One golden question per query type goes through the full pipeline (Chroma -> qwen3:8b) and
must pass DeepEval's faithfulness and answer-relevancy metrics at the thresholds from
settings, graded by ``RAG_JUDGE_MODEL`` over Ollama's OpenAI-compatible API.

Run with ``uv run pytest -m evaluation`` (a few minutes). Needs Chroma seeded, Ollama running
with both models pulled, and ``uv sync --all-extras``; skipped otherwise.
The full 30-question report comes from ``scripts/run_generation_evaluation.py``.
"""

import json
import os
import urllib.request
from typing import Any

import pytest

from rag_eval_platform.config.settings import Settings, get_settings
from rag_eval_platform.evaluation.golden_dataset import GoldenExample, load_golden_dataset
from rag_eval_platform.generation.generator import OLLAMA_PLACEHOLDER_KEY
from rag_eval_platform.pipeline import RagPipeline, create_pipeline
from rag_eval_platform.retrieval.vector_store import VectorStoreError

pytestmark = pytest.mark.evaluation

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")


def _one_per_query_type() -> list[GoldenExample]:
    seen: dict[str, GoldenExample] = {}
    for example in load_golden_dataset(get_settings().golden_dataset_path):
        seen.setdefault(example.query_type, example)
    return list(seen.values())


SAMPLE = _one_per_query_type()


@pytest.fixture(scope="module")
def settings() -> Settings:
    pytest.importorskip("deepeval")
    pytest.importorskip("openai")
    settings = get_settings()
    tags_url = settings.ollama_base_url.removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=3) as response:  # noqa: S310 - local URL
            models = {m["name"] for m in json.load(response)["models"]}
    except OSError:
        pytest.skip("Ollama is not running")
    for model in (settings.llm_model, settings.judge_model):
        if model not in models:
            pytest.skip(f"model {model} is not pulled")
    return settings


@pytest.fixture(scope="module")
def pipeline(settings: Settings) -> RagPipeline:
    try:
        pipeline = create_pipeline(settings)
        pipeline.retriever.retrieve("health check")
    except VectorStoreError as exc:
        pytest.skip(f"Chroma is not ready: {exc}")
    return pipeline


@pytest.fixture(scope="module")
def judge(settings: Settings) -> Any:
    from deepeval.models import LocalModel

    return LocalModel(
        model=settings.judge_model,
        base_url=settings.ollama_base_url,
        api_key=OLLAMA_PLACEHOLDER_KEY,
        temperature=settings.judge_temperature,
    )


@pytest.mark.parametrize("example", SAMPLE, ids=[f"{e.query_type}-{e.id}" for e in SAMPLE])
def test_answer_is_faithful_and_relevant(
    example: GoldenExample, pipeline: RagPipeline, judge: Any, settings: Settings
) -> None:
    from deepeval import assert_test
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    answer = pipeline.ask(example.question)
    assert not answer.is_refusal, f"refused an answerable golden question: {example.id}"

    test_case = LLMTestCase(
        input=example.question,
        actual_output=answer.text,
        expected_output=example.expected_answer,
        retrieval_context=[source.chunk.text for source in answer.sources],
    )
    assert_test(
        test_case,
        [
            FaithfulnessMetric(threshold=settings.min_faithfulness, model=judge, async_mode=False),
            AnswerRelevancyMetric(
                threshold=settings.min_answer_relevance, model=judge, async_mode=False
            ),
        ],
    )
