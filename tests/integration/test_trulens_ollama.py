"""The real TruLens triad with the real judge on Ollama (skips when either is missing)."""

import json
import urllib.request

import pytest

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.observability.trulens_scorer import create_triad_scorer

pytestmark = pytest.mark.integration

CONTEXT = (
    "Mean Reciprocal Rank (MRR) averages, over all queries, one divided by the rank of the "
    "first relevant document. A first relevant result at rank 2 contributes 0.5."
)


@pytest.fixture
def settings() -> Settings:
    pytest.importorskip("trulens.providers.litellm")
    settings = Settings(_env_file=None, trulens_enabled=True)
    try:
        with urllib.request.urlopen(f"{settings.ollama_root_url}/api/tags", timeout=3) as response:  # noqa: S310 - local URL
            names = {m["name"] for m in json.load(response)["models"]}
    except OSError:
        pytest.skip("Ollama is not running")
    if settings.trulens_judge_model not in names | {n.removesuffix(":latest") for n in names}:
        pytest.skip(f"judge {settings.trulens_judge_model} is not pulled")
    return settings


def test_a_grounded_on_topic_answer_scores_high_and_an_off_topic_one_low(
    settings: Settings,
) -> None:
    scorer = create_triad_scorer(settings)
    assert scorer is not None

    good = {s.metric: s for s in scorer.score(
        "What does MRR measure?", [CONTEXT],
        "MRR measures how high the first relevant document is ranked, averaged over queries [1].",
    )}  # fmt: skip
    bad = {s.metric: s for s in scorer.score(
        "What does MRR measure?", ["Paris is the capital of France."],
        "The Eiffel Tower is 330 metres tall.",
    )}  # fmt: skip

    for scores in (good, bad):
        assert set(scores) == {"context_relevance", "groundedness", "answer_relevance"}
        assert all(0.0 <= s.value <= 1.0 for s in scores.values())
    assert good["context_relevance"].value > bad["context_relevance"].value
    assert good["answer_relevance"].value > bad["answer_relevance"].value
    assert good["groundedness"].value >= 0.5
    assert good["groundedness"].reason
