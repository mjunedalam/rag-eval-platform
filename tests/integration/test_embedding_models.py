"""Integration tests that load a real Sentence Transformers model.

Needs ``uv sync --extra local-embeddings``; the first run downloads the model (~90 MB).
"""

import math

import pytest

from rag_eval_platform.ingestion.embedding import SentenceTransformerEmbedder

pytestmark = pytest.mark.integration

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture(scope="module")
def embedder() -> SentenceTransformerEmbedder:
    pytest.importorskip("sentence_transformers")
    return SentenceTransformerEmbedder.from_pretrained(MODEL_NAME)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_vectors_are_normalised_with_expected_dimension(
    embedder: SentenceTransformerEmbedder,
) -> None:
    (vector,) = embedder.embed_documents(["Force equals mass times acceleration."])

    assert len(vector) == 384
    assert math.isclose(math.sqrt(dot(vector, vector)), 1.0, rel_tol=1e-4)


def test_query_is_closest_to_the_semantically_matching_passage(
    embedder: SentenceTransformerEmbedder,
) -> None:
    passages = [
        "Newton's second law states that force equals mass times acceleration.",
        "Photosynthesis converts sunlight into chemical energy in plants.",
        "The French Revolution began in 1789.",
    ]
    vectors = embedder.embed_documents(passages)

    query = embedder.embed_query("What is the relationship between force and acceleration?")

    similarities = [dot(query, v) for v in vectors]
    assert similarities.index(max(similarities)) == 0
