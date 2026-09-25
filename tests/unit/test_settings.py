"""Tests for config.settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_eval_platform.config.settings import Settings, get_settings

SETTINGS_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_CHUNK_STRATEGY",
    "RAG_TOP_K",
    "RAG_VECTOR_STORE",
    "RAG_MIN_FAITHFULNESS",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate every test from the developer's shell env and local .env file."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


def test_defaults_load_without_any_environment() -> None:
    settings = Settings()

    assert settings.chunk_strategy == "recursive"
    assert settings.chunk_size == 800
    assert settings.chunk_overlap == 100
    assert settings.vector_store == "chroma"
    assert settings.top_k == 5
    assert settings.rerank is False
    assert settings.golden_dataset_path == Path("data/golden_dataset/qa_pairs.json")
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_default_thresholds_match_evaluation_methodology() -> None:
    settings = Settings()

    assert settings.min_recall_at_k == 0.80
    assert settings.min_mrr == 0.70
    assert settings.min_ndcg_at_k == 0.70
    assert settings.min_faithfulness == 0.85
    assert settings.min_answer_relevance == 0.80


def test_prefixed_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_TOP_K", "10")
    monkeypatch.setenv("RAG_VECTOR_STORE", "qdrant")
    monkeypatch.setenv("RAG_CHUNK_STRATEGY", "fixed")

    settings = Settings()

    assert settings.top_k == 10
    assert settings.vector_store == "qdrant"
    assert settings.chunk_strategy == "fixed"


def test_api_keys_read_from_unprefixed_env_and_stay_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-value")

    settings = Settings()

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "test-key-value"
    assert "test-key-value" not in repr(settings)


def test_values_are_read_from_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("RAG_TOP_K=7\n")

    assert Settings().top_k == 7


def test_settings_are_immutable() -> None:
    settings = Settings()

    with pytest.raises(ValidationError):
        settings.top_k = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    ("env_var", "value"),
    [
        ("RAG_TOP_K", "0"),
        ("RAG_CHUNK_SIZE", "0"),
        ("RAG_CHUNK_OVERLAP", "-1"),
        ("RAG_MIN_FAITHFULNESS", "1.5"),
        ("RAG_VECTOR_STORE", "pinecone"),
        ("RAG_CHUNK_STRATEGY", "unknown"),
    ],
)
def test_invalid_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, env_var: str, value: str
) -> None:
    monkeypatch.setenv(env_var, value)

    with pytest.raises(ValidationError):
        Settings()


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_CHUNK_SIZE", "100")
    monkeypatch.setenv("RAG_CHUNK_OVERLAP", "100")

    with pytest.raises(ValidationError, match="chunk_overlap"):
        Settings()


def test_get_settings_returns_cached_instance() -> None:
    assert get_settings() is get_settings()
