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
    "RAG_LLM_PROVIDER",
    "RAG_LLM_TEMPERATURE",
    "RAG_LLM_MAX_TOKENS",
    "RAG_LLM_REASONING_EFFORT",
    "RAG_JUDGE_PROVIDER",
    "RAG_JUDGE_MODEL",
    "RAG_JUDGE_TEMPERATURE",
    "RAG_JUDGE_MAX_TOKENS",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_BASE_URL",
    "RAG_LANGFUSE_ENABLED",
    "RAG_LANGFUSE_ENVIRONMENT",
    "RAG_TRULENS_ENABLED",
    "RAG_TRULENS_JUDGE_MODEL",
    "RAG_TRULENS_QUEUE_SIZE",
    "RAG_LLM_MODEL",
    "RAG_OLLAMA_BASE_URL",
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
    assert (settings.chroma_host, settings.chroma_port) == ("localhost", 8001)
    assert settings.top_k == 5
    assert settings.rerank is False
    assert settings.golden_dataset_path == Path("data/golden_dataset/qa_pairs.json")
    assert settings.openai_api_key is None
    assert settings.anthropic_api_key is None


def test_generation_defaults_use_local_ollama() -> None:
    settings = Settings()

    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "qwen3:8b"
    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.llm_temperature == 0.0
    assert settings.llm_max_tokens == 1024
    assert settings.gate_max_drop == 0.02
    assert settings.llm_detailed_max_tokens == 2048
    assert settings.llm_timeout_seconds == 120.0
    assert settings.llm_reasoning_effort == "none"


def test_judge_defaults_use_a_different_local_model() -> None:
    settings = Settings()

    assert settings.judge_provider == "ollama"
    assert settings.judge_model == "gemma3:12b"
    assert settings.judge_model != settings.llm_model
    assert settings.judge_temperature == 0.0
    assert settings.judge_max_tokens == 4096
    assert settings.judge_timeout_seconds == 300.0


def test_judge_must_differ_from_the_model_being_evaluated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_JUDGE_MODEL", "qwen3:8b")

    with pytest.raises(ValidationError, match="judge"):
        Settings()


def test_same_model_name_on_another_provider_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_JUDGE_PROVIDER", "openai")
    monkeypatch.setenv("RAG_JUDGE_MODEL", "qwen3:8b")

    assert Settings().judge_provider == "openai"


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
        ("RAG_LLM_PROVIDER", "gemini"),
        ("RAG_LLM_TEMPERATURE", "3"),
        ("RAG_LLM_MAX_TOKENS", "0"),
        ("RAG_LLM_REASONING_EFFORT", "extreme"),
        ("RAG_JUDGE_PROVIDER", "anthropic"),
        ("RAG_JUDGE_TEMPERATURE", "-1"),
        ("RAG_JUDGE_MAX_TOKENS", "0"),
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


def test_api_defaults() -> None:
    settings = Settings(_env_file=None)

    assert (settings.api_host, settings.api_port) == ("127.0.0.1", 8000)
    assert settings.api_key is None
    assert settings.api_rate_limit == 30


def test_an_empty_api_key_means_auth_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "")

    assert Settings(_env_file=None).api_key is None


def test_an_api_key_is_kept_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_KEY", "fake-test-value")

    settings = Settings(_env_file=None)

    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "fake-test-value"
    assert "fake-test-value" not in repr(settings)


def test_observability_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_OBSERVE_DB", raising=False)
    monkeypatch.delenv("RAG_OBSERVE_READONLY_DBS", raising=False)

    settings = Settings(_env_file=None)

    assert settings.observe_enabled is True
    assert settings.observe_db == Path("data/observability/rag.db")
    assert settings.observe_readonly_dbs == [Path("data/observability/rag-api.db")]
    assert settings.observe_store_text is True
    assert (settings.price_input_per_1m, settings.price_output_per_1m) == (0.0, 0.0)
    assert settings.drift_window == 20
    assert settings.drift_latency_rise == 0.5
    assert settings.drift_error_rise == 0.10
    assert settings.drift_refusal_rise == 0.15
    assert settings.drift_citation_rise == 0.10
    assert settings.drift_downvote_rise == 0.20
    assert settings.review_slow_ms == 30000


def test_observability_settings_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_OBSERVE_ENABLED", "false")
    monkeypatch.setenv("RAG_PRICE_INPUT_PER_1M", "0.15")
    monkeypatch.setenv("RAG_DRIFT_WINDOW", "5")

    settings = Settings(_env_file=None)

    assert settings.observe_enabled is False
    assert settings.price_input_per_1m == 0.15
    assert settings.drift_window == 5


def test_observe_readonly_dbs_parses_a_json_list_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_OBSERVE_READONLY_DBS", '["a/rag-api.db", "b/other.db"]')

    settings = Settings(_env_file=None)

    assert settings.observe_readonly_dbs == [Path("a/rag-api.db"), Path("b/other.db")]


def test_observe_readonly_dbs_can_be_emptied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_OBSERVE_READONLY_DBS", "[]")

    settings = Settings(_env_file=None)

    assert settings.observe_readonly_dbs == []


@pytest.mark.parametrize(
    ("name", "value"),
    [("RAG_DRIFT_WINDOW", "0"), ("RAG_PRICE_OUTPUT_PER_1M", "-1"),
     ("RAG_DRIFT_ERROR_RISE", "0"), ("RAG_REVIEW_SLOW_MS", "0")],
)  # fmt: skip
def test_observability_settings_are_validated(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


class TestLangfuseSettings:
    def test_defaults_are_off_until_keys_are_set(self) -> None:
        settings = Settings()

        assert settings.langfuse_public_key is None
        assert settings.langfuse_secret_key is None
        assert settings.langfuse_base_url == "https://cloud.langfuse.com"
        assert settings.langfuse_enabled is True
        assert settings.langfuse_environment == "development"
        assert settings.langfuse_ready is False

    def test_reads_langfuses_standard_unprefixed_names_and_hides_the_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake-public-value")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake-secret-value")
        monkeypatch.setenv("LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")

        settings = Settings()

        assert settings.langfuse_public_key == "fake-public-value"
        assert settings.langfuse_secret_key is not None
        assert settings.langfuse_secret_key.get_secret_value() == "fake-secret-value"
        assert "fake-secret-value" not in repr(settings)
        assert settings.langfuse_base_url == "https://us.cloud.langfuse.com"
        assert settings.langfuse_ready is True

    def test_disabled_is_not_ready_even_with_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "fake-public-value")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "fake-secret-value")
        monkeypatch.setenv("RAG_LANGFUSE_ENABLED", "false")

        assert Settings().langfuse_ready is False

    @pytest.mark.parametrize("environment", ["Production", "my env", "langfuse-test", ""])
    def test_rejects_environment_names_langfuse_does_not_allow(
        self, monkeypatch: pytest.MonkeyPatch, environment: str
    ) -> None:
        monkeypatch.setenv("RAG_LANGFUSE_ENVIRONMENT", environment)

        with pytest.raises(ValidationError):
            Settings()


class TestTrulensSettings:
    def test_defaults_score_every_answer_with_a_small_local_judge(self) -> None:
        settings = Settings()

        assert settings.trulens_enabled is True
        assert settings.trulens_judge_model == "gemma3:4b"
        assert settings.trulens_queue_size == 100

    def test_judge_must_differ_from_the_generator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_TRULENS_JUDGE_MODEL", "qwen3:8b")

        with pytest.raises(ValidationError, match="RAG_TRULENS_JUDGE_MODEL"):
            Settings()

    def test_queue_size_must_be_positive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAG_TRULENS_QUEUE_SIZE", "0")

        with pytest.raises(ValidationError):
            Settings()

    @pytest.mark.parametrize(
        ("base_url", "root"),
        [
            ("http://localhost:11434/v1", "http://localhost:11434"),
            ("http://host.docker.internal:11434/v1/", "http://host.docker.internal:11434"),
            ("http://localhost:11434", "http://localhost:11434"),
        ],
    )
    def test_ollama_root_drops_the_openai_compatible_suffix(
        self, monkeypatch: pytest.MonkeyPatch, base_url: str, root: str
    ) -> None:
        monkeypatch.setenv("RAG_OLLAMA_BASE_URL", base_url)

        assert Settings().ollama_root_url == root
