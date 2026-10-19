"""Centralised application settings.

Values come from environment variables prefixed with ``RAG_`` (or a local ``.env``
file); API keys use their conventional unprefixed names. Defaults are chosen so the
settings load with no environment at all, which keeps tests and CI key-free.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ChunkStrategy = Literal["fixed", "recursive", "semantic"]
EmbeddingProvider = Literal["sentence_transformers", "openai"]
VectorStoreBackend = Literal["chroma", "qdrant"]
LlmProvider = Literal["ollama", "openai", "anthropic"]
JudgeProvider = Literal["ollama", "openai"]
# "default" sends no reasoning_effort at all (for models that do not support the option).
ReasoningEffort = Literal["none", "low", "medium", "high", "default"]

# A metric score or threshold in [0, 1].
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class Settings(BaseSettings):
    """Immutable, validated configuration for every pipeline layer."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Ingestion
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    chunk_strategy: ChunkStrategy = "recursive"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)

    # Embedding
    embedding_provider: EmbeddingProvider = "sentence_transformers"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Vector store
    vector_store: VectorStoreBackend = "chroma"
    collection_name: str = "rag_documents"
    # Chroma runs as a server (docker/docker-compose.yml); 8001 keeps 8000 free for the API.
    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8001, gt=0, le=65535)
    qdrant_url: str = "http://localhost:6333"

    # Retrieval
    top_k: int = Field(default=5, gt=0)
    rerank: bool = False
    rerank_candidates: int = Field(default=20, gt=0)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation
    # Default: a local open-source model served by Ollama (no API key, no cost).
    llm_provider: LlmProvider = "ollama"
    llm_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1024, gt=0)
    llm_detailed_max_tokens: int = Field(default=2048, gt=0)  # the playground's detailed style
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    # "none" turns off hidden reasoning in thinking models like qwen3: ~4x faster answers.
    llm_reasoning_effort: ReasoningEffort = "none"
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_OPENAI_API_KEY")
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "RAG_ANTHROPIC_API_KEY")
    )

    # LLM judge for generation evaluation. It must differ from the model being evaluated,
    # to limit self-preference bias; changing it means re-baselining the scores.
    judge_provider: JudgeProvider = "ollama"
    judge_model: str = "gemma3:12b"
    judge_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    judge_max_tokens: int = Field(default=4096, gt=0)
    judge_timeout_seconds: float = Field(default=300.0, gt=0)

    # Evaluation thresholds (see docs/evaluation_methodology.md)
    golden_dataset_path: Path = Path("data/golden_dataset/qa_pairs.json")
    min_recall_at_k: Score = 0.80
    min_mrr: Score = 0.70
    min_ndcg_at_k: Score = 0.70
    min_faithfulness: Score = 0.85
    min_answer_relevance: Score = 0.80
    # Evaluation gate: the most a retrieval metric may drop below the committed baseline.
    gate_max_drop: Score = 0.02

    # API (Phase 6)
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, gt=0, lt=65536)
    api_key: SecretStr | None = None  # RAG_API_KEY; empty means auth is off
    api_rate_limit: int = Field(default=30, gt=0)  # requests per minute per client

    # Observability (Phase 7)
    log_level: str = "INFO"
    observe_enabled: bool = True
    observe_db: Path = Path("data/observability/rag.db")
    # Other observability logs the ⑧ Observe tab shows read-only (never written through this
    # process): by default, the Docker API's own rag-api.db. One writer per file — see
    # observability/store.py's docstring for why.
    observe_readonly_dbs: list[Path] = Field(
        default_factory=lambda: [Path("data/observability/rag-api.db")]
    )
    observe_store_text: bool = True  # false: store a SHA-256 of the question, no answer text
    price_input_per_1m: float = Field(default=0.0, ge=0.0)  # USD per 1M tokens (0 = local)
    price_output_per_1m: float = Field(default=0.0, ge=0.0)
    drift_window: int = Field(default=20, gt=0)  # answers per window: last N vs previous N
    drift_latency_rise: float = Field(default=0.5, gt=0)  # p95 may rise by 50%
    drift_error_rise: float = Field(default=0.10, gt=0)  # rates: absolute rise, 0.10 = 10 points
    drift_refusal_rise: float = Field(default=0.15, gt=0)
    drift_citation_rise: float = Field(default=0.10, gt=0)
    drift_downvote_rise: float = Field(default=0.20, gt=0)
    review_slow_ms: float = Field(default=30000.0, gt=0)

    # Langfuse tracing (Phase 7b). The keys use Langfuse's own unprefixed names and live only
    # in the git-ignored .env. Nothing is sent until tracing is enabled and both keys are set.
    langfuse_public_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY", "RAG_LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY", "RAG_LANGFUSE_SECRET_KEY"),
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_BASE_URL", "RAG_LANGFUSE_BASE_URL"),
    )
    langfuse_enabled: bool = True
    # Langfuse's rule for environment names: lowercase letters, digits, "-" and "_", and not
    # starting with "langfuse" (reserved).
    langfuse_environment: str = Field(
        default="development", pattern=r"^[a-z0-9_-]+$", max_length=40
    )

    @property
    def langfuse_ready(self) -> bool:
        """Tracing can actually send: switched on and both keys present."""
        return bool(self.langfuse_enabled and self.langfuse_public_key and self.langfuse_secret_key)

    @field_validator("langfuse_environment")
    @classmethod
    def _environment_not_reserved(cls, value: str) -> str:
        if value.startswith("langfuse"):
            raise ValueError("langfuse_environment must not start with 'langfuse' (reserved)")
        return value

    @field_validator("api_key", "langfuse_public_key", "langfuse_secret_key", mode="before")
    @classmethod
    def _empty_api_key_is_none(cls, value: object) -> object:
        return None if value in ("", None) else value

    @model_validator(mode="after")
    def _check_chunk_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def _check_judge_differs(self) -> Self:
        if (self.judge_provider, self.judge_model) == (self.llm_provider, self.llm_model):
            raise ValueError(
                f"the judge model ({self.judge_model}) must differ from the model being "
                "evaluated (RAG_LLM_MODEL); set RAG_JUDGE_MODEL to another model"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()
