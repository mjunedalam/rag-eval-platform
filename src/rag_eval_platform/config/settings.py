"""Centralised application settings.

Values come from environment variables prefixed with ``RAG_`` (or a local ``.env``
file); API keys use their conventional unprefixed names. Defaults are chosen so the
settings load with no environment at all, which keeps tests and CI key-free.
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ChunkStrategy = Literal["fixed", "recursive", "semantic"]
EmbeddingProvider = Literal["sentence_transformers", "openai"]
VectorStoreBackend = Literal["chroma", "qdrant"]
LlmProvider = Literal["anthropic", "openai"]

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
    chroma_path: Path = Path(".chroma")
    qdrant_url: str = "http://localhost:6333"

    # Retrieval
    top_k: int = Field(default=5, gt=0)
    rerank: bool = False
    rerank_candidates: int = Field(default=20, gt=0)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Generation
    llm_provider: LlmProvider = "anthropic"
    llm_model: str = "claude-sonnet-5"
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_OPENAI_API_KEY")
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY", "RAG_ANTHROPIC_API_KEY")
    )

    # Evaluation thresholds (see docs/evaluation_methodology.md)
    golden_dataset_path: Path = Path("data/golden_dataset/qa_pairs.json")
    min_recall_at_k: Score = 0.80
    min_mrr: Score = 0.70
    min_ndcg_at_k: Score = 0.70
    min_faithfulness: Score = 0.85
    min_answer_relevance: Score = 0.80

    # Observability
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_chunk_overlap(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return Settings()
