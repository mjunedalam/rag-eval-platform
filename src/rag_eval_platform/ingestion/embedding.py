"""Embedding providers.

Every provider implements the ``Embedder`` protocol, so the rest of the pipeline never
depends on a concrete backend. Both backends are optional installs:

- Sentence Transformers (default; local, free): ``uv sync --extra local-embeddings``
- OpenAI (hosted, needs ``OPENAI_API_KEY``): ``uv sync --extra openai``

Documents and queries must be embedded by the same model; switching models means
re-embedding the whole corpus.
"""

from collections.abc import Sequence
from typing import Any, Protocol, Self

from pydantic import SecretStr

from rag_eval_platform._optional import import_optional
from rag_eval_platform.config.settings import Settings

DEFAULT_LOCAL_BATCH_SIZE = 32
# OpenAI accepts up to 2048 inputs per request; smaller batches keep requests well under
# the per-request token limit.
DEFAULT_OPENAI_BATCH_SIZE = 256


class EmbeddingError(Exception):
    """An embedder could not be created or failed to embed."""


class Embedder(Protocol):
    """Turns chunk texts and queries into vectors in the same vector space."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class _Array(Protocol):
    def tolist(self) -> Any: ...


class _SentenceModel(Protocol):
    def encode_document(self, texts: list[str], **kwargs: Any) -> _Array: ...

    def encode_query(self, texts: list[str], **kwargs: Any) -> _Array: ...


class SentenceTransformerEmbedder:
    """Local embeddings with a Sentence Transformers model, e.g. all-MiniLM-L6-v2.

    Uses the model's document and query encoders, which apply any model-specific
    prompts (needed by families such as E5 and BGE), and normalises vectors to unit
    length so cosine similarity equals the dot product.
    """

    def __init__(self, model: _SentenceModel, batch_size: int = DEFAULT_LOCAL_BATCH_SIZE) -> None:
        self._model = model
        self._batch_size = batch_size

    @classmethod
    def from_pretrained(cls, model_name: str) -> Self:
        """Load a model by name (downloaded from Hugging Face on first use, then cached)."""
        module = import_optional("sentence_transformers", extra="local-embeddings")
        return cls(module.SentenceTransformer(model_name))

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode_document(list(texts), **self._encode_options())
        return [list(map(float, v)) for v in vectors.tolist()]

    def embed_query(self, text: str) -> list[float]:
        vectors = self._model.encode_query([text], **self._encode_options())
        return list(map(float, vectors.tolist()[0]))

    def _encode_options(self) -> dict[str, Any]:
        return {
            "batch_size": self._batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }


class _EmbeddingItem(Protocol):
    index: int
    embedding: list[float]


class _EmbeddingResponse(Protocol):
    data: Sequence[_EmbeddingItem]


class _EmbeddingsApi(Protocol):
    def create(self, *, input: Sequence[str], model: str) -> _EmbeddingResponse: ...


class _OpenAIClient(Protocol):
    @property
    def embeddings(self) -> _EmbeddingsApi: ...


class OpenAIEmbedder:
    """Hosted embeddings from the OpenAI API, e.g. text-embedding-3-small."""

    def __init__(
        self, client: _OpenAIClient, model: str, batch_size: int = DEFAULT_OPENAI_BATCH_SIZE
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    @classmethod
    def from_api_key(cls, api_key: SecretStr | None, model: str) -> Self:
        if api_key is None:
            raise EmbeddingError("OPENAI_API_KEY is not set (add it to .env)")
        module = import_optional("openai", extra="openai")
        return cls(module.OpenAI(api_key=api_key.get_secret_value()), model)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = self._client.embeddings.create(input=batch, model=self._model)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda i: i.index))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedder(settings: Settings) -> Embedder:
    """Build the embedder selected by ``RAG_EMBEDDING_PROVIDER`` / ``RAG_EMBEDDING_MODEL``."""
    if settings.embedding_provider == "openai":
        return OpenAIEmbedder.from_api_key(settings.openai_api_key, settings.embedding_model)
    return SentenceTransformerEmbedder.from_pretrained(settings.embedding_model)
