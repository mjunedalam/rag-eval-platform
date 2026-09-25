"""Tests for ingestion.embedding (with fake models; no downloads or API calls)."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import SecretStr

from rag_eval_platform.config.settings import Settings
from rag_eval_platform.ingestion import embedding
from rag_eval_platform.ingestion.embedding import (
    EmbeddingError,
    OpenAIEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)


class FakeArray:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


@dataclass
class FakeSentenceModel:
    calls: list[tuple[str, list[str], dict[str, Any]]] = field(default_factory=list)

    def _encode(self, kind: str, texts: list[str], **kwargs: Any) -> FakeArray:
        self.calls.append((kind, texts, kwargs))
        return FakeArray([[float(len(t)), 1.0] for t in texts])

    def encode_document(self, texts: list[str], **kwargs: Any) -> FakeArray:
        return self._encode("document", texts, **kwargs)

    def encode_query(self, texts: list[str], **kwargs: Any) -> FakeArray:
        return self._encode("query", texts, **kwargs)


class TestSentenceTransformerEmbedder:
    def test_embeds_documents_with_document_encoder_normalised(self) -> None:
        model = FakeSentenceModel()

        vectors = SentenceTransformerEmbedder(model, batch_size=8).embed_documents(["ab", "abc"])

        assert vectors == [[2.0, 1.0], [3.0, 1.0]]
        kind, texts, kwargs = model.calls[0]
        assert (kind, texts) == ("document", ["ab", "abc"])
        assert kwargs["normalize_embeddings"] is True
        assert kwargs["batch_size"] == 8

    def test_embeds_query_with_query_encoder(self) -> None:
        model = FakeSentenceModel()

        vector = SentenceTransformerEmbedder(model).embed_query("abcd")

        assert vector == [4.0, 1.0]
        assert model.calls[0][0] == "query"

    def test_empty_input_skips_the_model(self) -> None:
        model = FakeSentenceModel()

        assert SentenceTransformerEmbedder(model).embed_documents([]) == []
        assert model.calls == []


@dataclass
class FakeEmbeddingItem:
    index: int
    embedding: list[float]


@dataclass
class FakeOpenAIClient:
    calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def embeddings(self) -> "FakeOpenAIClient":
        return self

    def create(self, *, input: Sequence[str], model: str) -> Any:
        self.calls.append({"input": list(input), "model": model})
        # the API may return items out of order; the embedder must sort by index
        items = [FakeEmbeddingItem(i, [float(len(t))]) for i, t in enumerate(input)]
        return type("Response", (), {"data": list(reversed(items))})()


class TestOpenAIEmbedder:
    def test_batches_requests_and_keeps_input_order(self) -> None:
        client = FakeOpenAIClient()
        embedder = OpenAIEmbedder(client, model="text-embedding-3-small", batch_size=2)

        vectors = embedder.embed_documents(["a", "bb", "ccc"])

        assert vectors == [[1.0], [2.0], [3.0]]
        assert [c["input"] for c in client.calls] == [["a", "bb"], ["ccc"]]
        assert {c["model"] for c in client.calls} == {"text-embedding-3-small"}

    def test_embeds_query(self) -> None:
        assert OpenAIEmbedder(FakeOpenAIClient(), model="m").embed_query("abcd") == [4.0]

    def test_empty_input_makes_no_request(self) -> None:
        client = FakeOpenAIClient()

        assert OpenAIEmbedder(client, model="m").embed_documents([]) == []
        assert client.calls == []

    def test_from_api_key_requires_a_key(self) -> None:
        with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
            OpenAIEmbedder.from_api_key(None, model="m")


class TestCreateEmbedder:
    def test_builds_sentence_transformer_embedder_from_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        loaded: list[str] = []
        sentinel = object()

        def fake_from_pretrained(model_name: str) -> object:
            loaded.append(model_name)
            return sentinel

        monkeypatch.setattr(SentenceTransformerEmbedder, "from_pretrained", fake_from_pretrained)

        result = create_embedder(Settings(embedding_model="my/model"))

        assert result is sentinel
        assert loaded == ["my/model"]

    def test_builds_openai_embedder_from_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received: dict[str, Any] = {}
        sentinel = object()

        def fake_from_api_key(api_key: SecretStr | None, model: str) -> object:
            received.update(api_key=api_key, model=model)
            return sentinel

        monkeypatch.setattr(OpenAIEmbedder, "from_api_key", fake_from_api_key)
        settings = Settings(
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
            openai_api_key=SecretStr("test-key"),
        )

        assert create_embedder(settings) is sentinel
        assert received["model"] == "text-embedding-3-small"
        assert received["api_key"].get_secret_value() == "test-key"


def test_missing_optional_package_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> None:
        raise ModuleNotFoundError(f"No module named {name!r}")

    monkeypatch.setattr(embedding, "import_module", fail_import)

    with pytest.raises(EmbeddingError, match="uv sync --extra local-embeddings"):
        SentenceTransformerEmbedder.from_pretrained("any/model")


def test_from_api_key_builds_a_real_client_without_network() -> None:
    pytest.importorskip("openai")

    embedder = OpenAIEmbedder.from_api_key(SecretStr("test-key"), model="text-embedding-3-small")

    assert isinstance(embedder, OpenAIEmbedder)
