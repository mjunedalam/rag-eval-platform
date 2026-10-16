"""The API image must install every extra the API imports at startup."""

from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile"
COMPOSE = DOCKERFILE.parent / "docker-compose.yml"


def test_the_image_installs_the_embedding_and_llm_client_extras() -> None:
    syncs = [line for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
             if "uv sync" in line]  # fmt: skip

    assert syncs
    for line in syncs:
        assert "--extra local-embeddings" in line  # embeds each question
        assert "--extra openai" in line  # the OpenAI-compatible client that talks to Ollama


def test_the_api_container_can_write_the_observability_log() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    compose = COMPOSE.read_text(encoding="utf-8")

    assert "mkdir -p /app/data/observability" in dockerfile
    assert "chown app:app /app/data/observability" in dockerfile
    assert "../data/observability:/app/data/observability" in compose


def test_the_api_container_writes_its_own_observability_file() -> None:
    """One writer per SQLite file: the container never shares rag.db with the host playground
    (Docker Desktop's bind mount does not carry SQLite's POSIX locks between the Mac and the
    Linux VM), so it is pointed at its own rag-api.db inside the same mounted folder."""
    compose = COMPOSE.read_text(encoding="utf-8")

    assert "RAG_OBSERVE_DB: /app/data/observability/rag-api.db" in compose
