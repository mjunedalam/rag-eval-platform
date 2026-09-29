"""The API image must install every extra the API imports at startup."""

from pathlib import Path

DOCKERFILE = Path(__file__).resolve().parents[2] / "docker" / "Dockerfile"


def test_the_image_installs_the_embedding_and_llm_client_extras() -> None:
    syncs = [line for line in DOCKERFILE.read_text(encoding="utf-8").splitlines()
             if "uv sync" in line]  # fmt: skip

    assert syncs
    for line in syncs:
        assert "--extra local-embeddings" in line  # embeds each question
        assert "--extra openai" in line  # the OpenAI-compatible client that talks to Ollama
