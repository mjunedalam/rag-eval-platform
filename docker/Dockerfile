# syntax=docker/dockerfile:1
# The RAG API: multi-stage uv build, CPU-only torch (see pyproject), non-root, model baked in.
# Build from the repo root: docker build -f docker/Dockerfile -t rag-eval-api .

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 HF_HOME=/app/.hf
WORKDIR /app
# Dependencies first, so code changes don't reinstall torch.
COPY pyproject.toml uv.lock readme.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra local-embeddings --extra openai
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra local-embeddings --extra openai
# Download the embedding model now, so the container needs no internet.
RUN .venv/bin/python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

FROM python:3.12-slim-bookworm
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/app/.hf HF_HUB_OFFLINE=1 \
    RAG_API_HOST=0.0.0.0 RAG_API_PORT=8000 PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["python", "-m", "rag_eval_platform.api.serve"]
