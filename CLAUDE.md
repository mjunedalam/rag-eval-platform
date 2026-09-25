# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Phases 0 (setup) and 1 (ingestion) are done. `config/settings.py` and `config/logging_config.py` are implemented and tested, and `.github/workflows/ci.yml` runs lint, types, unit tests and a secret scan. The evaluation data and retrieval metrics are also in place. `data/raw/` holds a 14-document corpus on RAG and LLM evaluation; each document's filename is its doc id. `data/golden_dataset/qa_pairs.json` has 30 examples, and `evaluation/golden_dataset.py` and `evaluation/metrics.py` are implemented. A unit test fails if a golden example references a document that is missing from `data/raw/`, or if a corpus document has no golden question. So when you add, rename or remove a corpus document, update the golden dataset at the same time. Ingestion is implemented in `ingestion/`: `loaders`, `chunking`, `chunk_io`, `embedding` and `cli`, run through `scripts/run_ingestion.py`. Everything else is still a placeholder holding only a docstring or a `# TODO`: retrieval, generation, `evaluation/evaluator.py`, the observability and API modules, `scripts/run_evaluation.py`, `scripts/seed_vector_store.py`, `evaluation_gate.yml` and the Docker files. The intended design is written up in `docs/architecture.md` and `docs/evaluation_methodology.md`. Treat those two documents as the spec when implementing a module. The tools for each phase, and whether each is in use, chosen or only proposed, are listed in the "Tooling by phase" table in `docs/architecture.md`. Add a dependency only when its phase starts, and ask before adding a *proposed* one.

## Commands

The project uses [uv](https://docs.astral.sh/uv/), not pip. Python 3.12 is pinned in `.python-version`, and `requires-python` is `>=3.11`.

```bash
uv sync                                                  # create .venv, install project + dev group
uv sync --all-extras                                     # + optional embedding backends (torch, openai)
uv run python scripts/run_ingestion.py                   # data/raw -> data/processed/chunks.jsonl
uv run pytest                                            # all tests
uv run pytest tests/unit/test_chunking.py                # one file
uv run pytest tests/unit/test_chunking.py::test_name     # one test
uv run pytest tests/unit                                 # one tier: unit | integration | evaluation
uv run pytest -m integration                             # real-model tests (need --all-extras)
uv run pytest --cov                                      # with coverage (branch, missing lines)
uv run ruff check . && uv run ruff format --check .      # lint + format check (ruff format . to fix)
uv run mypy src tests                                    # strict type check
uv add <pkg>            # runtime dependency
uv add --dev <pkg>      # dev tool
```

Commit `uv.lock` together with any `pyproject.toml` change. CI runs `uv sync --locked`, so a stale lockfile fails the build. Before committing, run the same checks CI runs: Ruff lint, Ruff format, `mypy src tests` and the unit tests. mypy is `strict`, so every function needs full annotations. Pytest uses `--strict-markers`; the registered markers are `integration` and `evaluation`.

Configuration is read only through `get_settings()` (cached, frozen). Env vars use the `RAG_` prefix, e.g. `RAG_TOP_K=10`. The exceptions are `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`, which are unprefixed and held as `SecretStr`. Everything has a default, so tests and CI need no keys. Put local values in `.env` (copy it from `.env.example`); `.env` is git-ignored. Log with the standard `logging` module after calling `configure_logging()`, which writes one JSON object per line; pass structured fields with `extra={...}`.

## Architecture

The code uses a `src/` layout with the package at `src/rag_eval_platform/` (built with hatchling). It has two flows:

- **Offline ingestion** (`scripts/run_ingestion.py` → `scripts/seed_vector_store.py`): `ingestion/loaders` → `ingestion/chunking` → `ingestion/embedding` → `retrieval/vector_store`. Raw docs go in `data/raw/` and chunked output goes in `data/processed/`.
- **Online query** (FastAPI in `api/`, with `GET /health` and `POST /query`): embed the query with the *same* model used at ingestion → `retrieval/retriever` (top-k, with optional `retrieval/reranker` cross-encoder) → `generation/generator` → log through `observability/logger`.

No module wires the layers end to end yet. Design decisions that span several modules:

- **Pluggable backends through protocols.** Embedder, vector store, re-ranker and LLM client each sit behind an interface. Chroma is the local-dev store and Qdrant the production store. Business logic must not import a concrete backend directly.
- **Citations.** `generation/prompt_templates` numbers the retrieved chunks, and the LLM cites them as `[n]`. `generator` maps those markers back to source chunks. Citation validity is itself an evaluation metric.
- **Default chunking is recursive** (paragraph → line → word), using LangChain's `RecursiveCharacterTextSplitter`. Fixed-size with overlap is the alternative. `semantic` raises `NotImplementedError` for now. Sizes are in characters.
- **Ids flow end to end.** A `Document.id` is the file's path relative to the corpus root, e.g. `physics/02_newtons_laws.pdf`. A `Chunk.id` is `<doc_id>#<index>`, and each chunk keeps its `doc_id` so retrieval results can be scored at document level. PDFs record `page_starts`, so every chunk knows its 1-based `page` for citations. A PDF with no text layer (a scan) raises an error with an OCR hint.
- **Optional embedding backends.** `sentence-transformers` (with CPU-only torch on Linux) and `openai` are extras, not core dependencies, so CI never installs torch. `embedding.py` imports them lazily, and tests use fakes that satisfy the `Embedder` protocol. Tests that load a real model are marked `integration` and skip when the extra is missing. mypy skips these packages so local and CI results match.
- **Configuration.** Settings are typed and read from env / `.env` in `config/settings.py`. Chunk params, top-k, backends, model names and **evaluation thresholds** all live in config, never in code.
- **Access control and PII.** Permissions are applied as metadata filters *inside* the vector search, not after retrieval. PII is masked at ingestion, before embedding.

## Evaluation (the core of the project)

- **Golden dataset** (`data/golden_dataset/qa_pairs.json`, loaded and validated by `evaluation/golden_dataset.py`). Each item has these fields: `id`, `question`, `expected_answer`, `relevant_doc_ids`, `query_type`. Relevance is judged at the **document** level so the dataset stays valid when chunking changes. Before scoring, retrieved chunks are mapped to their doc id and de-duplicated in rank order.
- **Retrieval metrics** (`evaluation/metrics.py`): Precision@k, Recall@k, MRR, NDCG@k. They are deterministic, averaged, and also reported per `query_type`.
- **Generation metrics**: RAGAS (primary) and DeepEval (pytest-style), scored by a pinned LLM judge that is a different or stronger model than the one being evaluated. Changing the judge model or prompt means re-baselining.
- **Gate** (`evaluation/evaluator.py`, `scripts/run_evaluation.py`, `.github/workflows/evaluation_gate.yml`): fails the PR when any average is below its threshold. Starting thresholds are Recall@k 0.80, MRR 0.70, NDCG@k 0.70, Faithfulness 0.85, and Answer relevance 0.80. `ci.yml` is kept separate and covers lint plus unit tests.
- Retrieval or prompt changes (chunk size, strategy, top-k, embedding model, re-ranking, prompt wording) need a before/after metrics table in the PR.

## Secrets (hard rule)

- **Never commit `.env`.** The same goes for any `.env.*` other than `.env.example`, which must keep empty values, and for any file that contains a password, API key, token, private key or credential. Real values live only in the git-ignored `.env`.
- Stage files by name and read `git diff --cached` before every commit. Never bypass the local `.git/hooks/pre-commit` secret check with `--no-verify`.
- If a secret is ever committed, stop and remove it from git history; a follow-up delete commit is not enough. Then rotate the key.

## Conventions

- Commit messages use the conventional format `<type>: <description>`. Types: feat, fix, refactor, docs, test, chore, perf, ci, build.
- The README file is lowercase `readme.md`, and `pyproject.toml` refers to it by that name.
