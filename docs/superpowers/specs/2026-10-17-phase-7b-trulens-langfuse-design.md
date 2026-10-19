# Phase 7b — TruLens and Langfuse: design

**Status:** implemented. Langfuse tracing verified against Langfuse Cloud (2026-10-18, PR #12); TruLens scoring verified end to end with `gemma3:4b` (2026-10-19).
**Branch:** `feat/phase-7b-trulens-langfuse`
**Follows:** Phase 7a (observability core: SQLite log, ⑧ Observe, feedback, drift). **Followed by:** Phase 8 (security and governance).

## Goal

Add the two observability tools teams use for LLM applications, each for what it does best, on top of the Phase 7a SQLite log:

- **Langfuse** for *tracing*: every question becomes a trace with live spans (retrieve, re-rank, generate), the prompt, tokens, timings and scores, viewed in Langfuse's web UI.
- **TruLens** for *feedback functions*: its RAG triad (context relevance, groundedness, answer relevance) scores every answer automatically.

The SQLite log stays the single source of truth. Both tools are add-ons that can be switched off, and a failure in either never breaks or slows an answer.

**Success looks like:** ask a question in the playground, the API or `ask.py`; open Langfuse Cloud and see the full trace with its spans and scores; see the TruLens triad scores in ⑧ Observe a minute or two later. CI stays green without either tool installed.

## Decisions already made

| Topic | Decision |
|---|---|
| Langfuse hosting | **Langfuse Cloud free tier** (keys in `.env`). Self-hosting (six containers, 4–6 GB RAM) was rejected for this machine. The host is a setting, so a self-hosted Langfuse works later with no code change. |
| What Langfuse receives | **Full text everywhere**, including the playground: question, answer, retrieved chunk text and the prompt. The privacy switch from 7a still applies: with `RAG_OBSERVE_STORE_TEXT=false`, Langfuse gets SHA-256 hashes instead of text and no prompt. The playground shows that uploads are sent to Langfuse Cloud while it is enabled. |
| When TruLens scores | **Every answer, automatically**, on a background worker after the answer is delivered, so answers keep their normal speed. |
| TruLens judge | **`gemma3:4b`** on Ollama: about 4 GB, so it stays loaded next to `qwen3:8b` (about 10.5 GB together on 18 GB) and models are not swapped. The careful Phase 4 evaluation and the CI gate keep `gemma3:12b`. Settings reject a TruLens judge equal to the generator. |
| Integration style | **Thin adapters behind interfaces** (approach 1 of 3). Native `@observe`/`TruApp` instrumentation was rejected because it couples the core to two vendors and adds a second database; a pure OpenTelemetry route was rejected as more configuration for less learning value. |
| TruLens dashboard | Still excluded (it pins `plotly<7`; the playground needs `plotly>=7.1`). Scores are shown in ⑧ Observe, as the 7a spec decided. |

## Revisions from the official Langfuse guidance

Added after installing the official Langfuse agent skill (github.com/langfuse/skills) and reading the current docs (best practices, OpenAI Python integration, Python SDK instrumentation, masking). These override the matching details further down:

- **The LLM call uses the official `langfuse.openai` integration**, not a hand-built generation span: our `OpenAICompatibleClient` is created from `langfuse.openai` when tracing is on. It records model, prompt, output (including any thinking), tokens, cost, streaming and time to first token, and API errors, and nests under the active observation.
- **Names are verb-first and stable:** trace root `answer-question`; children `retrieve-context` (type `retriever`), `rerank-chunks` (type `retriever`, only with re-ranking) and `generate-answer` (type `generation`, from the integration). Run-specific values go in metadata, never in names.
- **Trace input/output** are set on the root observation: the question, and the answer.
- **Tags** (set at creation, immutable): the source (`api`, `cli`, `playground`) and the answer style.
- **Environment and release:** `environment` from `RAG_LANGFUSE_ENVIRONMENT` (default `development`), `release` = the package version.
- **Settings use Langfuse's standard names** `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` (replacing `RAG_LANGFUSE_HOST`; this project uses the US region, `https://us.cloud.langfuse.com`). The client is built from typed settings with explicit arguments, so it never depends on import order or on `os.environ`.
- **Text hiding** for `RAG_OBSERVE_STORE_TEXT=false` uses the SDK's masking hook instead of custom hashing.
- **Context across threads:** the playground streams on a background thread; that thread runs inside a copy of the caller's context (`contextvars`), so the generation stays nested in the question's trace.
- **Seeded trace ids without a fake parent (found in the self-audit):** passing `trace_context={"trace_id": ...}` without a parent span makes the SDK invent a random remote parent, so Langfuse reported no logical root. The root now gets the seeded id from a custom OpenTelemetry id generator passed to the client (`id_generator=`), and is created with no `trace_context`; a re-fetched trace shows `answer-question` as its logical root.
- **TruLens as built (2026-10-19):** TruLens 2.16. The SQLite log keeps only chunk ids, so the worker fetches the chunks' text from the answer's own Chroma collection (`ChromaVectorStore.texts`). Groundedness reasons are a per-claim list (`{"reasons": [{criteria, supporting_evidence, score}]}`), shown one claim per line; found by the real-judge integration test. Answers stored with `RAG_OBSERVE_STORE_TEXT=false` are skipped (no text to judge). The catch-up command also exits 1 when some answers still failed. Langfuse's legacy read APIs are gone for organisations created after 2026-09-16; anything reading back must use `v2/observations` and `v3/scores`.
- **Required self-audit:** after instrumenting, send a real trace, fetch it back with the Langfuse CLI, check it against the best-practices page, fix gaps and repeat until clean.

## Facts that shaped the design

Checked against the real packages (Langfuse SDK 5.0.2, TruLens 2.15.0):

- Langfuse cannot back-date a span's **start**: `start_observation` has no start-time argument; only `end(end_time=...)` exists. So traces are built **live**, while each stage runs, not reconstructed afterwards.
- `Langfuse.create_trace_id(seed=...)` derives a stable trace id from any string. Seeding it with the SQLite `query_id` lets late scores (TruLens, 👍/👎) attach to the right trace with nothing extra stored.
- The packages resolve with the project: 24 new packages; existing ones change only `openai` 3.3 → 2.54 (LiteLLM caps it below 3; the 2.x chat and streaming API is what the generator already uses), `rich` 14.3 → 13.9 and `filelock` 4.0 → 3.32. Plotly 7.1, RAGAS, DeepEval and Streamlit are unchanged.

## Architecture

```mermaid
flowchart LR
    Q["question"] --> T["Tracer.start(query_id)"]
    T --> R["retrieve span"] --> RR["rerank span"] --> G["generate span"] --> A["answer"]
    A --> L["QueryLogger.record(query_id)<br/>SQLite: source of truth"]
    L --> W["ScoreWorker<br/>background thread"]
    W --> J["TruLens triad<br/>gemma3:4b via LiteLLM"]
    J --> S["SQLite trulens_scores"]
    J --> LF["Langfuse trace scores"]
    T -.-> LF
    F["user feedback"] --> LF
    S --> O["⑧ Observe tab"]
```

### New modules (`src/rag_eval_platform/observability/`)

| Module | Responsibility | Depends on |
|---|---|---|
| `tracing.py` | The vendor-neutral interface: `Tracer.start(query_id, question, metadata) -> TraceHandle`; `TraceHandle.span(name, input) -> SpanHandle`, `TraceHandle.finish(answer=..., error=...)`; `SpanHandle.end(output=..., error=..., details)`. `NoopTracer` is the default. Handles are explicit objects (not "current span" context), so they are safe across the playground's background stream thread. | Nothing |
| `langfuse_tracer.py` | `LangfuseTracer` implements `Tracer` with the Langfuse SDK; trace id = `create_trace_id(seed=query_id)`. `score(query_id, name, value, comment)` attaches a score to that trace. `trace_url(query_id)` for links. Hashes text when `RAG_OBSERVE_STORE_TEXT=false`. Every SDK call is wrapped: errors log a warning and are swallowed. `create_tracer(settings)` returns `NoopTracer` when disabled, keys are missing or the package is not installed. | `langfuse` (lazy import) |
| `trulens_scorer.py` | `TriadScorer.score(question, contexts, answer) -> TriadScores` using TruLens' LiteLLM provider against Ollama (`ollama/<judge>`, with the Ollama address derived from `RAG_OLLAMA_BASE_URL` minus its `/v1` suffix): context relevance (mean over retrieved chunks), groundedness, answer relevance, each 0–1 with the judge's reason. | `trulens-*` (lazy import) |
| `score_worker.py` | `ScoreWorker`: one background thread and a bounded queue of `query_id`s. For each: load the query from SQLite, skip refusals and errors, score, save rows to `trulens_scores`, send the three scores to Langfuse. Never raises; a full queue drops the job with a warning. `stop()` drains politely on shutdown. | Scorer, store, tracer |

### Changes to existing code

- **Query id at the start.** `RagPipeline` and the playground create the `query_id` when the question arrives and pass it to the tracer and to `QueryLogger.record(..., query_id=...)` / `record_error(..., query_id=...)` (new optional argument; the logger still creates one when it is not given). The SQLite row and the Langfuse trace share one id.
- **Spans.** `RagPipeline.ask_timed`, `ask_stream` and the playground's live run open spans around retrieval, re-ranking and generation; streaming records the time to first token and ends the generate span when the stream finishes. `Retriever` gains an optional hook so re-ranking is its own span.
- **Logger hand-off.** `QueryLogger` gets an optional `on_recorded` callback; the factories wire it to the process's `ScoreWorker`.
- **Feedback.** `record_feedback` also sends `user_feedback` (1 or 0, with the comment) to Langfuse.
- **Store schema v2 → v3.** New table `trulens_scores (query_id, metric, value, reason, judge, ts)`, primary key `(query_id, metric)`; the v2 file upgrades in place, as v1 → v2 did. `LogSet` merges it like the other tables.

## Where tracing and scoring run

| Path | Trace | TruLens |
|---|---|---|
| API (`/query`, `/query/stream`) | Live trace per request | Background worker started in the app lifespan; stopped (and Langfuse flushed) on shutdown |
| Playground chat | Live trace per question | One worker per Streamlit server (`st.cache_resource`) |
| `ask.py` | Live trace; Langfuse flushed before exit | **Not scored in the background** (the process exits at once); the catch-up command scores it |
| API in Docker | Live trace (keys passed from `.env` through compose `env_file`, optional) | Worker in the container; judge reached at `host.docker.internal`, like the generator; the image includes the `monitoring` extra |

**Catch-up command.** `scripts/run_trulens.py --pending` scores every logged answer (not a refusal or error) that has no TruLens scores yet; `--last N` limits it. It covers `ask.py`, a full queue, and anything lost when a process stopped. Exit code 0 on success, 2 when it cannot run.

## What each tool receives

**Langfuse trace** (`name="rag-query"`):
- Trace input/output: question and answer. Tags: source (`api`, `cli`, `playground`). Metadata: collection, answer style, prompt version, top-k, re-ranking on/off.
- `retrieve` span (type `retriever`): the retrieved chunks (id, document, page, score, text).
- `rerank` span (only with re-ranking): candidate order before and after.
- `generate` span (type `generation`): model, parameters (temperature, max tokens, reasoning effort), the full prompt (system and user messages), the answer, input/output tokens, completion start time (first token).
- Errors: the failing span ends with level `ERROR` and the message. This goes only to the user's own Langfuse project, never to an API caller.
- `RAG_OBSERVE_STORE_TEXT=false`: hashes instead of question, answer and chunk text; no prompt.

**Scores on the trace:** `trulens_context_relevance`, `trulens_groundedness`, `trulens_answer_relevance` (0–1, reason as comment), and `user_feedback` (1 for 👍, 0 for 👎, with the comment).

**TruLens** receives nothing remote: it runs locally against Ollama.

## Settings

All have defaults; `.env.example` documents them with empty secret values.

| Setting | Default | Notes |
|---|---|---|
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` | empty | Langfuse's standard names, unprefixed like the other keys; the secret is a `SecretStr`. Only in `.env`. |
| `LANGFUSE_BASE_URL` | `https://cloud.langfuse.com` | Langfuse's standard name. EU by default; this project's `.env` uses `https://us.cloud.langfuse.com` |
| `RAG_LANGFUSE_ENVIRONMENT` | `development` | Langfuse environment, so test traces stay apart from production ones |
| `RAG_LANGFUSE_ENABLED` | `true` | Does nothing until both keys are set |
| `RAG_TRULENS_ENABLED` | `true` | Background scoring on or off |
| `RAG_TRULENS_JUDGE_MODEL` | `gemma3:4b` | Rejected if equal to `RAG_LLM_MODEL` |
| `RAG_TRULENS_QUEUE_SIZE` | `100` | Jobs beyond it are dropped with a warning |

## Error handling

- Tracing: every Langfuse call is wrapped; a failure logs a warning once per kind and the answer continues. The SDK sends in the background; `flush()` runs on API shutdown, after `ask.py` answers, and at playground exit.
- Scoring: a failed score saves no row, so the catch-up command retries it later. The worker thread catches everything; it never takes the process down.
- Missing pieces degrade quietly: no keys, the extra not installed, Ollama or the judge model missing → tracing or scoring is off, and the ⑧ tab says why.
- Secrets: keys never appear in logs, traces or error messages. The pre-commit hook and gitleaks keep guarding commits; CI never needs the keys.

## ⑧ Observe tab and header

- Header status pills: Langfuse (on, off or no keys) and TruLens (on, off, judge missing).
- Recent questions table: three TruLens columns (blank while pending).
- A triad tile (averages over the selected range) and a triad trend chart.
- Answer detail: the three scores with the judge's reasons, and **Open trace in Langfuse** (`trace_url`).
- Playground chat: a one-line note under the input while Langfuse is enabled: uploads and answers are sent to Langfuse Cloud.

## Dependencies

A new optional extra, `monitoring`: `trulens-core`, `trulens-feedback`, `trulens-providers-litellm`, `langfuse`. CI does not install it. mypy skips these packages (as it does `sentence_transformers` and `openai`), and every import is lazy through `import_optional`.

## Testing

**Unit (fakes, run in CI):**
- `NoopTracer` and the tracing interface; `LangfuseTracer` against a fake client that records calls: trace id seeded from `query_id`, spans and their order, generation details, hashing with `STORE_TEXT=false`, swallowed SDK errors, `NoopTracer` when keys or package are missing.
- `TriadScorer` with a fake provider: averaging context relevance, reasons kept, values clamped to 0–1.
- `ScoreWorker`: processes queued ids, skips refusals and errors, saves rows, sends scores, drops on a full queue, survives scorer and store errors, `stop()` drains.
- Store v2 → v3 upgrade in place; `trulens_scores` round trip; `LogSet` merge.
- Settings: judge equal to the generator rejected; defaults.
- Pipeline and playground: the same `query_id` reaches the tracer and the logger; spans for each stage; stream ends the generate span; errors end spans with `ERROR`.
- Feedback sends `user_feedback`; the catch-up command picks only unscored answers.

**Integration (skip when unavailable, never in CI):**
- Real TruLens triad with `gemma3:4b` on a tiny example (skips without Ollama or the model).
- Real Langfuse Cloud round trip (skips unless both keys are set).

## Docs

`docs/learning/phase-7b.md` (simple English with analogies, the concepts and a file-by-file code walkthrough, as in earlier phases), plus the readme (Phase 7 status, commands, setup), `CLAUDE.md`, the architecture tooling table (TruLens and Langfuse → in use) and `.env.example`.

## Out of scope

Self-hosting Langfuse; Langfuse prompt management and datasets; TruLens' own database, `TruApp` instrumentation and dashboard; sampling (every answer is scored); scoring with `gemma3:12b` in the background; alerting outside the app.
