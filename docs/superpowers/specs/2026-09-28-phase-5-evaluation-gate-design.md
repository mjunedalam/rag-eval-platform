# Phase 5 — Evaluation gate in CI: design

**Date:** 2026-09-28 · **Branch:** `feat/phase-5-evaluation-gate` · **Status:** approved (2026-09-28)

## 1. Goal

Every pull request is checked against the golden dataset, and a change that makes retrieval or answers worse is blocked before it merges. The gate must be free, need no API keys and keep the local-first setup: GitHub's runners have no GPU and 7 GB of RAM, so they cannot run `qwen3:8b` and `gemma3:12b` (the full generation evaluation takes about an hour on an M3 GPU).

As with every phase, the work ships with updated docs and a playground view that shows the gate working.

**Success criteria**

- A pull request that lowers Recall@k, MRR or NDCG@k below its threshold, or by more than the allowed drop from the committed baseline, fails the `Evaluation gate` check.
- A pull request that changes anything the generation scores depend on (prompt version, models, retrieval or chunk settings, golden set, corpus) fails until a fresh generation baseline is committed.
- The check writes a before/after table to the job summary and to one pull-request comment.
- The playground's **⑥ Gate** tab shows the gate's flow and checks, and can run a what-if gate with the sidebar settings in seconds.

## 2. Decisions

| Question | Decision |
|---|---|
| Generation quality in CI | **Committed report.** You run the judge locally, as today, and commit the result as a baseline. CI checks its scores against the thresholds and checks that it is **fresh**: made with the current settings, golden set and corpus. |
| Retrieval quality in CI | Evaluated **live** on every pull request (deterministic; CPU embeddings). |
| Regressions above the threshold | Also blocked: no retrieval metric may drop more than `RAG_GATE_MAX_DROP` (default 0.02) below the committed retrieval baseline. |
| Playground | A **live what-if gate** in a new ⑥ Gate tab. |
| New dependencies | None. CI installs the existing `local-embeddings` extra. |

## 3. Components

### 3.1 `evaluation/fingerprint.py`

`EvalFingerprint` is a frozen dataclass holding everything the scores depend on:

| Field | Source |
|---|---|
| `generator_model`, `llm_temperature`, `llm_reasoning_effort`, `prompt_version` | settings, `prompt_templates.PROMPT_VERSION` |
| `judge_model`, `citation_prompt_version` | settings, `citation_validity.CITATION_PROMPT_VERSION` |
| `embedding_model`, `chunk_strategy`, `chunk_size`, `chunk_overlap`, `top_k`, `rerank`, `rerank_candidates`, `reranker_model` | settings |
| `golden_sha256` | SHA-256 of the golden dataset file |
| `corpus_sha256` | SHA-256 over `data/raw/` (relative path and bytes of each file, in sorted order) |

- `current_fingerprint(settings) -> EvalFingerprint`
- `fingerprint_changes(saved, current) -> tuple[FieldChange, ...]`, where `FieldChange(field, saved, current)` reads as "chunk_size: 800 → 400".
- `to_dict` / `from_dict` for JSON. An unknown or missing field in a saved fingerprint counts as a change, so old baselines are stale rather than silently accepted.

### 3.2 Committed baselines in `baselines/` (tracked by git)

- `baselines/retrieval.json`: the reviewed retrieval report (`report_to_dict`) plus its fingerprint.
- `baselines/generation.json`: the generation report plus its fingerprint.
- Both are read and written through `evaluation/baselines.py`: `Baseline(report: dict[str, Any], fingerprint: EvalFingerprint)`, with `load_baseline(path) -> Baseline | None` (None when the file is missing) and `save_baseline(path, report, fingerprint)`.
- Written by `run_evaluation.py --save-baseline` and `run_generation_evaluation.py --save-baseline`. A generation baseline must cover the whole golden set: `--save-baseline` together with `--limit` is rejected.
- A one-time `scripts/save_baseline.py generation reports/generation_report.json` adopts an existing full report. It refuses when the report's recorded generator model, prompt version, judge model or citation prompt version differ from the current settings, or when it did not score every golden question. It then stamps the current fingerprint (documented: this trusts that the report was made with the current retrieval settings). This lets the 2026-09-27 baseline be adopted without another hour of judging.
- Updating a baseline is a deliberate change that shows in the pull request's diff, like changing a threshold.

### 3.3 `evaluation/gate.py` (pure)

```python
CheckStatus = Literal["pass", "fail"]


@dataclass(frozen=True)
class Check:
    group: Literal["retrieval", "generation"]
    name: str  # "recall@k", "mrr", "ndcg@k", "faithfulness", "answer_relevance", "freshness"
    value: float | None  # None for the freshness check or a metric that was never scored
    threshold: float | None
    baseline: float | None
    status: CheckStatus
    reason: str  # e.g. "0.79 < 0.80", "dropped 0.04 (max 0.02)", "chunk_size: 800 → 400"


@dataclass(frozen=True)
class GateResult:
    checks: tuple[Check, ...]
    changes: tuple[FieldChange, ...]  # why the generation baseline is stale; empty when fresh
    passed: bool  # every check passed
```

`run_gate(retrieval: RetrievalReport, retrieval_baseline: Baseline | None, generation_baseline: Baseline | None, current: EvalFingerprint, settings: Settings) -> GateResult`

Rules:

1. Each retrieval metric (Recall@k, MRR, NDCG@k) fails when below its threshold (`min_*` settings).
2. It also fails when it dropped more than `gate_max_drop` below the retrieval baseline. With no retrieval baseline, only rule 1 applies, and the result says so.
3. `freshness` fails when there is no generation baseline, or when its fingerprint differs from `current`. The reason lists every change and says: "run `scripts/run_generation_evaluation.py --save-baseline` and commit `baselines/generation.json`".
4. Faithfulness and answer relevance, read from the generation baseline, fail when below their thresholds or never scored (the rule the generation evaluator already uses).

`gate_markdown(result, retrieval, retrieval_baseline) -> str` renders the pass/block headline, a checks table (metric · baseline · now · Δ · threshold · status), the stale-baseline changes, and the per-query-type before/after table for retrieval.

### 3.4 `evaluation/gate_cli.py` + `scripts/run_gate.py`

Runs ingest → seed (into the configured collection) → retrieval evaluation → gate, and writes `reports/gate_report.json` and `reports/gate_summary.md`. Exit codes follow the other scripts: 0 pass, 1 blocked, 2 error (for example, Chroma unreachable or the extra missing). `--summary PATH` also appends the Markdown to a given file (CI passes `$GITHUB_STEP_SUMMARY`).

### 3.5 `.github/workflows/evaluation_gate.yml`

- Triggers on `pull_request` and on `push` to `main`, with the same Chroma service container as `ci.yml`.
- Installs with `uv sync --locked --extra local-embeddings` (CPU-only torch on Linux, already configured) and caches `~/.cache/huggingface`.
- Runs `uv run python scripts/run_gate.py --summary "$GITHUB_STEP_SUMMARY"`.
- Uploads `reports/` as a build artifact.
- On pull requests from this repository, creates or updates one comment (found by a hidden marker) with `gate_summary.md`, using `gh` and `pull-requests: write`. Fork pull requests skip the comment; the job summary still has the table.
- Blocking merges requires marking `Evaluate against golden dataset` as a required status check in branch protection. The docs explain this; changing the repository setting is the owner's call.

### 3.6 Settings

`gate_max_drop: Score = 0.02` (`RAG_GATE_MAX_DROP`), plus a commented entry in `.env.example`.

## 4. Playground: the ⑥ Gate tab

Pure logic lives in `playground/gate_view.py` (unit-tested); `tabs/gate.py` only draws. The Overview is unchanged.

- **CI flow diagram** (Graphviz DOT, the same style as the pipeline): Pull request → Build index → Evaluate retrieval → Check generation baseline → Compare → ✅ Pass / ⛔ Block. Each step is coloured by the latest result.
- **Checks board:** one gauge per check, showing the threshold and the baseline marker, green or red, with the reason. A stale generation baseline shows its list of changes.
- **Run the gate (what-if):** uses the sidebar's chunk strategy, size, overlap, top-k and re-rank settings. It rebuilds `data/raw/` into a separate **`gate_preview`** collection (`rag_documents` and `playground` are never touched), evaluates the 30 golden questions, and runs the gate against the committed baselines with a fingerprint built from those settings. It shows:
  - pass or block, with the checks board;
  - a before/after table and a per-query-type bar chart (baseline vs now);
  - the **flipped questions**: hit → miss or miss → hit, with the question and its expected documents.

  The result is saved to `st.session_state` before the next `st.*` call. A spinner shows progress (index, then evaluate).
- **Latest CI-style result:** when `reports/gate_report.json` exists, the tab shows it on load.

## 5. Docs

- `docs/learning/phase-5.md`: why a gate exists, the CI sequence diagram, a decision-rules flowchart, the fingerprint explained, how to update a baseline, how to make the check required, and what to try in the ⑥ Gate tab.
- `docs/evaluation_methodology.md` §4: rewritten to match what is built (live retrieval, committed generation baseline, freshness, max drop).
- `readme.md`, `CLAUDE.md`, `docs/architecture.md` (Phase 5 rows become *in use*; NumPy row fixed to *in use*), and `docs/learning/playground.md` (the new tab).

## 6. Error handling

- A missing or unreadable baseline file is reported as a failing check with a fix hint, never a crash. A malformed JSON file exits 2 with the file name.
- Chroma down, a missing `local-embeddings` extra, or a missing golden file exits 2 with a fix hint.
- The what-if run shows the same failures as `st.error`, without changing any saved state.

## 7. Testing

- Unit tests: fingerprint (hashing is stable and order-independent; changes listed; missing fields count as changed), every gate rule and its boundaries (equal to threshold passes; drop exactly `max_drop` passes), stale and missing baselines, Markdown output, baseline save/adopt refusals, the CLI's exit codes with fakes, and `gate_view` (flow-diagram states, flipped questions, board rows).
- Integration: `run_gate` end to end against Chroma with a fake embedder (runs in CI's integration job), and an `AppTest` smoke test that the ⑥ Gate tab renders.
- The workflow itself is verified on the pull request by its own run.

## 8. Out of scope

- Running generation or the judge in CI; hosted judges.
- Changing the thresholds or the golden set.
- Changing the repository's branch-protection settings.
