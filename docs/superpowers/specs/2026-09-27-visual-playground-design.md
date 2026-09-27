# Visual playground — design

**Date:** 2026-09-27 · **Branch:** `feat/visual-playground` · **Status:** approved in conversation, awaiting spec review

## 1. Goal

Turn the Streamlit playground into a **visual learning tool**: while using it on your own
documents, you watch every RAG phase happen, with a live diagram or chart for each concept
from Phases 1–4. Charts animate smoothly when new data arrives and grow during long steps.

**Success criteria**

1. Every concept below has a live view that updates when a setting changes or a new question
   is asked.
2. An **Overview** tab shows all five phases for the latest question on **one screen with no
   scrolling** (checked at 1440×900).
3. Animations play once when new data arrives and never replay on a tab switch.
4. Uploads stay private: `data/playground/` (git-ignored) and the separate `playground`
   Chroma collection; the evaluation corpus is untouched.

**Concepts to cover**

| Phase | Concepts |
|---|---|
| 1 · Ingest | loading, pages, chunking strategy, chunk size, overlap |
| 2 · Embed | embeddings as vectors, meaning space, cosine similarity |
| 3 · Retrieve | vector search, similarity scores, top-k, candidates, re-ranking |
| 4 · Generate | prompt (rules, numbered sources, question), streaming, tokens, citations, invalid citations, refusal, timings |
| 5 · Evaluate | LLM judge, faithfulness, answer relevance, citation validity, golden-set reports |

## 2. Decisions made

| Question | Decision |
|---|---|
| Evaluation in the UI | **Both**: a "Judge this answer" button (1–2 min, live) and an Evaluate tab charting the saved golden-set reports |
| Layout | **One tab per phase plus an Overview tab** that shows everything at once, no scrolling |
| Charts | **Plotly** (new dependency in the `ui` group, approved in conversation), plus Graphviz DOT for flow diagrams via `st.graphviz_chart` and numpy for PCA |
| Animation | Grow-in animation when new data arrives; live growth during indexing, streaming and judging |

## 3. Structure

Tabs: **Overview · ① Ingest · ② Embed · ③ Retrieve · ④ Generate · ⑤ Evaluate**

- **One question drives everything.** A single question box above the tabs. The resulting
  `QueryTrace` is stored in `st.session_state` (before any further `st.*` call, per the
  existing Streamlit rule) and every tab draws from it.
- **Pipeline diagram on every tab**: Ingest → Embed → Retrieve → Generate → Evaluate, with the
  tab's phase highlighted and live numbers on the edges (e.g. "1,505 chunks", "top-5 of 20",
  "cited 2/5", "faithfulness 0.93"). Phases without data are greyed out.
- **The sidebar settings stay as they are** (chunking, retrieval, generation).

### Overview (one screen, no scroll)

A fixed 2×3 grid of compact panels with fixed heights (~300 px):

| Ingest | Embed | Retrieve |
|---|---|---|
| chunk-size histogram + counts | 2-D meaning map, question as a star | similarity bars, top-k cut-off, re-rank arrows |
| **Generate** | **Evaluate** | **Timing** |
| answer with citations colour-linked to chunks | judge scores if judged, else baseline scores | retrieve vs generate (vs judge) bar |

The pipeline diagram sits above the grid in a single short row.

## 4. What each tab shows (★ = animated)

**① Ingest**
- Flow diagram: file → pages → text → chunks, with counts.
- ★ Chunk-size histogram, growing per document during indexing (`build_index`'s `on_progress`).
- Chunk layout: each document as a horizontal bar, chunks as blocks, **overlaps highlighted**.
- Table: pages and chunks per file.

**② Embed**
- The question's 384 numbers as a colour strip, next to the best-matching chunk's strip.
- ★ Meaning map: every chunk as a dot (PCA of the stored vectors), the question as a star,
  retrieved chunks highlighted, hover shows the chunk text. Dots spread out from the centre on draw.
  A **2-D / 3-D toggle**: 2-D by default (also used on the Overview), 3-D rotatable with Plotly.
- Cosine similarity explainer: the angle between the question and the top chunk.

**③ Retrieve**
- ★ Similarity bars for all candidates, with the top-k cut-off line.
- ★ Re-ranking slope chart: rank before → after for each candidate, lines crossing when reordered.
- Funnel: all chunks → candidates → top-k.

**④ Generate**
- The prompt as stacked blocks: system rules, numbered sources, question.
- ★ Streaming answer with a live token counter.
- Citations `[n]` colour-linked to their chunk; invalid citations red; refusal badge.
- ★ Timing bar: retrieve vs generate.

**⑤ Evaluate**
- "Judge this answer" button:
  - ★ Faithfulness and relevance gauges filling as each score arrives.
  - Each citation's yes/no verdict as a sentence ↔ chunk link.
- Golden-set reports (if `reports/*.json` exist): scores vs pass marks, per query type,
  retrieval hit vs faithfulness. Otherwise the command that creates them.

## 5. Animation

- **Grow-in**: when a view receives new data, it is redrawn in a placeholder over ~0.5 s from
  zero to its final values (bars rise, dots spread out). A spike checks whether Plotly's native
  `layout.transition` animates inside `st.plotly_chart`; where it does, it replaces the
  frame loop. **Result (2026-09-27):** it does not — Streamlit replaces the whole figure on
  update (bars jump straight to their final values), so the frame loop stays.
- **Live growth**: indexing (histogram per document), streaming (answer and token counter),
  judging (scores one metric at a time).
- **Play once**: each animated view records a fingerprint of the data it last animated in
  `st.session_state`; a rerun with the same data draws the final frame directly.

## 6. Data and code

### Data sources
- `QueryTrace` gains `query_embedding: tuple[float, ...]` and `timings` (retrieve, generate,
  and judge when present). The pre-rerank ranking already exists (`vector_results`).
- `ChromaVectorStore.export() -> tuple[tuple[Chunk, ...], tuple[tuple[float, ...], ...]]` reads
  all chunks and vectors back from Chroma, so Ingest/Embed survive an app restart. The shared
  `VectorStore` protocol is not changed; the playground depends on a small local protocol.
- Meaning map: numpy PCA to 2 or 3 dimensions, cached per index (collection count + chunk settings);
  indexes over 3,000 chunks are sampled, always keeping the retrieved chunks.
- Judge: `RagasJudge` + `CitationChecker` from Phase 4, cached per session with
  `st.cache_resource`; result stored in session state as soon as it arrives.
- Reports: `reports/retrieval_report.json` and `reports/generation_report.json` if present.

### Units

| File | Responsibility | Tested by |
|---|---|---|
| `retrieval/vector_store.py` | `ChromaVectorStore.export()` | unit (fake client) + integration (real Chroma) |
| `playground/core.py` | trace fields and timings, `judge_answer`, `load_reports` | unit, with fakes |
| `playground/visuals.py` (new) | pure data: pipeline DOT with active phase, chunk/overlap layout, PCA points, similarity and re-rank rows, prompt blocks, citation links, animation frames, data fingerprints | unit |
| `playground/charts.py` (new) | Plotly figures from visuals data | unit, skipped without Plotly (CI) |
| `playground/tabs/{overview,ingest,embed,retrieve,generate,evaluate}.py` (new) | draw one tab each | Streamlit `AppTest` smoke |
| `playground/app.py` (shrinks) | page shell, sidebar, question box, tab routing | Streamlit `AppTest` smoke |

`tabs/` and `app.py` are left out of coverage, like `app.py` today. Plotly goes in the `ui`
dependency group; mypy skips `plotly` like `streamlit`.

## 7. Empty and error states

| Situation | Behaviour |
|---|---|
| Nothing indexed | Diagram with phases greyed; "Upload documents in ① Ingest" |
| No question yet | Index-level views only; prompt to ask a question for the rest |
| Chroma / Ollama down | Existing status boxes; affected tabs show the command to run |
| `evaluation` extra missing / no reports | The exact `uv sync` or script command |
| Judge fails | Error message; the answer and other tabs keep working |

## 8. Testing and verification

- **Unit (TDD, CI):** all of `visuals.py`; `export()`; new `core.py` pieces. Coverage ≥ 80%.
- **Charts:** each Plotly builder returns a figure with the expected traces and values
  (`pytest.importorskip("plotly")`).
- **Smoke (integration):** every tab renders with an empty index and with a fake trace.
- **Real check before reporting done:** launch the app, drive it in a headless browser
  (upload a PDF, ask, judge), screenshot every tab at 1440×900, confirm the Overview needs no
  scrolling and that animations play once on new data but not on tab switches.

## 9. Delivery sequence

Each step leaves a working app:

1. Data layer: `export()`, trace fields and timings, `visuals.py`.
2. Pipeline diagram, ① Ingest, ② Embed.
3. ③ Retrieve, ④ Generate.
4. ⑤ Evaluate: live judge and reports.
5. Overview.
6. Animation polish.
7. Docs: readme, CLAUDE.md, and a learning-notes section on the visual playground.

## 10. Out of scope

- Editing chunks or the prompt from the UI.
- Running the full golden-set evaluation from the UI (it takes about an hour; the Evaluate tab
  shows its saved reports instead).
