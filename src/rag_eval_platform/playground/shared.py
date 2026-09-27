"""Streamlit helpers shared by the playground tabs: settings, cached resources, session
state, the pipeline header and play-once animated charts. Drawing only; the logic lives in
core.py, visuals.py and query_visuals.py."""

import json
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, get_args

import plotly.graph_objects as go
import streamlit as st

from rag_eval_platform.config.settings import ChunkStrategy, ReasoningEffort, get_settings
from rag_eval_platform.evaluation.citation_validity import CitationChecker, create_citation_checker
from rag_eval_platform.evaluation.judge import RagasJudge
from rag_eval_platform.generation.prompt_templates import AnswerStyle
from rag_eval_platform.ingestion.embedding import SentenceTransformerEmbedder
from rag_eval_platform.playground.animation import ease_steps, should_animate
from rag_eval_platform.playground.charts import CHART_CONFIG
from rag_eval_platform.playground.core import (
    PLAYGROUND_COLLECTION,
    IndexSnapshot,
    JudgeResult,
    QueryTrace,
    index_snapshot,
    trace_key,
)
from rag_eval_platform.playground.visuals import PipelineStats, pipeline_dot
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker
from rag_eval_platform.retrieval.vector_store import ChromaVectorStore, VectorStoreError

settings = get_settings()

TRACE, SUMMARY, INDEXED_WITH, INDEX_VERSION = "trace", "summary", "indexed_with", "index_version"
SKIPPED, JUDGED = "skipped", "judged"
PENDING_QUESTION, TAB_KEY = "pending_question", "phase-tab"
FIRST_OUTPUT_WAIT = "first_output_wait"
HISTORY, PENDING_REPLACE = "history", "pending_replace"
FRAMES, FRAME_SECONDS = 8, 0.05
TAB_LABELS = [
    "Overview", "① Ingest", "② Embed", "③ Retrieve", "④ Generate", "⑤ Evaluate",
]  # fmt: skip


@dataclass(frozen=True)
class Options:
    strategy: ChunkStrategy
    chunk_size: int
    chunk_overlap: int
    top_k: int
    rerank: bool
    rerank_candidates: int
    model: str
    reasoning_effort: ReasoningEffort
    temperature: float
    slow_motion: bool = False
    answer_style: AnswerStyle = "detailed"


@dataclass(frozen=True)
class TabContext:
    options: Options
    store: ChromaVectorStore | None
    models: list[str] | None
    snapshot: IndexSnapshot | None
    trace: QueryTrace | None
    judged: JudgeResult | None
    bubble: Any = None  # the chat reply being written, if a question is running now

    @property
    def stats(self) -> PipelineStats:
        snapshot, trace, judged = self.snapshot, self.trace, self.judged
        return PipelineStats(
            documents=len(snapshot.doc_ids) if snapshot else 0,
            chunks=len(snapshot.chunks) if snapshot else 0,
            candidates=len(trace.vector_results) if trace else 0,
            top_k=len(trace.final_results) if trace else 0,
            cited=len({c.number for c in trace.answer.citations}) if trace else 0,
            faithfulness=judged.scores.faithfulness if judged else None,
        )


# ---------- cached resources (loaded once per server, not on every click) ----------


@st.cache_resource(show_spinner="Loading the embedding model...")
def load_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder.from_pretrained(model_name)


@st.cache_resource(show_spinner="Loading the re-ranker...")
def load_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker.from_pretrained(model_name)


@st.cache_resource(show_spinner=False)
def connect_store(host: str, port: int) -> ChromaVectorStore:
    return ChromaVectorStore.connect(host, port, PLAYGROUND_COLLECTION)


@st.cache_resource(show_spinner="Loading the judge...")
def load_judge(judge_model: str) -> tuple[RagasJudge, CitationChecker]:
    embedder = load_embedder(settings.embedding_model)
    return RagasJudge.from_settings(settings, embedder), create_citation_checker(settings)


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models(base_url: str) -> list[str] | None:
    """Models available in Ollama, or None when Ollama is not reachable."""
    url = base_url.removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - configured URL
            return sorted(m["name"] for m in json.load(response)["models"])
    except (OSError, ValueError, KeyError):
        return None


@st.cache_data(show_spinner="Reading the index...", max_entries=3)
def _snapshot(_store: ChromaVectorStore, version: int, count: int) -> IndexSnapshot:
    return index_snapshot(_store)  # cached per index version and size; _store is not hashed


def current_snapshot(store: ChromaVectorStore | None) -> IndexSnapshot | None:
    if store is None:
        return None
    try:
        count = store.count()
        return _snapshot(store, st.session_state.get(INDEX_VERSION, 0), count) if count else None
    except VectorStoreError as exc:
        st.error(str(exc))
        return None


def current_judged(trace: QueryTrace | None) -> JudgeResult | None:
    """The judge result for this answer, if it has been judged."""
    if trace is None:
        return None
    judged: dict[str, JudgeResult] = st.session_state.get(JUDGED, {})
    return judged.get(trace_key(trace))


def save_judged(trace: QueryTrace, result: JudgeResult) -> None:
    """Remember a judge result for this answer (any chat turn can be judged)."""
    st.session_state[JUDGED] = {**st.session_state.get(JUDGED, {}), trace_key(trace): result}


# ---------- page parts ----------


def sidebar(models: list[str] | None) -> Options:
    st.sidebar.header("Settings")
    st.sidebar.caption("Change these and watch every tab react.")
    st.sidebar.subheader("① Chunking")
    strategy = st.sidebar.segmented_control(
        "Strategy", ["recursive", "fixed"], default=settings.chunk_strategy, required=True
    )
    chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2000, settings.chunk_size, 50)
    chunk_overlap = st.sidebar.slider(
        "Overlap (characters)", 0, min(400, chunk_size - 50), min(settings.chunk_overlap, 400), 10
    )
    st.sidebar.subheader("③ Retrieval")
    top_k = st.sidebar.slider("Top-k chunks sent to the model", 1, 10, settings.top_k)
    rerank = st.sidebar.toggle("Re-rank with a cross-encoder", value=settings.rerank)
    rerank_candidates = st.sidebar.slider(
        "Candidates before re-ranking", top_k, 30, max(settings.rerank_candidates, top_k),
        disabled=not rerank,
    )  # fmt: skip
    st.sidebar.subheader("④ Generation")
    choices = models or [settings.llm_model]
    model = st.sidebar.selectbox(
        "Model", choices,
        index=choices.index(settings.llm_model) if settings.llm_model in choices else 0,
    )  # fmt: skip
    effort = st.sidebar.segmented_control(
        "Thinking (reasoning effort)",
        [e for e in get_args(ReasoningEffort) if e != "default"],
        default=settings.llm_reasoning_effort, required=True,
    )  # fmt: skip
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, settings.llm_temperature, 0.1)
    answer_style = st.sidebar.segmented_control(
        "Answer style", list(get_args(AnswerStyle))[::-1], default="detailed", required=True,
        help="Detailed: a long, formatted answer (headings, lists, tables). Concise: a few "
        "sentences, the style the golden-set evaluation uses.",
    )  # fmt: skip
    st.sidebar.subheader("🎬 Playback")
    slow_motion = st.sidebar.toggle(
        "🐢 Slow motion", help="Slow the live run down to watch every step happen."
    )
    return Options(strategy, chunk_size, chunk_overlap, top_k, rerank, rerank_candidates,
                   model, effort, temperature, slow_motion, answer_style)  # fmt: skip


def service_status(models: list[str] | None) -> ChromaVectorStore | None:
    """One short line when everything is up; a red box for each service that is down."""
    store: ChromaVectorStore | None
    try:
        store = connect_store(settings.chroma_host, settings.chroma_port)
        chroma = f"🟢 Chroma · {store.count()} chunks in '{PLAYGROUND_COLLECTION}'"
    except VectorStoreError:
        store, chroma = None, ""
        st.error("Chroma is not running: `docker compose -f docker/docker-compose.yml up -d`")
    if models is None:
        st.error("Ollama is not running: `brew services start ollama`")
    ollama = f"🟢 Ollama · {len(models)} model(s)" if models is not None else ""
    st.caption("  ·  ".join(part for part in (chroma, ollama) if part))
    return store


def show_pipeline(active: str | None, ctx: TabContext) -> None:
    st.graphviz_chart(pipeline_dot(active, ctx.stats), width="stretch")


def animated_chart(
    key: str,
    data_key: str,
    build: Callable[[float], go.Figure],
    *,
    height: int | None = None,
    placeholder: Any = None,
    frames: int = FRAMES,
    frame_seconds: float = FRAME_SECONDS,
) -> None:
    """Draw ``build(progress)``; grow it in (0 -> 1) the first time ``data_key`` is shown."""
    placeholder = placeholder if placeholder is not None else st.empty()
    size = height or "content"
    if should_animate(st.session_state, key, data_key):
        for i, progress in enumerate(ease_steps(frames)):
            placeholder.plotly_chart(
                build(progress), key=f"{key}-{i}", config=CHART_CONFIG, height=size
            )
            time.sleep(frame_seconds)
    else:
        placeholder.plotly_chart(build(1.0), key=f"{key}-final", config=CHART_CONFIG, height=size)


def mark_shown(key: str, data_key: str) -> None:
    """Record that ``key`` already showed ``data_key``, so a later redraw does not replay it."""
    should_animate(st.session_state, key, data_key)
