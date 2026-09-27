"""RAG Playground: upload your own documents and try the pipeline step by step.

Run:   uv run streamlit run src/rag_eval_platform/playground/app.py
Needs: Chroma (`docker compose -f docker/docker-compose.yml up -d`), Ollama with a model
       (`ollama pull qwen3:8b`), and `uv sync --all-extras --all-groups`.

Uploads live in data/playground/ (git-ignored) and in their own Chroma collection, so the
sample corpus and the golden-dataset evaluation are never affected. All logic is in
core.py; this file only draws the page.
"""

import json
import urllib.request
from dataclasses import dataclass
from typing import get_args

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.config.settings import ChunkStrategy, ReasoningEffort, get_settings
from rag_eval_platform.generation.generator import (
    GenerationError,
    Generator,
    OpenAICompatibleClient,
)
from rag_eval_platform.generation.prompt_templates import build_messages
from rag_eval_platform.ingestion.embedding import SentenceTransformerEmbedder
from rag_eval_platform.ingestion.loaders import DocumentLoadError
from rag_eval_platform.playground.core import (
    PLAYGROUND_COLLECTION,
    UPLOAD_DIR,
    IndexSummary,
    QueryTrace,
    UploadError,
    build_index,
    retrieve_step,
    save_uploads,
)
from rag_eval_platform.retrieval.reranker import CrossEncoderReranker
from rag_eval_platform.retrieval.vector_store import (
    ChromaVectorStore,
    SearchResult,
    VectorStoreError,
)

settings = get_settings()


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


@st.cache_data(ttl=30, show_spinner=False)
def ollama_models(base_url: str) -> list[str] | None:
    """Models available in Ollama, or None when Ollama is not reachable."""
    url = base_url.removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310 - configured URL
            return sorted(m["name"] for m in json.load(response)["models"])
    except (OSError, ValueError, KeyError):
        return None


# ---------- page sections ----------


def sidebar(models: list[str] | None) -> Options:
    st.sidebar.header("Settings")
    st.sidebar.caption("Change these and see what happens.")

    st.sidebar.subheader("1 · Chunking")
    strategy = st.sidebar.segmented_control(
        "Strategy", ["recursive", "fixed"], default=settings.chunk_strategy, required=True
    )
    chunk_size = st.sidebar.slider("Chunk size (characters)", 200, 2000, settings.chunk_size, 50)
    chunk_overlap = st.sidebar.slider(
        "Overlap (characters)", 0, min(400, chunk_size - 50), min(settings.chunk_overlap, 400), 10
    )

    st.sidebar.subheader("2 · Retrieval")
    top_k = st.sidebar.slider("Top-k chunks sent to the model", 1, 10, settings.top_k)
    rerank = st.sidebar.toggle("Re-rank with a cross-encoder", value=settings.rerank)
    rerank_candidates = st.sidebar.slider(
        "Candidates before re-ranking", top_k, 30, max(settings.rerank_candidates, top_k),
        disabled=not rerank,
    )  # fmt: skip

    st.sidebar.subheader("3 · Generation")
    choices = models or [settings.llm_model]
    model = st.sidebar.selectbox(
        "Model",
        choices,
        index=choices.index(settings.llm_model) if settings.llm_model in choices else 0,
    )
    effort = st.sidebar.segmented_control(
        "Thinking (reasoning effort)",
        [e for e in get_args(ReasoningEffort) if e != "default"],
        default=settings.llm_reasoning_effort,
        required=True,
    )
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, settings.llm_temperature, 0.1)

    return Options(
        strategy, chunk_size, chunk_overlap, top_k, rerank, rerank_candidates,
        model, effort, temperature,
    )  # fmt: skip


def service_status(models: list[str] | None) -> ChromaVectorStore | None:
    left, right = st.columns(2)
    store: ChromaVectorStore | None
    try:
        store = connect_store(settings.chroma_host, settings.chroma_port)
        left.success(f"Chroma connected · {store.count()} chunks in '{PLAYGROUND_COLLECTION}'")
    except VectorStoreError:
        store = None
        left.error(
            "Chroma is not running. Start it: `docker compose -f docker/docker-compose.yml up -d`"
        )
    if models is None:
        right.error("Ollama is not running. Open the Ollama app (or run `ollama serve`).")
    else:
        right.success(f"Ollama connected · {len(models)} model(s)")
    return store


def upload_tab(options: Options, store: ChromaVectorStore | None) -> None:
    st.markdown(
        "Upload **PDFs** (with selectable text), **Markdown** or **text** files. "
        "Building the index **replaces** the previous upload."
    )
    files = st.file_uploader("Documents", type=["pdf", "md", "txt"], accept_multiple_files=True)
    clicked = st.button("Build index", type="primary", disabled=not files or store is None)
    st.caption(
        "Large PDFs take a while (a 400-page book is about 30 s). Wait for **Index ready** "
        "before clicking anything else: every click restarts the page."
    )
    if clicked and files and store is not None:
        try:
            with st.status("Building the index...", expanded=True) as status:
                uploaded = save_uploads([(f.name, f.getvalue()) for f in files], UPLOAD_DIR)
                st.session_state["skipped"] = uploaded.skipped
                embedder = load_embedder(settings.embedding_model)
                built = build_index(
                    UPLOAD_DIR,
                    strategy=options.strategy,
                    chunk_size=options.chunk_size,
                    chunk_overlap=options.chunk_overlap,
                    embedder=embedder,
                    store=store,
                    on_progress=st.write,
                )
                # Save before the next st.* call. Every st.* call is a point where Streamlit
                # stops the script if the user clicked meanwhile, which would throw away a
                # result that Chroma has already stored.
                st.session_state["summary"] = built
                st.session_state["indexed_with"] = (
                    options.strategy,
                    options.chunk_size,
                    options.chunk_overlap,
                )
                st.session_state.pop("history", None)  # old answers cite the old index
                status.update(label="Index ready", state="complete", expanded=False)
        except (UploadError, DocumentLoadError, VectorStoreError, OptionalDependencyError,
                ValueError) as exc:  # fmt: skip
            st.error(str(exc))
            return

    summary: IndexSummary | None = st.session_state.get("summary")
    if summary is None:
        return
    for note in st.session_state.get("skipped", ()):
        st.warning(f"Skipped duplicate: {note}")
    indexed_with = st.session_state.get("indexed_with")
    if indexed_with != (options.strategy, options.chunk_size, options.chunk_overlap):
        st.info("Chunk settings changed since the last build. Click **Build index** to apply them.")

    cols = st.columns(4)
    cols[0].metric("Documents", len(summary.documents))
    cols[1].metric("PDF pages", summary.page_count or "-")
    cols[2].metric("Chunks", len(summary.chunks))
    cols[3].metric("Avg chunk size", f"{summary.mean_chunk_chars:.0f} chars")
    st.dataframe(
        [
            {
                "document": d.id,
                "format": d.format,
                "pages": len(d.page_starts) or None,
                "characters": len(d.text),
                "chunks": sum(1 for c in summary.chunks if c.doc_id == d.id),
            }
            for d in summary.documents
        ],
        hide_index=True,
    )


def chunks_tab() -> None:
    summary: IndexSummary | None = st.session_state.get("summary")
    if summary is None:
        st.info("Build an index first to see its chunks.")
        return
    st.markdown(
        "These are the pieces your documents were cut into. Each one is embedded and stored "
        "separately; a question retrieves the closest chunks, not whole documents."
    )
    doc_ids = [d.id for d in summary.documents]
    selected = st.selectbox("Document", doc_ids)
    chunks = [c for c in summary.chunks if c.doc_id == selected]
    st.dataframe(
        [
            {"chunk": c.id, "page": c.page, "starts at char": c.start_index,
             "chars": len(c.text), "text": c.text[:120].replace("\n", " ")}
            for c in chunks
        ],
        hide_index=True,
    )  # fmt: skip
    index = st.number_input("Read chunk number", 0, len(chunks) - 1, 0)
    st.text(chunks[int(index)].text)


def ask_tab(options: Options, store: ChromaVectorStore | None, models: list[str] | None) -> None:
    if store is None or store.count() == 0:
        st.info("Build an index first (tab 1), then ask questions about your documents.")
        return

    history: list[QueryTrace] = st.session_state.get("history", [])
    if history and st.button("Clear conversation"):
        st.session_state["history"] = []
        history = []
    st.caption(
        "Each question is answered on its own from your documents (no chat memory). "
        "Wait for an answer to finish before clicking elsewhere: a click restarts the page."
    )

    for number, past in enumerate(history, start=1):
        with st.chat_message("user"):
            st.markdown(past.answer.question)
        with st.chat_message("assistant"):
            show_answer_text(past)
            show_details(past, key=f"turn-{number}")

    question = st.chat_input(
        "Ask about your documents...",
        disabled=models is None,
    )
    if not question:
        return

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            step = retrieve_step(
                question,
                embedder=load_embedder(settings.embedding_model),
                store=store,
                top_k=options.top_k,
                reranker=load_reranker(settings.reranker_model) if options.rerank else None,
                rerank_candidates=options.rerank_candidates,
            )
            llm_settings = settings.model_copy(
                update={
                    "llm_provider": "ollama",
                    "llm_model": options.model,
                    "llm_reasoning_effort": options.reasoning_effort,
                    "llm_temperature": options.temperature,
                }
            )
            generator = Generator(OpenAICompatibleClient.from_settings(llm_settings))
            stream = generator.stream(question, step.final_results)
            st.write_stream(stream)  # the answer appears as the model writes it
        except (GenerationError, VectorStoreError, OptionalDependencyError, ValueError) as exc:
            st.error(str(exc))
            return
        trace = QueryTrace(
            step.vector_results, step.final_results, stream.answer, step.retrieval_ms
        )
        # Saved before any further st.* call, so a click now cannot lose the answer.
        st.session_state["history"] = [*history, trace]
        if trace.answer.is_refusal:
            st.caption("The retrieved chunks did not contain the answer.")
        show_details(trace, key=f"turn-{len(history) + 1}")


def show_answer_text(trace: QueryTrace) -> None:
    answer = trace.answer
    if answer.is_refusal:
        st.warning(answer.text + "  \nThe retrieved chunks did not contain the answer.")
    else:
        st.markdown(answer.text)


def show_details(trace: QueryTrace, key: str) -> None:
    """Invalid citations, run metrics, sources, re-ranking and the prompt for one answer."""
    answer = trace.answer
    if answer.invalid_citations:
        st.error(f"Cited numbers that were never provided: {list(answer.invalid_citations)}")
    tokens = f"{answer.input_tokens} → {answer.output_tokens} tokens"
    st.caption(
        f"{answer.model} · retrieval {trace.retrieval_ms:.0f} ms · "
        f"generation {answer.latency_ms / 1000:.1f} s · {tokens}"
    )

    cited = {c.number for c in answer.citations}
    with st.expander(f"Sources ({len(trace.final_results)} given, {len(cited)} cited)"):
        for number, result in enumerate(trace.final_results, start=1):
            mark = " · ✅ **cited**" if number in cited else ""
            st.markdown(f"**[{number}]** {origin(result)} · score {result.score:.3f}{mark}")
            st.text(result.chunk.text)

    if trace.vector_results != trace.final_results:
        with st.expander("Re-ranking: before and after"):
            before, after = st.columns(2)
            before.caption("Vector search order (cosine similarity)")
            before.dataframe(ranking(trace.vector_results), hide_index=True, key=f"{key}-before")
            after.caption("After the cross-encoder (kept top-k)")
            after.dataframe(ranking(trace.final_results), hide_index=True, key=f"{key}-after")

    with st.expander("What the model saw (the full prompt)"):
        system, user = build_messages(answer.question, trace.final_results)
        st.caption("System message: the rules")
        st.code(system.content, language=None, wrap_lines=True)
        st.caption("User message: numbered sources + your question")
        st.code(user.content, language=None, wrap_lines=True)


def origin(result: SearchResult) -> str:
    chunk = result.chunk
    return chunk.doc_id if chunk.page is None else f"{chunk.doc_id}, page {chunk.page}"


def ranking(results: tuple[SearchResult, ...]) -> list[dict[str, object]]:
    return [
        {"rank": i, "chunk": r.chunk.id, "score": round(r.score, 3)}
        for i, r in enumerate(results, start=1)
    ]


# ---------- page ----------

st.set_page_config(page_title="RAG Playground", page_icon="📚", layout="wide")
st.title("RAG Playground")
st.caption("Upload → chunk → embed → retrieve → generate, with every step visible.")

available_models = ollama_models(settings.ollama_base_url)
chosen = sidebar(available_models)
playground_store = service_status(available_models)

upload, chunks, ask = st.tabs(["① Upload & index", "② Chunks", "③ Ask"])
with upload:
    upload_tab(chosen, playground_store)
with chunks:
    chunks_tab()
with ask:
    ask_tab(chosen, playground_store, available_models)
