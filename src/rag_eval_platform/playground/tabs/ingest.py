"""① Ingest: upload, index, and see how documents become chunks."""

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.ingestion.loaders import DocumentLoadError
from rag_eval_platform.playground import charts
from rag_eval_platform.playground.core import UPLOAD_DIR, UploadError, build_index, save_uploads
from rag_eval_platform.playground.shared import (
    HISTORY,
    INDEX_VERSION,
    INDEXED_WITH,
    JUDGED,
    SKIPPED,
    SUMMARY,
    TRACE,
    TabContext,
    animated_chart,
    load_embedder,
    settings,
    show_pipeline,
)
from rag_eval_platform.playground.visuals import (
    chunk_spans,
    index_counts,
    ingest_flow_dot,
    size_bins,
)
from rag_eval_platform.retrieval.vector_store import VectorStoreError


def render(ctx: TabContext) -> None:
    show_pipeline("Ingest", ctx)
    left, right = st.columns([1, 2], gap="large")
    with left:
        _upload(ctx)
    with right:
        _index_views(ctx)


def _upload(ctx: TabContext) -> None:
    st.markdown(
        "**Upload** PDFs (with selectable text), Markdown or text. "
        "Building the index **replaces** the previous upload."
    )
    files = st.file_uploader("Documents", type=["pdf", "md", "txt"], accept_multiple_files=True)
    clicked = st.button("Build index", type="primary", disabled=not files or ctx.store is None)
    options = ctx.options
    if clicked and files and ctx.store is not None:
        try:
            with st.status("Building the index...", expanded=True) as status:
                uploaded = save_uploads([(f.name, f.getvalue()) for f in files], UPLOAD_DIR)
                st.session_state[SKIPPED] = uploaded.skipped
                built = build_index(
                    UPLOAD_DIR,
                    strategy=options.strategy,
                    chunk_size=options.chunk_size,
                    chunk_overlap=options.chunk_overlap,
                    embedder=load_embedder(settings.embedding_model),
                    store=ctx.store,
                    on_progress=st.write,
                )
                # Saved before the next st.* call, which could stop the script on a click.
                st.session_state[SUMMARY] = built
                st.session_state[INDEXED_WITH] = (
                    options.strategy,
                    options.chunk_size,
                    options.chunk_overlap,
                )
                st.session_state[INDEX_VERSION] = st.session_state.get(INDEX_VERSION, 0) + 1
                st.session_state.pop(TRACE, None)  # old answers cite the old index
                st.session_state.pop(JUDGED, None)
                st.session_state.pop(HISTORY, None)  # the chat's answers cite the old index
                status.update(label="Index ready", state="complete", expanded=False)
        except (UploadError, DocumentLoadError, VectorStoreError, OptionalDependencyError,
                ValueError) as exc:  # fmt: skip
            st.error(str(exc))
            return
        st.rerun()  # redraw every view from the new index
    for note in st.session_state.get(SKIPPED, ()):
        st.warning(f"Skipped duplicate: {note}")
    current = (options.strategy, options.chunk_size, options.chunk_overlap)
    indexed_with = st.session_state.get(INDEXED_WITH)
    if indexed_with and indexed_with != current:
        st.info("Chunk settings changed since the last build. Click **Build index** to apply them.")


def _index_views(ctx: TabContext) -> None:
    snapshot = ctx.snapshot
    if snapshot is None:
        st.info("Nothing indexed yet: upload documents and click **Build index**.")
        return
    counts = index_counts(snapshot.chunks)
    st.graphviz_chart(ingest_flow_dot(counts), width="stretch")
    st.dataframe(
        [
            {
                "document": doc,
                "pages": index_counts([c for c in snapshot.chunks if c.doc_id == doc]).pages
                or None,
                "chunks": sum(1 for c in snapshot.chunks if c.doc_id == doc),
            }
            for doc in snapshot.doc_ids
        ],
        hide_index=True,
        key="ingest-docs",
    )
    version = st.session_state.get(INDEX_VERSION, 0)
    bins = size_bins([len(c.text) for c in snapshot.chunks])
    st.markdown(
        "**Chunk sizes.** Most chunks sit just under the chunk-size setting; small ones are "
        "the ends of documents or sections."
    )
    animated_chart(
        "ingest-sizes",
        f"{version}-{len(snapshot.chunks)}",
        lambda p: charts.chunk_size_chart(bins, p, height=260),
    )

    doc = st.selectbox("Document", snapshot.doc_ids, key="ingest-doc")
    chunks = [c for c in snapshot.chunks if c.doc_id == doc]
    st.markdown(
        "**Where each chunk sits.** Red = characters shared with the previous chunk (the "
        "overlap setting), so a sentence cut at a boundary survives whole in one of them."
    )
    st.plotly_chart(
        charts.chunk_layout_chart(chunk_spans(chunks), height=min(600, 80 + 22 * len(chunks))),
        key="ingest-layout",
        config=charts.CHART_CONFIG,
    )
    number = st.number_input("Read chunk number", 0, len(chunks) - 1, 0, key="ingest-read")
    chunk = chunks[int(number)]
    st.caption(f"{chunk.id} · page {chunk.page or '-'} · {len(chunk.text)} characters")
    st.text(chunk.text)
