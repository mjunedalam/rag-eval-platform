"""RAG Playground: watch every RAG phase work on your own documents.

Run:   uv run streamlit run src/rag_eval_platform/playground/app.py
Needs: Chroma (`docker compose -f docker/docker-compose.yml up -d`), Ollama with a model
       (`brew services start ollama`, `ollama pull qwen3:8b`), `uv sync --all-extras --all-groups`.

Uploads live in data/playground/ (git-ignored) and their own Chroma collection, so the
sample corpus and the golden-dataset evaluation are never affected. Logic lives in core.py,
visuals.py, query_visuals.py and live.py; this file and tabs/ only draw.

The chat sits on the left and the dashboard on the right. Asking a question switches the
dashboard to the Overview, which runs it live: every panel and stat updates as its phase
finishes while the reply streams into the chat with a status line.
"""

from collections.abc import Callable

import streamlit as st

from rag_eval_platform.playground.header import app_css, header_html
from rag_eval_platform.playground.live import LIVE_CSS
from rag_eval_platform.playground.session_metrics import (
    session_kpis,
    session_points,
    session_summary,
)
from rag_eval_platform.playground.shared import (
    HISTORY,
    TAB_KEY,
    TAB_LABELS,
    TRACE,
    TabContext,
    current_judged,
    current_snapshot,
    ollama_models,
    service_status,
    settings,
    sidebar,
)
from rag_eval_platform.playground.tabs import (
    chat_panel,
    embed,
    evaluate,
    gate,
    generate,
    ingest,
    overview,
    retrieve,
)

RENDERERS: dict[str, Callable[[TabContext], None]] = {
    "Overview": overview.render,
    "① Ingest": ingest.render,
    "② Embed": embed.render,
    "③ Retrieve": retrieve.render,
    "④ Generate": generate.render,
    "⑤ Evaluate": evaluate.render,
    "⑥ Gate": gate.render,
}


st.set_page_config(
    page_title="RAG Playground", page_icon="📚", layout="wide", initial_sidebar_state="collapsed"
)
# Theme in .streamlit/config.toml; APP_CSS adds cards, pills and a compact header, so the
# Overview (all phases at once) fits one screen without scrolling.
# st.html applies style-only HTML without Markdown parsing and without taking up space.
# Colours follow the light or dark theme (⋮ → Settings) through CSS light-dark().
st.html(app_css() + LIVE_CSS)
title_column, status_column = st.columns([1, 3], vertical_alignment="center")
title_column.title("RAG Playground")

available_models = ollama_models(settings.ollama_base_url)
options = sidebar(available_models)
store, pills = service_status(available_models)
snapshot = current_snapshot(store)
kpis = session_kpis(session_summary(session_points(st.session_state.get(HISTORY, ()))))
status_column.markdown(header_html(pills, kpis), unsafe_allow_html=True)

chat_column, dashboard_column = st.columns([32, 68], gap="medium")
with chat_column:
    bubble = chat_panel.render(snapshot, available_models)
with dashboard_column:
    trace = st.session_state.get(TRACE)
    ctx = TabContext(
        options, store, available_models, snapshot, trace, current_judged(trace), bubble
    )
    tabs = st.tabs(TAB_LABELS, key=TAB_KEY, on_change="rerun")
    for label, tab in zip(TAB_LABELS, tabs, strict=True):
        if tab.open:
            with tab:
                RENDERERS[label](ctx)
