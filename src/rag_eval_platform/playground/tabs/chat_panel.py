"""The chat pane (left side): conversation history with Claude-style replies.

Each reply shows its collapsible steps, the answer with coloured citations, source cards, a
meta line and actions (copy, regenerate, judge, show in dashboard, feedback). A new question
gets an empty reply bubble here; the live Overview (right side) writes into it while it runs.
"""

import html
from dataclasses import dataclass
from typing import Any

import streamlit as st

from rag_eval_platform._optional import OptionalDependencyError
from rag_eval_platform.evaluation.judge import JudgeError
from rag_eval_platform.generation.generator import GenerationError
from rag_eval_platform.playground.answer_metrics import answer_health
from rag_eval_platform.playground.chat import (
    Turn,
    cited_sources_html,
    run_steps,
    source_cards,
    steps_summary,
    suggestions,
    user_html,
    with_feedback,
)
from rag_eval_platform.playground.core import IndexSnapshot, judge_answer, trace_key
from rag_eval_platform.playground.live import LiveStats, status_html
from rag_eval_platform.playground.query_visuals import answer_spans, spans_markdown
from rag_eval_platform.playground.shared import (
    FIRST_OUTPUT_WAIT,
    HISTORY,
    PENDING_QUESTION,
    PENDING_REPLACE,
    TAB_KEY,
    TRACE,
    current_judged,
    load_judge,
    save_judged,
    settings,
)
from rag_eval_platform.playground.streaming import steps_html

CHAT_HEIGHT = 555  # the scrolling conversation; the input sits below it

# Claude's look: the user's message in a warm right-aligned bubble, the reply as plain serif
# text without an avatar, quiet collapsible rows, small grey icon actions, a rounded input.
CHAT_CSS = """<style>
.st-key-chat-pane .rag-user {display: flex; justify-content: flex-end; margin: 0.5rem 0 0.1rem;}
.st-key-chat-pane .rag-user span {background: var(--rag-soft); color: var(--rag-ink);
  line-height: 1.45;
  border-radius: 16px 16px 4px 16px; padding: 0.5rem 0.9rem; max-width: 85%;
  white-space: pre-wrap;}
.st-key-chat-pane [class*="st-key-chat-answer"] :is(p, li, td, th) {
  font-family: "Source Serif 4", Georgia, serif; font-size: 1rem; line-height: 1.6;}
.st-key-chat-pane [class*="st-key-chat-answer"] :is(h1, h2, h3) {font-size: 1.08rem;
  font-weight: 650; padding: 0.7rem 0 0.2rem;}
.st-key-chat-pane [class*="st-key-chat-answer"] table {border-collapse: collapse;
  font-size: 0.9rem; margin: 0.4rem 0;}
.st-key-chat-pane [class*="st-key-chat-answer"] :is(td, th) {font-size: 0.9rem;
  border: 1px solid var(--rag-border); padding: 0.3rem 0.55rem;}
.st-key-chat-pane [class*="st-key-chat-answer"] th {background: var(--rag-soft); font-weight: 650;}
.st-key-chat-pane .rag-sources {font-size: 0.8rem; color: var(--rag-muted);
  margin: 0.2rem 0 0.3rem;}
.st-key-chat-pane .rag-sources .label {font-weight: 600; margin-right: 0.3rem;}
.st-key-chat-pane .rag-source {display: inline-block; background: var(--rag-soft);
  border-radius: 10px; padding: 0 0.5rem; margin: 0.1rem 0.2rem 0.1rem 0;}
.st-key-chat-pane [data-testid="stExpander"] details {border: none; background: transparent;}
.st-key-chat-pane [data-testid="stExpander"] summary {padding: 0.1rem 0; color: var(--rag-muted);}
.st-key-chat-pane [data-testid="stExpander"] summary p {font-size: 0.82rem;}
.st-key-chat-pane [data-testid="stExpander"] summary:hover {color: #d97757;}
.st-key-chat-pane [class*="st-key-chat-actions"] button {color: var(--rag-muted); min-height: 0;
  padding: 0.1rem 0.4rem; border-radius: 6px;}
.st-key-chat-pane [class*="st-key-chat-actions"] button:hover {color: #d97757;
  background: var(--rag-soft);}
.st-key-chat-pane [data-testid="stChatInput"] {border-radius: 16px; background: var(--rag-surface);
  border: 1px solid var(--rag-border-strong); box-shadow: 0 1px 6px var(--rag-shadow);}
.st-key-chat-pane [data-testid="stChatInput"]:focus-within {border-color: #d97757;}
</style>"""


@dataclass(frozen=True)
class Bubble:
    """Placeholders of the reply being written, filled by the live Overview."""

    steps: Any  # finished steps, one ✓ line each
    status: Any  # what is happening now, with a rotating word
    text: Any


def render(snapshot: IndexSnapshot | None, models: list[str] | None) -> Bubble | None:
    """Draw the chat pane; returns the empty reply bubble when a question starts now."""
    st.html(CHAT_CSS)
    with st.container(key="chat-pane"):
        return _pane(snapshot, models)


def _pane(snapshot: IndexSnapshot | None, models: list[str] | None) -> Bubble | None:
    history: tuple[Turn, ...] = st.session_state.get(HISTORY, ())
    header, clear = st.columns([3, 2], vertical_alignment="center")
    header.markdown("**💬 Chat**")
    if clear.button("New chat", key="new-chat", icon=":material/edit_square:", type="tertiary",
                    disabled=not history):  # fmt: skip
        st.session_state[HISTORY] = ()
        st.session_state.pop(TRACE, None)
        st.rerun()

    box = st.container(height=CHAT_HEIGHT, border=True, autoscroll=True)
    ready = snapshot is not None and bool(models)
    picked: str | None = None
    with box:
        welcome = st.empty()
        if not history:
            with welcome.container():
                picked = _welcome(snapshot, ready)
        active = st.session_state.get(TRACE)
        for index, turn in enumerate(history):
            _turn(index, turn, snapshot, is_active=turn.trace == active)

    typed = st.chat_input(
        "Ask about your documents…" if ready else "Build an index in ① Ingest first",
        disabled=not ready,
        key="chat-input",
    )
    st.caption("No chat memory: each question stands alone.")
    question = (typed or picked or "").strip()
    if not question:
        return None
    st.session_state[PENDING_QUESTION] = question
    st.session_state[TAB_KEY] = "Overview"  # the dashboard runs it live, side by side
    welcome.empty()  # the conversation has started
    with box:
        return _start_reply(question)


def start_regenerate(index: int, question: str) -> None:
    st.session_state[PENDING_REPLACE] = index
    st.session_state[PENDING_QUESTION] = question
    st.session_state[TAB_KEY] = "Overview"


def _welcome(snapshot: IndexSnapshot | None, ready: bool) -> str | None:
    st.markdown("#### 👋 Ask about your documents")
    if not ready:
        st.caption("Upload and index documents in ① Ingest, and start Ollama, to begin.")
        return None
    st.caption("Try one of these, or type your own below:")
    for number, text in enumerate(suggestions(snapshot.doc_ids if snapshot else ())):
        if st.button(f"💡 {text}", key=f"suggest-{number}", width="stretch"):
            return text
    return None


def _start_reply(question: str) -> Bubble:
    st.markdown(user_html(question), unsafe_allow_html=True)
    # Like Claude: the timeline of finished steps on top, the answer below it, and the
    # rotating status as the last line, so it moves down as the answer grows.
    with st.container(key="chat-reply-live"):
        steps = st.empty()
        with st.container(key="chat-answer-live"):
            text = st.empty()
        status = st.empty()
        status.markdown(status_html("Starting…"), unsafe_allow_html=True)
    return Bubble(steps, status, text)


def _turn(index: int, turn: Turn, snapshot: IndexSnapshot | None, *, is_active: bool) -> None:
    trace, answer = turn.trace, turn.trace.answer
    key = trace_key(trace)
    st.markdown(user_html(answer.question), unsafe_allow_html=True)
    with st.container(key=f"chat-reply-{index}"):
        steps = run_steps(trace, len(snapshot.chunks) if snapshot else None, turn.first_token_s)
        with st.expander(f"✓ {steps_summary(steps)}"):
            lines = [
                f"{step.label} · {step.detail}"
                + (f" · {step.ms / 1000:.1f} s" if step.ms is not None else "")
                for step in steps
            ]
            st.markdown(steps_html(lines), unsafe_allow_html=True)
        if answer.is_refusal:
            st.warning("The sources did not contain the answer, so the model did not guess.")
        with st.container(key=f"chat-answer-{index}"):
            st.markdown(spans_markdown(answer_spans(answer)), unsafe_allow_html=True)
        cited = cited_sources_html(trace)
        if cited:
            st.markdown(cited, unsafe_allow_html=True)
        judged = current_judged(trace)
        if judged is not None:
            st.markdown(_score_badges(judged), unsafe_allow_html=True)
        _sources(trace, key)
        speed = LiveStats(tokens=answer.output_tokens, generate_s=answer.latency_ms / 1000,
                          first_token_s=turn.first_token_s).tokens_per_s  # fmt: skip
        speed_text = f" · {speed:.0f} tok/s" if speed is not None else ""
        health = answer_health(trace, turn.first_token_s)
        quality = "".join(
            f" · {name} {value:.0%}"
            for name, value in (("coverage", health.citation_coverage),
                                ("grounding", health.grounding))
            if value is not None
        )  # fmt: skip
        st.caption(f"{answer.model} · {answer.output_tokens} tokens{speed_text} · "
                   f"{(trace.retrieval_ms + answer.latency_ms) / 1000:.1f} s{quality}"
                   + (" · 📊 in dashboard" if is_active else ""))  # fmt: skip
        _actions(index, turn, key, is_active=is_active, judged=judged is not None)


def _sources(trace: Any, key: str) -> None:
    cards = source_cards(trace)
    cited = sum(card.cited for card in cards)
    with st.expander(f"📚 {len(cards)} sources · {cited} cited"):
        for card in cards:
            mark = " · cited" if card.cited else ""
            st.markdown(
                f'<span style="color:{card.color};font-weight:700">[{card.number}]</span> '
                f"{html.escape(card.origin)}{mark}",
                unsafe_allow_html=True,
            )
            st.caption(card.text[:400] + ("…" if len(card.text) > 400 else ""))


def _score_badges(judged: Any) -> str:
    marks = {"faithfulness": settings.min_faithfulness,
             "relevance": settings.min_answer_relevance, "citations": None}  # fmt: skip
    values = {"faithfulness": judged.scores.faithfulness,
              "relevance": judged.scores.answer_relevance,
              "citations": judged.citations.score}  # fmt: skip
    badges = []
    for name, value in values.items():
        if value is None:
            continue
        mark = marks[name]
        colour = "#2563eb" if mark is None else "#16a34a" if value >= mark else "#dc2626"
        badges.append(
            f'<span style="border:1px solid {colour};color:{colour};border-radius:10px;'
            f'padding:0 8px;margin-right:4px;font-size:0.8rem">⚖️ {name} {value:.2f}</span>'
        )
    return "".join(badges)


def _actions(index: int, turn: Turn, key: str, *, is_active: bool, judged: bool) -> None:
    answer = turn.trace.answer
    row = st.container(key=f"chat-actions-{index}", horizontal=True, gap="small",
                       vertical_alignment="center")  # fmt: skip
    with row.popover("📋", type="tertiary", help="Copy the answer"):
        st.code(answer.text, language=None, wrap_lines=True)
    if row.button("🔄", key=f"redo-{key}-{index}", help="Regenerate this answer", type="tertiary"):
        start_regenerate(index, answer.question)
        st.rerun()
    if row.button("⚖️", key=f"judge-{key}-{index}", help="Judge this answer (1 to 2 minutes)",
                  disabled=judged or answer.is_refusal, type="tertiary"):  # fmt: skip
        _judge(turn)
    if row.button("📊", key=f"show-{key}-{index}", help="Show this answer in the dashboard",
                  disabled=is_active, type="tertiary"):  # fmt: skip
        st.session_state[TRACE] = turn.trace
        st.session_state[FIRST_OUTPUT_WAIT] = (key, turn.first_token_s)
        st.rerun()
    with row:
        value = st.feedback("thumbs", key=f"rate-{key}-{index}")
    if value is not None and value != turn.feedback:
        history: tuple[Turn, ...] = st.session_state.get(HISTORY, ())
        if index < len(history):
            st.session_state[HISTORY] = with_feedback(history, index, value)


def _judge(turn: Turn) -> None:
    try:
        with st.status("Judging this answer…", expanded=False):
            judge, checker = load_judge(settings.judge_model)
            result = judge_answer(turn.trace, judge, checker)
            save_judged(turn.trace, result)  # saved before the next st.* call
    except OptionalDependencyError:
        st.error("The judge needs the evaluation extra: `uv sync --all-extras --all-groups`")
        return
    except (JudgeError, GenerationError, ValueError) as exc:
        st.error(str(exc))
        return
    st.rerun()
